from __future__ import annotations

import queue
import secrets
import threading
import time
from collections import deque
from typing import Optional

from PyQt5.QtCore import QObject, pyqtSignal

from cpu_verify import CpuVerifier, target_hex_to_assigned_work
from models import CandidateShare, MinerConfig, MiningJob, NonceWindow, SubmitResult, VerifiedShare
from opencl_miner import OpenCLGpuScanner
from stratum_connection import MiningConnection
from utils import nonce_to_hex_le, safe_bytes_from_hex


class RollingWorkMeter:
    def __init__(self, window_seconds: float) -> None:
        self.window_seconds = float(window_seconds)
        self._samples: deque[tuple[float, float]] = deque()
        self._total_work = 0.0

    def add_work(self, work: float, now: Optional[float] = None) -> None:
        if work <= 0:
            return
        ts = time.time() if now is None else float(now)
        self._samples.append((ts, float(work)))
        self._total_work += float(work)
        self._prune(ts)

    def rate(self, now: Optional[float] = None) -> float:
        ts = time.time() if now is None else float(now)
        self._prune(ts)
        if self.window_seconds <= 0:
            return 0.0
        return self._total_work / self.window_seconds

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            _, work = self._samples.popleft()
            self._total_work -= work
        if self._total_work < 0:
            self._total_work = 0.0


class MinerWorker(QObject):
    log = pyqtSignal(str)
    stats = pyqtSignal(dict)
    status = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: MinerConfig, verifier: Optional[CpuVerifier] = None) -> None:
        super().__init__()
        self.config = config

        needs_verifier_object = bool(
            config.require_dataset
            or config.enable_cpu_verify
            or bool(getattr(config, "enable_cpu_rescue_scan", False))
        )
        self.verifier = verifier
        if self.verifier is None and needs_verifier_object:
            self.verifier = CpuVerifier(
                dll_path=(config.randomx_dll_path or None),
                randomx_runtime_dll_path=(config.randomx_runtime_dll_path or None),
                preload_randomx_runtime=bool(config.preload_randomx_runtime),
                nonce_offset=config.nonce_offset,
                on_log=self.log.emit,
            )

        self._stop = threading.Event()
        self._job_lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._job: Optional[MiningJob] = None
        self._job_generation: int = 0

        self._scanner: Optional[OpenCLGpuScanner] = None
        self._client: Optional[MiningConnection] = None

        self._dataset_refresh_needed = False
        self._job_refresh_needed = False
        self._dataset_job_id: str = ""
        self._dataset_seed_hash_hex: str = ""

        self._candidate_queue: Optional[queue.Queue[tuple[int, CandidateShare] | None]] = None
        self._submit_queue: Optional[queue.Queue[tuple[int, CandidateShare, VerifiedShare, float] | None]] = None

        self._verify_threads: list[threading.Thread] = []
        self._submit_thread: Optional[threading.Thread] = None

        # Background nonce-window prefetch.
        self._window_queue_limit = max(2, int(getattr(config, "prefetch_window_queue_limit", 8)))
        self._window_queue: queue.Queue[tuple[int, str, str, NonceWindow] | None] = queue.Queue(
            maxsize=self._window_queue_limit
        )
        self._window_prefetch_thread: Optional[threading.Thread] = None
        self._window_prefetch_stop = threading.Event()

        self._accepted = 0
        self._rejected = 0
        self._candidates = 0
        self._verified = 0
        self._submitted_unverified = 0
        self._queued_dropped = 0
        self._verify_rejected = 0
        self._submit_stale = 0
        self._submit_duplicate = 0
        self._submit_invalid = 0
        self._submit_backend_error = 0
        self._job_changed_scan_drops = 0
        self._job_changed_submit_drops = 0

        self._teacher_hash_batches = 0
        self._teacher_hash_labels = 0
        self._teacher_hash_matches = 0
        self._teacher_tail_batches = 0
        self._teacher_tail_screened = 0
        self._teacher_tail_accepted = 0

        self._scan_seq = 0
        self._last_verified_share_scan_seq = 0
        self._last_rescue_scan_seq = 0
        self._no_share_scan_streak = 0

        self._cpu_rescue_runs = 0
        self._cpu_rescue_hits = 0
        self._cpu_rescue_empty = 0
        self._cpu_rescue_skipped_old_job = 0
        self._cpu_rescue_skipped_busy = 0

        self._nonce_cursor = secrets.randbits(32)
        self._hashes_done = 0
        self._start_time = 0.0
        self._last_stats_emit = 0.0
        self._scan_launches_last = 0
        self._last_effective_candidate_target = int(self.config.scan_candidate_target)
        self._last_effective_work_items = int(self.config.active_scan_window())
        self._last_job_age_ms = 0
        self._last_verify_pressure_q8 = 0
        self._last_submit_pressure_q8 = 0
        self._last_stale_risk_q8 = 0
        self._last_scan_window_source = "local"
        self._last_scan_window_count = int(self.config.active_scan_window())
        self._monerorpc_local_fallback_logged = False

        self._scan_rate_15m = RollingWorkMeter(15 * 60)
        self._scan_rate_1h = RollingWorkMeter(60 * 60)
        self._scan_rate_24h = RollingWorkMeter(24 * 60 * 60)

        self._verified_work_15m = RollingWorkMeter(15 * 60)
        self._accepted_work_15m = RollingWorkMeter(15 * 60)
        self._accepted_work_1h = RollingWorkMeter(60 * 60)
        self._accepted_work_24h = RollingWorkMeter(24 * 60 * 60)

        self._accepted_work_total = 0.0
        self._submit_blob_by_job_id: dict[str, str] = {}

    def _solo_like_verify_debug_enabled(self) -> bool:
        try:
            backend = str(self.config.mining_backend_name() or "").strip().lower()
        except Exception:
            backend = ""
        return bool(
            getattr(self.config, "use_solo", False)
            or getattr(self.config, "use_monero_rpc", False)
            or backend in {"solo", "monerorpc", "monero_rpc"}
        )

    def _submit_stale_grace_ms(self) -> int:
        return max(0, int(getattr(self.config, "submit_stale_grace_ms", 800)))

    def _candidate_debug_text(self, cand: CandidateShare) -> str:
        nonce = int(getattr(cand, "nonce", 0)) & 0xFFFFFFFF
        pred_tail = int(getattr(cand, "predicted_tail_u64", 0))
        rank = int(getattr(cand, "rank_score_u64", 0))
        bucket = int(getattr(cand, "tune_bucket", -1))
        tail_bin = int(getattr(cand, "tune_tail_bin", -1))
        rank_quality = int(getattr(cand, "rank_quality", 128))
        threshold_quality = int(getattr(cand, "threshold_quality", 128))
        predictor_match = 1 if bool(getattr(cand, "predictor_hash_match", False)) else 0

        gpu_hash_hex = str(getattr(cand, "gpu_hash_hex", "") or "").strip()
        if gpu_hash_hex:
            gpu_hash_hex = gpu_hash_hex[:24] + ("..." if len(gpu_hash_hex) > 24 else "")
        else:
            gpu_hash_hex = "-"

        return (
            f"nonce={nonce:08x} "
            f"pred_tail={pred_tail} "
            f"rank={rank} "
            f"bucket={bucket} "
            f"bin={tail_bin} "
            f"rq={rank_quality} "
            f"tq={threshold_quality} "
            f"pm={predictor_match} "
            f"gpu_hash={gpu_hash_hex}"
        )

    def _emit_verify_reject_debug(
        self,
        *,
        stage: str,
        job: Optional[MiningJob],
        total: int,
        kept: int,
        samples: list[CandidateShare],
        extra: str = "",
    ) -> None:
        if not self._solo_like_verify_debug_enabled():
            return

        now = time.time()
        min_interval_s = float(getattr(self.config, "solo_verify_debug_min_interval_s", 1.0))
        last_log_at = float(getattr(self, "_verify_debug_last_log_at", 0.0))

        if min_interval_s > 0.0 and (now - last_log_at) < min_interval_s:
            return

        self._verify_debug_last_log_at = now

        sample_cap = max(1, int(getattr(self.config, "solo_verify_debug_sample_items", 4)))
        sample_texts = [self._candidate_debug_text(c) for c in samples[:sample_cap]]
        if not sample_texts:
            sample_texts = ["-"]

        job_id = job.job_id if job is not None else "-"
        extra_text = f" {extra.strip()}" if extra else ""

        self.log.emit(
            f"[verify-debug] stage={stage} job={job_id} total={total} kept={kept}"
            f"{extra_text} samples={' || '.join(sample_texts)}"
        )

    def _needs_randomx_prepare(self) -> bool:
        return bool(
            self.verifier is not None
            and (
                self.config.require_dataset
                or self.config.enable_cpu_verify
                or self._cpu_rescue_enabled()
            )
        )

    def stop(self) -> None:
        self._stop.set()
        self._window_prefetch_stop.set()

    def _scan_mode(self) -> str:
        return self.config.normalized_scan_mode()

    def _verify_enabled(self) -> bool:
        return bool(
            self.config.enable_cpu_verify
            and self.verifier is not None
            and self.verifier.is_ready
        )

    def _cpu_rescue_enabled(self) -> bool:
        return bool(
            getattr(self.config, "enable_cpu_rescue_scan", False)
            and self.verifier is not None
            and self.verifier.is_ready
        )

    def _cpu_rescue_trigger_scans(self) -> int:
        return int(self.config.effective_cpu_rescue_trigger_scans())

    def _verify_batch_enabled(self) -> bool:
        return bool(
            self._verify_enabled()
            and self.config.enable_cpu_verify_batch
            and int(self.config.cpu_verify_batch_size) > 1
            and self.verifier is not None
            and self.verifier.has_batch_verify
        )

    def _hash_batch_enabled(self) -> bool:
        return bool(
            self._verify_enabled()
            and bool(getattr(self.config, "enable_cpu_hash_batch", True))
            and self.verifier is not None
            and self.verifier.has_batch_hash
        )

    def _tail_batch_enabled(self) -> bool:
        return bool(
            self._verify_enabled()
            and bool(getattr(self.config, "enable_cpu_tail_batch", True))
            and self.verifier is not None
            and self.verifier.has_batch_tail
        )

    def _unverified_submit_enabled(self) -> bool:
        if self.config.use_solo or self.config.use_monero_rpc:
            return False
        return bool(self.config.submit_unverified_shares)

    def _active_scan_window(self) -> int:
        return self.config.active_scan_window()

    def _desired_warm_batch_vms(self) -> int:
        if self.verifier is None or not self.verifier.is_ready:
            return 0

        desired = max(
            int(getattr(self.config, "cpu_verify_native_threads", 0)),
            int(getattr(self.config, "cpu_hash_batch_threads", 0)),
            int(getattr(self.config, "cpu_tail_batch_threads", 0)),
        )

        if self._verify_batch_enabled() or self._hash_batch_enabled() or self._tail_batch_enabled():
            desired = max(desired, 1)

        if self._cpu_rescue_enabled():
            desired = max(desired, 1)

        return max(0, desired)

    def _queue_fill_pct(self, q: Optional[queue.Queue], limit: int) -> float:
        if q is None or limit <= 0:
            return 0.0
        try:
            return max(0.0, min(1.0, float(q.qsize()) / float(limit)))
        except Exception:
            return 0.0

    def _classify_submit_result(self, result: SubmitResult) -> str:
        if result.accepted:
            return "pool_accepted"

        if getattr(result, "reject_class", ""):
            rc = str(getattr(result, "reject_class", "")).strip().lower()
            if rc in {
                "pool_stale",
                "pool_duplicate",
                "pool_invalid",
                "pool_backend_error",
                "pool_rejected",
                "stale",
                "duplicate",
                "invalid",
                "backend_error",
                "rejected",
            }:
                return {
                    "stale": "pool_stale",
                    "duplicate": "pool_duplicate",
                    "invalid": "pool_invalid",
                    "backend_error": "pool_backend_error",
                    "rejected": "pool_rejected",
                }.get(rc, rc)

        if getattr(result, "stale", False):
            return "pool_stale"
        if getattr(result, "duplicate", False):
            return "pool_duplicate"
        if getattr(result, "invalid", False):
            return "pool_invalid"
        if getattr(result, "backend_error", False):
            return "pool_backend_error"

        text = " ".join(
            [
                str(getattr(result, "status", "") or ""),
                str(getattr(result, "error", "") or ""),
                str(getattr(result, "reject_class", "") or ""),
                str(getattr(result, "raw", "") or ""),
            ]
        ).lower()

        if "stale" in text:
            return "pool_stale"
        if "duplicate" in text or "already submitted" in text or "duplicate_local" in text:
            return "pool_duplicate"
        if (
            "invalid" in text
            or "low difficulty" in text
            or "bad nonce" in text
            or "bad result" in text
            or "malformed" in text
            or "invalid_share" in text
        ):
            return "pool_invalid"
        if (
            "timeout" in text
            or "connect" in text
            or "gateway" in text
            or "session" in text
            or "not_open" in text
            or "socket" in text
            or "temporar" in text
            or "not connected" in text
        ):
            return "pool_backend_error"
        return "pool_rejected"

    def _build_scan_context(self, job: MiningJob) -> dict[str, int]:
        now = time.time()
        job_age_ms = max(0, int((now - float(getattr(job, "received_at", now))) * 1000.0))

        verify_pct = self._queue_fill_pct(self._candidate_queue, int(self.config.verify_queue_limit))
        submit_pct = self._queue_fill_pct(self._submit_queue, int(self.config.submit_queue_limit))

        age_soft = max(1, int(self.config.job_age_soft_ms))
        age_hard = max(age_soft + 1, int(self.config.job_age_hard_ms))
        if job_age_ms <= age_soft:
            age_pct = 0.0
        else:
            age_pct = max(
                0.0,
                min(1.0, float(job_age_ms - age_soft) / float(max(1, age_hard - age_soft))),
            )

        verify_q8 = max(0, min(255, int(round(verify_pct * 255.0))))
        submit_q8 = max(0, min(255, int(round(submit_pct * 255.0))))
        stale_risk_q8 = max(0, min(255, int(round(max(verify_pct, submit_pct, age_pct) * 255.0))))

        base_candidate_target = max(1, int(self.config.scan_candidate_target))
        base_work_items = max(1, int(self._active_scan_window()))

        effective_candidate_target = base_candidate_target
        effective_work_items = base_work_items

        if self.config.adaptive_queue_throttle:
            soft = max(
                float(self.config.clamped_verify_soft_pct()),
                float(self.config.clamped_submit_soft_pct()),
            )
            live_pressure = max(verify_pct, submit_pct, age_pct)
            if live_pressure > soft:
                severity = (live_pressure - soft) / max(1e-6, 1.0 - soft)
                factor = max(
                    float(self.config.clamped_min_dynamic_work_pct()),
                    1.0 - (0.75 * severity),
                )

                effective_candidate_target = max(
                    int(self.config.min_candidate_target),
                    min(base_candidate_target, int(round(base_candidate_target * factor))),
                )
                effective_work_items = max(
                    int(getattr(self.config, "min_dynamic_work_items", 16384)),
                    min(base_work_items, int(round(base_work_items * factor))),
                )

        self._last_effective_candidate_target = int(effective_candidate_target)
        self._last_effective_work_items = int(effective_work_items)
        self._last_job_age_ms = int(job_age_ms)
        self._last_verify_pressure_q8 = int(verify_q8)
        self._last_submit_pressure_q8 = int(submit_q8)
        self._last_stale_risk_q8 = int(stale_risk_q8)

        return {
            "job_age_ms": int(job_age_ms),
            "verify_pressure_q8": int(verify_q8),
            "submit_pressure_q8": int(submit_q8),
            "stale_risk_q8": int(stale_risk_q8),
            "candidate_target": int(effective_candidate_target),
            "work_items": int(effective_work_items),
        }

    def _acquire_scan_window(self, requested_span: int) -> Optional[NonceWindow]:
        span = max(1, int(requested_span or self._active_scan_window()))
        client = self._client

        if self.config.use_monero_rpc and client is not None:
            try:
                window = client.request_scan_window(span)
            except Exception as exc:
                self.log.emit(f"[monerorpc] nonce lease request failed: {exc}")
                if bool(getattr(self.config, "monero_rpc_require_leases", False)):
                    return None
                window = None

            if window is not None and int(getattr(window, "count", 0)) > 0:
                self._last_scan_window_source = str(getattr(window, "source", "monerorpc") or "monerorpc")
                self._last_scan_window_count = max(1, int(window.count))
                self._monerorpc_local_fallback_logged = False
                return window

            if bool(getattr(self.config, "monero_rpc_require_leases", False)):
                self.status.emit("waiting_for_lease")
                return None

            if not self._monerorpc_local_fallback_logged:
                self.log.emit("[monerorpc] lease unavailable, falling back to local nonce cursor")
                self._monerorpc_local_fallback_logged = True

        start = self._next_nonce_window(span)
        self._last_scan_window_source = "local"
        self._last_scan_window_count = span
        return NonceWindow(start_nonce=start, count=span, source="local")

    def _window_prefetch_loop(self) -> None:
        while not self._window_prefetch_stop.is_set() and not self._stop.is_set():
            try:
                if self._window_queue.full():
                    time.sleep(0.005)
                    continue

                job, job_generation = self._get_job_state()
                if job is None:
                    time.sleep(0.02)
                    continue

                scan_ctx = self._build_scan_context(job)
                requested_span = int(scan_ctx["work_items"])

                window = self._acquire_scan_window(requested_span)
                if window is None or int(getattr(window, "count", 0)) <= 0:
                    time.sleep(0.02)
                    continue

                item = (
                    int(job_generation),
                    str(job.job_id),
                    str(job.session_id),
                    window,
                )

                try:
                    self._window_queue.put(item, timeout=0.1)
                except queue.Full:
                    continue

            except Exception as exc:
                self.log.emit(f"[window-prefetch] error: {exc}")
                time.sleep(0.05)

    def _dequeue_prefetched_window(
        self,
        job: MiningJob,
        job_generation: int,
        timeout_s: float = 0.05,
    ) -> Optional[NonceWindow]:
        deadline = time.time() + max(0.01, float(timeout_s))

        while not self._stop.is_set():
            remaining = deadline - time.time()
            if remaining <= 0.0:
                return None

            try:
                item = self._window_queue.get(timeout=remaining)
            except queue.Empty:
                return None

            if item is None:
                try:
                    self._window_queue.task_done()
                except Exception:
                    pass
                return None

            queued_generation, queued_job_id, queued_session_id, window = item
            try:
                self._window_queue.task_done()
            except Exception:
                pass

            live_job, live_generation = self._get_job_state()
            if live_job is None:
                continue

            if (
                queued_generation != job_generation
                or live_generation != job_generation
                or queued_job_id != job.job_id
                or queued_session_id != job.session_id
                or live_job.job_id != job.job_id
                or live_job.session_id != job.session_id
            ):
                continue

            return window

        return None

    def run(self) -> None:
        scanner = OpenCLGpuScanner(self.config, self.log.emit)
        client = MiningConnection(
            self.config,
            on_log=self.log.emit,
            on_job=self._on_job,
            on_status=self.status.emit,
        )

        self._scanner = scanner
        self._client = client

        try:
            self._start_time = time.time()
            scanner.initialize()
            client.connect()
            self._start_pipeline_threads()

            if (
                self.config.enable_cpu_verify
                and self.config.enable_cpu_verify_batch
                and self.verifier is not None
                and not self.verifier.has_batch_verify
            ):
                self.log.emit(
                    "[verify] native batch export not found in DLL; "
                    "falling back to per-nonce verification"
                )

            if (
                self.config.enable_cpu_verify
                and bool(getattr(self.config, "enable_cpu_hash_batch", True))
                and self.verifier is not None
                and not self.verifier.has_batch_hash
            ):
                self.log.emit(
                    "[verify] native hash batch export not found in DLL; "
                    "falling back to verify batch / per-nonce exact labeling"
                )

            if (
                self.config.enable_cpu_verify
                and bool(getattr(self.config, "enable_cpu_tail_batch", True))
                and self.verifier is not None
                and not self.verifier.has_batch_tail
            ):
                self.log.emit(
                    "[verify] native tail batch export not found in DLL; "
                    "falling back to exact hash labeling"
                )

            if self.config.use_solo and not self._verify_enabled():
                self.log.emit("[worker] solo mode detected; raw/unverified submit is disabled")

            if self.config.use_monero_rpc and not self._verify_enabled():
                self.log.emit("[worker] monerorpc mode detected; raw/unverified submit is disabled")

            while not self._stop.is_set():
                self._process_pending_verifier_refresh()

                if self._needs_randomx_prepare():
                    with self._job_lock:
                        refresh_pending = self._dataset_refresh_needed or self._job_refresh_needed
                    if refresh_pending:
                        time.sleep(0.02)
                        self._emit_stats(force=False)
                        continue

                job, job_generation = self._get_job_state()
                if not job:
                    time.sleep(0.02)
                    self._emit_stats(force=False)
                    continue

                if self.config.require_dataset and (scanner.dataset_buf is None or scanner.dataset_words <= 0):
                    time.sleep(0.02)
                    self._emit_stats(force=False)
                    continue

                scan_ctx = self._build_scan_context(job)

                window = self._dequeue_prefetched_window(
                    job=job,
                    job_generation=job_generation,
                    timeout_s=0.05,
                )
                if window is None or int(window.count) <= 0:
                    self._emit_stats(force=False)
                    continue

                start_nonce = int(window.start_nonce) & 0xFFFFFFFF
                work_items = max(1, int(window.count))
                self._last_scan_window_source = str(window.source or "local")
                self._last_scan_window_count = int(work_items)

                candidates = scanner.scan(
                    job,
                    start_nonce,
                    job_age_ms=scan_ctx["job_age_ms"],
                    verify_pressure_q8=scan_ctx["verify_pressure_q8"],
                    submit_pressure_q8=scan_ctx["submit_pressure_q8"],
                    stale_risk_q8=scan_ctx["stale_risk_q8"],
                    scan_candidate_target_override=scan_ctx["candidate_target"],
                    work_items_override=work_items,
                )

                live_job, live_generation = self._get_job_state()
                if (
                    live_generation != job_generation
                    or live_job is None
                    or live_job.job_id != job.job_id
                    or live_job.session_id != job.session_id
                ):
                    if candidates:
                        with self._stats_lock:
                            self._job_changed_scan_drops += len(candidates)
                        self.log.emit(
                            f"[worker] job changed during scan, dropping stale candidate batch "
                            f"for job={job.job_id} count={len(candidates)}"
                        )
                    self._emit_stats(force=False)
                    continue

                self._scan_seq += 1
                scan_seq = int(self._scan_seq)

                for cand in candidates:
                    cand.scan_seq = scan_seq
                    cand.source = "gpu"

                self._no_share_scan_streak = max(
                    0,
                    int(self._scan_seq - self._last_verified_share_scan_seq),
                )

                if candidates and bool(getattr(self.config, "sort_candidates", True)):
                    candidates.sort(key=scanner.candidate_sort_key)

                scanned_items = int(max(0, getattr(scanner, "last_scan_work_items", work_items)))
                self._scan_launches_last = int(max(1, getattr(scanner, "last_scan_chunk_count", 1)))

                with self._stats_lock:
                    self._candidates += len(candidates)
                    self._hashes_done += scanned_items
                    scan_work = float(scanned_items)
                    self._scan_rate_15m.add_work(scan_work)
                    self._scan_rate_1h.add_work(scan_work)
                    self._scan_rate_24h.add_work(scan_work)

                if candidates:
                    self.log.emit(
                        f"[opencl] candidates={len(candidates)} "
                        f"job={job.job_id} start_nonce={start_nonce:08x} "
                        f"mode={self._scan_mode()} backend={self.config.mining_backend_name()} "
                        f"window_source={self._last_scan_window_source} "
                        f"window_count={work_items} "
                        f"job_age_ms={scan_ctx['job_age_ms']} "
                        f"verify_q8={scan_ctx['verify_pressure_q8']} "
                        f"submit_q8={scan_ctx['submit_pressure_q8']} "
                        f"stale_q8={scan_ctx['stale_risk_q8']} "
                        f"target={scan_ctx['candidate_target']} "
                        f"no_share_streak={self._no_share_scan_streak}"
                    )
                else:
                    self.log.emit(
                        f"[opencl] empty candidate pass "
                        f"job={job.job_id} start_nonce={start_nonce:08x} "
                        f"no_share_streak={self._no_share_scan_streak}"
                    )

                self._enqueue_candidates(job_generation, candidates)

                self._maybe_run_cpu_rescue(
                    job=job,
                    job_generation=job_generation,
                    scan_ctx=scan_ctx,
                    scan_seq=scan_seq,
                )

                self._emit_stats(force=False)

                if self.config.scan_pause_ms > 0:
                    time.sleep(self.config.scan_pause_ms / 1000.0)

        except Exception as exc:
            self.log.emit(f"[fatal] {exc}")
            self.status.emit("error")
        finally:
            self._stop.set()
            self._window_prefetch_stop.set()
            self._stop_pipeline_threads()

            try:
                if self.verifier is not None:
                    self.verifier.close()
            except Exception:
                pass
            try:
                scanner.close()
            except Exception:
                pass
            try:
                client.close()
            except Exception:
                pass

            self._emit_stats(force=True)
            self._scanner = None
            self._client = None
            self.finished.emit()

    def _start_pipeline_threads(self) -> None:
        verify_enabled = self._verify_enabled()
        verify_batch_enabled = self._verify_batch_enabled()
        hash_batch_enabled = self._hash_batch_enabled()
        tail_batch_enabled = self._tail_batch_enabled()
        rescue_enabled = self._cpu_rescue_enabled()
        unverified_submit = self._unverified_submit_enabled()

        if verify_enabled:
            self._candidate_queue = queue.Queue(maxsize=max(1, int(self.config.verify_queue_limit)))
        else:
            self._candidate_queue = None

        if verify_enabled or unverified_submit or rescue_enabled:
            self._submit_queue = queue.Queue(maxsize=max(1, int(self.config.submit_queue_limit)))
        else:
            self._submit_queue = None

        self._verify_threads = []
        if verify_enabled and self._candidate_queue is not None:
            verify_threads = max(1, int(getattr(self.config, "verify_threads", 1)))
            if (verify_batch_enabled or hash_batch_enabled or tail_batch_enabled) and verify_threads != 1:
                self.log.emit(
                    f"[worker] verify_threads clamped from {verify_threads} to 1 "
                    "because native batch labeling is enabled"
                )
                verify_threads = 1

            for idx in range(verify_threads):
                t = threading.Thread(target=self._verify_loop, name=f"VerifyThread-{idx}", daemon=True)
                t.start()
                self._verify_threads.append(t)

        if self._submit_queue is not None:
            self._submit_thread = threading.Thread(target=self._submit_loop, name="SubmitThread", daemon=True)
            self._submit_thread.start()
        else:
            self._submit_thread = None

        self._window_prefetch_stop.clear()
        self._window_prefetch_thread = threading.Thread(
            target=self._window_prefetch_loop,
            name="WindowPrefetchThread",
            daemon=True,
        )
        self._window_prefetch_thread.start()

        self.log.emit(
            f"[worker] pipeline: "
            f"verify={'on' if verify_enabled else 'off'} "
            f"verify_batch={'on' if verify_batch_enabled else 'off'} "
            f"hash_batch={'on' if hash_batch_enabled else 'off'} "
            f"tail_batch={'on' if tail_batch_enabled else 'off'} "
            f"cpu_rescue={'on' if rescue_enabled else 'off'} "
            f"cpu_rescue_after_no_share_scans={self._cpu_rescue_trigger_scans()} "
            f"cpu_rescue_job_age_max_ms={int(getattr(self.config, 'cpu_rescue_job_age_max_ms', 1800))} "
            f"cpu_rescue_window_size={int(getattr(self.config, 'cpu_rescue_window_size', 4096))} "
            f"cpu_rescue_batch_size={int(getattr(self.config, 'cpu_rescue_batch_size', 1024))} "
            f"hash_batch_min_size={int(getattr(self.config, 'cpu_hash_batch_min_size', 8))} "
            f"hash_batch_threads={int(getattr(self.config, 'cpu_hash_batch_threads', 0))} "
            f"tail_batch_min_size={int(getattr(self.config, 'cpu_tail_batch_min_size', 32))} "
            f"tail_batch_threads={int(getattr(self.config, 'cpu_tail_batch_threads', 0))} "
            f"batch_size={int(getattr(self.config, 'cpu_verify_batch_size', 1))} "
            f"batch_wait_ms={int(getattr(self.config, 'cpu_verify_batch_wait_ms', 0))} "
            f"native_threads={int(getattr(self.config, 'cpu_verify_native_threads', 0))} "
            f"submit_unverified={'on' if unverified_submit else 'off'} "
            f"verify_threads={len(self._verify_threads)} "
            f"verify_queue_limit={int(self.config.verify_queue_limit)} "
            f"submit_queue_limit={int(self.config.submit_queue_limit)} "
            f"submit_stale_grace_ms={self._submit_stale_grace_ms()} "
            f"adaptive_queue_throttle={'on' if self.config.adaptive_queue_throttle else 'off'} "
            f"prefetch_window_queue_limit={self._window_queue_limit}"
        )

    def _stop_pipeline_threads(self) -> None:
        self._window_prefetch_stop.set()

        if self._candidate_queue is not None:
            for _ in self._verify_threads:
                try:
                    self._candidate_queue.put_nowait(None)
                except Exception:
                    pass

        if self._submit_queue is not None:
            try:
                self._submit_queue.put_nowait(None)
            except Exception:
                pass

        try:
            self._window_queue.put_nowait(None)
        except Exception:
            pass

        for t in self._verify_threads:
            try:
                t.join(timeout=2.0)
            except Exception:
                pass
        self._verify_threads = []

        if self._submit_thread is not None:
            try:
                self._submit_thread.join(timeout=2.0)
            except Exception:
                pass
            self._submit_thread = None

        if self._window_prefetch_thread is not None:
            try:
                self._window_prefetch_thread.join(timeout=2.0)
            except Exception:
                pass
            self._window_prefetch_thread = None

    def _candidate_to_unverified_submit(self, cand: CandidateShare) -> VerifiedShare:
        assigned_work = target_hex_to_assigned_work(cand.target_hex)
        verified = VerifiedShare(
            nonce_hex=nonce_to_hex_le(cand.nonce),
            result_hex=(cand.gpu_hash_hex or ""),
            job_id=cand.job_id,
            session_id=cand.session_id,
            assigned_work=assigned_work,
            actual_work=0.0,
            credited_work=assigned_work,
            quality=0.0,
            actual_tail_u64=0,
            predicted_tail_u64=int(getattr(cand, "predicted_tail_u64", 0)),
            rank_score_u64=int(getattr(cand, "rank_score_u64", 0)),
            tune_bucket=int(getattr(cand, "tune_bucket", -1)),
            tune_tail_bin=int(getattr(cand, "tune_tail_bin", -1)),
            rank_quality=int(getattr(cand, "rank_quality", 128)),
            threshold_quality=int(getattr(cand, "threshold_quality", 128)),
            gpu_hash_hex=(cand.gpu_hash_hex or ""),
            predictor_hash_match=bool(getattr(cand, "predictor_hash_match", False)),
        )
        self._attach_solution_blob(verified, cand.nonce, cand.job_id)
        return verified

    def _attach_solution_blob(self, verified: VerifiedShare, nonce: int, job_id: str) -> None:
        if not (self.config.use_solo or self.config.use_monero_rpc):
            return

        blob_hex = ""
        with self._job_lock:
            blob_hex = (self._submit_blob_by_job_id.get(job_id, "") or "").strip()
            current_job = self._job

        if not blob_hex and current_job is not None and current_job.job_id == job_id:
            blob_hex = (getattr(current_job, "submit_blob_hex", "") or "").strip()

        raw0 = safe_bytes_from_hex(blob_hex)
        if not raw0:
            return

        raw = bytearray(raw0)
        off = int(self.config.nonce_offset)
        if len(raw) < off + 4:
            return

        nonce_u32 = int(nonce) & 0xFFFFFFFF
        raw[off:off + 4] = nonce_u32.to_bytes(4, "little", signed=False)
        verified.solution_blob_hex = raw.hex()

    def _enqueue_candidates(self, job_generation: int, candidates: list[CandidateShare]) -> None:
        if not candidates:
            return

        process_limit = int(getattr(self.config, "cpu_verify_limit", 0))
        if process_limit > 0 and len(candidates) > process_limit:
            self.log.emit(
                f"[worker] process_limit clipped candidates for job={candidates[0].job_id}: "
                f"processing={process_limit} total={len(candidates)}"
            )
            candidates = candidates[:process_limit]

        if self._verify_enabled():
            if self._candidate_queue is None:
                return

            enqueued = 0
            dropped = 0
            scanner = self._scanner
            for cand in candidates:
                try:
                    self._candidate_queue.put_nowait((job_generation, cand))
                    enqueued += 1
                except queue.Full:
                    dropped += 1
                    if scanner is not None and getattr(cand, "source", "gpu") == "gpu":
                        scanner.record_feedback(cand, "queue_drop", 0.0, 0.0)

            if dropped:
                with self._stats_lock:
                    self._queued_dropped += dropped
                self.log.emit(
                    f"[worker] candidate queue full, dropped={dropped} "
                    f"enqueued={enqueued} job={candidates[0].job_id}"
                )
            return

        if self._unverified_submit_enabled():
            if self._submit_queue is None:
                return

            enqueued = 0
            dropped = 0
            scanner = self._scanner
            now = time.time()
            for cand in candidates:
                verified = self._candidate_to_unverified_submit(cand)
                try:
                    self._submit_queue.put_nowait((job_generation, cand, verified, now))
                    enqueued += 1
                except queue.Full:
                    dropped += 1
                    if scanner is not None and getattr(cand, "source", "gpu") == "gpu":
                        scanner.record_feedback(cand, "queue_drop", verified.credited_work, verified.quality)

            if dropped:
                with self._stats_lock:
                    self._queued_dropped += dropped
                self.log.emit(
                    f"[worker] submit queue full, dropped={dropped} "
                    f"enqueued={enqueued} job={candidates[0].job_id}"
                )

    def _collect_verify_batch(
        self,
        first_item: tuple[int, CandidateShare],
    ) -> tuple[list[tuple[int, CandidateShare]], bool]:
        batch = [first_item]
        stop_after_batch = False

        if not (
            self._verify_batch_enabled()
            or self._hash_batch_enabled()
            or self._tail_batch_enabled()
        ) or self._candidate_queue is None:
            return batch, stop_after_batch

        target_size = max(
            1,
            int(getattr(self.config, "cpu_verify_batch_size", 64)),
            int(getattr(self.config, "cpu_hash_batch_min_size", 8)),
            int(getattr(self.config, "cpu_tail_batch_min_size", 32)),
        )
        wait_ms = max(0, int(getattr(self.config, "cpu_verify_batch_wait_ms", 2)))
        deadline = time.time() + (wait_ms / 1000.0)

        while len(batch) < target_size and not self._stop.is_set():
            try:
                if len(batch) == 1 and wait_ms > 0:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    next_item = self._candidate_queue.get(timeout=remaining)
                else:
                    next_item = self._candidate_queue.get_nowait()
            except queue.Empty:
                break

            if next_item is None:
                try:
                    self._candidate_queue.task_done()
                except Exception:
                    pass
                stop_after_batch = True
                break

            batch.append(next_item)

        return batch, stop_after_batch

    def _queue_verified_submit(
        self,
        job_generation: int,
        cand: CandidateShare,
        verified: VerifiedShare,
    ) -> None:
        if self._submit_queue is None:
            return

        queued = False
        enqueued_at = time.time()

        while not self._stop.is_set():
            current_job, _current_generation = self._get_job_state()
            if current_job is None:
                with self._stats_lock:
                    self._job_changed_submit_drops += 1
                break
            if current_job.session_id != cand.session_id:
                with self._stats_lock:
                    self._job_changed_submit_drops += 1
                break

            try:
                self._submit_queue.put((job_generation, cand, verified, enqueued_at), timeout=0.1)
                queued = True
                break
            except queue.Full:
                continue

        if not queued:
            scanner = self._scanner
            if scanner is not None and getattr(cand, "source", "gpu") == "gpu":
                scanner.record_feedback(cand, "queue_drop", verified.credited_work, verified.quality)
            self.log.emit(
                f"[worker] submit queue full/stale, dropping verified share "
                f"job={cand.job_id} nonce={verified.nonce_hex}"
            )

    def _handle_verified_share(
        self,
        job_generation: int,
        cand: CandidateShare,
        verified: VerifiedShare,
        *,
        record_scanner_feedback: bool = True,
    ) -> None:
        scanner = self._scanner

        self._attach_solution_blob(verified, cand.nonce, cand.job_id)

        if record_scanner_feedback and scanner is not None and getattr(cand, "source", "gpu") == "gpu":
            scanner.record_feedback(cand, "cpu_verified", verified.credited_work, verified.quality)

        share_scan_seq = int(getattr(cand, "scan_seq", 0))
        if share_scan_seq > self._last_verified_share_scan_seq:
            self._last_verified_share_scan_seq = share_scan_seq
        self._no_share_scan_streak = max(
            0,
            int(self._scan_seq - self._last_verified_share_scan_seq),
        )

        with self._stats_lock:
            self._verified += 1
            self._verified_work_15m.add_work(verified.credited_work)

        prefix = "[verify-rescue]" if getattr(cand, "source", "gpu") != "gpu" else "[verify]"
        self.log.emit(
            f"{prefix} share found nonce={verified.nonce_hex} "
            f"job={verified.job_id} session={verified.session_id} "
            f"pred_tail={verified.predicted_tail_u64} "
            f"act_tail={verified.actual_tail_u64} "
            f"bucket={verified.tune_bucket} bin={verified.tune_tail_bin} "
            f"actual_work={verified.actual_work:.6f} "
            f"assigned_work={verified.assigned_work:.6f} "
            f"credited_work={verified.credited_work:.6f} "
            f"quality={verified.quality:.6f}x "
            f"predictor_match={1 if verified.predictor_hash_match else 0} "
            f"scan_seq={share_scan_seq}"
        )

        self._queue_verified_submit(job_generation, cand, verified)

    def _cpu_rescue_batch_threads(self) -> int:
        verifier = self.verifier
        if verifier is None:
            return 0
        if verifier.has_batch_hash:
            return max(0, int(getattr(self.config, "cpu_hash_batch_threads", 0)))
        if verifier.has_batch_verify:
            return max(0, int(getattr(self.config, "cpu_verify_native_threads", 0)))
        return 0

    def _cpu_rescue_is_busy(self, scan_ctx: dict[str, int]) -> bool:
        verify_soft_q8 = int(round(float(self.config.clamped_verify_soft_pct()) * 255.0))
        submit_soft_q8 = int(round(float(self.config.clamped_submit_soft_pct()) * 255.0))
        return bool(
            int(scan_ctx.get("verify_pressure_q8", 0)) >= verify_soft_q8
            or int(scan_ctx.get("submit_pressure_q8", 0)) >= submit_soft_q8
        )

    def _maybe_run_cpu_rescue(
        self,
        job: MiningJob,
        job_generation: int,
        scan_ctx: dict[str, int],
        scan_seq: int,
    ) -> int:
        if not self._cpu_rescue_enabled():
            return 0

        current_streak = max(0, int(scan_seq - self._last_verified_share_scan_seq))
        self._no_share_scan_streak = current_streak

        trigger_scans = self._cpu_rescue_trigger_scans()
        if current_streak < trigger_scans:
            return 0

        if self._last_rescue_scan_seq > self._last_verified_share_scan_seq:
            if (scan_seq - self._last_rescue_scan_seq) < trigger_scans:
                return 0

        if self._cpu_rescue_is_busy(scan_ctx):
            with self._stats_lock:
                self._cpu_rescue_skipped_busy += 1
            return 0

        job_age_ms = int(scan_ctx.get("job_age_ms", 0))
        max_job_age_ms = max(0, int(getattr(self.config, "cpu_rescue_job_age_max_ms", 1800)))
        if job_age_ms > max_job_age_ms:
            with self._stats_lock:
                self._cpu_rescue_skipped_old_job += 1
            return 0

        window_size = max(1, int(getattr(self.config, "cpu_rescue_window_size", 4096)))
        batch_size = max(1, int(getattr(self.config, "cpu_rescue_batch_size", 1024)))
        window = self._acquire_scan_window(window_size)
        if window is None or int(window.count) <= 0:
            return 0

        live_job, live_generation = self._get_job_state()
        if (
            live_generation != job_generation
            or live_job is None
            or live_job.job_id != job.job_id
            or live_job.session_id != job.session_id
        ):
            return 0

        verifier = self.verifier
        if verifier is None:
            return 0

        self._last_rescue_scan_seq = scan_seq

        self.log.emit(
            f"[cpu-rescue] trigger job={job.job_id} "
            f"scan_seq={scan_seq} no_share_streak={current_streak} "
            f"job_age_ms={job_age_ms} "
            f"window_source={window.source or 'local'} "
            f"start_nonce={int(window.start_nonce) & 0xFFFFFFFF:08x} "
            f"count={int(window.count)}"
        )

        rescue_hits = 0
        try:
            rows = verifier.rescue_scan_window(
                job,
                int(window.start_nonce),
                int(window.count),
                batch_size=batch_size,
                max_threads=self._cpu_rescue_batch_threads(),
            )
        except Exception as exc:
            self.log.emit(f"[cpu-rescue] failed for job={job.job_id}: {exc}")
            return 0

        live_job, live_generation = self._get_job_state()
        if (
            live_generation != job_generation
            or live_job is None
            or live_job.job_id != job.job_id
            or live_job.session_id != job.session_id
        ):
            if rows:
                with self._stats_lock:
                    self._job_changed_scan_drops += len(rows)
            self.log.emit(
                f"[cpu-rescue] job changed before rescue-submit, dropping count={len(rows)} job={job.job_id}"
            )
            return 0

        for cand, verified in rows:
            cand.scan_seq = scan_seq
            cand.source = "cpu_rescue"
            self._handle_verified_share(
                job_generation,
                cand,
                verified,
                record_scanner_feedback=False,
            )
            rescue_hits += 1

        with self._stats_lock:
            self._cpu_rescue_runs += 1
            if rescue_hits > 0:
                self._cpu_rescue_hits += rescue_hits
            else:
                self._cpu_rescue_empty += 1

        self._no_share_scan_streak = max(
            0,
            int(self._scan_seq - self._last_verified_share_scan_seq),
        )

        self.log.emit(
            f"[cpu-rescue] done job={job.job_id} scan_seq={scan_seq} "
            f"scanned={int(window.count)} hits={rescue_hits} "
            f"batch_size={batch_size} threads={self._cpu_rescue_batch_threads()}"
        )
        return rescue_hits

    def _verify_loop(self) -> None:
        while not self._stop.is_set():
            if self._candidate_queue is None:
                time.sleep(0.05)
                continue

            try:
                first_item = self._candidate_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if first_item is None:
                try:
                    self._candidate_queue.task_done()
                except Exception:
                    pass
                break

            batch_items: list[tuple[int, CandidateShare]] = []
            stop_after_batch = False

            try:
                batch_items, stop_after_batch = self._collect_verify_batch(first_item)

                current_job, current_generation = self._get_job_state()
                if current_job is None:
                    continue

                scanner = self._scanner
                verifier = self.verifier
                if scanner is None or verifier is None or not self._verify_enabled():
                    continue

                live_items: list[tuple[int, CandidateShare]] = []
                for job_generation, cand in batch_items:
                    if (
                        current_generation == job_generation
                        and current_job.job_id == cand.job_id
                        and current_job.session_id == cand.session_id
                    ):
                        live_items.append((job_generation, cand))

                if not live_items:
                    continue

                if len(live_items) > 1 and self._tail_batch_enabled() and len(live_items) >= max(
                    2, int(getattr(self.config, "cpu_tail_batch_min_size", 32))
                ):
                    tail_threads = max(0, int(getattr(self.config, "cpu_tail_batch_threads", 0)))
                    shares = [cand for _, cand in live_items]
                    screened = verifier.screen_shares_batch_by_tail(
                        shares,
                        max_threads=tail_threads,
                    )

                    if len(screened) != len(live_items):
                        raise RuntimeError(
                            f"screen_shares_batch_by_tail returned {len(screened)} "
                            f"results for {len(live_items)} shares"
                        )

                    accepted_pairs: list[tuple[int, CandidateShare]] = []
                    tail_rejected_samples: list[CandidateShare] = []

                    for (job_generation, cand), item in zip(live_items, screened):
                        if not bool(getattr(item, "accepted_by_tail", False)):
                            with self._stats_lock:
                                self._verify_rejected += 1
                            scanner.record_feedback(cand, "cpu_rejected", 0.0, 0.0)
                            if len(tail_rejected_samples) < 8:
                                tail_rejected_samples.append(cand)
                            continue
                        accepted_pairs.append((job_generation, cand))

                    tail_rejected_count = len(live_items) - len(accepted_pairs)

                    if tail_rejected_count > 0:
                        self._emit_verify_reject_debug(
                            stage="tail_screen",
                            job=current_job,
                            total=len(live_items),
                            kept=len(accepted_pairs),
                            samples=tail_rejected_samples or [cand for _, cand in live_items[:4]],
                            extra=(
                                f"rejected={tail_rejected_count} "
                                f"tail_threads={tail_threads}"
                            ),
                        )

                    accepted_count = 0
                    match_count = 0
                    hash_rejected_samples: list[CandidateShare] = []

                    if accepted_pairs:
                        full_threads = max(0, int(getattr(self.config, "cpu_hash_batch_threads", 0)))
                        labeled = verifier.label_shares_batch_with_hashes(
                            [cand for _, cand in accepted_pairs],
                            max_threads=full_threads,
                        )

                        if len(labeled) != len(accepted_pairs):
                            raise RuntimeError(
                                f"label_shares_batch_with_hashes returned {len(labeled)} "
                                f"results for {len(accepted_pairs)} shares"
                            )

                        for (job_generation, cand), item in zip(accepted_pairs, labeled):
                            if bool(getattr(item, "predictor_hash_match", False)):
                                match_count += 1

                            if getattr(item, "verified", None) is None:
                                with self._stats_lock:
                                    self._verify_rejected += 1
                                scanner.record_feedback(cand, "cpu_rejected", 0.0, 0.0)
                                if len(hash_rejected_samples) < 8:
                                    hash_rejected_samples.append(cand)
                                continue

                            accepted_count += 1
                            self._handle_verified_share(job_generation, cand, item.verified)

                    hash_rejected_count = len(accepted_pairs) - accepted_count

                    if hash_rejected_count > 0:
                        self._emit_verify_reject_debug(
                            stage="hash_label_after_tail",
                            job=current_job,
                            total=len(accepted_pairs),
                            kept=accepted_count,
                            samples=hash_rejected_samples or [cand for _, cand in accepted_pairs[:4]],
                            extra=(
                                f"rejected={hash_rejected_count} "
                                f"predictor_matches={match_count}"
                            ),
                        )

                    with self._stats_lock:
                        self._teacher_tail_batches += 1
                        self._teacher_tail_screened += len(live_items)
                        self._teacher_tail_accepted += accepted_count
                        self._teacher_hash_labels += accepted_count
                        self._teacher_hash_matches += match_count

                    self.log.emit(
                        f"[verify-tail-batch] size={len(live_items)} "
                        f"screen_accept={len(accepted_pairs)} "
                        f"final_accept={accepted_count} "
                        f"predictor_matches={match_count} "
                        f"tail_threads={tail_threads}"
                    )

                elif len(live_items) > 1:
                    prefer_hash_batch = (
                        self._hash_batch_enabled()
                        and len(live_items) >= max(2, int(getattr(self.config, "cpu_hash_batch_min_size", 8)))
                    )

                    native_threads = max(
                        0,
                        int(
                            getattr(self.config, "cpu_hash_batch_threads", 0)
                            if prefer_hash_batch
                            else getattr(self.config, "cpu_verify_native_threads", 0)
                        ),
                    )

                    shares = [cand for _, cand in live_items]
                    labeled = verifier.label_shares_batch_with_hashes(
                        shares,
                        max_threads=native_threads,
                    )

                    if len(labeled) != len(live_items):
                        raise RuntimeError(
                            f"label_shares_batch_with_hashes returned {len(labeled)} "
                            f"results for {len(live_items)} shares"
                        )

                    accepted_count = 0
                    match_count = 0
                    rejected_samples: list[CandidateShare] = []

                    for (job_generation, cand), item in zip(live_items, labeled):
                        if bool(getattr(item, "predictor_hash_match", False)):
                            match_count += 1

                        if getattr(item, "verified", None) is None:
                            with self._stats_lock:
                                self._verify_rejected += 1
                            scanner.record_feedback(cand, "cpu_rejected", 0.0, 0.0)
                            if len(rejected_samples) < 8:
                                rejected_samples.append(cand)
                            continue

                        accepted_count += 1
                        self._handle_verified_share(job_generation, cand, item.verified)

                    rejected_count = len(live_items) - accepted_count
                    if rejected_count > 0:
                        self._emit_verify_reject_debug(
                            stage="batch_exact",
                            job=current_job,
                            total=len(live_items),
                            kept=accepted_count,
                            samples=rejected_samples or [cand for _, cand in live_items[:4]],
                            extra=(
                                f"rejected={rejected_count} "
                                f"predictor_matches={match_count} "
                                f"native_threads={native_threads}"
                            ),
                        )

                    if prefer_hash_batch:
                        with self._stats_lock:
                            self._teacher_hash_batches += 1
                            self._teacher_hash_labels += len(live_items)
                            self._teacher_hash_matches += match_count

                        self.log.emit(
                            f"[verify-hash-batch] size={len(live_items)} accepted={accepted_count} "
                            f"predictor_matches={match_count} native_threads={native_threads}"
                        )
                    else:
                        self.log.emit(
                            f"[verify-batch] size={len(live_items)} accepted={accepted_count} "
                            f"predictor_matches={match_count} native_threads={native_threads}"
                        )

                else:
                    job_generation, cand = live_items[0]
                    verified, _credited_work = verifier.verify_with_work(cand)
                    if not verified:
                        with self._stats_lock:
                            self._verify_rejected += 1
                        scanner.record_feedback(cand, "cpu_rejected", 0.0, 0.0)

                        self._emit_verify_reject_debug(
                            stage="single_exact",
                            job=current_job,
                            total=1,
                            kept=0,
                            samples=[cand],
                            extra="rejected=1 reason=verify_with_work_none",
                        )
                        continue

                    self._handle_verified_share(job_generation, cand, verified)

            finally:
                for _ in batch_items:
                    try:
                        self._candidate_queue.task_done()
                    except Exception:
                        pass

            if stop_after_batch:
                break

    def _submit_loop(self) -> None:
        while not self._stop.is_set():
            if self._submit_queue is None:
                time.sleep(0.05)
                continue

            try:
                item = self._submit_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                try:
                    self._submit_queue.task_done()
                except Exception:
                    pass
                break

            job_generation, cand, verified, enqueued_at = item

            try:
                current_job, _current_generation = self._get_job_state()
                if current_job is None:
                    with self._stats_lock:
                        self._job_changed_submit_drops += 1
                    continue

                if current_job.session_id != cand.session_id:
                    with self._stats_lock:
                        self._job_changed_submit_drops += 1
                    continue

                share_age_ms = max(0, int((time.time() - float(enqueued_at)) * 1000.0))
                if current_job.job_id != cand.job_id and share_age_ms > self._submit_stale_grace_ms():
                    with self._stats_lock:
                        self._submit_stale += 1
                        self._job_changed_submit_drops += 1
                    self.log.emit(
                        f"[submit] local stale drop nonce={verified.nonce_hex} "
                        f"job={verified.job_id} current={current_job.job_id} age_ms={share_age_ms}"
                    )
                    continue

                scanner = self._scanner
                client = self._client
                if client is None:
                    continue

                submit_result = client.submit(verified)
                outcome = self._classify_submit_result(submit_result)

                with self._stats_lock:
                    if submit_result.accepted:
                        self._accepted += 1
                        self._accepted_work_total += verified.credited_work
                        self._accepted_work_15m.add_work(verified.credited_work)
                        self._accepted_work_1h.add_work(verified.credited_work)
                        self._accepted_work_24h.add_work(verified.credited_work)
                    else:
                        self._rejected += 1
                        if outcome == "pool_stale":
                            self._submit_stale += 1
                        elif outcome == "pool_duplicate":
                            self._submit_duplicate += 1
                        elif outcome == "pool_invalid":
                            self._submit_invalid += 1
                        elif outcome == "pool_backend_error":
                            self._submit_backend_error += 1

                exact_submit = bool(verified.actual_work > 0.0 or getattr(cand, "source", "gpu") != "gpu")
                if exact_submit:
                    accepted_log_prefix = "[submit] accepted"
                    rejected_log_prefix = "[submit] rejected"
                else:
                    with self._stats_lock:
                        self._submitted_unverified += 1
                    accepted_log_prefix = "[submit-raw] accepted"
                    rejected_log_prefix = "[submit-raw] rejected"

                if scanner is not None and getattr(cand, "source", "gpu") == "gpu":
                    scanner.record_feedback(cand, outcome, verified.credited_work, verified.quality)

                if submit_result.accepted:
                    self.log.emit(
                        f"{accepted_log_prefix} nonce={verified.nonce_hex} "
                        f"job={verified.job_id} status={submit_result.status} "
                        f"class={outcome} credited_work={verified.credited_work:.6f}"
                    )
                else:
                    self.log.emit(
                        f"{rejected_log_prefix} nonce={verified.nonce_hex} "
                        f"job={verified.job_id} status={submit_result.status} "
                        f"class={outcome} error={submit_result.error}"
                    )

                self._emit_stats(force=False)

            finally:
                try:
                    self._submit_queue.task_done()
                except Exception:
                    pass

    def _emit_stats(self, force: bool) -> None:
        now = time.time()
        interval = max(0.05, float(int(getattr(self.config, "stats_update_ms", 500))) / 1000.0)

        with self._stats_lock:
            if not force and (now - self._last_stats_emit) < interval:
                return
            self._last_stats_emit = now

            accepted = self._accepted
            rejected = self._rejected
            candidates = self._candidates
            verified = self._verified
            submitted_unverified = self._submitted_unverified
            queued_dropped = self._queued_dropped
            accepted_work_total = self._accepted_work_total
            verify_rejected = self._verify_rejected
            submit_stale = self._submit_stale
            submit_duplicate = self._submit_duplicate
            submit_invalid = self._submit_invalid
            submit_backend_error = self._submit_backend_error
            job_changed_scan_drops = self._job_changed_scan_drops
            job_changed_submit_drops = self._job_changed_submit_drops

            teacher_hash_batches = self._teacher_hash_batches
            teacher_hash_labels = self._teacher_hash_labels
            teacher_hash_matches = self._teacher_hash_matches
            teacher_tail_batches = self._teacher_tail_batches
            teacher_tail_screened = self._teacher_tail_screened
            teacher_tail_accepted = self._teacher_tail_accepted

            no_share_scan_streak = self._no_share_scan_streak
            last_verified_share_scan_seq = self._last_verified_share_scan_seq
            last_rescue_scan_seq = self._last_rescue_scan_seq

            cpu_rescue_runs = self._cpu_rescue_runs
            cpu_rescue_hits = self._cpu_rescue_hits
            cpu_rescue_empty = self._cpu_rescue_empty
            cpu_rescue_skipped_old_job = self._cpu_rescue_skipped_old_job
            cpu_rescue_skipped_busy = self._cpu_rescue_skipped_busy

            elapsed = max(now - self._start_time, 1e-6)
            scan_rate_lifetime = int(self._hashes_done / elapsed)
            gpu_scan_rate_15m = int(self._scan_rate_15m.rate(now))
            gpu_scan_rate_1h = int(self._scan_rate_1h.rate(now))
            gpu_scan_rate_24h = int(self._scan_rate_24h.rate(now))
            verified_rate_15m = int(self._verified_work_15m.rate(now))
            p2pool_rate_15m = int(self._accepted_work_15m.rate(now))
            p2pool_rate_1h = int(self._accepted_work_1h.rate(now))
            p2pool_rate_24h = int(self._accepted_work_24h.rate(now))

        job, _ = self._get_job_state()
        verify_yield = (float(verified) / float(candidates)) if candidates > 0 else 0.0
        accept_yield = (float(accepted) / float(verified)) if verified > 0 else 0.0
        teacher_hash_match_rate = (
            float(teacher_hash_matches) / float(teacher_hash_labels)
            if teacher_hash_labels > 0 else 0.0
        )
        teacher_tail_accept_rate = (
            float(teacher_tail_accepted) / float(teacher_tail_screened)
            if teacher_tail_screened > 0 else 0.0
        )

        stats = {
            "accepted": accepted,
            "rejected": rejected,
            "candidates": candidates,
            "verified": verified,
            "verify_rejected": verify_rejected,
            "submitted_unverified": submitted_unverified,
            "queued_dropped": queued_dropped,
            "submit_stale": submit_stale,
            "submit_duplicate": submit_duplicate,
            "submit_invalid": submit_invalid,
            "submit_backend_error": submit_backend_error,
            "job_changed_scan_drops": job_changed_scan_drops,
            "job_changed_submit_drops": job_changed_submit_drops,
            "scan_rate_lifetime": scan_rate_lifetime,
            "gpu_scan_rate_15m": gpu_scan_rate_15m,
            "gpu_scan_rate_1h": gpu_scan_rate_1h,
            "gpu_scan_rate_24h": gpu_scan_rate_24h,
            "verified_rate_15m": verified_rate_15m,
            "p2pool_rate_15m": p2pool_rate_15m,
            "p2pool_rate_1h": p2pool_rate_1h,
            "p2pool_rate_24h": p2pool_rate_24h,
            "hashrate_est": p2pool_rate_15m,
            "accepted_work_total": accepted_work_total,
            "verify_yield": verify_yield,
            "accept_yield": accept_yield,
            "teacher_hash_batches": teacher_hash_batches,
            "teacher_hash_labels": teacher_hash_labels,
            "teacher_hash_matches": teacher_hash_matches,
            "teacher_hash_match_rate": teacher_hash_match_rate,
            "teacher_tail_batches": teacher_tail_batches,
            "teacher_tail_screened": teacher_tail_screened,
            "teacher_tail_accepted": teacher_tail_accepted,
            "teacher_tail_accept_rate": teacher_tail_accept_rate,
            "no_share_scan_streak": no_share_scan_streak,
            "last_verified_share_scan_seq": last_verified_share_scan_seq,
            "last_rescue_scan_seq": last_rescue_scan_seq,
            "cpu_rescue_runs": cpu_rescue_runs,
            "cpu_rescue_hits": cpu_rescue_hits,
            "cpu_rescue_empty": cpu_rescue_empty,
            "cpu_rescue_skipped_old_job": cpu_rescue_skipped_old_job,
            "cpu_rescue_skipped_busy": cpu_rescue_skipped_busy,
            "job_id": (job.job_id if job else "-"),
            "height": (job.height if job else 0),
            "backend": self.config.mining_backend_name(),
            "scan_mode": self._scan_mode(),
            "verification_enabled": self._verify_enabled(),
            "verification_batch_enabled": self._verify_batch_enabled(),
            "hash_batch_enabled": self._hash_batch_enabled(),
            "tail_batch_enabled": self._tail_batch_enabled(),
            "cpu_rescue_enabled": self._cpu_rescue_enabled(),
            "cpu_rescue_after_no_share_scans": self._cpu_rescue_trigger_scans(),
            "cpu_rescue_job_age_max_ms": int(getattr(self.config, "cpu_rescue_job_age_max_ms", 1800)),
            "cpu_rescue_window_size": int(getattr(self.config, "cpu_rescue_window_size", 4096)),
            "cpu_rescue_batch_size": int(getattr(self.config, "cpu_rescue_batch_size", 1024)),
            "cpu_verify_batch_size": int(getattr(self.config, "cpu_verify_batch_size", 1)),
            "cpu_verify_batch_wait_ms": int(getattr(self.config, "cpu_verify_batch_wait_ms", 0)),
            "cpu_verify_native_threads": int(getattr(self.config, "cpu_verify_native_threads", 0)),
            "cpu_hash_batch_min_size": int(getattr(self.config, "cpu_hash_batch_min_size", 8)),
            "cpu_hash_batch_threads": int(getattr(self.config, "cpu_hash_batch_threads", 0)),
            "cpu_tail_batch_min_size": int(getattr(self.config, "cpu_tail_batch_min_size", 32)),
            "cpu_tail_batch_threads": int(getattr(self.config, "cpu_tail_batch_threads", 0)),
            "job_tuning_enabled": bool(getattr(self.config, "enable_job_tuning", True)),
            "submit_unverified_shares": bool(self._unverified_submit_enabled()),
            "platform_index": int(self.config.platform_index),
            "device_index": int(self.config.device_index),
            "candidate_queue_depth": (self._candidate_queue.qsize() if self._candidate_queue is not None else 0),
            "submit_queue_depth": (self._submit_queue.qsize() if self._submit_queue is not None else 0),
            "prefetch_queue_depth": self._window_queue.qsize(),
            "scan_launches_last": int(self._scan_launches_last),
            "python_verify_threads_active": len(self._verify_threads),
            "effective_candidate_target_last": int(self._last_effective_candidate_target),
            "effective_work_items_last": int(self._last_effective_work_items),
            "job_age_ms_last": int(self._last_job_age_ms),
            "verify_pressure_q8_last": int(self._last_verify_pressure_q8),
            "submit_pressure_q8_last": int(self._last_submit_pressure_q8),
            "stale_risk_q8_last": int(self._last_stale_risk_q8),
            "scan_window_source_last": self._last_scan_window_source,
            "scan_window_count_last": int(self._last_scan_window_count),
            "split_tuning": True,
            "credit_tuning": True,
            "tail_bins": int(self.config.normalized_tail_bins()),
        }
        self.stats.emit(stats)

    def _process_pending_verifier_refresh(self) -> None:
        scanner = self._scanner
        if scanner is None:
            return

        with self._job_lock:
            if not (self._dataset_refresh_needed or self._job_refresh_needed):
                return
            job = self._job
            seed_refresh = self._dataset_refresh_needed
            job_refresh = self._job_refresh_needed

        if job is None:
            return

        if not self._needs_randomx_prepare():
            with self._job_lock:
                self._dataset_refresh_needed = False
                self._job_refresh_needed = False
                self._dataset_job_id = job.job_id
                self._dataset_seed_hash_hex = (job.seed_hash_hex or "").lower()
            return

        verifier = self.verifier
        if verifier is None or not verifier.is_ready:
            self.log.emit("[verify] RandomX verifier is not ready")
            time.sleep(0.5)
            return

        try:
            if seed_refresh:
                self.log.emit(
                    f"[verify] preparing RandomX seed for job_id={job.job_id} seed={job.seed_hash_hex}"
                )
                verifier.prepare_seed_for_job(job)

                warm_target = self._desired_warm_batch_vms()
                if warm_target > 0:
                    verifier.warm_batch_vms(warm_target)
                    self.log.emit(f"[verify] warmed batch vms target={warm_target}")

                if self.config.require_dataset:
                    if not verifier.has_dataset_exports:
                        raise RuntimeError(
                            "RandomX DLL does not export "
                            "bnrx_dataset_words64 / bnrx_export_dataset64"
                        )

                    self.log.emit(f"[verify] exporting RandomX dataset for seed={job.seed_hash_hex}")
                    dataset_u64 = verifier.export_dataset_u64()
                    scanner.bind_dataset(dataset_u64, verifier.current_dataset_fingerprint)

            if job_refresh:
                self.log.emit(f"[verify] binding verifier job job_id={job.job_id}")
                verifier.set_job(job)

            with self._job_lock:
                self._dataset_refresh_needed = False
                self._job_refresh_needed = False
                self._dataset_job_id = job.job_id
                self._dataset_seed_hash_hex = (job.seed_hash_hex or "").lower()

        except Exception as exc:
            self.log.emit(f"[verify] seed/job refresh failed for job={job.job_id}: {exc}")
            time.sleep(0.5)

    def _on_job(self, job: MiningJob) -> None:
        new_seed = (job.seed_hash_hex or "").lower()
        now = time.time()
        try:
            setattr(job, "received_at", now)
        except Exception:
            pass

        with self._job_lock:
            old_seed = self._dataset_seed_hash_hex
            self._job = job
            self._job_generation += 1
            self._nonce_cursor = secrets.randbits(32)

            needs_prepare = self._needs_randomx_prepare()
            self._dataset_refresh_needed = bool(new_seed != old_seed and needs_prepare)
            self._job_refresh_needed = bool(needs_prepare)

            self._dataset_job_id = job.job_id
            self._monerorpc_local_fallback_logged = False
            submit_blob_hex = str(getattr(job, "submit_blob_hex", "") or "").strip()
            if submit_blob_hex:
                self._submit_blob_by_job_id[job.job_id] = submit_blob_hex
                while len(self._submit_blob_by_job_id) > 64:
                    oldest = next(iter(self._submit_blob_by_job_id))
                    self._submit_blob_by_job_id.pop(oldest, None)

        self._scan_seq = 0
        self._last_verified_share_scan_seq = 0
        self._last_rescue_scan_seq = 0
        self._no_share_scan_streak = 0

        self._drain_queue(self._candidate_queue)
        self._drain_queue(self._window_queue)

    def _get_job_state(self) -> tuple[Optional[MiningJob], int]:
        with self._job_lock:
            return self._job, self._job_generation

    def _next_nonce_window(self, span: Optional[int] = None) -> int:
        step = max(1, int(span or self._active_scan_window()))
        with self._job_lock:
            start = self._nonce_cursor
            self._nonce_cursor = (self._nonce_cursor + step) & 0xFFFFFFFF
            return start

    @staticmethod
    def _drain_queue(q: Optional[queue.Queue]) -> None:
        if q is None:
            return
        while True:
            try:
                _item = q.get_nowait()
            except queue.Empty:
                break
            else:
                try:
                    q.task_done()
                except Exception:
                    pass