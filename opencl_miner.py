from __future__ import annotations

import ctypes
import os
import re
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pyopencl as cl

from models import CandidateShare, MinerConfig, MiningJob
from utils import resolve_resource_path, safe_bytes_from_hex, target_hex_to_prefilter_u64


@dataclass
class OpenCLDeviceInfo:
    platform_index: int
    device_index: int
    platform_name: str
    device_name: str


@dataclass
class _TuneState:
    seen: np.ndarray
    verified: np.ndarray
    cpu_rejected: np.ndarray
    pool_accepted: np.ndarray
    pool_rejected: np.ndarray
    pool_stale: np.ndarray
    pool_duplicate: np.ndarray
    pool_invalid: np.ndarray
    pool_backend_error: np.ndarray
    queue_drop: np.ndarray
    verified_work: np.ndarray
    quality_sum: np.ndarray
    pressure_sum: np.ndarray
    planes: np.ndarray
    dirty: bool = True


def _make_tune_state(cells: int, plane_count: int) -> _TuneState:
    zeros = lambda: np.zeros((cells,), dtype=np.float32)
    return _TuneState(
        seen=zeros(),
        verified=zeros(),
        cpu_rejected=zeros(),
        pool_accepted=zeros(),
        pool_rejected=zeros(),
        pool_stale=zeros(),
        pool_duplicate=zeros(),
        pool_invalid=zeros(),
        pool_backend_error=zeros(),
        queue_drop=zeros(),
        verified_work=zeros(),
        quality_sum=zeros(),
        pressure_sum=zeros(),
        planes=np.full((plane_count * cells,), 128, dtype=np.uint8),
        dirty=True,
    )


class OpenCLGpuScanner:
    _PLANE_RANK = 0
    _PLANE_THRESHOLD = 1
    _PLANE_CREDIT = 2
    _PLANE_CONFIDENCE = 3
    _PLANE_COUNT = 4

    def __init__(self, config: MinerConfig, on_log: Callable[[str], None]) -> None:
        self.config = config
        self.on_log = on_log

        self.ctx: Optional[cl.Context] = None
        self.queue: Optional[cl.CommandQueue] = None
        self.program: Optional[cl.Program] = None
        self.kernel = None
        self.device = None

        self.dataset_buf: Optional[cl.Buffer] = None
        self.dataset_words: int = 0
        self.dataset_fingerprint: Optional[bytes] = None

        self._seed_tune_buf: Optional[cl.Buffer] = None
        self._job_tune_buf: Optional[cl.Buffer] = None

        self._blob_buf: Optional[cl.Buffer] = None
        self._blob_bytes: Optional[bytes] = None
        self._blob_nbytes: int = 0

        self._seed_buf: Optional[cl.Buffer] = None
        self._seed_bytes: Optional[bytes] = None
        self._seed_nbytes: int = 0

        self._out_hashes_buf: Optional[cl.Buffer] = None
        self._out_nonces_buf: Optional[cl.Buffer] = None
        self._out_scores_buf: Optional[cl.Buffer] = None
        self._out_buckets_buf: Optional[cl.Buffer] = None
        self._out_rankq_buf: Optional[cl.Buffer] = None
        self._out_threshq_buf: Optional[cl.Buffer] = None
        self._out_tailbin_buf: Optional[cl.Buffer] = None
        self._out_count_buf: Optional[cl.Buffer] = None
        self._out_capacity: int = 0

        self._out_hashes_np: Optional[np.ndarray] = None
        self._out_nonces_np: Optional[np.ndarray] = None
        self._out_scores_np: Optional[np.ndarray] = None
        self._out_buckets_np: Optional[np.ndarray] = None
        self._out_rankq_np: Optional[np.ndarray] = None
        self._out_threshq_np: Optional[np.ndarray] = None
        self._out_tailbin_np: Optional[np.ndarray] = None
        self._out_count_np: Optional[np.ndarray] = None

        self._effective_local_work_size: Optional[int] = None

        self._bucket_count: int = max(
            1,
            self._extract_define_int(
                getattr(self.config, "build_options", "") or "",
                "BN_TUNE_WORDS",
                256,
            ),
        )
        self._local_stage_size: int = max(
            1,
            self._extract_define_int(
                getattr(self.config, "build_options", "") or "",
                "BN_LOCAL_STAGE_SIZE",
                128,
            ),
        )
        self._tail_bins: int = self.config.normalized_tail_bins()
        self._cells: int = self._bucket_count * self._tail_bins
        self._tune_neutral: int = 128
        self._tune_min: int = 16
        self._tune_max: int = 240

        self._seed_tune = _make_tune_state(self._cells, self._PLANE_COUNT)
        self._job_tune = _make_tune_state(self._cells, self._PLANE_COUNT)

        self._current_job_key: Optional[tuple[str, str, str, str]] = None

        self.last_scan_work_items: int = 0
        self.last_scan_chunk_count: int = 0
        self.last_effective_candidate_target: int = max(1, int(self.config.scan_candidate_target))
        self.last_job_age_ms: int = 0
        self.last_verify_pressure_q8: int = 0
        self.last_submit_pressure_q8: int = 0
        self.last_stale_risk_q8: int = 0

        self._cl_lock = threading.RLock()

    @staticmethod
    def list_devices() -> list[OpenCLDeviceInfo]:
        out: list[OpenCLDeviceInfo] = []
        for p_idx, platform in enumerate(cl.get_platforms()):
            for d_idx, device in enumerate(platform.get_devices()):
                out.append(
                    OpenCLDeviceInfo(
                        platform_index=p_idx,
                        device_index=d_idx,
                        platform_name=platform.name.strip(),
                        device_name=device.name.strip(),
                    )
                )
        return out

    def initialize(self) -> None:
        self._ensure_opencl_loader()

        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")

        if self.config.platform_index >= len(platforms):
            raise RuntimeError(f"Platform index out of range: {self.config.platform_index}")

        platform = platforms[self.config.platform_index]
        devices = platform.get_devices()
        if not devices:
            raise RuntimeError("No OpenCL devices found on selected platform")

        if self.config.device_index >= len(devices):
            raise RuntimeError(f"Device index out of range: {self.config.device_index}")

        device = devices[self.config.device_index]
        self.device = device
        self.ctx = cl.Context(devices=[device])
        self.queue = cl.CommandQueue(self.ctx, device=device)
        self.program = self._build_program(self.ctx, self.config.kernel_path, self.config.build_options)

        try:
            entry = self._selected_kernel_entry()
            self.kernel = getattr(self.program, entry)
        except AttributeError as exc:
            raise RuntimeError(f"Kernel entry not found: {self._selected_kernel_entry()}") from exc

        self._effective_local_work_size = self._choose_local_work_size()
        self._ensure_tune_buffers()

        self.on_log(f"[opencl] using {platform.name.strip()} / {device.name.strip()}")
        self.on_log(
            f"[opencl] mode={self._scan_mode()} "
            f"kernel={self._selected_kernel_entry()} "
            f"window={self.config.active_scan_window()} "
            f"lws={self._effective_local_work_size or 'auto'} "
            f"max_results={self.config.max_results} "
            f"buckets={self._bucket_count} "
            f"tail_bins={self._tail_bins} "
            f"local_stage_size={self._local_stage_size} "
            f"planes=rank+threshold+credit+confidence"
        )

        try:
            gmem = int(getattr(device, "global_mem_size", 0))
            if gmem > 0:
                self.on_log(f"[opencl] device global_mem={gmem / (1024**3):.2f} GiB")
        except Exception:
            pass

    def bind_dataset(self, dataset_u64: np.ndarray, dataset_fingerprint: Optional[bytes]) -> None:
        if self.ctx is None or self.queue is None:
            raise RuntimeError("OpenCL scanner not initialized")

        ds = np.ascontiguousarray(dataset_u64, dtype=np.uint64)

        if (
            self.dataset_buf is not None
            and self.dataset_fingerprint == dataset_fingerprint
            and self.dataset_words == int(ds.size)
        ):
            self.on_log("[opencl] RandomX dataset already bound for current seed")
            return

        mf = cl.mem_flags
        old_buf = self.dataset_buf
        new_buf = cl.Buffer(self.ctx, mf.READ_ONLY, max(1, int(ds.nbytes)))
        cl.enqueue_copy(self.queue, new_buf, ds).wait()
        self.queue.finish()

        self.dataset_buf = new_buf
        self.dataset_words = int(ds.size)
        self.dataset_fingerprint = dataset_fingerprint

        if old_buf is not None:
            self._release_buffer(old_buf)

        with self._cl_lock:
            self._reset_tune_state(self._seed_tune)
            self._reset_tune_state(self._job_tune)
            self._current_job_key = None
            self._upload_tuning_unlocked(force=True)

        mib = ds.nbytes / (1024.0 * 1024.0)
        self.on_log(
            f"[opencl] RandomX dataset ready on GPU: words={self.dataset_words} size={mib:.2f} MiB"
        )

    def record_feedback(
        self,
        share: CandidateShare,
        outcome: str,
        credited_work: float = 0.0,
        quality: float = 0.0,
    ) -> None:
        with self._cl_lock:
            cell = self._cell_from_candidate(share)
            if cell < 0:
                return

            self._write_state_feedback(self._seed_tune, cell, outcome, credited_work, quality)
            if self.config.enable_job_tuning and self._share_matches_current_job_unlocked(share):
                self._write_state_feedback(self._job_tune, cell, outcome, credited_work, quality)

    def bucket_score(self, share: CandidateShare) -> int:
        rank_q, threshold_q, credit_q, _conf_q = self._lookup_cell_qualities(share)
        return int((rank_q + threshold_q + credit_q) // 3)

    def candidate_sort_key(self, share: CandidateShare) -> tuple[int, int, int, int]:
        rank_q, threshold_q, credit_q, conf_q = self._lookup_cell_qualities(share)
        rank_score = int(getattr(share, "rank_score_u64", 0)) or int(getattr(share, "predicted_tail_u64", 0))
        predicted_tail = int(getattr(share, "predicted_tail_u64", 0))
        return (
            rank_score,
            -int(credit_q),
            -int(conf_q),
            predicted_tail if predicted_tail > 0 else 0xFFFFFFFFFFFFFFFF,
        )

    def close(self) -> None:
        with self._cl_lock:
            self._release_buffer(self.dataset_buf)
            self.dataset_buf = None
            self.dataset_words = 0
            self.dataset_fingerprint = None

            self._release_buffer(self._seed_tune_buf)
            self._seed_tune_buf = None
            self._release_buffer(self._job_tune_buf)
            self._job_tune_buf = None

            self._release_buffer(self._blob_buf)
            self._blob_buf = None
            self._blob_bytes = None
            self._blob_nbytes = 0

            self._release_buffer(self._seed_buf)
            self._seed_buf = None
            self._seed_bytes = None
            self._seed_nbytes = 0

            self._release_buffer(self._out_hashes_buf)
            self._out_hashes_buf = None
            self._release_buffer(self._out_nonces_buf)
            self._out_nonces_buf = None
            self._release_buffer(self._out_scores_buf)
            self._out_scores_buf = None
            self._release_buffer(self._out_buckets_buf)
            self._out_buckets_buf = None
            self._release_buffer(self._out_rankq_buf)
            self._out_rankq_buf = None
            self._release_buffer(self._out_threshq_buf)
            self._out_threshq_buf = None
            self._release_buffer(self._out_tailbin_buf)
            self._out_tailbin_buf = None
            self._release_buffer(self._out_count_buf)
            self._out_count_buf = None

            self._out_capacity = 0
            self._out_hashes_np = None
            self._out_nonces_np = None
            self._out_scores_np = None
            self._out_buckets_np = None
            self._out_rankq_np = None
            self._out_threshq_np = None
            self._out_tailbin_np = None
            self._out_count_np = None

            self._effective_local_work_size = None
            self._reset_tune_state(self._seed_tune)
            self._reset_tune_state(self._job_tune)
            self._current_job_key = None

    def scan(
        self,
        job: MiningJob,
        start_nonce: int,
        *,
        job_age_ms: int = 0,
        verify_pressure_q8: int = 0,
        submit_pressure_q8: int = 0,
        stale_risk_q8: int = 0,
        scan_candidate_target_override: Optional[int] = None,
        work_items_override: Optional[int] = None,
    ) -> list[CandidateShare]:
        if not all([self.ctx, self.queue, self.program, self.kernel]):
            raise RuntimeError("OpenCL scanner not initialized")

        self.last_scan_work_items = 0
        self.last_scan_chunk_count = 0
        self.last_effective_candidate_target = max(
            1,
            int(scan_candidate_target_override or self.config.scan_candidate_target),
        )
        self.last_job_age_ms = max(0, int(job_age_ms))
        self.last_verify_pressure_q8 = max(0, min(255, int(verify_pressure_q8)))
        self.last_submit_pressure_q8 = max(0, min(255, int(submit_pressure_q8)))
        self.last_stale_risk_q8 = max(0, min(255, int(stale_risk_q8)))

        mode = self._scan_mode()
        if mode == "hash_batch":
            return self._scan_hash_batch_mode(
                job,
                start_nonce,
                job_age_ms=job_age_ms,
                verify_pressure_q8=verify_pressure_q8,
                submit_pressure_q8=submit_pressure_q8,
                stale_risk_q8=stale_risk_q8,
                candidate_target_override=scan_candidate_target_override,
                work_items_override=work_items_override,
            )

        return self._scan_chunk_mode(
            job,
            start_nonce,
            job_age_ms=job_age_ms,
            verify_pressure_q8=verify_pressure_q8,
            submit_pressure_q8=submit_pressure_q8,
            stale_risk_q8=stale_risk_q8,
            candidate_target_override=scan_candidate_target_override,
            work_items_override=work_items_override,
        )

    def _scan_mode(self) -> str:
        return self.config.normalized_scan_mode()

    def _selected_kernel_entry(self) -> str:
        if self._scan_mode() == "hash_batch":
            return "blocknet_randomx_vm_hash_batch_ext"
        return "blocknet_randomx_vm_scan_ext"

    def _job_prefilter_target64(self, job: MiningJob) -> np.uint64:
        if getattr(job, "prefilter_target64", None) is not None:
            try:
                return np.uint64(int(job.prefilter_target64) & 0xFFFFFFFFFFFFFFFF)
            except Exception:
                pass
        return np.uint64(target_hex_to_prefilter_u64(job.target_hex))

    def _ensure_opencl_loader(self) -> None:
        loader = resolve_resource_path(self.config.opencl_loader)
        if os.name == "nt":
            ctypes.WinDLL(loader)
            self.on_log(f"[opencl] loaded {loader}")

    def _build_program(self, ctx: cl.Context, kernel_path: str, build_options: str) -> cl.Program:
        resolved_kernel_path = resolve_resource_path(kernel_path)

        self.on_log(f"[opencl] requested kernel_path={kernel_path}")
        self.on_log(f"[opencl] resolved kernel_path={resolved_kernel_path}")

        if not os.path.exists(resolved_kernel_path):
            raise FileNotFoundError(
                f"OpenCL kernel not found: requested={kernel_path} resolved={resolved_kernel_path}"
            )

        with open(resolved_kernel_path, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()

        prg = cl.Program(ctx, src)
        try:
            opts = build_options.split() if (build_options or "").strip() else []
            prg.build(options=opts)
        except Exception as exc:
            build_log = ""
            try:
                if self.device is not None:
                    build_log = prg.get_build_info(self.device, cl.program_build_info.LOG) or ""
            except Exception:
                pass
            raise RuntimeError(f"OpenCL build failed: {exc}\n{build_log}") from exc

        self.on_log("[opencl] kernel build succeeded")

        try:
            if self.device is not None:
                log = prg.get_build_info(self.device, cl.program_build_info.LOG)
                if log and str(log).strip():
                    self.on_log(f"[opencl-build-log]\n{log}")
        except Exception as exc:
            self.on_log(f"[opencl] failed to fetch build log: {exc}")

        return prg

    def _choose_local_work_size(self) -> Optional[int]:
        if self.device is None:
            requested = self.config.local_work_size
            if requested and requested > 0:
                return min(int(requested), int(self._local_stage_size))
            return None

        requested = self.config.local_work_size
        device_name = str(getattr(self.device, "name", "") or "").lower()
        vendor = str(getattr(self.device, "vendor", "") or "").lower()

        if requested is None or int(requested) <= 0:
            if "nvidia" in vendor or "nvidia" in device_name:
                requested = 128
            elif "amd" in vendor or "advanced micro devices" in vendor:
                requested = 256
            elif "intel" in vendor:
                requested = 64
            else:
                requested = 64

        try:
            max_wg = int(getattr(self.device, "max_work_group_size", requested))
            requested = max(1, min(int(requested), max_wg, int(self._local_stage_size)))
        except Exception:
            requested = max(1, min(int(requested), int(self._local_stage_size)))

        preferred_multiple = 0
        try:
            if self.kernel is not None and self.device is not None:
                preferred_multiple = int(
                    self.kernel.get_work_group_info(
                        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                        self.device,
                    )
                )
        except Exception:
            preferred_multiple = 0

        if preferred_multiple > 0 and requested >= preferred_multiple:
            requested = (requested // preferred_multiple) * preferred_multiple
            requested = max(preferred_multiple, requested)

        requested = max(1, min(int(requested), int(self._local_stage_size)))
        return requested

    def _ensure_tune_buffers(self) -> None:
        if self.ctx is None or self.queue is None:
            raise RuntimeError("OpenCL scanner not initialized")

        mf = cl.mem_flags
        size = max(1, int(self._PLANE_COUNT * self._cells))

        if self._seed_tune_buf is None:
            self._seed_tune_buf = cl.Buffer(self.ctx, mf.READ_ONLY, size)
        if self._job_tune_buf is None:
            self._job_tune_buf = cl.Buffer(self.ctx, mf.READ_ONLY, size)

        self._upload_tuning_unlocked(force=True)

    def _ensure_job_buffers(self, blob: bytes, seed: bytes) -> None:
        if self.ctx is None or self.queue is None:
            raise RuntimeError("OpenCL scanner not initialized")

        mf = cl.mem_flags

        if self._blob_buf is None or self._blob_nbytes != len(blob):
            self._release_buffer(self._blob_buf)
            self._blob_buf = cl.Buffer(self.ctx, mf.READ_ONLY, len(blob))
            self._blob_nbytes = len(blob)
            self._blob_bytes = None

        if self._blob_bytes != blob:
            blob_np = np.frombuffer(blob, dtype=np.uint8)
            cl.enqueue_copy(self.queue, self._blob_buf, blob_np).wait()
            self._blob_bytes = bytes(blob)

        if self._seed_buf is None or self._seed_nbytes != len(seed):
            self._release_buffer(self._seed_buf)
            self._seed_buf = cl.Buffer(self.ctx, mf.READ_ONLY, len(seed))
            self._seed_nbytes = len(seed)
            self._seed_bytes = None

        if self._seed_bytes != seed:
            seed_np = np.frombuffer(seed, dtype=np.uint8)
            cl.enqueue_copy(self.queue, self._seed_buf, seed_np).wait()
            self._seed_bytes = bytes(seed)

    def _ensure_output_buffers(self, max_results: int) -> None:
        if self.ctx is None:
            raise RuntimeError("OpenCL scanner not initialized")

        if (
            self._out_hashes_buf is not None
            and self._out_nonces_buf is not None
            and self._out_scores_buf is not None
            and self._out_buckets_buf is not None
            and self._out_rankq_buf is not None
            and self._out_threshq_buf is not None
            and self._out_tailbin_buf is not None
            and self._out_count_buf is not None
            and self._out_capacity == max_results
        ):
            return

        mf = cl.mem_flags

        self._release_buffer(self._out_hashes_buf)
        self._release_buffer(self._out_nonces_buf)
        self._release_buffer(self._out_scores_buf)
        self._release_buffer(self._out_buckets_buf)
        self._release_buffer(self._out_rankq_buf)
        self._release_buffer(self._out_threshq_buf)
        self._release_buffer(self._out_tailbin_buf)
        self._release_buffer(self._out_count_buf)

        self._out_hashes_np = np.empty((max_results, 32), dtype=np.uint8)
        self._out_nonces_np = np.empty((max_results,), dtype=np.uint32)
        self._out_scores_np = np.empty((max_results,), dtype=np.uint64)
        self._out_buckets_np = np.empty((max_results,), dtype=np.uint32)
        self._out_rankq_np = np.empty((max_results,), dtype=np.uint8)
        self._out_threshq_np = np.empty((max_results,), dtype=np.uint8)
        self._out_tailbin_np = np.empty((max_results,), dtype=np.uint8)
        self._out_count_np = np.zeros((1,), dtype=np.uint32)

        self._out_hashes_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self._out_hashes_np.nbytes)
        self._out_nonces_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self._out_nonces_np.nbytes)
        self._out_scores_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self._out_scores_np.nbytes)
        self._out_buckets_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self._out_buckets_np.nbytes)
        self._out_rankq_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self._out_rankq_np.nbytes)
        self._out_threshq_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self._out_threshq_np.nbytes)
        self._out_tailbin_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self._out_tailbin_np.nbytes)
        self._out_count_buf = cl.Buffer(self.ctx, mf.READ_WRITE, self._out_count_np.nbytes)
        self._out_capacity = max_results

    def _reset_out_count(self) -> None:
        if self.queue is None or self._out_count_buf is None:
            raise RuntimeError("OpenCL output buffers not initialized")

        zero = np.uint32(0)
        cl.enqueue_fill_buffer(
            self.queue,
            self._out_count_buf,
            zero,
            0,
            np.dtype(np.uint32).itemsize,
        ).wait()

    def _reset_tune_state(self, state: _TuneState) -> None:
        state.seen.fill(0.0)
        state.verified.fill(0.0)
        state.cpu_rejected.fill(0.0)
        state.pool_accepted.fill(0.0)
        state.pool_rejected.fill(0.0)
        state.pool_stale.fill(0.0)
        state.pool_duplicate.fill(0.0)
        state.pool_invalid.fill(0.0)
        state.pool_backend_error.fill(0.0)
        state.queue_drop.fill(0.0)
        state.verified_work.fill(0.0)
        state.quality_sum.fill(0.0)
        state.pressure_sum.fill(0.0)
        state.planes.fill(self._tune_neutral)
        state.dirty = True

    def _recompute_tune_planes(self, state: _TuneState) -> None:
        seen = state.seen
        verified = state.verified
        seen_safe = np.maximum(seen, 1e-6)
        verified_safe = np.maximum(verified, 1e-6)

        verify_rate = verified / seen_safe
        cpu_reject_rate = state.cpu_rejected / seen_safe
        accept_rate = state.pool_accepted / verified_safe
        reject_rate = state.pool_rejected / verified_safe
        stale_rate = state.pool_stale / verified_safe
        duplicate_rate = state.pool_duplicate / verified_safe
        invalid_rate = state.pool_invalid / verified_safe
        backend_error_rate = state.pool_backend_error / verified_safe
        queue_drop_rate = state.queue_drop / verified_safe

        avg_verified_work = state.verified_work / verified_safe
        avg_quality = state.quality_sum / verified_safe
        avg_pressure = state.pressure_sum / seen_safe

        neutral = float(self._tune_neutral)

        rank = np.full((self._cells,), neutral, dtype=np.float32)
        rank += verify_rate * 22.0 * float(self.config.tune_verified_reward)
        rank += accept_rate * 20.0 * float(self.config.tune_accepted_reward)
        rank += np.clip(np.log2(avg_verified_work + 1.0) * 8.0 * float(self.config.tune_work_bonus_scale), 0.0, 48.0)
        rank += np.clip(avg_quality * 14.0 * float(self.config.tune_quality_reward), 0.0, 48.0)
        rank -= cpu_reject_rate * 28.0 * float(self.config.tune_cpu_reject_penalty)
        rank -= reject_rate * 18.0 * float(self.config.tune_pool_reject_penalty)
        rank -= stale_rate * 16.0 * float(self.config.tune_stale_penalty)
        rank -= duplicate_rate * 8.0 * float(self.config.tune_duplicate_penalty)
        rank -= invalid_rate * 36.0 * float(self.config.tune_invalid_penalty)
        rank -= backend_error_rate * 8.0 * float(self.config.tune_backend_error_penalty)
        rank -= queue_drop_rate * 12.0
        rank -= np.clip(avg_pressure * 24.0 * float(self.config.tune_pressure_penalty), 0.0, 64.0)

        threshold = np.full((self._cells,), neutral, dtype=np.float32)
        threshold += verify_rate * 12.0 * float(self.config.tune_verified_reward)
        threshold += accept_rate * 10.0 * float(self.config.tune_accepted_reward)
        threshold += np.clip(avg_quality * 8.0 * float(self.config.tune_quality_reward), 0.0, 20.0)
        threshold -= cpu_reject_rate * 40.0 * float(self.config.tune_cpu_reject_penalty)
        threshold -= reject_rate * 22.0 * float(self.config.tune_pool_reject_penalty)
        threshold -= stale_rate * 18.0 * float(self.config.tune_stale_penalty)
        threshold -= duplicate_rate * 6.0 * float(self.config.tune_duplicate_penalty)
        threshold -= invalid_rate * 48.0 * float(self.config.tune_invalid_penalty)
        threshold -= backend_error_rate * 6.0 * float(self.config.tune_backend_error_penalty)
        threshold -= queue_drop_rate * 10.0
        threshold -= np.clip(avg_pressure * 30.0 * float(self.config.tune_pressure_penalty), 0.0, 72.0)

        credit = np.full((self._cells,), neutral, dtype=np.float32)
        credit += np.clip(np.log2(avg_verified_work + 1.0) * 16.0 * float(self.config.tune_work_bonus_scale), 0.0, 96.0)
        credit += np.clip(avg_quality * 24.0 * float(self.config.tune_quality_reward), 0.0, 64.0)
        credit += accept_rate * 18.0 * float(self.config.tune_accepted_reward)
        credit -= stale_rate * 10.0 * float(self.config.tune_stale_penalty)
        credit -= duplicate_rate * 6.0 * float(self.config.tune_duplicate_penalty)
        credit -= invalid_rate * 24.0 * float(self.config.tune_invalid_penalty)
        credit -= np.clip(avg_pressure * 10.0 * float(self.config.tune_pressure_penalty), 0.0, 32.0)

        confidence = np.clip((seen / max(1e-6, float(self.config.tune_confidence_div))) * 255.0, 0.0, 255.0)

        rank = np.clip(np.rint(rank), self._tune_min, self._tune_max).astype(np.uint8)
        threshold = np.clip(np.rint(threshold), self._tune_min, self._tune_max).astype(np.uint8)
        credit = np.clip(np.rint(credit), self._tune_min, self._tune_max).astype(np.uint8)
        confidence = np.clip(np.rint(confidence), 0, 255).astype(np.uint8)

        state.planes[0 * self._cells : 1 * self._cells] = rank
        state.planes[1 * self._cells : 2 * self._cells] = threshold
        state.planes[2 * self._cells : 3 * self._cells] = credit
        state.planes[3 * self._cells : 4 * self._cells] = confidence
        state.dirty = False

    def _write_state_feedback(
        self,
        state: _TuneState,
        cell: int,
        outcome: str,
        credited_work: float,
        quality: float,
    ) -> None:
        if cell < 0 or cell >= self._cells:
            return

        if outcome == "scan_seen":
            state.seen[cell] += 1.0
        elif outcome == "cpu_verified":
            state.verified[cell] += 1.0
            state.verified_work[cell] += max(0.0, float(credited_work))
            state.quality_sum[cell] += max(0.0, float(quality))
        elif outcome == "cpu_rejected":
            state.cpu_rejected[cell] += 1.0
        elif outcome == "pool_accepted":
            state.pool_accepted[cell] += 1.0
            state.verified_work[cell] += max(0.0, float(credited_work))
            state.quality_sum[cell] += max(0.0, float(quality))
        elif outcome == "pool_rejected":
            state.pool_rejected[cell] += 1.0
        elif outcome == "pool_stale":
            state.pool_stale[cell] += 1.0
        elif outcome == "pool_duplicate":
            state.pool_duplicate[cell] += 1.0
        elif outcome == "pool_invalid":
            state.pool_invalid[cell] += 1.0
        elif outcome == "pool_backend_error":
            state.pool_backend_error[cell] += 1.0
        elif outcome == "queue_drop":
            state.queue_drop[cell] += 1.0

        state.dirty = True

    def _record_scan_observation_unlocked(self, state: _TuneState, cell: int, pressure: float) -> None:
        if cell < 0 or cell >= self._cells:
            return
        state.seen[cell] += 1.0
        state.pressure_sum[cell] += max(0.0, float(pressure))
        state.dirty = True

    def _upload_tuning_unlocked(self, force: bool = False) -> None:
        if self.queue is None or self._seed_tune_buf is None or self._job_tune_buf is None:
            return

        if force or self._seed_tune.dirty:
            self._recompute_tune_planes(self._seed_tune)
            cl.enqueue_copy(self.queue, self._seed_tune_buf, self._seed_tune.planes).wait()

        if force or self._job_tune.dirty:
            self._recompute_tune_planes(self._job_tune)
            cl.enqueue_copy(self.queue, self._job_tune_buf, self._job_tune.planes).wait()

        self.queue.finish()

    def _bucket_from_hash_hex(self, gpu_hash_hex: str) -> int:
        try:
            raw = bytes.fromhex((gpu_hash_hex or "").strip())
            if len(raw) >= 16:
                word0 = int.from_bytes(raw[:8], "little", signed=False)
                word1 = int.from_bytes(raw[8:16], "little", signed=False)
                mixed = (word0 ^ ((word1 << 1) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
                return int(mixed % self._bucket_count)
        except Exception:
            pass
        return -1

    def _tail_bin_from_tail(self, tail_u64: int) -> int:
        bins = max(1, int(self._tail_bins))
        if bins <= 1:
            return 0
        tail_u64 = int(tail_u64) & 0xFFFFFFFFFFFFFFFF
        return min(bins - 1, max(0, int((tail_u64 * bins) >> 64)))

    def _cell_index(self, bucket: int, tail_bin: int) -> int:
        if bucket < 0 or tail_bin < 0:
            return -1
        if self._bucket_count <= 0 or self._tail_bins <= 0:
            return -1
        return (int(bucket) % self._bucket_count) * self._tail_bins + (int(tail_bin) % self._tail_bins)

    def _cell_from_candidate(self, share: CandidateShare) -> int:
        bucket = int(getattr(share, "tune_bucket", -1))
        if bucket < 0:
            bucket = self._bucket_from_hash_hex(getattr(share, "gpu_hash_hex", "") or "")
        if bucket < 0:
            try:
                bucket = int(int(getattr(share, "nonce", 0)) & 0xFFFFFFFF) % self._bucket_count
            except Exception:
                return -1

        predicted_tail = int(getattr(share, "predicted_tail_u64", 0))
        if predicted_tail <= 0:
            try:
                raw = bytes.fromhex(getattr(share, "gpu_hash_hex", "") or "")
                if len(raw) >= 32:
                    predicted_tail = int.from_bytes(raw[24:32], "little", signed=False)
            except Exception:
                predicted_tail = 0

        tail_bin = int(getattr(share, "tune_tail_bin", -1))
        if tail_bin < 0:
            tail_bin = self._tail_bin_from_tail(predicted_tail)

        return self._cell_index(bucket, tail_bin)

    def _lookup_cell_qualities(self, share: CandidateShare) -> tuple[int, int, int, int]:
        cell = self._cell_from_candidate(share)
        if cell < 0:
            return (128, 128, 128, 0)

        seed_rank = int(self._seed_tune.planes[self._PLANE_RANK * self._cells + cell])
        seed_threshold = int(self._seed_tune.planes[self._PLANE_THRESHOLD * self._cells + cell])
        seed_credit = int(self._seed_tune.planes[self._PLANE_CREDIT * self._cells + cell])
        seed_conf = int(self._seed_tune.planes[self._PLANE_CONFIDENCE * self._cells + cell])

        if self.config.enable_job_tuning and self._share_matches_current_job_unlocked(share):
            job_rank = int(self._job_tune.planes[self._PLANE_RANK * self._cells + cell])
            job_threshold = int(self._job_tune.planes[self._PLANE_THRESHOLD * self._cells + cell])
            job_credit = int(self._job_tune.planes[self._PLANE_CREDIT * self._cells + cell])
            job_conf = int(self._job_tune.planes[self._PLANE_CONFIDENCE * self._cells + cell])
        else:
            job_rank = job_threshold = job_credit = 128
            job_conf = 0

        def blend(seed_q: int, seed_c: int, job_q: int, job_c: int) -> int:
            sw = max(1, int(seed_c))
            jw = max(0, int(job_c)) * 2
            if jw <= 0:
                return seed_q
            total = sw + jw
            delta = ((seed_q - 128) * sw + (job_q - 128) * jw) / max(1, total)
            q = int(round(128 + delta))
            return max(0, min(255, q))

        rank_q = blend(seed_rank, seed_conf, job_rank, job_conf)
        threshold_q = blend(seed_threshold, seed_conf, job_threshold, job_conf)
        credit_q = blend(seed_credit, seed_conf, job_credit, job_conf)
        conf_q = max(seed_conf, job_conf)
        return rank_q, threshold_q, credit_q, conf_q

    def _job_key_from_job(self, job: MiningJob) -> tuple[str, str, str, str]:
        return (
            str(getattr(job, "session_id", "") or ""),
            str(getattr(job, "job_id", "") or ""),
            str(getattr(job, "seed_hash_hex", "") or "").lower(),
            str(getattr(job, "blob_hex", "") or "")[:128],
        )

    def _job_key_from_share(self, share: CandidateShare) -> tuple[str, str, str, str]:
        return (
            str(getattr(share, "session_id", "") or ""),
            str(getattr(share, "job_id", "") or ""),
            str(getattr(share, "seed_hash_hex", "") or "").lower(),
            str(getattr(share, "blob_hex", "") or "")[:128],
        )

    def _begin_job_unlocked(self, job: MiningJob) -> None:
        key = self._job_key_from_job(job)
        if key == self._current_job_key:
            return

        self._current_job_key = key
        self._reset_tune_state(self._job_tune)
        self.on_log(
            f"[opencl] job tuning reset for job={job.job_id} session={job.session_id or '-'}"
        )

    def _share_matches_current_job_unlocked(self, share: CandidateShare) -> bool:
        if self._current_job_key is None:
            return False
        return self._job_key_from_share(share) == self._current_job_key

    def _launch_local_size(self, work_items: int) -> Optional[tuple[int]]:
        lws = self._effective_local_work_size
        if lws is None or lws <= 0:
            return None
        if work_items % lws == 0:
            return (int(lws),)

        candidate = int(lws)
        while candidate > 1:
            candidate //= 2
            if candidate > 0 and work_items % candidate == 0:
                return (candidate,)
        return None

    def _trim_candidates_unlocked(
        self,
        candidates: list[CandidateShare],
        candidate_target: int,
    ) -> None:
        if len(candidates) <= candidate_target:
            return

        if self.config.sort_candidates:
            candidates.sort(key=self.candidate_sort_key)
            del candidates[candidate_target:]
        else:
            del candidates[candidate_target:]

    def _run_one_launch_unlocked(
        self,
        job: MiningJob,
        start_nonce: int,
        blob: bytes,
        seed: bytes,
        target64: np.uint64,
        work_items: int,
        max_results: int,
        *,
        job_age_ms: int,
        verify_pressure_q8: int,
        submit_pressure_q8: int,
        stale_risk_q8: int,
    ) -> tuple[list[CandidateShare], int]:
        if self.queue is None or self.dataset_buf is None:
            raise RuntimeError("OpenCL scanner not initialized")

        self._begin_job_unlocked(job)
        self._upload_tuning_unlocked(force=False)
        self._reset_out_count()

        evt = self.kernel(
            self.queue,
            (int(work_items),),
            self._launch_local_size(int(work_items)),
            self._blob_buf,
            np.uint32(len(blob)),
            np.uint32(self.config.nonce_offset),
            np.uint32(int(start_nonce) & 0xFFFFFFFF),
            target64,
            np.uint32(max_results),
            self._out_hashes_buf,
            self._out_nonces_buf,
            self._out_scores_buf,
            self._out_buckets_buf,
            self._out_rankq_buf,
            self._out_threshq_buf,
            self._out_tailbin_buf,
            self._out_count_buf,
            self._seed_buf,
            np.uint32(len(seed)),
            self.dataset_buf,
            np.uint32(self.dataset_words),
            self._seed_tune_buf,
            np.uint32(self._bucket_count),
            np.uint32(self._tail_bins),
            self._job_tune_buf,
            np.uint32(self._bucket_count if self.config.enable_job_tuning else 0),
            np.uint32(self._tail_bins if self.config.enable_job_tuning else 0),
            np.uint32(max(0, int(job_age_ms))),
            np.uint32(max(0, min(255, int(verify_pressure_q8)))),
            np.uint32(max(0, min(255, int(submit_pressure_q8)))),
            np.uint32(max(0, min(255, int(stale_risk_q8)))),
        )
        evt.wait()

        if any(
            x is None
            for x in (
                self._out_hashes_np,
                self._out_nonces_np,
                self._out_scores_np,
                self._out_buckets_np,
                self._out_rankq_np,
                self._out_threshq_np,
                self._out_tailbin_np,
                self._out_count_np,
            )
        ):
            raise RuntimeError("OpenCL output host buffers are not initialized")

        cl.enqueue_copy(self.queue, self._out_count_np, self._out_count_buf).wait()

        raw_count = int(self._out_count_np[0])
        count = min(raw_count, max_results)
        candidates: list[CandidateShare] = []

        if count > 0:
            cl.enqueue_copy(self.queue, self._out_nonces_np[:count], self._out_nonces_buf).wait()
            cl.enqueue_copy(self.queue, self._out_hashes_np[:count, :], self._out_hashes_buf).wait()
            cl.enqueue_copy(self.queue, self._out_scores_np[:count], self._out_scores_buf).wait()
            cl.enqueue_copy(self.queue, self._out_buckets_np[:count], self._out_buckets_buf).wait()
            cl.enqueue_copy(self.queue, self._out_rankq_np[:count], self._out_rankq_buf).wait()
            cl.enqueue_copy(self.queue, self._out_threshq_np[:count], self._out_threshq_buf).wait()
            cl.enqueue_copy(self.queue, self._out_tailbin_np[:count], self._out_tailbin_buf).wait()

            pressure = float(raw_count) / float(max(1, max_results))
            for i in range(count):
                row = self._out_hashes_np[i]
                gpu_hash_hex = bytes(row).hex()
                predicted_tail = int.from_bytes(bytes(row[24:32]), "little", signed=False)
                bucket = int(self._out_buckets_np[i]) % self._bucket_count
                tail_bin = int(self._out_tailbin_np[i]) % self._tail_bins
                cell = self._cell_index(bucket, tail_bin)

                self._record_scan_observation_unlocked(self._seed_tune, cell, pressure)
                if self.config.enable_job_tuning:
                    self._record_scan_observation_unlocked(self._job_tune, cell, pressure)

                candidates.append(
                    CandidateShare(
                        nonce=int(self._out_nonces_np[i]),
                        gpu_hash_hex=gpu_hash_hex,
                        job_id=job.job_id,
                        blob_hex=job.blob_hex,
                        session_id=job.session_id,
                        target_hex=job.target_hex,
                        seed_hash_hex=job.seed_hash_hex,
                        predicted_tail_u64=predicted_tail,
                        rank_score_u64=int(self._out_scores_np[i]),
                        tune_bucket=bucket,
                        tune_tail_bin=tail_bin,
                        rank_quality=int(self._out_rankq_np[i]),
                        threshold_quality=int(self._out_threshq_np[i]),
                        job_age_ms=max(0, int(job_age_ms)),
                        verify_pressure_q8=max(0, min(255, int(verify_pressure_q8))),
                        submit_pressure_q8=max(0, min(255, int(submit_pressure_q8))),
                        stale_risk_q8=max(0, min(255, int(stale_risk_q8))),
                    )
                )

        if raw_count >= max_results:
            self.on_log(
                f"[opencl] candidate buffer saturated for job={job.job_id}: "
                f"raw_count={raw_count}, max_results={max_results}"
            )

        return candidates, raw_count

    def _scan_chunk_mode(
        self,
        job: MiningJob,
        start_nonce: int,
        *,
        job_age_ms: int,
        verify_pressure_q8: int,
        submit_pressure_q8: int,
        stale_risk_q8: int,
        candidate_target_override: Optional[int],
        work_items_override: Optional[int],
    ) -> list[CandidateShare]:
        if self.dataset_buf is None or self.dataset_words <= 0:
            return []

        blob = safe_bytes_from_hex(job.blob_hex)
        if not blob:
            self.on_log(f"[opencl] empty/invalid blob for job={job.job_id}")
            return []

        if len(blob) < self.config.nonce_offset + 4:
            self.on_log(
                f"[opencl] blob too short for nonce offset {self.config.nonce_offset}: "
                f"len={len(blob)} job={job.job_id}"
            )
            return []

        seed = safe_bytes_from_hex(job.seed_hash_hex)
        if not seed and (job.algo or "rx/0").lower() == "rx/0":
            self.on_log(f"[opencl] missing seed_hash for rx/0 job={job.job_id}, skipping scan")
            return []

        if not seed:
            seed = blob[:32].ljust(32, b"\x00")

        target64 = self._job_prefilter_target64(job)
        max_results = int(self.config.max_results)
        total_work = max(1, int(work_items_override or self.config.global_work_size))
        chunk_size = max(1, int(getattr(self.config, "scan_chunk_size", 262144)))
        candidate_target = max(
            1,
            min(
                int(candidate_target_override or self.config.scan_candidate_target),
                max_results,
            ),
        )
        max_scan_time_ms = max(0, int(getattr(self.config, "max_scan_time_ms", 15)))

        all_candidates: list[CandidateShare] = []
        scanned = 0
        chunk_count = 0
        t0 = threading.get_native_id()  # dummy init to keep local symbol count simple
        del t0
        t_start = np.float64(0.0)
        t_start = np.float64(cl._cl._get_cl_version() if False else 0.0)  # no-op, keeps import use quiet

        import time as _time

        started = _time.perf_counter()
        early_stop_reason = ""

        with self._cl_lock:
            self._ensure_job_buffers(blob, seed)
            self._ensure_output_buffers(max_results)

            while scanned < total_work:
                remaining = total_work - scanned
                this_chunk = min(chunk_size, remaining)
                if this_chunk <= 0:
                    break

                chunk_start_nonce = (int(start_nonce) + scanned) & 0xFFFFFFFF
                chunk_candidates, _raw_count = self._run_one_launch_unlocked(
                    job=job,
                    start_nonce=chunk_start_nonce,
                    blob=blob,
                    seed=seed,
                    target64=target64,
                    work_items=int(this_chunk),
                    max_results=max_results,
                    job_age_ms=job_age_ms,
                    verify_pressure_q8=verify_pressure_q8,
                    submit_pressure_q8=submit_pressure_q8,
                    stale_risk_q8=stale_risk_q8,
                )

                if chunk_candidates:
                    all_candidates.extend(chunk_candidates)
                    self._trim_candidates_unlocked(all_candidates, candidate_target)

                scanned += int(this_chunk)
                chunk_count += 1

                if len(all_candidates) >= candidate_target:
                    early_stop_reason = f"candidate_target={candidate_target}"
                    break

                if max_scan_time_ms > 0:
                    elapsed_ms = (_time.perf_counter() - started) * 1000.0
                    if elapsed_ms >= float(max_scan_time_ms):
                        early_stop_reason = f"time_budget_ms={max_scan_time_ms}"
                        break

            self._upload_tuning_unlocked(force=False)
            self.queue.finish()

        self.last_scan_work_items = scanned
        self.last_scan_chunk_count = chunk_count

        if chunk_count > 1 or early_stop_reason:
            self.on_log(
                f"[opencl] chunked scan job={job.job_id} chunks={chunk_count} "
                f"scanned={scanned}/{total_work} kept={len(all_candidates)} "
                + (f"stop={early_stop_reason} " if early_stop_reason else "")
                + f"job_age_ms={job_age_ms} verify_q8={verify_pressure_q8} "
                  f"submit_q8={submit_pressure_q8} stale_q8={stale_risk_q8}"
            )

        return all_candidates

    def _scan_hash_batch_mode(
        self,
        job: MiningJob,
        start_nonce: int,
        *,
        job_age_ms: int,
        verify_pressure_q8: int,
        submit_pressure_q8: int,
        stale_risk_q8: int,
        candidate_target_override: Optional[int],
        work_items_override: Optional[int],
    ) -> list[CandidateShare]:
        if self.dataset_buf is None or self.dataset_words <= 0:
            return []

        blob = safe_bytes_from_hex(job.blob_hex)
        if not blob:
            self.on_log(f"[opencl] empty/invalid blob for job={job.job_id}")
            return []

        if len(blob) < self.config.nonce_offset + 4:
            self.on_log(
                f"[opencl] blob too short for nonce offset {self.config.nonce_offset}: "
                f"len={len(blob)} job={job.job_id}"
            )
            return []

        seed = safe_bytes_from_hex(job.seed_hash_hex)
        if not seed and (job.algo or "rx/0").lower() == "rx/0":
            self.on_log(f"[opencl] missing seed_hash for rx/0 job={job.job_id}, skipping hash_batch")
            return []

        if not seed:
            seed = blob[:32].ljust(32, b"\x00")

        target64 = self._job_prefilter_target64(job)
        max_results = int(self.config.max_results)
        work_items = max(1, int(work_items_override or getattr(self.config, "hash_batch_size", self.config.global_work_size)))
        candidate_target = max(
            1,
            min(
                int(candidate_target_override or self.config.scan_candidate_target),
                max_results,
            ),
        )

        with self._cl_lock:
            self._ensure_job_buffers(blob, seed)
            self._ensure_output_buffers(max_results)

            candidates, _raw_count = self._run_one_launch_unlocked(
                job=job,
                start_nonce=int(start_nonce),
                blob=blob,
                seed=seed,
                target64=target64,
                work_items=work_items,
                max_results=max_results,
                job_age_ms=job_age_ms,
                verify_pressure_q8=verify_pressure_q8,
                submit_pressure_q8=submit_pressure_q8,
                stale_risk_q8=stale_risk_q8,
            )

            self._trim_candidates_unlocked(candidates, candidate_target)
            self._upload_tuning_unlocked(force=False)
            self.queue.finish()

        self.last_scan_work_items = int(work_items)
        self.last_scan_chunk_count = 1

        self.on_log(
            f"[opencl] hash_batch job={job.job_id} work_items={work_items} kept={len(candidates)} "
            f"job_age_ms={job_age_ms} verify_q8={verify_pressure_q8} "
            f"submit_q8={submit_pressure_q8} stale_q8={stale_risk_q8}"
        )

        return candidates

    @staticmethod
    def _extract_define_int(build_options: str, name: str, default: int) -> int:
        if not build_options:
            return int(default)
        m = re.search(rf"-D{name}=([0-9]+)", build_options)
        if not m:
            return int(default)
        try:
            return int(m.group(1))
        except Exception:
            return int(default)

    @staticmethod
    def _release_buffer(buf: Optional[cl.Buffer]) -> None:
        if buf is None:
            return
        try:
            buf.release()
        except Exception:
            pass