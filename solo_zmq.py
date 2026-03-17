from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from models import MinerConfig, MiningJob, SubmitResult, VerifiedShare

try:
    import zmq
except Exception:
    zmq = None


@dataclass
class SoloTemplateState:
    height: int = 0
    prev_hash: str = ""
    seed_hash: str = ""
    job_id: str = ""
    submit_blob_hex: str = ""
    expected_reward_atomic: int = 0
    template_fingerprint: str = ""


class MoneroZmqReader:
    def __init__(self, url: str, on_log: Callable[[str], None], on_event: Callable[[], None]) -> None:
        self.url = (url or "").strip()
        self.on_log = on_log
        self.on_event = on_event

        self._ctx = None
        self._sock = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if not self.url:
            return
        if zmq is None:
            self.on_log("[solo-zmq] pyzmq is not installed; ZMQ notifications disabled")
            return

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.setsockopt(zmq.RCVTIMEO, 1000)
        self._sock.connect(self.url)

        self._thread = threading.Thread(target=self._run, name="MoneroZmqReader", daemon=True)
        self._thread.start()
        self.on_log(f"[solo-zmq] subscribed to {self.url}")

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sock is not None:
            try:
                self._sock.close(0)
            except Exception:
                pass
            self._sock = None

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._sock is None:
                time.sleep(0.25)
                continue
            try:
                _frames = self._sock.recv_multipart()
                self.on_event()
            except Exception:
                continue


class SoloMiningConnection:
    def __init__(
        self,
        config: MinerConfig,
        on_log: Callable[[str], None],
        on_job: Callable[[MiningJob], None],
        on_status: Callable[[str], None],
    ) -> None:
        self.config = config
        self.on_log = on_log
        self.on_job = on_job
        self.on_status = on_status

        self._stop = threading.Event()
        self._wake = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._zmq_reader: Optional[MoneroZmqReader] = None
        self._last = SoloTemplateState()

        self._template_lock = threading.Lock()
        self._expected_reward_by_job_id: dict[str, int] = {}

    def connect(self) -> None:
        self.on_status("connecting")
        self._refresh_template(force_emit=True)

        if self.config.solo_use_zmq and self.config.solo_zmq_pub_url.strip():
            self._zmq_reader = MoneroZmqReader(
                self.config.solo_zmq_pub_url.strip(),
                on_log=self.on_log,
                on_event=self._wake.set,
            )
            self._zmq_reader.start()

        self._poll_thread = threading.Thread(target=self._poll_loop, name="SoloPollThread", daemon=True)
        self._poll_thread.start()
        self.on_status("connected")

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None
        if self._zmq_reader is not None:
            self._zmq_reader.close()
            self._zmq_reader = None

    def submit(self, verified: VerifiedShare) -> SubmitResult:
        blob = (verified.solution_blob_hex or "").strip()
        if not blob:
            return SubmitResult(
                accepted=False,
                status="missing_solution_blob",
                error="solo submission requires solution_blob_hex",
                reject_class="invalid",
                invalid=True,
            )

        current_job_id = self._last.job_id
        if current_job_id and verified.job_id and verified.job_id != current_job_id:
            self.on_log(
                f"[submit] solo template moved share_job={verified.job_id} current={current_job_id}; "
                "submitting exact solved blob anyway"
            )

        self.on_log(
            f"[submit] sent backend=solo nonce={verified.nonce_hex} job={verified.job_id} "
            f"credited={verified.credited_work:.6f} "
            f"actual={verified.actual_work:.6f} "
            f"quality={verified.quality:.6f}"
        )

        try:
            resp = self._rpc_call("submit_block", [blob])
            result = resp.get("result") or {}
            status = str(result.get("status", "OK"))
            accepted = "OK" in status.upper()

            reward_atomic = 0
            with self._template_lock:
                reward_atomic = int(self._expected_reward_by_job_id.get(verified.job_id, 0) or 0)

            raw = {
                "rpc": resp,
                "solo_expected_reward_atomic": reward_atomic,
                "solo_expected_reward_xmr": (float(reward_atomic) / 1e12) if reward_atomic > 0 else 0.0,
                "solo_wallet_address": (self.config.solo_wallet_address or "").strip(),
            }

            if accepted:
                self.on_log(
                    f"[submit] solo block accepted nonce={verified.nonce_hex} job={verified.job_id} "
                    f"expected_reward_atomic={reward_atomic}"
                )
                return SubmitResult(
                    accepted=True,
                    status=status,
                    error="",
                    raw=raw,
                )

            return SubmitResult(
                accepted=False,
                status=status,
                error=status,
                raw=raw,
                reject_class="pool_rejected",
            )
        except Exception as exc:
            return SubmitResult(
                accepted=False,
                status="rpc_error",
                error=str(exc),
                reject_class="backend_error",
                backend_error=True,
            )

    def _poll_loop(self) -> None:
        fallback_s = max(0.25, float(self.config.solo_poll_fallback_s))
        while not self._stop.is_set():
            _fired = self._wake.wait(timeout=fallback_s)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                self._refresh_template(force_emit=False)
            except Exception as exc:
                self.on_log(f"[solo] template refresh failed: {exc}")

    def _refresh_template(self, force_emit: bool) -> None:
        _ = force_emit  # kept for signature compatibility

        wallet = (self.config.solo_wallet_address or "").strip()
        if not wallet:
            raise RuntimeError("solo wallet address is required")

        params = {
            "wallet_address": wallet,
            "reserve_size": int(max(0, self.config.solo_reserve_size)),
        }
        resp = self._rpc_call("get_block_template", params)
        result = resp.get("result") or {}

        if bool(result.get("untrusted", False)):
            self.on_log("[solo] ignoring untrusted block template")
            return

        height = int(result.get("height", 0))
        prev_hash = str(result.get("prev_hash", "")).strip().lower()
        seed_hash = str(result.get("seed_hash", "")).strip().lower()

        submit_blob_hex = str(result.get("blocktemplate_blob", "") or "").strip()
        hashing_blob_hex = str(result.get("blockhashing_blob", "") or submit_blob_hex).strip()
        expected_reward_atomic = int(result.get("expected_reward", 0) or 0)

        if not submit_blob_hex:
            raise RuntimeError("get_block_template returned empty blocktemplate_blob")
        if not hashing_blob_hex:
            raise RuntimeError("get_block_template returned empty blockhashing_blob")

        template_fingerprint = hashlib.sha256(
            submit_blob_hex.encode("ascii", errors="ignore")
        ).hexdigest()[:16]

        job_id = f"solo:{height}:{prev_hash[:16]}:{seed_hash[:16]}:{template_fingerprint}"

        if job_id == self._last.job_id:
            return

        prefilter_u64 = self._difficulty_result_to_prefilter_u64(result)
        target_hex = int(prefilter_u64 & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False).hex()

        job = MiningJob(
            job_id=job_id,
            blob_hex=hashing_blob_hex,
            submit_blob_hex=submit_blob_hex,
            target_hex=target_hex,
            session_id="solo",
            seed_hash_hex=seed_hash,
            height=height,
            algo="rx/0",
            reserved_offset=int(result.get("reserved_offset", 0) or 0),
            prefilter_target64=int(prefilter_u64),
            backend="solo",
        )

        self._last = SoloTemplateState(
            height=height,
            prev_hash=prev_hash,
            seed_hash=seed_hash,
            job_id=job_id,
            submit_blob_hex=submit_blob_hex,
            expected_reward_atomic=expected_reward_atomic,
            template_fingerprint=template_fingerprint,
        )

        with self._template_lock:
            self._expected_reward_by_job_id[job_id] = expected_reward_atomic
            while len(self._expected_reward_by_job_id) > 64:
                oldest = next(iter(self._expected_reward_by_job_id))
                self._expected_reward_by_job_id.pop(oldest, None)

        self.on_log(
            f"[solo] new template height={height} prev={prev_hash[:16]} "
            f"seed={seed_hash[:16]} tpl={template_fingerprint} "
            f"target64=0x{prefilter_u64:016x} expected_reward_atomic={expected_reward_atomic}"
        )
        self.on_job(job)

    def _rpc_call(self, method: str, params) -> dict:
        url = self._normalize_rpc_url(self.config.solo_daemon_rpc_url)
        body = json.dumps(
            {"jsonrpc": "2.0", "id": "0", "method": method, "params": params}
        ).encode("utf-8")

        req = urllib.request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} HTTP {exc.code}: {raw}") from exc
        except Exception as exc:
            raise RuntimeError(f"{method} request failed: {exc}") from exc

        data = json.loads(raw)
        if "error" in data and data["error"]:
            raise RuntimeError(f"{method} RPC error: {data['error']}")
        return data

    @staticmethod
    def _normalize_rpc_url(url: str) -> str:
        u = (url or "").strip()
        if not u:
            return "http://127.0.0.1:18081/json_rpc"
        if not u.startswith(("http://", "https://")):
            u = "http://" + u
        if not u.endswith("/json_rpc"):
            u = u.rstrip("/") + "/json_rpc"
        return u

    @staticmethod
    def _difficulty_result_to_prefilter_u64(result: dict) -> int:
        wide_hex = str(result.get("wide_difficulty", "") or "").strip().lower()
        if wide_hex.startswith("0x"):
            wide_hex = wide_hex[2:]

        difficulty = 0
        if wide_hex:
            try:
                difficulty = int(wide_hex, 16)
            except Exception:
                difficulty = 0

        if difficulty <= 0:
            try:
                difficulty = int(result.get("difficulty", 0) or 0)
                difficulty_top64 = int(result.get("difficulty_top64", 0) or 0)
                if difficulty_top64 > 0:
                    difficulty = (difficulty_top64 << 64) | difficulty
            except Exception:
                difficulty = 0

        if difficulty <= 0:
            return 0x0000FFFFFFFFFFFF

        full_target = ((1 << 256) - 1) // difficulty
        low64 = full_target & 0xFFFFFFFFFFFFFFFF
        if low64 <= 0:
            low64 = 1
        return int(low64)