from __future__ import annotations

import json
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from models import MinerConfig, MiningJob, SubmitResult, VerifiedShare


def _normalize_hex(text: str) -> str:
    return "".join(ch for ch in str(text or "").strip().lower() if not ch.isspace())


def _is_hex(text: str) -> bool:
    s = _normalize_hex(text)
    if not s:
        return False
    try:
        bytes.fromhex(s)
        return True
    except Exception:
        return False


def _stringify_error(error: Any) -> str:
    if isinstance(error, dict):
        msg = str(error.get("message") or error.get("error") or "").strip()
        code = error.get("code")
        if msg and code is not None:
            return f"{msg} (code={code})"
        if msg:
            return msg
        if code is not None:
            return f"code={code}"
    return str(error or "").strip()


def _classify_reject(status: str, error: str, raw: Any) -> tuple[str, bool, bool, bool, bool]:
    text = " ".join(
        [
            str(status or ""),
            str(error or ""),
            str(raw or ""),
        ]
    ).lower()

    stale = "stale" in text
    duplicate = (
        "duplicate" in text
        or "already submitted" in text
        or "duplicate_local" in text
    )
    invalid = (
        "invalid" in text
        or "low difficulty" in text
        or "bad nonce" in text
        or "bad result" in text
        or "invalid share" in text
        or "malformed" in text
    )
    backend_error = (
        "timeout" in text
        or "socket" in text
        or "connect" in text
        or "connection" in text
        or "ioerror" in text
        or "broken pipe" in text
        or "not connected" in text
    )

    if stale:
        return ("stale", True, False, False, False)
    if duplicate:
        return ("duplicate", False, True, False, False)
    if invalid:
        return ("invalid", False, False, True, False)
    if backend_error:
        return ("backend_error", False, False, False, True)
    return ("rejected", False, False, False, False)


@dataclass
class PendingRequest:
    event: threading.Event = field(default_factory=threading.Event)
    response: Optional[Dict[str, Any]] = None


class StratumClient:
    def __init__(
        self,
        config: MinerConfig,
        *,
        on_log: Callable[[str], None],
        on_job: Callable[[MiningJob], None],
        on_status: Callable[[str], None],
    ) -> None:
        self.config = config
        self.on_log = on_log
        self.on_job = on_job
        self.on_status = on_status

        self._sock: Optional[socket.socket] = None
        self._file = None
        self._reader_thread: Optional[threading.Thread] = None
        self._keepalive_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._write_lock = threading.Lock()
        self._id_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._recent_submit_lock = threading.Lock()

        self._req_id = 1
        self._pending: Dict[int, PendingRequest] = {}
        self._recent_submits: dict[tuple[str, str, str], float] = {}

        self._session_id = ""
        self._current_job: Optional[MiningJob] = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def current_job(self) -> Optional[MiningJob]:
        return self._current_job

    def connect(self) -> None:
        self.on_status("connecting")

        raw = socket.create_connection((self.config.host, self.config.port), timeout=10)

        if self.config.use_tls:
            ctx = ssl.create_default_context()
            self._sock = ctx.wrap_socket(raw, server_hostname=self.config.host)
        else:
            self._sock = raw

        self._sock.settimeout(None)
        self._file = self._sock.makefile("r", encoding="utf-8", newline="\n")
        self._stop.clear()

        self.on_status("connected")
        self.on_log(f"[stratum] connected to {self.config.host}:{self.config.port}")

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="StratumReader",
        )
        self._reader_thread.start()

        self._send_login()

        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop,
            daemon=True,
            name="StratumKeepalive",
        )
        self._keepalive_thread.start()

    def close(self) -> None:
        self._stop.set()

        try:
            if self._file:
                self._file.close()
        except Exception:
            pass

        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass

        with self._pending_lock:
            for pending in self._pending.values():
                pending.event.set()
            self._pending.clear()

        self._sock = None
        self._file = None
        self._session_id = ""
        self._current_job = None
        self.on_status("disconnected")

    def submit(self, share: VerifiedShare, timeout: float = 10.0) -> SubmitResult:
        if not self._sock:
            return SubmitResult(
                False,
                status="not_connected",
                error="socket not connected",
                reject_class="backend_error",
                backend_error=True,
            )

        session_id = (share.session_id or self._session_id).strip()
        if not session_id:
            return SubmitResult(
                False,
                status="not_connected",
                error="missing stratum session id",
                reject_class="backend_error",
                backend_error=True,
            )

        job_id = str(share.job_id or "").strip()
        nonce_hex = _normalize_hex(share.nonce_hex)
        result_hex = _normalize_hex(share.result_hex)

        if not job_id:
            return SubmitResult(
                False,
                status="invalid_share",
                error="missing job_id",
                reject_class="invalid",
                invalid=True,
            )
        if not nonce_hex:
            return SubmitResult(
                False,
                status="invalid_share",
                error="missing nonce_hex",
                reject_class="invalid",
                invalid=True,
            )
        if not result_hex:
            return SubmitResult(
                False,
                status="invalid_share",
                error="missing result_hex",
                reject_class="invalid",
                invalid=True,
            )
        if not _is_hex(nonce_hex):
            return SubmitResult(
                False,
                status="invalid_share",
                error="nonce_hex is not valid hex",
                reject_class="invalid",
                invalid=True,
            )
        if not _is_hex(result_hex):
            return SubmitResult(
                False,
                status="invalid_share",
                error="result_hex is not valid hex",
                reject_class="invalid",
                invalid=True,
            )

        current_job = self._current_job
        if current_job and current_job.job_id and current_job.job_id != job_id:
            return SubmitResult(
                False,
                status="STALE",
                error=f"stale share job={job_id}, current={current_job.job_id}",
                reject_class="stale",
                stale=True,
            )

        if self._mark_submit_seen(job_id, nonce_hex, result_hex):
            return SubmitResult(
                False,
                status="duplicate_local",
                error="share already submitted recently",
                reject_class="duplicate",
                duplicate=True,
            )

        req_id = self._next_id()
        pending = PendingRequest()

        with self._pending_lock:
            self._pending[req_id] = pending

        payload = {
            "id": req_id,
            "jsonrpc": "2.0",
            "method": "submit",
            "params": {
                "id": session_id,
                "job_id": job_id,
                "nonce": nonce_hex,
                "result": result_hex,
            },
        }

        try:
            self._send(payload)
            self.on_log(
                f"[submit] sent backend=stratum nonce={nonce_hex} job={job_id} "
                f"credited={share.credited_work:.6f} "
                f"actual={share.actual_work:.6f} "
                f"quality={share.quality:.6f}"
            )

            if not pending.event.wait(timeout):
                return SubmitResult(
                    False,
                    status="TIMEOUT",
                    error="submit timed out",
                    reject_class="backend_error",
                    backend_error=True,
                )

            resp = pending.response or {}
            error = resp.get("error")
            result = resp.get("result")

            if error:
                err_text = _stringify_error(error)
                reject_class, stale, duplicate, invalid, backend_error = _classify_reject(
                    "ERROR",
                    err_text,
                    resp,
                )
                return SubmitResult(
                    False,
                    status="ERROR",
                    error=err_text,
                    raw=resp,
                    reject_class=reject_class,
                    stale=stale,
                    duplicate=duplicate,
                    invalid=invalid,
                    backend_error=backend_error,
                )

            if isinstance(result, dict):
                status = str(result.get("status") or "").strip()
                if status and status.upper() not in ("OK", "ACCEPTED"):
                    err_text = str(result.get("error") or status)
                    reject_class, stale, duplicate, invalid, backend_error = _classify_reject(
                        status,
                        err_text,
                        resp,
                    )
                    return SubmitResult(
                        False,
                        status=status,
                        error=err_text,
                        raw=resp,
                        reject_class=reject_class,
                        stale=stale,
                        duplicate=duplicate,
                        invalid=invalid,
                        backend_error=backend_error,
                    )
                return SubmitResult(True, status=status or "OK", raw=resp, reject_class="accepted")

            if isinstance(result, str):
                status = result.strip()
                if status.upper() in ("OK", "ACCEPTED"):
                    return SubmitResult(True, status=status, raw=resp, reject_class="accepted")

                reject_class, stale, duplicate, invalid, backend_error = _classify_reject(
                    status,
                    status,
                    resp,
                )
                return SubmitResult(
                    False,
                    status=status or "ERROR",
                    error=status,
                    raw=resp,
                    reject_class=reject_class,
                    stale=stale,
                    duplicate=duplicate,
                    invalid=invalid,
                    backend_error=backend_error,
                )

            return SubmitResult(True, status="OK", raw=resp, reject_class="accepted")

        finally:
            with self._pending_lock:
                self._pending.pop(req_id, None)

    def _mark_submit_seen(self, job_id: str, nonce_hex: str, result_hex: str) -> bool:
        now = time.time()
        key = (job_id, nonce_hex, result_hex)

        with self._recent_submit_lock:
            cutoff = now - 120.0
            dead = [k for k, ts in self._recent_submits.items() if ts < cutoff]
            for k in dead:
                self._recent_submits.pop(k, None)

            if key in self._recent_submits:
                return True

            self._recent_submits[key] = now
            return False

    def _next_id(self) -> int:
        with self._id_lock:
            rid = self._req_id
            self._req_id += 1
            return rid

    def _send_login(self) -> None:
        payload = {
            "id": self._next_id(),
            "jsonrpc": "2.0",
            "method": "login",
            "params": {
                "login": self.config.login,
                "pass": self.config.password,
                "agent": self.config.agent,
            },
        }
        self._send(payload)
        self.on_log(f"[stratum] login as {self.config.login}")

    def _keepalive_loop(self) -> None:
        while not self._stop.wait(30.0):
            if not self._session_id:
                continue

            payload = {
                "id": self._next_id(),
                "jsonrpc": "2.0",
                "method": "keepalived",
                "params": {"id": self._session_id},
            }

            try:
                self._send(payload)
            except Exception as exc:
                self.on_log(f"[keepalive] error: {exc}")
                return

    def _send(self, payload: Dict[str, Any]) -> None:
        if not self._sock:
            raise RuntimeError("socket not connected")

        wire = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        with self._write_lock:
            self._sock.sendall(wire)

    def _reader_loop(self) -> None:
        try:
            while not self._stop.is_set() and self._file is not None:
                line = self._file.readline()
                if not line:
                    raise ConnectionError("stratum connection closed")
                self._handle_message(json.loads(line))
        except Exception as exc:
            self.on_log(f"[stratum] reader stopped: {exc}")
            self.close()

    def _handle_message(self, msg: Dict[str, Any]) -> None:
        method = msg.get("method")
        if method == "job":
            self._handle_job(msg.get("params") or {})
            return

        if "id" in msg:
            req_id = msg.get("id")
            with self._pending_lock:
                pending = self._pending.get(req_id)
            if pending:
                pending.response = msg
                pending.event.set()

        result = msg.get("result") or {}
        error = msg.get("error")
        if error:
            self.on_log(f"[stratum] error: {_stringify_error(error)}")
            return

        if isinstance(result, dict) and result.get("job"):
            self._session_id = str(result.get("id") or self._session_id or "")
            self._handle_job(result.get("job") or {})
            self.on_log(f"[stratum] session={self._session_id}")
            return

        status = result.get("status") if isinstance(result, dict) else None
        if status:
            self.on_log(f"[stratum] {status}")

    def _handle_job(self, params: Dict[str, Any]) -> None:
        job = MiningJob(
            job_id=str(params.get("job_id", "")),
            blob_hex=str(params.get("blob", "")),
            target_hex=str(params.get("target", "")),
            session_id=self._session_id,
            seed_hash_hex=str(params.get("seed_hash") or params.get("seed") or ""),
            height=int(params.get("height") or 0),
            algo=str(params.get("algo") or "rx/0"),
        )
        self._current_job = job
        self.on_log(
            f"[job] id={job.job_id} target={job.target_hex} height={job.height} algo={job.algo}"
        )
        self.on_job(job)