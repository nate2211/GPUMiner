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
    if isinstance(raw, dict):
        rc = str(raw.get("reject_class") or "").strip().lower()
        if rc == "accepted":
            return ("accepted", False, False, False, False)
        if rc == "stale":
            return ("stale", True, False, False, False)
        if rc == "duplicate":
            return ("duplicate", False, True, False, False)
        if rc == "invalid":
            return ("invalid", False, False, True, False)
        if rc == "backend_error":
            return ("backend_error", False, False, False, True)
        if rc == "rejected":
            return ("rejected", False, False, False, False)

        if bool(raw.get("stale")):
            return ("stale", True, False, False, False)
        if bool(raw.get("duplicate")):
            return ("duplicate", False, True, False, False)
        if bool(raw.get("invalid")):
            return ("invalid", False, False, True, False)
        if bool(raw.get("backend_error")):
            return ("backend_error", False, False, False, True)

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
        or "send_failed" in text
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
        self._reconnect_thread: Optional[threading.Thread] = None

        self._stop = threading.Event()

        self._write_lock = threading.Lock()
        self._id_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._recent_submit_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._connect_lock = threading.Lock()
        self._submit_lock = threading.Lock()
        self._reconnect_lock = threading.Lock()

        self._req_id = 1
        self._pending: Dict[int, PendingRequest] = {}
        self._pending_submit_keys: Dict[int, tuple[str, str, str]] = {}

        # key -> (timestamp, state)
        # state: "pending" | "accepted" | "rejected_final"
        self._recent_submits: dict[tuple[str, str, str], tuple[float, str]] = {}

        self._session_id = ""
        self._current_job: Optional[MiningJob] = None
        self._last_job_key: Optional[tuple[str, str, str, str, str]] = None

        self._conn_gen = 0

        self._submit_timeout_s = max(1.0, float(getattr(config, "submit_timeout_s", 10.0)))
        self._login_timeout_s = max(2.0, float(getattr(config, "login_timeout_s", 15.0)))
        self._socket_timeout_s = max(2.0, float(getattr(config, "socket_connect_timeout_s", 10.0)))
        self._keepalive_interval_s = max(10.0, float(getattr(config, "keepalive_interval_s", 30.0)))

        self._recent_submit_pending_ttl_s = max(
            5.0, float(getattr(config, "recent_submit_pending_ttl_s", 15.0))
        )
        self._recent_submit_final_ttl_s = max(
            30.0, float(getattr(config, "recent_submit_final_ttl_s", 120.0))
        )

        self._auto_reconnect = bool(getattr(config, "auto_reconnect", True))
        self._reconnect_initial_delay_s = max(
            1.0, float(getattr(config, "reconnect_initial_delay_s", 2.0))
        )
        self._reconnect_max_delay_s = max(
            self._reconnect_initial_delay_s,
            float(getattr(config, "reconnect_max_delay_s", 30.0)),
        )

    @property
    def session_id(self) -> str:
        with self._state_lock:
            return self._session_id

    @property
    def current_job(self) -> Optional[MiningJob]:
        with self._state_lock:
            return self._current_job

    def connect(self) -> None:
        self._stop.clear()

        if self._is_connected():
            return

        self._connect_once(is_reconnect=False)

    def close(self) -> None:
        self._stop.set()
        self._shutdown_connection(
            reason="closed by user",
            reconnect=False,
            clear_recent_pending=True,
        )
        self.on_status("disconnected")

    def submit(self, share: VerifiedShare, timeout: Optional[float] = None) -> SubmitResult:
        if timeout is None:
            timeout = self._submit_timeout_s

        if not self._is_connected():
            return SubmitResult(
                accepted=False,
                status="not_connected",
                error="socket not connected",
                reject_class="backend_error",
                backend_error=True,
            )

        session_id = (share.session_id or self.session_id).strip()
        if not session_id:
            return SubmitResult(
                accepted=False,
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
                accepted=False,
                status="invalid_share",
                error="missing job_id",
                reject_class="invalid",
                invalid=True,
            )
        if len(nonce_hex) != 8 or not _is_hex(nonce_hex):
            return SubmitResult(
                accepted=False,
                status="invalid_share",
                error="nonce_hex must be 8 hex chars",
                reject_class="invalid",
                invalid=True,
            )
        if len(result_hex) != 64 or not _is_hex(result_hex):
            return SubmitResult(
                accepted=False,
                status="invalid_share",
                error="result_hex must be 64 hex chars",
                reject_class="invalid",
                invalid=True,
            )

        current_job = self.current_job
        if current_job and current_job.job_id and current_job.job_id != job_id:
            self.on_log(
                f"[submit] local job moved on submit_job={job_id} current_job={current_job.job_id}; "
                "submitting anyway so server decides staleness"
            )

        key = (job_id, nonce_hex, result_hex)
        recent_state = self._get_recent_submit_state(key)
        if recent_state in {"pending", "accepted", "rejected_final"}:
            return SubmitResult(
                accepted=False,
                status="duplicate_local",
                error=f"share already submitted recently ({recent_state})",
                reject_class="duplicate",
                duplicate=True,
            )

        req_id: Optional[int] = None

        with self._submit_lock:
            recent_state = self._get_recent_submit_state(key)
            if recent_state in {"pending", "accepted", "rejected_final"}:
                return SubmitResult(
                    accepted=False,
                    status="duplicate_local",
                    error=f"share already submitted recently ({recent_state})",
                    reject_class="duplicate",
                    duplicate=True,
                )

            self._set_recent_submit_state(key, "pending")

            req_id = self._next_id()
            pending = PendingRequest()

            with self._pending_lock:
                self._pending[req_id] = pending
                self._pending_submit_keys[req_id] = key

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
            except Exception as exc:
                self._clear_recent_submit_state(key)
                with self._pending_lock:
                    self._pending.pop(req_id, None)
                    self._pending_submit_keys.pop(req_id, None)
                return SubmitResult(
                    accepted=False,
                    status="send_failed",
                    error=str(exc),
                    reject_class="backend_error",
                    backend_error=True,
                )

            try:
                if not pending.event.wait(float(timeout)):
                    self._clear_recent_submit_state(key)
                    return SubmitResult(
                        accepted=False,
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

                    if backend_error:
                        self._clear_recent_submit_state(key)
                    else:
                        self._set_recent_submit_state(key, "rejected_final")

                    return SubmitResult(
                        accepted=False,
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
                    accepted_flag = bool(result.get("accepted", False))

                    if accepted_flag or (status and status.upper() in {"OK", "ACCEPTED"}):
                        self._set_recent_submit_state(key, "accepted")
                        return SubmitResult(
                            accepted=True,
                            status=status or "OK",
                            raw=resp,
                            reject_class="accepted",
                        )

                    if status or result.get("error") or result.get("reject_class"):
                        err_text = str(result.get("error") or status or "rejected")
                        reject_class, stale, duplicate, invalid, backend_error = _classify_reject(
                            status,
                            err_text,
                            result,
                        )

                        if backend_error:
                            self._clear_recent_submit_state(key)
                        else:
                            self._set_recent_submit_state(key, "rejected_final")

                        return SubmitResult(
                            accepted=False,
                            status=status or "ERROR",
                            error=err_text,
                            raw=resp,
                            reject_class=reject_class,
                            stale=stale,
                            duplicate=duplicate,
                            invalid=invalid,
                            backend_error=backend_error,
                        )

                    self._set_recent_submit_state(key, "accepted")
                    return SubmitResult(
                        accepted=True,
                        status="OK",
                        raw=resp,
                        reject_class="accepted",
                    )

                if isinstance(result, str):
                    status = result.strip()
                    if status.upper() in {"OK", "ACCEPTED"}:
                        self._set_recent_submit_state(key, "accepted")
                        return SubmitResult(
                            accepted=True,
                            status=status,
                            raw=resp,
                            reject_class="accepted",
                        )

                    reject_class, stale, duplicate, invalid, backend_error = _classify_reject(
                        status,
                        status,
                        resp,
                    )

                    if backend_error:
                        self._clear_recent_submit_state(key)
                    else:
                        self._set_recent_submit_state(key, "rejected_final")

                    return SubmitResult(
                        accepted=False,
                        status=status or "ERROR",
                        error=status,
                        raw=resp,
                        reject_class=reject_class,
                        stale=stale,
                        duplicate=duplicate,
                        invalid=invalid,
                        backend_error=backend_error,
                    )

                if result is False:
                    self._set_recent_submit_state(key, "rejected_final")
                    return SubmitResult(
                        accepted=False,
                        status="ERROR",
                        error="pool returned false",
                        raw=resp,
                        reject_class="rejected",
                    )

                self._set_recent_submit_state(key, "accepted")
                return SubmitResult(
                    accepted=True,
                    status="OK",
                    raw=resp,
                    reject_class="accepted",
                )

            finally:
                if req_id is not None:
                    with self._pending_lock:
                        self._pending.pop(req_id, None)
                        self._pending_submit_keys.pop(req_id, None)

    def _connect_once(self, *, is_reconnect: bool) -> None:
        with self._connect_lock:
            if self._stop.is_set():
                raise RuntimeError("client is stopping")
            if self._is_connected():
                return

            self.on_status("reconnecting" if is_reconnect else "connecting")

            raw = socket.create_connection(
                (self.config.host, self.config.port),
                timeout=self._socket_timeout_s,
            )

            try:
                if self.config.use_tls:
                    ctx = ssl.create_default_context()
                    sock = ctx.wrap_socket(raw, server_hostname=self.config.host)
                else:
                    sock = raw

                sock.settimeout(None)
                file_obj = sock.makefile("r", encoding="utf-8", newline="\n")

                gen = self._install_connection(sock, file_obj)
                self.on_log(f"[stratum] connected to {self.config.host}:{self.config.port}")

                try:
                    self._send_login_and_wait()
                except Exception:
                    self._shutdown_connection(
                        reason="login failed",
                        reconnect=False,
                        clear_recent_pending=True,
                    )
                    raise

                self._start_keepalive_thread(gen)
            except Exception:
                try:
                    raw.close()
                except Exception:
                    pass
                raise

    def _install_connection(self, sock: socket.socket, file_obj: Any) -> int:
        with self._state_lock:
            self._conn_gen += 1
            gen = self._conn_gen
            self._sock = sock
            self._file = file_obj
            self._session_id = ""
            self._current_job = None
            self._last_job_key = None

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(gen,),
            daemon=True,
            name=f"StratumReader-{gen}",
        )
        self._reader_thread.start()
        return gen

    def _start_keepalive_thread(self, gen: int) -> None:
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop,
            args=(gen,),
            daemon=True,
            name=f"StratumKeepalive-{gen}",
        )
        self._keepalive_thread.start()

    def _start_reconnect_loop(self) -> None:
        if self._stop.is_set() or not self._auto_reconnect:
            return

        with self._reconnect_lock:
            if self._reconnect_thread is not None and self._reconnect_thread.is_alive():
                return

            self._reconnect_thread = threading.Thread(
                target=self._reconnect_loop,
                daemon=True,
                name="StratumReconnect",
            )
            self._reconnect_thread.start()

    def _reconnect_loop(self) -> None:
        delay = self._reconnect_initial_delay_s
        current = threading.current_thread()

        try:
            while not self._stop.is_set():
                if self._is_connected():
                    return

                try:
                    self._connect_once(is_reconnect=True)
                    self.on_log("[stratum] reconnected")
                    return
                except Exception as exc:
                    self.on_log(f"[reconnect] failed: {exc}")
                    if self._stop.wait(delay):
                        return
                    delay = min(self._reconnect_max_delay_s, delay * 2.0)
        finally:
            with self._reconnect_lock:
                if self._reconnect_thread is current:
                    self._reconnect_thread = None

    def _shutdown_connection(
        self,
        *,
        reason: str,
        reconnect: bool,
        clear_recent_pending: bool,
    ) -> None:
        with self._state_lock:
            sock = self._sock
            file_obj = self._file
            was_connected = self._sock is not None or self._file is not None or bool(self._session_id)

            self._sock = None
            self._file = None
            self._session_id = ""
            self._current_job = None
            self._last_job_key = None
            self._conn_gen += 1

        if file_obj is not None:
            try:
                file_obj.close()
            except Exception:
                pass

        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

        self._fail_all_pending(reason=reason, clear_recent_pending=clear_recent_pending)

        if reconnect and not self._stop.is_set() and self._auto_reconnect:
            if was_connected:
                self.on_log(f"[stratum] disconnected: {reason}")
            self.on_status("reconnecting")
            self._start_reconnect_loop()
        else:
            if was_connected:
                self.on_log(f"[stratum] disconnected: {reason}")
            self.on_status("disconnected")

    def _fail_all_pending(self, *, reason: str, clear_recent_pending: bool) -> None:
        with self._pending_lock:
            pending_items = list(self._pending.items())
            pending_submit_keys = dict(self._pending_submit_keys)
            self._pending.clear()
            self._pending_submit_keys.clear()

        for req_id, pending in pending_items:
            if clear_recent_pending:
                key = pending_submit_keys.get(req_id)
                if key is not None:
                    self._clear_recent_submit_state(key)

            pending.response = {
                "error": {
                    "message": reason,
                    "code": "DISCONNECTED",
                },
                "reject_class": "backend_error",
                "backend_error": True,
            }
            pending.event.set()

    def _prune_recent_submits(self, now: float) -> None:
        dead: list[tuple[str, str, str]] = []
        for k, (ts, state) in self._recent_submits.items():
            age = now - ts
            if state == "pending":
                if age > self._recent_submit_pending_ttl_s:
                    dead.append(k)
            else:
                if age > self._recent_submit_final_ttl_s:
                    dead.append(k)

        for k in dead:
            self._recent_submits.pop(k, None)

    def _get_recent_submit_state(self, key: tuple[str, str, str]) -> Optional[str]:
        now = time.time()
        with self._recent_submit_lock:
            self._prune_recent_submits(now)
            row = self._recent_submits.get(key)
            return row[1] if row else None

    def _set_recent_submit_state(self, key: tuple[str, str, str], state: str) -> None:
        with self._recent_submit_lock:
            self._recent_submits[key] = (time.time(), state)

    def _clear_recent_submit_state(self, key: tuple[str, str, str]) -> None:
        with self._recent_submit_lock:
            self._recent_submits.pop(key, None)

    def _is_connected(self) -> bool:
        with self._state_lock:
            return self._sock is not None and not self._stop.is_set()

    def _next_id(self) -> int:
        with self._id_lock:
            rid = self._req_id
            self._req_id += 1
            return rid

    def _send_login_and_wait(self) -> None:
        req_id = self._next_id()
        pending = PendingRequest()

        with self._pending_lock:
            self._pending[req_id] = pending

        payload = {
            "id": req_id,
            "jsonrpc": "2.0",
            "method": "login",
            "params": {
                "login": self.config.login,
                "pass": self.config.password,
                "agent": self.config.agent,
            },
        }

        try:
            self._send(payload)
            self.on_log(f"[stratum] login as {self.config.login}")

            if not pending.event.wait(self._login_timeout_s):
                raise RuntimeError("login timed out")

            resp = pending.response or {}
            error = resp.get("error")
            if error:
                raise RuntimeError(_stringify_error(error))

            result = resp.get("result") or {}
            if not isinstance(result, dict):
                raise RuntimeError(f"unexpected login result: {resp}")

            session_id = str(result.get("id") or "").strip()
            if not session_id:
                raise RuntimeError("login response missing session id")

            with self._state_lock:
                self._session_id = session_id

            job = result.get("job") or {}
            if isinstance(job, dict) and job:
                self._handle_job(job)

            status = str(result.get("status") or "authorized").strip()
            self.on_status("connected")
            self.on_log(f"[stratum] authorized login={self.config.login} session={session_id} status={status}")

        finally:
            with self._pending_lock:
                self._pending.pop(req_id, None)

    def _keepalive_loop(self, gen: int) -> None:
        while not self._stop.wait(self._keepalive_interval_s):
            with self._state_lock:
                if gen != self._conn_gen or self._sock is None:
                    return
                session_id = self._session_id

            if not session_id:
                continue

            payload = {
                "id": self._next_id(),
                "jsonrpc": "2.0",
                "method": "keepalived",
                "params": {"id": session_id},
            }

            try:
                self._send(payload)
            except Exception as exc:
                self.on_log(f"[keepalive] error: {exc}")
                self._shutdown_connection(
                    reason=f"keepalive failed: {exc}",
                    reconnect=True,
                    clear_recent_pending=True,
                )
                return

    def _send(self, payload: Dict[str, Any]) -> None:
        sock = self._sock
        if sock is None:
            raise RuntimeError("socket not connected")

        wire = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        with self._write_lock:
            sock.sendall(wire)

    def _reader_loop(self, gen: int) -> None:
        try:
            while not self._stop.is_set():
                with self._state_lock:
                    if gen != self._conn_gen:
                        return
                    file_obj = self._file

                if file_obj is None:
                    return

                line = file_obj.readline()
                if not line:
                    raise ConnectionError("stratum connection closed")
                self._handle_message(json.loads(line))
        except Exception as exc:
            self.on_log(f"[stratum] reader stopped: {exc}")
            self._shutdown_connection(
                reason=str(exc),
                reconnect=True,
                clear_recent_pending=True,
            )

    def _handle_message(self, msg: Dict[str, Any]) -> None:
        method = msg.get("method")
        if method == "job":
            self._handle_job(msg.get("params") or {})
            return

        if "id" in msg:
            req_id = msg.get("id")
            with self._pending_lock:
                pending = self._pending.get(req_id)
            if pending is not None:
                pending.response = msg
                pending.event.set()
                return

        error = msg.get("error")
        if error:
            self.on_log(f"[stratum] error: {_stringify_error(error)}")
            return

        result = msg.get("result") or {}
        if isinstance(result, dict):
            if result.get("job"):
                with self._state_lock:
                    if result.get("id"):
                        self._session_id = str(result.get("id") or self._session_id or "")
                self._handle_job(result.get("job") or {})
                if self.session_id:
                    self.on_log(f"[stratum] session={self.session_id}")
                return

            status = result.get("status")
            if status:
                self.on_log(f"[stratum] {status}")
                return

    def _handle_job(self, params: Dict[str, Any]) -> None:
        if not isinstance(params, dict):
            return

        session_id = self.session_id

        job_id = str(params.get("job_id") or "").strip()
        blob_hex = str(params.get("blob") or "").strip()
        target_hex = str(params.get("target") or "").strip()
        seed_hash_hex = str(params.get("seed_hash") or params.get("seed") or "").strip()
        algo = str(params.get("algo") or "rx/0").strip() or "rx/0"

        try:
            height = int(params.get("height") or 0)
        except Exception:
            height = 0

        if not job_id or not blob_hex or not target_hex:
            self.on_log(f"[job] ignoring incomplete payload: {params}")
            return

        job = MiningJob(
            job_id=job_id,
            blob_hex=blob_hex,
            target_hex=target_hex,
            session_id=session_id,
            seed_hash_hex=seed_hash_hex,
            height=height,
            algo=algo,
        )

        try:
            setattr(job, "received_at", time.time())
        except Exception:
            pass

        try:
            setattr(job, "submit_blob_hex", blob_hex)
        except Exception:
            pass

        key = (
            job.session_id,
            job.job_id,
            job.seed_hash_hex.lower(),
            job.target_hex.lower(),
            job.blob_hex.lower(),
        )

        with self._state_lock:
            self._current_job = job
            if key == self._last_job_key:
                return
            self._last_job_key = key

        self.on_log(
            f"[job] id={job.job_id} target={job.target_hex} height={job.height} algo={job.algo}"
        )
        self.on_job(job)