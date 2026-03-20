from __future__ import annotations

import json
import ssl
import threading
import time
from typing import Any, Callable, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from blocknet_mining_backend import BlockNetApiCfg
from models import MinerConfig, MiningJob, SubmitResult, VerifiedShare

JsonDict = Dict[str, Any]


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


def _make_ssl_context(verify_tls: bool) -> ssl.SSLContext:
    if verify_tls:
        return ssl.create_default_context()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _post_json_sync(cfg: BlockNetApiCfg, path: str, body: JsonDict) -> JsonDict:
    url = cfg.full_url(path)
    data = json.dumps(body or {}, separators=(",", ":")).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "BlockNetGpuMiner/1.0",
    }

    tok = (cfg.token or "").strip()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
        headers["X-Token"] = tok
        headers["X-BlockNet-Token"] = tok

    req = Request(url, data=data, headers=headers, method="POST")

    ssl_ctx = None
    if urlsplit(url).scheme.lower() == "https":
        ssl_ctx = _make_ssl_context(cfg.verify_tls)

    try:
        with urlopen(req, timeout=float(cfg.timeout_s), context=ssl_ctx) as resp:
            raw = resp.read() or b""
            status = getattr(resp, "status", 200)

            if not raw:
                return {
                    "ok": False,
                    "error": "empty response",
                    "status": status,
                    "headers": dict(resp.headers),
                }

            try:
                j = json.loads(raw.decode("utf-8", errors="replace"))
                if isinstance(j, dict):
                    if "status" not in j:
                        j["status"] = status
                    return j
                return {
                    "ok": False,
                    "error": "json was not an object",
                    "status": status,
                    "value": j,
                }
            except Exception:
                return {
                    "ok": False,
                    "error": "non-json response",
                    "status": status,
                    "headers": dict(resp.headers),
                    "body_preview": raw[:4000].decode("utf-8", errors="replace"),
                }

    except HTTPError as e:
        raw = b""
        try:
            raw = e.read() or b""
        except Exception:
            pass

        return {
            "ok": False,
            "error": f"http error {getattr(e, 'code', 0)}",
            "status": getattr(e, "code", 0),
            "headers": dict(getattr(e, "headers", {}) or {}),
            "body_preview": raw[:4000].decode("utf-8", errors="replace") if raw else "",
        }

    except URLError as e:
        return {
            "ok": False,
            "error": f"connect failed: {e}",
            "status": 0,
            "headers": {},
        }

    except Exception as e:
        return {
            "ok": False,
            "error": f"request failed: {e}",
            "status": 0,
            "headers": {},
        }


def _classify_blocknet_reject(status: str, error: str, raw: Any) -> tuple[str, bool, bool, bool, bool]:
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

    text = " ".join([str(status or ""), str(error or ""), str(raw or "")]).lower()

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
        or "malformed" in text
        or "invalid_share" in text
    )
    backend_error = (
        "timeout" in text
        or "gateway" in text
        or "session" in text
        or "not open" in text
        or "unknown_session" in text
        or "connect failed" in text
        or "request failed" in text
        or "socket" in text
        or "session_not_ready" in text
        or "session socket invalid" in text
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


class BlockNetClient:
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

        self._recover_lock = threading.Lock()
        self._submit_lock = threading.Lock()
        self._recent_submit_lock = threading.Lock()

        # key -> (ts, state)
        # state: "pending" | "accepted" | "rejected_final"
        self._recent_submits: dict[tuple[str, str, str], tuple[float, str]] = {}

        relay = (getattr(config, "blocknet_api_relay", "") or "").strip()
        if not relay:
            relay = f"{config.host}:{int(config.port)}"

        token = (getattr(config, "blocknet_api_token", "") or "").strip()
        prefix = (getattr(config, "blocknet_api_prefix", "/v1") or "/v1").strip() or "/v1"
        verify_tls = bool(getattr(config, "blocknet_verify_tls", False))
        timeout_s = float(getattr(config, "blocknet_timeout_s", 100.0))
        force_scheme = getattr(config, "blocknet_force_scheme", None)

        self.api_cfg = BlockNetApiCfg(
            relay=relay,
            token=token,
            prefix=prefix,
            timeout_s=timeout_s,
            verify_tls=verify_tls,
            force_scheme=force_scheme,
        )

        self._poll_interval_s = max(
            0.20,
            float(getattr(config, "blocknet_poll_interval_ms", 250)) / 1000.0,
        )
        self._poll_max_msgs = max(1, int(getattr(config, "blocknet_poll_max_msgs", 32)))
        self._poll_timeout_ms = max(0, int(getattr(config, "blocknet_poll_timeout_ms", 0)))
        self._submit_retry_count = max(1, int(getattr(config, "blocknet_submit_retry_count", 3)))
        self._submit_retry_base_ms = max(10, int(getattr(config, "blocknet_submit_retry_base_ms", 250)))

        self._stop = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None

        self._mu = threading.RLock()
        self._session: str = ""
        self._miner_id: str = ""
        self._connected = False
        self._job_seq: int = 0
        self._last_job_key: Optional[tuple[str, str, str, str, str]] = None
        self._current_job: Optional[MiningJob] = None

    @property
    def session(self) -> str:
        with self._mu:
            return self._session

    @property
    def session_id(self) -> str:
        with self._mu:
            return self._session

    @property
    def current_job(self) -> Optional[MiningJob]:
        with self._mu:
            return self._current_job

    def connect(self) -> None:
        if self._connected:
            return

        self.on_status("connecting")
        self._stop.clear()
        self._open_session_or_raise()

        self._connected = True
        self.on_status("connected")

        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="BlockNetPoll",
            daemon=True,
        )
        self._poll_thread.start()

    def close(self) -> None:
        self._stop.set()

        t = self._poll_thread
        self._poll_thread = None
        if t is not None and t.is_alive():
            t.join(timeout=2.0)

        with self._mu:
            session = self._session
            self._session = ""
            self._miner_id = ""
            self._connected = False
            self._job_seq = 0
            self._last_job_key = None
            self._current_job = None

        if session:
            try:
                _post_json_sync(self.api_cfg, "/p2pool/close", {"session": session})
            except Exception as exc:
                self.on_log(f"[blocknet] close failed: {exc}")

        self.on_status("disconnected")

    def submit(self, verified: VerifiedShare) -> SubmitResult:
        session = self.session
        if not session:
            return SubmitResult(
                accepted=False,
                status="not_open",
                error="BlockNet p2pool session not open",
                reject_class="backend_error",
                backend_error=True,
            )

        job_id = str(verified.job_id or "").strip()
        nonce_hex = _normalize_hex(verified.nonce_hex or "")
        result_hex = _normalize_hex(verified.result_hex or "")

        if not job_id:
            return SubmitResult(False, status="invalid_share", error="missing job_id", reject_class="invalid", invalid=True)
        if len(nonce_hex) != 8 or not _is_hex(nonce_hex):
            return SubmitResult(False, status="invalid_share", error="nonce_hex must be 8 hex chars", reject_class="invalid", invalid=True)
        if len(result_hex) != 64 or not _is_hex(result_hex):
            return SubmitResult(False, status="invalid_share", error="result_hex must be 64 hex chars", reject_class="invalid", invalid=True)

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

        with self._submit_lock:
            # check again under lock
            recent_state = self._get_recent_submit_state(key)
            if recent_state in {"pending", "accepted", "rejected_final"}:
                return SubmitResult(
                    accepted=False,
                    status="duplicate_local",
                    error=f"share already submitted recently ({recent_state})",
                    reject_class="duplicate",
                    duplicate=True,
                )

            payload: JsonDict = {
                "session": session,
                "job_id": job_id,
                "nonce": nonce_hex,
                "result": result_hex,
            }

            with self._mu:
                if self._job_seq > 0:
                    payload["job_seq"] = int(self._job_seq)

            self._set_recent_submit_state(key, "pending")

            self.on_log(
                f"[submit] sent backend=blocknet nonce={nonce_hex} job={job_id} "
                f"credited={verified.credited_work:.6f} "
                f"actual={verified.actual_work:.6f} "
                f"quality={verified.quality:.6f}"
            )

            transient_5xx = 0
            last_j: JsonDict | None = None

            for attempt in range(self._submit_retry_count):
                j = _post_json_sync(self.api_cfg, "/p2pool/submit", payload)
                last_j = j

                if j.get("ok"):
                    self._set_recent_submit_state(key, "accepted")
                    self._update_job_seq_from_response(j)
                    self._maybe_update_miner_id(j)
                    self._maybe_emit_job_from_response(j)
                    return SubmitResult(
                        accepted=True,
                        status=str(j.get("status") or j.get("code") or "accepted"),
                        error="",
                        raw=j,
                        reject_class="accepted",
                    )

                http_status = int(j.get("status") or 0)
                detail = str(
                    j.get("detail")
                    or j.get("error")
                    or j.get("body_preview")
                    or ""
                )
                detail_l = detail.lower()

                if self._is_session_error(j):
                    self.on_log("[blocknet] submit saw session error, recovering session")
                    self._clear_recent_submit_state(key)
                    self._recover_session()
                    new_session = self.session
                    if not new_session:
                        return SubmitResult(
                            accepted=False,
                            status="not_open",
                            error="BlockNet p2pool session recovery failed",
                            raw=j,
                            reject_class="backend_error",
                            backend_error=True,
                        )
                    payload["session"] = new_session
                    with self._mu:
                        if self._job_seq > 0:
                            payload["job_seq"] = int(self._job_seq)
                        else:
                            payload.pop("job_seq", None)
                    self._set_recent_submit_state(key, "pending")
                    continue

                if http_status in (502, 503, 504) or "bad gateway" in detail_l:
                    transient_5xx += 1
                    self.on_log(
                        f"[blocknet] submit transient failure status={http_status} "
                        f"attempt={attempt + 1}/{self._submit_retry_count} "
                        f"job={job_id} nonce={nonce_hex}"
                    )
                    self._clear_recent_submit_state(key)
                    time.sleep((self._submit_retry_base_ms / 1000.0) * (attempt + 1))
                    self._set_recent_submit_state(key, "pending")
                    continue

                status = str(j.get("status") or j.get("code") or "rejected")
                error = str(
                    j.get("detail")
                    or j.get("error")
                    or j.get("body_preview")
                    or "submit failed"
                )
                reject_class, stale, duplicate, invalid, backend_error = _classify_blocknet_reject(
                    status,
                    error,
                    j,
                )

                if backend_error:
                    self._clear_recent_submit_state(key)
                else:
                    self._set_recent_submit_state(key, "rejected_final")

                self._update_job_seq_from_response(j)
                self._maybe_update_miner_id(j)
                self._maybe_emit_job_from_response(j)

                return SubmitResult(
                    accepted=False,
                    status=status,
                    error=error,
                    raw=j,
                    reject_class=reject_class,
                    stale=stale,
                    duplicate=duplicate,
                    invalid=invalid,
                    backend_error=backend_error,
                )

            if transient_5xx:
                self.on_log("[blocknet] repeated submit 5xx errors, recovering session")
                self._recover_session()

            self._clear_recent_submit_state(key)

            j = last_j or {}
            status = str(j.get("status") or j.get("code") or "rejected")
            error = str(
                j.get("detail")
                or j.get("error")
                or j.get("body_preview")
                or "submit failed"
            )
            reject_class, stale, duplicate, invalid, backend_error = _classify_blocknet_reject(
                status,
                error,
                j,
            )
            return SubmitResult(
                accepted=False,
                status=status,
                error=error,
                raw=j,
                reject_class=reject_class,
                stale=stale,
                duplicate=duplicate,
                invalid=invalid,
                backend_error=backend_error,
            )

    def _prune_recent_submits(self, now: float) -> None:
        dead: list[tuple[str, str, str]] = []
        for k, (ts, state) in self._recent_submits.items():
            age = now - ts
            if state == "pending":
                if age > 5.0:
                    dead.append(k)
            else:
                if age > 120.0:
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

    def _open_session_or_raise(self) -> None:
        j = _post_json_sync(self.api_cfg, "/p2pool/open", {})
        if not j.get("ok"):
            raise RuntimeError(f"BlockNet p2pool open failed: {j}")

        session = str(j.get("session") or "")
        if not session:
            raise RuntimeError(f"BlockNet p2pool open missing session: {j}")

        miner_id = str(j.get("miner_id") or "")
        job_seq = int(j.get("job_seq", 0) or 0)

        with self._mu:
            self._session = session
            self._miner_id = miner_id
            self._job_seq = job_seq

        self.on_log(f"[blocknet] session opened: {session}")

        job = j.get("job") or {}
        if not job:
            poll_payload: JsonDict = {"session": session, "max_msgs": self._poll_max_msgs}
            if self._poll_timeout_ms > 0:
                poll_payload["timeout_ms"] = self._poll_timeout_ms
            if job_seq > 0:
                poll_payload["after_job_seq"] = job_seq

            poll = _post_json_sync(self.api_cfg, "/p2pool/poll", poll_payload)
            if poll.get("ok"):
                self._update_job_seq_from_response(poll)
                self._maybe_update_miner_id(poll)
                job = poll.get("job") or {}

        if not job:
            direct_payload: JsonDict = {"session": session}
            if job_seq > 0:
                direct_payload["job_seq"] = job_seq
            direct = _post_json_sync(self.api_cfg, "/p2pool/job", direct_payload)
            if direct.get("ok"):
                self._update_job_seq_from_response(direct)
                self._maybe_update_miner_id(direct)
                job = direct.get("job") or {}

        if job:
            self._emit_job(job)

    def _poll_loop(self) -> None:
        consecutive_failures = 0

        while not self._stop.is_set():
            session = self.session
            if not session:
                time.sleep(self._poll_interval_s)
                continue

            try:
                payload: JsonDict = {
                    "session": session,
                    "max_msgs": self._poll_max_msgs,
                }

                with self._mu:
                    current_job_seq = int(self._job_seq)

                if self._poll_timeout_ms > 0:
                    payload["timeout_ms"] = self._poll_timeout_ms
                if current_job_seq > 0:
                    payload["after_job_seq"] = current_job_seq

                j = _post_json_sync(self.api_cfg, "/p2pool/poll", payload)

                if not j.get("ok"):
                    if self._is_session_error(j):
                        raise RuntimeError(f"session error: {j}")
                    raise RuntimeError(str(j))

                consecutive_failures = 0
                self._update_job_seq_from_response(j)
                self._maybe_update_miner_id(j)

                job = j.get("job") or {}
                if job:
                    self._emit_job(job)

            except Exception as exc:
                if self._stop.is_set():
                    break

                consecutive_failures += 1
                self.on_log(f"[blocknet] poll error ({consecutive_failures}): {exc}")

                if consecutive_failures >= 3:
                    self.on_status("reconnecting")
                    self._recover_session()
                    consecutive_failures = 0

            time.sleep(self._poll_interval_s)

    def _recover_session(self) -> None:
        if self._stop.is_set():
            return

        if not self._recover_lock.acquire(blocking=False):
            return

        try:
            with self._mu:
                self._session = ""
                self._miner_id = ""
                self._connected = False
                self._job_seq = 0
                self._current_job = None
                self._last_job_key = None

            delay = 0.5
            while not self._stop.is_set():
                try:
                    self._open_session_or_raise()
                    with self._mu:
                        self._connected = True
                    self.on_status("connected")
                    return
                except Exception as exc:
                    self.on_log(f"[blocknet] reopen failed: {exc}")
                    time.sleep(delay)
                    delay = min(delay * 2.0, 5.0)
        finally:
            self._recover_lock.release()

    def _emit_job(self, wire_job: JsonDict) -> None:
        job = self._job_from_wire(wire_job)
        if job is None:
            return

        key = (
            job.session_id,
            job.job_id,
            job.seed_hash_hex.lower(),
            job.target_hex.lower(),
            job.blob_hex.lower(),
        )

        with self._mu:
            self._current_job = job

        if key == self._last_job_key:
            return

        self._last_job_key = key
        self.on_job(job)

    def _job_from_wire(self, j: JsonDict) -> Optional[MiningJob]:
        job_id = str(j.get("job_id") or j.get("id") or "").strip()
        blob_hex = str(j.get("blob") or j.get("blob_hex") or "").strip()
        target_hex = str(j.get("target") or j.get("target_hex") or "").strip()
        seed_hash_hex = str(j.get("seed_hash") or j.get("seed_hash_hex") or "").strip()
        algo = str(j.get("algo") or "rx/0").strip() or "rx/0"

        if not job_id or not blob_hex or not target_hex:
            self.on_log(f"[blocknet] ignoring incomplete job payload: {j}")
            return None

        try:
            height = int(j.get("height") or 0)
        except Exception:
            height = 0

        submit_blob_hex = str(
            j.get("submit_blob_hex")
            or j.get("submit_blob")
            or j.get("blocktemplate_blob")
            or ""
        ).strip()

        with self._mu:
            session_id = self._session

        try:
            job = MiningJob(
                job_id=job_id,
                blob_hex=blob_hex,
                target_hex=target_hex,
                session_id=session_id,
                seed_hash_hex=seed_hash_hex,
                height=height,
                algo=algo,
            )
        except TypeError:
            # fallback in case MiningJob accepts fewer fields
            job = MiningJob(
                job_id=job_id,
                blob_hex=blob_hex,
                target_hex=target_hex,
                session_id=session_id,
                seed_hash_hex=seed_hash_hex,
                height=height,
            )

        try:
            setattr(job, "algo", algo)
        except Exception:
            pass
        try:
            setattr(job, "received_at", time.time())
        except Exception:
            pass
        if submit_blob_hex:
            try:
                setattr(job, "submit_blob_hex", submit_blob_hex)
            except Exception:
                pass

        return job

    def _update_job_seq_from_response(self, j: JsonDict) -> None:
        try:
            job_seq = int(j.get("job_seq", 0) or 0)
        except Exception:
            job_seq = 0
        if job_seq > 0:
            with self._mu:
                self._job_seq = job_seq

    def _maybe_update_miner_id(self, j: JsonDict) -> None:
        miner_id = str(j.get("miner_id") or "").strip()
        if miner_id:
            with self._mu:
                self._miner_id = miner_id

    def _maybe_emit_job_from_response(self, j: JsonDict) -> None:
        job = j.get("job") or {}
        if isinstance(job, dict) and job:
            self._emit_job(job)

    def _is_session_error(self, j: JsonDict) -> bool:
        text = " ".join(
            [
                str(j.get("error") or ""),
                str(j.get("detail") or ""),
                str(j.get("body_preview") or ""),
                str(j),
            ]
        ).lower()
        return (
            "unknown_session" in text
            or "session_not_ready" in text
            or "session socket invalid" in text
            or "session not open" in text
            or "p2pool session not open" in text
        )