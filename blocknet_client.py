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
        self._recent_submits: dict[tuple[str, str, str], float] = {}

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

        self._stop = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None

        self._mu = threading.RLock()
        self._session: str = ""
        self._miner_id: str = ""
        self._connected = False
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
        if not nonce_hex:
            return SubmitResult(False, status="invalid_share", error="missing nonce_hex", reject_class="invalid", invalid=True)
        if not result_hex:
            return SubmitResult(False, status="invalid_share", error="missing result_hex", reject_class="invalid", invalid=True)
        if not _is_hex(nonce_hex):
            return SubmitResult(False, status="invalid_share", error="nonce_hex is not valid hex", reject_class="invalid", invalid=True)
        if not _is_hex(result_hex):
            return SubmitResult(False, status="invalid_share", error="result_hex is not valid hex", reject_class="invalid", invalid=True)

        current_job = self.current_job
        if current_job and current_job.job_id and current_job.job_id != job_id:
            return SubmitResult(
                accepted=False,
                status="STALE",
                error=f"stale share job={job_id}, current={current_job.job_id}",
                reject_class="stale",
                stale=True,
            )

        if self._mark_submit_seen(job_id, nonce_hex, result_hex):
            return SubmitResult(
                accepted=False,
                status="duplicate_local",
                error="share already submitted recently",
                reject_class="duplicate",
                duplicate=True,
            )

        with self._submit_lock:
            payload: JsonDict = {
                "session": session,
                "job_id": job_id,
                "nonce": nonce_hex,
                "result": result_hex,
            }

            self.on_log(
                f"[submit] sent backend=blocknet nonce={nonce_hex} job={job_id} "
                f"credited={verified.credited_work:.6f} "
                f"actual={verified.actual_work:.6f} "
                f"quality={verified.quality:.6f}"
            )

            transient_5xx = 0
            last_j: JsonDict | None = None

            for attempt in range(3):
                j = _post_json_sync(self.api_cfg, "/p2pool/submit", payload)
                last_j = j

                if j.get("ok"):
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

                if (
                    "unknown_session" in detail_l
                    or "session_not_ready" in detail_l
                    or "session socket invalid" in detail_l
                    or "session not open" in detail_l
                    or "p2pool session not open" in detail_l
                ):
                    self.on_log("[blocknet] submit saw session error, recovering session")
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
                    continue

                if http_status in (502, 503, 504) or "bad gateway" in detail_l:
                    transient_5xx += 1
                    self.on_log(
                        f"[blocknet] submit transient failure status={http_status} "
                        f"attempt={attempt + 1}/3 job={job_id} nonce={nonce_hex}"
                    )
                    time.sleep(0.25 * (attempt + 1))
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

    def _open_session_or_raise(self) -> None:
        j = _post_json_sync(self.api_cfg, "/p2pool/open", {})
        if not j.get("ok"):
            raise RuntimeError(f"BlockNet p2pool open failed: {j}")

        session = str(j.get("session") or "")
        if not session:
            raise RuntimeError(f"BlockNet p2pool open missing session: {j}")

        miner_id = str(j.get("miner_id") or "")

        with self._mu:
            self._session = session
            self._miner_id = miner_id

        self.on_log(f"[blocknet] session opened: {session}")

        job = j.get("job") or {}
        if not job:
            poll = _post_json_sync(
                self.api_cfg,
                "/p2pool/poll",
                {"session": session, "max_msgs": self._poll_max_msgs},
            )
            if poll.get("ok"):
                if poll.get("miner_id"):
                    with self._mu:
                        self._miner_id = str(poll.get("miner_id") or "")
                job = poll.get("job") or {}

        if not job:
            direct = _post_json_sync(self.api_cfg, "/p2pool/job", {"session": session})
            if direct.get("ok"):
                if direct.get("miner_id"):
                    with self._mu:
                        self._miner_id = str(direct.get("miner_id") or "")
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
                j = _post_json_sync(
                    self.api_cfg,
                    "/p2pool/poll",
                    {"session": session, "max_msgs": self._poll_max_msgs},
                )

                if not j.get("ok"):
                    raise RuntimeError(str(j))

                consecutive_failures = 0

                if j.get("miner_id"):
                    with self._mu:
                        self._miner_id = str(j.get("miner_id") or "")

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

        return MiningJob(
            job_id=job_id,
            blob_hex=blob_hex,
            target_hex=target_hex,
            session_id=self.session,
            seed_hash_hex=seed_hash_hex,
            height=height,
            algo=algo,
        )