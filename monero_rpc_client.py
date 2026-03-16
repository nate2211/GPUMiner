from __future__ import annotations

import hashlib
import json
import secrets
import ssl
import threading
import time
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from models import MinerConfig, MiningJob, NonceWindow, SubmitResult, VerifiedShare

JsonDict = dict[str, Any]


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return int(default)
            if text.lower().startswith("0x"):
                return int(text, 16)
            return int(text)
        return int(value)
    except Exception:
        return int(default)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _normalize_hex(text: Any) -> str:
    return "".join(ch for ch in str(text or "").strip().lower() if not ch.isspace())


def _normalize_url(base: str, force_scheme: Optional[str]) -> str:
    text = (base or "").strip()
    if not text:
        return ""
    if "://" not in text:
        scheme = (force_scheme or "http").strip().lower() or "http"
        text = f"{scheme}://{text}"
    parts = urlsplit(text)
    scheme = (force_scheme or parts.scheme or "http").strip().lower() or "http"
    path = (parts.path or "").rstrip("/")
    return urlunsplit((scheme, parts.netloc, path, "", ""))


def _join_url(base: str, prefix: str, path: str) -> str:
    base = (base or "").rstrip("/")
    pref = "/" + (prefix or "").strip("/")
    p = "/" + (path or "").strip("/")
    if pref == "/":
        pref = ""
    return f"{base}{pref}{p}"


def _make_ssl_context(verify_tls: bool) -> ssl.SSLContext:
    if verify_tls:
        return ssl.create_default_context()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _make_headers(token: str = "", *, extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "GPUMiner-PyQt5/0.2",
    }
    tok = (token or "").strip()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
        headers["X-Blocknet-Key"] = tok
        headers["X-MoneroRPC-Key"] = tok
        headers["X-Token"] = tok
    if extra:
        headers.update(extra)
    return headers


def _json_request(
    url: str,
    *,
    method: str,
    payload: Optional[dict[str, Any]],
    headers: dict[str, str],
    timeout_s: float,
    verify_tls: bool,
) -> JsonDict:
    data = None
    req_headers = dict(headers)

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")

    req = Request(url, data=data, headers=req_headers, method=method.upper())
    ssl_ctx = _make_ssl_context(verify_tls) if url.lower().startswith("https://") else None

    try:
        with urlopen(req, timeout=timeout_s, context=ssl_ctx) as resp:
            raw = resp.read() or b""
            if not raw:
                return {}
            try:
                obj = json.loads(raw.decode("utf-8", errors="replace"))
                return obj if isinstance(obj, dict) else {"value": obj}
            except Exception:
                return {"raw_text": raw.decode("utf-8", errors="replace")}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body or exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(str(exc.reason or exc)) from exc


def _unwrap_payload(obj: Any) -> JsonDict:
    cur = obj if isinstance(obj, dict) else {}
    while isinstance(cur, dict):
        nxt = None
        for key in ("result", "data"):
            value = cur.get(key)
            if isinstance(value, dict):
                nxt = value
                break
        if nxt is None:
            break
        cur = nxt
    return cur if isinstance(cur, dict) else {}


def _difficulty_to_target_hex(difficulty: int) -> str:
    diff = max(1, int(difficulty))
    target256 = ((1 << 256) - 1) // diff
    return target256.to_bytes(32, "little", signed=False).hex()


def _stable_job_id(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part or "").encode("utf-8", errors="replace"))
        h.update(b"|")
    return h.hexdigest()[:32]


class MoneroRpcClient:
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

        self.current_job: Optional[MiningJob] = None
        self.session_id: str = ""

        self._base_url = _normalize_url(config.monero_rpc_url, config.monero_rpc_force_scheme)
        self._timeout_s = max(1.0, float(config.monero_rpc_timeout_s))
        self._verify_tls = bool(config.monero_rpc_verify_tls)
        self._prefix = str(config.monero_rpc_prefix or "").strip() or "/v1"
        self._client_id = (config.monero_rpc_client_id or "").strip() or secrets.token_hex(8)

        self._headers = _make_headers(config.monero_rpc_token)
        self._stop = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._feeder_thread: Optional[threading.Thread] = None

        self._state_lock = threading.RLock()
        self._lease_lock = threading.RLock()
        self._lease_cache: list[NonceWindow] = []

        self._job_seq: int = 0
        self._feed_seq: int = 0
        self._last_poll_error: str = ""
        self._waiting_logged = False

        self._feeder_last_error: str = ""
        self._feeder_last_fp: str = ""
        self._feeder_last_push_at: float = 0.0

        # blocknet feeder session state
        self._blocknet_session: str = ""
        self._blocknet_headers = _make_headers(config.blocknet_api_token)

        # current upstream route for submit
        self._job_source: str = ""
        self._upstream_kind: str = ""
        self._upstream_session: str = ""
        self._upstream_job_id: str = ""

    def connect(self) -> None:
        if not self._base_url:
            raise RuntimeError("monero_rpc_url is empty")

        self._stop.clear()
        self.on_status("connecting")

        feeder_mode = self.config.normalized_monero_rpc_feeder_mode()
        if feeder_mode != "none":
            self._feeder_thread = threading.Thread(
                target=self._feeder_loop,
                name="MoneroRpcFeederThread",
                daemon=True,
            )
            self._feeder_thread.start()
            self.on_log(f"[monerorpc] embedded feeder enabled source={feeder_mode}")

        try:
            got_job = self._fetch_and_publish_job(log_missing=True)
        except Exception as exc:
            self.on_log(f"[monerorpc] initial job fetch failed: {exc}")
            got_job = False

        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="MoneroRpcPollThread",
            daemon=True,
        )
        self._poll_thread.start()

        self.on_status("connected" if got_job else "waiting_for_feeder")

    def close(self) -> None:
        self._stop.set()

        if self._poll_thread is not None:
            try:
                self._poll_thread.join(timeout=2.0)
            except Exception:
                pass
            self._poll_thread = None

        if self._feeder_thread is not None:
            try:
                self._feeder_thread.join(timeout=2.0)
            except Exception:
                pass
            self._feeder_thread = None

        try:
            self._blocknet_close_session()
        except Exception:
            pass

        with self._lease_lock:
            self._lease_cache.clear()

    def request_scan_window(self, span: int) -> Optional[NonceWindow]:
        wanted = max(1, int(span))

        with self._lease_lock:
            cached = self._consume_cached_window_locked(wanted)
            if cached is not None:
                return cached

        lease = self._fetch_lease(wanted)
        if lease is None:
            return None

        with self._lease_lock:
            self._lease_cache.append(lease)
            return self._consume_cached_window_locked(wanted)

    def submit(self, verified: VerifiedShare) -> SubmitResult:
        # For blocknet-fed broker jobs, allow direct upstream submit.
        route = self._get_upstream_submit_route()
        if route is not None:
            try:
                obj = self._submit_to_blocknet_upstream(
                    session=route["session"],
                    upstream_job_id=route["job_id"],
                    verified=verified,
                )
                return self._parse_submit_response(obj)
            except Exception as exc:
                return SubmitResult(
                    accepted=False,
                    status="backend_error",
                    error=str(exc),
                    raw={"error": str(exc), "route": route},
                    reject_class="pool_backend_error",
                    backend_error=True,
                )

        payload = {
            "miner_id": self._client_id,
            "job_id": verified.job_id,
            "nonce": verified.nonce_hex,
            "result": verified.result_hex,
            "solution_blob_hex": verified.solution_blob_hex,
            "assigned_work": verified.assigned_work,
            "actual_work": verified.actual_work,
            "credited_work": verified.credited_work,
            "quality": verified.quality,
            "actual_tail_u64": verified.actual_tail_u64,
            "predicted_tail_u64": verified.predicted_tail_u64,
            "rank_score_u64": verified.rank_score_u64,
            "tune_bucket": verified.tune_bucket,
            "tune_tail_bin": verified.tune_tail_bin,
            "rank_quality": verified.rank_quality,
            "threshold_quality": verified.threshold_quality,
            "gpu_hash_hex": verified.gpu_hash_hex,
            "predictor_hash_match": bool(verified.predictor_hash_match),
        }
        if self._job_seq > 0:
            payload["job_seq"] = int(self._job_seq)

        try:
            obj = self._broker_post("monero_rpc/submit/share", payload)
            return self._parse_submit_response(obj)
        except Exception as exc:
            return SubmitResult(
                accepted=False,
                status="backend_error",
                error=str(exc),
                raw={"error": str(exc)},
                reject_class="pool_backend_error",
                backend_error=True,
            )

    def _broker_get(self, path: str) -> JsonDict:
        return _json_request(
            _join_url(self._base_url, self._prefix, path),
            method="GET",
            payload=None,
            headers=self._headers,
            timeout_s=self._timeout_s,
            verify_tls=self._verify_tls,
        )

    def _broker_post(self, path: str, payload: JsonDict) -> JsonDict:
        return _json_request(
            _join_url(self._base_url, self._prefix, path),
            method="POST",
            payload=payload,
            headers=self._headers,
            timeout_s=self._timeout_s,
            verify_tls=self._verify_tls,
        )

    def _fetch_and_publish_job(self, *, log_missing: bool) -> bool:
        try:
            obj = self._broker_get("monero_rpc/job/current")
        except Exception as exc:
            msg = str(exc)
            if "HTTP 404" in msg or "no current job" in msg.lower():
                if log_missing and not self._waiting_logged:
                    self._waiting_logged = True
                    self.on_log(
                        "[monerorpc] broker has no current job yet; "
                        "waiting for feeder POST /v1/monero_rpc/job/push"
                    )
                self.on_status("waiting_for_feeder")
                return False
            raise

        payload = _unwrap_payload(obj)
        job_dict = payload.get("job") if isinstance(payload.get("job"), dict) else payload
        job = self._parse_job_dict(job_dict)
        if job is None:
            if log_missing and not self._waiting_logged:
                self._waiting_logged = True
                self.on_log(
                    "[monerorpc] broker has no current job yet; "
                    "waiting for feeder POST /v1/monero_rpc/job/push"
                )
            self.on_status("waiting_for_feeder")
            return False

        self._waiting_logged = False
        changed = self._set_current_job(job, job_dict)
        self.on_status("mining")
        return changed or True

    def _parse_job_dict(self, job_dict: Any) -> Optional[MiningJob]:
        if not isinstance(job_dict, dict):
            return None

        job_id = str(job_dict.get("job_id") or job_dict.get("id") or "").strip()
        blob_hex = _normalize_hex(job_dict.get("blob_hex") or job_dict.get("blob") or "")
        target_hex = _normalize_hex(job_dict.get("target_hex") or job_dict.get("target") or "")
        seed_hash_hex = _normalize_hex(
            job_dict.get("seed_hash_hex") or job_dict.get("seed_hash") or ""
        )

        if not job_id or not blob_hex or not target_hex:
            return None

        return MiningJob(
            job_id=job_id,
            blob_hex=blob_hex,
            target_hex=target_hex,
            session_id=str(job_dict.get("session_id") or job_dict.get("session") or ""),
            seed_hash_hex=seed_hash_hex,
            height=_coerce_int(job_dict.get("height"), 0),
            algo=str(job_dict.get("algo") or "rx/0"),
            submit_blob_hex=_normalize_hex(
                job_dict.get("submit_blob_hex")
                or job_dict.get("submit_blob")
                or job_dict.get("blocktemplate_blob")
                or blob_hex
            ),
            reserved_offset=_coerce_int(
                job_dict.get("reserved_offset", job_dict.get("reserve_offset")),
                0,
            ),
            backend="monerorpc",
        )

    def _extract_upstream_route(self, job_dict: Any) -> dict[str, str]:
        if not isinstance(job_dict, dict):
            return {
                "source": "",
                "kind": "",
                "session": "",
                "job_id": "",
            }

        upstream = job_dict.get("upstream")
        if not isinstance(upstream, dict):
            upstream = {}

        source = str(job_dict.get("source") or "").strip().lower()
        kind = str(upstream.get("kind") or "").strip().lower()
        session = str(upstream.get("session") or "").strip()
        upstream_job_id = str(upstream.get("job_id") or "").strip()

        if not kind and source == "blocknet_p2pool":
            kind = "blocknet_p2pool"

        if kind == "blocknet_p2pool":
            if not session and self._blocknet_session:
                session = self._blocknet_session
            if not upstream_job_id:
                upstream_job_id = str(job_dict.get("job_id") or job_dict.get("id") or "").strip()

        return {
            "source": source,
            "kind": kind,
            "session": session,
            "job_id": upstream_job_id,
        }

    def _set_current_job(self, job: MiningJob, raw_job_dict: Optional[JsonDict] = None) -> bool:
        route = self._extract_upstream_route(raw_job_dict or {})

        with self._state_lock:
            old = self.current_job
            changed = (
                old is None
                or old.job_id != job.job_id
                or old.blob_hex != job.blob_hex
                or old.target_hex != job.target_hex
                or old.seed_hash_hex != job.seed_hash_hex
                or old.session_id != job.session_id
            )
            if not changed:
                return False

            self.current_job = job
            self.session_id = job.session_id or self.session_id
            self._job_source = route["source"]
            self._upstream_kind = route["kind"]
            self._upstream_session = route["session"]
            self._upstream_job_id = route["job_id"]

        with self._lease_lock:
            self._lease_cache.clear()

        route_text = ""
        if self._upstream_kind:
            route_text = (
                f" route={self._upstream_kind}"
                f" upstream_job_id={self._upstream_job_id or '-'}"
                f" session={self._upstream_session or '-'}"
            )

        self.on_log(
            f"[monerorpc] new job job_id={job.job_id} "
            f"height={job.height} algo={job.algo} seed={job.seed_hash_hex[:16] or '-'}"
            f"{route_text}"
        )
        self.on_job(job)
        return True

    def _get_upstream_submit_route(self) -> Optional[dict[str, str]]:
        with self._state_lock:
            if (
                self._upstream_kind == "blocknet_p2pool"
                and self._upstream_session
                and self._upstream_job_id
            ):
                return {
                    "kind": self._upstream_kind,
                    "session": self._upstream_session,
                    "job_id": self._upstream_job_id,
                }
        return None

    def _poll_loop(self) -> None:
        interval_s = max(0.05, float(self.config.monero_rpc_poll_interval_ms) / 1000.0)
        while not self._stop.is_set():
            try:
                self._fetch_and_publish_job(log_missing=False)
                self._last_poll_error = ""
            except Exception as exc:
                msg = str(exc)
                if msg != self._last_poll_error:
                    self._last_poll_error = msg
                    self.on_log(f"[monerorpc] poll failed: {msg}")
            finally:
                self._stop.wait(interval_s)

    def _fetch_lease(self, wanted: int) -> Optional[NonceWindow]:
        payload = {
            "miner_id": self._client_id,
            "count": max(1, int(wanted)),
        }
        if self._job_seq > 0:
            payload["job_seq"] = int(self._job_seq)

        try:
            obj = self._broker_post("monero_rpc/lease/alloc", payload)
        except Exception as exc:
            self.on_log(f"[monerorpc] lease request failed: {exc}")
            return None

        payload = _unwrap_payload(obj)
        lease_dict = payload.get("lease") if isinstance(payload.get("lease"), dict) else payload

        start_nonce = _coerce_int(lease_dict.get("start_nonce"), -1)
        count = _coerce_int(lease_dict.get("count"), 0)
        if start_nonce < 0 or count <= 0:
            return None

        job_seq = _coerce_int(lease_dict.get("job_seq", payload.get("job_seq", self._job_seq)), 0)
        if job_seq > 0:
            self._job_seq = job_seq

        return NonceWindow(
            start_nonce=start_nonce & 0xFFFFFFFF,
            count=count,
            lease_id=str(lease_dict.get("lease_id") or ""),
            expires_at=0.0,
            source="monerorpc",
            job_seq=job_seq,
        )

    def _consume_cached_window_locked(self, wanted: int) -> Optional[NonceWindow]:
        while self._lease_cache and self._lease_cache[0].count <= 0:
            self._lease_cache.pop(0)

        if not self._lease_cache:
            return None

        head = self._lease_cache[0]
        take = max(1, min(int(wanted), int(head.count)))

        out = NonceWindow(
            start_nonce=int(head.start_nonce) & 0xFFFFFFFF,
            count=take,
            lease_id=head.lease_id,
            expires_at=head.expires_at,
            source=head.source,
            job_seq=head.job_seq,
        )

        head.start_nonce = (int(head.start_nonce) + take) & 0xFFFFFFFF
        head.count -= take
        if head.count <= 0:
            self._lease_cache.pop(0)

        return out

    def _parse_submit_response(self, obj: Any) -> SubmitResult:
        payload = _unwrap_payload(obj)
        submit_dict = payload.get("submit") if isinstance(payload.get("submit"), dict) else payload

        status = str(
            submit_dict.get("status")
            or submit_dict.get("result")
            or submit_dict.get("message")
            or ""
        ).strip()

        accepted_raw = submit_dict.get("accepted", submit_dict.get("ok"))
        if isinstance(accepted_raw, bool):
            accepted = accepted_raw
        else:
            accepted = status.lower() in {"ok", "accepted", "success", "found"}

        error = str(
            submit_dict.get("error")
            or submit_dict.get("message")
            or ""
        ).strip()

        raw_text = " ".join(
            [
                status,
                error,
                str(submit_dict.get("reject_class") or ""),
                str(submit_dict),
            ]
        ).lower()

        stale = "stale" in raw_text
        duplicate = "duplicate" in raw_text or "already submitted" in raw_text
        invalid = (
            "invalid" in raw_text
            or "low difficulty" in raw_text
            or "bad nonce" in raw_text
            or "bad result" in raw_text
        )
        backend_error = (
            not accepted
            and (
                "timeout" in raw_text
                or "connect" in raw_text
                or "gateway" in raw_text
                or "not connected" in raw_text
                or "temporar" in raw_text
                or "session" in raw_text
                or "unknown_session" in raw_text
            )
        )

        reject_class = str(submit_dict.get("reject_class") or "").strip().lower()
        if not reject_class:
            if accepted:
                reject_class = ""
            elif stale:
                reject_class = "pool_stale"
            elif duplicate:
                reject_class = "pool_duplicate"
            elif invalid:
                reject_class = "pool_invalid"
            elif backend_error:
                reject_class = "pool_backend_error"
            else:
                reject_class = "pool_rejected"

        return SubmitResult(
            accepted=accepted,
            status=status or ("accepted" if accepted else "rejected"),
            error=error,
            raw=obj,
            reject_class=reject_class,
            stale=stale,
            duplicate=duplicate,
            invalid=invalid,
            backend_error=backend_error,
        )

    # ----------------------------
    # Direct upstream submit for blocknet-fed jobs
    # ----------------------------

    def _submit_to_blocknet_upstream(
        self,
        *,
        session: str,
        upstream_job_id: str,
        verified: VerifiedShare,
    ) -> JsonDict:
        if not session:
            raise RuntimeError("missing blocknet upstream session")
        if not upstream_job_id:
            raise RuntimeError("missing blocknet upstream job_id")

        payload = {
            "session": session,
            "job_id": upstream_job_id,
            "nonce": _normalize_hex(verified.nonce_hex),
            "result": _normalize_hex(verified.result_hex),
        }

        obj = self._blocknet_post("p2pool/submit", payload)

        if not isinstance(obj, dict):
            raise RuntimeError(f"blocknet p2pool submit returned invalid response: {obj!r}")

        # normalize a few common blocknet response shapes
        if "accepted" not in obj and "ok" in obj:
            obj["accepted"] = bool(obj.get("ok"))
        if "status" not in obj:
            obj["status"] = "accepted" if bool(obj.get("accepted")) else "rejected"

        return obj

    # ----------------------------
    # Embedded feeder
    # ----------------------------

    def _feeder_loop(self) -> None:
        interval_s = max(0.10, float(self.config.monero_rpc_feeder_poll_interval_ms) / 1000.0)
        mode = self.config.normalized_monero_rpc_feeder_mode()

        while not self._stop.is_set():
            try:
                changed = False
                if mode == "solo":
                    changed = self._feed_once_solo()
                elif mode == "blocknet":
                    changed = self._feed_once_blocknet()
                self._feeder_last_error = ""
                if changed:
                    try:
                        self._fetch_and_publish_job(log_missing=False)
                    except Exception:
                        pass
            except Exception as exc:
                msg = str(exc)
                if msg != self._feeder_last_error:
                    self._feeder_last_error = msg
                    self.on_log(f"[monerorpc-feeder] {mode} feeder failed: {msg}")
            finally:
                self._stop.wait(interval_s)

    def _feed_once_solo(self) -> bool:
        wallet = (self.config.solo_wallet_address or "").strip()
        daemon_url = (self.config.solo_daemon_rpc_url or "").strip()
        if not wallet:
            raise RuntimeError("solo_wallet_address is required for solo feeder")
        if not daemon_url:
            raise RuntimeError("solo_daemon_rpc_url is required for solo feeder")

        rpc_url = daemon_url.rstrip("/")
        if not rpc_url.endswith("/json_rpc"):
            rpc_url = f"{rpc_url}/json_rpc"

        obj = _json_request(
            rpc_url,
            method="POST",
            payload={
                "jsonrpc": "2.0",
                "id": "0",
                "method": "get_block_template",
                "params": {
                    "wallet_address": wallet,
                    "reserve_size": int(self.config.solo_reserve_size),
                },
            },
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout_s=max(1.0, float(self.config.monero_rpc_timeout_s)),
            verify_tls=False,
        )

        result = obj.get("result") if isinstance(obj, dict) else None
        if not isinstance(result, dict):
            raise RuntimeError(f"bad monerod get_block_template response: {obj!r}")

        blob_hex = _normalize_hex(result.get("blocktemplate_blob") or "")
        seed_hash_hex = _normalize_hex(result.get("seed_hash") or "")
        height = _coerce_int(result.get("height"), 0)
        difficulty = _coerce_int(result.get("difficulty"), 0)
        reserved_offset = _coerce_int(result.get("reserved_offset"), 0)

        if not blob_hex:
            raise RuntimeError("monerod get_block_template missing blocktemplate_blob")
        if not seed_hash_hex:
            raise RuntimeError("monerod get_block_template missing seed_hash")
        if difficulty <= 0:
            raise RuntimeError("monerod get_block_template missing difficulty")

        target_hex = _difficulty_to_target_hex(difficulty)
        job_id = _stable_job_id(blob_hex, seed_hash_hex, target_hex, str(height), "solo")

        payload = {
            "job_id": job_id,
            "blob": blob_hex,
            "submit_blob_hex": blob_hex,
            "seed_hash": seed_hash_hex,
            "target": target_hex,
            "nonce_offset": int(self.config.nonce_offset),
            "reserved_offset": reserved_offset,
            "height": height,
            "algo": "rx/0",
            "source": "monerod",
        }
        return self._push_job_to_broker(payload)

    def _feed_once_blocknet(self) -> bool:
        job = self._blocknet_get_job()
        if not isinstance(job, dict):
            raise RuntimeError("blocknet feeder returned no job")

        blob_hex = _normalize_hex(job.get("blob_hex") or job.get("blob") or "")
        seed_hash_hex = _normalize_hex(job.get("seed_hash_hex") or job.get("seed_hash") or "")
        target_hex = _normalize_hex(job.get("target_hex") or job.get("target") or "")
        height = _coerce_int(job.get("height"), 0)
        algo = str(job.get("algo") or "rx/0")
        job_id = str(job.get("job_id") or job.get("id") or "").strip()

        if not blob_hex or not target_hex:
            raise RuntimeError(f"blocknet p2pool job missing blob/target: {job!r}")
        if not job_id:
            job_id = _stable_job_id(blob_hex, seed_hash_hex, target_hex, str(height), "blocknet")

        payload = {
            "job_id": job_id,
            "blob": blob_hex,
            "submit_blob_hex": _normalize_hex(job.get("submit_blob_hex") or blob_hex),
            "seed_hash": seed_hash_hex,
            "target": target_hex,
            "nonce_offset": int(self.config.nonce_offset),
            "height": height,
            "algo": algo,
            "source": "blocknet_p2pool",
            "upstream": {
                "kind": "blocknet_p2pool",
                "session": self._blocknet_session,
                "job_id": str(job.get("job_id") or job.get("id") or job_id),
            },
        }
        return self._push_job_to_broker(payload)

    def _push_job_to_broker(self, payload: JsonDict) -> bool:
        fp = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        now = time.time()
        changed = fp != self._feeder_last_fp
        if (not changed) and (now - self._feeder_last_push_at) < 5.0:
            return False

        obj = self._broker_post("monero_rpc/job/push", payload)
        if not bool(obj.get("ok", True)):
            raise RuntimeError(f"job push failed: {obj}")

        self._feeder_last_push_at = now
        self._feeder_last_fp = fp

        if changed:
            self.on_log(
                f"[monerorpc-feeder] pushed job source={payload.get('source')} "
                f"job_id={payload.get('job_id')} height={payload.get('height')}"
            )
        return changed

    # ----------------------------
    # BlockNet helpers
    # ----------------------------

    def _blocknet_base_url(self) -> str:
        base = _normalize_url(self.config.blocknet_api_relay, self.config.blocknet_force_scheme)
        if not base:
            raise RuntimeError("blocknet_api_relay is required for blocknet feeder/submit")
        return base

    def _blocknet_post(self, path: str, payload: JsonDict) -> JsonDict:
        return _json_request(
            _join_url(self._blocknet_base_url(), self.config.blocknet_api_prefix, path),
            method="POST",
            payload=payload,
            headers=self._blocknet_headers,
            timeout_s=max(1.0, float(self.config.blocknet_timeout_s)),
            verify_tls=bool(self.config.blocknet_verify_tls),
        )

    def _blocknet_open_session(self) -> JsonDict:
        obj = self._blocknet_post("p2pool/open", {})
        if not obj.get("ok"):
            raise RuntimeError(f"blocknet p2pool open failed: {obj}")
        self._blocknet_session = str(obj.get("session") or "")
        if not self._blocknet_session:
            raise RuntimeError(f"blocknet p2pool open missing session: {obj}")
        job = obj.get("job")
        return job if isinstance(job, dict) else {}

    def _blocknet_close_session(self) -> None:
        if not self._blocknet_session:
            return
        session = self._blocknet_session
        self._blocknet_session = ""
        try:
            self._blocknet_post("p2pool/close", {"session": session})
        except Exception:
            pass

    def _blocknet_get_job(self) -> JsonDict:
        if not self._blocknet_session:
            job = self._blocknet_open_session()
            if job:
                return job

        try:
            poll = self._blocknet_post(
                "p2pool/poll",
                {
                    "session": self._blocknet_session,
                    "max_msgs": int(self.config.blocknet_poll_max_msgs),
                },
            )
            if poll.get("ok"):
                job = poll.get("job")
                if isinstance(job, dict) and job:
                    return job
        except Exception as exc:
            text = str(exc).lower()
            if "session" in text or "not open" in text or "unknown_session" in text:
                self._blocknet_session = ""
            else:
                raise

        if not self._blocknet_session:
            job = self._blocknet_open_session()
            if job:
                return job

        job_resp = self._blocknet_post("p2pool/job", {"session": self._blocknet_session})
        if not job_resp.get("ok"):
            raise RuntimeError(f"blocknet p2pool job failed: {job_resp}")

        job = job_resp.get("job")
        if not isinstance(job, dict):
            raise RuntimeError(f"blocknet p2pool job missing payload: {job_resp}")
        return job