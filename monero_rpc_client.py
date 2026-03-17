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


def _coerce_boolish(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return bool(default)


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


def _normalize_upstream_kind(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"blocknet", "blocknet_p2pool"}:
        return "blocknet"
    return text


def _full_difficulty_from_source(src: JsonDict) -> int:
    wide_raw = str(src.get("wide_difficulty") or "").strip().lower()
    if wide_raw.startswith("0x"):
        wide_raw = wide_raw[2:]
    if wide_raw:
        try:
            return int(wide_raw, 16)
        except Exception:
            pass

    diff_lo = _coerce_int(src.get("difficulty"), 0)
    diff_hi = _coerce_int(src.get("difficulty_top64"), 0)
    full = (int(diff_hi) << 64) | (int(diff_lo) & 0xFFFFFFFFFFFFFFFF)
    return full if full > 0 else 0


def _difficulty_fields_from_target_hex(target_hex: Any) -> dict[str, Any]:
    text = _normalize_hex(target_hex)
    if not text or (len(text) % 2) != 0:
        return {}

    try:
        raw = bytes.fromhex(text)
    except Exception:
        return {}

    if not raw:
        return {}

    target_int = int.from_bytes(raw, "little", signed=False)
    if target_int <= 0:
        return {}

    bits = len(raw) * 8
    max_target = (1 << bits) - 1
    diff = max_target // target_int
    if diff <= 0:
        diff = 1

    return {
        "difficulty": int(diff & 0xFFFFFFFFFFFFFFFF),
        "difficulty_top64": int((diff >> 64) & 0xFFFFFFFFFFFFFFFF),
        "wide_difficulty": hex(diff),
    }


def _job_source_kind(job_dict: JsonDict) -> tuple[str, str]:
    source = str(job_dict.get("source") or "").strip().lower()
    upstream = job_dict.get("upstream")
    kind = ""

    if isinstance(upstream, dict):
        kind = _normalize_upstream_kind(upstream.get("kind") or "")

    if not kind and source in {"blocknet", "blocknet_p2pool"}:
        kind = "blocknet"

    return source, kind


def _target_hex_for_job(job_dict: JsonDict) -> str:
    direct = _normalize_hex(job_dict.get("target_hex") or job_dict.get("target") or "")
    source, kind = _job_source_kind(job_dict)

    if kind == "blocknet" or source in {"blocknet", "blocknet_p2pool"}:
        return direct

    if len(direct) >= 64:
        return direct[:64]

    full_diff = _full_difficulty_from_source(job_dict)
    if full_diff > 0:
        return _difficulty_to_target_hex(full_diff)

    return direct

def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _as_dict(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def _backend_name(cfg: MinerConfig) -> str:
    return str(getattr(cfg, "backend", "") or "").strip().lower()


def _extract_p2pool_job_id(payload: JsonDict) -> str:
    root = _as_dict(payload)
    upstream = _as_dict(root.get("upstream"))
    job = _as_dict(root.get("job"))
    meta = _as_dict(root.get("meta"))
    p2pool = _as_dict(root.get("p2pool"))
    result = _as_dict(root.get("result"))

    # IMPORTANT:
    # Only return EXPLICIT upstream/p2pool ids here.
    # Do NOT fall back to generic job_id/id because broker current-job
    # rewrites those into synthetic template ids like 3632254::...
    return _first_text(
        root.get("p2pool_job_id"),
        root.get("submit_job_id"),
        root.get("pool_job_id"),
        root.get("upstream_job_id"),

        upstream.get("job_id"),
        upstream.get("p2pool_job_id"),
        upstream.get("submit_job_id"),

        job.get("p2pool_job_id"),
        job.get("submit_job_id"),
        job.get("pool_job_id"),
        job.get("upstream_job_id"),

        meta.get("p2pool_job_id"),
        meta.get("submit_job_id"),
        meta.get("pool_job_id"),
        meta.get("upstream_job_id"),

        p2pool.get("job_id"),
        p2pool.get("p2pool_job_id"),
        p2pool.get("submit_job_id"),

        result.get("p2pool_job_id"),
        result.get("submit_job_id"),
        result.get("pool_job_id"),
        result.get("upstream_job_id"),
    )


def _attach_job_runtime_fields(
    job_obj: MiningJob,
    *,
    template_job_id: str,
    submit_job_id: str,
    upstream_job_id: str,
) -> None:
    try:
        setattr(job_obj, "template_job_id", str(template_job_id or "").strip())
    except Exception:
        pass
    try:
        setattr(job_obj, "submit_job_id", str(submit_job_id or "").strip())
    except Exception:
        pass
    try:
        setattr(job_obj, "upstream_job_id", str(upstream_job_id or "").strip())
    except Exception:
        pass


def _select_job_identity(cfg: MinerConfig, job_dict: JsonDict) -> tuple[str, str, str, str, str]:
    source, kind = _job_source_kind(job_dict)
    backend = _backend_name(cfg)

    broker_job_id = _first_text(
        job_dict.get("broker_job_id"),
        job_dict.get("template_job_id"),
        job_dict.get("job_id"),
        job_dict.get("id"),
    )

    upstream_job_id = _extract_p2pool_job_id(job_dict)

    is_blocknet = (
        kind == "blocknet"
        or source in {"blocknet", "blocknet_p2pool"}
        or (backend == "blocknet" and bool(upstream_job_id))
    )

    if is_blocknet:
        visible_job_id = upstream_job_id or broker_job_id
        submit_job_id = upstream_job_id or broker_job_id
        template_job_id = broker_job_id or upstream_job_id
    else:
        visible_job_id = broker_job_id or upstream_job_id
        submit_job_id = upstream_job_id or broker_job_id or visible_job_id
        template_job_id = broker_job_id or visible_job_id

    return (
        str(visible_job_id or "").strip(),
        str(submit_job_id or "").strip(),
        str(template_job_id or "").strip(),
        str(source or "").strip().lower(),
        str(kind or "").strip().lower(),
    )


def _verified_submit_job_id(verified: VerifiedShare) -> str:
    job_obj = getattr(verified, "job", None)
    return _first_text(
        getattr(verified, "submit_job_id", ""),
        getattr(verified, "job_id", ""),
        getattr(job_obj, "submit_job_id", ""),
        getattr(job_obj, "upstream_job_id", ""),
        getattr(job_obj, "job_id", ""),
    )

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
        self._blocknet_headers = _make_headers(config.blocknet_api_token)

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

        self._blocknet_session: str = ""
        self._blocknet_route_cache: dict[str, dict[str, Any]] = {}

        self._job_source: str = ""
        self._upstream_kind: str = ""
        self._upstream_session: str = ""
        self._upstream_job_id: str = ""

        self._solo_last_template_sig: str = ""
        self._solo_last_template_height: int = 0
        self._solo_last_template_prev_hash: str = ""
        self._solo_last_push_sig: str = ""
        self._solo_last_push_at: float = 0.0
        self._solo_force_repush_s: float = 15.0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # broker transport
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # blocknet transport
    # ------------------------------------------------------------------

    def _blocknet_route_match_key(self, job_dict: Any) -> str:
        if not isinstance(job_dict, dict):
            return ""

        blob_hex = _normalize_hex(
            job_dict.get("blob_hex")
            or job_dict.get("blob")
            or job_dict.get("blockhashing_blob")
            or ""
        )
        submit_blob_hex = _normalize_hex(
            job_dict.get("submit_blob_hex")
            or job_dict.get("submit_blob")
            or job_dict.get("blocktemplate_blob")
            or blob_hex
        )
        seed_hash_hex = _normalize_hex(
            job_dict.get("seed_hash_hex")
            or job_dict.get("seed_hash")
            or ""
        )
        height = _coerce_int(job_dict.get("height"), 0)
        reserved_offset = _coerce_int(
            job_dict.get("reserved_offset", job_dict.get("reserve_offset")),
            0,
        )
        nonce_offset = _coerce_int(job_dict.get("nonce_offset"), -1)
        prev_hash = _normalize_hex(job_dict.get("prev_hash") or "")
        next_seed_hash_hex = _normalize_hex(
            job_dict.get("next_seed_hash_hex")
            or job_dict.get("next_seed_hash")
            or ""
        )
        session = str(
            job_dict.get("upstream_session")
            or job_dict.get("session")
            or job_dict.get("session_id")
            or self._blocknet_session
            or ""
        ).strip()

        # IMPORTANT:
        # Do NOT include target_hex, template_sig, or generic job_id here.
        # Broker rewrites those.
        return _stable_job_id(
            session,
            blob_hex,
            submit_blob_hex,
            seed_hash_hex,
            str(height),
            str(reserved_offset),
            str(nonce_offset),
            prev_hash,
            next_seed_hash_hex,
            "blocknet_route_match",
        )

    def _blocknet_route_aliases(self, job_dict: Any) -> list[str]:
        if not isinstance(job_dict, dict):
            return []

        aliases: list[str] = []

        match_key = self._blocknet_route_match_key(job_dict)
        if match_key:
            aliases.append(f"mk:{match_key}")

        for key in (
                "template_sig",
                "broker_job_id",
                "p2pool_job_id",
                "upstream_job_id",
                "submit_job_id",
                "pool_job_id",
                "job_id",
                "id",
        ):
            text = str(job_dict.get(key) or "").strip()
            if text:
                aliases.append(f"{key}:{text}")

        upstream = job_dict.get("upstream")
        if isinstance(upstream, dict):
            for key in ("job_id", "p2pool_job_id", "submit_job_id"):
                text = str(upstream.get(key) or "").strip()
                if text:
                    aliases.append(f"upstream.{key}:{text}")

        return aliases
    def _blocknet_route_fingerprint(self, job_dict: Any) -> str:
        if not isinstance(job_dict, dict):
            return ""

        blob_hex = _normalize_hex(
            job_dict.get("blob_hex")
            or job_dict.get("blob")
            or job_dict.get("blockhashing_blob")
            or ""
        )
        submit_blob_hex = _normalize_hex(
            job_dict.get("submit_blob_hex")
            or job_dict.get("submit_blob")
            or job_dict.get("blocktemplate_blob")
            or blob_hex
        )
        seed_hash_hex = _normalize_hex(job_dict.get("seed_hash_hex") or job_dict.get("seed_hash") or "")
        target_hex = _target_hex_for_job(job_dict)
        height = _coerce_int(job_dict.get("height"), 0)
        reserved_offset = _coerce_int(
            job_dict.get("reserved_offset", job_dict.get("reserve_offset")),
            0,
        )
        nonce_offset = _coerce_int(job_dict.get("nonce_offset"), -1)
        prev_hash = _normalize_hex(job_dict.get("prev_hash") or "")
        next_seed_hash_hex = _normalize_hex(
            job_dict.get("next_seed_hash_hex") or job_dict.get("next_seed_hash") or ""
        )
        template_sig = str(job_dict.get("template_sig") or "").strip()

        return _stable_job_id(
            blob_hex,
            submit_blob_hex,
            seed_hash_hex,
            target_hex,
            str(height),
            str(reserved_offset),
            str(nonce_offset),
            prev_hash,
            next_seed_hash_hex,
            template_sig,
            "blocknet_route_fp",
        )

    def _remember_blocknet_route(
            self,
            job_dict: JsonDict,
            *,
            p2pool_job_id: str,
            session: str,
    ) -> None:
        p2pool_job_id = str(p2pool_job_id or "").strip()
        if not p2pool_job_id:
            return

        entry = {
            "job_id": p2pool_job_id,
            "session": str(session or "").strip(),
            "ts": time.time(),
        }

        keys = set(self._blocknet_route_aliases(job_dict))
        keys.add(f"p2pool:{p2pool_job_id}")

        with self._state_lock:
            for key in keys:
                self._blocknet_route_cache[key] = entry

            if len(self._blocknet_route_cache) > 1024:
                items = sorted(
                    self._blocknet_route_cache.items(),
                    key=lambda kv: float(kv[1].get("ts", 0.0)),
                )
                for key, _ in items[:-768]:
                    self._blocknet_route_cache.pop(key, None)

    def _resolve_blocknet_route(self, job_dict: Any) -> Optional[dict[str, str]]:
        if not isinstance(job_dict, dict):
            return None

        keys = self._blocknet_route_aliases(job_dict)

        with self._state_lock:
            for key in keys:
                entry = self._blocknet_route_cache.get(key)
                if entry:
                    return {
                        "job_id": str(entry.get("job_id") or "").strip(),
                        "session": str(entry.get("session") or "").strip(),
                    }

        return None

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
        if not bool(obj.get("ok", False)):
            raise RuntimeError(f"blocknet p2pool open failed: {obj}")

        session = str(obj.get("session") or "").strip()
        if not session:
            raise RuntimeError(f"blocknet p2pool open missing session: {obj}")

        self._blocknet_session = session
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
            if bool(poll.get("ok", False)):
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
        if not bool(job_resp.get("ok", False)):
            raise RuntimeError(f"blocknet p2pool job failed: {job_resp}")

        job = job_resp.get("job")
        if not isinstance(job, dict):
            raise RuntimeError(f"blocknet p2pool job missing payload: {job_resp}")
        return job

    # ------------------------------------------------------------------
    # job parse / state
    # ------------------------------------------------------------------

    def _route_fingerprint(self, route: dict[str, str]) -> str:
        return "|".join(
            [
                str(route.get("source") or ""),
                str(route.get("kind") or ""),
                str(route.get("session") or ""),
                str(route.get("job_id") or ""),
            ]
        )

    def _extract_upstream_route(self, job_dict: Any) -> dict[str, str]:
        if not isinstance(job_dict, dict):
            return {"source": "", "kind": "", "session": "", "job_id": ""}

        upstream = job_dict.get("upstream")
        if not isinstance(upstream, dict):
            upstream = {}

        source = str(job_dict.get("source") or "").strip().lower()
        kind = _normalize_upstream_kind(
            upstream.get("kind")
            or job_dict.get("upstream_kind")
            or ""
        )

        session = str(
            upstream.get("session")
            or job_dict.get("upstream_session")
            or job_dict.get("session")
            or job_dict.get("session_id")
            or ""
        ).strip()

        upstream_job_id = _first_text(
            upstream.get("job_id"),
            job_dict.get("upstream_job_id"),
            job_dict.get("p2pool_job_id"),
            job_dict.get("submit_job_id"),
            job_dict.get("pool_job_id"),
            _extract_p2pool_job_id(job_dict),
            job_dict.get("job_id"),
            job_dict.get("id"),
        )

        if not kind and source in {"blocknet", "blocknet_p2pool"}:
            kind = "blocknet"

        if kind == "blocknet":
            cached = self._resolve_blocknet_route(job_dict)
            if cached is not None:
                if cached.get("job_id"):
                    upstream_job_id = cached["job_id"]
                if not session and cached.get("session"):
                    session = cached["session"]
            elif not session and self._blocknet_session:
                session = self._blocknet_session

        return {
            "source": source,
            "kind": kind,
            "session": session,
            "job_id": str(upstream_job_id or "").strip(),
        }

    def _parse_job_dict(self, job_dict: Any) -> Optional[MiningJob]:
        if not isinstance(job_dict, dict):
            return None

        source, kind = _job_source_kind(job_dict)
        broker_job_id = _first_text(
            job_dict.get("job_id"),
            job_dict.get("id"),
        )

        cached_route = None
        if kind == "blocknet" or source in {"blocknet", "blocknet_p2pool"}:
            cached_route = self._resolve_blocknet_route(job_dict)

        explicit_upstream_job_id = _extract_p2pool_job_id(job_dict)

        if cached_route is not None and cached_route.get("job_id"):
            visible_job_id = cached_route["job_id"]
            submit_job_id = cached_route["job_id"]
            template_job_id = broker_job_id or cached_route["job_id"]
        elif kind == "blocknet" or source in {"blocknet", "blocknet_p2pool"}:
            visible_job_id = explicit_upstream_job_id or broker_job_id
            submit_job_id = explicit_upstream_job_id or broker_job_id
            template_job_id = broker_job_id or explicit_upstream_job_id
        else:
            visible_job_id = broker_job_id
            submit_job_id = explicit_upstream_job_id or broker_job_id
            template_job_id = broker_job_id

        blob_hex = _normalize_hex(
            job_dict.get("blob_hex")
            or job_dict.get("blob")
            or job_dict.get("blockhashing_blob")
            or ""
        )
        target_hex = _target_hex_for_job(job_dict)
        seed_hash_hex = _normalize_hex(job_dict.get("seed_hash_hex") or job_dict.get("seed_hash") or "")

        if not visible_job_id or not blob_hex or not target_hex:
            return None

        submit_blob_hex = _normalize_hex(
            job_dict.get("submit_blob_hex")
            or job_dict.get("submit_blob")
            or job_dict.get("blocktemplate_blob")
            or blob_hex
        )

        session_id = str(
            job_dict.get("session_id")
            or job_dict.get("session")
            or ""
        ).strip()

        if cached_route is not None and cached_route.get("session") and not session_id:
            session_id = cached_route["session"]

        job = MiningJob(
            job_id=visible_job_id,
            blob_hex=blob_hex,
            target_hex=target_hex,
            session_id=session_id,
            seed_hash_hex=seed_hash_hex,
            height=_coerce_int(job_dict.get("height"), 0),
            algo=str(job_dict.get("algo") or "rx/0"),
            submit_blob_hex=submit_blob_hex,
            reserved_offset=_coerce_int(
                job_dict.get("reserved_offset", job_dict.get("reserve_offset")),
                0,
            ),
            backend="monerorpc",
        )

        _attach_job_runtime_fields(
            job,
            template_job_id=template_job_id,
            submit_job_id=submit_job_id,
            upstream_job_id=submit_job_id,
        )

        return job

    def _set_current_job(self, job: MiningJob, raw_job_dict: Optional[JsonDict] = None) -> bool:
        route = self._extract_upstream_route(raw_job_dict or {})

        with self._state_lock:
            old = self.current_job
            old_route = {
                "source": self._job_source,
                "kind": self._upstream_kind,
                "session": self._upstream_session,
                "job_id": self._upstream_job_id,
            }

            old_template_job_id = getattr(old, "template_job_id", "") if old is not None else ""
            new_template_job_id = getattr(job, "template_job_id", "")

            job_changed = (
                old is None
                or old.job_id != job.job_id
                or old.blob_hex != job.blob_hex
                or old.target_hex != job.target_hex
                or old.seed_hash_hex != job.seed_hash_hex
                or old.session_id != job.session_id
                or old.submit_blob_hex != job.submit_blob_hex
                or old.reserved_offset != job.reserved_offset
                or old_template_job_id != new_template_job_id
            )

            route_changed = self._route_fingerprint(old_route) != self._route_fingerprint(route)
            changed = job_changed or route_changed

            self.current_job = job
            self.session_id = job.session_id or self.session_id

            self._job_source = route["source"]
            self._upstream_kind = route["kind"]
            self._upstream_session = route["session"]
            self._upstream_job_id = route["job_id"]

        if not changed:
            return False

        with self._lease_lock:
            self._lease_cache.clear()

        template_job_id = getattr(job, "template_job_id", "")
        route_text = ""
        if self._upstream_kind:
            route_text = (
                f" route={self._upstream_kind}"
                f" upstream_job_id={self._upstream_job_id or '-'}"
                f" session={self._upstream_session or '-'}"
            )

        self.on_log(
            f"[monerorpc] new job job_id={job.job_id} "
            f"template_job_id={template_job_id or '-'} "
            f"height={job.height} algo={job.algo} "
            f"seed={job.seed_hash_hex[:16] or '-'} "
            f"target={job.target_hex[:16] or '-'}{route_text}"
        )
        self.on_job(job)
        return True

    def _get_upstream_submit_route(self) -> Optional[dict[str, str]]:
        with self._state_lock:
            if (
                self._upstream_kind == "blocknet"
                and self._upstream_session
                and self._upstream_job_id
            ):
                return {
                    "kind": self._upstream_kind,
                    "session": self._upstream_session,
                    "job_id": self._upstream_job_id,
                }
        return None

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

        job_seq = _coerce_int(payload.get("job_seq", obj.get("job_seq", 0)), 0)
        if job_seq > 0:
            self._job_seq = job_seq

        feed_seq = _coerce_int(payload.get("feed_seq", obj.get("feed_seq", 0)), 0)
        if feed_seq > 0:
            self._feed_seq = feed_seq

        # IMPORTANT:
        # Broker may have rewritten blocknet job_id into synthetic template id.
        # Reattach the cached upstream P2Pool job id before parsing.
        if isinstance(job_dict, dict):
            source, kind = _job_source_kind(job_dict)
            if kind == "blocknet" or source in {"blocknet", "blocknet_p2pool"}:
                cached = self._resolve_blocknet_route(job_dict)
                if cached is not None:
                    job_dict = dict(job_dict)
                    if cached.get("job_id"):
                        job_dict["upstream_job_id"] = cached["job_id"]
                        job_dict["p2pool_job_id"] = cached["job_id"]
                        upstream = job_dict.get("upstream")
                        if isinstance(upstream, dict):
                            upstream = dict(upstream)
                        else:
                            upstream = {}
                        upstream["kind"] = "blocknet"
                        upstream["job_id"] = cached["job_id"]
                        job_dict["upstream"] = upstream
                    if cached.get("session"):
                        job_dict.setdefault("upstream_session", cached["session"])
                        job_dict.setdefault("session", cached["session"])
                        job_dict.setdefault("session_id", cached["session"])

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
        self._set_current_job(job, job_dict)
        self.on_status("mining")
        return True

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

    # ------------------------------------------------------------------
    # lease allocation
    # ------------------------------------------------------------------

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

    def _fetch_lease(self, wanted: int) -> Optional[NonceWindow]:
        count = min(max(1, int(wanted)), max(1, int(self.config.monero_rpc_lease_size)))
        payload = {
            "miner_id": self._client_id,
            "count": count,
        }

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

    # ------------------------------------------------------------------
    # submit handling
    # ------------------------------------------------------------------

    def _is_sessionish_error(self, text: str) -> bool:
        s = (text or "").strip().lower()
        return (
            "session" in s
            or "unknown_session" in s
            or "not open" in s
            or "not_open" in s
            or "expired" in s
            or "invalid session" in s
        )

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

        accepted = bool(obj.get("accepted", obj.get("ok", False)))
        status = str(obj.get("status") or ("accepted" if accepted else "rejected"))
        error = str(obj.get("error") or obj.get("message") or "")

        return {
            "accepted": accepted,
            "ok": accepted,
            "status": status,
            "error": error,
            "reject_class": str(obj.get("reject_class") or ""),
            "message": str(obj.get("message") or ""),
            "upstream": "blocknet",
            "raw_upstream": obj,
        }

    def _should_try_blocknet_solo_bridge(self) -> bool:
        with self._state_lock:
            source = (self._job_source or "").strip().lower()

        relay = (self.config.blocknet_api_relay or "").strip()
        if not relay:
            return False

        return source in {"monerod", "solo", "solo_mining"}

    def _submit_to_blocknet_monero_bridge(self, verified: VerifiedShare) -> JsonDict:
        submit_job_id = _verified_submit_job_id(verified)

        payload = {
            "miner_id": self._client_id,
            "job_id": submit_job_id,
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
            payload["expected_job_seq"] = int(self._job_seq)

        obj = self._blocknet_post("monero_rpc/submit/share", payload)
        if not isinstance(obj, dict):
            raise RuntimeError(f"blocknet monero bridge returned invalid response: {obj!r}")

        accepted = bool(obj.get("accepted", obj.get("ok", False)))
        status = str(obj.get("status") or ("accepted" if accepted else "rejected"))
        error = str(obj.get("error") or obj.get("message") or "")

        return {
            "accepted": accepted,
            "ok": accepted,
            "status": status,
            "error": error,
            "reject_class": str(obj.get("reject_class") or ""),
            "message": str(obj.get("message") or ""),
            "upstream": "blocknet_monero_bridge",
            "raw_upstream": obj,
        }

    def _refresh_blocknet_route_after_submit_error(self) -> Optional[dict[str, str]]:
        try:
            self._blocknet_close_session()
        except Exception:
            pass

        try:
            changed = self._feed_once_blocknet()
            if changed:
                self.on_log("[monerorpc] refreshed blocknet session and pushed new upstream route")
            else:
                self.on_log("[monerorpc] refreshed blocknet session")
        except Exception as exc:
            self.on_log(f"[monerorpc] blocknet route refresh failed: {exc}")
            return None

        try:
            self._fetch_and_publish_job(log_missing=False)
        except Exception as exc:
            self.on_log(f"[monerorpc] broker refresh after blocknet route refresh failed: {exc}")
            return None

        return self._get_upstream_submit_route()

    def submit(self, verified: VerifiedShare) -> SubmitResult:
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
                msg = str(exc)

                if self._is_sessionish_error(msg):
                    self.on_log("[monerorpc] upstream submit hit stale blocknet session, refreshing route")
                    retry_route = self._refresh_blocknet_route_after_submit_error()
                    if retry_route is not None:
                        try:
                            obj = self._submit_to_blocknet_upstream(
                                session=retry_route["session"],
                                upstream_job_id=retry_route["job_id"],
                                verified=verified,
                            )
                            return self._parse_submit_response(obj)
                        except Exception as retry_exc:
                            msg = f"{msg} | retry_failed={retry_exc}"

                return SubmitResult(
                    accepted=False,
                    status="backend_error",
                    error=msg,
                    raw={"error": msg, "route": route},
                    reject_class="pool_backend_error",
                    backend_error=True,
                )

        if self._should_try_blocknet_solo_bridge():
            try:
                obj = self._submit_to_blocknet_monero_bridge(verified)
                return self._parse_submit_response(obj)
            except Exception as exc:
                self.on_log(
                    f"[monerorpc] blocknet solo bridge submit failed, "
                    f"falling back to broker submit: {exc}"
                )

        submit_job_id = _verified_submit_job_id(verified)

        payload = {
            "miner_id": self._client_id,
            "job_id": submit_job_id,
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
            payload["expected_job_seq"] = int(self._job_seq)

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

    # ------------------------------------------------------------------
    # feeder loop
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # solo feeder helpers
    # ------------------------------------------------------------------

    def _difficulty_fields_from_source(self, src: JsonDict) -> dict[str, Any]:
        out: dict[str, Any] = {}

        diff_lo = _coerce_int(src.get("difficulty"), 0)
        diff_hi = _coerce_int(src.get("difficulty_top64"), 0)

        wide_raw = str(src.get("wide_difficulty") or "").strip()
        wide_hex = ""
        if wide_raw:
            if wide_raw.lower().startswith("0x"):
                wide_hex = wide_raw.lower()
            else:
                norm = _normalize_hex(wide_raw)
                if norm:
                    wide_hex = f"0x{norm}"

        if not wide_hex and (diff_lo > 0 or diff_hi > 0):
            full = (int(diff_hi) << 64) | (int(diff_lo) & 0xFFFFFFFFFFFFFFFF)
            if full > 0:
                wide_hex = hex(full)

        if diff_lo > 0:
            out["difficulty"] = int(diff_lo)
        if diff_hi > 0:
            out["difficulty_top64"] = int(diff_hi)
        if wide_hex:
            out["wide_difficulty"] = wide_hex

        return out

    def _solo_template_signature(self, result: JsonDict) -> str:
        hashing_blob = _normalize_hex(result.get("blockhashing_blob") or result.get("blob") or "")
        submit_blob = _normalize_hex(result.get("blocktemplate_blob") or result.get("submit_blob") or "")
        seed_hash = _normalize_hex(result.get("seed_hash") or "")
        next_seed_hash = _normalize_hex(result.get("next_seed_hash") or "")
        prev_hash = _normalize_hex(result.get("prev_hash") or result.get("prev_id") or result.get("top_hash") or "")

        height = _coerce_int(result.get("height"), 0)
        difficulty = _coerce_int(result.get("difficulty"), 0)
        difficulty_top64 = _coerce_int(result.get("difficulty_top64"), 0)
        reserved_offset = _coerce_int(result.get("reserved_offset"), 0)
        major_version = _coerce_int(result.get("major_version"), 0)
        minor_version = _coerce_int(result.get("minor_version"), 0)
        untrusted = _coerce_boolish(result.get("untrusted"), False)

        blob_material = submit_blob or hashing_blob
        blob_fp = hashlib.sha256(blob_material.encode("ascii")).hexdigest() if blob_material else ""

        wide_raw = str(result.get("wide_difficulty") or "").strip()
        wide_norm = wide_raw.lower() if wide_raw else ""

        return _stable_job_id(
            prev_hash,
            seed_hash,
            next_seed_hash,
            str(height),
            str(difficulty),
            str(difficulty_top64),
            wide_norm,
            str(reserved_offset),
            str(major_version),
            str(minor_version),
            str(untrusted),
            blob_fp,
            "solo_template",
        )

    def _should_repush_solo_template(self, template_sig: str) -> bool:
        now = time.time()
        if template_sig != self._solo_last_push_sig:
            return True
        return (now - self._solo_last_push_at) >= self._solo_force_repush_s

    def _mark_solo_template_pushed(
        self,
        *,
        template_sig: str,
        height: int,
        prev_hash: str,
    ) -> None:
        self._solo_last_template_sig = template_sig
        self._solo_last_template_height = int(height)
        self._solo_last_template_prev_hash = prev_hash
        self._solo_last_push_sig = template_sig
        self._solo_last_push_at = time.time()

    def _feed_once_solo(self) -> bool:
        wallet = (self.config.solo_wallet_address or "").strip()
        if not wallet:
            raise RuntimeError("solo_wallet_address is required for solo feeder")

        try:
            refresh_obj = self._broker_post(
                "monero_rpc/job/refresh",
                {
                    "wallet_address": wallet,
                    "reserve_size": int(self.config.solo_reserve_size),
                    "source": "monerod",
                },
            )
            if bool(refresh_obj.get("ok", False)):
                changed = bool(refresh_obj.get("changed", True))

                job_seq = _coerce_int(refresh_obj.get("job_seq"), 0)
                if job_seq > 0:
                    self._job_seq = job_seq

                feed_seq = _coerce_int(refresh_obj.get("feed_seq"), 0)
                if feed_seq > 0:
                    self._feed_seq = feed_seq

                job = refresh_obj.get("job")
                if isinstance(job, dict):
                    height = _coerce_int(job.get("height"), 0)
                    self.on_log(
                        f"[monerorpc-feeder] broker solo refresh ok changed={1 if changed else 0} "
                        f"height={height} job_id={job.get('job_id') or job.get('id') or '-'}"
                    )
                return changed
        except Exception:
            pass

        daemon_url = (self.config.solo_daemon_rpc_url or "").strip()
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

        hashing_blob_hex = _normalize_hex(result.get("blockhashing_blob") or result.get("blob") or "")
        submit_blob_hex = _normalize_hex(result.get("blocktemplate_blob") or result.get("submit_blob") or "")
        seed_hash_hex = _normalize_hex(result.get("seed_hash") or "")
        next_seed_hash_hex = _normalize_hex(result.get("next_seed_hash") or "")
        prev_hash = _normalize_hex(result.get("prev_hash") or result.get("prev_id") or result.get("top_hash") or "")

        height = _coerce_int(result.get("height"), 0)
        difficulty = _coerce_int(result.get("difficulty"), 0)
        reserved_offset = _coerce_int(result.get("reserved_offset"), 0)
        untrusted = _coerce_boolish(result.get("untrusted"), False)

        if not hashing_blob_hex:
            raise RuntimeError("monerod get_block_template missing blockhashing_blob")
        if not submit_blob_hex:
            raise RuntimeError("monerod get_block_template missing blocktemplate_blob")
        if not seed_hash_hex:
            raise RuntimeError("monerod get_block_template missing seed_hash")
        if difficulty <= 0 and not str(result.get("wide_difficulty") or "").strip():
            raise RuntimeError("monerod get_block_template missing difficulty/wide_difficulty")
        if untrusted:
            raise RuntimeError("monerod returned untrusted block template")

        difficulty_fields = self._difficulty_fields_from_source(result)
        target_hex = _target_hex_for_job(result)
        if not target_hex:
            raise RuntimeError("cannot derive normalized full target from solo template")

        template_sig = self._solo_template_signature(result)
        submit_blob_fp = hashlib.sha256(submit_blob_hex.encode("ascii")).hexdigest()
        job_id = _stable_job_id(
            template_sig,
            seed_hash_hex,
            next_seed_hash_hex,
            str(height),
            submit_blob_fp,
            "solo",
        )

        if not self._should_repush_solo_template(template_sig):
            return False

        payload: JsonDict = {
            "job_id": job_id,
            "blob": hashing_blob_hex,
            "blocktemplate_blob": submit_blob_hex,
            "seed_hash": seed_hash_hex,
            "target": target_hex,
            "nonce_offset": int(self.config.nonce_offset),
            "reserved_offset": reserved_offset,
            "height": height,
            "algo": "rx/0",
            "source": "monerod",
            "prev_hash": prev_hash,
            "untrusted": False,
            "template_sig": template_sig,
            "template_meta": {
                "prev_hash": prev_hash,
                "height": height,
                "reserved_offset": reserved_offset,
                "difficulty": difficulty,
            },
        }

        if next_seed_hash_hex:
            payload["next_seed_hash"] = next_seed_hash_hex

        payload.update(difficulty_fields)

        pushed = self._push_job_to_broker(payload)
        if pushed:
            self._mark_solo_template_pushed(
                template_sig=template_sig,
                height=height,
                prev_hash=prev_hash,
            )
            self.on_log(
                f"[monerorpc-feeder] solo template pushed "
                f"height={height} diff={difficulty or '-'} prev={prev_hash[:16] or '-'} "
                f"hash_blob={hashing_blob_hex[:16]} submit_blob={submit_blob_hex[:16]}"
            )

        return pushed

    # ------------------------------------------------------------------
    # blocknet feeder
    # ------------------------------------------------------------------

    def _normalize_blocknet_target_hex(self, src: JsonDict) -> str:
        direct = _normalize_hex(src.get("target_hex") or src.get("target") or "")

        # Best case: already a full 32-byte little-endian target.
        if direct and len(direct) >= 64 and (len(direct) % 2) == 0:
            return direct[:64]

        # Prefer reconstructing from full difficulty when possible.
        full_diff = _full_difficulty_from_source(src)
        if full_diff > 0:
            return _difficulty_to_target_hex(full_diff)

        # Last resort: accept a 32-byte direct target if present.
        if direct and (len(direct) % 2) == 0:
            try:
                raw = bytes.fromhex(direct)
                if len(raw) == 32:
                    return raw.hex()
            except Exception:
                pass

        return direct

    def _blocknet_template_sig(
            self,
            *,
            p2pool_job_id: str,
            blob_hex: str,
            blocktemplate_blob_hex: str,
            seed_hash_hex: str,
            target_hex: str,
            height: int,
            reserved_offset: int,
            nonce_offset: int,
            prev_hash: str,
            next_seed_hash_hex: str,
    ) -> str:
        return _stable_job_id(
            p2pool_job_id,
            blob_hex,
            blocktemplate_blob_hex,
            seed_hash_hex,
            target_hex,
            str(height),
            str(reserved_offset),
            str(nonce_offset),
            prev_hash,
            next_seed_hash_hex,
            "blocknet_template",
        )

    def _feed_once_blocknet(self) -> bool:
        job = self._blocknet_get_job()
        if not isinstance(job, dict):
            raise RuntimeError("blocknet feeder returned no job")

        blob_hex = _normalize_hex(
            job.get("blob_hex")
            or job.get("blob")
            or job.get("blockhashing_blob")
            or ""
        )
        blocktemplate_blob_hex = _normalize_hex(
            job.get("blocktemplate_blob")
            or job.get("submit_blob_hex")
            or job.get("submit_blob")
            or blob_hex
        )
        seed_hash_hex = _normalize_hex(job.get("seed_hash_hex") or job.get("seed_hash") or "")
        target_hex = self._normalize_blocknet_target_hex(job)
        height = _coerce_int(job.get("height"), 0)
        algo = str(job.get("algo") or "rx/0").strip() or "rx/0"
        reserved_offset = _coerce_int(job.get("reserved_offset"), 0)
        upstream_nonce_offset = _coerce_int(job.get("nonce_offset"), int(self.config.nonce_offset))
        prev_hash = _normalize_hex(job.get("prev_hash") or "")
        next_seed_hash_hex = _normalize_hex(
            job.get("next_seed_hash_hex") or job.get("next_seed_hash") or ""
        )
        untrusted = _coerce_boolish(job.get("untrusted"), False)

        # Use the true upstream P2Pool job id as the worker-visible job_id.
        p2pool_job_id = str(
            job.get("upstream_job_id")
            or job.get("p2pool_job_id")
            or job.get("job_id")
            or job.get("id")
            or ""
        ).strip()

        if not blob_hex:
            raise RuntimeError(f"blocknet job missing hashing blob: {job!r}")
        if not blocktemplate_blob_hex:
            raise RuntimeError(f"blocknet job missing blocktemplate blob: {job!r}")
        if not seed_hash_hex:
            raise RuntimeError(f"blocknet job missing seed hash: {job!r}")
        if not target_hex:
            raise RuntimeError(f"blocknet job missing/invalid full target: {job!r}")

        # Your worker uses a fixed config.nonce_offset everywhere.
        # Reject incompatible upstream jobs instead of silently mining bad shares.
        if upstream_nonce_offset != int(self.config.nonce_offset):
            raise RuntimeError(
                f"blocknet job nonce_offset={upstream_nonce_offset} is incompatible with "
                f"worker nonce_offset={int(self.config.nonce_offset)}"
            )

        # If upstream omitted job_id, derive one from the exact template.
        if not p2pool_job_id:
            p2pool_job_id = _stable_job_id(
                blob_hex,
                blocktemplate_blob_hex,
                seed_hash_hex,
                target_hex,
                str(height),
                prev_hash,
                next_seed_hash_hex,
                "blocknet_p2pool_job",
            )

        template_sig = self._blocknet_template_sig(
            p2pool_job_id=p2pool_job_id,
            blob_hex=blob_hex,
            blocktemplate_blob_hex=blocktemplate_blob_hex,
            seed_hash_hex=seed_hash_hex,
            target_hex=target_hex,
            height=height,
            reserved_offset=reserved_offset,
            nonce_offset=upstream_nonce_offset,
            prev_hash=prev_hash,
            next_seed_hash_hex=next_seed_hash_hex,
        )

        difficulty_fields: dict[str, Any] = self._difficulty_fields_from_source(job)
        if not difficulty_fields:
            difficulty_fields = _difficulty_fields_from_target_hex(target_hex)
        if not difficulty_fields:
            difficulty_fields = {
                "difficulty": 1,
                "difficulty_top64": 0,
                "wide_difficulty": "0x1",
            }

        payload: JsonDict = {
            # IMPORTANT:
            # Make broker/current-job job_id match the true upstream P2Pool job id.
            "job_id": p2pool_job_id,
            "upstream_job_id": p2pool_job_id,
            "p2pool_job_id": p2pool_job_id,

            # Optional debug identity for broker-side visibility only.
            "broker_job_id": _stable_job_id(
                self._blocknet_session,
                p2pool_job_id,
                template_sig,
                "blocknet_broker_route",
            ),

            "blob": blob_hex,
            "blob_hex": blob_hex,
            "blockhashing_blob": blob_hex,

            "blocktemplate_blob": blocktemplate_blob_hex,
            "submit_blob_hex": blocktemplate_blob_hex,
            "submit_blob": blocktemplate_blob_hex,

            "seed_hash": seed_hash_hex,
            "seed_hash_hex": seed_hash_hex,

            "target": target_hex,
            "target_hex": target_hex,

            "nonce_offset": upstream_nonce_offset,
            "reserved_offset": reserved_offset,
            "height": height,
            "algo": algo,
            "source": "blocknet",

            "session": self._blocknet_session,
            "session_id": self._blocknet_session,

            "untrusted": untrusted,
            "template_sig": template_sig,

            "upstream": {
                "kind": "blocknet",
                "session": self._blocknet_session,
                "job_id": p2pool_job_id,
            },
        }

        if prev_hash:
            payload["prev_hash"] = prev_hash
        if next_seed_hash_hex:
            payload["next_seed_hash"] = next_seed_hash_hex

        payload.update(difficulty_fields)
        self._remember_blocknet_route(
            payload,
            p2pool_job_id=p2pool_job_id,
            session=self._blocknet_session,
        )
        pushed = self._push_job_to_broker(payload)
        if pushed:
            self.on_log(
                f"[monerorpc-feeder] blocknet job pushed "
                f"job_id={p2pool_job_id} "
                f"session={self._blocknet_session or '-'} "
                f"height={height} "
                f"nonce_offset={upstream_nonce_offset} "
                f"reserved_offset={reserved_offset} "
                f"wide_diff={payload.get('wide_difficulty', '-')}"
            )
        return pushed

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

        job_seq = _coerce_int(obj.get("job_seq"), 0)
        if job_seq > 0:
            self._job_seq = job_seq

        feed_seq = _coerce_int(obj.get("feed_seq"), 0)
        if feed_seq > 0:
            self._feed_seq = feed_seq

        # IMPORTANT:
        # If broker returns a rewritten/current job payload, store aliases for that too,
        # so later GET /job/current can be mapped back to the upstream P2Pool job id.
        if str(payload.get("source") or "").strip().lower() in {"blocknet", "blocknet_p2pool"}:
            upstream_job_id = str(
                payload.get("p2pool_job_id")
                or payload.get("upstream_job_id")
                or payload.get("job_id")
                or ""
            ).strip()

            if upstream_job_id:
                self._remember_blocknet_route(
                    payload,
                    p2pool_job_id=upstream_job_id,
                    session=str(payload.get("session") or payload.get("session_id") or ""),
                )

                unwrapped = _unwrap_payload(obj)
                broker_job = unwrapped.get("job") if isinstance(unwrapped.get("job"), dict) else unwrapped
                if isinstance(broker_job, dict) and broker_job:
                    merged = dict(payload)
                    merged.update(broker_job)
                    self._remember_blocknet_route(
                        merged,
                        p2pool_job_id=upstream_job_id,
                        session=str(
                            merged.get("upstream_session")
                            or merged.get("session")
                            or merged.get("session_id")
                            or payload.get("session")
                            or payload.get("session_id")
                            or ""
                        ),
                    )

        self._feeder_last_push_at = now
        self._feeder_last_fp = fp

        if changed:
            self.on_log(
                f"[monerorpc-feeder] pushed job source={payload.get('source')} "
                f"job_id={payload.get('job_id')} height={payload.get('height')}"
            )
        return changed