from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional

from models import MinerConfig, MiningJob, SubmitResult, VerifiedShare

try:
    import zmq
except Exception:
    zmq = None


def _normalize_hex(text: str) -> str:
    return "".join(ch for ch in str(text or "").strip().lower() if not ch.isspace())


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        if isinstance(value, str):
            t = value.strip()
            if not t:
                return int(default)
            if t.lower().startswith("0x"):
                return int(t, 16)
            return int(t)
        return int(value)
    except Exception:
        return int(default)


def _short_hex(text: str, left: int = 12, right: int = 8) -> str:
    s = _normalize_hex(text)
    if not s:
        return "-"
    if len(s) <= left + right:
        return s
    return f"{s[:left]}...{s[-right:]}"


def _blob_len(hex_text: str) -> int:
    s = _normalize_hex(hex_text)
    if not s:
        return 0
    return len(s) // 2


def _atomic_to_xmr_text(amount_atomic: int) -> str:
    try:
        return f"{float(int(amount_atomic)) / 1e12:.12f}"
    except Exception:
        return "0.000000000000"


def _stable_fingerprint(*parts: Any, out_len: int = 16) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="replace"))
        h.update(b"|")
    return h.hexdigest()[:out_len]


def _is_hex(text: str) -> bool:
    s = _normalize_hex(text)
    if not s or (len(s) % 2) != 0:
        return False
    try:
        bytes.fromhex(s)
        return True
    except Exception:
        return False


def _rpc_error_text(error: Any) -> str:
    if isinstance(error, dict):
        code = error.get("code")
        msg = str(error.get("message") or error.get("error") or "").strip()
        if msg and code is not None:
            return f"{msg} (code={code})"
        if msg:
            return msg
        if code is not None:
            return f"code={code}"
    return str(error or "").strip()


def _full_difficulty_from_result(result: dict) -> int:
    wide_hex = str(result.get("wide_difficulty", "") or "").strip().lower()
    if wide_hex.startswith("0x"):
        wide_hex = wide_hex[2:]
    if wide_hex:
        try:
            return int(wide_hex, 16)
        except Exception:
            pass

    diff_lo = _coerce_int(result.get("difficulty"), 0)
    diff_hi = _coerce_int(result.get("difficulty_top64"), 0)
    full = (int(diff_hi) << 64) | (int(diff_lo) & 0xFFFFFFFFFFFFFFFF)
    return full if full > 0 else 0


def _difficulty_result_to_target_hex(result: dict) -> str:
    direct = _normalize_hex(result.get("target") or result.get("target_hex") or "")
    if direct and len(direct) >= 64:
        return direct[:64]

    difficulty = _full_difficulty_from_result(result)
    if difficulty <= 0:
        raise RuntimeError("get_block_template missing usable difficulty / wide_difficulty")

    target256 = ((1 << 256) - 1) // difficulty
    return target256.to_bytes(32, "little", signed=False).hex()


@dataclass
class SoloTemplateState:
    refresh_id: int = 0

    height: int = 0
    prev_hash: str = ""
    seed_hash: str = ""
    next_seed_hash: str = ""

    job_id: str = ""
    mining_fingerprint: str = ""
    template_fingerprint: str = ""

    submit_blob_hex: str = ""
    hashing_blob_hex: str = ""
    submit_blob_len: int = 0
    hashing_blob_len: int = 0

    target_hex: str = ""
    difficulty: int = 0
    reserved_offset: int = 0
    expected_reward_atomic: int = 0


class MoneroZmqReader:
    def __init__(
        self,
        url: str,
        on_log: Callable[[str], None],
        on_event: Callable[[str], None],
        topic_prefixes: Optional[list[str]] = None,
    ) -> None:
        self.url = (url or "").strip()
        self.on_log = on_log
        self.on_event = on_event

        self.topic_prefixes = list(
            topic_prefixes
            or [
                "json-minimal-chain_main",
                "json-full-miner_data",
                "json-minimal-txpool_add",
            ]
        )

        self._ctx = None
        self._sock = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._last_chain_height: Optional[int] = None
        self._last_chain_hash: str = ""

    def start(self) -> None:
        if not self.url:
            return
        if zmq is None:
            self.on_log("[solo-zmq] pyzmq is not installed; ZMQ notifications disabled")
            return

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, 1000)

        for topic in self.topic_prefixes:
            self._sock.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))

        self._sock.connect(self.url)

        self._thread = threading.Thread(target=self._run, name="MoneroZmqReader", daemon=True)
        self._thread.start()

        self.on_log(
            f"[solo-zmq] subscribed url={self.url} topics={','.join(self.topic_prefixes)}"
        )

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
                frames = self._sock.recv_multipart()
            except Exception:
                continue

            if not frames:
                continue

            try:
                topic = frames[0].decode("utf-8", errors="replace").strip()
            except Exception:
                topic = "<decode-error>"

            payload_bytes = frames[1] if len(frames) >= 2 else b""
            payload_obj = self._decode_payload(payload_bytes)
            summary = self._summarize_event(topic, payload_obj, payload_bytes)

            self.on_log(f"[solo-zmq] event topic={topic} {summary}".rstrip())
            try:
                self.on_event(f"zmq:{topic}")
            except Exception:
                pass

    @staticmethod
    def _decode_payload(payload_bytes: bytes) -> Any:
        if not payload_bytes:
            return None
        try:
            text = payload_bytes.decode("utf-8", errors="replace").strip()
        except Exception:
            return None
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return text

    def _summarize_event(self, topic: str, payload_obj: Any, payload_bytes: bytes) -> str:
        if isinstance(payload_obj, dict):
            if "chain_main" in topic:
                return self._summarize_chain_main(payload_obj)
            if "miner_data" in topic:
                return self._summarize_miner_data(payload_obj)
            if "txpool_add" in topic:
                return self._summarize_txpool_add(payload_obj)

            keys = ",".join(sorted(payload_obj.keys())[:8])
            return f"keys={keys} bytes={len(payload_bytes)}"

        if isinstance(payload_obj, list):
            return f"list_len={len(payload_obj)} bytes={len(payload_bytes)}"

        text = str(payload_obj or "").strip()
        if text:
            return f"bytes={len(payload_bytes)} text={text[:180].replace(chr(10), ' ')}"
        return f"bytes={len(payload_bytes)}"

    def _summarize_chain_main(self, payload: dict[str, Any]) -> str:
        height = _coerce_int(payload.get("height"), 0)
        block_hash = _normalize_hex(
            payload.get("hash")
            or payload.get("id")
            or payload.get("block_id")
            or payload.get("top_hash")
            or ""
        )
        prev_hash = _normalize_hex(payload.get("prev_id") or payload.get("prev_hash") or "")
        major_version = _coerce_int(payload.get("major_version"), 0)
        timestamp = _coerce_int(payload.get("timestamp"), 0)
        tx_count = _coerce_int(
            payload.get("num_txes")
            or payload.get("tx_count")
            or payload.get("n_txes"),
            0,
        )

        extras: list[str] = []
        if self._last_chain_height is not None:
            if height > 0 and height != self._last_chain_height + 1:
                extras.append(f"height_gap={self._last_chain_height}->{height}")
            if self._last_chain_hash and prev_hash and prev_hash != self._last_chain_hash:
                extras.append(
                    f"prev_mismatch={_short_hex(prev_hash)}!=last={_short_hex(self._last_chain_hash)}"
                )

        if height > 0:
            self._last_chain_height = height
        if block_hash:
            self._last_chain_hash = block_hash

        base = (
            f"height={height} hash={_short_hex(block_hash)} prev={_short_hex(prev_hash)} "
            f"txs={tx_count} major={major_version} ts={timestamp}"
        )
        if extras:
            base += " " + " ".join(extras)
        return base

    def _summarize_miner_data(self, payload: dict[str, Any]) -> str:
        height = _coerce_int(payload.get("height"), 0)
        prev_id = _normalize_hex(payload.get("prev_id") or payload.get("prev_hash") or "")
        seed_hash = _normalize_hex(payload.get("seed_hash") or "")
        diff = _coerce_int(
            payload.get("difficulty")
            or payload.get("wide_difficulty")
            or payload.get("difficulty_top64"),
            0,
        )

        backlog = payload.get("tx_backlog")
        backlog_count = len(backlog) if isinstance(backlog, list) else 0
        median_weight = _coerce_int(payload.get("median_weight"), 0)
        already_generated = _coerce_int(payload.get("already_generated_coins"), 0)

        return (
            f"height={height} prev={_short_hex(prev_id)} seed={_short_hex(seed_hash)} "
            f"diff={diff} backlog={backlog_count} median_weight={median_weight} "
            f"generated={already_generated}"
        )

    def _summarize_txpool_add(self, payload: dict[str, Any]) -> str:
        txs = payload.get("txs") or payload.get("transactions") or payload.get("tx_hashes") or []
        count = len(txs) if isinstance(txs, list) else 0

        first_hash = ""
        if isinstance(txs, list) and txs:
            first = txs[0]
            if isinstance(first, dict):
                first_hash = _normalize_hex(first.get("id_hash") or first.get("hash") or "")
            else:
                first_hash = _normalize_hex(str(first))

        return f"tx_count={count} first={_short_hex(first_hash)}"


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

        self.current_job: Optional[MiningJob] = None
        self.session_id: str = "solo"

        self._stop = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._zmq_reader: Optional[MoneroZmqReader] = None

        self._wake_reasons: queue.Queue[str] = queue.Queue(maxsize=256)

        self._template_lock = threading.Lock()
        self._expected_reward_by_job_id: dict[str, int] = {}
        self._submit_blob_by_job_id: dict[str, str] = {}

        self._last = SoloTemplateState()
        self._refresh_id = 0
        self._last_info_log_at = 0.0

    def connect(self) -> None:
        self.on_status("connecting")
        self._refresh_template(force_emit=True, reason="connect")

        if self.config.solo_use_zmq and self.config.solo_zmq_pub_url.strip():
            topic_prefixes = getattr(self.config, "solo_zmq_topics", None)
            self._zmq_reader = MoneroZmqReader(
                self.config.solo_zmq_pub_url.strip(),
                on_log=self.on_log,
                on_event=self._queue_refresh_reason,
                topic_prefixes=topic_prefixes,
            )
            self._zmq_reader.start()

        self._log_daemon_info(reason="connect", force=True)

        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="SoloPollThread",
            daemon=True,
        )
        self._poll_thread.start()

        if self.current_job is not None:
            self.on_status("mining")
        else:
            self.on_status("connected")

    def close(self) -> None:
        self._stop.set()
        self._queue_refresh_reason("close")

        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None

        if self._zmq_reader is not None:
            self._zmq_reader.close()
            self._zmq_reader = None

    def submit(self, verified: VerifiedShare) -> SubmitResult:
        solved_blob_hex, blob_source = self._resolve_solution_blob(verified)
        if not solved_blob_hex:
            return SubmitResult(
                accepted=False,
                status="missing_solution_blob",
                error="solo submission requires a valid solved block blob",
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
            f"[submit] solo-ready job={verified.job_id} nonce={verified.nonce_hex} "
            f"result={_short_hex(verified.result_hex)} blob_source={blob_source} "
            f"blob_len={_blob_len(solved_blob_hex)} "
            f"credited={verified.credited_work:.6f} actual={verified.actual_work:.6f} "
            f"quality={verified.quality:.6f}"
        )

        try:
            resp, rpc_ms = self._rpc_call("submit_block", [solved_blob_hex], allow_rpc_error=True)
        except Exception as exc:
            self.on_log(f"[submit] solo rpc error job={verified.job_id} error={exc}")
            return SubmitResult(
                accepted=False,
                status="rpc_error",
                error=str(exc),
                reject_class="backend_error",
                backend_error=True,
            )

        reward_atomic = 0
        with self._template_lock:
            reward_atomic = int(self._expected_reward_by_job_id.get(verified.job_id, 0) or 0)

        raw = {
            "rpc": resp,
            "rpc_ms": rpc_ms,
            "solo_expected_reward_atomic": reward_atomic,
            "solo_expected_reward_xmr": (float(reward_atomic) / 1e12) if reward_atomic > 0 else 0.0,
            "solo_wallet_address": (self.config.solo_wallet_address or "").strip(),
            "blob_source": blob_source,
        }

        rpc_error = resp.get("error")
        if rpc_error:
            err_text = _rpc_error_text(rpc_error)
            lower = err_text.lower()

            reject_class = "pool_rejected"
            stale = False
            invalid = False

            if current_job_id and verified.job_id and verified.job_id != current_job_id:
                reject_class = "pool_stale"
                stale = True
            if "invalid" in lower or "bad" in lower:
                reject_class = "pool_invalid"
                invalid = True

            self.on_log(
                f"[submit] solo block rejected nonce={verified.nonce_hex} job={verified.job_id} "
                f"error={err_text} rpc_ms={rpc_ms:.1f}"
            )
            return SubmitResult(
                accepted=False,
                status="ERROR",
                error=err_text,
                raw=raw,
                reject_class=reject_class,
                stale=stale,
                invalid=invalid,
            )

        result = resp.get("result")

        if isinstance(result, dict):
            status = str(result.get("status") or "OK").strip()
        elif isinstance(result, str):
            status = result.strip() or "OK"
        else:
            status = "OK"

        accepted = status.upper() in {"OK", "ACCEPTED"}

        if accepted:
            self.on_log(
                f"[submit] solo block accepted nonce={verified.nonce_hex} job={verified.job_id} "
                f"expected_reward_atomic={reward_atomic} ({_atomic_to_xmr_text(reward_atomic)} XMR) "
                f"rpc_ms={rpc_ms:.1f}"
            )
            self._log_post_submit_snapshots(verified.job_id)
            self._queue_refresh_reason("submit_block:accepted")
            return SubmitResult(
                accepted=True,
                status=status,
                error="",
                raw=raw,
                reject_class="accepted",
            )

        self.on_log(
            f"[submit] solo block rejected nonce={verified.nonce_hex} job={verified.job_id} "
            f"status={status or '-'} rpc_ms={rpc_ms:.1f}"
        )
        return SubmitResult(
            accepted=False,
            status=status or "ERROR",
            error=status or "Block not accepted",
            raw=raw,
            reject_class="pool_rejected",
        )

    def _resolve_solution_blob(self, verified: VerifiedShare) -> tuple[str, str]:
        nonce_hex = _normalize_hex(verified.nonce_hex)
        if not nonce_hex:
            return "", "missing_nonce"

        candidates: list[tuple[str, str]] = []

        carried = _normalize_hex(verified.solution_blob_hex or "")
        if carried:
            candidates.append((carried, "verified.solution_blob_hex"))

        with self._template_lock:
            archived = _normalize_hex(self._submit_blob_by_job_id.get(str(verified.job_id or ""), "") or "")
        if archived:
            candidates.append((archived, "job_archive"))

        current_submit = _normalize_hex(getattr(self.current_job, "submit_blob_hex", "") or "")
        current_job_id = str(getattr(self.current_job, "job_id", "") or "")
        if current_submit and current_job_id and current_job_id == str(verified.job_id or ""):
            candidates.append((current_submit, "current_job.submit_blob_hex"))

        last_submit = _normalize_hex(self._last.submit_blob_hex or "")
        if last_submit and self._last.job_id and self._last.job_id == str(verified.job_id or ""):
            candidates.append((last_submit, "_last.submit_blob_hex"))

        for blob_hex, source in candidates:
            rebuilt = self._rebuild_solution_blob(blob_hex, nonce_hex)
            if rebuilt:
                return rebuilt, source

        return "", "unavailable"

    def _rebuild_solution_blob(self, submit_blob_hex: str, nonce_hex: str) -> str:
        blob_hex = _normalize_hex(submit_blob_hex)
        nonce_hex = _normalize_hex(nonce_hex)

        if not blob_hex or not nonce_hex:
            return ""
        if not _is_hex(blob_hex):
            return ""
        if not _is_hex(nonce_hex):
            return ""

        try:
            raw = bytearray(bytes.fromhex(blob_hex))
            nonce_bytes = bytes.fromhex(nonce_hex)
        except Exception:
            return ""

        off = int(getattr(self.config, "nonce_offset", 39))
        if off < 0 or (off + 4) > len(raw):
            return ""
        if len(nonce_bytes) != 4:
            return ""

        raw[off:off + 4] = nonce_bytes
        return raw.hex()

    def _queue_refresh_reason(self, reason: str) -> None:
        if self._stop.is_set():
            return
        try:
            self._wake_reasons.put_nowait(str(reason or "event"))
        except queue.Full:
            pass

    def _poll_loop(self) -> None:
        fallback_s = max(0.25, float(self.config.solo_poll_fallback_s))

        while not self._stop.is_set():
            try:
                reason = self._wake_reasons.get(timeout=fallback_s)
            except queue.Empty:
                reason = f"timer:{fallback_s:.2f}s"

            if self._stop.is_set():
                break

            try:
                self._refresh_template(force_emit=False, reason=reason)
                self._log_daemon_info(reason=reason, force=False)
            except Exception as exc:
                self.on_log(f"[solo] template refresh failed reason={reason}: {exc}")

    def _refresh_template(self, force_emit: bool, reason: str) -> None:
        wallet = (self.config.solo_wallet_address or "").strip()
        if not wallet:
            raise RuntimeError("solo wallet address is required")

        params = {
            "wallet_address": wallet,
            "reserve_size": int(max(0, self.config.solo_reserve_size)),
        }

        resp, rpc_ms = self._rpc_call("get_block_template", params)
        result = resp.get("result") or {}

        if bool(result.get("untrusted", False)):
            self.on_log("[solo] ignoring untrusted block template")
            return

        height = int(result.get("height", 0))
        prev_hash = _normalize_hex(result.get("prev_hash", "") or result.get("prev_id", "") or "")
        seed_hash = _normalize_hex(result.get("seed_hash", "") or "")
        next_seed_hash = _normalize_hex(result.get("next_seed_hash", "") or "")

        submit_blob_hex = _normalize_hex(
            result.get("blocktemplate_blob", "") or result.get("submit_blob", "") or ""
        )
        hashing_blob_hex = _normalize_hex(
            result.get("blockhashing_blob", "") or result.get("blob", "") or submit_blob_hex
        )

        expected_reward_atomic = int(result.get("expected_reward", 0) or 0)
        reserved_offset = int(result.get("reserved_offset", 0) or 0)
        target_hex = _difficulty_result_to_target_hex(result)
        difficulty = _full_difficulty_from_result(result)

        if not submit_blob_hex:
            raise RuntimeError("get_block_template returned empty blocktemplate_blob")
        if not hashing_blob_hex:
            raise RuntimeError("get_block_template returned empty blockhashing_blob")
        if not seed_hash:
            raise RuntimeError("get_block_template returned empty seed_hash")

        submit_blob_len = _blob_len(submit_blob_hex)
        hashing_blob_len = _blob_len(hashing_blob_hex)

        template_fingerprint = _stable_fingerprint(submit_blob_hex)
        mining_fingerprint = _stable_fingerprint(
            height,
            prev_hash,
            seed_hash,
            target_hex,
            reserved_offset,
            hashing_blob_hex,
        )

        job_id = f"solo:{height}:{prev_hash[:16]}:{seed_hash[:16]}:{mining_fingerprint}"

        self._refresh_id += 1
        refresh_id = self._refresh_id

        new_state = SoloTemplateState(
            refresh_id=refresh_id,
            height=height,
            prev_hash=prev_hash,
            seed_hash=seed_hash,
            next_seed_hash=next_seed_hash,
            job_id=job_id,
            mining_fingerprint=mining_fingerprint,
            template_fingerprint=template_fingerprint,
            submit_blob_hex=submit_blob_hex,
            hashing_blob_hex=hashing_blob_hex,
            submit_blob_len=submit_blob_len,
            hashing_blob_len=hashing_blob_len,
            target_hex=target_hex,
            difficulty=difficulty,
            reserved_offset=reserved_offset,
            expected_reward_atomic=expected_reward_atomic,
        )

        changes = self._describe_template_changes(self._last, new_state)
        mining_changed = force_emit or (job_id != self._last.job_id)
        template_changed = bool(changes)

        if template_changed:
            self.on_log(
                f"[solo] template changed refresh_id={refresh_id} reason={reason} "
                f"changes={'; '.join(changes)} rpc_ms={rpc_ms:.1f}"
            )
        else:
            self.on_log(
                f"[solo] template unchanged refresh_id={refresh_id} reason={reason} rpc_ms={rpc_ms:.1f}"
            )

        if next_seed_hash:
            self.on_log(f"[solo] next_seed={_short_hex(next_seed_hash)}")

        with self._template_lock:
            self._expected_reward_by_job_id[job_id] = expected_reward_atomic
            self._submit_blob_by_job_id[job_id] = submit_blob_hex

            while len(self._expected_reward_by_job_id) > 128:
                oldest = next(iter(self._expected_reward_by_job_id))
                self._expected_reward_by_job_id.pop(oldest, None)

            while len(self._submit_blob_by_job_id) > 128:
                oldest = next(iter(self._submit_blob_by_job_id))
                self._submit_blob_by_job_id.pop(oldest, None)

        if mining_changed:
            job = MiningJob(
                job_id=job_id,
                blob_hex=hashing_blob_hex,
                submit_blob_hex=submit_blob_hex,
                target_hex=target_hex,
                session_id="solo",
                seed_hash_hex=seed_hash,
                height=height,
                algo="rx/0",
                reserved_offset=reserved_offset,
                backend="solo",
            )

            self.current_job = job
            self._last = new_state

            self.on_log(
                f"[solo] new mining job job_id={job_id} height={height} "
                f"prev={_short_hex(prev_hash)} seed={_short_hex(seed_hash)} tpl={template_fingerprint} "
                f"target={_short_hex(target_hex)} blob_len={hashing_blob_len} "
                f"submit_blob_len={submit_blob_len} reserved_offset={reserved_offset}"
            )

            self.on_job(job)
            self.on_status("mining")
            return

        self._last = new_state

        if self.current_job is not None:
            self.current_job.submit_blob_hex = submit_blob_hex
            self.current_job.reserved_offset = reserved_offset

        if template_changed:
            self.on_log(
                f"[solo] submit blob updated in place job_id={job_id} tpl={template_fingerprint} "
                f"submit_blob_len={submit_blob_len} reward_atomic={expected_reward_atomic} "
                f"({_atomic_to_xmr_text(expected_reward_atomic)} XMR)"
            )

    def _describe_template_changes(
        self,
        old: SoloTemplateState,
        new: SoloTemplateState,
    ) -> list[str]:
        if old.refresh_id == 0:
            return [
                f"job_id init->{new.job_id}",
                f"tpl init->{new.template_fingerprint}",
                f"reward_atomic init->{new.expected_reward_atomic} ({_atomic_to_xmr_text(new.expected_reward_atomic)} XMR)",
                f"submit_blob_len init->{new.submit_blob_len}",
                f"hashing_blob_len init->{new.hashing_blob_len}",
            ]

        changes: list[str] = []

        if old.job_id != new.job_id:
            changes.append(f"job_id {old.job_id}->{new.job_id}")
        if old.template_fingerprint != new.template_fingerprint:
            changes.append(f"tpl {old.template_fingerprint}->{new.template_fingerprint}")
        if old.height != new.height:
            changes.append(f"height {old.height}->{new.height}")
        if old.prev_hash != new.prev_hash:
            changes.append(f"prev {_short_hex(old.prev_hash)}->{_short_hex(new.prev_hash)}")
        if old.seed_hash != new.seed_hash:
            changes.append(f"seed {_short_hex(old.seed_hash)}->{_short_hex(new.seed_hash)}")
        if old.target_hex != new.target_hex:
            changes.append(f"target {_short_hex(old.target_hex)}->{_short_hex(new.target_hex)}")
        if old.difficulty != new.difficulty:
            changes.append(f"diff {old.difficulty}->{new.difficulty}")
        if old.reserved_offset != new.reserved_offset:
            changes.append(f"reserved_offset {old.reserved_offset}->{new.reserved_offset}")
        if old.expected_reward_atomic != new.expected_reward_atomic:
            changes.append(
                "reward_atomic "
                f"{old.expected_reward_atomic}->{new.expected_reward_atomic} "
                f"({_atomic_to_xmr_text(old.expected_reward_atomic)}->{_atomic_to_xmr_text(new.expected_reward_atomic)} XMR)"
            )
        if old.submit_blob_len != new.submit_blob_len:
            changes.append(f"submit_blob_len {old.submit_blob_len}->{new.submit_blob_len}")
        if old.hashing_blob_len != new.hashing_blob_len:
            changes.append(f"hashing_blob_len {old.hashing_blob_len}->{new.hashing_blob_len}")

        return changes

    def _log_post_submit_snapshots(self, submitted_job_id: str) -> None:
        try:
            header_resp, rpc_ms = self._rpc_call("get_last_block_header", {})
            hdr = (header_resp.get("result", {}) or {}).get("block_header", {}) or {}

            height = _coerce_int(hdr.get("height"), 0)
            block_hash = _normalize_hex(hdr.get("hash") or "")
            prev_hash = _normalize_hex(hdr.get("prev_hash") or "")
            reward = _coerce_int(hdr.get("reward"), 0)
            difficulty = _coerce_int(
                hdr.get("wide_difficulty") or hdr.get("difficulty") or hdr.get("difficulty_top64"),
                0,
            )
            nonce = _coerce_int(hdr.get("nonce"), 0)
            depth = _coerce_int(hdr.get("depth"), 0)
            num_txes = _coerce_int(hdr.get("num_txes"), 0)
            miner_tx_hash = _normalize_hex(hdr.get("miner_tx_hash") or "")

            self.on_log(
                f"[solo-post] last_block_header job={submitted_job_id} "
                f"height={height} hash={_short_hex(block_hash)} prev={_short_hex(prev_hash)} "
                f"reward_atomic={reward} ({_atomic_to_xmr_text(reward)} XMR) "
                f"diff={difficulty} nonce={nonce} depth={depth} txs={num_txes} "
                f"miner_tx={_short_hex(miner_tx_hash)} rpc_ms={rpc_ms:.1f}"
            )
        except Exception as exc:
            self.on_log(f"[solo-post] get_last_block_header failed job={submitted_job_id}: {exc}")

        try:
            info_resp, rpc_ms = self._rpc_call("get_info", {})
            info = info_resp.get("result", {}) or {}

            height = _coerce_int(info.get("height"), 0)
            target_height = _coerce_int(info.get("target_height"), 0)
            top_block_hash = _normalize_hex(info.get("top_block_hash") or "")
            tx_pool_size = _coerce_int(info.get("tx_pool_size"), 0)
            syncing = bool(info.get("busy_syncing", False))
            synchronized = bool(info.get("synchronized", False))
            incoming = _coerce_int(info.get("incoming_connections_count"), 0)
            outgoing = _coerce_int(info.get("outgoing_connections_count"), 0)

            self.on_log(
                f"[solo-post] daemon_info height={height} target_height={target_height} "
                f"top={_short_hex(top_block_hash)} tx_pool={tx_pool_size} "
                f"synchronized={int(synchronized)} busy_syncing={int(syncing)} "
                f"peers_in={incoming} peers_out={outgoing} rpc_ms={rpc_ms:.1f}"
            )
        except Exception as exc:
            self.on_log(f"[solo-post] get_info failed job={submitted_job_id}: {exc}")

    def _log_daemon_info(self, reason: str, force: bool) -> None:
        now = time.time()
        interval_s = float(getattr(self.config, "solo_info_log_interval_s", 30.0))
        if not force and (now - self._last_info_log_at) < max(5.0, interval_s):
            return

        try:
            resp, rpc_ms = self._rpc_call("get_info", {})
            info = resp.get("result", {}) or {}

            height = _coerce_int(info.get("height"), 0)
            target_height = _coerce_int(info.get("target_height"), 0)
            top_block_hash = _normalize_hex(info.get("top_block_hash") or "")
            tx_pool_size = _coerce_int(info.get("tx_pool_size"), 0)
            free_space = _coerce_int(info.get("free_space"), 0)
            synchronized = bool(info.get("synchronized", False))
            busy_syncing = bool(info.get("busy_syncing", False))
            incoming = _coerce_int(info.get("incoming_connections_count"), 0)
            outgoing = _coerce_int(info.get("outgoing_connections_count"), 0)

            self._last_info_log_at = now

            self.on_log(
                f"[solo-info] reason={reason} height={height} target_height={target_height} "
                f"top={_short_hex(top_block_hash)} tx_pool={tx_pool_size} free_space={free_space} "
                f"synchronized={int(synchronized)} busy_syncing={int(busy_syncing)} "
                f"peers_in={incoming} peers_out={outgoing} rpc_ms={rpc_ms:.1f}"
            )
        except Exception as exc:
            self.on_log(f"[solo-info] get_info failed reason={reason}: {exc}")

    def _rpc_call(
        self,
        method: str,
        params: Any = None,
        *,
        allow_rpc_error: bool = False,
    ) -> tuple[dict, float]:
        url = self._normalize_rpc_url(self.config.solo_daemon_rpc_url)

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": "0",
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        body = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            rpc_ms = (time.perf_counter() - t0) * 1000.0
            raw = exc.read().decode("utf-8", errors="replace")

            try:
                data = json.loads(raw)
                if allow_rpc_error and data.get("error"):
                    self.on_log(
                        f"[solo-rpc] rpc_error method={method} rpc_ms={rpc_ms:.1f} "
                        f"error={_rpc_error_text(data.get('error'))}"
                    )
                    return data, rpc_ms
            except Exception:
                pass

            self.on_log(f"[solo-rpc] error method={method} rpc_ms={rpc_ms:.1f} http={exc.code}")
            raise RuntimeError(f"{method} HTTP {exc.code}: {raw}") from exc
        except Exception as exc:
            rpc_ms = (time.perf_counter() - t0) * 1000.0
            self.on_log(f"[solo-rpc] error method={method} rpc_ms={rpc_ms:.1f} error={exc}")
            raise RuntimeError(f"{method} request failed: {exc}") from exc

        rpc_ms = (time.perf_counter() - t0) * 1000.0

        try:
            data = json.loads(raw)
        except Exception as exc:
            self.on_log(f"[solo-rpc] error method={method} rpc_ms={rpc_ms:.1f} bad_json={exc}")
            raise RuntimeError(f"{method} returned invalid JSON: {exc}") from exc

        if "error" in data and data["error"]:
            err_text = _rpc_error_text(data["error"])
            if allow_rpc_error:
                self.on_log(f"[solo-rpc] rpc_error method={method} rpc_ms={rpc_ms:.1f} error={err_text}")
                return data, rpc_ms

            self.on_log(f"[solo-rpc] rpc_error method={method} rpc_ms={rpc_ms:.1f} error={err_text}")
            raise RuntimeError(f"{method} RPC error: {data['error']}")

        self.on_log(f"[solo-rpc] ok method={method} rpc_ms={rpc_ms:.1f}")
        return data, rpc_ms

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