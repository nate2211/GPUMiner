from __future__ import annotations

import ctypes
import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from models import CandidateShare, MiningJob, VerifiedShare
from utils import nonce_to_hex_le, safe_bytes_from_hex

MAX_U64 = 0xFFFFFFFFFFFFFFFF
MAX_U256 = (1 << 256) - 1


def _normalize_hex(text: Optional[str]) -> str:
    if not text:
        return ""
    return "".join(ch for ch in text.strip().lower() if not ch.isspace())


def _nonce_array(nonces: list[int] | np.ndarray | None) -> np.ndarray:
    if nonces is None:
        return np.empty((0,), dtype=np.uint32)
    if isinstance(nonces, np.ndarray):
        return np.ascontiguousarray(nonces, dtype=np.uint32)
    return np.ascontiguousarray(list(nonces), dtype=np.uint32)


def _nonce_range_array(start_nonce: int, count: int) -> np.ndarray:
    count = max(0, int(count))
    if count <= 0:
        return np.empty((0,), dtype=np.uint32)
    base = np.uint64(int(start_nonce) & 0xFFFFFFFF)
    seq = (np.arange(count, dtype=np.uint64) + base) & np.uint64(0xFFFFFFFF)
    return np.ascontiguousarray(seq.astype(np.uint32))


def parse_target_hex_to_bytes(target_hex: str) -> bytes:
    s = _normalize_hex(target_hex)
    raw = safe_bytes_from_hex(s)
    if not raw:
        return b""
    if len(raw) >= 32:
        return raw[:32]
    return raw.ljust(32, b"\x00")


def target_hex_uses_full_256(target_hex: str) -> bool:
    raw = safe_bytes_from_hex(_normalize_hex(target_hex))
    return bool(raw) and len(raw) >= 32


def target_hex_to_int(target_hex: str) -> int:
    raw = parse_target_hex_to_bytes(target_hex)
    if not raw:
        return 0
    return int.from_bytes(raw, "little", signed=False)


def parse_target_hex_to_u64(target_hex: str) -> int:
    s = _normalize_hex(target_hex)
    raw = safe_bytes_from_hex(s)
    if not raw:
        return 0

    if len(raw) == 4:
        t32 = int.from_bytes(raw, "little", signed=False)
        if t32 == 0:
            return 0

        denom = 0xFFFFFFFF // t32
        if denom == 0:
            return MAX_U64
        return MAX_U64 // denom

    if len(raw) >= 8:
        return int.from_bytes(raw[:8], "little", signed=False)

    return int.from_bytes(raw.ljust(8, b"\x00"), "little", signed=False)


def target_hex_to_assigned_work(target_hex: str) -> float:
    raw = safe_bytes_from_hex(_normalize_hex(target_hex))
    if not raw:
        return 0.0

    if len(raw) >= 32:
        target_int = int.from_bytes(parse_target_hex_to_bytes(target_hex), "little", signed=False)
        if target_int <= 0:
            return 0.0
        return float(MAX_U256) / float(target_int)

    target64 = parse_target_hex_to_u64(target_hex)
    if target64 <= 0:
        return 0.0
    return float(MAX_U64) / float(target64)


def hash_bytes_to_actual_hash_int(hash32: bytes) -> int:
    if not hash32 or len(hash32) < 32:
        return 0
    return int.from_bytes(hash32[:32], "little", signed=False)


def hash_bytes_to_actual_tail_u64(hash32: bytes) -> int:
    if not hash32 or len(hash32) < 32:
        return 0
    return int.from_bytes(hash32[24:32], "little", signed=False)


def tail_u64_to_actual_work(tail_u64: int) -> float:
    v = int(tail_u64) & MAX_U64
    if v <= 0:
        return float(MAX_U64)
    return float(MAX_U64) / float(v)


def hash_bytes_to_actual_work(hash32: bytes, target_hex: str) -> float:
    raw = safe_bytes_from_hex(_normalize_hex(target_hex))
    if not raw:
        return 0.0

    if len(raw) >= 32:
        v = hash_bytes_to_actual_hash_int(hash32)
        if v <= 0:
            return float(MAX_U256)
        return float(MAX_U256) / float(v)

    return tail_u64_to_actual_work(hash_bytes_to_actual_tail_u64(hash32))


def hash_meets_target(hash32: bytes, target_hex: str) -> bool:
    raw = safe_bytes_from_hex(_normalize_hex(target_hex))
    if not raw or len(hash32) < 32:
        return False

    if len(raw) >= 32:
        target_int = int.from_bytes(parse_target_hex_to_bytes(target_hex), "little", signed=False)
        if target_int <= 0:
            return False
        hash_int = hash_bytes_to_actual_hash_int(hash32)
        return hash_int <= target_int

    target64 = parse_target_hex_to_u64(target_hex)
    if target64 <= 0:
        return False
    return hash_bytes_to_actual_tail_u64(hash32) <= target64


@dataclass(frozen=True)
class _PreparedSeed:
    seed: bytes
    seed_hex_norm: str
    fingerprint: bytes
    dataset_fingerprint: bytes


@dataclass(frozen=True)
class _PreparedJob:
    job_id: str
    blob: bytes
    target_hex: str
    target_b: bytes
    blob_hex_norm: str
    target_raw: bytes
    target_int: int
    target64: int
    assigned_work: float
    fingerprint: bytes
    seed_ctx: _PreparedSeed
    full_target: bool


@dataclass
class HashLabelResult:
    share: CandidateShare
    exact_hash_hex: str = ""
    exact_tail_u64: int = 0
    predictor_hash_match: bool = False
    verified: Optional[VerifiedShare] = None
    credited_work: float = 0.0
    accepted_by_tail: bool = False
    tail_only: bool = False


class _NativeHandle:
    __slots__ = (
        "_lib",
        "_handle",
        "_lock",
        "_current_seed_fingerprint",
        "_current_job_fingerprint",
        "_current_dataset_fingerprint",
    )

    def __init__(self, lib: ctypes.CDLL) -> None:
        self._lib = lib
        self._handle = lib.bnrx_create()
        if not self._handle:
            raise RuntimeError("bnrx_create returned null")
        self._lock = threading.RLock()
        self._current_seed_fingerprint: Optional[bytes] = None
        self._current_job_fingerprint: Optional[bytes] = None
        self._current_dataset_fingerprint: Optional[bytes] = None

    @property
    def current_job_fingerprint(self) -> Optional[bytes]:
        return self._current_job_fingerprint

    @property
    def current_dataset_fingerprint(self) -> Optional[bytes]:
        return self._current_dataset_fingerprint

    @property
    def has_prepare_seed(self) -> bool:
        return hasattr(self._lib, "bnrx_prepare_seed")

    @property
    def has_set_job(self) -> bool:
        return hasattr(self._lib, "bnrx_set_job")

    @property
    def has_warm_batch_vms(self) -> bool:
        return hasattr(self._lib, "bnrx_warm_batch_vms")

    @property
    def has_batch_verify(self) -> bool:
        return hasattr(self._lib, "bnrx_verify_nonce_batch")

    @property
    def has_hash_nonce(self) -> bool:
        return hasattr(self._lib, "bnrx_hash_nonce")

    @property
    def has_batch_hash(self) -> bool:
        return hasattr(self._lib, "bnrx_hash_nonce_batch")

    @property
    def has_batch_tail(self) -> bool:
        return hasattr(self._lib, "bnrx_tail_nonce_batch")

    def close(self) -> None:
        with self._lock:
            if self._handle:
                try:
                    self._lib.bnrx_destroy(self._handle)
                finally:
                    self._handle = None
            self._current_seed_fingerprint = None
            self._current_job_fingerprint = None
            self._current_dataset_fingerprint = None

    def _last_error_unlocked(self) -> str:
        if not self._handle or not hasattr(self._lib, "bnrx_last_error"):
            return ""
        try:
            raw = self._lib.bnrx_last_error(self._handle)
            if not raw:
                return ""
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="replace")
            return str(raw)
        except Exception:
            return ""

    def last_error(self) -> str:
        with self._lock:
            return self._last_error_unlocked()

    def prepare_seed(self, seed_ctx: _PreparedSeed) -> None:
        if self._current_seed_fingerprint == seed_ctx.fingerprint:
            return
        if not self.has_prepare_seed:
            raise RuntimeError("native verifier does not export bnrx_prepare_seed")

        seed_arr = _to_ubyte_array(seed_ctx.seed)

        with self._lock:
            if not self._handle:
                raise RuntimeError("native verifier handle is closed")

            if self._current_seed_fingerprint == seed_ctx.fingerprint:
                return

            rc = int(
                self._lib.bnrx_prepare_seed(
                    self._handle,
                    seed_arr,
                    ctypes.c_size_t(len(seed_ctx.seed)),
                )
            )
            if rc != 0:
                raise RuntimeError(
                    self._last_error_unlocked() or f"bnrx_prepare_seed failed with rc={rc}"
                )

            self._current_seed_fingerprint = seed_ctx.fingerprint
            self._current_dataset_fingerprint = seed_ctx.dataset_fingerprint
            self._current_job_fingerprint = None

    def set_job(self, prepared: _PreparedJob, nonce_offset: int) -> None:
        if (
            self._current_job_fingerprint == prepared.fingerprint
            and self._current_seed_fingerprint == prepared.seed_ctx.fingerprint
        ):
            return

        blob_arr = _to_ubyte_array(prepared.blob)

        with self._lock:
            if not self._handle:
                raise RuntimeError("native verifier handle is closed")

            if (
                self._current_job_fingerprint == prepared.fingerprint
                and self._current_seed_fingerprint == prepared.seed_ctx.fingerprint
            ):
                return

            if self.has_set_job:
                rc = int(
                    self._lib.bnrx_set_job(
                        self._handle,
                        blob_arr,
                        ctypes.c_size_t(len(prepared.blob)),
                        ctypes.c_uint32(int(nonce_offset)),
                        ctypes.c_char_p(prepared.target_b),
                    )
                )
            else:
                seed_arr = _to_ubyte_array(prepared.seed_ctx.seed)
                rc = int(
                    self._lib.bnrx_prepare_job(
                        self._handle,
                        blob_arr,
                        ctypes.c_size_t(len(prepared.blob)),
                        ctypes.c_uint32(int(nonce_offset)),
                        seed_arr,
                        ctypes.c_size_t(len(prepared.seed_ctx.seed)),
                        ctypes.c_char_p(prepared.target_b),
                    )
                )

            if rc != 0:
                raise RuntimeError(
                    self._last_error_unlocked() or f"bnrx_set_job failed with rc={rc}"
                )

            self._current_job_fingerprint = prepared.fingerprint
            self._current_seed_fingerprint = prepared.seed_ctx.fingerprint
            self._current_dataset_fingerprint = prepared.seed_ctx.dataset_fingerprint

    def warm_batch_vms(self, wanted: int) -> None:
        if not self.has_warm_batch_vms:
            return

        with self._lock:
            if not self._handle:
                raise RuntimeError("native verifier handle is closed")

            rc = int(
                self._lib.bnrx_warm_batch_vms(
                    self._handle,
                    ctypes.c_size_t(max(0, int(wanted))),
                )
            )
            if rc != 0:
                raise RuntimeError(
                    self._last_error_unlocked() or f"bnrx_warm_batch_vms failed with rc={rc}"
                )

    def prepare(self, prepared: _PreparedJob, nonce_offset: int) -> None:
        if self.has_prepare_seed and self.has_set_job:
            self.prepare_seed(prepared.seed_ctx)
            self.set_job(prepared, nonce_offset)
            return

        blob_arr = _to_ubyte_array(prepared.blob)
        seed_arr = _to_ubyte_array(prepared.seed_ctx.seed)

        with self._lock:
            if not self._handle:
                raise RuntimeError("native verifier handle is closed")

            rc = int(
                self._lib.bnrx_prepare_job(
                    self._handle,
                    blob_arr,
                    ctypes.c_size_t(len(prepared.blob)),
                    ctypes.c_uint32(int(nonce_offset)),
                    seed_arr,
                    ctypes.c_size_t(len(prepared.seed_ctx.seed)),
                    ctypes.c_char_p(prepared.target_b),
                )
            )
            if rc != 0:
                raise RuntimeError(
                    self._last_error_unlocked() or f"bnrx_prepare_job failed with rc={rc}"
                )

            self._current_seed_fingerprint = prepared.seed_ctx.fingerprint
            self._current_job_fingerprint = prepared.fingerprint
            self._current_dataset_fingerprint = prepared.seed_ctx.dataset_fingerprint

    def verify_nonce(self, nonce: int) -> tuple[int, bytes]:
        out_hash = (ctypes.c_ubyte * 32)()

        with self._lock:
            if not self._handle:
                raise RuntimeError("native verifier handle is closed")

            rc = int(
                self._lib.bnrx_verify_nonce(
                    self._handle,
                    ctypes.c_uint32(int(nonce) & 0xFFFFFFFF),
                    out_hash,
                )
            )
            return rc, bytes(out_hash)

    def verify_nonces_batch(
        self,
        nonces: list[int] | np.ndarray,
        *,
        max_threads: int = 0,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        if not self.has_batch_verify:
            raise RuntimeError("native verifier does not export bnrx_verify_nonce_batch")

        nonce_np = _nonce_array(nonces)
        count = int(nonce_np.size)
        out_accepts = np.zeros((count,), dtype=np.uint8)
        out_hashes = np.empty((count, 32), dtype=np.uint8)

        if count <= 0:
            return 0, out_accepts, out_hashes

        with self._lock:
            if not self._handle:
                raise RuntimeError("native verifier handle is closed")

            rc = int(
                self._lib.bnrx_verify_nonce_batch(
                    self._handle,
                    nonce_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                    ctypes.c_size_t(count),
                    out_hashes.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                    out_accepts.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                    ctypes.c_size_t(max(0, int(max_threads))),
                )
            )

        return rc, out_accepts, out_hashes

    def hash_nonce(self, nonce: int) -> bytes:
        if not self.has_hash_nonce:
            raise RuntimeError("native verifier does not export bnrx_hash_nonce")

        out_hash = (ctypes.c_ubyte * 32)()

        with self._lock:
            if not self._handle:
                raise RuntimeError("native verifier handle is closed")

            rc = int(
                self._lib.bnrx_hash_nonce(
                    self._handle,
                    ctypes.c_uint32(int(nonce) & 0xFFFFFFFF),
                    out_hash,
                )
            )
            if rc != 0:
                raise RuntimeError(
                    self._last_error_unlocked() or f"bnrx_hash_nonce failed with rc={rc}"
                )

        return bytes(out_hash)

    def hash_nonces_batch(
        self,
        nonces: list[int] | np.ndarray,
        *,
        max_threads: int = 0,
    ) -> tuple[int, np.ndarray]:
        if not self.has_batch_hash:
            raise RuntimeError("native verifier does not export bnrx_hash_nonce_batch")

        nonce_np = _nonce_array(nonces)
        count = int(nonce_np.size)
        out_hashes = np.empty((count, 32), dtype=np.uint8)

        if count <= 0:
            return 0, out_hashes

        with self._lock:
            if not self._handle:
                raise RuntimeError("native verifier handle is closed")

            rc = int(
                self._lib.bnrx_hash_nonce_batch(
                    self._handle,
                    nonce_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                    ctypes.c_size_t(count),
                    out_hashes.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                    ctypes.c_size_t(max(0, int(max_threads))),
                )
            )

        return rc, out_hashes

    def tail_nonces_batch(
        self,
        nonces: list[int] | np.ndarray,
        *,
        max_threads: int = 0,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        if not self.has_batch_tail:
            raise RuntimeError("native verifier does not export bnrx_tail_nonce_batch")

        nonce_np = _nonce_array(nonces)
        count = int(nonce_np.size)
        out_tails = np.empty((count,), dtype=np.uint64)
        out_accepts = np.zeros((count,), dtype=np.uint8)

        if count <= 0:
            return 0, out_accepts, out_tails

        with self._lock:
            if not self._handle:
                raise RuntimeError("native verifier handle is closed")

            rc = int(
                self._lib.bnrx_tail_nonce_batch(
                    self._handle,
                    nonce_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                    ctypes.c_size_t(count),
                    out_tails.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                    out_accepts.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                    ctypes.c_size_t(max(0, int(max_threads))),
                )
            )

        return rc, out_accepts, out_tails

    def dataset_words64(self) -> int:
        with self._lock:
            if not self._handle:
                raise RuntimeError("native verifier handle is closed")
            return int(self._lib.bnrx_dataset_words64(self._handle))

    def export_dataset64(self) -> np.ndarray:
        words = self.dataset_words64()
        if words <= 0:
            raise RuntimeError(self.last_error() or "bnrx_dataset_words64 returned 0")

        arr = np.empty(words, dtype=np.uint64)
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))

        with self._lock:
            if not self._handle:
                raise RuntimeError("native verifier handle is closed")

            rc = int(
                self._lib.bnrx_export_dataset64(
                    self._handle,
                    ptr,
                    ctypes.c_size_t(words),
                )
            )
            if rc != 0:
                raise RuntimeError(
                    self._last_error_unlocked() or f"bnrx_export_dataset64 failed with rc={rc}"
                )

        return arr


class CpuVerifier:
    _ENV_DLL_KEYS = (
        "BLOCKNET_RANDOMX_VERIFY_DLL",
        "RANDOMX_VERIFY_DLL",
        "BNRX_VERIFY_DLL",
    )

    _ENV_RUNTIME_KEYS = (
        "BLOCKNET_RANDOMX_RUNTIME_DLL",
        "RANDOMX_RUNTIME_DLL",
        "RANDOMX_DLL",
    )

    _DEFAULT_LIB_NAMES = (
        "MiningProject.dll",
        "blocknet_randomx_verify.dll",
        "randomx_verify.dll",
        "bnrx_verify.dll",
        "blocknet_randomx_verify.so",
        "randomx_verify.so",
        "libblocknet_randomx_verify.so",
        "librandomx_verify.so",
        "blocknet_randomx_verify.dylib",
        "librandomx_verify.dylib",
    )

    _DEFAULT_RUNTIME_NAMES = (
        "randomx-dll.dll",
        "randomx.dll",
        "librandomx.so",
        "librandomx.dylib",
    )

    def __init__(
        self,
        dll_path: Optional[str] = None,
        *,
        randomx_runtime_dll_path: Optional[str] = None,
        preload_randomx_runtime: bool = True,
        nonce_offset: int = 39,
        on_log: Optional[Callable[[str], None]] = None,
        strict: bool = False,
    ) -> None:
        self.nonce_offset = int(nonce_offset)
        self.on_log = on_log
        self.strict = bool(strict)

        self._state_lock = threading.RLock()
        self._handles_lock = threading.Lock()
        self._tls = threading.local()

        self._lib: Optional[ctypes.CDLL] = None
        self._export_handle: Optional[_NativeHandle] = None
        self._thread_handles: dict[int, _NativeHandle] = {}
        self._disabled_reason: Optional[str] = None
        self._current_prepared_job: Optional[_PreparedJob] = None
        self._warm_batch_vms_target: int = 0

        self._randomx_runtime_lib: Optional[ctypes.CDLL] = None
        self._randomx_runtime_path: Optional[str] = None
        self._dll_directory_handles: list[object] = []

        try:
            if preload_randomx_runtime:
                self._randomx_runtime_lib = self._maybe_preload_randomx_runtime(randomx_runtime_dll_path)

            resolved = self._resolve_library_path(dll_path)
            if not resolved:
                raise FileNotFoundError(
                    "No RandomX verifier library found. "
                    "Set BLOCKNET_RANDOMX_VERIFY_DLL or place a verifier DLL/SO next to the app."
                )

            self._lib = self._load_library(resolved)
            self._export_handle = _NativeHandle(self._lib)

            self._log(f"[verify] native verifier loaded: {resolved}")
            if self._randomx_runtime_path:
                self._log(f"[verify] preloaded RandomX runtime: {self._randomx_runtime_path}")

            self._log(
                "[verify] mode=per-thread handles + dedicated export handle "
                "(native side is expected to share dataset/cache by seed)"
            )
            self._log(f"[verify] prepare_seed_export={'yes' if self.has_prepare_seed else 'no'}")
            self._log(f"[verify] set_job_export={'yes' if self.has_set_job else 'no'}")
            self._log(f"[verify] warm_batch_vms_export={'yes' if self.has_warm_batch_vms else 'no'}")
            self._log(f"[verify] batch_verify_export={'yes' if self.has_batch_verify else 'no'}")
            self._log(f"[verify] batch_hash_export={'yes' if self.has_batch_hash else 'no'}")
            self._log(f"[verify] batch_tail_export={'yes' if self.has_batch_tail else 'no'}")
        except Exception as exc:
            self._disabled_reason = str(exc)
            self._log(f"[verify] disabled: {self._disabled_reason}")
            if self.strict:
                raise

    @property
    def is_ready(self) -> bool:
        return self._lib is not None and self._export_handle is not None and not self._disabled_reason

    @property
    def disabled_reason(self) -> str:
        return self._disabled_reason or ""

    @property
    def has_dataset_exports(self) -> bool:
        lib = self._lib
        return (
            lib is not None
            and hasattr(lib, "bnrx_dataset_words64")
            and hasattr(lib, "bnrx_export_dataset64")
        )

    @property
    def has_prepare_seed(self) -> bool:
        lib = self._lib
        return lib is not None and hasattr(lib, "bnrx_prepare_seed")

    @property
    def has_set_job(self) -> bool:
        lib = self._lib
        return lib is not None and hasattr(lib, "bnrx_set_job")

    @property
    def has_warm_batch_vms(self) -> bool:
        lib = self._lib
        return lib is not None and hasattr(lib, "bnrx_warm_batch_vms")

    @property
    def has_batch_verify(self) -> bool:
        lib = self._lib
        return lib is not None and hasattr(lib, "bnrx_verify_nonce_batch")

    @property
    def has_hash_nonce(self) -> bool:
        lib = self._lib
        return lib is not None and hasattr(lib, "bnrx_hash_nonce")

    @property
    def has_batch_hash(self) -> bool:
        lib = self._lib
        return lib is not None and hasattr(lib, "bnrx_hash_nonce_batch")

    @property
    def has_batch_tail(self) -> bool:
        lib = self._lib
        return lib is not None and hasattr(lib, "bnrx_tail_nonce_batch")

    @property
    def current_dataset_fingerprint(self) -> Optional[bytes]:
        with self._state_lock:
            prepared = self._current_prepared_job
            return prepared.seed_ctx.dataset_fingerprint if prepared is not None else None

    def close(self) -> None:
        with self._handles_lock:
            export_handle = self._export_handle
            self._export_handle = None

            handles = list(self._thread_handles.values())
            self._thread_handles.clear()

        if export_handle is not None:
            try:
                export_handle.close()
            except Exception:
                pass

        for handle in handles:
            try:
                handle.close()
            except Exception:
                pass

        with self._state_lock:
            self._current_prepared_job = None
            self._lib = None
            self._randomx_runtime_lib = None
            self._randomx_runtime_path = None
            self._dll_directory_handles.clear()
            self._warm_batch_vms_target = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def prepare_seed_for_job(self, job: MiningJob) -> None:
        prepared = self._build_prepared_from_job(job)

        with self._state_lock:
            if not self.is_ready:
                raise RuntimeError(self.disabled_reason or "native verifier is not available")
            current = self._current_prepared_job
            if current is not None and current.seed_ctx.fingerprint == prepared.seed_ctx.fingerprint:
                return

        export_handle = self._require_export_handle()
        if self.has_prepare_seed:
            export_handle.prepare_seed(prepared.seed_ctx)
        else:
            export_handle.prepare(prepared, self.nonce_offset)

        with self._state_lock:
            self._current_prepared_job = prepared

    def set_job(self, job: MiningJob) -> None:
        prepared = self._build_prepared_from_job(job)

        with self._state_lock:
            if not self.is_ready:
                raise RuntimeError(self.disabled_reason or "native verifier is not available")
            current = self._current_prepared_job
            if (
                current is not None
                and current.fingerprint == prepared.fingerprint
                and current.seed_ctx.fingerprint == prepared.seed_ctx.fingerprint
            ):
                return

        export_handle = self._require_export_handle()
        if self.has_prepare_seed and self.has_set_job:
            export_handle.prepare_seed(prepared.seed_ctx)
            export_handle.set_job(prepared, self.nonce_offset)
        else:
            export_handle.prepare(prepared, self.nonce_offset)

        with self._state_lock:
            self._current_prepared_job = prepared

    def warm_batch_vms(self, wanted: int) -> None:
        wanted = max(0, int(wanted))

        with self._state_lock:
            self._warm_batch_vms_target = wanted
            prepared = self._current_prepared_job

        if wanted <= 0 or prepared is None or not self.has_warm_batch_vms:
            return

        handles: list[_NativeHandle] = []
        export_handle = self._require_export_handle()

        with self._handles_lock:
            handles.append(export_handle)
            handles.extend(self._thread_handles.values())

        for handle in handles:
            try:
                if self.has_prepare_seed:
                    handle.prepare_seed(prepared.seed_ctx)
                else:
                    handle.prepare(prepared, self.nonce_offset)
                handle.warm_batch_vms(wanted)
            except Exception as exc:
                self._log(f"[verify] warm batch vms skipped for handle: {exc}")

    def prepare_job(self, job: MiningJob) -> None:
        prepared = self._build_prepared_from_job(job)

        with self._state_lock:
            if not self.is_ready:
                raise RuntimeError(self.disabled_reason or "native verifier is not available")
            current = self._current_prepared_job
            if (
                current is not None
                and current.fingerprint == prepared.fingerprint
                and current.seed_ctx.fingerprint == prepared.seed_ctx.fingerprint
            ):
                return

        export_handle = self._require_export_handle()
        export_handle.prepare(prepared, self.nonce_offset)

        with self._state_lock:
            self._current_prepared_job = prepared

        self._log(
            f"[verify] prepared job job_id={job.job_id} "
            f"height={job.height} algo={job.algo} "
            f"full_target={1 if prepared.full_target else 0} "
            f"target64={prepared.target64} assigned_work={prepared.assigned_work:.6f}"
        )

    def export_dataset_u64(self) -> np.ndarray:
        with self._state_lock:
            if not self.is_ready:
                raise RuntimeError(self.disabled_reason or "native verifier is not available")
            prepared = self._current_prepared_job

        if prepared is None:
            raise RuntimeError("no job is prepared")

        if not self.has_dataset_exports:
            raise RuntimeError(
                "native verifier is missing dataset exports: "
                "bnrx_dataset_words64 / bnrx_export_dataset64"
            )

        export_handle = self._require_export_handle()
        if self.has_prepare_seed:
            export_handle.prepare_seed(prepared.seed_ctx)
        else:
            export_handle.prepare(prepared, self.nonce_offset)

        arr = export_handle.export_dataset64()
        mib = arr.nbytes / (1024.0 * 1024.0)
        self._log(f"[verify] exported RandomX dataset: words={arr.size} size={mib:.2f} MiB")
        return arr

    def verify(self, share: CandidateShare) -> Optional[VerifiedShare]:
        verified, _credited_work = self.verify_with_work(share)
        return verified

    def verify_with_work(self, share: CandidateShare) -> tuple[Optional[VerifiedShare], float]:
        results = self.label_shares_batch_with_hashes([share], max_threads=0)
        if not results:
            return None, 0.0
        return results[0].verified, results[0].credited_work

    def verify_batch_with_work(
        self,
        shares: list[CandidateShare],
        *,
        max_threads: int = 0,
    ) -> list[tuple[Optional[VerifiedShare], float]]:
        labeled = self.label_shares_batch_with_hashes(shares, max_threads=max_threads)
        return [(item.verified, item.credited_work) for item in labeled]

    def rescue_scan_window(
        self,
        job: MiningJob,
        start_nonce: int,
        count: int,
        *,
        batch_size: int = 1024,
        max_threads: int = 0,
    ) -> list[tuple[CandidateShare, VerifiedShare]]:
        count = max(0, int(count))
        batch_size = max(1, int(batch_size))
        if count <= 0 or not self.is_ready:
            return []

        prepared = self._build_prepared_from_job(job)
        handle = self._get_thread_handle()
        handle.prepare(prepared, self.nonce_offset)

        hits: list[tuple[CandidateShare, VerifiedShare]] = []
        scanned = 0

        while scanned < count:
            this_batch = min(batch_size, count - scanned)
            nonce_np = _nonce_range_array((int(start_nonce) + scanned) & 0xFFFFFFFF, this_batch)
            hashes_np: Optional[np.ndarray] = None

            if this_batch > 1 and self.has_batch_hash:
                try:
                    rc, hashes_np = handle.hash_nonces_batch(nonce_np, max_threads=max_threads)
                    if rc != 0:
                        raise RuntimeError(
                            handle.last_error() or f"bnrx_hash_nonce_batch failed with rc={rc}"
                        )
                except Exception as exc:
                    self._log(f"[verify] rescue hash batch fallback size={this_batch} reason={exc}")
                    hashes_np = None

            if hashes_np is None and this_batch > 1 and self.has_batch_verify:
                try:
                    rc, _accepts_np, hashes_np = handle.verify_nonces_batch(
                        nonce_np,
                        max_threads=max_threads,
                    )
                    if rc != 0:
                        raise RuntimeError(
                            handle.last_error() or f"bnrx_verify_nonce_batch failed with rc={rc}"
                        )
                except Exception as exc:
                    self._log(f"[verify] rescue verify batch fallback size={this_batch} reason={exc}")
                    hashes_np = None

            if hashes_np is not None:
                for idx, nonce_u32 in enumerate(nonce_np):
                    cand = CandidateShare(
                        nonce=int(nonce_u32),
                        gpu_hash_hex="",
                        job_id=job.job_id,
                        blob_hex=job.blob_hex,
                        session_id=job.session_id,
                        target_hex=job.target_hex,
                        seed_hash_hex=job.seed_hash_hex,
                        source="cpu_rescue",
                    )
                    item = self._label_share_from_hash(prepared, cand, hashes_np[idx].tobytes())
                    if item.verified is not None:
                        hits.append((cand, item.verified))
            else:
                for nonce_u32 in nonce_np:
                    nonce_i = int(nonce_u32)
                    try:
                        if self.has_hash_nonce:
                            out_hash = handle.hash_nonce(nonce_i)
                        else:
                            rc, out_hash = handle.verify_nonce(nonce_i)
                            if rc < 0:
                                raise RuntimeError(
                                    handle.last_error() or f"bnrx_verify_nonce failed with rc={rc}"
                                )
                    except Exception as exc:
                        self._log(f"[verify] rescue single nonce failed nonce={nonce_i:08x}: {exc}")
                        continue

                    cand = CandidateShare(
                        nonce=nonce_i,
                        gpu_hash_hex="",
                        job_id=job.job_id,
                        blob_hex=job.blob_hex,
                        session_id=job.session_id,
                        target_hex=job.target_hex,
                        seed_hash_hex=job.seed_hash_hex,
                        source="cpu_rescue",
                    )
                    item = self._label_share_from_hash(prepared, cand, out_hash)
                    if item.verified is not None:
                        hits.append((cand, item.verified))

            scanned += this_batch

        hits.sort(key=lambda row: (-float(row[1].credited_work), int(row[0].nonce)))
        return hits

    def screen_shares_batch_by_tail(
        self,
        shares: list[CandidateShare],
        *,
        max_threads: int = 0,
    ) -> list[HashLabelResult]:
        shares = list(shares or [])
        if not shares:
            return []

        results: list[HashLabelResult] = [HashLabelResult(share=s) for s in shares]

        if not self.is_ready:
            return results

        if not self.has_batch_tail:
            return self.label_shares_batch_with_hashes(shares, max_threads=max_threads)

        grouped: dict[tuple[bytes, bytes], tuple[_PreparedJob, list[tuple[int, CandidateShare]]]] = {}
        for idx, share in enumerate(shares):
            try:
                prepared = self._prepared_for_share(share)
            except Exception as exc:
                self._log(
                    f"[verify] skipped candidate during tail preparation nonce="
                    f"{int(getattr(share, 'nonce', 0)) & 0xFFFFFFFF:08x}: {exc}"
                )
                continue

            key = (prepared.seed_ctx.fingerprint, prepared.fingerprint)
            entry = grouped.get(key)
            if entry is None:
                grouped[key] = (prepared, [(idx, share)])
            else:
                entry[1].append((idx, share))

        handle = self._get_thread_handle()

        for prepared, indexed_shares in grouped.values():
            if prepared.full_target:
                fallback = self.label_shares_batch_with_hashes(
                    [share for _, share in indexed_shares],
                    max_threads=max_threads,
                )
                for (result_idx, _), item in zip(indexed_shares, fallback):
                    results[result_idx] = item
                continue

            try:
                handle.prepare(prepared, self.nonce_offset)
            except Exception as exc:
                self._log(f"[verify] tail batch prepare failed: {exc}")
                continue

            nonces = [int(share.nonce) & 0xFFFFFFFF for _, share in indexed_shares]

            try:
                rc, accepts_np, tails_np = handle.tail_nonces_batch(nonces, max_threads=max_threads)
                if rc != 0:
                    raise RuntimeError(
                        handle.last_error() or f"bnrx_tail_nonce_batch failed with rc={rc}"
                    )
            except Exception as exc:
                self._log(f"[verify] tail batch fallback size={len(indexed_shares)} reason={exc}")
                fallback = self.label_shares_batch_with_hashes(
                    [share for _, share in indexed_shares],
                    max_threads=max_threads,
                )
                for (result_idx, _), item in zip(indexed_shares, fallback):
                    results[result_idx] = item
                continue

            for batch_idx, (result_idx, share) in enumerate(indexed_shares):
                tail_u64 = int(tails_np[batch_idx])
                accepted = bool(int(accepts_np[batch_idx]) != 0)

                try:
                    share.exact_tail_u64 = tail_u64
                except Exception:
                    pass

                credited_work = 0.0
                if accepted and prepared.target64 > 0:
                    assigned_work = prepared.assigned_work
                    actual_work = tail_u64_to_actual_work(tail_u64)
                    credited_work = max(assigned_work, actual_work)

                results[result_idx] = HashLabelResult(
                    share=share,
                    exact_tail_u64=tail_u64,
                    credited_work=credited_work,
                    accepted_by_tail=accepted,
                    tail_only=True,
                )

        return results

    def hash_shares_batch(
        self,
        shares: list[CandidateShare],
        *,
        max_threads: int = 0,
    ) -> list[bytes]:
        labeled = self.label_shares_batch_with_hashes(shares, max_threads=max_threads)
        out: list[bytes] = []
        for item in labeled:
            if item.exact_hash_hex:
                try:
                    out.append(bytes.fromhex(item.exact_hash_hex))
                except Exception:
                    out.append(b"")
            else:
                out.append(b"")
        return out

    def label_shares_batch_with_hashes(
        self,
        shares: list[CandidateShare],
        *,
        max_threads: int = 0,
    ) -> list[HashLabelResult]:
        shares = list(shares or [])
        if not shares:
            return []

        results: list[HashLabelResult] = [HashLabelResult(share=s) for s in shares]

        if not self.is_ready:
            return results

        grouped: dict[tuple[bytes, bytes], tuple[_PreparedJob, list[tuple[int, CandidateShare]]]] = {}
        for idx, share in enumerate(shares):
            try:
                prepared = self._prepared_for_share(share)
            except Exception as exc:
                self._log(
                    f"[verify] skipped candidate during batch preparation nonce="
                    f"{int(getattr(share, 'nonce', 0)) & 0xFFFFFFFF:08x}: {exc}"
                )
                continue

            key = (prepared.seed_ctx.fingerprint, prepared.fingerprint)
            entry = grouped.get(key)
            if entry is None:
                grouped[key] = (prepared, [(idx, share)])
            else:
                entry[1].append((idx, share))

        handle = self._get_thread_handle()

        for prepared, indexed_shares in grouped.values():
            try:
                handle.prepare(prepared, self.nonce_offset)
            except Exception as exc:
                self._log(f"[verify] batch prepare failed: {exc}")
                continue

            nonces = [int(share.nonce) & 0xFFFFFFFF for _, share in indexed_shares]
            hashes_np: Optional[np.ndarray] = None

            if len(indexed_shares) > 1 and self.has_batch_hash:
                try:
                    rc, hashes_np = handle.hash_nonces_batch(nonces, max_threads=max_threads)
                    if rc != 0:
                        raise RuntimeError(
                            handle.last_error() or f"bnrx_hash_nonce_batch failed with rc={rc}"
                        )
                except Exception as exc:
                    self._log(f"[verify] hash batch fallback size={len(indexed_shares)} reason={exc}")
                    hashes_np = None

            if hashes_np is None and len(indexed_shares) > 1 and self.has_batch_verify:
                try:
                    rc, _accepts_np, hashes_np = handle.verify_nonces_batch(
                        nonces,
                        max_threads=max_threads,
                    )
                    if rc != 0:
                        raise RuntimeError(
                            handle.last_error() or f"bnrx_verify_nonce_batch failed with rc={rc}"
                        )
                except Exception as exc:
                    self._log(f"[verify] verify batch fallback size={len(indexed_shares)} reason={exc}")
                    hashes_np = None

            if hashes_np is not None:
                for batch_idx, (result_idx, share) in enumerate(indexed_shares):
                    out_hash = hashes_np[batch_idx].tobytes()
                    results[result_idx] = self._label_share_from_hash(prepared, share, out_hash)
                continue

            for result_idx, share in indexed_shares:
                try:
                    if self.has_hash_nonce:
                        out_hash = handle.hash_nonce(int(share.nonce) & 0xFFFFFFFF)
                    else:
                        rc, out_hash = handle.verify_nonce(int(share.nonce) & 0xFFFFFFFF)
                        if rc < 0:
                            raise RuntimeError(
                                handle.last_error() or f"bnrx_verify_nonce failed with rc={rc}"
                            )
                    results[result_idx] = self._label_share_from_hash(prepared, share, out_hash)
                except Exception as exc:
                    self._log(
                        f"[verify] exact label failed nonce="
                        f"{int(getattr(share, 'nonce', 0)) & 0xFFFFFFFF:08x}: {exc}"
                    )

        return results

    def estimate_job_work(self, job: MiningJob) -> float:
        prepared = self._build_prepared_from_job(job)
        return prepared.assigned_work

    def estimate_share_work(self, share: CandidateShare) -> float:
        prepared = self._build_prepared_from_share(share)
        return prepared.assigned_work

    def _label_share_from_hash(
        self,
        prepared: _PreparedJob,
        share: CandidateShare,
        out_hash: bytes,
    ) -> HashLabelResult:
        exact_hash_hex = out_hash.hex() if out_hash else ""
        exact_tail_u64 = hash_bytes_to_actual_tail_u64(out_hash)
        predictor_hash_match = (
            bool(exact_hash_hex)
            and _normalize_hex(getattr(share, "gpu_hash_hex", "") or "") == exact_hash_hex
        )

        try:
            share.exact_hash_hex = exact_hash_hex
            share.exact_tail_u64 = exact_tail_u64
            share.predictor_hash_match = predictor_hash_match
        except Exception:
            pass

        if not hash_meets_target(out_hash, prepared.target_hex):
            return HashLabelResult(
                share=share,
                exact_hash_hex=exact_hash_hex,
                exact_tail_u64=exact_tail_u64,
                predictor_hash_match=predictor_hash_match,
                verified=None,
                credited_work=0.0,
            )

        assigned_work = prepared.assigned_work
        actual_work = hash_bytes_to_actual_work(out_hash, prepared.target_hex)
        credited_work = max(assigned_work, actual_work)
        quality = (actual_work / assigned_work) if assigned_work > 0.0 else 0.0

        verified = VerifiedShare(
            nonce_hex=nonce_to_hex_le(share.nonce),
            result_hex=exact_hash_hex,
            job_id=share.job_id,
            session_id=share.session_id,
            assigned_work=assigned_work,
            actual_work=actual_work,
            credited_work=credited_work,
            quality=quality,
            actual_tail_u64=exact_tail_u64,
            predicted_tail_u64=int(getattr(share, "predicted_tail_u64", 0)),
            rank_score_u64=int(getattr(share, "rank_score_u64", 0)),
            tune_bucket=int(getattr(share, "tune_bucket", -1)),
            tune_tail_bin=int(getattr(share, "tune_tail_bin", -1)),
            rank_quality=int(getattr(share, "rank_quality", 128)),
            threshold_quality=int(getattr(share, "threshold_quality", 128)),
            gpu_hash_hex=(getattr(share, "gpu_hash_hex", "") or ""),
            predictor_hash_match=predictor_hash_match,
        )

        return HashLabelResult(
            share=share,
            exact_hash_hex=exact_hash_hex,
            exact_tail_u64=exact_tail_u64,
            predictor_hash_match=predictor_hash_match,
            verified=verified,
            credited_work=credited_work,
            accepted_by_tail=True,
        )

    def _prepared_for_share(self, share: CandidateShare) -> _PreparedJob:
        with self._state_lock:
            current = self._current_prepared_job

        if current is not None and self._share_matches_prepared(current, share):
            return current

        return self._build_prepared_from_share(share)

    def _share_matches_prepared(self, prepared: _PreparedJob, share: CandidateShare) -> bool:
        blob_hex_norm = _normalize_hex(share.blob_hex)
        target_hex_norm = _normalize_hex(share.target_hex)
        share_seed_norm = _normalize_hex(share.seed_hash_hex)

        if not blob_hex_norm or not target_hex_norm:
            return False
        if blob_hex_norm != prepared.blob_hex_norm:
            return False
        if target_hex_norm != prepared.target_hex:
            return False
        if share_seed_norm and share_seed_norm != prepared.seed_ctx.seed_hex_norm:
            return False
        return True

    def _build_prepared_seed(self, seed_hex_norm: str, blob: bytes) -> _PreparedSeed:
        seed = safe_bytes_from_hex(seed_hex_norm) if seed_hex_norm else b""
        if not seed:
            seed = blob[:32].ljust(32, b"\x00")
            seed_hex_norm = seed.hex()

        fp = self._dataset_fingerprint(seed)
        return _PreparedSeed(
            seed=seed,
            seed_hex_norm=seed_hex_norm,
            fingerprint=fp,
            dataset_fingerprint=fp,
        )

    def _build_prepared_from_job(self, job: MiningJob) -> _PreparedJob:
        blob_hex_norm = _normalize_hex(job.blob_hex)
        blob = safe_bytes_from_hex(blob_hex_norm)
        if not blob:
            raise ValueError("job.blob_hex is empty or invalid")

        if self.nonce_offset < 0 or (self.nonce_offset + 4) > len(blob):
            raise ValueError(f"invalid nonce_offset={self.nonce_offset} for blob length {len(blob)}")

        seed_hex_norm = _normalize_hex(job.seed_hash_hex)
        seed_ctx = self._build_prepared_seed(seed_hex_norm, blob)

        target_hex = _normalize_hex(job.target_hex)
        if not target_hex:
            raise ValueError("job.target_hex is empty")

        target_raw = parse_target_hex_to_bytes(target_hex)
        if not target_raw:
            raise ValueError("job.target_hex is invalid")

        full_target = target_hex_uses_full_256(target_hex)
        target_b = target_hex.encode("ascii", errors="strict")
        target64 = parse_target_hex_to_u64(target_hex)
        target_int = int.from_bytes(target_raw, "little", signed=False) if full_target else 0
        assigned_work = target_hex_to_assigned_work(target_hex)
        fingerprint = self._job_fingerprint(blob, target_hex)

        return _PreparedJob(
            job_id=str(getattr(job, "job_id", "") or ""),
            blob=blob,
            target_hex=target_hex,
            target_b=target_b,
            blob_hex_norm=blob_hex_norm,
            target_raw=target_raw,
            target_int=target_int,
            target64=target64,
            assigned_work=assigned_work,
            fingerprint=fingerprint,
            seed_ctx=seed_ctx,
            full_target=full_target,
        )

    def _build_prepared_from_share(self, share: CandidateShare) -> _PreparedJob:
        blob_hex_norm = _normalize_hex(share.blob_hex)
        blob = safe_bytes_from_hex(blob_hex_norm)
        if not blob:
            raise ValueError("candidate blob_hex is empty or invalid")

        if self.nonce_offset < 0 or (self.nonce_offset + 4) > len(blob):
            raise ValueError(f"invalid nonce_offset={self.nonce_offset} for blob length {len(blob)}")

        seed_hex_norm = _normalize_hex(share.seed_hash_hex)
        seed_ctx = self._build_prepared_seed(seed_hex_norm, blob)

        target_hex = _normalize_hex(share.target_hex)
        if not target_hex:
            raise ValueError("candidate target_hex is empty")

        target_raw = parse_target_hex_to_bytes(target_hex)
        if not target_raw:
            raise ValueError("candidate target_hex is invalid")

        full_target = target_hex_uses_full_256(target_hex)
        target_b = target_hex.encode("ascii", errors="strict")
        target64 = parse_target_hex_to_u64(target_hex)
        target_int = int.from_bytes(target_raw, "little", signed=False) if full_target else 0
        assigned_work = target_hex_to_assigned_work(target_hex)
        fingerprint = self._job_fingerprint(blob, target_hex)

        return _PreparedJob(
            job_id=str(getattr(share, "job_id", "") or ""),
            blob=blob,
            target_hex=target_hex,
            target_b=target_b,
            blob_hex_norm=blob_hex_norm,
            target_raw=target_raw,
            target_int=target_int,
            target64=target64,
            assigned_work=assigned_work,
            fingerprint=fingerprint,
            seed_ctx=seed_ctx,
            full_target=full_target,
        )

    def _get_thread_handle(self) -> _NativeHandle:
        handle = getattr(self._tls, "native_handle", None)
        if handle is not None:
            return handle

        lib = self._lib
        if lib is None:
            raise RuntimeError(self.disabled_reason or "native verifier is not available")

        with self._state_lock:
            prepared = self._current_prepared_job
            warm_target = self._warm_batch_vms_target

        handle = _NativeHandle(lib)
        tid = threading.get_ident()

        with self._handles_lock:
            self._thread_handles[tid] = handle

        self._tls.native_handle = handle
        self._log(f"[verify] created native handle for thread={tid}")

        if prepared is not None:
            try:
                handle.prepare(prepared, self.nonce_offset)
                if warm_target > 0 and handle.has_warm_batch_vms:
                    handle.warm_batch_vms(warm_target)
            except Exception as exc:
                self._log(f"[verify] thread handle warm prepare skipped: {exc}")

        return handle

    def _require_export_handle(self) -> _NativeHandle:
        handle = self._export_handle
        if handle is None:
            raise RuntimeError(self.disabled_reason or "native verifier is not available")
        return handle

    def _resolve_library_path(self, explicit: Optional[str]) -> Optional[str]:
        candidates: list[Path] = []

        if explicit:
            candidates.append(Path(explicit).expanduser())

        for key in self._ENV_DLL_KEYS:
            value = os.environ.get(key, "").strip()
            if value:
                candidates.append(Path(value).expanduser())

        here = Path.cwd()
        module_dir = Path(__file__).resolve().parent

        for base in (here, module_dir):
            for name in self._DEFAULT_LIB_NAMES:
                candidates.append(base / name)

        for p in candidates:
            try:
                rp = p.resolve()
            except Exception:
                rp = p
            if rp.exists() and rp.is_file():
                return str(rp)

        return None

    def _resolve_runtime_library_path(self, explicit: Optional[str]) -> Optional[str]:
        candidates: list[Path] = []

        if explicit:
            candidates.append(Path(explicit).expanduser())

        for key in self._ENV_RUNTIME_KEYS:
            value = os.environ.get(key, "").strip()
            if value:
                candidates.append(Path(value).expanduser())

        here = Path.cwd()
        module_dir = Path(__file__).resolve().parent

        for base in (here, module_dir):
            for name in self._DEFAULT_RUNTIME_NAMES:
                candidates.append(base / name)

        for p in candidates:
            try:
                rp = p.resolve()
            except Exception:
                rp = p
            if rp.exists() and rp.is_file():
                return str(rp)

        return None

    def _maybe_preload_randomx_runtime(self, explicit: Optional[str]) -> Optional[ctypes.CDLL]:
        runtime_path = self._resolve_runtime_library_path(explicit)
        if not runtime_path:
            if explicit:
                msg = f"RandomX runtime not found: {explicit}"
                if self.strict:
                    raise FileNotFoundError(msg)
                self._log(f"[verify] {msg}")
            return None

        self._randomx_runtime_path = runtime_path
        for key in self._ENV_RUNTIME_KEYS:
            os.environ[key] = runtime_path

        runtime_dir = str(Path(runtime_path).resolve().parent)
        self._add_dll_search_path(runtime_dir)

        try:
            lib = ctypes.CDLL(runtime_path)
            return lib
        except Exception as exc:
            if self.strict:
                raise
            self._log(f"[verify] failed to preload RandomX runtime {runtime_path}: {exc}")
            return None

    def _add_dll_search_path(self, directory: str) -> None:
        if not directory:
            return
        if os.name != "nt":
            return
        try:
            handle = os.add_dll_directory(directory)
            self._dll_directory_handles.append(handle)
        except Exception:
            pass

    def _load_library(self, path: str) -> ctypes.CDLL:
        lib_path = str(Path(path).resolve())

        self._add_dll_search_path(str(Path(lib_path).parent))
        if self._randomx_runtime_path:
            self._add_dll_search_path(str(Path(self._randomx_runtime_path).resolve().parent))

        lib = ctypes.CDLL(lib_path)

        required = ("bnrx_create", "bnrx_destroy", "bnrx_prepare_job", "bnrx_verify_nonce")
        for name in required:
            if not hasattr(lib, name):
                raise RuntimeError(f"native verifier is missing required export: {name}")

        lib.bnrx_create.argtypes = []
        lib.bnrx_create.restype = ctypes.c_void_p

        lib.bnrx_destroy.argtypes = [ctypes.c_void_p]
        lib.bnrx_destroy.restype = None

        if hasattr(lib, "bnrx_prepare_seed"):
            lib.bnrx_prepare_seed.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.c_size_t,
            ]
            lib.bnrx_prepare_seed.restype = ctypes.c_int

        if hasattr(lib, "bnrx_set_job"):
            lib.bnrx_set_job.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.c_char_p,
            ]
            lib.bnrx_set_job.restype = ctypes.c_int

        if hasattr(lib, "bnrx_warm_batch_vms"):
            lib.bnrx_warm_batch_vms.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            lib.bnrx_warm_batch_vms.restype = ctypes.c_int

        lib.bnrx_prepare_job.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.c_char_p,
        ]
        lib.bnrx_prepare_job.restype = ctypes.c_int

        lib.bnrx_verify_nonce.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_ubyte),
        ]
        lib.bnrx_verify_nonce.restype = ctypes.c_int

        if hasattr(lib, "bnrx_verify_nonce_batch"):
            lib.bnrx_verify_nonce_batch.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.c_size_t,
            ]
            lib.bnrx_verify_nonce_batch.restype = ctypes.c_int

        if hasattr(lib, "bnrx_hash_nonce"):
            lib.bnrx_hash_nonce.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_ubyte),
            ]
            lib.bnrx_hash_nonce.restype = ctypes.c_int

        if hasattr(lib, "bnrx_hash_nonce_batch"):
            lib.bnrx_hash_nonce_batch.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.c_size_t,
            ]
            lib.bnrx_hash_nonce_batch.restype = ctypes.c_int

        if hasattr(lib, "bnrx_tail_nonce_batch"):
            lib.bnrx_tail_nonce_batch.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.c_size_t,
            ]
            lib.bnrx_tail_nonce_batch.restype = ctypes.c_int

        if hasattr(lib, "bnrx_last_error"):
            lib.bnrx_last_error.argtypes = [ctypes.c_void_p]
            lib.bnrx_last_error.restype = ctypes.c_char_p

        if hasattr(lib, "bnrx_dataset_words64"):
            lib.bnrx_dataset_words64.argtypes = [ctypes.c_void_p]
            lib.bnrx_dataset_words64.restype = ctypes.c_size_t

        if hasattr(lib, "bnrx_export_dataset64"):
            lib.bnrx_export_dataset64.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
            ]
            lib.bnrx_export_dataset64.restype = ctypes.c_int

        self._log(
            "[verify] dataset exports loaded="
            f"{hasattr(lib, 'bnrx_dataset_words64') and hasattr(lib, 'bnrx_export_dataset64')}"
        )
        self._log(f"[verify] prepare seed loaded={hasattr(lib, 'bnrx_prepare_seed')}")
        self._log(f"[verify] set job loaded={hasattr(lib, 'bnrx_set_job')}")
        self._log(f"[verify] warm batch vms loaded={hasattr(lib, 'bnrx_warm_batch_vms')}")
        self._log(f"[verify] batch verify loaded={hasattr(lib, 'bnrx_verify_nonce_batch')}")
        self._log(f"[verify] batch hash loaded={hasattr(lib, 'bnrx_hash_nonce_batch')}")
        self._log(f"[verify] batch tail loaded={hasattr(lib, 'bnrx_tail_nonce_batch')}")
        return lib

    @staticmethod
    def _job_fingerprint(blob: bytes, target_hex: str) -> bytes:
        h = hashlib.sha256()
        h.update(blob)
        h.update(b"|")
        h.update(target_hex.encode("ascii", errors="ignore"))
        return h.digest()

    @staticmethod
    def _dataset_fingerprint(seed: bytes) -> bytes:
        h = hashlib.sha256()
        h.update(b"rx-dataset|")
        h.update(seed)
        return h.digest()

    def _log(self, text: str) -> None:
        if self.on_log is None:
            return
        try:
            self.on_log(text)
        except Exception:
            pass


def _to_ubyte_array(data: bytes):
    arr_t = ctypes.c_ubyte * len(data)
    return arr_t.from_buffer_copy(data)