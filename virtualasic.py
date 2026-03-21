from __future__ import annotations

import ctypes
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from models import CandidateShare, MinerConfig, MiningJob
from opencl_miner import OpenCLGpuScanner
from utils import resolve_resource_path, safe_bytes_from_hex, target_hex_to_prefilter_u64


class _NoOpQueue:
    def finish(self) -> None:
        return None


class _VirtualAsicApi:
    def __init__(self, dll_path: str) -> None:
        if os.name != "nt":
            raise RuntimeError("VirtualASIC scanning is only implemented for Windows")

        self.dll_path = resolve_resource_path(dll_path)
        self.lib = ctypes.WinDLL(self.dll_path)
        self._bind()
        self.engine = self.lib.vasic_create()
        if not self.engine:
            raise RuntimeError(f"vasic_create failed for {self.dll_path}")

    def _bind(self) -> None:
        lib = self.lib
        lib.vasic_create.argtypes = []
        lib.vasic_create.restype = ctypes.c_void_p
        lib.vasic_destroy.argtypes = [ctypes.c_void_p]
        lib.vasic_destroy.restype = None
        lib.vasic_reset.argtypes = [ctypes.c_void_p]
        lib.vasic_reset.restype = ctypes.c_int
        lib.vasic_set_core_count.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.vasic_set_core_count.restype = ctypes.c_int
        lib.vasic_copy_last_error.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        lib.vasic_copy_last_error.restype = ctypes.c_int
        lib.vasic_create_buffer.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.vasic_create_buffer.restype = ctypes.c_uint32
        lib.vasic_release_buffer.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.vasic_release_buffer.restype = ctypes.c_int
        lib.vasic_write_buffer.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
        lib.vasic_write_buffer.restype = ctypes.c_int
        lib.vasic_read_buffer.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
        lib.vasic_read_buffer.restype = ctypes.c_int
        lib.vasic_load_kernel_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.vasic_load_kernel_file.restype = ctypes.c_uint32
        lib.vasic_release_kernel.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.vasic_release_kernel.restype = ctypes.c_int
        lib.vasic_set_kernel_arg_buffer.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        lib.vasic_set_kernel_arg_buffer.restype = ctypes.c_int
        lib.vasic_set_kernel_arg_u32.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        lib.vasic_set_kernel_arg_u32.restype = ctypes.c_int
        lib.vasic_enqueue_ndrange.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
        lib.vasic_enqueue_ndrange.restype = ctypes.c_int

    def close(self) -> None:
        if getattr(self, "engine", None):
            try:
                self.lib.vasic_destroy(self.engine)
            finally:
                self.engine = None

    def last_error(self) -> str:
        if not getattr(self, "engine", None):
            return ""
        buf = ctypes.create_string_buffer(4096)
        n = int(self.lib.vasic_copy_last_error(self.engine, buf, ctypes.sizeof(buf)))
        if n <= 0:
            return ""
        return buf.value.decode("utf-8", errors="replace")

    def _check(self, ok: int, action: str) -> None:
        if int(ok):
            return
        raise RuntimeError(self.last_error() or action)

    def create_buffer(self, size_bytes: int) -> int:
        size_bytes = max(1, int(size_bytes))
        buf_id = int(self.lib.vasic_create_buffer(self.engine, ctypes.c_uint32(size_bytes)))
        if buf_id <= 0:
            raise RuntimeError(self.last_error() or f"vasic_create_buffer failed size={size_bytes}")
        return buf_id

    def release_buffer(self, buffer_id: int) -> None:
        if int(buffer_id) > 0:
            self._check(self.lib.vasic_release_buffer(self.engine, ctypes.c_uint32(int(buffer_id))), "vasic_release_buffer failed")

    def write_buffer(self, buffer_id: int, data: np.ndarray | bytes | bytearray, offset: int = 0) -> None:
        if isinstance(data, np.ndarray):
            arr = np.ascontiguousarray(data)
            ptr = arr.ctypes.data_as(ctypes.c_void_p)
            size = int(arr.nbytes)
            keeper = arr
        else:
            raw = bytes(data)
            keeper = ctypes.create_string_buffer(raw, len(raw))
            ptr = ctypes.cast(keeper, ctypes.c_void_p)
            size = len(raw)
        self._check(
            self.lib.vasic_write_buffer(
                self.engine,
                ctypes.c_uint32(int(buffer_id)),
                ctypes.c_uint32(int(offset)),
                ptr,
                ctypes.c_uint32(int(size)),
            ),
            "vasic_write_buffer failed",
        )

    def read_into(self, buffer_id: int, out: np.ndarray, offset: int = 0) -> None:
        arr = np.ascontiguousarray(out)
        self._check(
            self.lib.vasic_read_buffer(
                self.engine,
                ctypes.c_uint32(int(buffer_id)),
                ctypes.c_uint32(int(offset)),
                arr.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_uint32(int(arr.nbytes)),
            ),
            "vasic_read_buffer failed",
        )
        out[...] = arr.reshape(out.shape)

    def load_kernel_file(self, kernel_name: str, file_path: str) -> int:
        kid = int(
            self.lib.vasic_load_kernel_file(
                self.engine,
                kernel_name.encode("utf-8"),
                file_path.encode("utf-8"),
            )
        )
        if kid <= 0:
            raise RuntimeError(self.last_error() or f"vasic_load_kernel_file failed: {kernel_name}")
        return kid

    def release_kernel(self, kernel_id: int) -> None:
        if int(kernel_id) > 0:
            self._check(self.lib.vasic_release_kernel(self.engine, ctypes.c_uint32(int(kernel_id))), "vasic_release_kernel failed")

    def set_arg_buffer(self, kernel_id: int, arg_index: int, buffer_id: int) -> None:
        self._check(
            self.lib.vasic_set_kernel_arg_buffer(
                self.engine,
                ctypes.c_uint32(int(kernel_id)),
                ctypes.c_uint32(int(arg_index)),
                ctypes.c_uint32(int(buffer_id)),
            ),
            f"vasic_set_kernel_arg_buffer failed arg={arg_index}",
        )

    def set_arg_u32(self, kernel_id: int, arg_index: int, value: int) -> None:
        self._check(
            self.lib.vasic_set_kernel_arg_u32(
                self.engine,
                ctypes.c_uint32(int(kernel_id)),
                ctypes.c_uint32(int(arg_index)),
                ctypes.c_uint32(int(value) & 0xFFFFFFFF),
            ),
            f"vasic_set_kernel_arg_u32 failed arg={arg_index}",
        )

    def enqueue(self, kernel_id: int, global_size: int) -> None:
        self._check(
            self.lib.vasic_enqueue_ndrange(
                self.engine,
                ctypes.c_uint32(int(kernel_id)),
                ctypes.c_uint32(int(global_size)),
            ),
            "vasic_enqueue_ndrange failed",
        )


class VirtualAsicScanner(OpenCLGpuScanner):
    def __init__(self, config: MinerConfig, on_log: Callable[[str], None]) -> None:
        super().__init__(config, on_log)
        self._api: Optional[_VirtualAsicApi] = None
        self._kernel_id: int = 0
        self._kernel_path_resolved: str = ""
        self._buffer_sizes: dict[str, int] = {}

    def initialize(self) -> None:
        if os.name != "nt":
            raise RuntimeError("VirtualASIC is only supported on Windows")

        # Optional: make sure the configured OpenCL loader is already resident.
        loader = str(getattr(self.config, "opencl_loader", "") or "").strip()
        if loader:
            ctypes.WinDLL(resolve_resource_path(loader))

        self._api = _VirtualAsicApi(self.config.virtualasic_dll_path or "virtualasic.dll")
        preferred = int(self.config.local_work_size or 0)
        if preferred > 0:
            try:
                self._api._check(self._api.lib.vasic_set_core_count(self._api.engine, ctypes.c_uint32(preferred)), "vasic_set_core_count failed")
            except Exception as exc:
                self.on_log(f"[virtualasic] set_core_count skipped: {exc}")

        self._kernel_path_resolved = resolve_resource_path(self.config.kernel_path)
        if not os.path.exists(self._kernel_path_resolved):
            raise FileNotFoundError(
                f"VirtualASIC kernel not found: requested={self.config.kernel_path} resolved={self._kernel_path_resolved}"
            )

        self._kernel_id = self._api.load_kernel_file(self._selected_kernel_entry(), self._kernel_path_resolved)
        self.program = self._kernel_id
        self.kernel = self._kernel_id
        self.queue = _NoOpQueue()
        self.ctx = True

        self.on_log(f"[virtualasic] loaded {self._api.dll_path}")
        self.on_log(
            f"[virtualasic] kernel={self._selected_kernel_entry()} file={self._kernel_path_resolved} "
            f"mode={self._scan_mode()} hybrid=cpu+gpu decorators=on"
        )

        self._ensure_tune_buffers()

    def _selected_kernel_entry(self) -> str:
        if self._scan_mode() == "hash_batch":
            return "blocknet_randomx_vm_hash_batch_vasic"
        return "blocknet_randomx_vm_scan_vasic"

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
        if self._api is None or not self._kernel_id:
            raise RuntimeError("VirtualASIC scanner not initialized")

        self.last_scan_work_items = 0
        self.last_scan_chunk_count = 0
        self.last_effective_candidate_target = max(1, int(scan_candidate_target_override or self.config.scan_candidate_target))
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

    def _ensure_tune_buffers(self) -> None:
        if self._api is None:
            raise RuntimeError("VirtualASIC scanner not initialized")

        size = max(1, int(self._PLANE_COUNT * self._cells))
        if self._seed_tune_buf is None:
            self._seed_tune_buf = self._api.create_buffer(size)
            self._buffer_sizes["seed_tune"] = size
        if self._job_tune_buf is None:
            self._job_tune_buf = self._api.create_buffer(size)
            self._buffer_sizes["job_tune"] = size

        self._upload_tuning_unlocked(force=True)

    def bind_dataset(self, dataset_u64: np.ndarray, dataset_fingerprint: Optional[bytes]) -> None:
        if self._api is None:
            raise RuntimeError("VirtualASIC scanner not initialized")

        ds = np.ascontiguousarray(dataset_u64, dtype=np.uint64)
        want_bytes = int(ds.nbytes)
        if self.dataset_buf is None or self._buffer_sizes.get("dataset") != want_bytes:
            if self.dataset_buf is not None:
                self._api.release_buffer(int(self.dataset_buf))
            self.dataset_buf = self._api.create_buffer(want_bytes)
            self._buffer_sizes["dataset"] = want_bytes

        if want_bytes > 0:
            self._api.write_buffer(int(self.dataset_buf), ds)

        self.dataset_words = int(ds.size)
        self.dataset_fingerprint = dataset_fingerprint

        with self._cl_lock:
            self._reset_tune_state(self._seed_tune)
            self._reset_tune_state(self._job_tune)
            self._current_job_key = None
            self._upload_tuning_unlocked(force=True)

        mib = ds.nbytes / (1024.0 * 1024.0)
        self.on_log(f"[virtualasic] RandomX dataset ready: words={self.dataset_words} size={mib:.2f} MiB")

    def _ensure_buf(self, key: str, attr: str, size_bytes: int) -> int:
        if self._api is None:
            raise RuntimeError("VirtualASIC scanner not initialized")
        cur = getattr(self, attr)
        if cur is None or self._buffer_sizes.get(key) != int(size_bytes):
            if cur is not None:
                self._api.release_buffer(int(cur))
            cur = self._api.create_buffer(int(size_bytes))
            setattr(self, attr, cur)
            self._buffer_sizes[key] = int(size_bytes)
        return int(cur)

    def _ensure_job_buffers(self, blob: bytes, seed: bytes) -> None:
        blob_id = self._ensure_buf("blob", "_blob_buf", len(blob))
        if self._blob_bytes != blob:
            self._api.write_buffer(blob_id, blob)
            self._blob_bytes = bytes(blob)
            self._blob_nbytes = len(blob)

        seed_id = self._ensure_buf("seed", "_seed_buf", len(seed))
        if self._seed_bytes != seed:
            self._api.write_buffer(seed_id, seed)
            self._seed_bytes = bytes(seed)
            self._seed_nbytes = len(seed)

    def _ensure_output_buffers(self, max_results: int) -> None:
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

        self._out_hashes_np = np.empty((max_results, 32), dtype=np.uint8)
        self._out_nonces_np = np.empty((max_results,), dtype=np.uint32)
        self._out_scores_np = np.empty((max_results,), dtype=np.uint64)
        self._out_buckets_np = np.empty((max_results,), dtype=np.uint32)
        self._out_rankq_np = np.empty((max_results,), dtype=np.uint8)
        self._out_threshq_np = np.empty((max_results,), dtype=np.uint8)
        self._out_tailbin_np = np.empty((max_results,), dtype=np.uint8)
        self._out_count_np = np.zeros((1,), dtype=np.uint32)

        self._out_hashes_buf = self._ensure_buf("out_hashes", "_out_hashes_buf", int(self._out_hashes_np.nbytes))
        self._out_nonces_buf = self._ensure_buf("out_nonces", "_out_nonces_buf", int(self._out_nonces_np.nbytes))
        self._out_scores_buf = self._ensure_buf("out_scores", "_out_scores_buf", int(self._out_scores_np.nbytes))
        self._out_buckets_buf = self._ensure_buf("out_buckets", "_out_buckets_buf", int(self._out_buckets_np.nbytes))
        self._out_rankq_buf = self._ensure_buf("out_rankq", "_out_rankq_buf", int(self._out_rankq_np.nbytes))
        self._out_threshq_buf = self._ensure_buf("out_threshq", "_out_threshq_buf", int(self._out_threshq_np.nbytes))
        self._out_tailbin_buf = self._ensure_buf("out_tailbin", "_out_tailbin_buf", int(self._out_tailbin_np.nbytes))
        self._out_count_buf = self._ensure_buf("out_count", "_out_count_buf", int(self._out_count_np.nbytes))
        self._out_capacity = int(max_results)

    def _reset_out_count(self) -> None:
        if self._api is None or self._out_count_buf is None:
            raise RuntimeError("VirtualASIC output buffers not initialized")
        zero = np.zeros((1,), dtype=np.uint32)
        self._api.write_buffer(int(self._out_count_buf), zero)

    def _upload_tuning_unlocked(self, force: bool = False) -> None:
        if self._api is None or self._seed_tune_buf is None or self._job_tune_buf is None:
            return

        if force or self._seed_tune.dirty:
            self._recompute_tune_planes(self._seed_tune)
            self._api.write_buffer(int(self._seed_tune_buf), self._seed_tune.planes)

        if force or self._job_tune.dirty:
            self._recompute_tune_planes(self._job_tune)
            self._api.write_buffer(int(self._job_tune_buf), self._job_tune.planes)

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
        if self._api is None or self.dataset_buf is None:
            raise RuntimeError("VirtualASIC scanner not initialized")

        self._begin_job_unlocked(job)
        self._upload_tuning_unlocked(force=False)
        self._reset_out_count()

        target64_int = int(target64) & 0xFFFFFFFFFFFFFFFF
        target_lo = target64_int & 0xFFFFFFFF
        target_hi = (target64_int >> 32) & 0xFFFFFFFF

        # buffer args
        self._api.set_arg_buffer(self._kernel_id, 0, int(self._blob_buf))
        self._api.set_arg_u32(self._kernel_id, 1, len(blob))
        self._api.set_arg_u32(self._kernel_id, 2, int(self.config.nonce_offset))
        self._api.set_arg_u32(self._kernel_id, 3, int(start_nonce) & 0xFFFFFFFF)
        self._api.set_arg_u32(self._kernel_id, 4, target_lo)
        self._api.set_arg_u32(self._kernel_id, 5, target_hi)
        self._api.set_arg_u32(self._kernel_id, 6, int(max_results))
        self._api.set_arg_buffer(self._kernel_id, 7, int(self._out_hashes_buf))
        self._api.set_arg_buffer(self._kernel_id, 8, int(self._out_nonces_buf))
        self._api.set_arg_buffer(self._kernel_id, 9, int(self._out_scores_buf))
        self._api.set_arg_buffer(self._kernel_id, 10, int(self._out_buckets_buf))
        self._api.set_arg_buffer(self._kernel_id, 11, int(self._out_rankq_buf))
        self._api.set_arg_buffer(self._kernel_id, 12, int(self._out_threshq_buf))
        self._api.set_arg_buffer(self._kernel_id, 13, int(self._out_tailbin_buf))
        self._api.set_arg_buffer(self._kernel_id, 14, int(self._out_count_buf))
        self._api.set_arg_buffer(self._kernel_id, 15, int(self._seed_buf))
        self._api.set_arg_u32(self._kernel_id, 16, len(seed))
        self._api.set_arg_buffer(self._kernel_id, 17, int(self.dataset_buf))
        self._api.set_arg_u32(self._kernel_id, 18, int(self.dataset_words))
        self._api.set_arg_buffer(self._kernel_id, 19, int(self._seed_tune_buf))
        self._api.set_arg_u32(self._kernel_id, 20, int(self._bucket_count))
        self._api.set_arg_u32(self._kernel_id, 21, int(self._tail_bins))
        self._api.set_arg_buffer(self._kernel_id, 22, int(self._job_tune_buf))
        self._api.set_arg_u32(self._kernel_id, 23, int(self._bucket_count if self.config.enable_job_tuning else 0))
        self._api.set_arg_u32(self._kernel_id, 24, int(self._tail_bins if self.config.enable_job_tuning else 0))
        self._api.set_arg_u32(self._kernel_id, 25, max(0, int(job_age_ms)))
        self._api.set_arg_u32(self._kernel_id, 26, max(0, min(255, int(verify_pressure_q8))))
        self._api.set_arg_u32(self._kernel_id, 27, max(0, min(255, int(submit_pressure_q8))))
        self._api.set_arg_u32(self._kernel_id, 28, max(0, min(255, int(stale_risk_q8))))

        self._api.enqueue(self._kernel_id, int(work_items))

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
            raise RuntimeError("VirtualASIC output host buffers are not initialized")

        self._api.read_into(int(self._out_count_buf), self._out_count_np)
        raw_count = int(self._out_count_np[0])
        count = min(raw_count, max_results)
        candidates: list[CandidateShare] = []

        if count > 0:
            self._api.read_into(int(self._out_nonces_buf), self._out_nonces_np)
            self._api.read_into(int(self._out_hashes_buf), self._out_hashes_np)
            self._api.read_into(int(self._out_scores_buf), self._out_scores_np)
            self._api.read_into(int(self._out_buckets_buf), self._out_buckets_np)
            self._api.read_into(int(self._out_rankq_buf), self._out_rankq_np)
            self._api.read_into(int(self._out_threshq_buf), self._out_threshq_np)
            self._api.read_into(int(self._out_tailbin_buf), self._out_tailbin_np)

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
                f"[virtualasic] candidate buffer saturated for job={job.job_id}: raw_count={raw_count}, max_results={max_results}"
            )

        return candidates, raw_count

    def _begin_job_unlocked(self, job: MiningJob) -> None:
        key = self._job_key_from_job(job)
        if key == self._current_job_key:
            return

        self._current_job_key = key
        self._reset_tune_state(self._job_tune)
        self.on_log(
            f"[virtualasic] job tuning reset for job={job.job_id} session={job.session_id or '-'}"
        )

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
            self.on_log(f"[virtualasic] empty/invalid blob for job={job.job_id}")
            return []

        if len(blob) < self.config.nonce_offset + 4:
            self.on_log(
                f"[virtualasic] blob too short for nonce offset {self.config.nonce_offset}: "
                f"len={len(blob)} job={job.job_id}"
            )
            return []

        seed = safe_bytes_from_hex(job.seed_hash_hex)
        if not seed and (job.algo or "rx/0").lower() == "rx/0":
            self.on_log(f"[virtualasic] missing seed_hash for rx/0 job={job.job_id}, skipping scan")
            return []

        if not seed:
            seed = blob[:32].ljust(32, b"\x00")

        full_target = self._is_full_target(job.target_hex)
        target64 = self._job_prefilter_target64(job)

        requested_candidate_target = max(
            1,
            int(candidate_target_override or self.config.scan_candidate_target),
        )
        max_results = self._effective_max_results(requested_candidate_target, full_target=full_target)
        candidate_target = min(requested_candidate_target, max_results)

        total_work = max(1, int(work_items_override or self.config.global_work_size))
        chunk_size = max(1, int(getattr(self.config, "scan_chunk_size", 262144)))
        max_scan_time_ms = max(0, int(getattr(self.config, "max_scan_time_ms", 15)))

        all_candidates: list[CandidateShare] = []
        scanned = 0
        chunk_count = 0
        started = time.perf_counter()
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
                chunk_candidates, raw_count = self._run_one_launch_unlocked(
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
                    if job_age_ms >= 250 or raw_count >= max_results:
                        early_stop_reason = f"candidate_target={candidate_target}"
                        break

                if max_scan_time_ms > 0:
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    if elapsed_ms >= float(max_scan_time_ms):
                        early_stop_reason = f"time_budget_ms={max_scan_time_ms}"
                        break

            self._upload_tuning_unlocked(force=False)

        self.last_scan_work_items = scanned
        self.last_scan_chunk_count = chunk_count

        if chunk_count > 1 or early_stop_reason:
            self.on_log(
                f"[virtualasic] chunked scan job={job.job_id} chunks={chunk_count} "
                f"scanned={scanned}/{total_work} kept={len(all_candidates)} "
                + (f"stop={early_stop_reason} " if early_stop_reason else "")
                + f"full_target={1 if full_target else 0} "
                  f"job_age_ms={job_age_ms} verify_q8={verify_pressure_q8} "
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
            self.on_log(f"[virtualasic] empty/invalid blob for job={job.job_id}")
            return []

        if len(blob) < self.config.nonce_offset + 4:
            self.on_log(
                f"[virtualasic] blob too short for nonce offset {self.config.nonce_offset}: "
                f"len={len(blob)} job={job.job_id}"
            )
            return []

        seed = safe_bytes_from_hex(job.seed_hash_hex)
        if not seed and (job.algo or "rx/0").lower() == "rx/0":
            self.on_log(f"[virtualasic] missing seed_hash for rx/0 job={job.job_id}, skipping hash_batch")
            return []

        if not seed:
            seed = blob[:32].ljust(32, b"\x00")

        full_target = self._is_full_target(job.target_hex)
        target64 = self._job_prefilter_target64(job)

        requested_candidate_target = max(
            1,
            int(candidate_target_override or self.config.scan_candidate_target),
        )
        max_results = self._effective_max_results(requested_candidate_target, full_target=full_target)
        candidate_target = min(requested_candidate_target, max_results)

        work_items = max(
            1,
            int(work_items_override or getattr(self.config, "hash_batch_size", self.config.global_work_size)),
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

        self.last_scan_work_items = int(work_items)
        self.last_scan_chunk_count = 1

        self.on_log(
            f"[virtualasic] hash_batch job={job.job_id} work_items={work_items} kept={len(candidates)} "
            f"full_target={1 if full_target else 0} "
            f"job_age_ms={job_age_ms} verify_q8={verify_pressure_q8} "
            f"submit_q8={submit_pressure_q8} stale_q8={stale_risk_q8}"
        )
        return candidates

    def close(self) -> None:
        try:
            if self._api is not None:
                for attr in (
                    "dataset_buf",
                    "_seed_tune_buf",
                    "_job_tune_buf",
                    "_blob_buf",
                    "_seed_buf",
                    "_out_hashes_buf",
                    "_out_nonces_buf",
                    "_out_scores_buf",
                    "_out_buckets_buf",
                    "_out_rankq_buf",
                    "_out_threshq_buf",
                    "_out_tailbin_buf",
                    "_out_count_buf",
                ):
                    value = getattr(self, attr, None)
                    if value is not None:
                        try:
                            self._api.release_buffer(int(value))
                        except Exception:
                            pass
                        setattr(self, attr, None)

                if self._kernel_id:
                    try:
                        self._api.release_kernel(int(self._kernel_id))
                    except Exception:
                        pass
                    self._kernel_id = 0
        finally:
            if self._api is not None:
                try:
                    self._api.close()
                finally:
                    self._api = None
            self.program = None
            self.kernel = None
            self.queue = None
            self.ctx = None
            self.dataset_words = 0
            self.dataset_fingerprint = None
            self._buffer_sizes.clear()
