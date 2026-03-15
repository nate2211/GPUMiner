from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MinerConfig:
    # backend: "stratum" | "blocknet" | "solo"
    mining_backend: str = "stratum"

    # direct stratum
    host: str = "127.0.0.1"
    port: int = 3333
    login: str = "x"
    password: str = "x"
    agent: str = "GPUMiner-PyQt5/0.2"
    use_tls: bool = False

    # BlockNet relay / P2Pool
    blocknet_api_relay: str = ""
    blocknet_api_token: str = ""
    blocknet_api_prefix: str = "/v1"
    blocknet_verify_tls: bool = False
    blocknet_timeout_s: float = 30.0
    blocknet_force_scheme: Optional[str] = None
    blocknet_poll_interval_ms: int = 50
    blocknet_poll_max_msgs: int = 32

    # Solo mining via monerod
    solo_wallet_address: str = ""
    solo_daemon_rpc_url: str = "http://127.0.0.1:18081"
    solo_zmq_pub_url: str = "tcp://127.0.0.1:18083"
    solo_use_zmq: bool = True
    solo_poll_fallback_s: float = 2.0
    solo_reserve_size: int = 60

    # OpenCL / RandomX
    kernel_path: str = "kernels/blocknet_randomx_vm_opencl.cl"
    kernel_entry: str = "blocknet_randomx_vm_scan_ext"
    opencl_loader: str = "OpenCL.dll"
    build_options: str = (
        "-cl-std=CL1.2 "
        "-DBN_PREFILTER_ROUNDS=96 "
        "-DBN_TAIL_PREDICT_ROUNDS=8 "
        "-DBN_TUNE_WORDS=256 "
        "-DBN_LOCAL_STAGE_SIZE=128 "
        "-DBN_LOCAL_TOPK=8"
    )
    platform_index: int = 0
    device_index: int = 0

    # Native verifier wrapper DLL/SO that exports bnrx_*.
    randomx_dll_path: str = ""

    # Optional official RandomX runtime.
    randomx_runtime_dll_path: str = ""
    preload_randomx_runtime: bool = True

    require_dataset: bool = True

    # Verification / submit behavior
    enable_cpu_verify: bool = True
    submit_unverified_shares: bool = False

    # Native CPU batch verify
    enable_cpu_verify_batch: bool = True
    cpu_verify_batch_size: int = 64
    cpu_verify_batch_wait_ms: int = 2
    cpu_verify_native_threads: int = 0

    # Native CPU exact hash batch teacher
    enable_cpu_hash_batch: bool = True
    cpu_hash_batch_min_size: int = 8
    cpu_hash_batch_threads: int = 0

    # Native CPU tail-only prescreen batch
    enable_cpu_tail_batch: bool = True
    cpu_tail_batch_min_size: int = 32
    cpu_tail_batch_threads: int = 0

    # GPU scan mode
    gpu_scan_mode: str = "chunk"  # "chunk" | "hash_batch"
    hash_batch_size: int = 262_144

    # GPU scan settings
    global_work_size: int = 2_097_152
    local_work_size: Optional[int] = 128
    max_results: int = 4096
    scan_pause_ms: int = 0
    nonce_offset: int = 39

    # Chunk mode settings
    scan_chunk_size: int = 262_144
    scan_candidate_target: int = 512
    max_scan_time_ms: int = 15

    # Verification / submit pipeline
    cpu_verify_limit: int = 1024
    verify_threads: int = 1
    verify_queue_limit: int = 2048
    submit_queue_limit: int = 512

    # Adaptive live-mining shaping
    adaptive_queue_throttle: bool = True
    verify_queue_soft_pct: float = 0.70
    submit_queue_soft_pct: float = 0.70
    min_candidate_target: int = 64
    min_dynamic_work_pct: float = 0.25
    job_age_soft_ms: int = 800
    job_age_hard_ms: int = 2500

    # Candidate handling / tuning
    enable_bucket_tuning: bool = True
    enable_job_tuning: bool = True
    sort_candidates: bool = True

    # Tail-bin + richer credit tuning
    tune_tail_bins: int = 16
    tune_confidence_div: float = 8.0
    tune_quality_reward: float = 1.0
    tune_stale_penalty: float = 1.0
    tune_duplicate_penalty: float = 0.25
    tune_invalid_penalty: float = 2.5
    tune_backend_error_penalty: float = 0.20

    # Stats
    stats_update_ms: int = 500

    # Shared tuning controls used by rank / threshold / credit learners
    ema_alpha: float = 0.08
    tune_verified_reward: float = 1.0
    tune_accepted_reward: float = 2.0
    tune_cpu_reject_penalty: float = 1.5
    tune_pool_reject_penalty: float = 2.0
    tune_pressure_penalty: float = 8.0
    tune_work_bonus_scale: float = 2.0

    def normalized_scan_mode(self) -> str:
        mode = (self.gpu_scan_mode or "chunk").strip().lower()
        return mode if mode in {"chunk", "hash_batch"} else "chunk"

    def active_scan_window(self) -> int:
        if self.normalized_scan_mode() == "hash_batch":
            return max(1, int(self.hash_batch_size))
        return max(1, int(self.global_work_size))

    def mining_backend_name(self) -> str:
        name = (self.mining_backend or "stratum").strip().lower()
        return name if name in {"stratum", "blocknet", "solo"} else "stratum"

    def normalized_tail_bins(self) -> int:
        n = max(1, int(self.tune_tail_bins))
        p = 1
        while p < n and p < 64:
            p <<= 1
        return max(1, min(64, p))

    def clamped_verify_soft_pct(self) -> float:
        return max(0.10, min(0.98, float(self.verify_queue_soft_pct)))

    def clamped_submit_soft_pct(self) -> float:
        return max(0.10, min(0.98, float(self.submit_queue_soft_pct)))

    def clamped_min_dynamic_work_pct(self) -> float:
        return max(0.05, min(1.00, float(self.min_dynamic_work_pct)))

    @property
    def use_blocknet(self) -> bool:
        return self.mining_backend_name() == "blocknet"

    @property
    def use_solo(self) -> bool:
        return self.mining_backend_name() == "solo"


@dataclass
class MiningJob:
    job_id: str
    blob_hex: str
    target_hex: str
    session_id: str = ""
    seed_hash_hex: str = ""
    height: int = 0
    algo: str = "rx/0"
    received_at: float = field(default_factory=time.time)

    # optional fields for solo mining
    submit_blob_hex: str = ""
    reserved_offset: int = 0
    prefilter_target64: Optional[int] = None
    backend: str = "stratum"


@dataclass
class CandidateShare:
    nonce: int
    gpu_hash_hex: str
    job_id: str
    blob_hex: str
    session_id: str
    target_hex: str
    seed_hash_hex: str = ""

    predicted_tail_u64: int = 0
    rank_score_u64: int = 0
    tune_bucket: int = -1
    tune_tail_bin: int = -1
    rank_quality: int = 128
    threshold_quality: int = 128

    job_age_ms: int = 0
    verify_pressure_q8: int = 0
    submit_pressure_q8: int = 0
    stale_risk_q8: int = 0

    # exact-teacher audit fields
    exact_hash_hex: str = ""
    exact_tail_u64: int = 0
    predictor_hash_match: bool = False


@dataclass
class VerifiedShare:
    nonce_hex: str
    result_hex: str
    job_id: str
    session_id: str
    solution_blob_hex: str = ""

    assigned_work: float = 0.0
    actual_work: float = 0.0
    credited_work: float = 0.0
    quality: float = 0.0

    actual_tail_u64: int = 0
    predicted_tail_u64: int = 0
    rank_score_u64: int = 0
    tune_bucket: int = -1
    tune_tail_bin: int = -1
    rank_quality: int = 128
    threshold_quality: int = 128

    # optional audit fields
    gpu_hash_hex: str = ""
    predictor_hash_match: bool = False


@dataclass
class SubmitResult:
    accepted: bool
    status: str = ""
    error: str = ""
    raw: Any = None

    reject_class: str = ""
    stale: bool = False
    duplicate: bool = False
    invalid: bool = False
    backend_error: bool = False