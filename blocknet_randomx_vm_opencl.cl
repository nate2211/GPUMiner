/*
 * blocknet_randomx_fused_all_kernels.cl
 * RandomX-structured predictive OpenCL core with ABI-compatible wrappers.
 *
 * Included entry kernels:
 *   - blocknet_randomx_basic_hash          (simple per-work-item hash dump)
 *   - blocknet_randomx_basic_scan          (simple direct target scan)
 *   - blocknet_randomx_basic               (tuned-core compatibility alias)
 *   - blocknet_randomx_fast                (tuned-core compatibility alias)
 *   - blocknet_randomx_optimized           (compat wrapper from optimized snippet)
 *   - blocknet_randomx_vm_scan_basic       (tuned-core basic alias, ulong target)
 *   - blocknet_randomx_vm_hash_batch_basic (tuned-core basic alias, ulong target)
 *   - blocknet_randomx_vm_scan_fast        (tuned-core fast alias, ulong target)
 *   - blocknet_randomx_vm_hash_batch_fast  (tuned-core fast alias, ulong target)
 *   - blocknet_randomx_vm_scan_ext         (original ext wrapper, ulong target)
 *   - blocknet_randomx_vm_hash_batch_ext   (original ext wrapper, ulong target)
 *   - blocknet_randomx_vm_scan_vasic       (original vasic wrapper, target lo/hi)
 *   - blocknet_randomx_vm_hash_batch_vasic (original vasic wrapper, target lo/hi)
 *
 * Notes:
 *   - This is OpenCL C. Do not include <CL/cl.h> inside this file.
 *   - Every public kernel name and argument list from the uploaded source is preserved.
 *   - The predictor now follows the RandomX data flow: BLAKE2b input seed, RandomX AES
 *     scratchpad/program generation, RandomX opcode frequencies, integer VM operations,
 *     aligned 64-byte Dataset reads, chained programs, scratchpad fingerprinting, and
 *     BLAKE2b finalization.
 *   - It deliberately uses bounded predictor constants so the existing ABI remains
 *     launchable without a separate 2 MiB mutable scratchpad allocation per hash.
 *     Therefore its 32-byte result is a predictor/ranking digest, NOT a consensus
 *     RandomX hash. CPU/native full-RandomX verification remains mandatory.
 */

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#ifndef BN_HASH_BYTES
#define BN_HASH_BYTES 32u
#endif

#ifndef BN_MAX_BLOB_BYTES
#define BN_MAX_BLOB_BYTES 512u
#endif

#ifndef BN_PREFILTER_ROUNDS
#define BN_PREFILTER_ROUNDS 80u
#endif

#ifndef BN_TAIL_PREDICT_ROUNDS
#define BN_TAIL_PREDICT_ROUNDS 8u
#endif

#ifndef BN_TUNE_WORDS
#define BN_TUNE_WORDS 256u
#endif

#ifndef BN_TUNE_NEUTRAL
#define BN_TUNE_NEUTRAL 128u
#endif

#ifndef BN_LOCAL_STAGE_SIZE
#define BN_LOCAL_STAGE_SIZE 128u
#endif

#ifndef BN_LOCAL_TOPK
#define BN_LOCAL_TOPK 64u
#endif

#ifndef BN_PREFILTER_ROUNDS_FAST
#define BN_PREFILTER_ROUNDS_FAST 48u
#endif

#ifndef BN_FINAL_MIX_ROUNDS_FAST
#define BN_FINAL_MIX_ROUNDS_FAST 8u
#endif

#ifndef BN_FAST_ABSORB_STRIDE
#define BN_FAST_ABSORB_STRIDE 4u
#endif

/*
 * ABI-preserving RandomX predictor profile.
 *
 * Consensus RandomX uses a 2 MiB scratchpad, 256 instructions, 2048 program
 * iterations, and 8 programs. Those values require a multi-kernel pipeline and
 * a host-provided mutable scratchpad buffer. The current public ABI has no such
 * buffer, so the defaults below intentionally bound the private working set.
 *
 * They may be raised with OpenCL build options for experiments, but raising
 * them does not make this digest consensus-valid: the scratchpad is still
 * smaller than the mandatory RandomX L3 scratchpad.
 */
#ifndef BN_RX_PREDICT_SCRATCH_WORDS
#define BN_RX_PREDICT_SCRATCH_WORDS 64u
#endif

#ifndef BN_RX_PREDICT_PROGRAM_SIZE
#define BN_RX_PREDICT_PROGRAM_SIZE 32u
#endif

#ifndef BN_RX_PREDICT_ITERATIONS
#define BN_RX_PREDICT_ITERATIONS 8u
#endif

#ifndef BN_RX_PREDICT_PROGRAMS
#define BN_RX_PREDICT_PROGRAMS 2u
#endif

#ifndef BN_RX_PREDICT_MAX_BRANCH_STEPS
#define BN_RX_PREDICT_MAX_BRANCH_STEPS (BN_RX_PREDICT_PROGRAM_SIZE * 4u)
#endif

#if BN_RX_PREDICT_SCRATCH_WORDS < 64u
#error "BN_RX_PREDICT_SCRATCH_WORDS must be at least 64"
#endif

#if (BN_RX_PREDICT_SCRATCH_WORDS & (BN_RX_PREDICT_SCRATCH_WORDS - 1u)) != 0u
#error "BN_RX_PREDICT_SCRATCH_WORDS must be a power of two"
#endif

#if (BN_RX_PREDICT_SCRATCH_WORDS & 7u) != 0u
#error "BN_RX_PREDICT_SCRATCH_WORDS must contain whole 64-byte lines"
#endif

#if BN_RX_PREDICT_PROGRAM_SIZE < 16u || BN_RX_PREDICT_PROGRAM_SIZE > 256u
#error "BN_RX_PREDICT_PROGRAM_SIZE must be in [16, 256]"
#endif

#if BN_RX_PREDICT_ITERATIONS < 1u
#error "BN_RX_PREDICT_ITERATIONS must be positive"
#endif

#if BN_RX_PREDICT_PROGRAMS < 1u || BN_RX_PREDICT_PROGRAMS > 8u
#error "BN_RX_PREDICT_PROGRAMS must be in [1, 8]"
#endif

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define BN_RX_HAVE_FP64 1
#else
#define BN_RX_HAVE_FP64 0
#endif

#define BN_U64_C(x) ((ulong)(x##UL))

#define BN_PLANE_RANK       0u
#define BN_PLANE_THRESHOLD  1u
#define BN_PLANE_CREDIT     2u
#define BN_PLANE_CONFIDENCE 3u
#define BN_PLANE_COUNT      4u

#define BN_STAGE_REJECT 0u
#define BN_STAGE_PASS   1u
#define BN_STAGE_NEAR   2u

inline ulong bn_rotl64(ulong x, uint r) {
    return (x << (r & 63u)) | (x >> ((64u - r) & 63u));
}

inline ulong bn_rotr64(ulong x, uint r) {
    return (x >> (r & 63u)) | (x << ((64u - r) & 63u));
}

inline ulong bn_mix64(ulong x) {
    x ^= x >> 30;
    x *= BN_U64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= BN_U64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return x;
}

inline ulong bn_avalanche64(ulong x) {
    x ^= x >> 33;
    x *= BN_U64_C(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x *= BN_U64_C(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    return x;
}

inline ulong bn_mulh64(ulong a, ulong b) {
    return mul_hi(a, b);
}

inline uint bn_popcount64(ulong x) {
    return popcount((uint)x) + popcount((uint)(x >> 32));
}

inline uint bn_min_u32(uint a, uint b) {
    return (a < b) ? a : b;
}

inline uint bn_max_u32(uint a, uint b) {
    return (a > b) ? a : b;
}

inline ulong bn_min_u64(ulong a, ulong b) {
    return (a < b) ? a : b;
}

inline ulong bn_max_u64(ulong a, ulong b) {
    return (a > b) ? a : b;
}

inline ulong bn_min3_u64(ulong a, ulong b, ulong c) {
    return bn_min_u64(a, bn_min_u64(b, c));
}

inline ulong bn_max3_u64(ulong a, ulong b, ulong c) {
    return bn_max_u64(a, bn_max_u64(b, c));
}

inline ulong bn_median3_u64(ulong a, ulong b, ulong c) {
    if (a < b) {
        if (b < c) return b;
        return (a < c) ? c : a;
    } else {
        if (a < c) return a;
        return (b < c) ? c : b;
    }
}

inline ulong bn_rng_step(__private ulong* s) {
    ulong x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * BN_U64_C(0x2545F4914F6CDD1D);
}

inline void bn_store_u64_le_g(__global uchar* p, ulong v) {
    p[0] = (uchar)(v & 0xFFUL);
    p[1] = (uchar)((v >> 8) & 0xFFUL);
    p[2] = (uchar)((v >> 16) & 0xFFUL);
    p[3] = (uchar)((v >> 24) & 0xFFUL);
    p[4] = (uchar)((v >> 32) & 0xFFUL);
    p[5] = (uchar)((v >> 40) & 0xFFUL);
    p[6] = (uchar)((v >> 48) & 0xFFUL);
    p[7] = (uchar)((v >> 56) & 0xFFUL);
}

inline void bn_write_hash32(__global uchar* dst32, ulong h0, ulong h1, ulong h2, ulong h3) {
    bn_store_u64_le_g(dst32 +  0u, h0);
    bn_store_u64_le_g(dst32 +  8u, h1);
    bn_store_u64_le_g(dst32 + 16u, h2);
    bn_store_u64_le_g(dst32 + 24u, h3);
}

inline uint bn_log2_pow2(uint v) {
    uint s = 0u;
    while (v > 1u) {
        v >>= 1u;
        ++s;
    }
    return s;
}

inline uint bn_tail_bin_from_tail(ulong tail64, uint tail_bins) {
    if (tail_bins <= 1u) {
        return 0u;
    }
    uint bits = bn_log2_pow2(tail_bins);
    uint shift = 64u - bits;
    return (uint)(tail64 >> shift);
}

inline uint bn_tune_bucket(ulong h0, ulong h1, uint nonce_u32) {
#if BN_TUNE_WORDS > 0
    ulong key = bn_mix64(h0 ^ bn_rotl64(h1, 13u) ^ (ulong)nonce_u32);
    return (uint)(key % (ulong)BN_TUNE_WORDS);
#else
    (void)h0;
    (void)h1;
    (void)nonce_u32;
    return 0u;
#endif
}

inline uint bn_tune_index(uint bucket, uint tail_bin, uint buckets, uint tail_bins) {
    return ((bucket % buckets) * tail_bins) + (tail_bin % tail_bins);
}

inline uint bn_read_tune_quality(
    __global const uchar* tune,
    uint buckets,
    uint tail_bins,
    uint plane,
    uint bucket,
    uint tail_bin,
    uint fallback
) {
    if (buckets == 0u || tail_bins == 0u) {
        return fallback;
    }
    uint stride = buckets * tail_bins;
    uint idx = bn_tune_index(bucket, tail_bin, buckets, tail_bins);
    return (uint)tune[(plane * stride) + idx];
}

inline uint bn_blend_quality(uint seed_q, uint seed_conf, uint job_q, uint job_conf) {
    ulong sw = (ulong)bn_max_u32(1u, seed_conf);
    ulong jw = (ulong)(bn_max_u32(0u, job_conf) * 2u);

    if (jw == 0UL) {
        return seed_q;
    }

    long seed_term = ((long)seed_q - (long)BN_TUNE_NEUTRAL) * (long)sw;
    long job_term = ((long)job_q - (long)BN_TUNE_NEUTRAL) * (long)jw;
    long total = (long)(sw + jw);
    long q = (long)BN_TUNE_NEUTRAL + ((seed_term + job_term) / ((total > 0L) ? total : 1L));

    if (q < 0L) q = 0L;
    if (q > 255L) q = 255L;
    return (uint)q;
}

inline ulong bn_adjust_target64(ulong target64, uint threshold_quality) {
    if (target64 <= 1UL) {
        return target64;
    }

    if (threshold_quality < BN_TUNE_NEUTRAL) {
        uint delta = BN_TUNE_NEUTRAL - threshold_quality;
        ulong tighten = ((target64 >> 3) * (ulong)delta) / (ulong)BN_TUNE_NEUTRAL;
        if (tighten >= target64) {
            return 1UL;
        }
        return target64 - tighten;
    }

    if (threshold_quality > BN_TUNE_NEUTRAL) {
        uint delta = threshold_quality - BN_TUNE_NEUTRAL;
        ulong loosen = ((target64 >> 2) * (ulong)delta) / (ulong)BN_TUNE_NEUTRAL;
        if (target64 > (~0UL - loosen)) {
            return ~0UL;
        }
        return target64 + loosen;
    }

    return target64;
}

inline ulong bn_apply_operational_tightening(
    ulong target64,
    uint job_age_ms,
    uint verify_pressure_q8,
    uint submit_pressure_q8,
    uint stale_risk_q8
) {
    if (target64 <= 1UL) {
        return target64;
    }

    if (job_age_ms < 250u) {
        verify_pressure_q8 >>= 2;
        submit_pressure_q8 >>= 2;
        stale_risk_q8 >>= 2;
    } else if (job_age_ms < 750u) {
        verify_pressure_q8 >>= 1;
        submit_pressure_q8 >>= 1;
        stale_risk_q8 >>= 1;
    }

    uint pressure = bn_max_u32(verify_pressure_q8, bn_max_u32(submit_pressure_q8, stale_risk_q8));
    ulong tighten_pressure = ((target64 >> 5) * (ulong)pressure) / 255UL;
    uint age_ms = bn_min_u32(job_age_ms, 5000u);
    ulong tighten_age = ((target64 >> 5) * (ulong)age_ms) / 5000UL;
    ulong tighten = tighten_pressure + tighten_age;

    if (tighten >= target64) {
        return 1UL;
    }
    return bn_max_u64(1UL, target64 - tighten);
}

inline ulong bn_apply_early_job_relaxation(
    ulong target64,
    uint job_age_ms,
    uint verify_pressure_q8,
    uint submit_pressure_q8,
    uint stale_risk_q8,
    uint confidence_quality
) {
    uint pressure = bn_max_u32(verify_pressure_q8, bn_max_u32(submit_pressure_q8, stale_risk_q8));
    if (job_age_ms > 1400u || pressure > 96u) {
        return target64;
    }

    ulong bonus = target64 >> 1;
    if (confidence_quality >= 96u) {
        bonus += (target64 >> 3);
    } else if (confidence_quality >= 48u) {
        bonus += (target64 >> 4);
    }

    if (target64 > (~0UL - bonus)) {
        return ~0UL;
    }
    return target64 + bonus;
}

inline ulong bn_near_target64(ulong target64, uint confidence_quality) {
    ulong bonus = target64 >> 1;

    if (confidence_quality >= 96u) {
        bonus += (target64 >> 2);
    } else if (confidence_quality >= 48u) {
        bonus += (target64 >> 3);
    } else if (confidence_quality < 16u) {
        bonus >>= 1;
    }

    if (target64 > (~0UL - bonus)) {
        return ~0UL;
    }
    return target64 + bn_max_u64(1UL, bonus);
}

inline ulong bn_rank_score(ulong tail64, uint rank_quality) {
    if (rank_quality >= BN_TUNE_NEUTRAL) {
        return tail64;
    }

    uint delta = BN_TUNE_NEUTRAL - rank_quality;
    ulong penalty = ((ulong)delta) << 46;
    ulong limit = ~0UL - penalty;
    return (tail64 >= limit) ? ~0UL : (tail64 + penalty);
}

inline ulong bn_add_penalty_sat(ulong score, ulong penalty) {
    ulong limit = ~0UL - penalty;
    if (score >= limit) {
        return ~0UL;
    }
    return score + penalty;
}

inline ulong bn_apply_credit_bonus(ulong score, uint credit_quality, uint confidence_quality) {
    if (credit_quality <= BN_TUNE_NEUTRAL || confidence_quality == 0u) {
        return score;
    }

    uint delta = credit_quality - BN_TUNE_NEUTRAL;
    ulong bonus = (((ulong)delta) * (ulong)(confidence_quality + 1u)) << 22;

    if (bonus >= score) {
        return 0UL;
    }
    return score - bonus;
}

inline ulong bn_apply_operational_penalty(
    ulong score,
    uint job_age_ms,
    uint verify_pressure_q8,
    uint submit_pressure_q8,
    uint stale_risk_q8
) {
    if (job_age_ms < 250u) {
        verify_pressure_q8 >>= 2;
        submit_pressure_q8 >>= 2;
        stale_risk_q8 >>= 2;
    } else if (job_age_ms < 750u) {
        verify_pressure_q8 >>= 1;
        submit_pressure_q8 >>= 1;
        stale_risk_q8 >>= 1;
    }

    uint pressure = bn_max_u32(verify_pressure_q8, bn_max_u32(submit_pressure_q8, stale_risk_q8));
    ulong penalty = (((ulong)pressure) << 30);
    uint age_ms = bn_min_u32(job_age_ms, 4000u);
    penalty += (((ulong)age_ms) << 18);

    ulong limit = ~0UL - penalty;
    if (score >= limit) {
        return ~0UL;
    }
    return score + penalty;
}

inline uchar bn_blob_byte_overlay(
    __global const uchar* blob,
    uint blob_len,
    uint nonce_offset,
    uint nonce_u32,
    uint idx
) {
    if (idx >= blob_len) {
        return (uchar)0u;
    }

    if (idx >= nonce_offset && idx < nonce_offset + 4u && idx < blob_len) {
        uint sh = (idx - nonce_offset) * 8u;
        return (uchar)((nonce_u32 >> sh) & 0xFFu);
    }

    return blob[idx];
}

inline ulong bn_blob_load_u64_overlay(
    __global const uchar* blob,
    uint blob_len,
    uint nonce_offset,
    uint nonce_u32,
    uint off
) {
    ulong v = 0UL;
    for (uint i = 0u; i < 8u; ++i) {
        uint ix = off + i;
        if (ix >= blob_len) break;
        v |= ((ulong)bn_blob_byte_overlay(blob, blob_len, nonce_offset, nonce_u32, ix)) << (i * 8u);
    }
    return v;
}

inline ulong bn_global_load_u64_repeat(__global const uchar* p, uint len, uint off) {
    if (len == 0u) {
        return 0UL;
    }

    ulong v = 0UL;
    for (uint i = 0u; i < 8u; ++i) {
        uint ix = (off + i) % len;
        v |= ((ulong)p[ix]) << (8u * i);
    }
    return v;
}

inline ulong bn_dataset_read_mix_fast(
    __global const ulong* dataset64,
    uint dataset_words,
    ulong addr
) {
    if (dataset_words == 0u) {
        return 0UL;
    }

    ulong a0 = addr ^ bn_rotl64(addr, 17u) ^ BN_U64_C(0x9E3779B97F4A7C15);
    uint base0 = (uint)(a0 % (ulong)dataset_words);
    uint base1 = (base0 + 8u + (uint)((addr >> 29) & 15UL)) % dataset_words;

    ulong v0 = dataset64[(base0 + 0u) % dataset_words];
    ulong v1 = dataset64[(base0 + 1u) % dataset_words];
    ulong v2 = dataset64[(base0 + 2u) % dataset_words];
    ulong v3 = dataset64[(base0 + 3u) % dataset_words];
    ulong u0 = dataset64[(base1 + 0u) % dataset_words];
    ulong u1 = dataset64[(base1 + 1u) % dataset_words];

    ulong m0 = bn_mix64(v0 ^ bn_rotl64(v2, 11u) ^ addr);
    ulong m1 = bn_mix64(v1 ^ bn_rotr64(v3, 13u) ^ bn_rotr64(addr, 7u));
    ulong m2 = bn_mix64(u0 ^ bn_rotl64(u1, 19u) ^ bn_rotl64(addr, 23u));

    return bn_avalanche64(m0 ^ bn_rotl64(m1, (uint)(addr & 31u)) ^ bn_rotr64(m2, (uint)((addr >> 37) & 31UL)));
}

inline ulong bn_hash_balance_penalty(__private const ulong hv[4]) {
    int pc0 = (int)bn_popcount64(hv[0]);
    int pc1 = (int)bn_popcount64(hv[1]);
    int pc2 = (int)bn_popcount64(hv[2]);
    int pc3 = (int)bn_popcount64(hv[3]);

    int d0 = pc0 - 32; if (d0 < 0) d0 = -d0;
    int d1 = pc1 - 32; if (d1 < 0) d1 = -d1;
    int d2 = pc2 - 32; if (d2 < 0) d2 = -d2;
    int d3 = pc3 - 32; if (d3 < 0) d3 = -d3;

    ulong nib =
        ((hv[0] >> 60) & 0xFUL) ^
        ((hv[1] >> 56) & 0xFUL) ^
        ((hv[2] >> 52) & 0xFUL) ^
        ((hv[3] >> 48) & 0xFUL);

    return (((ulong)(d0 + d1 + d2 + d3)) << 18) + (nib << 12);
}

inline void bn_compute_tail_ensemble(
    __private const ulong hv[4],
    __private ulong* tail_best,
    __private ulong* tail_consensus,
    __private ulong* tail_worst,
    __private uint* disagreement_q8
) {
    ulong t0 = hv[3];
    ulong t1 = bn_mix64(
        hv[0] ^
        bn_rotl64(hv[1], 11u) ^
        bn_rotr64(hv[2], 7u) ^
        BN_U64_C(0x9E3779B97F4A7C15)
    );
    ulong t2 = bn_mix64(
        hv[1] ^
        bn_rotl64(hv[2], 17u) ^
        bn_rotr64(hv[3], 13u) ^
        BN_U64_C(0xD1B54A32D192ED03)
    );
    ulong t3 = bn_mix64(
        hv[2] ^
        bn_rotl64(hv[0], 5u) ^
        bn_rotr64(hv[3], 19u) ^
        BN_U64_C(0x94D049BB133111EB)
    );

    ulong mn = bn_min_u64(bn_min_u64(t0, t1), bn_min_u64(t2, t3));
    ulong mx = bn_max_u64(bn_max_u64(t0, t1), bn_max_u64(t2, t3));
    ulong md = bn_median3_u64(t0, t1, t2);
    ulong spread = mx - mn;
    uint spread_q8 = bn_min_u32(255u, (uint)(spread >> 53));

    *tail_best = mn;
    *tail_consensus = mn + ((md - mn) >> 1);
    *tail_worst = mx;
    *disagreement_q8 = spread_q8;
}

inline ulong bn_soft_pass_tail(ulong tail_best, ulong tail_consensus) {
    return tail_best + ((tail_consensus - tail_best) >> 1);
}

inline ulong bn_compose_rank_score(
    __private const ulong hv[4],
    ulong soft_tail,
    uint disagreement_q8,
    uint rank_quality,
    uint credit_quality,
    uint confidence_quality,
    uint stage_class
) {
    ulong score = bn_rank_score(soft_tail, rank_quality);

    ulong mix0 = hv[0] ^ bn_rotl64(hv[1], 9u) ^ bn_rotr64(hv[2], 13u);
    ulong mix1 = hv[1] ^ bn_rotl64(hv[2], 7u) ^ bn_rotr64(hv[0], 11u);
    ulong secondary = (mix0 >> 44) & 0xFFFFFUL;
    ulong spread = (ulong)(bn_popcount64(mix1) & 0xFFu);

    score = bn_add_penalty_sat(score, (secondary << 8) | spread);
    score = bn_add_penalty_sat(score, bn_hash_balance_penalty(hv));
    score = bn_add_penalty_sat(score, ((ulong)disagreement_q8) << 28);

    if (confidence_quality < 32u) {
        score = bn_add_penalty_sat(score, ((ulong)(32u - confidence_quality)) << 26);
    }

    if (stage_class == BN_STAGE_NEAR) {
        score = bn_add_penalty_sat(score, BN_U64_C(1) << 52);
    }

    score = bn_apply_credit_bonus(score, credit_quality, confidence_quality);
    return score;
}

inline uint bn_effective_local_topk(
    uint job_age_ms,
    uint verify_pressure_q8,
    uint submit_pressure_q8,
    uint stale_risk_q8
) {
    uint topk = BN_LOCAL_TOPK;
    uint pressure = bn_max_u32(verify_pressure_q8, bn_max_u32(submit_pressure_q8, stale_risk_q8));

    if (job_age_ms < 250u && pressure < 96u) {
        return bn_max_u32(1u, bn_min_u32((uint)BN_LOCAL_TOPK, topk));
    }

    if (pressure >= 240u || job_age_ms >= 3600u) {
        topk = bn_max_u32(8u, topk / 2u);
    } else if (pressure >= 192u || job_age_ms >= 2800u) {
        topk = bn_max_u32(12u, topk - 8u);
    } else if (pressure >= 128u || job_age_ms >= 1800u) {
        topk = bn_max_u32(16u, topk - 4u);
    }

    return bn_max_u32(1u, bn_min_u32((uint)BN_LOCAL_TOPK, topk));
}

inline uint bn_effective_local_near_limit(
    uint effective_topk,
    uint job_age_ms,
    uint verify_pressure_q8,
    uint submit_pressure_q8,
    uint stale_risk_q8
) {
    uint pressure = bn_max_u32(verify_pressure_q8, bn_max_u32(submit_pressure_q8, stale_risk_q8));

    if (effective_topk <= 1u) {
        return 0u;
    }

    if (job_age_ms < 250u && pressure < 96u) {
        return bn_min_u32(bn_max_u32(2u, effective_topk / 2u), effective_topk - 1u);
    }

    if (pressure >= 248u || job_age_ms >= 3800u) {
        return 0u;
    }
    if (pressure >= 192u || job_age_ms >= 3000u) {
        return 1u;
    }
    if (pressure >= 128u || job_age_ms >= 1800u) {
        return bn_min_u32(2u, effective_topk - 1u);
    }
    return bn_min_u32(bn_max_u32(2u, effective_topk / 2u), effective_topk - 1u);
}

inline ulong bn_local_pick_penalty(
    uint cand_class,
    uint cand_bucket,
    uint cand_tailbin,
    __private const uint* chosen_bucket,
    __private const uchar* chosen_tailbin,
    uint selected_count
) {
    uint dup_cell = 0u;
    uint dup_bucket = 0u;
    uint dup_bin = 0u;

    for (uint j = 0u; j < selected_count; ++j) {
        if (chosen_bucket[j] == cand_bucket && (uint)chosen_tailbin[j] == cand_tailbin) {
            ++dup_cell;
        } else {
            if (chosen_bucket[j] == cand_bucket) ++dup_bucket;
            if ((uint)chosen_tailbin[j] == cand_tailbin) ++dup_bin;
        }
    }

    ulong penalty = 0UL;
    penalty += ((ulong)dup_cell) << 42;
    penalty += ((ulong)dup_bucket) << 36;
    penalty += ((ulong)dup_bin) << 32;

    if (cand_class == BN_STAGE_NEAR) {
        penalty += BN_U64_C(1) << 48;
    }

    return penalty;
}

// ============================================================
// RANDOMX-STRUCTURED PREDICTOR PRIMITIVES
// ============================================================

/*
 * BLAKE2b is used by consensus RandomX for the input seed, chained
 * register-file seeds, and the final 256-bit digest. These routines implement
 * that primitive directly in OpenCL C and support the nonce overlay without
 * copying the input blob.
 */
__constant ulong BN_BLAKE2B_IV[8] = {
    BN_U64_C(0x6a09e667f3bcc908), BN_U64_C(0xbb67ae8584caa73b),
    BN_U64_C(0x3c6ef372fe94f82b), BN_U64_C(0xa54ff53a5f1d36f1),
    BN_U64_C(0x510e527fade682d1), BN_U64_C(0x9b05688c2b3e6c1f),
    BN_U64_C(0x1f83d9abfb41bd6b), BN_U64_C(0x5be0cd19137e2179)
};

__constant uchar BN_BLAKE2B_SIGMA[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 },
    {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4 },
    { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8 },
    { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13 },
    { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9 },
    {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11 },
    {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10 },
    { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5 },
    {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 }
};

inline void bn_blake2b_g(
    __private ulong* a,
    __private ulong* b,
    __private ulong* c,
    __private ulong* d,
    ulong x,
    ulong y
) {
    *a = *a + *b + x;
    *d = bn_rotr64(*d ^ *a, 32u);
    *c = *c + *d;
    *b = bn_rotr64(*b ^ *c, 24u);
    *a = *a + *b + y;
    *d = bn_rotr64(*d ^ *a, 16u);
    *c = *c + *d;
    *b = bn_rotr64(*b ^ *c, 63u);
}

inline void bn_blake2b_compress(
    __private ulong h[8],
    __private const ulong m[16],
    ulong t0,
    ulong t1,
    uint is_last
) {
    ulong v[16];

    for (uint i = 0u; i < 8u; ++i) {
        v[i] = h[i];
        v[i + 8u] = BN_BLAKE2B_IV[i];
    }

    v[12] ^= t0;
    v[13] ^= t1;
    if (is_last != 0u) {
        v[14] = ~v[14];
    }

    for (uint r = 0u; r < 12u; ++r) {
        __constant const uchar* s = BN_BLAKE2B_SIGMA[r];

        bn_blake2b_g(&v[0], &v[4], &v[ 8], &v[12], m[s[ 0]], m[s[ 1]]);
        bn_blake2b_g(&v[1], &v[5], &v[ 9], &v[13], m[s[ 2]], m[s[ 3]]);
        bn_blake2b_g(&v[2], &v[6], &v[10], &v[14], m[s[ 4]], m[s[ 5]]);
        bn_blake2b_g(&v[3], &v[7], &v[11], &v[15], m[s[ 6]], m[s[ 7]]);
        bn_blake2b_g(&v[0], &v[5], &v[10], &v[15], m[s[ 8]], m[s[ 9]]);
        bn_blake2b_g(&v[1], &v[6], &v[11], &v[12], m[s[10]], m[s[11]]);
        bn_blake2b_g(&v[2], &v[7], &v[ 8], &v[13], m[s[12]], m[s[13]]);
        bn_blake2b_g(&v[3], &v[4], &v[ 9], &v[14], m[s[14]], m[s[15]]);
    }

    for (uint i = 0u; i < 8u; ++i) {
        h[i] ^= v[i] ^ v[i + 8u];
    }
}

inline void bn_blake2b_overlay(
    __global const uchar* blob,
    uint blob_len,
    uint nonce_offset,
    uint nonce_u32,
    uint out_words,
    __private ulong outv[8]
) {
    ulong h[8];
    ulong m[16];
    ulong t0 = 0UL;
    ulong t1 = 0UL;
    uint out_bytes = out_words * 8u;
    uint blocks = bn_max_u32(1u, (blob_len + 127u) >> 7);

    for (uint i = 0u; i < 8u; ++i) {
        h[i] = BN_BLAKE2B_IV[i];
    }
    h[0] ^= (BN_U64_C(0x01010000) ^ (ulong)out_bytes);

    for (uint b = 0u; b < blocks; ++b) {
        uint off = b << 7;
        uint block_bytes = (off < blob_len) ? bn_min_u32(128u, blob_len - off) : 0u;

        for (uint i = 0u; i < 16u; ++i) {
            m[i] = bn_blob_load_u64_overlay(
                blob,
                blob_len,
                nonce_offset,
                nonce_u32,
                off + (i << 3)
            );
        }

        ulong prev = t0;
        t0 += (ulong)block_bytes;
        if (t0 < prev) {
            ++t1;
        }

        bn_blake2b_compress(h, m, t0, t1, (b + 1u == blocks) ? 1u : 0u);
    }

    for (uint i = 0u; i < 8u; ++i) {
        outv[i] = (i < out_words) ? h[i] : 0UL;
    }
}

inline void bn_blake2b_private_words(
    __private const ulong* words,
    uint word_count,
    uint out_words,
    __private ulong outv[8]
) {
    ulong h[8];
    ulong m[16];
    ulong t0 = 0UL;
    ulong t1 = 0UL;
    uint out_bytes = out_words * 8u;
    uint blocks = bn_max_u32(1u, (word_count + 15u) >> 4);

    for (uint i = 0u; i < 8u; ++i) {
        h[i] = BN_BLAKE2B_IV[i];
    }
    h[0] ^= (BN_U64_C(0x01010000) ^ (ulong)out_bytes);

    for (uint b = 0u; b < blocks; ++b) {
        uint first = b << 4;
        uint remaining = (first < word_count) ? word_count - first : 0u;
        uint used = bn_min_u32(16u, remaining);

        for (uint i = 0u; i < 16u; ++i) {
            m[i] = (i < used) ? words[first + i] : 0UL;
        }

        ulong bytes = (ulong)(used << 3);
        ulong prev = t0;
        t0 += bytes;
        if (t0 < prev) {
            ++t1;
        }

        bn_blake2b_compress(h, m, t0, t1, (b + 1u == blocks) ? 1u : 0u);
    }

    for (uint i = 0u; i < 8u; ++i) {
        outv[i] = (i < out_words) ? h[i] : 0UL;
    }
}

__constant uchar BN_AES_SBOX[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

__constant uchar BN_AES_ISBOX[256] = {
    0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
    0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
    0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
    0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
    0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
    0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
    0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
    0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
    0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
    0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
    0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d
};

__constant uint BN_RX_AES1_KEYS[16] = {
    0x6daca553u,0x62716609u,0xdbb5552bu,0xb4f44917u,
    0x6d7caf07u,0x846a710du,0x1725d378u,0x0da1dc4eu,
    0x3f1262f1u,0x9f947ec6u,0xf4c0794fu,0x3e20e345u,
    0x6aef8135u,0xb1ba317cu,0x16314c88u,0x49169154u
};

__constant uint BN_RX_AES4_KEYS[32] = {
    0x6421aaddu,0xd1833ddbu,0x2f546d2bu,0x99e5d23fu,
    0xb20e3450u,0xb6913f55u,0x06f79d53u,0xa5dfcde5u,
    0x5c3ed904u,0x515e7bafu,0x0aa4679fu,0x171c02bfu,
    0x85623763u,0xe78f5d08u,0xcd673785u,0xd8ded291u,
    0xb5826f73u,0xe3d6a7a6u,0x3d518b6du,0x229effb4u,
    0xc7566bf3u,0x9c10b3d9u,0xe9024d4eu,0xb272b7d2u,
    0xf273c9e7u,0xf765a38bu,0x2ba9660au,0xf63befa7u,
    0x7a7cd609u,0x915839deu,0x0c06d1fdu,0xc0b0762du
};

__constant uint BN_RX_AES_HASH_STATE[16] = {
    0x92b52c0du,0x9fa856deu,0xcc82db47u,0xd7983aadu,
    0x338d996eu,0x15c7b798u,0xf59e125au,0xace78057u,
    0x6a770017u,0xae62c7d0u,0x5079506bu,0xe8a07ce4u,
    0x630a240cu,0x07ad828du,0x79a10005u,0x7e994948u
};

__constant uint BN_RX_AES_HASH_XKEYS[8] = {
    0xf6fa8389u,0x8b24949fu,0x90dc56bfu,0x06890201u,
    0x61b263d1u,0x51f4e03cu,0xee1043c6u,0xed18f99bu
};

inline uchar bn_aes_xtime(uchar x) {
    return (uchar)(((uint)x << 1) ^ (((x & (uchar)0x80u) != 0) ? 0x1bu : 0u));
}

inline void bn_aes_key_xor(
    __private uchar state[16],
    uint k0,
    uint k1,
    uint k2,
    uint k3
) {
    uint k[4] = { k0, k1, k2, k3 };
    for (uint i = 0u; i < 16u; ++i) {
        state[i] ^= (uchar)((k[i >> 2] >> ((i & 3u) << 3)) & 0xffu);
    }
}

inline void bn_aes_enc_round(
    __private uchar state[16],
    uint k0,
    uint k1,
    uint k2,
    uint k3
) {
    uchar t[16];
    uchar o[16];

    for (uint c = 0u; c < 4u; ++c) {
        for (uint r = 0u; r < 4u; ++r) {
            uint src_c = (c + r) & 3u;
            t[(c << 2) + r] = BN_AES_SBOX[state[(src_c << 2) + r]];
        }
    }

    for (uint c = 0u; c < 4u; ++c) {
        uint p = c << 2;
        uchar a0 = t[p + 0u];
        uchar a1 = t[p + 1u];
        uchar a2 = t[p + 2u];
        uchar a3 = t[p + 3u];
        uchar x0 = bn_aes_xtime(a0);
        uchar x1 = bn_aes_xtime(a1);
        uchar x2 = bn_aes_xtime(a2);
        uchar x3 = bn_aes_xtime(a3);

        o[p + 0u] = x0 ^ (x1 ^ a1) ^ a2 ^ a3;
        o[p + 1u] = a0 ^ x1 ^ (x2 ^ a2) ^ a3;
        o[p + 2u] = a0 ^ a1 ^ x2 ^ (x3 ^ a3);
        o[p + 3u] = (x0 ^ a0) ^ a1 ^ a2 ^ x3;
    }

    for (uint i = 0u; i < 16u; ++i) {
        state[i] = o[i];
    }
    bn_aes_key_xor(state, k0, k1, k2, k3);
}

inline void bn_aes_dec_round(
    __private uchar state[16],
    uint k0,
    uint k1,
    uint k2,
    uint k3
) {
    uchar t[16];
    uchar o[16];

    for (uint c = 0u; c < 4u; ++c) {
        for (uint r = 0u; r < 4u; ++r) {
            uint src_c = (c + 4u - r) & 3u;
            t[(c << 2) + r] = BN_AES_ISBOX[state[(src_c << 2) + r]];
        }
    }

    for (uint c = 0u; c < 4u; ++c) {
        uint p = c << 2;
        uchar a0 = t[p + 0u];
        uchar a1 = t[p + 1u];
        uchar a2 = t[p + 2u];
        uchar a3 = t[p + 3u];
        uchar a0x2 = bn_aes_xtime(a0);
        uchar a1x2 = bn_aes_xtime(a1);
        uchar a2x2 = bn_aes_xtime(a2);
        uchar a3x2 = bn_aes_xtime(a3);
        uchar a0x4 = bn_aes_xtime(a0x2);
        uchar a1x4 = bn_aes_xtime(a1x2);
        uchar a2x4 = bn_aes_xtime(a2x2);
        uchar a3x4 = bn_aes_xtime(a3x2);
        uchar a0x8 = bn_aes_xtime(a0x4);
        uchar a1x8 = bn_aes_xtime(a1x4);
        uchar a2x8 = bn_aes_xtime(a2x4);
        uchar a3x8 = bn_aes_xtime(a3x4);

        uchar a0x9  = a0x8 ^ a0;
        uchar a0x11 = a0x8 ^ a0x2 ^ a0;
        uchar a0x13 = a0x8 ^ a0x4 ^ a0;
        uchar a0x14 = a0x8 ^ a0x4 ^ a0x2;
        uchar a1x9  = a1x8 ^ a1;
        uchar a1x11 = a1x8 ^ a1x2 ^ a1;
        uchar a1x13 = a1x8 ^ a1x4 ^ a1;
        uchar a1x14 = a1x8 ^ a1x4 ^ a1x2;
        uchar a2x9  = a2x8 ^ a2;
        uchar a2x11 = a2x8 ^ a2x2 ^ a2;
        uchar a2x13 = a2x8 ^ a2x4 ^ a2;
        uchar a2x14 = a2x8 ^ a2x4 ^ a2x2;
        uchar a3x9  = a3x8 ^ a3;
        uchar a3x11 = a3x8 ^ a3x2 ^ a3;
        uchar a3x13 = a3x8 ^ a3x4 ^ a3;
        uchar a3x14 = a3x8 ^ a3x4 ^ a3x2;

        o[p + 0u] = a0x14 ^ a1x11 ^ a2x13 ^ a3x9;
        o[p + 1u] = a0x9  ^ a1x14 ^ a2x11 ^ a3x13;
        o[p + 2u] = a0x13 ^ a1x9  ^ a2x14 ^ a3x11;
        o[p + 3u] = a0x11 ^ a1x13 ^ a2x9  ^ a3x14;
    }

    for (uint i = 0u; i < 16u; ++i) {
        state[i] = o[i];
    }
    bn_aes_key_xor(state, k0, k1, k2, k3);
}

inline void bn_store_u64_le_p(__private uchar* p, ulong v) {
    for (uint i = 0u; i < 8u; ++i) {
        p[i] = (uchar)((v >> (i << 3)) & 0xffUL);
    }
}

inline ulong bn_load_u64_le_p(__private const uchar* p) {
    ulong v = 0UL;
    for (uint i = 0u; i < 8u; ++i) {
        v |= ((ulong)p[i]) << (i << 3);
    }
    return v;
}

inline void bn_rx_words_to_aes_state(
    __private const ulong words[8],
    __private uchar state[64]
) {
    for (uint i = 0u; i < 8u; ++i) {
        bn_store_u64_le_p(state + (i << 3), words[i]);
    }
}

inline void bn_rx_aes_state_to_words(
    __private const uchar state[64],
    __private ulong words[8]
) {
    for (uint i = 0u; i < 8u; ++i) {
        words[i] = bn_load_u64_le_p(state + (i << 3));
    }
}

inline void bn_rx_aes1_step(__private uchar state[64]) {
    bn_aes_dec_round(state +  0u, BN_RX_AES1_KEYS[ 0], BN_RX_AES1_KEYS[ 1], BN_RX_AES1_KEYS[ 2], BN_RX_AES1_KEYS[ 3]);
    bn_aes_enc_round(state + 16u, BN_RX_AES1_KEYS[ 4], BN_RX_AES1_KEYS[ 5], BN_RX_AES1_KEYS[ 6], BN_RX_AES1_KEYS[ 7]);
    bn_aes_dec_round(state + 32u, BN_RX_AES1_KEYS[ 8], BN_RX_AES1_KEYS[ 9], BN_RX_AES1_KEYS[10], BN_RX_AES1_KEYS[11]);
    bn_aes_enc_round(state + 48u, BN_RX_AES1_KEYS[12], BN_RX_AES1_KEYS[13], BN_RX_AES1_KEYS[14], BN_RX_AES1_KEYS[15]);
}

inline void bn_rx_aes4_step(__private uchar state[64]) {
    for (uint r = 0u; r < 4u; ++r) {
        uint a = r << 2;
        uint b = (r + 4u) << 2;
        bn_aes_dec_round(state +  0u, BN_RX_AES4_KEYS[a + 0u], BN_RX_AES4_KEYS[a + 1u], BN_RX_AES4_KEYS[a + 2u], BN_RX_AES4_KEYS[a + 3u]);
        bn_aes_enc_round(state + 16u, BN_RX_AES4_KEYS[a + 0u], BN_RX_AES4_KEYS[a + 1u], BN_RX_AES4_KEYS[a + 2u], BN_RX_AES4_KEYS[a + 3u]);
        bn_aes_dec_round(state + 32u, BN_RX_AES4_KEYS[b + 0u], BN_RX_AES4_KEYS[b + 1u], BN_RX_AES4_KEYS[b + 2u], BN_RX_AES4_KEYS[b + 3u]);
        bn_aes_enc_round(state + 48u, BN_RX_AES4_KEYS[b + 0u], BN_RX_AES4_KEYS[b + 1u], BN_RX_AES4_KEYS[b + 2u], BN_RX_AES4_KEYS[b + 3u]);
    }
}

inline void bn_rx_aes_hash_scratch(
    __private const ulong scratch[BN_RX_PREDICT_SCRATCH_WORDS],
    __private ulong outv[8]
) {
    uchar state[64];

    for (uint lane = 0u; lane < 4u; ++lane) {
        uint k = lane << 2;
        for (uint w = 0u; w < 4u; ++w) {
            uint kw = BN_RX_AES_HASH_STATE[k + w];
            uint p = (lane << 4) + (w << 2);
            state[p + 0u] = (uchar)(kw & 0xffu);
            state[p + 1u] = (uchar)((kw >> 8) & 0xffu);
            state[p + 2u] = (uchar)((kw >> 16) & 0xffu);
            state[p + 3u] = (uchar)((kw >> 24) & 0xffu);
        }
    }

    for (uint base = 0u; base < BN_RX_PREDICT_SCRATCH_WORDS; base += 8u) {
        for (uint lane = 0u; lane < 4u; ++lane) {
            ulong lo = scratch[base + (lane << 1) + 0u];
            ulong hi = scratch[base + (lane << 1) + 1u];
            uint k0 = (uint)lo;
            uint k1 = (uint)(lo >> 32);
            uint k2 = (uint)hi;
            uint k3 = (uint)(hi >> 32);

            if ((lane & 1u) == 0u) {
                bn_aes_enc_round(state + (lane << 4), k0, k1, k2, k3);
            } else {
                bn_aes_dec_round(state + (lane << 4), k0, k1, k2, k3);
            }
        }
    }

    for (uint xr = 0u; xr < 2u; ++xr) {
        uint k = xr << 2;
        for (uint lane = 0u; lane < 4u; ++lane) {
            if ((lane & 1u) == 0u) {
                bn_aes_enc_round(
                    state + (lane << 4),
                    BN_RX_AES_HASH_XKEYS[k + 0u],
                    BN_RX_AES_HASH_XKEYS[k + 1u],
                    BN_RX_AES_HASH_XKEYS[k + 2u],
                    BN_RX_AES_HASH_XKEYS[k + 3u]
                );
            } else {
                bn_aes_dec_round(
                    state + (lane << 4),
                    BN_RX_AES_HASH_XKEYS[k + 0u],
                    BN_RX_AES_HASH_XKEYS[k + 1u],
                    BN_RX_AES_HASH_XKEYS[k + 2u],
                    BN_RX_AES_HASH_XKEYS[k + 3u]
                );
            }
        }
    }

    bn_rx_aes_state_to_words(state, outv);
}

// RandomX v1 opcode ceilings. The frequencies total exactly 256.
#define BN_RX_CEIL_IADD_RS  16u
#define BN_RX_CEIL_IADD_M   23u
#define BN_RX_CEIL_ISUB_R   39u
#define BN_RX_CEIL_ISUB_M   46u
#define BN_RX_CEIL_IMUL_R   62u
#define BN_RX_CEIL_IMUL_M   66u
#define BN_RX_CEIL_IMULH_R  70u
#define BN_RX_CEIL_IMULH_M  71u
#define BN_RX_CEIL_ISMULH_R 75u
#define BN_RX_CEIL_ISMULH_M 76u
#define BN_RX_CEIL_IMUL_RCP 84u
#define BN_RX_CEIL_INEG_R   86u
#define BN_RX_CEIL_IXOR_R   101u
#define BN_RX_CEIL_IXOR_M   106u
#define BN_RX_CEIL_IROR_R   114u
#define BN_RX_CEIL_IROL_R   116u
#define BN_RX_CEIL_ISWAP_R  120u
#define BN_RX_CEIL_FSWAP_R  124u
#define BN_RX_CEIL_FADD_R   140u
#define BN_RX_CEIL_FADD_M   145u
#define BN_RX_CEIL_FSUB_R   161u
#define BN_RX_CEIL_FSUB_M   166u
#define BN_RX_CEIL_FSCAL_R  172u
#define BN_RX_CEIL_FMUL_R   204u
#define BN_RX_CEIL_FDIV_M   208u
#define BN_RX_CEIL_FSQRT_R  214u
#define BN_RX_CEIL_CBRANCH  239u
#define BN_RX_CEIL_CFROUND  240u
#define BN_RX_CEIL_ISTORE   256u

inline ulong bn_smulh64(ulong a, ulong b) {
    return (ulong)mul_hi((long)a, (long)b);
}

inline uint bn_is_zero_or_power_of_two_u32(uint x) {
    return ((x & (x - 1u)) == 0u) ? 1u : 0u;
}

inline ulong bn_rx_reciprocal(uint divisor) {
    if (divisor == 0u || bn_is_zero_or_power_of_two_u32(divisor) != 0u) {
        return 0UL;
    }

    ulong d = (ulong)divisor;
    ulong p2exp63 = BN_U64_C(0x8000000000000000);
    ulong q = p2exp63 / d;
    ulong r = p2exp63 % d;
    uint shift = 32u - clz(divisor);
    return (q << shift) + ((r << shift) / d);
}

inline ulong bn_rx_small_positive_float_bits(ulong entropy) {
    ulong exponent = entropy >> 59;
    ulong mantissa = entropy & BN_U64_C(0x000fffffffffffff);
    exponent = (exponent + 1023UL) & 0x7ffUL;
    return (exponent << 52) | mantissa;
}

inline ulong bn_rx_float_mask(ulong entropy) {
    ulong exponent = BN_U64_C(0x300) | ((entropy >> 60) << 4);
    return (entropy & BN_U64_C(0x3fffff)) | (exponent << 52);
}

inline ulong bn_rx_swap32_halves(ulong x) {
    return (x << 32) | (x >> 32);
}

inline ulong bn_rx_fp_from_i32(uint x) {
#if BN_RX_HAVE_FP64
    return as_ulong((double)((int)x));
#else
    return bn_mix64((ulong)x ^ BN_U64_C(0x3ff0000000000000));
#endif
}

inline ulong bn_rx_fp_e_from_i32(uint x, ulong e_mask) {
#if BN_RX_HAVE_FP64
    ulong bits = as_ulong((double)((int)x));
    bits &= BN_U64_C(0x00ffffffffffffff);
    bits |= e_mask;
    return bits;
#else
    return (bn_mix64((ulong)x) & BN_U64_C(0x000fffffffffffff)) | e_mask;
#endif
}

inline ulong bn_rx_fp_add(ulong a, ulong b) {
#if BN_RX_HAVE_FP64
    return as_ulong(as_double(a) + as_double(b));
#else
    return bn_mix64(a + bn_rotl64(b, 17u));
#endif
}

inline ulong bn_rx_fp_sub(ulong a, ulong b) {
#if BN_RX_HAVE_FP64
    return as_ulong(as_double(a) - as_double(b));
#else
    return bn_mix64(a - bn_rotl64(b, 11u));
#endif
}

inline ulong bn_rx_fp_mul(ulong a, ulong b) {
#if BN_RX_HAVE_FP64
    return as_ulong(as_double(a) * as_double(b));
#else
    return bn_mix64(a ^ bn_mulh64(a | 1UL, b | 1UL) ^ b);
#endif
}

inline ulong bn_rx_fp_div(ulong a, ulong b) {
#if BN_RX_HAVE_FP64
    double d = as_double(b);
    if (d == 0.0) {
        d = 1.0;
    }
    return as_ulong(as_double(a) / d);
#else
    return bn_mix64(a ^ bn_rotr64(b | 1UL, 23u));
#endif
}

inline ulong bn_rx_fp_sqrt(ulong a) {
#if BN_RX_HAVE_FP64
    double d = as_double(a);
    if (d < 0.0) {
        d = -d;
    }
    return as_ulong(sqrt(d));
#else
    return bn_mix64(a ^ (a >> 1));
#endif
}

inline uint bn_rx_scratch_index(ulong byte_address, uint mod_mem, uint force_l3) {
    uint words;

    if (force_l3 != 0u) {
        words = BN_RX_PREDICT_SCRATCH_WORDS;
    } else if (mod_mem != 0u) {
        words = BN_RX_PREDICT_SCRATCH_WORDS >> 3;
    } else {
        words = BN_RX_PREDICT_SCRATCH_WORDS >> 1;
    }

    return (uint)((byte_address >> 3) & (ulong)(words - 1u));
}

inline ulong bn_rx_scratch_load(
    __private const ulong scratch[BN_RX_PREDICT_SCRATCH_WORDS],
    ulong address_reg,
    uint imm32,
    uint mod_mem,
    uint force_l3
) {
    ulong address = address_reg + (ulong)(long)(int)imm32;
    return scratch[bn_rx_scratch_index(address, mod_mem, force_l3)];
}

inline void bn_rx_scratch_store(
    __private ulong scratch[BN_RX_PREDICT_SCRATCH_WORDS],
    ulong address_reg,
    uint imm32,
    uint mod_mem,
    uint force_l3,
    ulong value
) {
    ulong address = address_reg + (ulong)(long)(int)imm32;
    scratch[bn_rx_scratch_index(address, mod_mem, force_l3)] = value;
}

inline void bn_rx_generate_program(
    __private uchar aes_state[64],
    __private ulong config[16],
    __private ulong program[BN_RX_PREDICT_PROGRAM_SIZE]
) {
    ulong block[8];

    for (uint base = 0u; base < 16u; base += 8u) {
        bn_rx_aes4_step(aes_state);
        bn_rx_aes_state_to_words(aes_state, block);
        for (uint i = 0u; i < 8u; ++i) {
            config[base + i] = block[i];
        }
    }

    for (uint base = 0u; base < BN_RX_PREDICT_PROGRAM_SIZE; base += 8u) {
        bn_rx_aes4_step(aes_state);
        bn_rx_aes_state_to_words(aes_state, block);
        for (uint i = 0u; i < 8u && base + i < BN_RX_PREDICT_PROGRAM_SIZE; ++i) {
            program[base + i] = block[i];
        }
    }
}

inline void bn_rx_build_branch_targets(
    __private const ulong program[BN_RX_PREDICT_PROGRAM_SIZE],
    __private ushort branch_target[BN_RX_PREDICT_PROGRAM_SIZE]
) {
    ushort last_write[8];

    for (uint r = 0u; r < 8u; ++r) {
        last_write[r] = (ushort)0xffffu;
    }

    for (uint pc = 0u; pc < BN_RX_PREDICT_PROGRAM_SIZE; ++pc) {
        ulong inst = program[pc];
        uint opcode = (uint)(inst & 0xffUL);
        uint dst = (uint)((inst >> 8) & 7UL);
        uint src = (uint)((inst >> 16) & 7UL);
        uint imm32 = (uint)(inst >> 32);

        branch_target[pc] = (ushort)0xffffu;

        if (opcode < BN_RX_CEIL_IROL_R) {
            if (opcode >= BN_RX_CEIL_IMUL_RCP - 8u && opcode < BN_RX_CEIL_IMUL_RCP) {
                if (bn_is_zero_or_power_of_two_u32(imm32) == 0u) {
                    last_write[dst] = (ushort)pc;
                }
            } else {
                last_write[dst] = (ushort)pc;
            }
        } else if (opcode < BN_RX_CEIL_ISWAP_R) {
            if (src != dst) {
                last_write[dst] = (ushort)pc;
                last_write[src] = (ushort)pc;
            }
        } else if (opcode >= BN_RX_CEIL_FSQRT_R && opcode < BN_RX_CEIL_CBRANCH) {
            branch_target[pc] = last_write[dst];
            for (uint r = 0u; r < 8u; ++r) {
                last_write[r] = (ushort)pc;
            }
        }
    }
}

inline void bn_rx_execute_program(
    __private const ulong program[BN_RX_PREDICT_PROGRAM_SIZE],
    __private const ushort branch_target[BN_RX_PREDICT_PROGRAM_SIZE],
    __private ulong r[8],
    __private ulong f[8],
    __private ulong e[8],
    __private const ulong a[8],
    __private const ulong e_mask[2],
    __private ulong scratch[BN_RX_PREDICT_SCRATCH_WORDS],
    __private uint* fprc
) {
    uint pc = 0u;
    uint steps = 0u;

    while (pc < BN_RX_PREDICT_PROGRAM_SIZE && steps < BN_RX_PREDICT_MAX_BRANCH_STEPS) {
        ulong inst = program[pc];
        uint opcode = (uint)(inst & 0xffUL);
        uint dst = (uint)((inst >> 8) & 7UL);
        uint src = (uint)((inst >> 16) & 7UL);
        uint mod = (uint)((inst >> 24) & 0xffUL);
        uint imm32 = (uint)(inst >> 32);
        uint next_pc = pc + 1u;

        if (opcode < BN_RX_CEIL_IADD_RS) {
            uint shift = (mod >> 2) & 3u;
            ulong displacement = (dst == 5u) ? (ulong)(long)(int)imm32 : 0UL;
            r[dst] += (r[src] << shift) + displacement;
        } else if (opcode < BN_RX_CEIL_IADD_M) {
            r[dst] += bn_rx_scratch_load(scratch, (src == dst) ? 0UL : r[src], imm32, mod & 3u, (src == dst) ? 1u : 0u);
        } else if (opcode < BN_RX_CEIL_ISUB_R) {
            r[dst] -= (src == dst) ? (ulong)(long)(int)imm32 : r[src];
        } else if (opcode < BN_RX_CEIL_ISUB_M) {
            r[dst] -= bn_rx_scratch_load(scratch, (src == dst) ? 0UL : r[src], imm32, mod & 3u, (src == dst) ? 1u : 0u);
        } else if (opcode < BN_RX_CEIL_IMUL_R) {
            r[dst] *= (src == dst) ? (ulong)(long)(int)imm32 : r[src];
        } else if (opcode < BN_RX_CEIL_IMUL_M) {
            r[dst] *= bn_rx_scratch_load(scratch, (src == dst) ? 0UL : r[src], imm32, mod & 3u, (src == dst) ? 1u : 0u);
        } else if (opcode < BN_RX_CEIL_IMULH_R) {
            r[dst] = bn_mulh64(r[dst], r[src]);
        } else if (opcode < BN_RX_CEIL_IMULH_M) {
            r[dst] = bn_mulh64(r[dst], bn_rx_scratch_load(scratch, (src == dst) ? 0UL : r[src], imm32, mod & 3u, (src == dst) ? 1u : 0u));
        } else if (opcode < BN_RX_CEIL_ISMULH_R) {
            r[dst] = bn_smulh64(r[dst], r[src]);
        } else if (opcode < BN_RX_CEIL_ISMULH_M) {
            r[dst] = bn_smulh64(r[dst], bn_rx_scratch_load(scratch, (src == dst) ? 0UL : r[src], imm32, mod & 3u, (src == dst) ? 1u : 0u));
        } else if (opcode < BN_RX_CEIL_IMUL_RCP) {
            ulong reciprocal = bn_rx_reciprocal(imm32);
            if (reciprocal != 0UL) {
                r[dst] *= reciprocal;
            }
        } else if (opcode < BN_RX_CEIL_INEG_R) {
            r[dst] = ~r[dst] + 1UL;
        } else if (opcode < BN_RX_CEIL_IXOR_R) {
            r[dst] ^= (src == dst) ? (ulong)(long)(int)imm32 : r[src];
        } else if (opcode < BN_RX_CEIL_IXOR_M) {
            r[dst] ^= bn_rx_scratch_load(scratch, (src == dst) ? 0UL : r[src], imm32, mod & 3u, (src == dst) ? 1u : 0u);
        } else if (opcode < BN_RX_CEIL_IROR_R) {
            ulong count = (src == dst) ? (ulong)imm32 : r[src];
            r[dst] = bn_rotr64(r[dst], (uint)(count & 63UL));
        } else if (opcode < BN_RX_CEIL_IROL_R) {
            ulong count = (src == dst) ? (ulong)imm32 : r[src];
            r[dst] = bn_rotl64(r[dst], (uint)(count & 63UL));
        } else if (opcode < BN_RX_CEIL_ISWAP_R) {
            if (src != dst) {
                ulong tmp = r[dst];
                r[dst] = r[src];
                r[src] = tmp;
            }
        } else if (opcode < BN_RX_CEIL_FSWAP_R) {
            uint reg = dst & 7u;
            if (reg < 4u) {
                uint q = reg << 1;
                f[q + 0u] = bn_rx_swap32_halves(f[q + 0u]);
                f[q + 1u] = bn_rx_swap32_halves(f[q + 1u]);
            } else {
                uint q = (reg - 4u) << 1;
                e[q + 0u] = bn_rx_swap32_halves(e[q + 0u]);
                e[q + 1u] = bn_rx_swap32_halves(e[q + 1u]);
            }
        } else if (opcode < BN_RX_CEIL_FADD_R) {
            uint fd = (dst & 3u) << 1;
            uint fs = (src & 3u) << 1;
            f[fd + 0u] = bn_rx_fp_add(f[fd + 0u], a[fs + 0u]);
            f[fd + 1u] = bn_rx_fp_add(f[fd + 1u], a[fs + 1u]);
        } else if (opcode < BN_RX_CEIL_FADD_M) {
            ulong qword = bn_rx_scratch_load(scratch, r[src], imm32, mod & 3u, 0u);
            uint fd = (dst & 3u) << 1;
            f[fd + 0u] = bn_rx_fp_add(f[fd + 0u], bn_rx_fp_from_i32((uint)qword));
            f[fd + 1u] = bn_rx_fp_add(f[fd + 1u], bn_rx_fp_from_i32((uint)(qword >> 32)));
        } else if (opcode < BN_RX_CEIL_FSUB_R) {
            uint fd = (dst & 3u) << 1;
            uint fs = (src & 3u) << 1;
            f[fd + 0u] = bn_rx_fp_sub(f[fd + 0u], a[fs + 0u]);
            f[fd + 1u] = bn_rx_fp_sub(f[fd + 1u], a[fs + 1u]);
        } else if (opcode < BN_RX_CEIL_FSUB_M) {
            ulong qword = bn_rx_scratch_load(scratch, r[src], imm32, mod & 3u, 0u);
            uint fd = (dst & 3u) << 1;
            f[fd + 0u] = bn_rx_fp_sub(f[fd + 0u], bn_rx_fp_from_i32((uint)qword));
            f[fd + 1u] = bn_rx_fp_sub(f[fd + 1u], bn_rx_fp_from_i32((uint)(qword >> 32)));
        } else if (opcode < BN_RX_CEIL_FSCAL_R) {
            uint fd = (dst & 3u) << 1;
            f[fd + 0u] ^= BN_U64_C(0x80f0000000000000);
            f[fd + 1u] ^= BN_U64_C(0x80f0000000000000);
        } else if (opcode < BN_RX_CEIL_FMUL_R) {
            uint ed = (dst & 3u) << 1;
            uint as = (src & 3u) << 1;
            e[ed + 0u] = bn_rx_fp_mul(e[ed + 0u], a[as + 0u]);
            e[ed + 1u] = bn_rx_fp_mul(e[ed + 1u], a[as + 1u]);
        } else if (opcode < BN_RX_CEIL_FDIV_M) {
            ulong qword = bn_rx_scratch_load(scratch, r[src], imm32, mod & 3u, 0u);
            uint ed = (dst & 3u) << 1;
            e[ed + 0u] = bn_rx_fp_div(e[ed + 0u], bn_rx_fp_e_from_i32((uint)qword, e_mask[0]));
            e[ed + 1u] = bn_rx_fp_div(e[ed + 1u], bn_rx_fp_e_from_i32((uint)(qword >> 32), e_mask[1]));
        } else if (opcode < BN_RX_CEIL_FSQRT_R) {
            uint ed = (dst & 3u) << 1;
            e[ed + 0u] = bn_rx_fp_sqrt(e[ed + 0u]);
            e[ed + 1u] = bn_rx_fp_sqrt(e[ed + 1u]);
        } else if (opcode < BN_RX_CEIL_CBRANCH) {
            uint shift = (mod >> 4) + 8u;
            ulong condition_mask = BN_U64_C(0xff) << shift;
            ulong cimm = (ulong)(long)(int)imm32;
            cimm |= 1UL << shift;
            if (shift > 0u) {
                cimm &= ~(1UL << (shift - 1u));
            }
            r[dst] += cimm;
            if ((r[dst] & condition_mask) == 0UL) {
                ushort target = branch_target[pc];
                next_pc = (target == (ushort)0xffffu) ? 0u : (uint)target + 1u;
            }
        } else if (opcode < BN_RX_CEIL_CFROUND) {
            *fprc = (uint)(bn_rotr64(r[src], imm32 & 63u) & 3UL);
        } else {
            uint force_l3 = ((mod >> 4) >= 14u) ? 1u : 0u;
            bn_rx_scratch_store(scratch, r[dst], imm32, mod & 3u, force_l3, r[src]);
        }

        pc = next_pc;
        ++steps;
    }
}

inline void bn_rx_register_file(
    __private const ulong r[8],
    __private const ulong f[8],
    __private const ulong e[8],
    __private const ulong a_or_hash[8],
    __private ulong words[32]
) {
    for (uint i = 0u; i < 8u; ++i) {
        words[i +  0u] = r[i];
        words[i +  8u] = f[i];
        words[i + 16u] = e[i];
        words[i + 24u] = a_or_hash[i];
    }
}

/*
 * ABI-compatible RandomX-structured predictor.
 *
 * This executes the same high-level stages and opcode distribution as RandomX,
 * but with the bounded profile declared above. The full Dataset is consumed as
 * aligned 64-byte items. The seed pointer is intentionally not hashed into H:
 * in consensus RandomX the seed key influences the prebuilt Dataset, while
 * Hash512(H) is calculated from the nonce-bearing blob alone.
 */
inline void bn_gpu_dataset_prefilter_hash_fast(
    __global const uchar* blob,
    uint blob_len,
    uint nonce_offset,
    uint nonce_u32,
    __global const uchar* seed,
    uint seed_len,
    __global const ulong* dataset64,
    uint dataset_words,
    __private ulong outv[4]
) {
    if (
        blob_len == 0u ||
        blob_len > BN_MAX_BLOB_BYTES ||
        nonce_offset > blob_len ||
        (blob_len - nonce_offset) < 4u ||
        dataset_words < 8u
    ) {
        outv[0] = 0UL;
        outv[1] = 0UL;
        outv[2] = 0UL;
        outv[3] = ~0UL;
        return;
    }

    (void)seed;
    (void)seed_len;

    ulong initial_seed[8];
    uchar aes_state[64];
    ulong aes_words[8];
    ulong scratch[BN_RX_PREDICT_SCRATCH_WORDS];
    ulong config[16];
    ulong program[BN_RX_PREDICT_PROGRAM_SIZE];
    ushort branch_target[BN_RX_PREDICT_PROGRAM_SIZE];
    ulong r[8];
    ulong f[8];
    ulong e[8];
    ulong a[8];
    ulong e_mask[2];
    ulong reg_file[32];
    ulong chain_hash[8];
    ulong scratch_hash[8];
    uint fprc = 0u;
    uint dataset_items = dataset_words >> 3;

    bn_blake2b_overlay(
        blob,
        blob_len,
        nonce_offset,
        nonce_u32,
        8u,
        initial_seed
    );
    bn_rx_words_to_aes_state(initial_seed, aes_state);

    /*
     * AesGenerator1R scratchpad fill. The state after the final generated
     * 64-byte line is reused as AesGenerator4R state, matching RandomX.
     */
    for (uint base = 0u; base < BN_RX_PREDICT_SCRATCH_WORDS; base += 8u) {
        bn_rx_aes1_step(aes_state);
        bn_rx_aes_state_to_words(aes_state, aes_words);
        for (uint i = 0u; i < 8u; ++i) {
            scratch[base + i] = aes_words[i];
        }
    }

    for (uint i = 0u; i < 8u; ++i) {
        r[i] = 0UL;
        f[i] = 0UL;
        e[i] = 0UL;
        a[i] = 0UL;
    }

    for (uint program_index = 0u; program_index < BN_RX_PREDICT_PROGRAMS; ++program_index) {
        bn_rx_generate_program(aes_state, config, program);
        bn_rx_build_branch_targets(program, branch_target);

        for (uint i = 0u; i < 8u; ++i) {
            a[i] = bn_rx_small_positive_float_bits(config[i]);
        }
        e_mask[0] = bn_rx_float_mask(config[14]);
        e_mask[1] = bn_rx_float_mask(config[15]);

        uint ma = ((uint)config[8]) & ~63u;
        uint mx = (uint)config[10];
        ulong address_registers = config[12];
        uint read_reg0 = 0u + (uint)(address_registers & 1UL);
        address_registers >>= 1;
        uint read_reg1 = 2u + (uint)(address_registers & 1UL);
        address_registers >>= 1;
        uint read_reg2 = 4u + (uint)(address_registers & 1UL);
        address_registers >>= 1;
        uint read_reg3 = 6u + (uint)(address_registers & 1UL);
        uint dataset_offset_item = (uint)(config[13] % (ulong)dataset_items);

        for (uint iteration = 0u; iteration < BN_RX_PREDICT_ITERATIONS; ++iteration) {
            ulong sp_mix = r[read_reg0] ^ r[read_reg1];
            uint sp_addr0 = (uint)((sp_mix >> 3) & (ulong)(BN_RX_PREDICT_SCRATCH_WORDS - 8u));
            uint sp_addr1 = (uint)(((sp_mix >> 32) >> 3) & (ulong)(BN_RX_PREDICT_SCRATCH_WORDS - 8u));

            for (uint i = 0u; i < 8u; ++i) {
                r[i] ^= scratch[sp_addr0 + i];
            }

            for (uint i = 0u; i < 4u; ++i) {
                ulong fq = scratch[sp_addr1 + i];
                ulong eq = scratch[sp_addr1 + 4u + i];
                uint p = i << 1;
                f[p + 0u] = bn_rx_fp_from_i32((uint)fq);
                f[p + 1u] = bn_rx_fp_from_i32((uint)(fq >> 32));
                e[p + 0u] = bn_rx_fp_e_from_i32((uint)eq, e_mask[0]);
                e[p + 1u] = bn_rx_fp_e_from_i32((uint)(eq >> 32), e_mask[1]);
            }

            bn_rx_execute_program(
                program,
                branch_target,
                r,
                f,
                e,
                a,
                e_mask,
                scratch,
                &fprc
            );

            mx ^= (uint)r[read_reg2] ^ (uint)r[read_reg3];
            mx &= ~63u;

            ulong item_number =
                (((ulong)ma >> 6) + (ulong)dataset_offset_item) %
                (ulong)dataset_items;
            uint dataset_base = (uint)(item_number << 3);

            for (uint i = 0u; i < 8u; ++i) {
                ulong next_r = r[i] ^ dataset64[dataset_base + i];
                r[i] = next_r;
                scratch[sp_addr1 + i] = next_r;
                scratch[sp_addr0 + i] = f[i] ^ e[i];
            }

            uint tmp = ma;
            ma = mx;
            mx = tmp;
        }

        if (program_index + 1u < BN_RX_PREDICT_PROGRAMS) {
            bn_rx_register_file(r, f, e, a, reg_file);
            bn_blake2b_private_words(reg_file, 32u, 8u, chain_hash);
            bn_rx_words_to_aes_state(chain_hash, aes_state);
        }
    }

    bn_rx_aes_hash_scratch(scratch, scratch_hash);
    bn_rx_register_file(r, f, e, scratch_hash, reg_file);
    bn_blake2b_private_words(reg_file, 32u, 4u, chain_hash);

    outv[0] = chain_hash[0];
    outv[1] = chain_hash[1];
    outv[2] = chain_hash[2];
    outv[3] = chain_hash[3];
}

inline void bn_gpu_dataset_prefilter_hash_legacy(
    __global const uchar* blob,
    uint blob_len,
    uint nonce_offset,
    uint nonce_u32,
    __global const uchar* seed,
    uint seed_len,
    __global const ulong* dataset64,
    uint dataset_words,
    __private ulong outv[4]
) {
    if (blob_len == 0u || blob_len > BN_MAX_BLOB_BYTES || dataset_words == 0u) {
        outv[0] = 0UL;
        outv[1] = 0UL;
        outv[2] = 0UL;
        outv[3] = ~0UL;
        return;
    }

    ulong nonce64 = (ulong)nonce_u32;

    ulong s0 = BN_U64_C(0x243F6A8885A308D3) ^ nonce64 ^ ((ulong)blob_len << 32);
    ulong s1 = BN_U64_C(0x13198A2E03707344) ^ bn_rotl64(nonce64, 7u) ^ ((ulong)seed_len << 24);
    ulong s2 = BN_U64_C(0xA4093822299F31D0) ^ bn_rotr64(nonce64, 9u) ^ ((ulong)blob_len << 11);
    ulong s3 = BN_U64_C(0x082EFA98EC4E6C89) ^ bn_rotl64(nonce64, 13u) ^ ((ulong)seed_len << 19);

    ulong rng = bn_avalanche64(s0 ^ s1 ^ s2 ^ s3 ^ nonce64);
    ulong acc0 = BN_U64_C(0x9E3779B97F4A7C15) ^ nonce64;
    ulong acc1 = BN_U64_C(0xD1B54A32D192ED03) ^ bn_rotl64(nonce64, 17u);

    uint blob_words = (blob_len + 7u) >> 3;
    for (uint i = 0u; i < blob_words; ++i) {
        ulong w = bn_blob_load_u64_overlay(blob, blob_len, nonce_offset, nonce_u32, i * 8u);
        ulong ds = bn_dataset_read_mix_fast(dataset64, dataset_words, w ^ s0 ^ bn_rotl64(s1, i + 1u));
        ulong r = bn_rng_step(&rng);

        acc0 = bn_mix64(acc0 ^ w ^ ds ^ r ^ ((ulong)i << 29));
        acc1 = bn_mix64(acc1 + bn_rotl64(w, ((i * 7u) + 5u) & 63u) + bn_rotr64(ds ^ r, ((i * 3u) + 1u) & 63u));

        s0 = bn_mix64(s0 ^ acc0 ^ ds);
        s1 = bn_mix64(s1 + acc1 + bn_rotl64(acc0, 9u));
        s2 = bn_mix64(s2 ^ bn_rotl64(ds ^ s0, 13u) ^ ((ulong)i << 17) ^ r);
        s3 = bn_mix64(s3 + bn_rotr64(w ^ s1, 11u) + ((ulong)i << 32));
    }

    uint seed_words = (seed_len + 7u) >> 3;
    for (uint i = 0u; i < seed_words; ++i) {
        ulong w = bn_global_load_u64_repeat(seed, seed_len, i * 8u);
        ulong ds = bn_dataset_read_mix_fast(dataset64, dataset_words, w ^ s2 ^ bn_rotl64(s3, i + 3u));
        ulong r = bn_rng_step(&rng);

        acc0 = bn_mix64(acc0 + w + bn_rotl64(ds, 9u));
        acc1 = bn_mix64(acc1 ^ w ^ bn_rotr64(ds, 7u) ^ r);

        s0 = bn_mix64(s0 + acc0 + ds);
        s1 = bn_mix64(s1 ^ acc1 ^ bn_rotl64(w, 21u));
        s2 = bn_mix64(s2 + ds + bn_rotl64(w, 27u));
        s3 = bn_mix64(s3 ^ ds ^ bn_rotr64(w, 15u) ^ r);
    }

    for (uint i = 0u; i < BN_PREFILTER_ROUNDS_FAST; ++i) {
        ulong r0 = bn_rng_step(&rng);
        ulong r1 = bn_rng_step(&rng);

        ulong a0 = s0 ^ bn_rotl64(s2, 7u) ^ acc0 ^ r0 ^ ((ulong)i << 9);
        ulong a1 = s1 ^ bn_rotr64(s3, 11u) ^ acc1 ^ r1 ^ ((ulong)i << 15);

        ulong d0 = bn_dataset_read_mix_fast(dataset64, dataset_words, a0);
        ulong d1 = bn_dataset_read_mix_fast(dataset64, dataset_words, a1);

        ulong x0 = bn_mix64(d0 ^ bn_rotl64(d1, (i + 5u) & 63u) ^ s3 ^ acc0);
        ulong x1 = bn_mix64(d1 ^ bn_rotr64(d0, (i + 11u) & 63u) ^ s0 ^ acc1);

        ulong hi0 = bn_mulh64((s1 ^ acc0) | 1UL, d0 | 1UL);
        ulong hi1 = bn_mulh64((s2 ^ acc1) | 1UL, d1 | 1UL);

        s0 = bn_mix64(s0 + x0 + bn_rotl64(s3, 9u) + hi0);
        s1 = bn_mix64(s1 ^ x1 ^ bn_rotr64(s0, 13u) ^ hi1);
        s2 = bn_mix64(s2 + x1 + bn_rotl64(s1, 17u) + hi0);
        s3 = bn_mix64(s3 ^ x0 ^ bn_rotr64(s2, 23u) ^ hi1);

        acc0 = bn_mix64(acc0 ^ x0 ^ hi1 ^ ((ulong)i << 37));
        acc1 = bn_mix64(acc1 + x1 + hi0 + ((ulong)i << 41));

        if ((i & (BN_FAST_ABSORB_STRIDE - 1u)) == 0u) {
            uint bix = ((i >> 1) % bn_max_u32(1u, blob_words)) * 8u;
            uint six = ((i >> 1) % bn_max_u32(1u, seed_words)) * 8u;
            ulong bw = bn_blob_load_u64_overlay(blob, blob_len, nonce_offset, nonce_u32, bix);
            ulong sw = bn_global_load_u64_repeat(seed, seed_len, six);
            s0 ^= bn_mix64(bw ^ acc1);
            s2 ^= bn_mix64(sw ^ acc0);
        }
    }

    ulong f0 = s0 ^ acc0;
    ulong f1 = s1 ^ acc1;
    ulong f2 = s2 ^ bn_rotl64(acc0, 17u);
    ulong f3 = s3 ^ bn_rotr64(acc1, 19u);

    for (uint i = 0u; i < BN_FINAL_MIX_ROUNDS_FAST; ++i) {
        ulong d0 = bn_dataset_read_mix_fast(dataset64, dataset_words, f0 ^ f2 ^ nonce64 ^ ((ulong)i << 21));
        ulong d1 = bn_dataset_read_mix_fast(dataset64, dataset_words, f1 ^ f3 ^ rng ^ ((ulong)i << 27));
        ulong m  = bn_mix64(d0 ^ bn_rotl64(d1, (i * 9u + 7u) & 63u) ^ bn_rng_step(&rng));

        f0 = bn_avalanche64(f0 ^ m ^ bn_rotl64(f3, 7u));
        f1 = bn_avalanche64(f1 + m + bn_rotr64(f0, 11u));
        f2 = bn_avalanche64(f2 ^ m ^ bn_rotl64(f1, 17u));
        f3 = bn_avalanche64(f3 + m + bn_rotr64(f2, 23u));
    }

    outv[0] = bn_avalanche64(f0 ^ s1 ^ bn_rotl64(f2, 9u));
    outv[1] = bn_avalanche64(f1 ^ s2 ^ bn_rotr64(f3, 7u));
    outv[2] = bn_avalanche64(f2 ^ s3 ^ bn_rotl64(f0, 13u));
    outv[3] = bn_avalanche64(f3 ^ s0 ^ bn_rotr64(f1, 17u));
}

inline void bn_stage_candidate_local(
    uint lid,
    uint nonce_u32,
    __private const ulong hv[4],
    ulong rank_score,
    uint stage_class,
    uint tune_bucket,
    uint tune_tail_bin,
    uint rank_quality,
    uint threshold_quality,
    __local ulong* l_score,
    __local uint* l_nonce,
    __local ulong* l_h0,
    __local ulong* l_h1,
    __local ulong* l_h2,
    __local ulong* l_h3,
    __local uint* l_bucket,
    __local uchar* l_rankq,
    __local uchar* l_threshq,
    __local uchar* l_tailbin,
    __local uchar* l_class
) {
    if (lid < BN_LOCAL_STAGE_SIZE) {
        if (stage_class != BN_STAGE_REJECT) {
            l_score[lid] = rank_score;
            l_nonce[lid] = nonce_u32;
            l_h0[lid] = hv[0];
            l_h1[lid] = hv[1];
            l_h2[lid] = hv[2];
            l_h3[lid] = hv[3];
            l_bucket[lid] = tune_bucket;
            l_rankq[lid] = (uchar)(rank_quality & 0xFFu);
            l_threshq[lid] = (uchar)(threshold_quality & 0xFFu);
            l_tailbin[lid] = (uchar)(tune_tail_bin & 0xFFu);
            l_class[lid] = (uchar)(stage_class & 0xFFu);
        } else {
            l_score[lid] = ~0UL;
            l_nonce[lid] = 0u;
            l_h0[lid] = 0UL;
            l_h1[lid] = 0UL;
            l_h2[lid] = 0UL;
            l_h3[lid] = ~0UL;
            l_bucket[lid] = 0u;
            l_rankq[lid] = (uchar)BN_TUNE_NEUTRAL;
            l_threshq[lid] = (uchar)BN_TUNE_NEUTRAL;
            l_tailbin[lid] = (uchar)0u;
            l_class[lid] = (uchar)BN_STAGE_REJECT;
        }
    }
}

inline void bn_rank_and_stage(
    uint lid,
    uint nonce_u32,
    __private const ulong hv[4],
    ulong target64,
    __global const uchar* seed_tune,
    uint seed_tune_buckets,
    uint seed_tune_tail_bins,
    __global const uchar* job_tune,
    uint job_tune_buckets,
    uint job_tune_tail_bins,
    uint job_age_ms,
    uint verify_pressure_q8,
    uint submit_pressure_q8,
    uint stale_risk_q8,
    __local ulong* l_score,
    __local uint* l_nonce,
    __local ulong* l_h0,
    __local ulong* l_h1,
    __local ulong* l_h2,
    __local ulong* l_h3,
    __local uint* l_bucket,
    __local uchar* l_rankq,
    __local uchar* l_threshq,
    __local uchar* l_tailbin,
    __local uchar* l_class
) {
    ulong tail_best, tail_consensus, tail_worst;
    uint disagreement_q8;

    bn_compute_tail_ensemble(
        hv,
        &tail_best,
        &tail_consensus,
        &tail_worst,
        &disagreement_q8
    );

    ulong soft_tail = bn_soft_pass_tail(tail_best, tail_consensus);
    uint active_tail_bins = bn_max_u32(1u, bn_max_u32(seed_tune_tail_bins, job_tune_tail_bins));
    uint bucket = bn_tune_bucket(hv[0], hv[1], nonce_u32);
    uint tail_bin = bn_tail_bin_from_tail(soft_tail, active_tail_bins);

    uint seed_rank_q = bn_read_tune_quality(
        seed_tune, seed_tune_buckets, seed_tune_tail_bins, BN_PLANE_RANK,
        bucket, tail_bin, BN_TUNE_NEUTRAL
    );
    uint seed_threshold_q = bn_read_tune_quality(
        seed_tune, seed_tune_buckets, seed_tune_tail_bins, BN_PLANE_THRESHOLD,
        bucket, tail_bin, BN_TUNE_NEUTRAL
    );
    uint seed_credit_q = bn_read_tune_quality(
        seed_tune, seed_tune_buckets, seed_tune_tail_bins, BN_PLANE_CREDIT,
        bucket, tail_bin, BN_TUNE_NEUTRAL
    );
    uint seed_conf_q = bn_read_tune_quality(
        seed_tune, seed_tune_buckets, seed_tune_tail_bins, BN_PLANE_CONFIDENCE,
        bucket, tail_bin, 0u
    );

    uint job_rank_q = bn_read_tune_quality(
        job_tune, job_tune_buckets, job_tune_tail_bins, BN_PLANE_RANK,
        bucket, tail_bin, BN_TUNE_NEUTRAL
    );
    uint job_threshold_q = bn_read_tune_quality(
        job_tune, job_tune_buckets, job_tune_tail_bins, BN_PLANE_THRESHOLD,
        bucket, tail_bin, BN_TUNE_NEUTRAL
    );
    uint job_credit_q = bn_read_tune_quality(
        job_tune, job_tune_buckets, job_tune_tail_bins, BN_PLANE_CREDIT,
        bucket, tail_bin, BN_TUNE_NEUTRAL
    );
    uint job_conf_q = bn_read_tune_quality(
        job_tune, job_tune_buckets, job_tune_tail_bins, BN_PLANE_CONFIDENCE,
        bucket, tail_bin, 0u
    );

    uint rank_q = bn_blend_quality(seed_rank_q, seed_conf_q, job_rank_q, job_conf_q);
    uint threshold_q = bn_blend_quality(seed_threshold_q, seed_conf_q, job_threshold_q, job_conf_q);
    uint credit_q = bn_blend_quality(seed_credit_q, seed_conf_q, job_credit_q, job_conf_q);
    uint confidence_q = bn_max_u32(seed_conf_q, job_conf_q);

    ulong adjusted_target = bn_adjust_target64(target64, threshold_q);
    adjusted_target = bn_apply_operational_tightening(
        adjusted_target,
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8
    );
    adjusted_target = bn_apply_early_job_relaxation(
        adjusted_target,
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8,
        confidence_q
    );

    ulong near_target = bn_near_target64(adjusted_target, confidence_q);

    uint stage_class = BN_STAGE_REJECT;
    if (soft_tail <= adjusted_target || tail_best <= adjusted_target) {
        stage_class = BN_STAGE_PASS;
    } else if (tail_best <= near_target && confidence_q >= 16u) {
        stage_class = BN_STAGE_NEAR;
    }

    ulong score = bn_compose_rank_score(
        hv,
        soft_tail,
        disagreement_q8,
        rank_q,
        credit_q,
        confidence_q,
        stage_class
    );

    score = bn_apply_operational_penalty(
        score,
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8
    );

    bn_stage_candidate_local(
        lid,
        nonce_u32,
        hv,
        score,
        stage_class,
        bucket,
        tail_bin,
        rank_q,
        threshold_q,
        l_score,
        l_nonce,
        l_h0,
        l_h1,
        l_h2,
        l_h3,
        l_bucket,
        l_rankq,
        l_threshq,
        l_tailbin,
        l_class
    );
}

inline void bn_flush_local_topk(
    uint lid,
    uint local_size,
    uint max_results,
    uint job_age_ms,
    uint verify_pressure_q8,
    uint submit_pressure_q8,
    uint stale_risk_q8,
    __local ulong* l_score,
    __local uint* l_nonce,
    __local ulong* l_h0,
    __local ulong* l_h1,
    __local ulong* l_h2,
    __local ulong* l_h3,
    __local uint* l_bucket,
    __local uchar* l_rankq,
    __local uchar* l_threshq,
    __local uchar* l_tailbin,
    __local uchar* l_class,
    __global uchar* out_hashes,
    __global uint* out_nonces,
    __global ulong* out_scores,
    __global uint* out_buckets,
    __global uchar* out_rankq,
    __global uchar* out_threshq,
    __global uchar* out_tailbin,
    __global uint* out_count
) {
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid != 0u) {
        return;
    }

    uint active = bn_min_u32(local_size, (uint)BN_LOCAL_STAGE_SIZE);
    uint effective_topk = bn_effective_local_topk(
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8
    );
    effective_topk = bn_min_u32(effective_topk, max_results);

    uint near_limit = bn_effective_local_near_limit(
        effective_topk,
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8
    );

    uchar used[BN_LOCAL_STAGE_SIZE];
    uint selected_ix[BN_LOCAL_TOPK];
    uint chosen_bucket[BN_LOCAL_TOPK];
    uchar chosen_tailbin[BN_LOCAL_TOPK];

    for (uint i = 0u; i < active; ++i) {
        used[i] = (uchar)0u;
    }

    uint selected_count = 0u;
    uint near_picked = 0u;

    while (selected_count < effective_topk) {
        ulong best_score = ~0UL;
        uint best_i = BN_LOCAL_STAGE_SIZE;

        for (uint i = 0u; i < active; ++i) {
            if (used[i] != (uchar)0u) continue;
            if ((uint)l_class[i] != BN_STAGE_PASS) continue;
            if (l_score[i] == ~0UL) continue;

            uint dup_cell = 0u;
            for (uint j = 0u; j < selected_count; ++j) {
                if (chosen_bucket[j] == l_bucket[i] && (uint)chosen_tailbin[j] == (uint)l_tailbin[i]) {
                    dup_cell = 1u;
                    break;
                }
            }
            if (dup_cell != 0u) continue;

            if (l_score[i] < best_score) {
                best_score = l_score[i];
                best_i = i;
            }
        }

        if (best_i >= active || best_score == ~0UL) {
            break;
        }

        used[best_i] = (uchar)1u;
        selected_ix[selected_count] = best_i;
        chosen_bucket[selected_count] = l_bucket[best_i];
        chosen_tailbin[selected_count] = l_tailbin[best_i];
        ++selected_count;
    }

    while (selected_count < effective_topk && near_picked < near_limit) {
        ulong best_score = ~0UL;
        uint best_i = BN_LOCAL_STAGE_SIZE;

        for (uint i = 0u; i < active; ++i) {
            if (used[i] != (uchar)0u) continue;
            if ((uint)l_class[i] != BN_STAGE_NEAR) continue;
            if (l_score[i] == ~0UL) continue;

            uint dup_cell = 0u;
            for (uint j = 0u; j < selected_count; ++j) {
                if (chosen_bucket[j] == l_bucket[i] && (uint)chosen_tailbin[j] == (uint)l_tailbin[i]) {
                    dup_cell = 1u;
                    break;
                }
            }
            if (dup_cell != 0u) continue;

            if (l_score[i] < best_score) {
                best_score = l_score[i];
                best_i = i;
            }
        }

        if (best_i >= active || best_score == ~0UL) {
            break;
        }

        used[best_i] = (uchar)1u;
        selected_ix[selected_count] = best_i;
        chosen_bucket[selected_count] = l_bucket[best_i];
        chosen_tailbin[selected_count] = l_tailbin[best_i];
        ++selected_count;
        ++near_picked;
    }

    while (selected_count < effective_topk) {
        ulong best_adj_score = ~0UL;
        uint best_i = BN_LOCAL_STAGE_SIZE;

        for (uint i = 0u; i < active; ++i) {
            if (used[i] != (uchar)0u) continue;
            if ((uint)l_class[i] == BN_STAGE_REJECT) continue;
            if (l_score[i] == ~0UL) continue;
            if ((uint)l_class[i] == BN_STAGE_NEAR && near_picked >= near_limit) continue;

            ulong adj = bn_add_penalty_sat(
                l_score[i],
                bn_local_pick_penalty(
                    (uint)l_class[i],
                    l_bucket[i],
                    (uint)l_tailbin[i],
                    chosen_bucket,
                    chosen_tailbin,
                    selected_count
                )
            );

            if (adj < best_adj_score) {
                best_adj_score = adj;
                best_i = i;
            }
        }

        if (best_i >= active || best_adj_score == ~0UL) {
            break;
        }

        used[best_i] = (uchar)1u;
        selected_ix[selected_count] = best_i;
        chosen_bucket[selected_count] = l_bucket[best_i];
        chosen_tailbin[selected_count] = l_tailbin[best_i];

        if ((uint)l_class[best_i] == BN_STAGE_NEAR) {
            ++near_picked;
        }
        ++selected_count;
    }

    if (selected_count == 0u) {
        return;
    }

    uint base_slot = atomic_add((volatile __global uint*)out_count, selected_count);

    for (uint k = 0u; k < selected_count; ++k) {
        uint slot = base_slot + k;
        if (slot >= max_results) {
            break;
        }

        uint ix = selected_ix[k];
        out_nonces[slot] = l_nonce[ix];
        out_scores[slot] = l_score[ix];
        out_buckets[slot] = l_bucket[ix];
        out_rankq[slot] = l_rankq[ix];
        out_threshq[slot] = l_threshq[ix];
        out_tailbin[slot] = l_tailbin[ix];

        bn_write_hash32(
            out_hashes + (slot * BN_HASH_BYTES),
            l_h0[ix],
            l_h1[ix],
            l_h2[ix],
            l_h3[ix]
        );
    }
}

inline void bn_run_core(
    __global const uchar* blob,
    const uint blob_len,
    const uint nonce_offset,
    const uint start_nonce,
    const ulong target64,
    const uint max_results,
    __global uchar* out_hashes,
    __global uint* out_nonces,
    __global ulong* out_scores,
    __global uint* out_buckets,
    __global uchar* out_rankq,
    __global uchar* out_threshq,
    __global uchar* out_tailbin,
    __global uint* out_count,
    __global const uchar* seed,
    const uint seed_len,
    __global const ulong* dataset64,
    const uint dataset_words,
    __global const uchar* seed_tune,
    const uint seed_tune_buckets,
    const uint seed_tune_tail_bins,
    __global const uchar* job_tune,
    const uint job_tune_buckets,
    const uint job_tune_tail_bins,
    const uint job_age_ms,
    const uint verify_pressure_q8,
    const uint submit_pressure_q8,
    const uint stale_risk_q8,
    __local ulong* l_score,
    __local uint* l_nonce,
    __local ulong* l_h0,
    __local ulong* l_h1,
    __local ulong* l_h2,
    __local ulong* l_h3,
    __local uint* l_bucket,
    __local uchar* l_rankq,
    __local uchar* l_threshq,
    __local uchar* l_tailbin,
    __local uchar* l_class
) {
    const uint lid = get_local_id(0);
    const uint local_size = get_local_size(0);
    const uint nonce_u32 = start_nonce + (uint)get_global_id(0);

    __private ulong hv[4];

    bn_gpu_dataset_prefilter_hash_fast(
        blob,
        blob_len,
        nonce_offset,
        nonce_u32,
        seed,
        seed_len,
        dataset64,
        dataset_words,
        hv
    );

    bn_rank_and_stage(
        lid,
        nonce_u32,
        hv,
        target64,
        seed_tune,
        seed_tune_buckets,
        seed_tune_tail_bins,
        job_tune,
        job_tune_buckets,
        job_tune_tail_bins,
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8,
        l_score,
        l_nonce,
        l_h0,
        l_h1,
        l_h2,
        l_h3,
        l_bucket,
        l_rankq,
        l_threshq,
        l_tailbin,
        l_class
    );

    bn_flush_local_topk(
        lid,
        local_size,
        max_results,
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8,
        l_score,
        l_nonce,
        l_h0,
        l_h1,
        l_h2,
        l_h3,
        l_bucket,
        l_rankq,
        l_threshq,
        l_tailbin,
        l_class,
        out_hashes,
        out_nonces,
        out_scores,
        out_buckets,
        out_rankq,
        out_threshq,
        out_tailbin,
        out_count
    );
}

// @vasic_mode candidate_merge
// @vasic_count_arg 14
// @vasic_merge_buffer 7:32
// @vasic_merge_buffer 8:4
// @vasic_merge_buffer 9:8
// @vasic_merge_buffer 10:4
// @vasic_merge_buffer 11:1
// @vasic_merge_buffer 12:1
// @vasic_merge_buffer 13:1
// @vasic_partition global_offset
__kernel void blocknet_randomx_vm_scan_vasic(
    __global const uchar* blob,
    const uint blob_len,
    const uint nonce_offset,
    const uint start_nonce,
    const uint target64_lo,
    const uint target64_hi,
    const uint max_results,
    __global uchar* out_hashes,
    __global uint* out_nonces,
    __global ulong* out_scores,
    __global uint* out_buckets,
    __global uchar* out_rankq,
    __global uchar* out_threshq,
    __global uchar* out_tailbin,
    __global uint* out_count,
    __global const uchar* seed,
    const uint seed_len,
    __global const ulong* dataset64,
    const uint dataset_words,
    __global const uchar* seed_tune,
    const uint seed_tune_buckets,
    const uint seed_tune_tail_bins,
    __global const uchar* job_tune,
    const uint job_tune_buckets,
    const uint job_tune_tail_bins,
    const uint job_age_ms,
    const uint verify_pressure_q8,
    const uint submit_pressure_q8,
    const uint stale_risk_q8
) {
    const ulong target64 = ((ulong)target64_lo) | (((ulong)target64_hi) << 32);

    __local ulong l_score[BN_LOCAL_STAGE_SIZE];
    __local uint  l_nonce[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h0[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h1[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h2[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h3[BN_LOCAL_STAGE_SIZE];
    __local uint  l_bucket[BN_LOCAL_STAGE_SIZE];
    __local uchar l_rankq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_threshq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_tailbin[BN_LOCAL_STAGE_SIZE];
    __local uchar l_class[BN_LOCAL_STAGE_SIZE];

    bn_run_core(
        blob,
        blob_len,
        nonce_offset,
        start_nonce,
        target64,
        max_results,
        out_hashes,
        out_nonces,
        out_scores,
        out_buckets,
        out_rankq,
        out_threshq,
        out_tailbin,
        out_count,
        seed,
        seed_len,
        dataset64,
        dataset_words,
        seed_tune,
        seed_tune_buckets,
        seed_tune_tail_bins,
        job_tune,
        job_tune_buckets,
        job_tune_tail_bins,
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8,
        l_score,
        l_nonce,
        l_h0,
        l_h1,
        l_h2,
        l_h3,
        l_bucket,
        l_rankq,
        l_threshq,
        l_tailbin,
        l_class
    );
}

// @vasic_mode candidate_merge
// @vasic_count_arg 14
// @vasic_merge_buffer 7:32
// @vasic_merge_buffer 8:4
// @vasic_merge_buffer 9:8
// @vasic_merge_buffer 10:4
// @vasic_merge_buffer 11:1
// @vasic_merge_buffer 12:1
// @vasic_merge_buffer 13:1
// @vasic_partition global_offset
__kernel void blocknet_randomx_vm_hash_batch_vasic(
    __global const uchar* blob,
    const uint blob_len,
    const uint nonce_offset,
    const uint start_nonce,
    const uint target64_lo,
    const uint target64_hi,
    const uint max_results,
    __global uchar* out_hashes,
    __global uint* out_nonces,
    __global ulong* out_scores,
    __global uint* out_buckets,
    __global uchar* out_rankq,
    __global uchar* out_threshq,
    __global uchar* out_tailbin,
    __global uint* out_count,
    __global const uchar* seed,
    const uint seed_len,
    __global const ulong* dataset64,
    const uint dataset_words,
    __global const uchar* seed_tune,
    const uint seed_tune_buckets,
    const uint seed_tune_tail_bins,
    __global const uchar* job_tune,
    const uint job_tune_buckets,
    const uint job_tune_tail_bins,
    const uint job_age_ms,
    const uint verify_pressure_q8,
    const uint submit_pressure_q8,
    const uint stale_risk_q8
) {
    const ulong target64 = ((ulong)target64_lo) | (((ulong)target64_hi) << 32);

    __local ulong l_score[BN_LOCAL_STAGE_SIZE];
    __local uint  l_nonce[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h0[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h1[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h2[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h3[BN_LOCAL_STAGE_SIZE];
    __local uint  l_bucket[BN_LOCAL_STAGE_SIZE];
    __local uchar l_rankq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_threshq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_tailbin[BN_LOCAL_STAGE_SIZE];
    __local uchar l_class[BN_LOCAL_STAGE_SIZE];

    bn_run_core(
        blob,
        blob_len,
        nonce_offset,
        start_nonce,
        target64,
        max_results,
        out_hashes,
        out_nonces,
        out_scores,
        out_buckets,
        out_rankq,
        out_threshq,
        out_tailbin,
        out_count,
        seed,
        seed_len,
        dataset64,
        dataset_words,
        seed_tune,
        seed_tune_buckets,
        seed_tune_tail_bins,
        job_tune,
        job_tune_buckets,
        job_tune_tail_bins,
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8,
        l_score,
        l_nonce,
        l_h0,
        l_h1,
        l_h2,
        l_h3,
        l_bucket,
        l_rankq,
        l_threshq,
        l_tailbin,
        l_class
    );
}

__kernel void blocknet_randomx_vm_scan_ext(
    __global const uchar* blob,
    const uint blob_len,
    const uint nonce_offset,
    const uint start_nonce,
    const ulong target64,
    const uint max_results,
    __global uchar* out_hashes,
    __global uint* out_nonces,
    __global ulong* out_scores,
    __global uint* out_buckets,
    __global uchar* out_rankq,
    __global uchar* out_threshq,
    __global uchar* out_tailbin,
    __global uint* out_count,
    __global const uchar* seed,
    const uint seed_len,
    __global const ulong* dataset64,
    const uint dataset_words,
    __global const uchar* seed_tune,
    const uint seed_tune_buckets,
    const uint seed_tune_tail_bins,
    __global const uchar* job_tune,
    const uint job_tune_buckets,
    const uint job_tune_tail_bins,
    const uint job_age_ms,
    const uint verify_pressure_q8,
    const uint submit_pressure_q8,
    const uint stale_risk_q8
) {
    __local ulong l_score[BN_LOCAL_STAGE_SIZE];
    __local uint  l_nonce[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h0[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h1[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h2[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h3[BN_LOCAL_STAGE_SIZE];
    __local uint  l_bucket[BN_LOCAL_STAGE_SIZE];
    __local uchar l_rankq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_threshq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_tailbin[BN_LOCAL_STAGE_SIZE];
    __local uchar l_class[BN_LOCAL_STAGE_SIZE];

    bn_run_core(
        blob,
        blob_len,
        nonce_offset,
        start_nonce,
        target64,
        max_results,
        out_hashes,
        out_nonces,
        out_scores,
        out_buckets,
        out_rankq,
        out_threshq,
        out_tailbin,
        out_count,
        seed,
        seed_len,
        dataset64,
        dataset_words,
        seed_tune,
        seed_tune_buckets,
        seed_tune_tail_bins,
        job_tune,
        job_tune_buckets,
        job_tune_tail_bins,
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8,
        l_score,
        l_nonce,
        l_h0,
        l_h1,
        l_h2,
        l_h3,
        l_bucket,
        l_rankq,
        l_threshq,
        l_tailbin,
        l_class
    );
}

__kernel void blocknet_randomx_vm_hash_batch_ext(
    __global const uchar* blob,
    const uint blob_len,
    const uint nonce_offset,
    const uint start_nonce,
    const ulong target64,
    const uint max_results,
    __global uchar* out_hashes,
    __global uint* out_nonces,
    __global ulong* out_scores,
    __global uint* out_buckets,
    __global uchar* out_rankq,
    __global uchar* out_threshq,
    __global uchar* out_tailbin,
    __global uint* out_count,
    __global const uchar* seed,
    const uint seed_len,
    __global const ulong* dataset64,
    const uint dataset_words,
    __global const uchar* seed_tune,
    const uint seed_tune_buckets,
    const uint seed_tune_tail_bins,
    __global const uchar* job_tune,
    const uint job_tune_buckets,
    const uint job_tune_tail_bins,
    const uint job_age_ms,
    const uint verify_pressure_q8,
    const uint submit_pressure_q8,
    const uint stale_risk_q8
) {
    __local ulong l_score[BN_LOCAL_STAGE_SIZE];
    __local uint  l_nonce[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h0[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h1[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h2[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h3[BN_LOCAL_STAGE_SIZE];
    __local uint  l_bucket[BN_LOCAL_STAGE_SIZE];
    __local uchar l_rankq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_threshq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_tailbin[BN_LOCAL_STAGE_SIZE];
    __local uchar l_class[BN_LOCAL_STAGE_SIZE];

    bn_run_core(
        blob,
        blob_len,
        nonce_offset,
        start_nonce,
        target64,
        max_results,
        out_hashes,
        out_nonces,
        out_scores,
        out_buckets,
        out_rankq,
        out_threshq,
        out_tailbin,
        out_count,
        seed,
        seed_len,
        dataset64,
        dataset_words,
        seed_tune,
        seed_tune_buckets,
        seed_tune_tail_bins,
        job_tune,
        job_tune_buckets,
        job_tune_tail_bins,
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8,
        l_score,
        l_nonce,
        l_h0,
        l_h1,
        l_h2,
        l_h3,
        l_bucket,
        l_rankq,
        l_threshq,
        l_tailbin,
        l_class
    );
}

// ============================================================
// ADDITIONAL FUSED COMPATIBILITY KERNELS
// ============================================================

// Very small baseline: one work item writes one 32-byte hash and nonce at global_id.
// Useful for smoke tests and host-side verification without candidate merging.
__kernel void blocknet_randomx_basic_hash(
    __global const uchar* blob,
    const uint blob_len,
    const uint nonce_offset,
    const uint start_nonce,
    __global uchar* out_hashes,
    __global uint* out_nonces,
    __global const uchar* seed,
    const uint seed_len,
    __global const ulong* dataset64,
    const uint dataset_words
) {
    const uint gid = (uint)get_global_id(0);
    const uint nonce_u32 = start_nonce + gid;
    __private ulong hv[4];

    bn_gpu_dataset_prefilter_hash_fast(
        blob,
        blob_len,
        nonce_offset,
        nonce_u32,
        seed,
        seed_len,
        dataset64,
        dataset_words,
        hv
    );

    out_nonces[gid] = nonce_u32;
    bn_write_hash32(out_hashes + (gid * BN_HASH_BYTES), hv[0], hv[1], hv[2], hv[3]);
}

// Very small baseline scanner: direct tail check, no tune planes, no local top-k.
// out_count should be cleared by the host before launch.
__kernel void blocknet_randomx_basic_scan(
    __global const uchar* blob,
    const uint blob_len,
    const uint nonce_offset,
    const uint start_nonce,
    const ulong target64,
    const uint max_results,
    __global uchar* out_hashes,
    __global uint* out_nonces,
    __global uint* out_count,
    __global const uchar* seed,
    const uint seed_len,
    __global const ulong* dataset64,
    const uint dataset_words
) {
    const uint nonce_u32 = start_nonce + (uint)get_global_id(0);
    __private ulong hv[4];

    bn_gpu_dataset_prefilter_hash_fast(
        blob,
        blob_len,
        nonce_offset,
        nonce_u32,
        seed,
        seed_len,
        dataset64,
        dataset_words,
        hv
    );

    if (hv[3] <= target64) {
        uint slot = atomic_add((volatile __global uint*)out_count, 1u);
        if (slot < max_results) {
            out_nonces[slot] = nonce_u32;
            bn_write_hash32(out_hashes + (slot * BN_HASH_BYTES), hv[0], hv[1], hv[2], hv[3]);
        }
    }
}

__kernel void blocknet_randomx_basic(
    __global const uchar* blob,
    const uint blob_len,
    const uint nonce_offset,
    const uint start_nonce,
    const ulong target64,
    const uint max_results,
    __global uchar* out_hashes,
    __global uint* out_nonces,
    __global ulong* out_scores,
    __global uint* out_buckets,
    __global uchar* out_rankq,
    __global uchar* out_threshq,
    __global uchar* out_tailbin,
    __global uint* out_count,
    __global const uchar* seed,
    const uint seed_len,
    __global const ulong* dataset64,
    const uint dataset_words,
    __global const uchar* seed_tune,
    const uint seed_tune_buckets,
    const uint seed_tune_tail_bins,
    __global const uchar* job_tune,
    const uint job_tune_buckets,
    const uint job_tune_tail_bins,
    const uint job_age_ms,
    const uint verify_pressure_q8,
    const uint submit_pressure_q8,
    const uint stale_risk_q8
) {
    __local ulong l_score[BN_LOCAL_STAGE_SIZE];
    __local uint  l_nonce[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h0[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h1[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h2[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h3[BN_LOCAL_STAGE_SIZE];
    __local uint  l_bucket[BN_LOCAL_STAGE_SIZE];
    __local uchar l_rankq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_threshq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_tailbin[BN_LOCAL_STAGE_SIZE];
    __local uchar l_class[BN_LOCAL_STAGE_SIZE];

    bn_run_core(
        blob,
        blob_len,
        nonce_offset,
        start_nonce,
        target64,
        max_results,
        out_hashes,
        out_nonces,
        out_scores,
        out_buckets,
        out_rankq,
        out_threshq,
        out_tailbin,
        out_count,
        seed,
        seed_len,
        dataset64,
        dataset_words,
        seed_tune,
        seed_tune_buckets,
        seed_tune_tail_bins,
        job_tune,
        job_tune_buckets,
        job_tune_tail_bins,
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8,
        l_score,
        l_nonce,
        l_h0,
        l_h1,
        l_h2,
        l_h3,
        l_bucket,
        l_rankq,
        l_threshq,
        l_tailbin,
        l_class
    );
}

__kernel void blocknet_randomx_fast(
    __global const uchar* blob,
    const uint blob_len,
    const uint nonce_offset,
    const uint start_nonce,
    const ulong target64,
    const uint max_results,
    __global uchar* out_hashes,
    __global uint* out_nonces,
    __global ulong* out_scores,
    __global uint* out_buckets,
    __global uchar* out_rankq,
    __global uchar* out_threshq,
    __global uchar* out_tailbin,
    __global uint* out_count,
    __global const uchar* seed,
    const uint seed_len,
    __global const ulong* dataset64,
    const uint dataset_words,
    __global const uchar* seed_tune,
    const uint seed_tune_buckets,
    const uint seed_tune_tail_bins,
    __global const uchar* job_tune,
    const uint job_tune_buckets,
    const uint job_tune_tail_bins,
    const uint job_age_ms,
    const uint verify_pressure_q8,
    const uint submit_pressure_q8,
    const uint stale_risk_q8
) {
    __local ulong l_score[BN_LOCAL_STAGE_SIZE];
    __local uint  l_nonce[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h0[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h1[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h2[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h3[BN_LOCAL_STAGE_SIZE];
    __local uint  l_bucket[BN_LOCAL_STAGE_SIZE];
    __local uchar l_rankq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_threshq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_tailbin[BN_LOCAL_STAGE_SIZE];
    __local uchar l_class[BN_LOCAL_STAGE_SIZE];

    bn_run_core(
        blob,
        blob_len,
        nonce_offset,
        start_nonce,
        target64,
        max_results,
        out_hashes,
        out_nonces,
        out_scores,
        out_buckets,
        out_rankq,
        out_threshq,
        out_tailbin,
        out_count,
        seed,
        seed_len,
        dataset64,
        dataset_words,
        seed_tune,
        seed_tune_buckets,
        seed_tune_tail_bins,
        job_tune,
        job_tune_buckets,
        job_tune_tail_bins,
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8,
        l_score,
        l_nonce,
        l_h0,
        l_h1,
        l_h2,
        l_h3,
        l_bucket,
        l_rankq,
        l_threshq,
        l_tailbin,
        l_class
    );
}

__kernel void blocknet_randomx_optimized(
    __global const uchar* blob,
    const uint blob_len,
    const uint nonce_offset,
    const uint start_nonce,
    const ulong target64,
    const uint max_results,
    __global uchar* out_hashes,
    __global uint* out_nonces,
    __global ulong* out_scores,
    __global uint* out_buckets,
    __global uchar* out_rankq,
    __global uchar* out_threshq,
    __global uchar* out_tailbin,
    __global uint* out_count,
    __global const uchar* seed,
    const uint seed_len,
    __global const ulong* dataset64,
    const uint dataset_words,
    __global const uchar* seed_tune,
    const uint seed_tune_buckets,
    const uint seed_tune_tail_bins,
    __global const uchar* job_tune,
    const uint job_tune_buckets,
    const uint job_tune_tail_bins,
    const uint job_age_ms,
    const uint verify_pressure_q8,
    const uint submit_pressure_q8,
    const uint stale_risk_q8
) {
    __local ulong l_score[BN_LOCAL_STAGE_SIZE];
    __local uint  l_nonce[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h0[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h1[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h2[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h3[BN_LOCAL_STAGE_SIZE];
    __local uint  l_bucket[BN_LOCAL_STAGE_SIZE];
    __local uchar l_rankq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_threshq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_tailbin[BN_LOCAL_STAGE_SIZE];
    __local uchar l_class[BN_LOCAL_STAGE_SIZE];

    bn_run_core(
        blob,
        blob_len,
        nonce_offset,
        start_nonce,
        target64,
        max_results,
        out_hashes,
        out_nonces,
        out_scores,
        out_buckets,
        out_rankq,
        out_threshq,
        out_tailbin,
        out_count,
        seed,
        seed_len,
        dataset64,
        dataset_words,
        seed_tune,
        seed_tune_buckets,
        seed_tune_tail_bins,
        job_tune,
        job_tune_buckets,
        job_tune_tail_bins,
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8,
        l_score,
        l_nonce,
        l_h0,
        l_h1,
        l_h2,
        l_h3,
        l_bucket,
        l_rankq,
        l_threshq,
        l_tailbin,
        l_class
    );
}

__kernel void blocknet_randomx_vm_scan_basic(
    __global const uchar* blob,
    const uint blob_len,
    const uint nonce_offset,
    const uint start_nonce,
    const ulong target64,
    const uint max_results,
    __global uchar* out_hashes,
    __global uint* out_nonces,
    __global ulong* out_scores,
    __global uint* out_buckets,
    __global uchar* out_rankq,
    __global uchar* out_threshq,
    __global uchar* out_tailbin,
    __global uint* out_count,
    __global const uchar* seed,
    const uint seed_len,
    __global const ulong* dataset64,
    const uint dataset_words,
    __global const uchar* seed_tune,
    const uint seed_tune_buckets,
    const uint seed_tune_tail_bins,
    __global const uchar* job_tune,
    const uint job_tune_buckets,
    const uint job_tune_tail_bins,
    const uint job_age_ms,
    const uint verify_pressure_q8,
    const uint submit_pressure_q8,
    const uint stale_risk_q8
) {
    __local ulong l_score[BN_LOCAL_STAGE_SIZE];
    __local uint  l_nonce[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h0[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h1[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h2[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h3[BN_LOCAL_STAGE_SIZE];
    __local uint  l_bucket[BN_LOCAL_STAGE_SIZE];
    __local uchar l_rankq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_threshq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_tailbin[BN_LOCAL_STAGE_SIZE];
    __local uchar l_class[BN_LOCAL_STAGE_SIZE];

    bn_run_core(
        blob,
        blob_len,
        nonce_offset,
        start_nonce,
        target64,
        max_results,
        out_hashes,
        out_nonces,
        out_scores,
        out_buckets,
        out_rankq,
        out_threshq,
        out_tailbin,
        out_count,
        seed,
        seed_len,
        dataset64,
        dataset_words,
        seed_tune,
        seed_tune_buckets,
        seed_tune_tail_bins,
        job_tune,
        job_tune_buckets,
        job_tune_tail_bins,
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8,
        l_score,
        l_nonce,
        l_h0,
        l_h1,
        l_h2,
        l_h3,
        l_bucket,
        l_rankq,
        l_threshq,
        l_tailbin,
        l_class
    );
}

__kernel void blocknet_randomx_vm_hash_batch_basic(
    __global const uchar* blob,
    const uint blob_len,
    const uint nonce_offset,
    const uint start_nonce,
    const ulong target64,
    const uint max_results,
    __global uchar* out_hashes,
    __global uint* out_nonces,
    __global ulong* out_scores,
    __global uint* out_buckets,
    __global uchar* out_rankq,
    __global uchar* out_threshq,
    __global uchar* out_tailbin,
    __global uint* out_count,
    __global const uchar* seed,
    const uint seed_len,
    __global const ulong* dataset64,
    const uint dataset_words,
    __global const uchar* seed_tune,
    const uint seed_tune_buckets,
    const uint seed_tune_tail_bins,
    __global const uchar* job_tune,
    const uint job_tune_buckets,
    const uint job_tune_tail_bins,
    const uint job_age_ms,
    const uint verify_pressure_q8,
    const uint submit_pressure_q8,
    const uint stale_risk_q8
) {
    __local ulong l_score[BN_LOCAL_STAGE_SIZE];
    __local uint  l_nonce[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h0[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h1[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h2[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h3[BN_LOCAL_STAGE_SIZE];
    __local uint  l_bucket[BN_LOCAL_STAGE_SIZE];
    __local uchar l_rankq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_threshq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_tailbin[BN_LOCAL_STAGE_SIZE];
    __local uchar l_class[BN_LOCAL_STAGE_SIZE];

    bn_run_core(
        blob,
        blob_len,
        nonce_offset,
        start_nonce,
        target64,
        max_results,
        out_hashes,
        out_nonces,
        out_scores,
        out_buckets,
        out_rankq,
        out_threshq,
        out_tailbin,
        out_count,
        seed,
        seed_len,
        dataset64,
        dataset_words,
        seed_tune,
        seed_tune_buckets,
        seed_tune_tail_bins,
        job_tune,
        job_tune_buckets,
        job_tune_tail_bins,
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8,
        l_score,
        l_nonce,
        l_h0,
        l_h1,
        l_h2,
        l_h3,
        l_bucket,
        l_rankq,
        l_threshq,
        l_tailbin,
        l_class
    );
}

__kernel void blocknet_randomx_vm_scan_fast(
    __global const uchar* blob,
    const uint blob_len,
    const uint nonce_offset,
    const uint start_nonce,
    const ulong target64,
    const uint max_results,
    __global uchar* out_hashes,
    __global uint* out_nonces,
    __global ulong* out_scores,
    __global uint* out_buckets,
    __global uchar* out_rankq,
    __global uchar* out_threshq,
    __global uchar* out_tailbin,
    __global uint* out_count,
    __global const uchar* seed,
    const uint seed_len,
    __global const ulong* dataset64,
    const uint dataset_words,
    __global const uchar* seed_tune,
    const uint seed_tune_buckets,
    const uint seed_tune_tail_bins,
    __global const uchar* job_tune,
    const uint job_tune_buckets,
    const uint job_tune_tail_bins,
    const uint job_age_ms,
    const uint verify_pressure_q8,
    const uint submit_pressure_q8,
    const uint stale_risk_q8
) {
    __local ulong l_score[BN_LOCAL_STAGE_SIZE];
    __local uint  l_nonce[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h0[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h1[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h2[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h3[BN_LOCAL_STAGE_SIZE];
    __local uint  l_bucket[BN_LOCAL_STAGE_SIZE];
    __local uchar l_rankq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_threshq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_tailbin[BN_LOCAL_STAGE_SIZE];
    __local uchar l_class[BN_LOCAL_STAGE_SIZE];

    bn_run_core(
        blob,
        blob_len,
        nonce_offset,
        start_nonce,
        target64,
        max_results,
        out_hashes,
        out_nonces,
        out_scores,
        out_buckets,
        out_rankq,
        out_threshq,
        out_tailbin,
        out_count,
        seed,
        seed_len,
        dataset64,
        dataset_words,
        seed_tune,
        seed_tune_buckets,
        seed_tune_tail_bins,
        job_tune,
        job_tune_buckets,
        job_tune_tail_bins,
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8,
        l_score,
        l_nonce,
        l_h0,
        l_h1,
        l_h2,
        l_h3,
        l_bucket,
        l_rankq,
        l_threshq,
        l_tailbin,
        l_class
    );
}

__kernel void blocknet_randomx_vm_hash_batch_fast(
    __global const uchar* blob,
    const uint blob_len,
    const uint nonce_offset,
    const uint start_nonce,
    const ulong target64,
    const uint max_results,
    __global uchar* out_hashes,
    __global uint* out_nonces,
    __global ulong* out_scores,
    __global uint* out_buckets,
    __global uchar* out_rankq,
    __global uchar* out_threshq,
    __global uchar* out_tailbin,
    __global uint* out_count,
    __global const uchar* seed,
    const uint seed_len,
    __global const ulong* dataset64,
    const uint dataset_words,
    __global const uchar* seed_tune,
    const uint seed_tune_buckets,
    const uint seed_tune_tail_bins,
    __global const uchar* job_tune,
    const uint job_tune_buckets,
    const uint job_tune_tail_bins,
    const uint job_age_ms,
    const uint verify_pressure_q8,
    const uint submit_pressure_q8,
    const uint stale_risk_q8
) {
    __local ulong l_score[BN_LOCAL_STAGE_SIZE];
    __local uint  l_nonce[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h0[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h1[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h2[BN_LOCAL_STAGE_SIZE];
    __local ulong l_h3[BN_LOCAL_STAGE_SIZE];
    __local uint  l_bucket[BN_LOCAL_STAGE_SIZE];
    __local uchar l_rankq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_threshq[BN_LOCAL_STAGE_SIZE];
    __local uchar l_tailbin[BN_LOCAL_STAGE_SIZE];
    __local uchar l_class[BN_LOCAL_STAGE_SIZE];

    bn_run_core(
        blob,
        blob_len,
        nonce_offset,
        start_nonce,
        target64,
        max_results,
        out_hashes,
        out_nonces,
        out_scores,
        out_buckets,
        out_rankq,
        out_threshq,
        out_tailbin,
        out_count,
        seed,
        seed_len,
        dataset64,
        dataset_words,
        seed_tune,
        seed_tune_buckets,
        seed_tune_tail_bins,
        job_tune,
        job_tune_buckets,
        job_tune_tail_bins,
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8,
        l_score,
        l_nonce,
        l_h0,
        l_h1,
        l_h2,
        l_h3,
        l_bucket,
        l_rankq,
        l_threshq,
        l_tailbin,
        l_class
    );
}

