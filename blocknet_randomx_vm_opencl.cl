#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#ifndef BN_HASH_BYTES
#define BN_HASH_BYTES 32u
#endif

#ifndef BN_MAX_BLOB_BYTES
#define BN_MAX_BLOB_BYTES 256u
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
#define BN_LOCAL_TOPK 8u
#endif

/*
    RandomX-shaped predictor controls.
    These DO NOT make this exact RandomX. They only make the surrogate
    much more VM/scratchpad/program oriented while preserving the host ABI.
*/
#ifndef BN_RXP_SCRATCH_WORDS
#define BN_RXP_SCRATCH_WORDS 256u
#endif

#ifndef BN_RXP_PROGRAM_SIZE
#define BN_RXP_PROGRAM_SIZE 64u
#endif

#ifndef BN_RXP_PROGRAM_ITERS
#define BN_RXP_PROGRAM_ITERS 96u
#endif

#ifndef BN_RXP_PROGRAM_COUNT
#define BN_RXP_PROGRAM_COUNT 4u
#endif

#ifndef BN_RXP_L1_WORDS
#define BN_RXP_L1_WORDS 32u
#endif

#ifndef BN_RXP_L2_WORDS
#define BN_RXP_L2_WORDS 128u
#endif

#ifndef BN_RXP_JUMP_MASK_BITS
#define BN_RXP_JUMP_MASK_BITS 8u
#endif
#ifndef BN_RXP_LINE_WORDS
#define BN_RXP_LINE_WORDS 8u
#endif

#ifndef BN_RXP_LINE_BYTES
#define BN_RXP_LINE_BYTES 64u
#endif

#ifndef BN_RXP_L1_LINES
#define BN_RXP_L1_LINES 8u
#endif

#ifndef BN_RXP_L2_LINES
#define BN_RXP_L2_LINES 16u
#endif

#ifndef BN_RXP_L3_LINES
#define BN_RXP_L3_LINES 32u
#endif

#ifndef BN_RXP_FP_LANES
#define BN_RXP_FP_LANES 4u
#endif
#define BN_U64_C(x) ((ulong)(x##UL))

#define BN_PLANE_RANK 0u
#define BN_PLANE_THRESHOLD 1u
#define BN_PLANE_CREDIT 2u
#define BN_PLANE_CONFIDENCE 3u
#define BN_PLANE_COUNT 4u

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

inline void bn_store_u32_le_p(__private uchar* p, uint v) {
    p[0] = (uchar)(v & 0xFFu);
    p[1] = (uchar)((v >> 8) & 0xFFu);
    p[2] = (uchar)((v >> 16) & 0xFFu);
    p[3] = (uchar)((v >> 24) & 0xFFu);
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

inline uint bn_reserve_slot(__global uint* out_count) {
    return atomic_inc((volatile __global uint*)out_count);
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
    (void)tune;
    if (buckets == 0u || tail_bins == 0u) {
        return fallback;
    }
    uint stride = buckets * tail_bins;
    uint idx = bn_tune_index(bucket, tail_bin, buckets, tail_bins);
    return (uint)tune[(plane * stride) + idx];
}

inline uint bn_blend_quality(uint seed_q, uint seed_conf, uint job_q, uint job_conf) {
    ulong sw = (ulong)max(1u, seed_conf);
    ulong jw = (ulong)(max(0u, job_conf) * 2u);

    if (jw == 0UL) {
        return seed_q;
    }

    long seed_term = ((long)seed_q - (long)BN_TUNE_NEUTRAL) * (long)sw;
    long job_term = ((long)job_q - (long)BN_TUNE_NEUTRAL) * (long)jw;
    long total = (long)(sw + jw);
    long q = (long)BN_TUNE_NEUTRAL + ((seed_term + job_term) / max(1L, total));

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
        ulong tighten = ((target64 >> 2) * (ulong)delta) / (ulong)BN_TUNE_NEUTRAL;
        if (tighten >= target64) {
            return 1UL;
        }
        return target64 - tighten;
    }

    if (threshold_quality > BN_TUNE_NEUTRAL) {
        uint delta = threshold_quality - BN_TUNE_NEUTRAL;
        ulong loosen = ((target64 >> 3) * (ulong)delta) / (ulong)BN_TUNE_NEUTRAL;
        if (target64 > (~0UL - loosen)) {
            return ~0UL;
        }
        return target64 + loosen;
    }

    return target64;
}

/* much softer than the strict version */
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

    uint pressure = max(verify_pressure_q8, max(submit_pressure_q8, stale_risk_q8));

    ulong tighten_pressure = ((target64 >> 4) * (ulong)pressure) / 255UL;
    uint age_ms = min(job_age_ms, 4000u);
    ulong tighten_age = ((target64 >> 4) * (ulong)age_ms) / 4000UL;

    ulong tighten = tighten_pressure + tighten_age;
    if (tighten >= target64) {
        return 1UL;
    }
    return max(1UL, target64 - tighten);
}

/* small early/fresh-job relaxation to keep candidates flowing */
inline ulong bn_apply_early_job_relaxation(
    ulong target64,
    uint job_age_ms,
    uint verify_pressure_q8,
    uint submit_pressure_q8,
    uint stale_risk_q8,
    uint confidence_quality
) {
    uint pressure = max(verify_pressure_q8, max(submit_pressure_q8, stale_risk_q8));
    if (job_age_ms > 900u || pressure > 64u) {
        return target64;
    }

    ulong bonus = target64 >> 2; /* +25% */
    if (confidence_quality >= 96u) {
        bonus += (target64 >> 4); /* up to +31.25% */
    }

    if (target64 > (~0UL - bonus)) {
        return ~0UL;
    }
    return target64 + bonus;
}

inline ulong bn_near_target64(
    ulong target64,
    uint confidence_quality
) {
    ulong bonus = target64 >> 2; /* +25% */

    if (confidence_quality >= 96u) {
        bonus += (target64 >> 3); /* +12.5% more */
    } else if (confidence_quality < 32u) {
        bonus >>= 1;
    }

    if (target64 > (~0UL - bonus)) {
        return ~0UL;
    }
    return target64 + max(1UL, bonus);
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
    uint pressure = max(verify_pressure_q8, max(submit_pressure_q8, stale_risk_q8));

    ulong penalty = (((ulong)pressure) << 30);
    uint age_ms = min(job_age_ms, 4000u);
    penalty += (((ulong)age_ms) << 18);

    ulong limit = ~0UL - penalty;
    if (score >= limit) {
        return ~0UL;
    }
    return score + penalty;
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
        bn_rotl64(hv[2], 17u) ^
        bn_rotr64(hv[3], 7u) ^
        BN_U64_C(0x9E3779B97F4A7C15)
    );
    ulong t2 = bn_mix64(
        hv[1] ^
        bn_rotr64(hv[2], 11u) ^
        bn_rotl64(hv[0], 3u) ^
        bn_rotl64(hv[3], 29u) ^
        BN_U64_C(0xD1B54A32D192ED03)
    );

    ulong mn = bn_min3_u64(t0, t1, t2);
    ulong mx = bn_max3_u64(t0, t1, t2);
    ulong md = bn_median3_u64(t0, t1, t2);

    ulong spread = mx - mn;
    uint spread_q8 = min(255u, (uint)(spread >> 53));

    *tail_best = mn;
    *tail_consensus = md;
    *tail_worst = mx;
    *disagreement_q8 = spread_q8;
}

inline ulong bn_soft_pass_tail(
    ulong tail_best,
    ulong tail_consensus
) {
    /* halfway between best and consensus: candidate-heavy but not reckless */
    return tail_best + ((tail_consensus - tail_best) >> 1);
}

inline ulong bn_uncertainty_penalty(
    __private const ulong hv[4],
    uint disagreement_q8
) {
    int pc0 = (int)bn_popcount64(hv[0]);
    int pc1 = (int)bn_popcount64(hv[1]);
    int pc2 = (int)bn_popcount64(hv[2]);
    int pc3 = (int)bn_popcount64(hv[3]);

    uint imbalance =
        (uint)abs(pc0 - 32) +
        (uint)abs(pc1 - 32) +
        (uint)abs(pc2 - 32) +
        (uint)abs(pc3 - 32);

    ulong penalty = ((ulong)imbalance << 18) + ((ulong)disagreement_q8 << 28);
    return penalty;
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
    score = bn_add_penalty_sat(score, bn_uncertainty_penalty(hv, disagreement_q8));

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
    uint pressure = max(verify_pressure_q8, max(submit_pressure_q8, stale_risk_q8));

    if (pressure >= 224u || job_age_ms >= 3200u) {
        topk = max(1u, topk / 2u);
    } else if (pressure >= 160u || job_age_ms >= 2400u) {
        topk = max(2u, topk - 2u);
    }

    return max(1u, min((uint)BN_LOCAL_TOPK, topk));
}

inline uint bn_effective_local_near_limit(
    uint effective_topk,
    uint job_age_ms,
    uint verify_pressure_q8,
    uint submit_pressure_q8,
    uint stale_risk_q8
) {
    uint pressure = max(verify_pressure_q8, max(submit_pressure_q8, stale_risk_q8));

    if (effective_topk <= 1u) {
        return 0u;
    }
    if (pressure >= 240u || job_age_ms >= 3500u) {
        return 0u;
    }
    if (pressure >= 128u || job_age_ms >= 2200u) {
        return 1u;
    }
    if (pressure >= 64u || job_age_ms >= 1200u) {
        return min(2u, effective_topk - 1u);
    }
    return max(1u, effective_topk / 2u);
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

inline void bn_prepare_work_blob(
    __private uchar* work_blob,
    __global const uchar* blob,
    uint blob_len,
    uint nonce_offset,
    uint nonce_u32
) {
    for (uint i = 0u; i < blob_len; ++i) {
        work_blob[i] = blob[i];
    }
    if (nonce_offset + 4u <= blob_len) {
        bn_store_u32_le_p(work_blob + nonce_offset, nonce_u32);
    }
}

inline ulong bn_private_load_u64_repeat(
    __private const uchar* p,
    uint len,
    uint off
) {
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

inline ulong bn_global_load_u64_repeat(
    __global const uchar* p,
    uint len,
    uint off
) {
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

inline ulong bn_dataset_read_mix(
    __global const ulong* dataset64,
    uint dataset_words,
    ulong addr
) {
    if (dataset_words == 0u) {
        return 0UL;
    }

    ulong line_base = addr & ~(ulong)(BN_RXP_LINE_WORDS - 1u);
    uint ix0 = (uint)(line_base % (ulong)dataset_words);
    uint ix1 = (ix0 + 1u) % dataset_words;
    uint ix2 = (ix0 + 3u) % dataset_words;
    uint ix3 = (ix0 + 7u) % dataset_words;
    uint ix4 = (ix0 + 13u) % dataset_words;
    uint ix5 = (ix0 + 29u) % dataset_words;

    ulong a = dataset64[ix0];
    ulong b = dataset64[ix1];
    ulong c = dataset64[ix2];
    ulong d = dataset64[ix3];
    ulong e = dataset64[ix4];
    ulong f = dataset64[ix5];

    ulong lane = bn_mix64(
        a ^
        bn_rotl64(b, 7u) ^
        bn_rotr64(c, 11u) ^
        d ^
        bn_rotl64(e, 19u) ^
        bn_rotr64(f, 23u) ^
        addr
    );

    return bn_avalanche64(lane ^ bn_rotl64(addr, (uint)(addr & 31u)));
}

inline uint bn_sp_index(uint level_sel, ulong addr) {
    uint mask_words;
    if (level_sel == 0u) {
        mask_words = BN_RXP_L1_WORDS;
    } else if (level_sel == 1u) {
        mask_words = BN_RXP_L2_WORDS;
    } else {
        mask_words = BN_RXP_SCRATCH_WORDS;
    }
    return (uint)(addr & (ulong)(mask_words - 1u));
}

inline float bn_u64_to_unit_float(ulong x) {
    uint bits = ((uint)x & 0x007FFFFFu) | 0x3F000000u;
    return as_float(bits) - 0.5f;
}

inline int bn_clamp_i32(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

inline ulong bn_pack_festate(float f, int e) {
    uint fb = as_uint(f);
    return ((ulong)fb) ^ (((ulong)((uint)(e & 0xFFFF))) << 32);
}

inline void bn_line_zero(__private ulong* line) {
    for (uint i = 0u; i < BN_RXP_LINE_WORDS; ++i) {
        line[i] = 0UL;
    }
}

inline void bn_line_copy(__private ulong* dst, __private const ulong* src) {
    for (uint i = 0u; i < BN_RXP_LINE_WORDS; ++i) {
        dst[i] = src[i];
    }
}

inline ulong bn_line_fold(__private const ulong* line) {
    ulong v = BN_U64_C(0x9E3779B97F4A7C15);
    for (uint i = 0u; i < BN_RXP_LINE_WORDS; ++i) {
        v = bn_mix64(v ^ line[i] ^ ((ulong)i << 32));
    }
    return v;
}

inline uint bn_cache_slot(ulong tag, uint lines) {
    return (uint)(tag % (ulong)lines);
}

inline void bn_cache_read_line(
    __private const ulong* data,
    uint slot,
    __private ulong* line_out
) {
    uint base = slot * BN_RXP_LINE_WORDS;
    for (uint i = 0u; i < BN_RXP_LINE_WORDS; ++i) {
        line_out[i] = data[base + i];
    }
}

inline void bn_cache_write_line(
    __private ulong* data,
    uint slot,
    __private const ulong* line_in
) {
    uint base = slot * BN_RXP_LINE_WORDS;
    for (uint i = 0u; i < BN_RXP_LINE_WORDS; ++i) {
        data[base + i] = line_in[i];
    }
}

inline ulong bn_virtual_tag(ulong word_addr, uint region_sel) {
    ulong line_id = word_addr >> 3;
    ulong salt = ((ulong)(region_sel + 1u) * BN_U64_C(0x9E3779B97F4A7C15));
    ulong virt = bn_mix64(line_id ^ salt ^ bn_rotl64(word_addr, region_sel + 3u));
    uint dataset_backed = (region_sel >= 2u) ? 1u : 0u;
    return (virt << 1) | (ulong)dataset_backed;
}

inline void bn_virtual_line_load(
    ulong tag,
    ulong salt,
    __private ulong* scratch,
    __global const ulong* dataset64,
    uint dataset_words,
    __private ulong* line_out
) {
    ulong line_no = tag >> 1;
    uint dataset_backed = (uint)(tag & 1UL);

    if (dataset_backed == 0u) {
        uint line_count = BN_RXP_SCRATCH_WORDS / BN_RXP_LINE_WORDS;
        uint base_line = (uint)(line_no % (ulong)line_count);
        uint base = base_line * BN_RXP_LINE_WORDS;
        for (uint i = 0u; i < BN_RXP_LINE_WORDS; ++i) {
            line_out[i] = scratch[(base + i) % BN_RXP_SCRATCH_WORDS];
        }
    } else {
        for (uint i = 0u; i < BN_RXP_LINE_WORDS; ++i) {
            ulong a = (line_no * (ulong)BN_RXP_LINE_WORDS) + (ulong)i;
            line_out[i] = bn_dataset_read_mix(
                dataset64,
                dataset_words,
                a ^ salt ^ ((ulong)i * BN_U64_C(0xD1B54A32D192ED03))
            );
        }
    }
}

inline void bn_virtual_line_writeback(
    ulong tag,
    __private const ulong* line_in,
    __private ulong* scratch,
    __private ulong* seed_mix,
    __private ulong regs[8]
) {
    ulong line_no = tag >> 1;
    uint dataset_backed = (uint)(tag & 1UL);

    if (dataset_backed == 0u) {
        uint line_count = BN_RXP_SCRATCH_WORDS / BN_RXP_LINE_WORDS;
        uint base_line = (uint)(line_no % (ulong)line_count);
        uint base = base_line * BN_RXP_LINE_WORDS;
        for (uint i = 0u; i < BN_RXP_LINE_WORDS; ++i) {
            scratch[(base + i) % BN_RXP_SCRATCH_WORDS] = line_in[i];
        }
    } else {
        uint mirror_line_count = BN_RXP_SCRATCH_WORDS / BN_RXP_LINE_WORDS;
        uint base_line = (uint)((line_no ^ (*seed_mix >> 17)) % (ulong)mirror_line_count);
        uint base = base_line * BN_RXP_LINE_WORDS;

        ulong fold = bn_line_fold(line_in) ^ tag;
        *seed_mix = bn_mix64(*seed_mix ^ fold);

        for (uint i = 0u; i < BN_RXP_LINE_WORDS; ++i) {
            ulong v = bn_mix64(line_in[i] ^ bn_rotl64(fold, i + 1u));
            scratch[(base + i) % BN_RXP_SCRATCH_WORDS] ^= v;
            regs[i & 7u] ^= bn_rotr64(v, (i * 7u + 3u) & 63u);
        }
    }
}

inline void bn_cache_install_l3(
    ulong tag,
    __private const ulong* line_in,
    uchar dirty,
    __private ulong* l3_data,
    __private ulong* l3_tag,
    __private uchar* l3_valid,
    __private uchar* l3_dirty,
    __private ulong* seed_mix,
    __private ulong regs[8],
    __private ulong* scratch
) {
    uint slot = bn_cache_slot(tag, BN_RXP_L3_LINES);

    if (l3_valid[slot] != (uchar)0 && l3_tag[slot] != tag && l3_dirty[slot] != (uchar)0) {
        __private ulong victim[BN_RXP_LINE_WORDS];
        bn_cache_read_line(l3_data, slot, victim);
        bn_virtual_line_writeback(l3_tag[slot], victim, scratch, seed_mix, regs);
    }

    bn_cache_write_line(l3_data, slot, line_in);
    l3_tag[slot] = tag;
    l3_valid[slot] = (uchar)1u;
    l3_dirty[slot] = dirty;
}

inline void bn_cache_install_l2(
    ulong tag,
    __private const ulong* line_in,
    uchar dirty,
    __private ulong* l2_data,
    __private ulong* l2_tag,
    __private uchar* l2_valid,
    __private uchar* l2_dirty,
    __private ulong* l3_data,
    __private ulong* l3_tag,
    __private uchar* l3_valid,
    __private uchar* l3_dirty,
    __private ulong* seed_mix,
    __private ulong regs[8],
    __private ulong* scratch
) {
    uint slot = bn_cache_slot(tag, BN_RXP_L2_LINES);

    if (l2_valid[slot] != (uchar)0 && l2_tag[slot] != tag) {
        __private ulong victim[BN_RXP_LINE_WORDS];
        bn_cache_read_line(l2_data, slot, victim);
        bn_cache_install_l3(
            l2_tag[slot],
            victim,
            l2_dirty[slot],
            l3_data,
            l3_tag,
            l3_valid,
            l3_dirty,
            seed_mix,
            regs,
            scratch
        );
    }

    bn_cache_write_line(l2_data, slot, line_in);
    l2_tag[slot] = tag;
    l2_valid[slot] = (uchar)1u;
    l2_dirty[slot] = dirty;
}

inline void bn_cache_install_l1(
    ulong tag,
    __private const ulong* line_in,
    uchar dirty,
    __private ulong* l1_data,
    __private ulong* l1_tag,
    __private uchar* l1_valid,
    __private uchar* l1_dirty,
    __private ulong* l2_data,
    __private ulong* l2_tag,
    __private uchar* l2_valid,
    __private uchar* l2_dirty,
    __private ulong* l3_data,
    __private ulong* l3_tag,
    __private uchar* l3_valid,
    __private uchar* l3_dirty,
    __private ulong* seed_mix,
    __private ulong regs[8],
    __private ulong* scratch
) {
    uint slot = bn_cache_slot(tag, BN_RXP_L1_LINES);

    if (l1_valid[slot] != (uchar)0 && l1_tag[slot] != tag) {
        __private ulong victim[BN_RXP_LINE_WORDS];
        bn_cache_read_line(l1_data, slot, victim);
        bn_cache_install_l2(
            l1_tag[slot],
            victim,
            l1_dirty[slot],
            l2_data,
            l2_tag,
            l2_valid,
            l2_dirty,
            l3_data,
            l3_tag,
            l3_valid,
            l3_dirty,
            seed_mix,
            regs,
            scratch
        );
    }

    bn_cache_write_line(l1_data, slot, line_in);
    l1_tag[slot] = tag;
    l1_valid[slot] = (uchar)1u;
    l1_dirty[slot] = dirty;
}

inline uint bn_cache_resolve_l1(
    ulong tag,
    ulong salt,
    __private ulong* scratch,
    __global const ulong* dataset64,
    uint dataset_words,
    __private ulong* l1_data,
    __private ulong* l1_tag,
    __private uchar* l1_valid,
    __private uchar* l1_dirty,
    __private ulong* l2_data,
    __private ulong* l2_tag,
    __private uchar* l2_valid,
    __private uchar* l2_dirty,
    __private ulong* l3_data,
    __private ulong* l3_tag,
    __private uchar* l3_valid,
    __private uchar* l3_dirty,
    __private ulong* seed_mix,
    __private ulong regs[8]
) {
    uint l1s = bn_cache_slot(tag, BN_RXP_L1_LINES);
    if (l1_valid[l1s] != (uchar)0 && l1_tag[l1s] == tag) {
        return l1s;
    }

    __private ulong line[BN_RXP_LINE_WORDS];
    uint l2s = bn_cache_slot(tag, BN_RXP_L2_LINES);
    if (l2_valid[l2s] != (uchar)0 && l2_tag[l2s] == tag) {
        bn_cache_read_line(l2_data, l2s, line);
        bn_cache_install_l1(
            tag, line, (uchar)0u,
            l1_data, l1_tag, l1_valid, l1_dirty,
            l2_data, l2_tag, l2_valid, l2_dirty,
            l3_data, l3_tag, l3_valid, l3_dirty,
            seed_mix, regs, scratch
        );
        return bn_cache_slot(tag, BN_RXP_L1_LINES);
    }

    uint l3s = bn_cache_slot(tag, BN_RXP_L3_LINES);
    if (l3_valid[l3s] != (uchar)0 && l3_tag[l3s] == tag) {
        bn_cache_read_line(l3_data, l3s, line);
        bn_cache_install_l2(
            tag, line, (uchar)0u,
            l2_data, l2_tag, l2_valid, l2_dirty,
            l3_data, l3_tag, l3_valid, l3_dirty,
            seed_mix, regs, scratch
        );
        bn_cache_install_l1(
            tag, line, (uchar)0u,
            l1_data, l1_tag, l1_valid, l1_dirty,
            l2_data, l2_tag, l2_valid, l2_dirty,
            l3_data, l3_tag, l3_valid, l3_dirty,
            seed_mix, regs, scratch
        );
        return bn_cache_slot(tag, BN_RXP_L1_LINES);
    }

    bn_virtual_line_load(tag, salt, scratch, dataset64, dataset_words, line);
    bn_cache_install_l3(
        tag, line, (uchar)0u,
        l3_data, l3_tag, l3_valid, l3_dirty,
        seed_mix, regs, scratch
    );
    bn_cache_install_l2(
        tag, line, (uchar)0u,
        l2_data, l2_tag, l2_valid, l2_dirty,
        l3_data, l3_tag, l3_valid, l3_dirty,
        seed_mix, regs, scratch
    );
    bn_cache_install_l1(
        tag, line, (uchar)0u,
        l1_data, l1_tag, l1_valid, l1_dirty,
        l2_data, l2_tag, l2_valid, l2_dirty,
        l3_data, l3_tag, l3_valid, l3_dirty,
        seed_mix, regs, scratch
    );
    return bn_cache_slot(tag, BN_RXP_L1_LINES);
}

inline ulong bn_cache_read64(
    ulong tag,
    uint word_ix,
    ulong salt,
    __private ulong* scratch,
    __global const ulong* dataset64,
    uint dataset_words,
    __private ulong* l1_data,
    __private ulong* l1_tag,
    __private uchar* l1_valid,
    __private uchar* l1_dirty,
    __private ulong* l2_data,
    __private ulong* l2_tag,
    __private uchar* l2_valid,
    __private uchar* l2_dirty,
    __private ulong* l3_data,
    __private ulong* l3_tag,
    __private uchar* l3_valid,
    __private uchar* l3_dirty,
    __private ulong* seed_mix,
    __private ulong regs[8]
) {
    uint slot = bn_cache_resolve_l1(
        tag, salt, scratch, dataset64, dataset_words,
        l1_data, l1_tag, l1_valid, l1_dirty,
        l2_data, l2_tag, l2_valid, l2_dirty,
        l3_data, l3_tag, l3_valid, l3_dirty,
        seed_mix, regs
    );
    return l1_data[(slot * BN_RXP_LINE_WORDS) + (word_ix & (BN_RXP_LINE_WORDS - 1u))];
}

inline void bn_cache_write64(
    ulong tag,
    uint word_ix,
    ulong value,
    ulong salt,
    __private ulong* scratch,
    __global const ulong* dataset64,
    uint dataset_words,
    __private ulong* l1_data,
    __private ulong* l1_tag,
    __private uchar* l1_valid,
    __private uchar* l1_dirty,
    __private ulong* l2_data,
    __private ulong* l2_tag,
    __private uchar* l2_valid,
    __private uchar* l2_dirty,
    __private ulong* l3_data,
    __private ulong* l3_tag,
    __private uchar* l3_valid,
    __private uchar* l3_dirty,
    __private ulong* seed_mix,
    __private ulong regs[8]
) {
    uint slot = bn_cache_resolve_l1(
        tag, salt, scratch, dataset64, dataset_words,
        l1_data, l1_tag, l1_valid, l1_dirty,
        l2_data, l2_tag, l2_valid, l2_dirty,
        l3_data, l3_tag, l3_valid, l3_dirty,
        seed_mix, regs
    );
    l1_data[(slot * BN_RXP_LINE_WORDS) + (word_ix & (BN_RXP_LINE_WORDS - 1u))] = value;
    l1_dirty[slot] = (uchar)1u;
}

inline void bn_cache_flush_all(
    __private ulong* scratch,
    __private ulong* l1_data,
    __private ulong* l1_tag,
    __private uchar* l1_valid,
    __private uchar* l1_dirty,
    __private ulong* l2_data,
    __private ulong* l2_tag,
    __private uchar* l2_valid,
    __private uchar* l2_dirty,
    __private ulong* l3_data,
    __private ulong* l3_tag,
    __private uchar* l3_valid,
    __private uchar* l3_dirty,
    __private ulong* seed_mix,
    __private ulong regs[8]
) {
    for (uint i = 0u; i < BN_RXP_L1_LINES; ++i) {
        if (l1_valid[i] != (uchar)0 && l1_dirty[i] != (uchar)0) {
            __private ulong line[BN_RXP_LINE_WORDS];
            bn_cache_read_line(l1_data, i, line);
            bn_cache_install_l2(
                l1_tag[i], line, (uchar)1u,
                l2_data, l2_tag, l2_valid, l2_dirty,
                l3_data, l3_tag, l3_valid, l3_dirty,
                seed_mix, regs, scratch
            );
            l1_dirty[i] = (uchar)0u;
        }
    }

    for (uint i = 0u; i < BN_RXP_L2_LINES; ++i) {
        if (l2_valid[i] != (uchar)0 && l2_dirty[i] != (uchar)0) {
            __private ulong line[BN_RXP_LINE_WORDS];
            bn_cache_read_line(l2_data, i, line);
            bn_cache_install_l3(
                l2_tag[i], line, (uchar)1u,
                l3_data, l3_tag, l3_valid, l3_dirty,
                seed_mix, regs, scratch
            );
            l2_dirty[i] = (uchar)0u;
        }
    }

    for (uint i = 0u; i < BN_RXP_L3_LINES; ++i) {
        if (l3_valid[i] != (uchar)0 && l3_dirty[i] != (uchar)0) {
            __private ulong line[BN_RXP_LINE_WORDS];
            bn_cache_read_line(l3_data, i, line);
            bn_virtual_line_writeback(l3_tag[i], line, scratch, seed_mix, regs);
            l3_dirty[i] = (uchar)0u;
        }
    }
}

/* gid-independent predictor state */
inline void bn_init_predict_vm(
    __private const uchar* work_blob,
    uint blob_len,
    __global const uchar* seed,
    uint seed_len,
    __global const ulong* dataset64,
    uint dataset_words,
    uint nonce_u32,
    __private ulong regs[8],
    __private ulong* ma,
    __private ulong* mx,
    __private ulong* seed_mix,
    __private ulong* rng_state
) {
    ulong nonce64 = (ulong)nonce_u32;

    ulong s0 = BN_U64_C(0x243F6A8885A308D3) ^ nonce64;
    ulong s1 = BN_U64_C(0x13198A2E03707344) ^ ((ulong)blob_len << 32) ^ bn_rotl64(nonce64, 7u);
    ulong s2 = BN_U64_C(0xA4093822299F31D0) ^ ((ulong)seed_len << 24) ^ bn_rotl64(nonce64, 13u);
    ulong s3 = BN_U64_C(0x082EFA98EC4E6C89) ^ bn_rotr64(nonce64, 9u);

    for (uint i = 0u; i < blob_len; ++i) {
        ulong v = (ulong)work_blob[i] | ((ulong)(i + 1u) << 8);
        s0 = bn_mix64(s0 ^ v ^ bn_rotl64(s3, 5u));
        s1 = bn_mix64(s1 + v + bn_rotr64(s0, 11u));
        s2 = bn_mix64(s2 ^ bn_rotl64(v + s1, 17u));
        s3 = bn_mix64(s3 + bn_rotr64(v ^ s2, 23u));
    }

    for (uint i = 0u; i < seed_len; ++i) {
        ulong v = (ulong)seed[i] | ((ulong)(i + 3u) << 9);
        s0 = bn_mix64(s0 ^ v ^ bn_rotl64(s2, 7u));
        s1 = bn_mix64(s1 + bn_rotr64(v ^ s0, 13u));
        s2 = bn_mix64(s2 ^ bn_rotl64(v + s1, 19u));
        s3 = bn_mix64(s3 + bn_rotr64(v ^ s2, 29u));
    }

    ulong ds0 = bn_dataset_read_mix(dataset64, dataset_words, s0 ^ s2);
    ulong ds1 = bn_dataset_read_mix(dataset64, dataset_words, s1 ^ s3);
    ulong ds2 = bn_dataset_read_mix(dataset64, dataset_words, s2 ^ s0 ^ nonce64);
    ulong ds3 = bn_dataset_read_mix(dataset64, dataset_words, s3 ^ s1 ^ nonce64);

    regs[0] = bn_avalanche64(s0 ^ ds0);
    regs[1] = bn_avalanche64(s1 ^ ds1);
    regs[2] = bn_avalanche64(s2 ^ ds2);
    regs[3] = bn_avalanche64(s3 ^ ds3);
    regs[4] = bn_avalanche64(s0 ^ s2 ^ ds1);
    regs[5] = bn_avalanche64(s1 ^ s3 ^ ds2);
    regs[6] = bn_avalanche64(s0 ^ s1 ^ ds3);
    regs[7] = bn_avalanche64(s2 ^ s3 ^ ds0);

    *ma = regs[0] ^ regs[4] ^ ds2;
    *mx = regs[3] ^ regs[7] ^ ds1;
    *seed_mix = bn_avalanche64(s0 ^ s1 ^ s2 ^ s3 ^ ds0 ^ ds1 ^ ds2 ^ ds3);
    *rng_state = bn_avalanche64(*seed_mix ^ regs[1] ^ regs[6] ^ nonce64);
}

inline void bn_fill_predict_scratchpad(
    __private ulong regs[8],
    __private ulong* ma,
    __private ulong* mx,
    __private ulong* rng_state,
    __private ulong* seed_mix,
    __global const ulong* dataset64,
    uint dataset_words,
    __private ulong scratch[BN_RXP_SCRATCH_WORDS]
) {
    uint line_count = BN_RXP_SCRATCH_WORDS / BN_RXP_LINE_WORDS;

    for (uint line = 0u; line < line_count; ++line) {
        ulong base = bn_rng_step(rng_state) ^ regs[line & 7u] ^ *ma ^ bn_rotl64(*mx, line & 31u);
        ulong ds = bn_dataset_read_mix(dataset64, dataset_words, base ^ ((ulong)line << 7));

        for (uint w = 0u; w < BN_RXP_LINE_WORDS; ++w) {
            ulong v = bn_mix64(
                base ^
                ds ^
                regs[(line + w) & 7u] ^
                bn_rotl64(*seed_mix, (line + w + 3u) & 63u) ^
                ((ulong)line << 32) ^
                (ulong)w
            );
            scratch[(line * BN_RXP_LINE_WORDS) + w] = v;
            base = bn_mix64(base + v + ((ulong)w << 17));
        }

        ulong fold = bn_line_fold(scratch + (line * BN_RXP_LINE_WORDS));
        regs[line & 7u] ^= bn_rotl64(fold, (line + 11u) & 63u);
        *ma = bn_mix64(*ma + fold + ds);
        *mx = bn_mix64(*mx ^ bn_rotr64(fold, (line + 5u) & 63u));
        *seed_mix = bn_mix64(*seed_mix ^ fold ^ (ulong)line);
    }
}

inline void bn_generate_program_words(
    __private ulong* rng_state,
    __private ulong regs[8],
    __private ulong* seed_mix,
    uint program_index,
    __private uint prog[BN_RXP_PROGRAM_SIZE]
) {
    ulong st = *rng_state ^ regs[program_index & 7u] ^ *seed_mix ^ ((ulong)program_index << 32);

    for (uint i = 0u; i < BN_RXP_PROGRAM_SIZE; ++i) {
        ulong w0 = bn_rng_step(&st);
        ulong w1 = bn_rng_step(&st);
        ulong w2 = bn_rng_step(&st);

        uint op = (uint)(w0 & 15u);
        uint dst = (uint)((w0 >> 4) & 7u);
        uint src = (uint)((w0 >> 7) & 7u);
        uint aux = (uint)((w0 >> 10) & 7u);
        uint sh = (uint)((w0 >> 13) & 63u);
        uint regA = (uint)((w1 >> 3) & 3u);
        uint regB = (uint)((w1 >> 5) & 3u);
        uint regionA = (uint)((w1 >> 7) & 3u);
        uint regionB = (uint)((w1 >> 9) & 3u);
        uint jump = (uint)((w2 >> 11) & (BN_RXP_PROGRAM_SIZE - 1u));

        prog[i] =
            (op) |
            (dst << 4) |
            (src << 7) |
            (aux << 10) |
            (sh << 13) |
            (regA << 19) |
            (regB << 21) |
            (regionA << 23) |
            (regionB << 25) |
            (jump << 27);
    }

    *rng_state = bn_mix64(st ^ regs[(program_index + 3u) & 7u]);
}

inline void bn_exec_predict_program(
    __private uint prog[BN_RXP_PROGRAM_SIZE],
    __private ulong regs[8],
    __private ulong* ma,
    __private ulong* mx,
    __private ulong* seed_mix,
    __private ulong* rng_state,
    __private ulong scratch[BN_RXP_SCRATCH_WORDS],
    __global const ulong* dataset64,
    uint dataset_words,
    uint program_index
) {
    __private float fstate[BN_RXP_FP_LANES];
    __private int estate[BN_RXP_FP_LANES];

    __private ulong l1_data[BN_RXP_L1_LINES * BN_RXP_LINE_WORDS];
    __private ulong l1_tag[BN_RXP_L1_LINES];
    __private uchar l1_valid[BN_RXP_L1_LINES];
    __private uchar l1_dirty[BN_RXP_L1_LINES];

    __private ulong l2_data[BN_RXP_L2_LINES * BN_RXP_LINE_WORDS];
    __private ulong l2_tag[BN_RXP_L2_LINES];
    __private uchar l2_valid[BN_RXP_L2_LINES];
    __private uchar l2_dirty[BN_RXP_L2_LINES];

    __private ulong l3_data[BN_RXP_L3_LINES * BN_RXP_LINE_WORDS];
    __private ulong l3_tag[BN_RXP_L3_LINES];
    __private uchar l3_valid[BN_RXP_L3_LINES];
    __private uchar l3_dirty[BN_RXP_L3_LINES];

    for (uint i = 0u; i < BN_RXP_FP_LANES; ++i) {
        float a = bn_u64_to_unit_float(regs[i]);
        float b = bn_u64_to_unit_float(regs[i + 4u]);
        fstate[i] = a + b;
        estate[i] = ((int)((regs[i] >> 58) & 31UL)) - 16;
    }

    for (uint i = 0u; i < BN_RXP_L1_LINES; ++i) {
        l1_tag[i] = 0UL;
        l1_valid[i] = (uchar)0u;
        l1_dirty[i] = (uchar)0u;
    }
    for (uint i = 0u; i < BN_RXP_L2_LINES; ++i) {
        l2_tag[i] = 0UL;
        l2_valid[i] = (uchar)0u;
        l2_dirty[i] = (uchar)0u;
    }
    for (uint i = 0u; i < BN_RXP_L3_LINES; ++i) {
        l3_tag[i] = 0UL;
        l3_valid[i] = (uchar)0u;
        l3_dirty[i] = (uchar)0u;
    }

    uint pc = 0u;

    for (uint iter = 0u; iter < BN_RXP_PROGRAM_ITERS; ++iter) {
        uint iw = prog[pc & (BN_RXP_PROGRAM_SIZE - 1u)];

        uint op = iw & 15u;
        uint dst = (iw >> 4) & 7u;
        uint src = (iw >> 7) & 7u;
        uint aux = (iw >> 10) & 7u;
        uint shift = ((iw >> 13) & 63u) + 1u;
        uint fsel = (iw >> 19) & 3u;
        uint esel = (iw >> 21) & 3u;
        uint regionA = (iw >> 23) & 3u;
        uint regionB = (iw >> 25) & 3u;
        uint jump = (iw >> 27) & (BN_RXP_PROGRAM_SIZE - 1u);

        float ff = fstate[fsel];
        int ee = estate[esel];

        ulong fpack = bn_pack_festate(ff, ee);
        ulong imm = bn_mix64(
            ((ulong)iw << 32) ^
            bn_rng_step(rng_state) ^
            regs[(dst + 1u) & 7u] ^
            *seed_mix ^
            ((ulong)iter << 17) ^
            ((ulong)program_index << 48) ^
            fpack
        );

        ulong addr0w = bn_mix64(
            regs[src] ^
            *ma ^
            imm ^
            (((ulong)((uint)(ee & 63))) << 25) ^
            (((ulong)as_uint(ff + 1.0f)) << 7)
        );
        ulong addr1w = bn_mix64(
            regs[aux] ^
            *mx ^
            bn_rotl64(imm, 13u) ^
            (((ulong)as_uint(ff * 0.5f + 1.0f)) << 11)
        );

        ulong tag0 = bn_virtual_tag(addr0w, regionA);
        ulong tag1 = bn_virtual_tag(addr1w, regionB);
        uint word0 = (uint)(addr0w & (ulong)(BN_RXP_LINE_WORDS - 1u));
        uint word1 = (uint)(addr1w & (ulong)(BN_RXP_LINE_WORDS - 1u));

        ulong prefetch0 = bn_virtual_tag(addr0w + (ulong)BN_RXP_LINE_WORDS + (ulong)(iter & 7u), regionA);
        ulong prefetch1 = bn_virtual_tag(addr1w + (ulong)BN_RXP_LINE_WORDS + (ulong)((iter + 3u) & 7u), regionB);

        (void)bn_cache_resolve_l1(
            prefetch0, *seed_mix ^ imm, scratch, dataset64, dataset_words,
            l1_data, l1_tag, l1_valid, l1_dirty,
            l2_data, l2_tag, l2_valid, l2_dirty,
            l3_data, l3_tag, l3_valid, l3_dirty,
            seed_mix, regs
        );
        (void)bn_cache_resolve_l1(
            prefetch1, *seed_mix ^ bn_rotl64(imm, 9u), scratch, dataset64, dataset_words,
            l1_data, l1_tag, l1_valid, l1_dirty,
            l2_data, l2_tag, l2_valid, l2_dirty,
            l3_data, l3_tag, l3_valid, l3_dirty,
            seed_mix, regs
        );

        ulong m0 = bn_cache_read64(
            tag0, word0, *seed_mix ^ imm, scratch, dataset64, dataset_words,
            l1_data, l1_tag, l1_valid, l1_dirty,
            l2_data, l2_tag, l2_valid, l2_dirty,
            l3_data, l3_tag, l3_valid, l3_dirty,
            seed_mix, regs
        );
        ulong m1 = bn_cache_read64(
            tag1, word1, *seed_mix ^ bn_rotl64(imm, 7u), scratch, dataset64, dataset_words,
            l1_data, l1_tag, l1_valid, l1_dirty,
            l2_data, l2_tag, l2_valid, l2_dirty,
            l3_data, l3_tag, l3_valid, l3_dirty,
            seed_mix, regs
        );

        if (op == 0u) {
            regs[dst] = bn_mix64(regs[dst] + bn_rotl64(m0 ^ regs[src], shift));
            regs[src] ^= bn_rotr64(m1 + imm, shift);
        } else if (op == 1u) {
            ulong hi = bn_mulh64(regs[src] | 1UL, m0 | 1UL);
            regs[dst] = bn_mix64(regs[dst] ^ hi ^ bn_rotl64(m1, shift));
            regs[aux] += bn_rotr64(hi ^ imm, (shift + 7u) & 63u);
        } else if (op == 2u) {
            ulong nv = bn_mix64(m0 ^ regs[dst] ^ imm);
            bn_cache_write64(
                tag0, word0, nv,
                *seed_mix ^ imm, scratch, dataset64, dataset_words,
                l1_data, l1_tag, l1_valid, l1_dirty,
                l2_data, l2_tag, l2_valid, l2_dirty,
                l3_data, l3_tag, l3_valid, l3_dirty,
                seed_mix, regs
            );
            regs[dst] ^= nv;
        } else if (op == 3u) {
            ulong nv = bn_mix64(m1 + regs[src] + bn_rotl64(imm, 5u));
            bn_cache_write64(
                tag1, word1, nv,
                *seed_mix ^ bn_rotl64(imm, 5u), scratch, dataset64, dataset_words,
                l1_data, l1_tag, l1_valid, l1_dirty,
                l2_data, l2_tag, l2_valid, l2_dirty,
                l3_data, l3_tag, l3_valid, l3_dirty,
                seed_mix, regs
            );
            regs[src] = bn_mix64(regs[src] + nv);
        } else if (op == 4u) {
            float fv = ff * (1.0625f + 0.015625f * (float)(word0 + 1u));
            fv += (float)((uint)(m0 ^ m1) & 0xFFFFu) * 0.000015258789f;
            int ne = ee + (int)((m1 >> 60) & 7UL) - 3;
            if (fv > 4.0f) {
                fv *= 0.5f;
                ne += 1;
            } else if (fv < -4.0f) {
                fv *= 0.5f;
                ne += 1;
            }
            ne = bn_clamp_i32(ne, -63, 63);
            fstate[fsel] = fv;
            estate[esel] = ne;
            regs[dst] ^= bn_mix64(bn_pack_festate(fv, ne) ^ m0 ^ bn_rotl64(m1, 9u));
        } else if (op == 5u) {
            ulong lane0 = bn_cache_read64(
                tag0, (word0 + 1u) & (BN_RXP_LINE_WORDS - 1u), *seed_mix ^ m0, scratch, dataset64, dataset_words,
                l1_data, l1_tag, l1_valid, l1_dirty,
                l2_data, l2_tag, l2_valid, l2_dirty,
                l3_data, l3_tag, l3_valid, l3_dirty,
                seed_mix, regs
            );
            ulong lane1 = bn_cache_read64(
                tag1, (word1 + 2u) & (BN_RXP_LINE_WORDS - 1u), *seed_mix ^ m1, scratch, dataset64, dataset_words,
                l1_data, l1_tag, l1_valid, l1_dirty,
                l2_data, l2_tag, l2_valid, l2_dirty,
                l3_data, l3_tag, l3_valid, l3_dirty,
                seed_mix, regs
            );
            regs[dst] = bn_mix64(regs[dst] ^ lane0 ^ bn_rotl64(lane1, shift));
            *ma = bn_mix64(*ma + lane0 + regs[src]);
            *mx = bn_mix64(*mx ^ lane1 ^ regs[aux]);
        } else if (op == 6u) {
            ulong cond = regs[dst] ^ m0 ^ bn_pack_festate(ff, ee);
            if (((cond >> BN_RXP_JUMP_MASK_BITS) & (ulong)((1u << BN_RXP_JUMP_MASK_BITS) - 1u)) == 0UL) {
                pc = jump;
            }
        } else if (op == 7u) {
            ulong a = bn_mix64(m0 ^ bn_rotl64(m1, 11u) ^ regs[src]);
            ulong b = bn_mix64(m1 ^ bn_rotr64(m0, 7u) ^ regs[aux]);
            bn_cache_write64(
                tag0, (word0 + 3u) & (BN_RXP_LINE_WORDS - 1u), a,
                *seed_mix ^ a, scratch, dataset64, dataset_words,
                l1_data, l1_tag, l1_valid, l1_dirty,
                l2_data, l2_tag, l2_valid, l2_dirty,
                l3_data, l3_tag, l3_valid, l3_dirty,
                seed_mix, regs
            );
            bn_cache_write64(
                tag1, (word1 + 5u) & (BN_RXP_LINE_WORDS - 1u), b,
                *seed_mix ^ b, scratch, dataset64, dataset_words,
                l1_data, l1_tag, l1_valid, l1_dirty,
                l2_data, l2_tag, l2_valid, l2_dirty,
                l3_data, l3_tag, l3_valid, l3_dirty,
                seed_mix, regs
            );
            regs[dst] ^= a ^ bn_rotr64(b, 17u);
        } else if (op == 8u) {
            ulong ds0 = bn_dataset_read_mix(dataset64, dataset_words, addr0w ^ regs[dst] ^ *seed_mix);
            ulong ds1 = bn_dataset_read_mix(dataset64, dataset_words, addr1w ^ regs[src] ^ bn_rotl64(*ma, 9u));
            regs[dst] = bn_mix64(regs[dst] + ds0 + bn_rotl64(m0, 7u));
            regs[src] = bn_mix64(regs[src] ^ ds1 ^ bn_rotr64(m1, 13u));
            *ma = bn_mix64(*ma + ds0);
            *mx = bn_mix64(*mx ^ ds1);
        } else if (op == 9u) {
            float fv = fstate[fsel];
            int ev = estate[esel];
            ulong t = bn_mix64(m0 ^ m1 ^ regs[dst] ^ bn_pack_festate(fv, ev));
            regs[dst] = bn_rotl64(t, shift);
            fstate[fsel] = fv + (float)((uint)t & 0x3FFu) * 0.00048828125f;
            estate[esel] = bn_clamp_i32(ev + (int)((t >> 61) & 3UL) - 1, -63, 63);
        } else if (op == 10u) {
            ulong swapv = regs[dst];
            regs[dst] = bn_mix64(regs[src] ^ m0);
            regs[src] = bn_mix64(swapv ^ m1 ^ imm);
        } else if (op == 11u) {
            ulong lane = bn_cache_read64(
                tag0, (word0 + 6u) & (BN_RXP_LINE_WORDS - 1u), *mx ^ imm, scratch, dataset64, dataset_words,
                l1_data, l1_tag, l1_valid, l1_dirty,
                l2_data, l2_tag, l2_valid, l2_dirty,
                l3_data, l3_tag, l3_valid, l3_dirty,
                seed_mix, regs
            );
            regs[dst] ^= bn_mix64(lane ^ m1 ^ bn_rotl64(imm, shift));
            *seed_mix = bn_mix64(*seed_mix ^ lane ^ regs[dst]);
        } else if (op == 12u) {
            ulong wr = bn_mix64(regs[dst] ^ regs[src] ^ regs[aux] ^ m0 ^ m1 ^ imm);
            bn_cache_write64(
                tag0, word0, wr,
                *seed_mix ^ wr, scratch, dataset64, dataset_words,
                l1_data, l1_tag, l1_valid, l1_dirty,
                l2_data, l2_tag, l2_valid, l2_dirty,
                l3_data, l3_tag, l3_valid, l3_dirty,
                seed_mix, regs
            );
            bn_cache_write64(
                tag1, word1, bn_rotr64(wr, 9u),
                *seed_mix ^ bn_rotr64(wr, 9u), scratch, dataset64, dataset_words,
                l1_data, l1_tag, l1_valid, l1_dirty,
                l2_data, l2_tag, l2_valid, l2_dirty,
                l3_data, l3_tag, l3_valid, l3_dirty,
                seed_mix, regs
            );
        } else if (op == 13u) {
            ulong branch_mix = bn_mix64(regs[dst] ^ *ma ^ *mx ^ m0 ^ m1);
            if (((branch_mix >> 59) & 7UL) <= 1UL) {
                pc = (pc + jump + 1u) & (BN_RXP_PROGRAM_SIZE - 1u);
            }
            regs[aux] ^= branch_mix;
        } else if (op == 14u) {
            ulong hi = bn_mulh64(m0 | 1UL, m1 | 1UL);
            regs[dst] = bn_mix64(regs[dst] + hi + bn_rotl64(imm, 3u));
            regs[aux] ^= bn_rotr64(hi, shift);
            *ma = bn_mix64(*ma + hi);
            *mx = bn_mix64(*mx ^ hi);
        } else {
            ulong line_mix = bn_mix64(m0 ^ bn_rotl64(m1, 17u) ^ regs[dst] ^ imm);
            float fv = fstate[fsel] * 0.875f + (float)((uint)line_mix & 0x7FFu) * 0.000244140625f;
            int ev = bn_clamp_i32(estate[esel] + (int)((line_mix >> 62) & 3UL) - 1, -63, 63);
            fstate[fsel] = fv;
            estate[esel] = ev;
            regs[dst] ^= bn_pack_festate(fv, ev);
            *seed_mix = bn_mix64(*seed_mix ^ line_mix ^ bn_pack_festate(fv, ev));
        }

        regs[(dst + 5u) & 7u] ^= bn_rotl64(regs[dst], (iter + shift) & 63u);
        *ma = bn_mix64(*ma + regs[(dst + 3u) & 7u] + m0);
        *mx = bn_mix64(*mx ^ regs[(src + 5u) & 7u] ^ m1);
        *seed_mix = bn_mix64(*seed_mix ^ regs[dst] ^ regs[src] ^ m0 ^ m1 ^ (ulong)iter);

        pc = (pc + 1u) & (BN_RXP_PROGRAM_SIZE - 1u);
    }

    bn_cache_flush_all(
        scratch,
        l1_data, l1_tag, l1_valid, l1_dirty,
        l2_data, l2_tag, l2_valid, l2_dirty,
        l3_data, l3_tag, l3_valid, l3_dirty,
        seed_mix, regs
    );

    for (uint i = 0u; i < BN_RXP_FP_LANES; ++i) {
        ulong fe = bn_pack_festate(fstate[i], estate[i]);
        regs[i] ^= bn_mix64(fe ^ regs[i + 4u] ^ ((ulong)i << 40));
        *seed_mix = bn_mix64(*seed_mix ^ fe ^ regs[i]);
    }

    for (uint i = 0u; i < BN_RXP_L1_LINES; ++i) {
        if (l1_valid[i] != (uchar)0) {
            *seed_mix = bn_mix64(*seed_mix ^ l1_tag[i] ^ ((ulong)l1_dirty[i] << 48));
        }
    }
    for (uint i = 0u; i < BN_RXP_L2_LINES; ++i) {
        if (l2_valid[i] != (uchar)0) {
            regs[i & 7u] ^= bn_mix64(l2_tag[i] ^ ((ulong)l2_dirty[i] << 40));
        }
    }
    for (uint i = 0u; i < BN_RXP_L3_LINES; ++i) {
        if (l3_valid[i] != (uchar)0) {
            *ma ^= bn_mix64(l3_tag[i] ^ ((ulong)l3_dirty[i] << 32));
        }
    }
}

inline void bn_digest_predict_state(
    __private ulong regs[8],
    __private ulong* ma,
    __private ulong* mx,
    __private ulong* seed_mix,
    __private ulong scratch[BN_RXP_SCRATCH_WORDS],
    __global const ulong* dataset64,
    uint dataset_words,
    uint nonce_u32,
    __private ulong outv[4]
) {
    ulong s0 = regs[0] ^ bn_rotl64(regs[4], 11u) ^ *ma ^ *seed_mix;
    ulong s1 = regs[1] ^ bn_rotr64(regs[5], 7u) ^ *mx;
    ulong s2 = regs[2] ^ bn_rotl64(regs[6], 17u) ^ *seed_mix;
    ulong s3 = regs[3] ^ bn_rotr64(regs[7], 13u) ^ *ma ^ *mx;

    uint line_count = BN_RXP_SCRATCH_WORDS / BN_RXP_LINE_WORDS;
    for (uint line = 0u; line < line_count; ++line) {
        ulong fold = bn_line_fold(scratch + (line * BN_RXP_LINE_WORDS));
        ulong ds = bn_dataset_read_mix(
            dataset64,
            dataset_words,
            fold ^ ((ulong)line << 9) ^ (ulong)nonce_u32
        );

        s0 = bn_mix64(s0 ^ fold ^ bn_rotl64(ds, 5u));
        s1 = bn_mix64(s1 + fold + bn_rotr64(ds, 11u));
        s2 = bn_mix64(s2 ^ bn_rotl64(fold ^ ds, 17u));
        s3 = bn_mix64(s3 + bn_rotr64(fold + ds, 23u));
    }

    for (uint i = 0u; i < BN_TAIL_PREDICT_ROUNDS + 12u; ++i) {
        ulong ds0 = bn_dataset_read_mix(dataset64, dataset_words, s0 ^ s2 ^ (ulong)i ^ (ulong)nonce_u32);
        ulong ds1 = bn_dataset_read_mix(dataset64, dataset_words, s1 ^ s3 ^ ((ulong)i << 12) ^ *seed_mix);
        ulong x = bn_mix64(ds0 ^ bn_rotl64(ds1, (i + 5u) & 63u) ^ *seed_mix);

        s0 = bn_avalanche64(s0 ^ x ^ bn_rotl64(s3, 9u));
        s1 = bn_avalanche64(s1 + x + bn_rotr64(s0, 13u));
        s2 = bn_avalanche64(s2 ^ x ^ bn_rotl64(s1, 19u));
        s3 = bn_avalanche64(s3 + x + bn_rotr64(s2, 27u));
    }

    outv[0] = bn_avalanche64(s0 ^ regs[0] ^ regs[5]);
    outv[1] = bn_avalanche64(s1 ^ regs[1] ^ regs[6]);
    outv[2] = bn_avalanche64(s2 ^ regs[2] ^ regs[7]);
    outv[3] = bn_avalanche64(s3 ^ regs[3] ^ regs[4] ^ *seed_mix);
}

inline void bn_gpu_dataset_prefilter_hash(
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
    __private uchar work_blob[BN_MAX_BLOB_BYTES];
    __private ulong regs[8];
    __private ulong scratch[BN_RXP_SCRATCH_WORDS];
    __private uint prog[BN_RXP_PROGRAM_SIZE];

    if (blob_len == 0u || blob_len > BN_MAX_BLOB_BYTES || dataset_words == 0u) {
        outv[0] = 0UL;
        outv[1] = 0UL;
        outv[2] = 0UL;
        outv[3] = ~0UL;
        return;
    }

    bn_prepare_work_blob(work_blob, blob, blob_len, nonce_offset, nonce_u32);

    ulong ma, mx, seed_mix, rng_state;
    bn_init_predict_vm(
        work_blob,
        blob_len,
        seed,
        seed_len,
        dataset64,
        dataset_words,
        nonce_u32,
        regs,
        &ma,
        &mx,
        &seed_mix,
        &rng_state
    );

    bn_fill_predict_scratchpad(
        regs,
        &ma,
        &mx,
        &rng_state,
        &seed_mix,
        dataset64,
        dataset_words,
        scratch
    );

    for (uint p = 0u; p < BN_RXP_PROGRAM_COUNT; ++p) {
        bn_generate_program_words(&rng_state, regs, &seed_mix, p, prog);
        bn_exec_predict_program(
            prog,
            regs,
            &ma,
            &mx,
            &seed_mix,
            &rng_state,
            scratch,
            dataset64,
            dataset_words,
            p
        );

        ulong fold = bn_mix64(
            regs[p & 7u] ^
            regs[(p + 3u) & 7u] ^
            ma ^
            bn_rotl64(mx, (p + 7u) & 63u) ^
            seed_mix ^
            bn_rng_step(&rng_state)
        );

        regs[(p + 0u) & 7u] ^= fold;
        regs[(p + 2u) & 7u] = bn_mix64(regs[(p + 2u) & 7u] + fold);
        regs[(p + 5u) & 7u] ^= bn_rotr64(fold, 17u);
        seed_mix = bn_mix64(seed_mix ^ fold ^ (ulong)p);

        uint ix = (uint)(fold & (BN_RXP_SCRATCH_WORDS - 1u));
        scratch[ix] ^= bn_dataset_read_mix(dataset64, dataset_words, fold ^ regs[ix & 7u]);
    }

    bn_digest_predict_state(
        regs,
        &ma,
        &mx,
        &seed_mix,
        scratch,
        dataset64,
        dataset_words,
        nonce_u32,
        outv
    );
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

    uint active = local_size;
    if (active > BN_LOCAL_STAGE_SIZE) {
        active = BN_LOCAL_STAGE_SIZE;
    }

    uint effective_topk = bn_effective_local_topk(
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8
    );
    effective_topk = min(effective_topk, max_results);

    uint near_limit = bn_effective_local_near_limit(
        effective_topk,
        job_age_ms,
        verify_pressure_q8,
        submit_pressure_q8,
        stale_risk_q8
    );

    uchar used[BN_LOCAL_STAGE_SIZE];
    for (uint i = 0u; i < active; ++i) {
        used[i] = (uchar)0u;
    }

    uint chosen_bucket[BN_LOCAL_TOPK];
    uchar chosen_tailbin[BN_LOCAL_TOPK];

    uint selected_count = 0u;
    uint near_picked = 0u;

    /* pass first */
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
        chosen_bucket[selected_count] = l_bucket[best_i];
        chosen_tailbin[selected_count] = l_tailbin[best_i];

        uint slot = bn_reserve_slot(out_count);
        if (slot < max_results) {
            out_nonces[slot] = l_nonce[best_i];
            out_scores[slot] = l_score[best_i];
            out_buckets[slot] = l_bucket[best_i];
            out_rankq[slot] = l_rankq[best_i];
            out_threshq[slot] = l_threshq[best_i];
            out_tailbin[slot] = l_tailbin[best_i];
            bn_write_hash32(
                out_hashes + (slot * BN_HASH_BYTES),
                l_h0[best_i],
                l_h1[best_i],
                l_h2[best_i],
                l_h3[best_i]
            );
        }

        ++selected_count;
    }

    /* controlled near reserve */
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
        chosen_bucket[selected_count] = l_bucket[best_i];
        chosen_tailbin[selected_count] = l_tailbin[best_i];

        uint slot = bn_reserve_slot(out_count);
        if (slot < max_results) {
            out_nonces[slot] = l_nonce[best_i];
            out_scores[slot] = l_score[best_i];
            out_buckets[slot] = l_bucket[best_i];
            out_rankq[slot] = l_rankq[best_i];
            out_threshq[slot] = l_threshq[best_i];
            out_tailbin[slot] = l_tailbin[best_i];
            bn_write_hash32(
                out_hashes + (slot * BN_HASH_BYTES),
                l_h0[best_i],
                l_h1[best_i],
                l_h2[best_i],
                l_h3[best_i]
            );
        }

        ++selected_count;
        ++near_picked;
    }

    /* fill with diversity-aware adjusted rank */
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
        chosen_bucket[selected_count] = l_bucket[best_i];
        chosen_tailbin[selected_count] = l_tailbin[best_i];

        if ((uint)l_class[best_i] == BN_STAGE_NEAR) {
            ++near_picked;
        }

        uint slot = bn_reserve_slot(out_count);
        if (slot < max_results) {
            out_nonces[slot] = l_nonce[best_i];
            out_scores[slot] = l_score[best_i];
            out_buckets[slot] = l_bucket[best_i];
            out_rankq[slot] = l_rankq[best_i];
            out_threshq[slot] = l_threshq[best_i];
            out_tailbin[slot] = l_tailbin[best_i];
            bn_write_hash32(
                out_hashes + (slot * BN_HASH_BYTES),
                l_h0[best_i],
                l_h1[best_i],
                l_h2[best_i],
                l_h3[best_i]
            );
        }

        ++selected_count;
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

    uint active_tail_bins = max(1u, max(seed_tune_tail_bins, job_tune_tail_bins));
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
    uint confidence_q = max(seed_conf_q, job_conf_q);

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

    if (soft_tail < adjusted_target || tail_best < adjusted_target) {
        stage_class = BN_STAGE_PASS;
    } else if (tail_best < near_target && confidence_q >= 16u) {
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
    const uint lid = get_local_id(0);
    const uint local_size = get_local_size(0);
    const uint nonce_u32 = start_nonce + (uint)get_global_id(0);

    __private ulong hv[4];

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

    bn_gpu_dataset_prefilter_hash(
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
    const uint lid = get_local_id(0);
    const uint local_size = get_local_size(0);
    const uint nonce_u32 = start_nonce + (uint)get_global_id(0);

    __private ulong hv[4];

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

    bn_gpu_dataset_prefilter_hash(
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