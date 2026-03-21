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
#define BN_LOCAL_TOPK 64u
#endif

#ifndef BN_PREFILTER_ROUNDS_FAST
#define BN_PREFILTER_ROUNDS_FAST 20u
#endif

#ifndef BN_FINAL_MIX_ROUNDS_FAST
#define BN_FINAL_MIX_ROUNDS_FAST 6u
#endif

#ifndef BN_FAST_ABSORB_STRIDE
#define BN_FAST_ABSORB_STRIDE 4u
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