/// Hardware-accelerated NTT for Apple Silicon (aarch64 with NEON).
///
/// This module provides an optimized NTT implementation using:
/// 1. Montgomery arithmetic (eliminates expensive modular division)
/// 2. NEON vector intrinsics (processes butterflies 2-at-a-time)
/// 3. Precomputed twiddle factor tables (trades memory for speed)
/// 4. Loop unrolling and prefetch-friendly memory access patterns
///
/// The existing `ntt.rs` module is preserved as the reference/fallback implementation.

use crate::montgomery::MontCtx32;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Precomputed tables for the negacyclic NTT over `Z_q[X]/(X^N + 1)`.
///
/// The butterfly tables hold powers of `omega = psi^2` — see the warning in
/// `ntt.rs`. The psi tables carry the negacyclic twist, which is part of the
/// transform, not setup.
#[derive(Debug, Clone)]
pub struct TwiddleTable {
    /// Twiddle factors for the forward NTT, indexed by stage and position.
    /// Layout: for stage s (len = 2^(s+1)), half = 2^s butterflies per group,
    /// we store `half` twiddle factors. Total storage: N-1 entries.
    /// Powers of `omega = psi^2`, in Montgomery form.
    pub forward: Vec<u64>,
    /// Twiddle factors for the inverse NTT (powers of `omega^(-1)`, Montgomery form).
    pub inverse: Vec<u64>,
    /// Fused forward twist: `mont_psi_powers[i] = psi^i * R^2 mod q`.
    /// One `mont_mul(coeff[i], mont_psi_powers[i])` applies the psi twist *and*
    /// converts into the Montgomery domain, so the twist replaces the standalone
    /// `to_mont` pass instead of adding a pass.
    pub mont_psi_powers: Vec<u64>,
    /// Inverse untwist: `psi_inv_powers[i] = psi^(-i) mod q` in **ordinary** form.
    /// One `mont_mul(coeff[i], psi_inv_powers[i])` both untwists and leaves the
    /// Montgomery domain, replacing the standalone `from_mont` pass.
    pub psi_inv_powers: Vec<u64>,
    /// N (polynomial degree)
    pub n: usize,
}

/// Precompute all tables for a given N, psi, and Montgomery context.
///
/// `psi` must be a primitive 2N-th root of unity mod q (see
/// `ntt::find_primitive_2n_root`); `omega = psi^2` is derived here so the
/// butterfly tables can never accidentally be built from psi itself.
pub fn precompute_twiddles(n: usize, psi: u64, mont: &MontCtx32) -> TwiddleTable {
    let q = mont.q;
    // omega = psi^2 mod q. Products of reduced residues below q < 2^32 fit u64.
    let omega = (psi % q) * (psi % q) % q;
    let omega_mont = mont.to_mont(omega);

    // Compute omega_inv in Montgomery form
    // omega_inv = omega^(q-2) mod q (Fermat's little theorem)
    let omega_inv_mont = mont.mont_pow(omega_mont, mont.q - 2);

    let mut forward = Vec::with_capacity(n);
    let mut inverse = Vec::with_capacity(n);

    // For each stage: len goes 2, 4, 8, ..., n
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        // w_step = omega^(n/len) in Montgomery form
        let step_exp = (n / len) as u64;
        let w_step_fwd = mont.mont_pow(omega_mont, step_exp);
        let w_step_inv = mont.mont_pow(omega_inv_mont, step_exp);

        // Compute successive powers: 1, w_step, w_step^2, ..., w_step^(half-1)
        let one_mont = mont.r_mod_q; // 1 in Montgomery form = R mod q
        let mut w_fwd = one_mont;
        let mut w_inv = one_mont;

        for _ in 0..half {
            forward.push(w_fwd);
            inverse.push(w_inv);
            w_fwd = mont.mont_mul(w_fwd, w_step_fwd);
            w_inv = mont.mont_mul(w_inv, w_step_inv);
        }

        len <<= 1;
    }

    // psi twist tables.
    // Forward: double conversion, psi^i * R^2 mod q, so a single mont_mul both
    // twists and enters the Montgomery domain.
    // Inverse: psi^(-i) in ordinary form, so a single mont_mul both untwists and
    // exits the Montgomery domain.
    let psi_inv = mont_inverse(psi % q, mont);
    let mut mont_psi_powers = Vec::with_capacity(n);
    let mut psi_inv_powers = Vec::with_capacity(n);
    let mut psi_pow = 1u64;
    let mut psi_inv_pow = 1u64;
    for _ in 0..n {
        mont_psi_powers.push(mont.to_mont(mont.to_mont(psi_pow)));
        psi_inv_powers.push(psi_inv_pow);
        psi_pow = psi_pow * (psi % q) % q;
        psi_inv_pow = psi_inv_pow * psi_inv % q;
    }

    TwiddleTable {
        forward,
        inverse,
        mont_psi_powers,
        psi_inv_powers,
        n,
    }
}

/// Modular inverse via Fermat, computed in the Montgomery domain.
fn mont_inverse(a: u64, mont: &MontCtx32) -> u64 {
    mont.from_mont(mont.mont_pow(mont.to_mont(a), mont.q - 2))
}

/// Bit-reversal permutation for in-place NTT.
#[inline]
fn bit_reverse(data: &mut [u64]) {
    let n = data.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
}

/// Scalar Montgomery butterfly: processes a single butterfly pair.
#[inline(always)]
fn scalar_butterfly(data: &mut [u64], i: usize, j: usize, w: u64, mont: &MontCtx32) {
    let u = data[i];
    let v = mont.mont_mul(data[j], w);
    data[i] = mont.mont_add(u, v);
    data[j] = mont.mont_sub(u, v);
}

/// Forward negacyclic NTT over `Z_q[X]/(X^N + 1)` using Montgomery arithmetic
/// with precomputed tables.
///
/// The psi twist and the entry into the Montgomery domain are fused into one
/// pass: `mont_psi_powers[i] = psi^i * R^2 mod q`, so
/// `mont_mul(a[i], psi^i * R^2) = (a[i] * psi^i) * R mod q`.
///
/// On aarch64, uses NEON loads/stores with 4x-unrolled butterflies.
/// Falls back to scalar Montgomery on other architectures.
pub fn forward_ntt_accel(data: &mut [u64], twiddles: &TwiddleTable, mont: &MontCtx32) -> Result<(), ()> {
    let n = data.len();
    if n == 0 || (n & (n - 1)) != 0 || n != twiddles.n || twiddles.mont_psi_powers.len() != n {
        return Err(());
    }

    // Fused negacyclic twist + Montgomery domain entry (one pass, not two).
    for i in 0..n {
        data[i] = mont.mont_mul(data[i], twiddles.mont_psi_powers[i]);
    }

    bit_reverse(data);

    // Process butterfly stages
    #[cfg(target_arch = "aarch64")]
    {
        ntt_butterfly_stages_neon(data, &twiddles.forward, mont);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        ntt_butterfly_stages_scalar(data, &twiddles.forward, mont);
    }

    // Convert back from Montgomery form
    for val in data.iter_mut() {
        *val = mont.from_mont(*val);
    }

    Ok(())
}

/// Inverse negacyclic NTT over `Z_q[X]/(X^N + 1)`.
/// Runs the cyclic inverse core with `omega^(-1)`, scales by `N^(-1)`, then
/// untwists by `psi^(-i)`. The untwist is fused with the exit from the
/// Montgomery domain (`psi_inv_powers` are in ordinary form), so the untwist
/// replaces the `from_mont` pass rather than adding one.
pub fn inverse_ntt_accel(data: &mut [u64], twiddles: &TwiddleTable, mont: &MontCtx32) -> Result<(), ()> {
    let n = data.len();
    if n == 0 || (n & (n - 1)) != 0 || n != twiddles.n || twiddles.psi_inv_powers.len() != n {
        return Err(());
    }

    // Convert all data to Montgomery form
    for val in data.iter_mut() {
        *val = mont.to_mont(*val);
    }

    bit_reverse(data);

    // Process butterfly stages with inverse twiddles
    #[cfg(target_arch = "aarch64")]
    {
        ntt_butterfly_stages_neon(data, &twiddles.inverse, mont);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        ntt_butterfly_stages_scalar(data, &twiddles.inverse, mont);
    }

    // Scale by N^(-1) mod q (in Montgomery form)
    // N^(-1) mod q via Fermat: N^(q-2) mod q
    let n_inv_mont = {
        let n_mont = mont.to_mont(n as u64);
        mont.mont_pow(n_mont, mont.q - 2)
    };

    for val in data.iter_mut() {
        *val = mont.mont_mul(*val, n_inv_mont);
    }

    // Fused psi^(-i) untwist + Montgomery domain exit (one pass, not two):
    // mont_mul(aR, psi^(-i)) = a * psi^(-i) mod q.
    for i in 0..n {
        data[i] = mont.mont_mul(data[i], twiddles.psi_inv_powers[i]);
    }

    Ok(())
}

/// Verify the negacyclic convolution identity on the accelerated path:
/// `INTT(NTT(a) . NTT(b)) == a * b in Z_q[X]/(X^N + 1)`.
///
/// Uses the same deterministic sparse operands and the same O(N) schoolbook
/// reference as the reference-path gate in `ntt`, so a failure here is either a
/// bad (N, q) pair or a divergence in the accelerated implementation.
pub fn verify_negacyclic_convolution_accel(twiddles: &TwiddleTable, mont: &MontCtx32) -> bool {
    let n = twiddles.n;
    let q = mont.q;
    let (a, b, expected) = crate::ntt::convolution_test_vectors(n, q);

    let mut fa = a;
    let mut fb = b;
    if forward_ntt_accel(&mut fa, twiddles, mont).is_err() {
        return false;
    }
    if forward_ntt_accel(&mut fb, twiddles, mont).is_err() {
        return false;
    }

    // Pointwise product. Both operands are < q < 2^32, so the product fits u64.
    let mut fc: Vec<u64> = (0..n).map(|i| fa[i] * fb[i] % q).collect();
    if inverse_ntt_accel(&mut fc, twiddles, mont).is_err() {
        return false;
    }

    fc == expected
}

/// Scalar butterfly stages (fallback for non-aarch64).
#[cfg(not(target_arch = "aarch64"))]
fn ntt_butterfly_stages_scalar(data: &mut [u64], twiddle_table: &[u64], mont: &MontCtx32) {
    let n = data.len();
    let mut twiddle_offset = 0;
    let mut len = 2;

    while len <= n {
        let half = len / 2;
        let twiddles = &twiddle_table[twiddle_offset..twiddle_offset + half];
        twiddle_offset += half;

        let mut i = 0;
        while i < n {
            for j in 0..half {
                let w = twiddles[j];
                scalar_butterfly(data, i + j, i + j + half, w, mont);
            }
            i += len;
        }
        len <<= 1;
    }
}

/// NEON-accelerated butterfly stages for aarch64.
/// Processes butterflies 2 at a time using 128-bit NEON registers where beneficial.
/// Falls back to scalar for small stages or odd-count butterflies.
#[cfg(target_arch = "aarch64")]
fn ntt_butterfly_stages_neon(data: &mut [u64], twiddle_table: &[u64], mont: &MontCtx32) {
    let n = data.len();
    let mut twiddle_offset = 0;
    let mut len = 2;

    while len <= n {
        let half = len / 2;
        let twiddles = &twiddle_table[twiddle_offset..twiddle_offset + half];
        twiddle_offset += half;

        // For small half (< 4), scalar is fine since NEON overhead isn't worth it
        if half < 4 {
            let mut i = 0;
            while i < n {
                for j in 0..half {
                    let w = twiddles[j];
                    scalar_butterfly(data, i + j, i + j + half, w, mont);
                }
                i += len;
            }
        } else {
            // Process groups with 4× unrolled scalar (Montgomery mul doesn't vectorize
            // well with NEON u64 since NEON lacks native 64×64→128 multiply).
            // Instead, we unroll 4 butterflies for ILP (instruction-level parallelism).
            let mut i = 0;
            while i < n {
                let mut j = 0;
                // 4× unrolled loop
                while j + 3 < half {
                    unsafe {
                        neon_butterfly_4x(data, i + j, i + j + half, &twiddles[j..j + 4], mont);
                    }
                    j += 4;
                }
                // Handle remainder
                while j < half {
                    let w = twiddles[j];
                    scalar_butterfly(data, i + j, i + j + half, w, mont);
                    j += 1;
                }
                i += len;
            }
        }
        len <<= 1;
    }
}

/// Process 4 butterflies with explicit ILP (instruction-level parallelism).
/// On Apple Silicon's wide out-of-order core, 4 independent mul chains can execute
/// simultaneously across the multiple execution units.
///
/// We also use NEON loads/stores for prefetch-friendly memory access patterns.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_butterfly_4x(
    data: &mut [u64],
    top_base: usize,
    bot_base: usize,
    twiddles: &[u64],
    mont: &MontCtx32,
) {
    // Load 4 top values and 4 bottom values
    // Using NEON for 2×u64 loads (128-bit) where alignment allows
    let ptr_top = data.as_ptr().add(top_base);
    let ptr_bot = data.as_ptr().add(bot_base);

    // Load pairs using NEON
    let u01 = vld1q_u64(ptr_top);
    let u23 = vld1q_u64(ptr_top.add(2));
    let v01 = vld1q_u64(ptr_bot);
    let v23 = vld1q_u64(ptr_bot.add(2));

    // Extract scalar values for Montgomery multiplication
    // (NEON doesn't have native u64×u64 multiply, so we do scalar mul but NEON load/store)
    let u0 = vgetq_lane_u64(u01, 0);
    let u1 = vgetq_lane_u64(u01, 1);
    let u2 = vgetq_lane_u64(u23, 0);
    let u3 = vgetq_lane_u64(u23, 1);

    let v0 = vgetq_lane_u64(v01, 0);
    let v1 = vgetq_lane_u64(v01, 1);
    let v2 = vgetq_lane_u64(v23, 0);
    let v3 = vgetq_lane_u64(v23, 1);

    // Montgomery multiply: wv = w * v in Montgomery form
    // These 4 independent multiplications can execute in parallel on M4's wide core
    let wv0 = mont.mont_mul(v0, twiddles[0]);
    let wv1 = mont.mont_mul(v1, twiddles[1]);
    let wv2 = mont.mont_mul(v2, twiddles[2]);
    let wv3 = mont.mont_mul(v3, twiddles[3]);

    // Butterfly: top = u + wv, bottom = u - wv
    let top0 = mont.mont_add(u0, wv0);
    let top1 = mont.mont_add(u1, wv1);
    let top2 = mont.mont_add(u2, wv2);
    let top3 = mont.mont_add(u3, wv3);

    let bot0 = mont.mont_sub(u0, wv0);
    let bot1 = mont.mont_sub(u1, wv1);
    let bot2 = mont.mont_sub(u2, wv2);
    let bot3 = mont.mont_sub(u3, wv3);

    // Store results using NEON
    let result_top_01 = vcombine_u64(vcreate_u64(top0), vcreate_u64(top1));
    let result_top_23 = vcombine_u64(vcreate_u64(top2), vcreate_u64(top3));
    let result_bot_01 = vcombine_u64(vcreate_u64(bot0), vcreate_u64(bot1));
    let result_bot_23 = vcombine_u64(vcreate_u64(bot2), vcreate_u64(bot3));

    let ptr_top_mut = data.as_mut_ptr().add(top_base);
    let ptr_bot_mut = data.as_mut_ptr().add(bot_base);

    vst1q_u64(ptr_top_mut, result_top_01);
    vst1q_u64(ptr_top_mut.add(2), result_top_23);
    vst1q_u64(ptr_bot_mut, result_bot_01);
    vst1q_u64(ptr_bot_mut.add(2), result_bot_23);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt;

    /// Helper: build psi, a Montgomery context, tables, and the matching
    /// reference plan for testing.
    fn setup_ntt(n: usize, q: u64) -> Option<(MontCtx32, TwiddleTable, ntt::NegacyclicPlan)> {
        let psi = ntt::find_primitive_2n_root(n as u64, q)?;
        let plan = ntt::NegacyclicPlan::new(n, q)?;
        assert_eq!(plan.psi, psi);
        let mont = MontCtx32::new(q as u32);
        let twiddles = precompute_twiddles(n, psi, &mont);
        Some((mont, twiddles, plan))
    }

    #[test]
    fn test_accel_ntt_roundtrip_n4() {
        let n = 4;
        let q = 17u64;
        let (mont, twiddles, _plan) = setup_ntt(n, q).unwrap();

        let original: Vec<u64> = vec![1, 2, 3, 4];
        let mut data = original.clone();

        forward_ntt_accel(&mut data, &twiddles, &mont).unwrap();
        assert_ne!(data, original, "Forward NTT should change data");

        inverse_ntt_accel(&mut data, &twiddles, &mont).unwrap();
        assert_eq!(data, original, "Roundtrip should restore original");
    }

    #[test]
    fn test_accel_ntt_roundtrip_n8() {
        let n = 8;
        let q = 97u64;
        let (mont, twiddles, _plan) = setup_ntt(n, q).unwrap();

        let original: Vec<u64> = (0..n).map(|i| (i as u64 * 7 + 3) % q).collect();
        let mut data = original.clone();

        forward_ntt_accel(&mut data, &twiddles, &mont).unwrap();
        inverse_ntt_accel(&mut data, &twiddles, &mont).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn test_accel_ntt_roundtrip_n16() {
        let n = 16;
        let q = 97u64;
        let (mont, twiddles, _plan) = setup_ntt(n, q).unwrap();

        let original: Vec<u64> = (0..n).map(|i| (i as u64 * 13 + 5) % q).collect();
        let mut data = original.clone();

        forward_ntt_accel(&mut data, &twiddles, &mont).unwrap();
        inverse_ntt_accel(&mut data, &twiddles, &mont).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn test_accel_matches_reference_n4() {
        let n = 4;
        let q = 17u64;
        let (mont, twiddles, plan) = setup_ntt(n, q).unwrap();

        let original: Vec<u64> = vec![1, 2, 3, 4];

        // Reference negacyclic NTT
        let mut ref_data = original.clone();
        ntt::forward_ntt_negacyclic(&mut ref_data, &plan).unwrap();

        // Accelerated negacyclic NTT
        let mut accel_data = original.clone();
        forward_ntt_accel(&mut accel_data, &twiddles, &mont).unwrap();

        assert_eq!(
            accel_data, ref_data,
            "Accelerated NTT must match reference NTT exactly"
        );
    }

    #[test]
    fn test_accel_matches_reference_n8() {
        let n = 8;
        let q = 97u64;
        let (mont, twiddles, plan) = setup_ntt(n, q).unwrap();

        let original: Vec<u64> = (0..n).map(|i| (i as u64 * 7 + 3) % q).collect();

        let mut ref_data = original.clone();
        ntt::forward_ntt_negacyclic(&mut ref_data, &plan).unwrap();

        let mut accel_data = original.clone();
        forward_ntt_accel(&mut accel_data, &twiddles, &mont).unwrap();

        assert_eq!(accel_data, ref_data);
    }

    #[test]
    fn test_accel_matches_reference_n16() {
        let n = 16;
        let q = 97u64;
        let (mont, twiddles, plan) = setup_ntt(n, q).unwrap();

        let original: Vec<u64> = (0..n).map(|i| (i as u64 * 13 + 5) % q).collect();

        let mut ref_data = original.clone();
        ntt::forward_ntt_negacyclic(&mut ref_data, &plan).unwrap();

        let mut accel_data = original.clone();
        forward_ntt_accel(&mut accel_data, &twiddles, &mont).unwrap();

        assert_eq!(accel_data, ref_data);
    }

    #[test]
    fn test_accel_matches_reference_n512() {
        let n = 512;
        // 12289 is 2N-friendly for N = 512: 12288 / 1024 == 12.
        // (7681 is *not*: 7680 % 1024 == 512.)
        let q = 12289u64;
        let (mont, twiddles, plan) = setup_ntt(n, q).unwrap();

        let original: Vec<u64> = (0..n).map(|i| (i as u64 * 31 + 17) % q).collect();

        let mut ref_data = original.clone();
        ntt::forward_ntt_negacyclic(&mut ref_data, &plan).unwrap();

        let mut accel_data = original.clone();
        forward_ntt_accel(&mut accel_data, &twiddles, &mont).unwrap();

        assert_eq!(accel_data, ref_data);
    }

    #[test]
    fn test_accel_matches_reference_inverse() {
        // Bit-identical inverse transforms too, not just forward.
        for (n, q) in [(8usize, 97u64), (16, 7681), (512, 12289)] {
            let (mont, twiddles, plan) = setup_ntt(n, q).unwrap();
            let original: Vec<u64> = (0..n).map(|i| (i as u64 * 29 + 11) % q).collect();

            let mut ref_data = original.clone();
            ntt::forward_ntt_negacyclic(&mut ref_data, &plan).unwrap();
            ntt::inverse_ntt_negacyclic(&mut ref_data, &plan).unwrap();

            let mut accel_data = original.clone();
            forward_ntt_accel(&mut accel_data, &twiddles, &mont).unwrap();
            inverse_ntt_accel(&mut accel_data, &twiddles, &mont).unwrap();

            assert_eq!(accel_data, ref_data, "inverse mismatch at N={}, q={}", n, q);
            assert_eq!(accel_data, original);
        }
    }

    #[test]
    fn test_accel_ntt_roundtrip_n512() {
        let n = 512;
        let q = 12289u64;
        let (mont, twiddles, _plan) = setup_ntt(n, q).unwrap();

        let original: Vec<u64> = (0..n).map(|i| (i as u64 * 31 + 17) % q).collect();
        let mut data = original.clone();

        forward_ntt_accel(&mut data, &twiddles, &mont).unwrap();
        inverse_ntt_accel(&mut data, &twiddles, &mont).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn test_accel_ntt_n2_smallest() {
        // Edge case: smallest possible NTT
        let n = 2;
        let q = 17u64; // 16 % 4 == 0 ✓
        let (mont, twiddles, plan) = setup_ntt(n, q).unwrap();

        let original: Vec<u64> = vec![5, 12];

        // Reference
        let mut ref_data = original.clone();
        ntt::forward_ntt_negacyclic(&mut ref_data, &plan).unwrap();

        // Accelerated
        let mut accel_data = original.clone();
        forward_ntt_accel(&mut accel_data, &twiddles, &mont).unwrap();

        assert_eq!(accel_data, ref_data);

        // Roundtrip
        inverse_ntt_accel(&mut accel_data, &twiddles, &mont).unwrap();
        assert_eq!(accel_data, original);
    }

    #[test]
    fn test_accel_ntt_zero_input() {
        let n = 8;
        let q = 97u64;
        let (mont, twiddles, _plan) = setup_ntt(n, q).unwrap();

        let original: Vec<u64> = vec![0; n];
        let mut data = original.clone();

        forward_ntt_accel(&mut data, &twiddles, &mont).unwrap();
        // NTT of all zeros should be all zeros
        assert_eq!(data, original);

        inverse_ntt_accel(&mut data, &twiddles, &mont).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn test_accel_ntt_invalid_params() {
        let mont = MontCtx32::new(97);
        let twiddles = TwiddleTable {
            forward: vec![],
            inverse: vec![],
            mont_psi_powers: vec![],
            psi_inv_powers: vec![],
            n: 8,
        };

        // Wrong length
        let mut data = vec![1, 2, 3]; // not power of 2
        assert!(forward_ntt_accel(&mut data, &twiddles, &mont).is_err());

        // Mismatched n
        let mut data = vec![1, 2, 3, 4]; // n=4, but twiddles.n=8
        assert!(forward_ntt_accel(&mut data, &twiddles, &mont).is_err());
    }

    #[test]
    fn test_accel_matches_reference_large_values() {
        // Test with values close to q (boundary conditions)
        let n = 16;
        let q = 7681u64; // 7680 % 32 == 0 ✓
        let (mont, twiddles, plan) = setup_ntt(n, q).unwrap();

        let original: Vec<u64> = (0..n).map(|i| q - 1 - (i as u64)).collect();

        let mut ref_data = original.clone();
        ntt::forward_ntt_negacyclic(&mut ref_data, &plan).unwrap();

        let mut accel_data = original.clone();
        forward_ntt_accel(&mut accel_data, &twiddles, &mont).unwrap();

        assert_eq!(accel_data, ref_data);
    }

    #[test]
    fn test_precompute_twiddles_count() {
        let n = 16;
        let q = 97u64;
        let psi = ntt::find_primitive_2n_root(n as u64, q).unwrap();
        let mont = MontCtx32::new(q as u32);
        let twiddles = precompute_twiddles(n, psi, &mont);

        // Total twiddles should be: 1 + 2 + 4 + 8 = 15 = N-1
        assert_eq!(twiddles.forward.len(), n - 1);
        assert_eq!(twiddles.inverse.len(), n - 1);
        // psi tables are one entry per coefficient
        assert_eq!(twiddles.mont_psi_powers.len(), n);
        assert_eq!(twiddles.psi_inv_powers.len(), n);
    }

    #[test]
    fn test_accel_twiddles_hold_powers_of_omega_not_psi() {
        // Stage 1 (len = 2) stores a single twiddle, 1. Stage 2 (len = 4) stores
        // [1, omega^(n/4)]. Confirm the table is built from omega = psi^2.
        let n = 16usize;
        let q = 97u64;
        let psi = ntt::find_primitive_2n_root(n as u64, q).unwrap();
        let omega = psi * psi % q;
        let mont = MontCtx32::new(q as u32);
        let twiddles = precompute_twiddles(n, psi, &mont);

        // Last stage (len = n) holds 1, omega, omega^2, ... omega^(n/2 - 1)
        let last_stage_offset = n / 2 - 1; // 1 + 2 + 4 = 7 for n = 16
        let w1 = mont.from_mont(twiddles.forward[last_stage_offset + 1]);
        assert_eq!(w1, omega, "twiddle table must hold powers of omega = psi^2");
        assert_ne!(w1, psi, "twiddle table must NOT hold powers of psi");
    }

    #[test]
    fn test_accel_psi_tables_are_correctly_fused() {
        let n = 16usize;
        let q = 97u64;
        let psi = ntt::find_primitive_2n_root(n as u64, q).unwrap();
        let mont = MontCtx32::new(q as u32);
        let twiddles = precompute_twiddles(n, psi, &mont);
        let psi_inv = ntt::mod_inv(psi, q);

        for i in 0..n {
            let expected_psi_pow = ntt::mod_pow(psi, i as u64, q);
            // mont_mul(x, mont_psi_powers[i]) must equal to_mont(x * psi^i)
            let x = (i as u64 * 7 + 3) % q;
            assert_eq!(
                mont.mont_mul(x, twiddles.mont_psi_powers[i]),
                mont.to_mont(x * expected_psi_pow % q),
                "fused twist wrong at i={}",
                i
            );
            // psi_inv_powers are ordinary-form psi^(-i)
            assert_eq!(
                twiddles.psi_inv_powers[i],
                ntt::mod_pow(psi_inv, i as u64, q),
                "untwist table wrong at i={}",
                i
            );
        }
    }

    #[test]
    fn test_accel_convolution_identity() {
        for (n, q) in [(2usize, 17u64), (4, 17), (8, 97), (16, 97), (16, 7681), (512, 12289)] {
            let (mont, twiddles, plan) = setup_ntt(n, q).unwrap();
            assert!(
                verify_negacyclic_convolution_accel(&twiddles, &mont),
                "accelerated convolution identity failed at N={}, q={}",
                n,
                q
            );
            // And the reference path agrees.
            assert!(ntt::verify_negacyclic_convolution(&plan));
        }
    }
}
