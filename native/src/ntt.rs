/// Number Theoretic Transform (NTT) module.
///
/// Implements the **negacyclic** NTT over the ring `Z_q[X]/(X^N + 1)`, which is
/// the ring FHE schemes (BFV/BGV/CKKS/TFHE) actually work in. This is *not* the
/// cyclic transform over `Z_q[X]/(X^N - 1)`: the difference is a per-coefficient
/// twist by powers of psi, a primitive **2N**-th root of unity.
///
/// Structure:
///   * `psi` is a primitive 2N-th root of unity mod q. It exists iff
///     `q = 1 (mod 2N)`, and it satisfies `psi^N = -1 (mod q)`. That sign flip
///     is exactly the `X^N = -1` relation of the ring.
///   * `omega = psi^2` is a primitive N-th root of unity and drives the ordinary
///     radix-2 Cooley-Tukey butterflies.
///   * Forward transform: twist `a[i] -> a[i] * psi^i`, then the cyclic core
///     with `omega`. Inverse: cyclic core with `omega^(-1)`, scale by `N^(-1)`,
///     then untwist by `psi^(-i)`.
///
/// # Twiddle tables hold powers of omega, not powers of psi
///
/// The butterfly twiddle tables (here, and the Montgomery tables in
/// `ntt_accel`) must be filled with powers of `omega = psi^2`, **not** powers of
/// `psi`. Filling them with powers of `psi` halves every effective exponent and
/// the transform stops computing a convolution at all. Do not expect a
/// round-trip check to catch that: it only fails if the forward and inverse
/// tables are built from *different* roots. When the inverse table is derived as
/// the elementwise inverse of the psi-filled forward table — which is how it is
/// normally built — forward-then-inverse still recovers the input exactly and
/// the defect stays hidden.
///
/// The subtler and far more dangerous mistake is dropping the psi twist. What
/// remains is a perfectly valid *cyclic* NTT: it round-trips exactly, so a
/// forward-then-inverse check passes with flying colours, while the product it
/// computes is modulo `X^N - 1` rather than `X^N + 1`. A round-trip check cannot
/// see that. Only the negacyclic convolution identity can, which is why
/// `verify_negacyclic_convolution` exists and is used as the gate.
///
/// # Overflow
///
/// Every modulus handled here is `q < 2^32`, so any product of two reduced
/// residues is `< 2^64` and fits a `u64` exactly. The `wrapping_mul` calls below
/// therefore never actually wrap; they are used only to tell the compiler not to
/// emit overflow checks. Preserve the `q < 2^32` precondition if you touch this.

/// Modular exponentiation: base^exp mod modulus
/// Uses binary exponentiation. Safe for `modulus < 2^32` (see module note on
/// overflow): all intermediate products stay below 2^64.
pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u64 = 1;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result.wrapping_mul(base) % modulus;
        }
        exp >>= 1;
        base = base.wrapping_mul(base) % modulus;
    }
    result
}

/// Modular inverse using Fermat's little theorem (requires p to be prime).
/// Returns base^(p-2) mod p.
pub fn mod_inv(base: u64, p: u64) -> u64 {
    mod_pow(base, p - 2, p)
}

/// Modular multiplication for `a, b < q < 2^32`. See the module note on overflow.
#[inline(always)]
fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
    a.wrapping_mul(b) % q
}

/// Find psi, a primitive 2N-th root of unity modulo q.
///
/// Requires q prime and `q = 1 (mod 2N)`. Candidate generators are searched
/// upward and each candidate root is accepted only after two algebraic order
/// tests:
///   * `psi^(2N) = 1 (mod q)` — psi is a 2N-th root of unity at all;
///   * `psi^N = q - 1` — psi has order *exactly* 2N, and the transform built
///     from it is negacyclic rather than cyclic.
///
/// The second test is the load-bearing one. Without it a root of order N (or any
/// proper divisor of 2N) would be accepted and the transform would silently
/// compute a cyclic convolution while still round-tripping.
///
/// Returns `None` if no such root exists. There is no fallback: a caller that
/// cannot get a psi must fail, not degrade to a cyclic transform.
pub fn find_primitive_2n_root(n: u64, q: u64) -> Option<u64> {
    if n < 2 || q < 3 {
        return None;
    }
    let two_n = n.checked_mul(2)?;
    // A primitive 2N-th root of unity exists in Z_q* iff 2N divides q-1.
    if (q - 1) % two_n != 0 {
        return None;
    }

    let exponent = (q - 1) / two_n;

    for g in 2..q {
        let psi = mod_pow(g, exponent, q);
        if psi <= 1 {
            continue;
        }
        // Order exactly 2N: psi^(2N) == 1 AND psi^N == -1 (mod q).
        // Because 2N is a power of two, psi^N == -1 forces the order to be 2N
        // exactly (any smaller order would divide N and give psi^N == 1).
        if mod_pow(psi, two_n, q) == 1 && mod_pow(psi, n, q) == q - 1 {
            return Some(psi);
        }
    }
    None
}

/// Precomputed constants for the negacyclic transform over `Z_q[X]/(X^N + 1)`.
///
/// Everything that depends only on `(N, q)` lives here so it is never inside a
/// timed region. The twist tables are part of the transform, not setup.
#[derive(Debug, Clone)]
pub struct NegacyclicPlan {
    pub n: usize,
    pub q: u64,
    /// Primitive 2N-th root of unity. `psi^N == q - 1`.
    pub psi: u64,
    /// `psi^(-1) mod q`. The tabulated `psi_inv_powers` are what the transform
    /// uses; this is kept for inspection and for the plan-level inverse checks.
    #[allow(dead_code)]
    pub psi_inv: u64,
    /// `omega = psi^2`, a primitive N-th root of unity. Drives the butterflies.
    pub omega: u64,
    pub omega_inv: u64,
    /// `N^(-1) mod q`, for the inverse transform.
    pub n_inv: u64,
    /// `psi^i mod q` for i in 0..N (forward twist).
    pub psi_powers: Vec<u64>,
    /// `psi^(-i) mod q` for i in 0..N (inverse untwist).
    pub psi_inv_powers: Vec<u64>,
}

impl NegacyclicPlan {
    /// Build a plan for degree `n` and modulus `q`, or `None` if the pair does
    /// not admit a negacyclic NTT. Fails loudly (returns `None`) on any failed
    /// algebraic check rather than falling back to a weaker transform.
    pub fn new(n: usize, q: u64) -> Option<Self> {
        if n < 2 || (n & (n - 1)) != 0 {
            return None; // N must be a power of two, at least 2
        }
        if q < 3 || q >= (1u64 << 32) {
            return None; // see module note on overflow
        }

        let n64 = n as u64;
        let psi = find_primitive_2n_root(n64, q)?;

        // Re-assert the defining properties of psi at plan level. Cheap, and it
        // means every plan in existence has been checked, not just the search.
        if mod_pow(psi, 2 * n64, q) != 1 || mod_pow(psi, n64, q) != q - 1 {
            return None;
        }

        let omega = mul_mod(psi, psi, q);
        // omega must have order exactly N. Since N is a power of two it suffices
        // to rule out order N/2, i.e. omega^(N/2) != 1. (It equals psi^N == -1.)
        if mod_pow(omega, n64, q) != 1 || mod_pow(omega, n64 / 2, q) == 1 {
            return None;
        }

        let psi_inv = mod_inv(psi, q);
        if mul_mod(psi, psi_inv, q) != 1 {
            return None; // q not prime, so Fermat inversion is invalid
        }
        let omega_inv = mod_inv(omega, q);
        if mul_mod(omega, omega_inv, q) != 1 {
            return None;
        }
        let n_inv = mod_inv(n64 % q, q);
        if mul_mod(n64 % q, n_inv, q) != 1 {
            return None;
        }

        let mut psi_powers = Vec::with_capacity(n);
        let mut psi_inv_powers = Vec::with_capacity(n);
        psi_powers.push(1u64);
        psi_inv_powers.push(1u64);
        for i in 1..n {
            psi_powers.push(mul_mod(psi_powers[i - 1], psi, q));
            psi_inv_powers.push(mul_mod(psi_inv_powers[i - 1], psi_inv, q));
        }

        Some(NegacyclicPlan {
            n,
            q,
            psi,
            psi_inv,
            omega,
            omega_inv,
            n_inv,
            psi_powers,
            psi_inv_powers,
        })
    }
}

/// Bit-reversal permutation for in-place NTT.
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

/// Cyclic forward NTT core (Cooley-Tukey radix-2 butterflies) in Z_q.
///
/// `data` must have length N (a power of 2) and `omega` must be a primitive
/// N-th root of unity mod q. On its own this computes the transform over
/// `Z_q[X]/(X^N - 1)`; the negacyclic wrappers below add the psi twist.
pub fn forward_ntt(data: &mut [u64], omega: u64, q: u64) -> Result<(), ()> {
    let n = data.len();
    if n == 0 || (n & (n - 1)) != 0 {
        return Err(()); // N must be a power of 2
    }

    bit_reverse(data);

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let w_step = mod_pow(omega, (n / len) as u64, q);
        let mut i = 0;
        while i < n {
            let mut w: u64 = 1;
            for j in 0..half {
                let u = data[i + j];
                let v = data[i + j + half].wrapping_mul(w) % q;
                data[i + j] = (u + v) % q;
                data[i + j + half] = (u + q - v) % q;
                w = w.wrapping_mul(w_step) % q;
            }
            i += len;
        }
        len <<= 1;
    }
    Ok(())
}

/// Forward negacyclic NTT over `Z_q[X]/(X^N + 1)`.
///
/// Twists coefficient `i` by `psi^i`, then runs the cyclic core with
/// `omega = psi^2`. The twist is part of the transform and belongs inside any
/// timed region.
pub fn forward_ntt_negacyclic(data: &mut [u64], plan: &NegacyclicPlan) -> Result<(), ()> {
    if data.len() != plan.n {
        return Err(());
    }
    let q = plan.q;
    for i in 0..plan.n {
        data[i] = data[i].wrapping_mul(plan.psi_powers[i]) % q;
    }
    forward_ntt(data, plan.omega, q)
}

/// Inverse negacyclic NTT over `Z_q[X]/(X^N + 1)`.
///
/// Runs the cyclic inverse core with `omega^(-1)`, scales by `N^(-1)`, then
/// untwists coefficient `i` by `psi^(-i)`.
pub fn inverse_ntt_negacyclic(data: &mut [u64], plan: &NegacyclicPlan) -> Result<(), ()> {
    if data.len() != plan.n {
        return Err(());
    }
    let q = plan.q;
    forward_ntt(data, plan.omega_inv, q)?;
    for i in 0..plan.n {
        let scaled = data[i].wrapping_mul(plan.n_inv) % q;
        data[i] = scaled.wrapping_mul(plan.psi_inv_powers[i]) % q;
    }
    Ok(())
}

/// Deterministic sparse operands for the negacyclic convolution identity, plus
/// the schoolbook reference product.
///
/// The support of both operands always contains index `N-1`, so at least one
/// product has degree `>= N` and must wrap with a sign flip under `X^N = -1`.
/// A cyclic transform gets that sign wrong and fails the identity.
///
/// Because the support has O(1) entries, both the operands and the reference
/// product cost O(N) to build regardless of N.
pub fn convolution_test_vectors(n: usize, q: u64) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let n64 = n as u64;

    let mut idx_a: Vec<usize> = vec![n - 1, 0];
    let mut idx_b: Vec<usize> = vec![n - 1, 1 % n];
    for k in 1u64..=6 {
        idx_a.push(((k * 1_013_904_223) % n64) as usize);
        idx_b.push(((k * 1_664_525) % n64) as usize);
    }
    idx_a.sort_unstable();
    idx_a.dedup();
    idx_b.sort_unstable();
    idx_b.dedup();

    let mut a = vec![0u64; n];
    let mut b = vec![0u64; n];
    let step = 6_364_136_223_846_793_005u64 % q;
    let mut v = 12_345_678_901_234_567u64 % q;
    for &i in &idx_a {
        a[i] = v;
        v = mul_mod(v, step, q);
    }
    for &i in &idx_b {
        b[i] = v;
        v = mul_mod(v, step, q);
    }

    // Schoolbook negacyclic convolution over the sparse support.
    let mut expected = vec![0u64; n];
    for &i in &idx_a {
        for &j in &idx_b {
            let mut prod = mul_mod(a[i], b[j], q);
            let mut k = i + j;
            if k >= n {
                k -= n;
                // X^N = -1
                prod = if prod == 0 { 0 } else { q - prod };
            }
            expected[k] = (expected[k] + prod) % q;
        }
    }

    (a, b, expected)
}

/// Verify the negacyclic convolution identity for a plan:
/// `INTT(NTT(a) . NTT(b)) == a * b in Z_q[X]/(X^N + 1)`.
///
/// This is the gate. Unlike a round-trip check it fails if the twiddle tables
/// hold powers of psi instead of powers of omega, if psi has the wrong order, or
/// if the transform is cyclic rather than negacyclic.
pub fn verify_negacyclic_convolution(plan: &NegacyclicPlan) -> bool {
    let n = plan.n;
    let q = plan.q;
    let (a, b, expected) = convolution_test_vectors(n, q);

    let mut fa = a;
    let mut fb = b;
    if forward_ntt_negacyclic(&mut fa, plan).is_err() {
        return false;
    }
    if forward_ntt_negacyclic(&mut fb, plan).is_err() {
        return false;
    }

    let mut fc: Vec<u64> = (0..n).map(|i| mul_mod(fa[i], fb[i], q)).collect();
    if inverse_ntt_negacyclic(&mut fc, plan).is_err() {
        return false;
    }

    fc == expected
}

/// Validates that forward + inverse negacyclic NTT roundtrips correctly for
/// given parameters. Kept as a cheap regression check; the convolution identity
/// is the real gate.
#[allow(dead_code)]
pub fn validate_ntt_roundtrip(n: usize, q: u64) -> bool {
    let plan = match NegacyclicPlan::new(n, q) {
        Some(p) => p,
        None => return false,
    };

    let original: Vec<u64> = (0..n).map(|i| (i as u64) % q).collect();
    let mut data = original.clone();

    if forward_ntt_negacyclic(&mut data, &plan).is_err() {
        return false;
    }
    if inverse_ntt_negacyclic(&mut data, &plan).is_err() {
        return false;
    }

    data == original
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense O(N^2) schoolbook negacyclic convolution, for cross-checking the
    /// sparse reference at small N.
    fn dense_negacyclic_mul(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
        let n = a.len();
        let mut out = vec![0u64; n];
        for i in 0..n {
            for j in 0..n {
                let mut prod = (a[i] * b[j]) % q;
                let mut k = i + j;
                if k >= n {
                    k -= n;
                    prod = if prod == 0 { 0 } else { q - prod };
                }
                out[k] = (out[k] + prod) % q;
            }
        }
        out
    }

    #[test]
    fn test_mod_pow_basic() {
        assert_eq!(mod_pow(2, 10, 1024), 0); // 2^10 = 1024, 1024 % 1024 = 0
        assert_eq!(mod_pow(2, 10, 1000), 24); // 1024 % 1000 = 24
        assert_eq!(mod_pow(3, 0, 7), 1);
        assert_eq!(mod_pow(5, 1, 7), 5);
    }

    #[test]
    fn test_mod_inv() {
        // 3 * 5 = 15 ≡ 1 (mod 7)
        assert_eq!(mod_inv(3, 7), 5);
        // Verify: base * inv ≡ 1 (mod p)
        let p = 97u64;
        for base in 2..10 {
            let inv = mod_inv(base, p);
            assert_eq!((base * inv) % p, 1);
        }
    }

    #[test]
    fn test_psi_is_primitive_2n_root() {
        // psi^(2N) == 1 and psi^N == q - 1 at several (N, q) pairs.
        let cases: Vec<(u64, u64)> = vec![
            (2, 17),
            (4, 17),
            (8, 97),
            (16, 97),
            (16, 7681),
            (512, 12289),
            (1024, 12289),
            (4096, 40961),
        ];
        for (n, q) in cases {
            let psi = find_primitive_2n_root(n, q)
                .unwrap_or_else(|| panic!("no psi for N={}, q={}", n, q));
            assert_eq!(
                mod_pow(psi, 2 * n, q),
                1,
                "psi^(2N) != 1 for N={}, q={}",
                n,
                q
            );
            assert_eq!(
                mod_pow(psi, n, q),
                q - 1,
                "psi^N != q-1 for N={}, q={}",
                n,
                q
            );
            // Order is exactly 2N, so psi^N must not be 1.
            assert_ne!(mod_pow(psi, n, q), 1);
        }
    }

    #[test]
    fn test_omega_has_order_exactly_n() {
        let cases: Vec<(usize, u64)> = vec![(4, 17), (8, 97), (16, 97), (512, 12289)];
        for (n, q) in cases {
            let plan = NegacyclicPlan::new(n, q).unwrap();
            let n64 = n as u64;
            assert_eq!(plan.omega, (plan.psi * plan.psi) % q);
            // Inverses stored on the plan are genuine inverses.
            assert_eq!(plan.psi * plan.psi_inv % q, 1, "psi_inv wrong for N={}", n);
            assert_eq!(plan.omega * plan.omega_inv % q, 1);
            assert_eq!(n64 % q * plan.n_inv % q, 1);
            assert_eq!(mod_pow(plan.omega, n64, q), 1, "omega^N != 1 for N={}", n);
            assert_ne!(
                mod_pow(plan.omega, n64 / 2, q),
                1,
                "omega^(N/2) == 1, order is smaller than N for N={}",
                n
            );
        }
    }

    #[test]
    fn test_find_primitive_2n_root_invalid() {
        // q = 17, N = 6: 2N = 12, 16 % 12 != 0, so no root exists
        assert!(find_primitive_2n_root(6, 17).is_none());
        // q = 7, N = 4: 2N = 8, 6 % 8 != 0
        assert!(find_primitive_2n_root(4, 7).is_none());
    }

    #[test]
    fn test_rejects_q_congruent_mod_n_but_not_2n() {
        // 7681 - 1 = 7680. 7680 % 512 == 0, so a *cyclic* N-th root exists,
        // but 7680 % 1024 == 512 != 0, so there is no primitive 2N-th root and
        // the negacyclic transform must be rejected.
        assert_eq!(7680 % 512, 0);
        assert_ne!(7680 % 1024, 0);
        assert!(find_primitive_2n_root(512, 7681).is_none());
        assert!(NegacyclicPlan::new(512, 7681).is_none());
        assert!(!validate_ntt_roundtrip(512, 7681));

        // Same shape at a tiny degree: q = 97, N = 48 is not a power of two, so
        // use N = 16 vs q = 17: 16 % 16 == 0 but 16 % 32 != 0.
        assert_eq!((17 - 1) % 16, 0);
        assert_ne!((17 - 1) % 32, 0);
        assert!(find_primitive_2n_root(16, 17).is_none());
        assert!(NegacyclicPlan::new(16, 17).is_none());
    }

    #[test]
    fn test_ntt_roundtrip_small() {
        // q = 17, N = 4: 16 % 8 == 0 ✓
        let n = 4;
        let q = 17u64;
        let plan = NegacyclicPlan::new(n, q).unwrap();

        let original: Vec<u64> = vec![1, 2, 3, 4];
        let mut data = original.clone();

        forward_ntt_negacyclic(&mut data, &plan).unwrap();
        // After forward NTT, data should differ from original
        assert_ne!(data, original);

        inverse_ntt_negacyclic(&mut data, &plan).unwrap();
        // After inverse, should match original
        assert_eq!(data, original);
    }

    #[test]
    fn test_ntt_roundtrip_n8() {
        // q = 97, N = 8: 96 % 16 == 0 ✓
        let n = 8;
        let q = 97u64;
        let plan = NegacyclicPlan::new(n, q).unwrap();

        let original: Vec<u64> = (0..n).map(|i| (i as u64 * 7 + 3) % q).collect();
        let mut data = original.clone();

        forward_ntt_negacyclic(&mut data, &plan).unwrap();
        inverse_ntt_negacyclic(&mut data, &plan).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn test_ntt_roundtrip_n16() {
        // q = 97, N = 16: 96 % 32 == 0 ✓
        let n = 16;
        let q = 97u64;
        let plan = NegacyclicPlan::new(n, q).unwrap();

        let original: Vec<u64> = (0..n).map(|i| (i as u64 * 13 + 5) % q).collect();
        let mut data = original.clone();

        forward_ntt_negacyclic(&mut data, &plan).unwrap();
        inverse_ntt_negacyclic(&mut data, &plan).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn test_validate_ntt_roundtrip_valid() {
        assert!(validate_ntt_roundtrip(4, 17));
        assert!(validate_ntt_roundtrip(8, 97));
        assert!(validate_ntt_roundtrip(16, 97));
    }

    #[test]
    fn test_validate_ntt_roundtrip_invalid() {
        // q = 7, N = 4: 6 % 8 != 0 → no primitive 2N-th root
        assert!(!validate_ntt_roundtrip(4, 7));
        // N not a power of 2
        assert!(!validate_ntt_roundtrip(3, 17));
    }

    #[test]
    fn test_ntt_larger_prime() {
        // q = 12289 is 2N-friendly for N = 512: 12288 / 1024 == 12
        let n = 512;
        let q = 12289u64;
        assert_eq!((q - 1) % (2 * n as u64), 0);
        assert!(validate_ntt_roundtrip(n, q));
    }

    #[test]
    fn test_negacyclic_convolution_identity() {
        let cases: Vec<(usize, u64)> = vec![
            (2, 17),
            (4, 17),
            (8, 97),
            (16, 97),
            (16, 7681),
            (512, 12289),
            (1024, 12289),
        ];
        for (n, q) in cases {
            let plan = NegacyclicPlan::new(n, q).unwrap();
            assert!(
                verify_negacyclic_convolution(&plan),
                "convolution identity failed for N={}, q={}",
                n,
                q
            );
        }
    }

    #[test]
    fn test_sparse_reference_matches_dense_reference() {
        // At small N the sparse O(N) reference product must agree with a dense
        // O(N^2) schoolbook negacyclic multiply.
        for (n, q) in [(4usize, 17u64), (8, 97), (16, 97), (16, 7681)] {
            let (a, b, expected) = convolution_test_vectors(n, q);
            let dense = dense_negacyclic_mul(&a, &b, q);
            assert_eq!(
                expected, dense,
                "sparse reference != dense reference for N={}, q={}",
                n, q
            );
        }
    }

    #[test]
    fn test_negacyclic_wraparound_sign_flip_by_hand() {
        // N = 4, q = 17. a = X^3, b = X^3. In Z_q[X]/(X^4 + 1):
        //   X^3 * X^3 = X^6 = X^2 * X^4 = -X^2, so the product is 16*X^2.
        let n = 4;
        let q = 17u64;
        let plan = NegacyclicPlan::new(n, q).unwrap();

        let a = vec![0u64, 0, 0, 1];
        let b = vec![0u64, 0, 0, 1];
        let expected = vec![0u64, 0, q - 1, 0];
        assert_eq!(dense_negacyclic_mul(&a, &b, q), expected);

        let mut fa = a.clone();
        let mut fb = b.clone();
        forward_ntt_negacyclic(&mut fa, &plan).unwrap();
        forward_ntt_negacyclic(&mut fb, &plan).unwrap();
        let mut fc: Vec<u64> = (0..n).map(|i| (fa[i] * fb[i]) % q).collect();
        inverse_ntt_negacyclic(&mut fc, &plan).unwrap();
        assert_eq!(fc, expected, "X^3 * X^3 should be -X^2 mod (X^4 + 1)");
    }

    #[test]
    fn test_cyclic_transform_roundtrips_but_fails_the_gate() {
        // Why the convolution identity has to be the gate: strip the psi twist
        // and what remains is a valid cyclic NTT. It round-trips perfectly, so a
        // round-trip check passes, but it convolves modulo X^N - 1.
        let n = 8usize;
        let q = 97u64;
        let plan = NegacyclicPlan::new(n, q).unwrap();

        let mut cyclic = plan.clone();
        cyclic.psi_powers = vec![1u64; n];
        cyclic.psi_inv_powers = vec![1u64; n];

        let original: Vec<u64> = (0..n).map(|i| (i as u64 * 5 + 1) % q).collect();
        let mut data = original.clone();
        forward_ntt_negacyclic(&mut data, &cyclic).unwrap();
        inverse_ntt_negacyclic(&mut data, &cyclic).unwrap();
        assert_eq!(data, original, "cyclic transform round-trips perfectly");

        assert!(
            !verify_negacyclic_convolution(&cyclic),
            "the gate must reject a cyclic transform"
        );
        assert!(verify_negacyclic_convolution(&plan));
    }
}
