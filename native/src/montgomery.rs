/// Montgomery arithmetic for modular multiplication without division.
///
/// Uses R = 2^32 since all our moduli q fit in u32 (max ~4 billion).
/// This keeps products in u64 range without needing u128 arithmetic.
///
/// In Montgomery form, a value `a` is represented as `aR mod q`.
/// Multiplication becomes: `mont_mul(aR, bR) = abR mod q` (one R cancels).

/// Montgomery context for a specific modulus q (must fit in u32).
/// All arithmetic stays in Montgomery domain to avoid costly mod operations.
#[derive(Debug, Clone, Copy)]
pub struct MontCtx32 {
    /// The modulus (stored as u64 for convenience in arithmetic)
    pub q: u64,
    /// The modulus as u32
    #[allow(dead_code)]
    pub q32: u32,
    /// -q^(-1) mod R, where R = 2^32
    pub q_inv_neg: u32,
    /// R mod q (for converting to Montgomery form)
    pub r_mod_q: u64,
    /// R^2 mod q (for converting to Montgomery form via mont_mul)
    pub r2_mod_q: u64,
}

impl MontCtx32 {
    /// Create a new Montgomery context for modulus `q`.
    /// Requires: q is odd (all NTT-friendly primes are odd), q < 2^32.
    pub fn new(q: u32) -> Self {
        assert!(q > 1, "Modulus must be > 1");
        assert!(q & 1 == 1, "Modulus must be odd for Montgomery reduction");

        let q64 = q as u64;

        // Compute -q^(-1) mod 2^32 using Newton's method.
        // Start with q_inv such that q * q_inv ≡ 1 (mod 2^k), then negate.
        // Newton iteration: x' = x * (2 - q*x) mod 2^32
        let mut inv: u32 = 1;
        for _ in 0..5 {
            // 5 iterations give 2^32 bits of precision (doubles each time: 1→2→4→8→16→32)
            inv = inv.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(inv)));
        }
        // We want -q^(-1) mod R, i.e., the value such that q * q_inv_neg ≡ -1 (mod R)
        let q_inv_neg = inv.wrapping_neg(); // This gives us -q^{-1} mod 2^32
        // Verify: q * q_inv_neg ≡ -1 (mod 2^32)
        debug_assert!(q.wrapping_mul(q_inv_neg) == u32::MAX, "q_inv_neg computation failed");

        // R mod q, where R = 2^32
        let r_mod_q = (1u64 << 32) % q64;

        // R^2 mod q
        let r2_mod_q = (r_mod_q * r_mod_q) % q64;

        MontCtx32 {
            q: q64,
            q32: q,
            q_inv_neg,
            r_mod_q,
            r2_mod_q,
        }
    }

    /// Convert a normal value to Montgomery form: a -> aR mod q
    #[inline(always)]
    pub fn to_mont(&self, a: u64) -> u64 {
        self.mont_mul(a, self.r2_mod_q)
    }

    /// Convert from Montgomery form back to normal: aR -> a
    #[inline(always)]
    pub fn from_mont(&self, a: u64) -> u64 {
        self.mont_reduce(a)
    }

    /// Montgomery multiplication: given aR mod q and bR mod q,
    /// returns abR mod q (i.e., the product stays in Montgomery form).
    #[inline(always)]
    pub fn mont_mul(&self, a: u64, b: u64) -> u64 {
        let t = a * b; // Both a, b < q < 2^32, so product fits in u64
        self.mont_reduce(t)
    }

    /// Montgomery reduction (REDC) for R = 2^32.
    /// Given t < q * R, returns t * R^(-1) mod q.
    #[inline(always)]
    pub fn mont_reduce(&self, t: u64) -> u64 {
        // m = (t mod R) * q_inv_neg mod R
        let m = (t as u32).wrapping_mul(self.q_inv_neg);
        // result = (t + m*q) / R
        let result = (t + (m as u64) * self.q) >> 32;
        // Final conditional subtraction
        if result >= self.q {
            result - self.q
        } else {
            result
        }
    }

    /// Modular addition in Montgomery form (works for both normal and Montgomery values).
    #[inline(always)]
    pub fn mont_add(&self, a: u64, b: u64) -> u64 {
        let sum = a + b;
        if sum >= self.q {
            sum - self.q
        } else {
            sum
        }
    }

    /// Modular subtraction in Montgomery form.
    #[inline(always)]
    pub fn mont_sub(&self, a: u64, b: u64) -> u64 {
        if a >= b {
            a - b
        } else {
            a + self.q - b
        }
    }

    /// Montgomery exponentiation: base^exp mod q, result in Montgomery form.
    /// `base` must already be in Montgomery form.
    #[inline]
    pub fn mont_pow(&self, mut base: u64, mut exp: u64) -> u64 {
        // Start with 1 in Montgomery form = R mod q
        let mut result = self.r_mod_q;
        while exp > 0 {
            if exp & 1 == 1 {
                result = self.mont_mul(result, base);
            }
            exp >>= 1;
            base = self.mont_mul(base, base);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mont_ctx_creation() {
        let ctx = MontCtx32::new(97);
        assert_eq!(ctx.q, 97);
        assert_eq!(ctx.q32, 97);
        // Verify q * q_inv_neg ≡ -1 (mod 2^32)
        assert_eq!(ctx.q32.wrapping_mul(ctx.q_inv_neg), u32::MAX);
    }

    #[test]
    fn test_mont_roundtrip() {
        let ctx = MontCtx32::new(97);
        for a in 0..97u64 {
            let mont_a = ctx.to_mont(a);
            let back = ctx.from_mont(mont_a);
            assert_eq!(back, a, "Roundtrip failed for a={}", a);
        }
    }

    #[test]
    fn test_mont_mul_correctness() {
        let ctx = MontCtx32::new(97);
        for a in 0..97u64 {
            for b in 0..97u64 {
                let expected = (a * b) % 97;
                let mont_a = ctx.to_mont(a);
                let mont_b = ctx.to_mont(b);
                let mont_product = ctx.mont_mul(mont_a, mont_b);
                let result = ctx.from_mont(mont_product);
                assert_eq!(
                    result, expected,
                    "mont_mul failed for a={}, b={}: got {}, expected {}",
                    a, b, result, expected
                );
            }
        }
    }

    #[test]
    fn test_mont_add_sub() {
        let ctx = MontCtx32::new(97);
        for a in 0..97u64 {
            for b in 0..97u64 {
                let mont_a = ctx.to_mont(a);
                let mont_b = ctx.to_mont(b);

                let sum = ctx.mont_add(mont_a, mont_b);
                let expected_sum = (a + b) % 97;
                assert_eq!(ctx.from_mont(sum), expected_sum, "add failed for a={}, b={}", a, b);

                let diff = ctx.mont_sub(mont_a, mont_b);
                let expected_diff = (a + 97 - b) % 97;
                assert_eq!(ctx.from_mont(diff), expected_diff, "sub failed for a={}, b={}", a, b);
            }
        }
    }

    #[test]
    fn test_mont_pow() {
        let ctx = MontCtx32::new(97);
        // 3^5 mod 97 = 243 mod 97 = 49
        let base = ctx.to_mont(3);
        let result = ctx.mont_pow(base, 5);
        assert_eq!(ctx.from_mont(result), 49);

        // 2^10 mod 97 = 1024 mod 97 = 54
        let base2 = ctx.to_mont(2);
        let result2 = ctx.mont_pow(base2, 10);
        assert_eq!(ctx.from_mont(result2), 54);
    }

    #[test]
    fn test_mont_large_prime() {
        // Test with a larger NTT-friendly prime: q = 7681 (7680 = 15 * 512)
        let ctx = MontCtx32::new(7681);
        assert_eq!(ctx.q32.wrapping_mul(ctx.q_inv_neg), u32::MAX);

        // Roundtrip
        for a in 0..100u64 {
            let mont_a = ctx.to_mont(a);
            assert_eq!(ctx.from_mont(mont_a), a);
        }

        // Multiplication
        let a = 5000u64;
        let b = 3000u64;
        let expected = (a * b) % 7681;
        let result = ctx.from_mont(ctx.mont_mul(ctx.to_mont(a), ctx.to_mont(b)));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mont_ntt_friendly_primes() {
        // Test all typical NTT-friendly primes we'd use
        let primes: Vec<u32> = vec![17, 97, 257, 7681, 12289, 40961, 65537, 786433, 134217729];
        for &p in &primes {
            if p & 1 == 0 {
                continue; // skip even (134217729 is odd: 2^27 + 1... wait, let's check)
            }
            let ctx = MontCtx32::new(p);
            assert_eq!(ctx.q32.wrapping_mul(ctx.q_inv_neg), u32::MAX, "Failed for prime {}", p);

            // Quick mul check
            let a = (p as u64 / 2) + 1;
            let b = (p as u64 / 3) + 1;
            let expected = (a * b) % (p as u64);
            let result = ctx.from_mont(ctx.mont_mul(ctx.to_mont(a), ctx.to_mont(b)));
            assert_eq!(result, expected, "Mont mul failed for prime {}", p);
        }
    }
}
