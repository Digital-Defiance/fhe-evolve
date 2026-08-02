import { describe, it, expect } from 'vitest';
import {
  estimateSecurityBits,
  estimateNoiseBudget,
  estimateSecurity,
  SECURITY_FLOOR_BITS,
} from './security-estimator.js';

describe('security-estimator', () => {
  describe('SECURITY_FLOOR_BITS', () => {
    it('should be 128', () => {
      expect(SECURITY_FLOOR_BITS).toBe(128);
    });
  });

  describe('estimateSecurityBits', () => {
    it('returns ~128 bits for N=1024, q=2^27, sigma=3.2 (HE Standard reference)', () => {
      const bits = estimateSecurityBits(1024, 2 ** 27, 3200);
      expect(bits).toBeGreaterThanOrEqual(127);
      expect(bits).toBeLessThanOrEqual(130);
    });

    it('returns well above 128 bits for high-N with moderate q', () => {
      // N=8192 with q=2^27 is massively over-provisioned
      const bits = estimateSecurityBits(8192, 2 ** 27, 3200);
      expect(bits).toBeGreaterThan(500);
    });

    it('returns below 128 bits for N=1024 with large q', () => {
      // N=1024, q=2^30 exceeds the max modulus for 128-bit at this dimension
      const bits = estimateSecurityBits(1024, 2 ** 30, 3200);
      expect(bits).toBeLessThan(128);
    });

    it('increases with larger N (more secure)', () => {
      const q = 2 ** 27;
      const sigma = 3200;
      const bits1024 = estimateSecurityBits(1024, q, sigma);
      const bits4096 = estimateSecurityBits(4096, q, sigma);
      expect(bits4096).toBeGreaterThan(bits1024);
    });

    it('decreases with larger q (less secure)', () => {
      const N = 2048;
      const sigma = 3200;
      const bitsSmallQ = estimateSecurityBits(N, 2 ** 20, sigma);
      const bitsLargeQ = estimateSecurityBits(N, 2 ** 30, sigma);
      expect(bitsLargeQ).toBeLessThan(bitsSmallQ);
    });

    it('increases with larger sigma (more noise = more secure)', () => {
      const N = 2048;
      const q = 2 ** 27;
      const bitsSmallSigma = estimateSecurityBits(N, q, 1600); // sigma=1.6
      const bitsLargeSigma = estimateSecurityBits(N, q, 6400); // sigma=6.4
      expect(bitsLargeSigma).toBeGreaterThan(bitsSmallSigma);
    });

    it('returns 0 for degenerate parameters', () => {
      expect(estimateSecurityBits(0, 2 ** 27, 3200)).toBe(0);
      expect(estimateSecurityBits(1024, 0, 3200)).toBe(0);
      expect(estimateSecurityBits(1024, 2 ** 27, 0)).toBe(0);
    });

    it('returns 0 when q <= sigma (unusable scheme)', () => {
      // q=1 with sigma=3.2 means q/sigma < 1
      const bits = estimateSecurityBits(1024, 1, 3200);
      expect(bits).toBe(0);
    });
  });

  describe('estimateNoiseBudget', () => {
    it('returns positive budget for typical parameters', () => {
      // q=2^27, sigma=3.2 → budget ≈ log2(2^27 / 6.4) - 1 ≈ 27 - 2.68 - 1 ≈ 23.3
      const budget = estimateNoiseBudget(2 ** 27, 3200);
      expect(budget).toBeGreaterThan(20);
      expect(budget).toBeLessThan(30);
    });

    it('returns 0 for degenerate parameters', () => {
      expect(estimateNoiseBudget(0, 3200)).toBe(0);
      expect(estimateNoiseBudget(100, 0)).toBe(0);
    });

    it('increases with larger q', () => {
      const budgetSmall = estimateNoiseBudget(2 ** 20, 3200);
      const budgetLarge = estimateNoiseBudget(2 ** 30, 3200);
      expect(budgetLarge).toBeGreaterThan(budgetSmall);
    });

    it('decreases with larger sigma', () => {
      const budgetSmallSigma = estimateNoiseBudget(2 ** 27, 1600);
      const budgetLargeSigma = estimateNoiseBudget(2 ** 27, 6400);
      expect(budgetSmallSigma).toBeGreaterThan(budgetLargeSigma);
    });
  });

  describe('estimateSecurity', () => {
    it('marks high-N, moderate-q parameters as secure', () => {
      const result = estimateSecurity(4096, 2 ** 27, 3200);
      expect(result.isSecure).toBe(true);
      expect(result.securityBits).toBeGreaterThanOrEqual(128);
      expect(result.noiseBudgetBits).toBeGreaterThan(0);
    });

    it('marks trivially insecure parameters as not secure', () => {
      // N=1024, q=2^30 — large modulus with small dimension is weak
      const result = estimateSecurity(1024, 2 ** 30, 3200);
      expect(result.isSecure).toBe(false);
      expect(result.securityBits).toBeLessThan(128);
    });

    it('returns zero security for degenerate parameters', () => {
      const result = estimateSecurity(1024, 1, 3200);
      expect(result.securityBits).toBe(0);
      expect(result.isSecure).toBe(false);
    });

    it('includes noise budget in the result', () => {
      const result = estimateSecurity(8192, 2 ** 27, 3200);
      expect(result.noiseBudgetBits).toBeGreaterThan(0);
    });

    it('returns all expected fields', () => {
      const result = estimateSecurity(2048, 2 ** 27, 3200);
      expect(result).toHaveProperty('securityBits');
      expect(result).toHaveProperty('isSecure');
      expect(result).toHaveProperty('noiseBudgetBits');
      expect(typeof result.securityBits).toBe('number');
      expect(typeof result.isSecure).toBe('boolean');
      expect(typeof result.noiseBudgetBits).toBe('number');
    });
  });
});
