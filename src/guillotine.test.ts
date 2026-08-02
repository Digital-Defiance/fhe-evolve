import { describe, it, expect } from 'vitest';
import { Genome } from './genome.js';
import { guillotine, evaluateGenome, SECURITY_FLOOR_BITS } from './guillotine.js';

describe('guillotine', () => {
  describe('SECURITY_FLOOR_BITS', () => {
    it('re-exports the 128-bit security floor', () => {
      expect(SECURITY_FLOOR_BITS).toBe(128);
    });
  });

  describe('guillotine()', () => {
    it('passes genomes with high N and moderate q (known-secure)', () => {
      const genomes = [
        // N=8192, q=2^27, sigma=3.2, Bg=4096 — massively secure
        new Genome({ N: 8192, q: 134217728, sigma: 3200, Bg: 4096 }),
      ];

      const result = guillotine(genomes);
      expect(result).toHaveLength(1);
      expect(result[0].genome).toBe(genomes[0]);
      expect(result[0].securityBits).toBeGreaterThanOrEqual(128);
      expect(result[0].noiseBudgetBits).toBeGreaterThan(0);
    });

    it('discards genomes with low N and high q (known-insecure)', () => {
      const genomes = [
        // N=1024, q=2^30, sigma=3.2 — exceeds safe modulus for this dimension
        new Genome({ N: 1024, q: 1073741824, sigma: 3200, Bg: 512 }),
      ];

      const result = guillotine(genomes);
      expect(result).toHaveLength(0);
    });

    it('filters a mixed population correctly', () => {
      const genomes = [
        new Genome({ N: 8192, q: 134217728, sigma: 3200, Bg: 4096 }),  // Secure (>>128)
        new Genome({ N: 1024, q: 1073741824, sigma: 3200, Bg: 512 }), // Insecure (q=2^30)
        new Genome({ N: 4096, q: 134217728, sigma: 3200, Bg: 2048 }), // Secure (>>128)
        new Genome({ N: 1024, q: 2147483648, sigma: 3200, Bg: 256 }), // Insecure (q=2^31)
      ];

      const result = guillotine(genomes);
      expect(result).toHaveLength(2);
      for (const sg of result) {
        expect(sg.securityBits).toBeGreaterThanOrEqual(128);
      }
    });

    it('returns an empty array when given an empty population', () => {
      expect(guillotine([])).toEqual([]);
    });

    it('annotates survivors with noise budget', () => {
      const genomes = [
        new Genome({ N: 16384, q: 134217728, sigma: 3200, Bg: 8192 }),
      ];

      const result = guillotine(genomes);
      expect(result).toHaveLength(1);
      expect(result[0].noiseBudgetBits).toBeGreaterThan(0);
    });

    it('passes a borderline-secure genome (N=1024, q=2^27)', () => {
      const genomes = [
        new Genome({ N: 1024, q: 134217728, sigma: 3200, Bg: 512 }),
      ];

      const result = guillotine(genomes);
      expect(result).toHaveLength(1);
      expect(result[0].securityBits).toBeGreaterThanOrEqual(128);
    });
  });

  describe('evaluateGenome()', () => {
    it('returns a full security estimate for a single genome', () => {
      const genome = new Genome({ N: 4096, q: 134217728, sigma: 3200, Bg: 2048 });
      const estimate = evaluateGenome(genome);

      expect(estimate).toHaveProperty('securityBits');
      expect(estimate).toHaveProperty('isSecure');
      expect(estimate).toHaveProperty('noiseBudgetBits');
      expect(typeof estimate.securityBits).toBe('number');
      expect(typeof estimate.isSecure).toBe('boolean');
    });

    it('matches the guillotine filtering decision', () => {
      const secure = new Genome({ N: 8192, q: 134217728, sigma: 3200, Bg: 4096 });
      const insecure = new Genome({ N: 1024, q: 2147483648, sigma: 3200, Bg: 512 });

      expect(evaluateGenome(secure).isSecure).toBe(true);
      expect(evaluateGenome(insecure).isSecure).toBe(false);
    });
  });
});
