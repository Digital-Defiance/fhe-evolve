import { describe, it, expect } from 'vitest';
import { PopulationManager, NTT_PRIME_POOL } from './population-manager.js';
import { Genome, GENOME_FIELD_COUNT } from './genome.js';
import type { ScoredGenome } from './types.js';

/**
 * Check that a (N, q) pair satisfies the negacyclic NTT constraint:
 * q is prime and (q - 1) % (2 * N) === 0, so a primitive 2N-th root of unity
 * exists modulo q for the ring Z_q[X]/(X^N + 1).
 */
function isNttCompatible(N: number, q: number): boolean {
  if (q < 2 || (q - 1) % (2 * N) !== 0) return false;
  // Simple primality check
  if (q === 2 || q === 3) return true;
  if (q % 2 === 0 || q % 3 === 0) return false;
  for (let i = 5; i * i <= q; i += 6) {
    if (q % i === 0 || q % (i + 2) === 0) return false;
  }
  return true;
}

describe('PopulationManager', () => {
  describe('NTT Prime Pool', () => {
    it('has primes for all valid N values', () => {
      const validN = [1024, 2048, 4096, 8192, 16384];
      for (const N of validN) {
        const primes = NTT_PRIME_POOL.get(N);
        expect(primes).toBeDefined();
        expect(primes!.length).toBeGreaterThan(0);
      }
    });

    it('all primes in the pool satisfy NTT constraint', () => {
      for (const [N, primes] of NTT_PRIME_POOL) {
        for (const q of primes) {
          expect(isNttCompatible(N, q)).toBe(true);
        }
      }
    });
  });

  describe('seed', () => {
    it('creates a population of the configured size', () => {
      const pm = new PopulationManager({ populationSize: 50 });
      pm.seed();
      expect(pm.getPopulation().length).toBe(50);
    });

    it('starts at generation 0', () => {
      const pm = new PopulationManager();
      pm.seed();
      expect(pm.getGeneration()).toBe(0);
    });

    it('includes known-safe seed parameters verbatim', () => {
      const pm = new PopulationManager({ populationSize: 10 });
      pm.seed();
      const pop = pm.getPopulation();
      // First genome should be the first seed: N=1024, q=12289
      expect(pop[0].N).toBe(1024);
      expect(pop[0].q).toBe(12289);
      expect(pop[0].sigma).toBe(3200);
    });

    it('all seeded genomes have NTT-compatible (N, q) pairs', () => {
      const pm = new PopulationManager({ populationSize: 100 });
      pm.seed();
      for (const genome of pm.getPopulation()) {
        expect(isNttCompatible(genome.N, genome.q)).toBe(true);
      }
    });
  });

  describe('mutate', () => {
    it('returns a new Genome instance', () => {
      const pm = new PopulationManager({ mutationRate: 1.0, mutationStrength: 0.1 });
      const original = new Genome({ N: 4096, q: 40961, sigma: 3200, Bg: 2048 });
      const mutated = pm.mutate(original);
      expect(mutated).toBeInstanceOf(Genome);
      expect(mutated).not.toBe(original);
    });

    it('keeps N as a valid power of 2', () => {
      const pm = new PopulationManager({ mutationRate: 1.0, mutationStrength: 0.5 });
      const original = new Genome({ N: 4096, q: 40961, sigma: 3200, Bg: 2048 });
      const validN = [1024, 2048, 4096, 8192, 16384];
      for (let i = 0; i < 50; i++) {
        const mutated = pm.mutate(original);
        expect(validN).toContain(mutated.N);
      }
    });

    it('always produces NTT-compatible (N, q) pairs', () => {
      const pm = new PopulationManager({ mutationRate: 1.0, mutationStrength: 0.5 });
      const original = new Genome({ N: 4096, q: 40961, sigma: 3200, Bg: 2048 });
      for (let i = 0; i < 100; i++) {
        const mutated = pm.mutate(original);
        expect(isNttCompatible(mutated.N, mutated.q)).toBe(true);
      }
    });

    it('keeps sigma and Bg within bounds', () => {
      const pm = new PopulationManager({ mutationRate: 1.0, mutationStrength: 0.5 });
      const original = new Genome({ N: 4096, q: 40961, sigma: 3200, Bg: 2048 });
      for (let i = 0; i < 50; i++) {
        const m = pm.mutate(original);
        expect(m.sigma).toBeGreaterThanOrEqual(3200);
        expect(m.sigma).toBeLessThanOrEqual(10000);
        expect(m.Bg).toBeGreaterThanOrEqual(4);
        expect(m.Bg).toBeLessThanOrEqual(16384);
      }
    });
  });

  describe('crossover', () => {
    it('produces a genome with NTT-compatible (N, q) pair', () => {
      const pm = new PopulationManager();
      // Use NTT-compatible parents with different N values
      const a = new Genome({ N: 1024, q: 12289, sigma: 2000, Bg: 64 });
      const b = new Genome({ N: 4096, q: 786433, sigma: 5000, Bg: 8192 });

      for (let i = 0; i < 100; i++) {
        const child = pm.crossover(a, b);
        expect([a.N, b.N]).toContain(child.N);
        expect(isNttCompatible(child.N, child.q)).toBe(true);
        expect([a.sigma, b.sigma]).toContain(child.sigma);
        expect([a.Bg, b.Bg]).toContain(child.Bg);
      }
    });

    it('fixes incompatible (N, q) pairs from mixed parents', () => {
      const pm = new PopulationManager();
      // Both parents are individually valid under q ≡ 1 (mod 2N):
      //   N=8192, q=65537: 65536 % 16384 === 0 ✓
      //   N=1024, q=12289: 12288 % 2048  === 0 ✓
      // But the cross pairing conflicts:
      //   N=8192, q=12289: 12288 % 16384 === 12288 ≠ 0 ✗ → fixup must kick in
      const a = new Genome({ N: 8192, q: 65537, sigma: 2000, Bg: 64 });
      const b = new Genome({ N: 1024, q: 12289, sigma: 5000, Bg: 8192 });

      let sawFixup = false;
      for (let i = 0; i < 200; i++) {
        const child = pm.crossover(a, b);
        // Must always be NTT-compatible regardless of which N/q was chosen
        expect(isNttCompatible(child.N, child.q)).toBe(true);
        // If child got N=8192 but q=12289 was picked, the fixup must have kicked in
        if (child.N === 8192 && child.q !== 12289) {
          sawFixup = true;
        }
      }
      // With 200 iterations, we should see at least one fixup
      expect(sawFixup).toBe(true);
    });
  });

  describe('evolve', () => {
    it('increments the generation counter', () => {
      const pm = new PopulationManager({ populationSize: 20, eliteRatio: 0.5 });
      pm.seed();
      const scored: ScoredGenome[] = pm.getPopulation().map((genome, i) => ({
        genome,
        score: i, // higher index = better score
      }));
      pm.evolve(scored);
      expect(pm.getGeneration()).toBe(1);
    });

    it('maintains population size after evolution', () => {
      const pm = new PopulationManager({ populationSize: 30, eliteRatio: 0.1 });
      pm.seed();
      const scored: ScoredGenome[] = pm.getPopulation().map((genome, i) => ({
        genome,
        score: i * 10,
      }));
      pm.evolve(scored);
      expect(pm.getPopulation().length).toBe(30);
    });

    it('filters out failed genomes (score -1)', () => {
      const pm = new PopulationManager({ populationSize: 10, eliteRatio: 0.5 });
      pm.seed();
      const scored: ScoredGenome[] = pm.getPopulation().map((genome, i) => ({
        genome,
        score: i < 5 ? -1 : i * 10, // first 5 are failures
      }));
      pm.evolve(scored);
      // Should still produce full population
      expect(pm.getPopulation().length).toBe(10);
    });

    it('all evolved genomes have NTT-compatible (N, q) pairs', () => {
      const pm = new PopulationManager({ populationSize: 50, eliteRatio: 0.2 });
      pm.seed();
      const scored: ScoredGenome[] = pm.getPopulation().map((genome, i) => ({
        genome,
        score: i * 10,
      }));
      pm.evolve(scored);
      for (const genome of pm.getPopulation()) {
        expect(isNttCompatible(genome.N, genome.q)).toBe(true);
      }
    });
  });

  describe('packForFFI', () => {
    it('packs population into flat Uint32Array with correct layout', () => {
      const pm = new PopulationManager({ populationSize: 3 });
      pm.seed();
      const packed = pm.packForFFI();
      expect(packed).toBeInstanceOf(Uint32Array);
      expect(packed.length).toBe(3 * GENOME_FIELD_COUNT);

      // Verify first genome matches
      const pop = pm.getPopulation();
      expect(packed[0]).toBe(pop[0].N);
      expect(packed[1]).toBe(pop[0].q);
      expect(packed[2]).toBe(pop[0].sigma);
      expect(packed[3]).toBe(pop[0].Bg);
    });
  });
});
