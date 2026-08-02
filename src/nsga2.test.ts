import { describe, it, expect } from 'vitest';
import { Genome } from './genome.js';
import { dominates, nonDominatedSort, assignCrowdingDistance, nsga2Select } from './nsga2.js';
import type { EvaluatedGenome, MultiObjectiveFitness } from './types.js';

/** Helper to create an EvaluatedGenome with given fitness values. */
function makeEval(latency: number, security: number, noise: number): EvaluatedGenome {
  const fitness: MultiObjectiveFitness = {
    latencyMs: latency,
    securityBits: security,
    noiseBudgetBits: noise,
    rank: 0,
    crowdingDistance: 0,
  };
  return {
    genome: new Genome({ N: 1024, q: 12289, sigma: 3200, Bg: 512 }),
    fitness,
  };
}

describe('NSGA-II', () => {
  describe('dominates', () => {
    it('A dominates B when strictly better on one objective, equal on others', () => {
      const a = makeEval(1.0, 130, 20); // faster
      const b = makeEval(2.0, 130, 20);
      expect(dominates(a, b)).toBe(true);
      expect(dominates(b, a)).toBe(false);
    });

    it('A dominates B when better on all objectives', () => {
      const a = makeEval(1.0, 200, 30);
      const b = makeEval(2.0, 150, 20);
      expect(dominates(a, b)).toBe(true);
      expect(dominates(b, a)).toBe(false);
    });

    it('neither dominates when tradeoff exists', () => {
      const a = makeEval(1.0, 128, 10); // faster but less secure
      const b = makeEval(2.0, 200, 30); // slower but more secure
      expect(dominates(a, b)).toBe(false);
      expect(dominates(b, a)).toBe(false);
    });

    it('equal genomes do not dominate each other', () => {
      const a = makeEval(1.0, 130, 20);
      const b = makeEval(1.0, 130, 20);
      expect(dominates(a, b)).toBe(false);
      expect(dominates(b, a)).toBe(false);
    });

    it('A does not dominate B if worse on any single objective', () => {
      const a = makeEval(1.0, 130, 19); // worse noise
      const b = makeEval(2.0, 130, 20);
      expect(dominates(a, b)).toBe(false); // a has better latency but worse noise
    });
  });

  describe('nonDominatedSort', () => {
    it('returns empty array for empty input', () => {
      const fronts = nonDominatedSort([]);
      expect(fronts).toEqual([]);
    });

    it('single genome is the Pareto front', () => {
      const g = makeEval(1.0, 130, 20);
      const fronts = nonDominatedSort([g]);
      expect(fronts.length).toBe(1);
      expect(fronts[0].length).toBe(1);
      expect(fronts[0][0].fitness.rank).toBe(1);
    });

    it('non-dominated genomes share the same front', () => {
      // These trade off: fast+low security vs slow+high security
      const a = makeEval(1.0, 128, 10);
      const b = makeEval(3.0, 200, 30);
      const fronts = nonDominatedSort([a, b]);
      expect(fronts.length).toBe(1);
      expect(fronts[0].length).toBe(2);
      expect(a.fitness.rank).toBe(1);
      expect(b.fitness.rank).toBe(1);
    });

    it('dominated genome goes to second front', () => {
      const a = makeEval(1.0, 200, 30); // dominates b and c
      const b = makeEval(3.0, 150, 25); // does not dominate c (worse latency, better sec/noise)
      const c = makeEval(2.0, 128, 10); // not dominated by b (better latency)
      // a dominates both b and c. b and c don't dominate each other (tradeoff).
      const fronts = nonDominatedSort([a, b, c]);
      expect(fronts.length).toBe(2);
      expect(a.fitness.rank).toBe(1);
      expect(b.fitness.rank).toBe(2);
      expect(c.fitness.rank).toBe(2);
    });

    it('assigns correct ranks to multiple fronts', () => {
      const genomes = [
        makeEval(1.0, 200, 30), // front 1
        makeEval(2.0, 190, 28), // front 2 (dominated by first)
        makeEval(3.0, 180, 25), // front 3
        makeEval(4.0, 170, 20), // front 4
      ];
      const fronts = nonDominatedSort(genomes);
      expect(fronts.length).toBe(4);
      expect(genomes[0].fitness.rank).toBe(1);
      expect(genomes[1].fitness.rank).toBe(2);
      expect(genomes[2].fitness.rank).toBe(3);
      expect(genomes[3].fitness.rank).toBe(4);
    });

    it('all genomes on same front when none dominate', () => {
      // Classic tradeoff: each is best in one objective
      const a = makeEval(1.0, 128, 5);  // best latency
      const b = makeEval(5.0, 250, 10); // best security
      const c = makeEval(3.0, 130, 50); // best noise
      const fronts = nonDominatedSort([a, b, c]);
      expect(fronts.length).toBe(1);
      expect(fronts[0].length).toBe(3);
    });
  });

  describe('assignCrowdingDistance', () => {
    it('handles empty front', () => {
      assignCrowdingDistance([]);
      // Should not throw
    });

    it('single genome gets infinite distance', () => {
      const g = makeEval(1.0, 130, 20);
      assignCrowdingDistance([g]);
      expect(g.fitness.crowdingDistance).toBe(Infinity);
    });

    it('two genomes both get infinite distance', () => {
      const a = makeEval(1.0, 130, 20);
      const b = makeEval(2.0, 200, 30);
      assignCrowdingDistance([a, b]);
      expect(a.fitness.crowdingDistance).toBe(Infinity);
      expect(b.fitness.crowdingDistance).toBe(Infinity);
    });

    it('boundary genomes get infinite distance, interior gets finite', () => {
      const a = makeEval(1.0, 128, 10);
      const b = makeEval(2.0, 160, 20);
      const c = makeEval(3.0, 200, 30);
      assignCrowdingDistance([a, b, c]);
      expect(a.fitness.crowdingDistance).toBe(Infinity);
      expect(c.fitness.crowdingDistance).toBe(Infinity);
      expect(b.fitness.crowdingDistance).toBeGreaterThan(0);
      expect(isFinite(b.fitness.crowdingDistance)).toBe(true);
    });
  });

  describe('nsga2Select', () => {
    it('returns all genomes when selectCount >= population size', () => {
      const pop = [makeEval(1.0, 130, 20), makeEval(2.0, 150, 25)];
      const selected = nsga2Select(pop, 5);
      expect(selected.length).toBe(2);
    });

    it('returns empty for empty population', () => {
      const selected = nsga2Select([], 5);
      expect(selected.length).toBe(0);
    });

    it('selects exactly selectCount genomes', () => {
      const pop = [
        makeEval(1.0, 200, 30),
        makeEval(2.0, 180, 25),
        makeEval(3.0, 160, 20),
        makeEval(4.0, 140, 15),
        makeEval(5.0, 128, 10),
      ];
      const selected = nsga2Select(pop, 3);
      expect(selected.length).toBe(3);
    });

    it('prefers rank 1 (Pareto front) genomes', () => {
      // a and b are non-dominated (Pareto front), c is dominated
      const a = makeEval(1.0, 128, 10); // best latency
      const b = makeEval(5.0, 200, 30); // best security
      const c = makeEval(3.0, 140, 15); // dominated by b (worse latency, worse security, worse noise)
      // Actually c is not dominated by b since c has better latency than b.
      // Let's make a properly dominated genome:
      const d = makeEval(6.0, 190, 28); // dominated by b (worse latency, less security, less noise)
      const selected = nsga2Select([a, b, c, d], 2);
      // Should prefer rank 1 genomes
      expect(selected.length).toBe(2);
      // All of a, b, c are on front 1, d is on front 2
      // With selectCount=2 and 3 on front 1, should pick by crowding distance
      expect(selected.every((s) => s.fitness.rank === 1)).toBe(true);
    });

    it('uses crowding distance as tiebreaker within same rank', () => {
      // 5 genomes all on Pareto front (tradeoffs), select 3
      const pop = [
        makeEval(1.0, 128, 5),
        makeEval(2.0, 140, 10),
        makeEval(3.0, 160, 20),
        makeEval(4.0, 180, 30),
        makeEval(5.0, 200, 40),
      ];
      const selected = nsga2Select(pop, 3);
      expect(selected.length).toBe(3);
      // Boundary points (extremes) should be preferred due to infinite crowding distance
      const latencies = selected.map((s) => s.fitness.latencyMs);
      expect(latencies).toContain(1.0); // boundary
      expect(latencies).toContain(5.0); // boundary
    });
  });
});
