/**
 * NSGA-II non-dominated sorting and crowding distance assignment.
 * Used to rank genomes in multi-objective space.
 *
 * Objectives (all minimized internally):
 *   - latencyMs: minimize
 *   - -securityBits: minimize (i.e., maximize security)
 *   - -noiseBudgetBits: minimize (i.e., maximize noise budget)
 */

import type { EvaluatedGenome } from './types.js';

/**
 * Check if genome A dominates genome B.
 *
 * A dominates B if A is better or equal on ALL objectives AND strictly
 * better on at least one:
 *   - A.latencyMs <= B.latencyMs
 *   - A.securityBits >= B.securityBits
 *   - A.noiseBudgetBits >= B.noiseBudgetBits
 *   - AND at least one strict inequality
 */
export function dominates(a: EvaluatedGenome, b: EvaluatedGenome): boolean {
  const aFit = a.fitness;
  const bFit = b.fitness;

  const betterOrEqualLatency = aFit.latencyMs <= bFit.latencyMs;
  const betterOrEqualSecurity = aFit.securityBits >= bFit.securityBits;
  const betterOrEqualNoise = aFit.noiseBudgetBits >= bFit.noiseBudgetBits;

  if (!betterOrEqualLatency || !betterOrEqualSecurity || !betterOrEqualNoise) {
    return false;
  }

  // At least one strict inequality
  const strictlyBetterLatency = aFit.latencyMs < bFit.latencyMs;
  const strictlyBetterSecurity = aFit.securityBits > bFit.securityBits;
  const strictlyBetterNoise = aFit.noiseBudgetBits > bFit.noiseBudgetBits;

  return strictlyBetterLatency || strictlyBetterSecurity || strictlyBetterNoise;
}

/**
 * Non-dominated sorting: partition genomes into successive Pareto fronts.
 *
 * front[0] = Pareto front (rank 1)
 * front[1] = second front (rank 2), etc.
 *
 * Also assigns `rank` to each genome's fitness.
 */
export function nonDominatedSort(genomes: EvaluatedGenome[]): EvaluatedGenome[][] {
  if (genomes.length === 0) return [];

  const n = genomes.length;

  // dominationCount[i] = number of genomes that dominate genome i
  const dominationCount = new Array<number>(n).fill(0);
  // dominatedSet[i] = indices of genomes that genome i dominates
  const dominatedSet: number[][] = Array.from({ length: n }, () => []);

  // Calculate domination relationships
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (dominates(genomes[i], genomes[j])) {
        dominatedSet[i].push(j);
        dominationCount[j]++;
      } else if (dominates(genomes[j], genomes[i])) {
        dominatedSet[j].push(i);
        dominationCount[i]++;
      }
    }
  }

  const fronts: EvaluatedGenome[][] = [];
  let currentFrontIndices: number[] = [];

  // First front: genomes not dominated by anyone
  for (let i = 0; i < n; i++) {
    if (dominationCount[i] === 0) {
      currentFrontIndices.push(i);
    }
  }

  let rank = 1;
  while (currentFrontIndices.length > 0) {
    const front: EvaluatedGenome[] = [];
    const nextFrontIndices: number[] = [];

    for (const idx of currentFrontIndices) {
      genomes[idx].fitness.rank = rank;
      front.push(genomes[idx]);

      // Reduce domination count for genomes dominated by this one
      for (const dominated of dominatedSet[idx]) {
        dominationCount[dominated]--;
        if (dominationCount[dominated] === 0) {
          nextFrontIndices.push(dominated);
        }
      }
    }

    fronts.push(front);
    currentFrontIndices = nextFrontIndices;
    rank++;
  }

  return fronts;
}

/**
 * Assign crowding distance to genomes within a single front.
 * Modifies genomes in place, setting fitness.crowdingDistance.
 *
 * Crowding distance measures how isolated a genome is in objective space.
 * Higher crowding distance = more diversity = preferred for selection.
 */
export function assignCrowdingDistance(front: EvaluatedGenome[]): void {
  const n = front.length;

  if (n === 0) return;

  // Reset crowding distances
  for (const g of front) {
    g.fitness.crowdingDistance = 0;
  }

  if (n <= 2) {
    // Edge case: 1-2 genomes get infinite crowding distance
    for (const g of front) {
      g.fitness.crowdingDistance = Infinity;
    }
    return;
  }

  // Objectives to consider (extract value and whether to maximize)
  const objectives: Array<{ getValue: (g: EvaluatedGenome) => number }> = [
    { getValue: (g) => g.fitness.latencyMs },         // minimize
    { getValue: (g) => -g.fitness.securityBits },     // minimize (negate to maximize security)
    { getValue: (g) => -g.fitness.noiseBudgetBits },  // minimize (negate to maximize noise)
  ];

  for (const obj of objectives) {
    // Sort by this objective
    const sorted = [...front].sort((a, b) => obj.getValue(a) - obj.getValue(b));

    // Boundary points get infinite distance
    sorted[0].fitness.crowdingDistance = Infinity;
    sorted[n - 1].fitness.crowdingDistance = Infinity;

    // Range of this objective
    const fMin = obj.getValue(sorted[0]);
    const fMax = obj.getValue(sorted[n - 1]);
    const range = fMax - fMin;

    if (range === 0) continue; // All values identical for this objective

    // Interior points: distance = (neighbor_right - neighbor_left) / range
    for (let i = 1; i < n - 1; i++) {
      const dist = (obj.getValue(sorted[i + 1]) - obj.getValue(sorted[i - 1])) / range;
      sorted[i].fitness.crowdingDistance += dist;
    }
  }
}

/**
 * NSGA-II selection: select the best N genomes using rank (lower is better)
 * as the primary criterion and crowding distance (higher is better) as tiebreaker.
 *
 * This performs non-dominated sorting, assigns crowding distances, then
 * fills the selection pool front-by-front. If a front doesn't fit entirely,
 * genomes within that front are sorted by crowding distance (descending).
 */
export function nsga2Select(population: EvaluatedGenome[], selectCount: number): EvaluatedGenome[] {
  if (population.length === 0) return [];
  if (selectCount >= population.length) return [...population];

  const fronts = nonDominatedSort(population);
  const selected: EvaluatedGenome[] = [];

  for (const front of fronts) {
    assignCrowdingDistance(front);

    if (selected.length + front.length <= selectCount) {
      // Entire front fits
      selected.push(...front);
    } else {
      // Partial front: sort by crowding distance (descending) and take what we need
      const remaining = selectCount - selected.length;
      const sorted = [...front].sort((a, b) => b.fitness.crowdingDistance - a.fitness.crowdingDistance);
      selected.push(...sorted.slice(0, remaining));
      break;
    }
  }

  return selected;
}
