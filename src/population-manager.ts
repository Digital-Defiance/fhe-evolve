/**
 * PopulationManager handles the evolutionary lifecycle:
 *  - Seeding the initial population with known-safe FHE parameters
 *  - Mutating genomes (random perturbation within valid ranges)
 *  - Crossover (combining parameters from two parents)
 *  - Selection of the top 5% to produce the next generation
 *
 * CRITICAL: All (N, q) pairs must satisfy the negacyclic NTT constraint:
 *   q is prime AND (q - 1) % (2 * N) === 0
 * FHE operates in Z_q[X]/(X^N + 1), which requires a primitive 2N-th root of
 * unity modulo q. The weaker q ≡ 1 (mod N) condition only guarantees a
 * primitive N-th root, which is the cyclic transform over X^N − 1.
 */

import { Genome, GENOME_FIELD_COUNT } from './genome.js';
import type { FheParams, PopulationConfig, ScoredGenome } from './types.js';

// ─── NTT-Friendly Prime Pool ────────────────────────────────────────────────

/** Simple primality test (deterministic for values < 2^32). */
function isPrime(n: number): boolean {
  if (n < 2) return false;
  if (n === 2 || n === 3) return true;
  if (n % 2 === 0 || n % 3 === 0) return false;
  for (let i = 5; i * i <= n; i += 6) {
    if (n % i === 0 || n % (i + 2) === 0) return false;
  }
  return true;
}

/**
 * Find NTT-friendly primes for a given N.
 * These are primes q = 2kN + 1, i.e., q ≡ 1 (mod 2N), which guarantees a
 * primitive 2N-th root of unity and is therefore valid for the negacyclic
 * ring Z_q[X]/(X^N + 1) used by FHE.
 * We cap at u32 max (~4.29 billion) since q is packed as a u32 for FFI.
 */
function findNttFriendlyPrimes(N: number, maxQ: number, count: number): number[] {
  const primes: number[] = [];
  for (let k = 1; primes.length < count; k++) {
    const candidate = 2 * k * N + 1;
    if (candidate > maxQ) break;
    if (isPrime(candidate)) primes.push(candidate);
  }
  return primes;
}

/** Maximum q value (fits in u32). */
const MAX_Q = 2 ** 32 - 1;

/** Valid polynomial degrees (must be powers of 2). */
const VALID_N = [1024, 2048, 4096, 8192, 16384];

/**
 * Pre-computed pool of NTT-friendly primes for each valid N.
 * For each N, we find up to 50 primes where q = 2*k*N + 1 is prime.
 * This gives the evolutionary algorithm room to explore different q values
 * while always satisfying the negacyclic NTT constraint.
 */
export const NTT_PRIME_POOL: Map<number, number[]> = new Map(
  VALID_N.map((N) => [N, findNttFriendlyPrimes(N, MAX_Q, 50)]),
);

/**
 * Pick a random NTT-friendly prime for the given N from the pool.
 */
function randomNttPrime(N: number): number {
  const primes = NTT_PRIME_POOL.get(N);
  if (!primes || primes.length === 0) {
    throw new Error(`No NTT-friendly primes available for N=${N}`);
  }
  return primes[Math.floor(Math.random() * primes.length)];
}

/**
 * Ensure q is NTT-compatible with N. If not, replace with a random compatible prime.
 */
function ensureNttCompatible(N: number, q: number): number {
  if (isPrime(q) && (q - 1) % (2 * N) === 0) {
    return q;
  }
  return randomNttPrime(N);
}

// ─── Seed Parameters (NTT-friendly) ────────────────────────────────────────

/** Known-safe FHE parameter sets used as seeds — all satisfy q ≡ 1 (mod 2N) and q is prime. */
const SEED_PARAMS: FheParams[] = [
  { N: 1024, q: 12289, sigma: 3200, Bg: 512 },
  { N: 1024, q: 40961, sigma: 3200, Bg: 512 },
  { N: 1024, q: 65537, sigma: 3200, Bg: 1024 },
  { N: 2048, q: 65537, sigma: 3200, Bg: 1024 },
  { N: 2048, q: 40961, sigma: 3200, Bg: 1024 },
  { N: 4096, q: 114689, sigma: 3200, Bg: 2048 },
  { N: 4096, q: 65537, sigma: 3200, Bg: 2048 },
  { N: 4096, q: 786433, sigma: 3200, Bg: 2048 },
];

/** Minimum and maximum bounds for each parameter. */
const PARAM_BOUNDS = {
  N: { min: 1024, max: 16384 },
  sigma: { min: 3200, max: 10000 },          // 3.2 to 10.0 (scaled x1000)
  Bg: { min: 4, max: 16384 },
} as const;

const DEFAULT_CONFIG: PopulationConfig = {
  populationSize: 100,
  eliteRatio: 0.05,
  mutationRate: 0.8,
  mutationStrength: 0.2,
};

export class PopulationManager {
  private population: Genome[] = [];
  private generation = 0;
  private readonly config: PopulationConfig;

  constructor(config?: Partial<PopulationConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /** Current generation number. */
  getGeneration(): number {
    return this.generation;
  }

  /** Current population. */
  getPopulation(): readonly Genome[] {
    return this.population;
  }

  /** Population size from config. */
  getPopulationSize(): number {
    return this.config.populationSize;
  }

  /**
   * Seed the initial population by cycling through known-safe parameters
   * and applying light mutations to fill the requested population size.
   */
  seed(): void {
    this.population = [];
    this.generation = 0;

    for (let i = 0; i < this.config.populationSize; i++) {
      const base = SEED_PARAMS[i % SEED_PARAMS.length];
      if (i < SEED_PARAMS.length) {
        // First few genomes are the exact known-safe seeds
        this.population.push(new Genome(base));
      } else {
        // Remaining are lightly mutated versions of the seeds
        this.population.push(this.mutate(new Genome(base)));
      }
    }
  }

  /**
   * Mutate a genome by applying random perturbations within valid bounds.
   * Each parameter has a `mutationRate` probability of being perturbed.
   *
   * CRITICAL: q is always chosen from the NTT-friendly prime pool for the current N.
   * When N changes, q is also updated to remain compatible.
   */
  mutate(genome: Genome): Genome {
    const params: FheParams = {
      N: genome.N,
      q: genome.q,
      sigma: genome.sigma,
      Bg: genome.Bg,
    };

    // N must stay a power of 2 — pick a neighboring valid value
    let nChanged = false;
    if (Math.random() < this.config.mutationRate) {
      const currentIdx = VALID_N.indexOf(params.N);
      const idx = currentIdx >= 0 ? currentIdx : Math.floor(Math.random() * VALID_N.length);
      const shift = Math.random() < 0.5 ? -1 : 1;
      const newIdx = Math.max(0, Math.min(VALID_N.length - 1, idx + shift));
      if (VALID_N[newIdx] !== params.N) {
        params.N = VALID_N[newIdx];
        nChanged = true;
      }
    }

    // q: pick from the NTT-friendly prime pool for current N
    // Always re-pick if N changed; otherwise mutate with probability mutationRate
    if (nChanged || Math.random() < this.config.mutationRate) {
      params.q = randomNttPrime(params.N);
    }

    // sigma: perturbation within bounds
    if (Math.random() < this.config.mutationRate) {
      params.sigma = this.perturbParam(params.sigma, PARAM_BOUNDS.sigma.min, PARAM_BOUNDS.sigma.max);
    }

    // Bg: perturbation within bounds
    if (Math.random() < this.config.mutationRate) {
      params.Bg = this.perturbParam(params.Bg, PARAM_BOUNDS.Bg.min, PARAM_BOUNDS.Bg.max);
    }

    return new Genome(params);
  }

  /**
   * Single-point crossover: for each parameter, randomly pick from parent A or B.
   * Post-crossover fixup ensures (N, q) compatibility.
   */
  crossover(parentA: Genome, parentB: Genome): Genome {
    const N = Math.random() < 0.5 ? parentA.N : parentB.N;
    const q = Math.random() < 0.5 ? parentA.q : parentB.q;
    const sigma = Math.random() < 0.5 ? parentA.sigma : parentB.sigma;
    const Bg = Math.random() < 0.5 ? parentA.Bg : parentB.Bg;

    // Post-crossover fixup: ensure q is NTT-compatible with the chosen N
    const fixedQ = ensureNttCompatible(N, q);

    return new Genome({ N, q: fixedQ, sigma, Bg });
  }

  /**
   * Select the top performers from scored genomes and generate the next generation
   * via crossover and mutation of the elite subset.
   *
   * @param scoredPopulation - Population with fitness scores (higher is better, -1 is failure)
   */
  evolve(scoredPopulation: ScoredGenome[]): void {
    // Filter out failures (score === -1) and sort descending by score
    const viable = scoredPopulation
      .filter((s) => s.score >= 0)
      .sort((a, b) => b.score - a.score);

    const eliteCount = Math.max(
      2, // Need at least 2 parents for crossover
      Math.ceil(viable.length * this.config.eliteRatio),
    );

    const elites = viable.slice(0, eliteCount).map((s) => s.genome);

    // Build next generation
    const nextGen: Genome[] = [];

    // Keep elites unchanged
    for (const elite of elites) {
      nextGen.push(elite);
    }

    // Fill the rest with crossover + mutation of elites
    while (nextGen.length < this.config.populationSize) {
      const parentA = elites[Math.floor(Math.random() * elites.length)];
      const parentB = elites[Math.floor(Math.random() * elites.length)];
      const child = this.crossover(parentA, parentB);
      nextGen.push(this.mutate(child));
    }

    this.population = nextGen;
    this.generation++;
  }

  /**
   * Pack the entire current population into a flat Uint32Array
   * suitable for passing across the FFI boundary.
   *
   * Layout: [N0, q0, σ0, Bg0, N1, q1, σ1, Bg1, ...]
   */
  packForFFI(): Uint32Array {
    const buffer = new Uint32Array(this.population.length * GENOME_FIELD_COUNT);
    for (let i = 0; i < this.population.length; i++) {
      this.population[i].packInto(buffer, i * GENOME_FIELD_COUNT);
    }
    return buffer;
  }

  /**
   * Apply a random perturbation to a numeric parameter, clamped to bounds.
   */
  private perturbParam(value: number, min: number, max: number): number {
    const delta = value * this.config.mutationStrength * (Math.random() * 2 - 1);
    const perturbed = Math.round(value + delta);
    return Math.max(min, Math.min(max, perturbed));
  }
}
