/**
 * Core types for the FHE Parameter Evolution Simulation.
 */

/** Raw FHE parameters representing a single genome. */
export interface FheParams {
  /** Polynomial degree (power of 2, e.g. 1024, 2048, 4096, 8192, 16384) */
  N: number;
  /** Ciphertext modulus */
  q: number;
  /** Noise standard deviation (scaled by 1000 for integer representation) */
  sigma: number;
  /** Gadget decomposition base */
  Bg: number;
}

/** A genome paired with its fitness score from the benchmark. */
export interface ScoredGenome {
  genome: import('./genome.js').Genome;
  score: number;
}

/** A genome that has passed the security sieve, annotated with its noise budget. */
export interface SecureGenome {
  genome: import('./genome.js').Genome;
  /** Estimated security level in bits (>= 128). */
  securityBits: number;
  /** Theoretical noise growth budget (log2 of max noise before decryption failure). */
  noiseBudgetBits: number;
}

/** Result from the security estimator for a single parameter set. */
export interface SecurityEstimate {
  /** Estimated security level in bits. */
  securityBits: number;
  /** Whether the parameter set meets the 128-bit security floor. */
  isSecure: boolean;
  /** Theoretical noise growth budget in bits. */
  noiseBudgetBits: number;
}

/** Configuration for population seeding and evolution. */
export interface PopulationConfig {
  /** Number of genomes in each generation */
  populationSize: number;
  /** Fraction of top performers that survive (default: 0.05 = top 5%) */
  eliteRatio: number;
  /** Probability of mutation per parameter (0-1) */
  mutationRate: number;
  /** Maximum relative perturbation during mutation (0-1) */
  mutationStrength: number;
}

// ─── Multi-Objective / Pareto Types ─────────────────────────────────────────

/** Multi-objective fitness scores for a genome. */
export interface MultiObjectiveFitness {
  /** NTT latency in milliseconds (lower is better). */
  latencyMs: number;
  /** Estimated security level in bits (higher is better). */
  securityBits: number;
  /** Noise growth budget in bits (higher is better). */
  noiseBudgetBits: number;
  /** NSGA-II rank (1 = Pareto front, 2 = second front, etc.) */
  rank: number;
  /** Crowding distance within the rank (higher = more isolated = more diverse). */
  crowdingDistance: number;
}

/** A genome with full multi-objective evaluation. */
export interface EvaluatedGenome {
  genome: import('./genome.js').Genome;
  fitness: MultiObjectiveFitness;
}

/** A point on the Pareto frontier. */
export interface ParetoPoint {
  N: number;
  q: number;
  sigma: number;
  Bg: number;
  latencyMs: number;
  securityBits: number;
  noiseBudgetBits: number;
  generation: number;
}

/** Configuration for the Pareto explorer. */
export interface ParetoExplorerConfig {
  /** Generations per N-tier. */
  generationsPerTier: number;
  /** Population size per tier. */
  populationSize: number;
  /** Elite ratio for selection. */
  eliteRatio: number;
  /** Mutation rate. */
  mutationRate: number;
  /** Mutation strength. */
  mutationStrength: number;
  /** NTT iterations for statistical significance. */
  nttIterations: number;
  /** Output directory for results. */
  outputDir: string;
  /** If true, suppress per-generation console output. */
  quiet: boolean;
  /** Log every N generations (reduces noise). */
  logInterval: number;
}

