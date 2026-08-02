export { Genome, GENOME_FIELD_COUNT } from './genome.js';
export { PopulationManager } from './population-manager.js';
export {
  estimateSecurity,
  estimateSecurityBits,
  estimateNoiseBudget,
  SECURITY_FLOOR_BITS,
} from './security-estimator.js';
export { guillotine, evaluateGenome } from './guillotine.js';
export {
  loadNativeModule,
  validateBatch,
  executeBenchmark,
  executeBenchmarkDeep,
  NOISE_FAILURE_SCORE,
} from './ffi.js';
export type { NativeModule, BenchmarkResult } from './ffi.js';
export { BurnLoop } from './burn-loop.js';
export type { BurnLoopConfig, BurnLoopResult } from './burn-loop.js';
export { Logger } from './logger.js';
export type { GenerationStats, LeaderboardEntry, LoggerConfig } from './logger.js';
export { dominates, nonDominatedSort, assignCrowdingDistance, nsga2Select } from './nsga2.js';
export { ParetoExplorer, formatDuration } from './pareto-explorer.js';
export type { ParetoExplorerResult, ConvergenceEntry } from './pareto-explorer.js';
export { writeParetoResults, generateSummary, HE_STANDARD_2018 } from './pareto-output.js';
export type {
  FheParams,
  ScoredGenome,
  PopulationConfig,
  SecureGenome,
  SecurityEstimate,
  MultiObjectiveFitness,
  EvaluatedGenome,
  ParetoPoint,
  ParetoExplorerConfig,
} from './types.js';
