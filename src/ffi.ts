/**
 * FFI boundary wrapper for the Node.js ↔ Rust native module.
 *
 * This module provides a clean TypeScript interface for passing packed
 * FHE parameter batches to the Rust execution sandbox and receiving
 * benchmark scores back.
 *
 * Data contract (from design.md):
 *   Input:  Uint32Array — [N, q, σ, Bg, N, q, σ, Bg, ...]
 *   Output: Float64Array — [latency_ms, latency_ms, ...] (-1.0 for noise failure)
 */

import { createRequire } from 'node:module';
import { GENOME_FIELD_COUNT } from './genome.js';

/**
 * Snapshot of the Rust benchmark thread pool: P-core sized, interactive QoS.
 * Used by the bimodality smoke check to confirm pinning is live.
 */
export interface BenchmarkPoolInfo {
  threads: number;
  performanceCores: number;
  qosAvailable: boolean;
}

/**
 * The native module interface as exposed by napi-rs.
 * The actual module is loaded from the compiled .node binary.
 */
export interface NativeModule {
  runBenchmark(parameterBatch: Uint32Array): Float64Array;
  runBenchmarkDeep(parameterBatch: Uint32Array, iterations: number): Float64Array;
  validateBatch(parameterBatch: Uint32Array): number;
  echoGenome(parameterBatch: Uint32Array, index: number): Uint32Array;
  /** Present after the performance-core pinning change; optional for older mocks. */
  benchmarkPoolInfo?: () => BenchmarkPoolInfo;
}

/** Sentinel value indicating noise-induced corruption during benchmark. */
export const NOISE_FAILURE_SCORE = -1.0;

/**
 * Resolves the native module path and loads it.
 * Throws if the native binary has not been compiled.
 */
export function loadNativeModule(): NativeModule {
  const require = createRequire(import.meta.url);

  // napi-rs with cargo build produces a .dylib on macOS ARM64.
  // The build:native script copies it to project root as native.node.
  const possiblePaths = [
    '../native.node',
    '../native/target/release/libfhe_evolve_native.dylib',
    '../native/fhe-evolve-native.darwin-arm64.node',
    '../fhe-evolve-native.darwin-arm64.node',
    '../native/index.node',
  ];

  for (const relPath of possiblePaths) {
    try {
      return require(relPath) as NativeModule;
    } catch {
      // Try next path
    }
  }

  throw new Error(
    'Native module not found. Run `npm run build:native` to compile the Rust crate.',
  );
}

/**
 * Validates that a packed parameter batch has the correct structure.
 * @param batch - The packed Uint32Array from PopulationManager.packForFFI()
 * @returns The number of genomes in the batch
 * @throws If the buffer length is not a multiple of GENOME_FIELD_COUNT
 */
export function validateBatch(batch: Uint32Array): number {
  if (batch.length % GENOME_FIELD_COUNT !== 0) {
    throw new Error(
      `Invalid batch: buffer length ${batch.length} is not a multiple of ${GENOME_FIELD_COUNT}`,
    );
  }
  return batch.length / GENOME_FIELD_COUNT;
}

/**
 * Scores from the benchmark, annotated with genome index information.
 */
export interface BenchmarkResult {
  /** Raw Float64Array of scores (one per genome). */
  scores: Float64Array;
  /** Number of genomes that were benchmarked. */
  genomeCount: number;
  /** Number of genomes that failed due to noise (-1.0 scores). */
  failureCount: number;
}

function annotateScores(scores: Float64Array, genomeCount: number): BenchmarkResult {
  let failureCount = 0;
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] === NOISE_FAILURE_SCORE) {
      failureCount++;
    }
  }
  return { scores, genomeCount, failureCount };
}

/**
 * Execute the benchmark on a packed parameter batch.
 *
 * This is the primary FFI entry point. It:
 * 1. Validates the input batch structure
 * 2. Passes the flat Uint32Array across the FFI boundary to Rust
 * 3. Receives back a Float64Array of execution scores
 * 4. Returns annotated results
 *
 * @param nativeModule - The loaded native module instance
 * @param batch - Packed Uint32Array from PopulationManager.packForFFI()
 * @returns Benchmark results with scores and metadata
 */
export function executeBenchmark(
  nativeModule: NativeModule,
  batch: Uint32Array,
): BenchmarkResult {
  const genomeCount = validateBatch(batch);
  return annotateScores(nativeModule.runBenchmark(batch), genomeCount);
}

/**
 * Execute a deep benchmark with multiple NTT iterations per genome.
 * Each genome score is the minimum of several independent medians (min-of-k
 * in the Rust sandbox), for robustness against sticky efficiency-core placement.
 *
 * @param nativeModule - The loaded native module instance
 * @param batch - Packed Uint32Array from PopulationManager.packForFFI()
 * @param iterations - Number of NTT iterations per median trial (e.g. 100)
 * @returns Benchmark results with latency scores
 */
export function executeBenchmarkDeep(
  nativeModule: NativeModule,
  batch: Uint32Array,
  iterations: number,
): BenchmarkResult {
  const genomeCount = validateBatch(batch);
  return annotateScores(nativeModule.runBenchmarkDeep(batch, iterations), genomeCount);
}
