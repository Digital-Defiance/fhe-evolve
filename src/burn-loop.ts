/**
 * The Burn Loop — continuous generation loop connecting the Orchestrator,
 * the Sieve (Guillotine), and the Sandbox (Rust FFI benchmark).
 *
 * Data flow per generation:
 *  1. Run population through the Guillotine (filter insecure genomes)
 *  2. Pack secure genomes into Uint32Array
 *  3. Pass batch to Rust sandbox via FFI (executeBenchmark)
 *  4. Map scores back to ScoredGenome[]
 *  5. Call PopulationManager.evolve() with the scored population
 *  6. Log generation stats
 */

import { Genome, GENOME_FIELD_COUNT } from './genome.js';
import { PopulationManager } from './population-manager.js';
import { guillotine } from './guillotine.js';
import { executeBenchmark, NOISE_FAILURE_SCORE, type NativeModule } from './ffi.js';
import { Logger, type GenerationStats } from './logger.js';
import type { PopulationConfig, ScoredGenome, SecureGenome } from './types.js';

/** Configuration for the burn loop. */
export interface BurnLoopConfig {
  /** Number of generations to run. 0 = infinite (until shutdown). */
  maxGenerations: number;
  /** Population configuration passed to PopulationManager. */
  population: Partial<PopulationConfig>;
  /** Number of entries on the all-time leaderboard. */
  leaderboardSize: number;
  /** If true, suppress console output. */
  silent: boolean;
}

/** Result summary from a completed burn loop run. */
export interface BurnLoopResult {
  /** Total generations completed. */
  generations: number;
  /** Whether the loop was stopped via shutdown signal. */
  wasShutdown: boolean;
  /** The logger instance with full history and leaderboard. */
  logger: Logger;
}

const DEFAULT_BURN_CONFIG: BurnLoopConfig = {
  maxGenerations: 100,
  population: {},
  leaderboardSize: 10,
  silent: false,
};

export class BurnLoop {
  private readonly config: BurnLoopConfig;
  private readonly populationManager: PopulationManager;
  private readonly logger: Logger;
  private readonly nativeModule: NativeModule;
  private shutdownRequested = false;

  constructor(nativeModule: NativeModule, config?: Partial<BurnLoopConfig>) {
    this.config = { ...DEFAULT_BURN_CONFIG, ...config };
    this.nativeModule = nativeModule;
    this.populationManager = new PopulationManager(this.config.population);
    this.logger = new Logger({
      leaderboardSize: this.config.leaderboardSize,
      silent: this.config.silent,
    });
  }

  /**
   * Request a graceful shutdown after the current generation completes.
   */
  shutdown(): void {
    this.shutdownRequested = true;
  }

  /**
   * Check if shutdown has been requested.
   */
  isShutdownRequested(): boolean {
    return this.shutdownRequested;
  }

  /**
   * Get the underlying logger (for accessing leaderboard/history after run).
   */
  getLogger(): Logger {
    return this.logger;
  }

  /**
   * Get the population manager (for inspection/testing).
   */
  getPopulationManager(): PopulationManager {
    return this.populationManager;
  }

  /**
   * Run the full burn loop.
   *
   * Initializes the population, then iterates through generations:
   * filter → benchmark → score → evolve → log → repeat.
   */
  run(): BurnLoopResult {
    this.populationManager.seed();
    this.shutdownRequested = false;

    let generationsCompleted = 0;

    while (!this.shouldStop(generationsCompleted)) {
      this.runGeneration();
      generationsCompleted++;
    }

    // Final leaderboard dump
    this.logger.logLeaderboard();

    return {
      generations: generationsCompleted,
      wasShutdown: this.shutdownRequested,
      logger: this.logger,
    };
  }

  /**
   * Execute a single generation cycle.
   * Exposed for testing individual steps.
   */
  runGeneration(): GenerationStats {
    const population = this.populationManager.getPopulation();
    const generation = this.populationManager.getGeneration();

    // Step 1: Filter through the Guillotine
    const secureGenomes = guillotine(population);

    if (secureGenomes.length === 0) {
      // No genomes passed security — reseed and try again
      this.populationManager.seed();
      return {
        generation,
        populationSize: population.length,
        secureCount: 0,
        bestLatencyMs: Infinity,
        avgLatencyMs: Infinity,
        survivalRate: 0,
        fastestGenome: population[0],
        timestamp: Date.now(),
      };
    }

    // Step 2: Pack secure genomes into Uint32Array
    const batch = this.packSecureGenomes(secureGenomes);

    // Step 3: Pass to Rust sandbox via FFI
    const benchmarkResult = executeBenchmark(this.nativeModule, batch);

    // Step 4: Map scores back to ScoredGenome[]
    const scoredGenomes = this.mapScores(secureGenomes, benchmarkResult.scores);

    // Check if we have any viable results for evolution
    const viableCount = scoredGenomes.filter((s) => s.score >= 0).length;

    if (viableCount < 2) {
      // Not enough viable genomes to evolve — reseed
      this.populationManager.seed();
      const stats: GenerationStats = {
        generation,
        populationSize: population.length,
        secureCount: secureGenomes.length,
        bestLatencyMs: Infinity,
        avgLatencyMs: Infinity,
        survivalRate: secureGenomes.length / population.length,
        fastestGenome: secureGenomes[0].genome,
        timestamp: Date.now(),
      };
      this.logger.logGeneration(stats);
      return stats;
    }

    // Step 5: Evolve the population using scored results
    this.populationManager.evolve(scoredGenomes);

    // Step 6: Log generation stats
    const stats = this.computeStats(
      generation,
      population.length,
      secureGenomes.length,
      scoredGenomes,
    );

    this.logger.logGeneration(stats);

    return stats;
  }

  /**
   * Pack an array of SecureGenomes into a flat Uint32Array for FFI.
   */
  private packSecureGenomes(secureGenomes: SecureGenome[]): Uint32Array {
    const buffer = new Uint32Array(secureGenomes.length * GENOME_FIELD_COUNT);
    for (let i = 0; i < secureGenomes.length; i++) {
      secureGenomes[i].genome.packInto(buffer, i * GENOME_FIELD_COUNT);
    }
    return buffer;
  }

  /**
   * Map benchmark scores back to ScoredGenome format.
   * Lower latency = better score, so we invert: score = 1/latency.
   * Noise failures (-1.0) remain as -1.
   */
  private mapScores(secureGenomes: SecureGenome[], scores: Float64Array): ScoredGenome[] {
    const result: ScoredGenome[] = [];
    for (let i = 0; i < secureGenomes.length; i++) {
      const rawLatency = scores[i];
      // For evolution, we want higher = better. Invert latency so faster configs rank higher.
      const score = rawLatency === NOISE_FAILURE_SCORE ? -1 : 1 / rawLatency;
      result.push({
        genome: secureGenomes[i].genome,
        score,
      });
    }
    return result;
  }

  /**
   * Compute generation statistics from scored results.
   */
  private computeStats(
    generation: number,
    populationSize: number,
    secureCount: number,
    scoredGenomes: ScoredGenome[],
  ): GenerationStats {
    const viable = scoredGenomes.filter((s) => s.score >= 0);

    // Find the fastest (lowest raw latency = highest inverted score)
    let bestScore = -Infinity;
    let fastestGenome: Genome = scoredGenomes[0].genome;

    let totalScore = 0;
    for (const sg of viable) {
      totalScore += sg.score;
      if (sg.score > bestScore) {
        bestScore = sg.score;
        fastestGenome = sg.genome;
      }
    }

    // Convert back from inverted score to latency for display
    const bestLatencyMs = bestScore > 0 ? 1 / bestScore : Infinity;
    const avgLatencyMs = viable.length > 0 ? viable.length / totalScore : Infinity;

    return {
      generation,
      populationSize,
      secureCount,
      bestLatencyMs,
      avgLatencyMs,
      survivalRate: secureCount / populationSize,
      fastestGenome,
      timestamp: Date.now(),
    };
  }

  /**
   * Determine if the loop should stop.
   */
  private shouldStop(generationsCompleted: number): boolean {
    if (this.shutdownRequested) return true;
    if (this.config.maxGenerations > 0 && generationsCompleted >= this.config.maxGenerations) {
      return true;
    }
    return false;
  }
}
