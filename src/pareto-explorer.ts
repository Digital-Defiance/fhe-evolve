/**
 * ParetoExplorer — the main scientific engine for multi-objective FHE parameter optimization.
 *
 * Replaces the simple BurnLoop for serious Pareto frontier exploration runs.
 * Runs separate populations per N bracket (1024, 2048, 4096, 8192, 16384),
 * using NSGA-II selection to map the tradeoff space between security,
 * noise budget, and latency.
 */

import { Genome, GENOME_FIELD_COUNT } from './genome.js';
import { PopulationManager } from './population-manager.js';
import { guillotine } from './guillotine.js';
import { executeBenchmarkDeep, NOISE_FAILURE_SCORE, type NativeModule } from './ffi.js';
import { nsga2Select } from './nsga2.js';
import type {
  EvaluatedGenome,
  MultiObjectiveFitness,
  ParetoExplorerConfig,
  ParetoPoint,
  SecureGenome,
} from './types.js';

/** Valid N tiers for exploration. */
const N_TIERS = [1024, 2048, 4096, 8192, 16384] as const;

/** Minimum generations before showing ETA (too noisy before this). */
const MIN_GENS_FOR_ETA = 5;

/** Format a duration in milliseconds to a human-readable string. */
export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '??';
  const totalSeconds = Math.round(ms / 1000);
  if (totalSeconds < 60) return `${totalSeconds}s`;
  if (totalSeconds < 3600) {
    const m = Math.floor(totalSeconds / 60);
    const s = totalSeconds % 60;
    return s > 0 ? `${m}m ${s}s` : `${m}m`;
  }
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.round((totalSeconds % 3600) / 60);
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

/**
 * Compute the NTT cost factor for a given N: N * log2(N).
 * Used to project timing across tiers.
 */
function nttCostFactor(N: number): number {
  return N * Math.log2(N);
}

/** Default configuration for serious exploration runs. */
const DEFAULT_CONFIG: ParetoExplorerConfig = {
  generationsPerTier: 10000,
  populationSize: 500,
  eliteRatio: 0.1,
  mutationRate: 0.7,
  mutationStrength: 0.15,
  nttIterations: 100,
  outputDir: './results',
  quiet: false,
  logInterval: 100,
};

/** Per-generation convergence stats. */
export interface ConvergenceEntry {
  tier: number;
  generation: number;
  paretoFrontSize: number;
  bestLatencyMs: number;
  bestSecurityBits: number;
  bestNoiseBudgetBits: number;
  populationSize: number;
  secureCount: number;
}

/** Result from a full Pareto exploration run. */
export interface ParetoExplorerResult {
  /** All Pareto-optimal points merged across tiers. */
  paretoFrontier: ParetoPoint[];
  /** Per-tier Pareto fronts. */
  perTierFronts: Map<number, ParetoPoint[]>;
  /** Per-generation convergence stats for analysis. */
  convergence: ConvergenceEntry[];
  /** Total wall-clock time in milliseconds. */
  wallClockMs: number;
  /** Whether the run was interrupted. */
  wasInterrupted: boolean;
}

export class ParetoExplorer {
  private readonly config: ParetoExplorerConfig;
  private readonly nativeModule: NativeModule;
  private interrupted = false;

  constructor(nativeModule: NativeModule, config?: Partial<ParetoExplorerConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.nativeModule = nativeModule;
  }

  /** Request graceful interruption. */
  interrupt(): void {
    this.interrupted = true;
  }

  /** Check if interrupted. */
  isInterrupted(): boolean {
    return this.interrupted;
  }

  /** Run the full exploration across all N tiers. */
  run(): ParetoExplorerResult {
    const startTime = Date.now();
    this.interrupted = false;

    const perTierFronts = new Map<number, ParetoPoint[]>();
    const convergence: ConvergenceEntry[] = [];

    // Track completed tier timings for ETA projection
    const completedTierTimings = new Map<number, number>(); // N -> elapsed ms

    for (let tierIdx = 0; tierIdx < N_TIERS.length; tierIdx++) {
      const N = N_TIERS[tierIdx];
      if (this.interrupted) break;

      // Project tier duration based on completed timings or scaling
      const tierEstimate = this.projectTierDuration(N, completedTierTimings);

      if (!this.config.quiet) {
        const estStr = tierEstimate > 0 ? ` (est. ~${formatDuration(tierEstimate)})` : '';
        console.log(`\n═══ Tier N=${N}${estStr} ═══`);
      }

      const tierStartTime = Date.now();
      const remainingTiers = N_TIERS.slice(tierIdx + 1);
      const tierResult = this.runTierInternal(N, convergence, tierStartTime, completedTierTimings, remainingTiers);
      const tierElapsed = Date.now() - tierStartTime;
      completedTierTimings.set(N, tierElapsed);

      perTierFronts.set(N, tierResult);

      if (!this.config.quiet) {
        console.log(`  Tier N=${N} completed in ${formatDuration(tierElapsed)}`);
        console.log(`  Pareto front: ${tierResult.length} points`);
        if (tierResult.length > 0) {
          const best = tierResult.reduce((a, b) => a.latencyMs < b.latencyMs ? a : b);
          console.log(`  Best latency: ${best.latencyMs.toFixed(3)}ms (security: ${best.securityBits.toFixed(1)} bits)`);
        }

        // Show remaining tiers ETA
        if (remainingTiers.length > 0 && !this.interrupted) {
          const remainingEta = this.computeRemainingTiersEta(remainingTiers, completedTierTimings, N, tierElapsed);
          if (remainingEta > 0) {
            console.log(`  ...remaining tiers ETA: ${formatDuration(remainingEta)}`);
          }
        }
      }
    }

    // Merge all Pareto fronts
    const allPoints: ParetoPoint[] = [];
    for (const front of perTierFronts.values()) {
      allPoints.push(...front);
    }

    // Deduplicate: same N/q/σ/Bg → keep the one with better latency
    const paretoFrontier = this.deduplicatePoints(allPoints);

    const wallClockMs = Date.now() - startTime;

    return {
      paretoFrontier,
      perTierFronts,
      convergence,
      wallClockMs,
      wasInterrupted: this.interrupted,
    };
  }

  /** Run a single N-tier and return its Pareto front. */
  runTier(N: number): ParetoPoint[] {
    const convergence: ConvergenceEntry[] = [];
    const tierStartTime = Date.now();
    return this.runTierInternal(N, convergence, tierStartTime, new Map(), []);
  }

  private runTierInternal(
    N: number,
    convergence: ConvergenceEntry[],
    tierStartTime: number,
    completedTierTimings: Map<number, number>,
    remainingTiers: readonly number[],
  ): ParetoPoint[] {
    // Create a population manager locked to this N tier
    const pm = new PopulationManager({
      populationSize: this.config.populationSize,
      eliteRatio: this.config.eliteRatio,
      mutationRate: this.config.mutationRate,
      mutationStrength: this.config.mutationStrength,
    });

    // Seed with this N tier's population
    pm.seed();

    // Force all genomes to use this N.
    //
    // The mutation operator can change N, so every mutation inside a tier must
    // be followed by re-locking N. This used to construct an N-locked genome
    // and then mutate it, which let mutation undo the lock immediately. The
    // result was that each tier's population -- and therefore its Pareto front
    // -- contained genomes belonging to the tier above it: 16 N=2048 genomes in
    // the N=1024 front, 17 N=4096 in the N=2048 front, and so on. The top tier
    // was the only clean one, because there is no tier above it to leak from.
    //
    // Latency roughly doubles per doubling of N, so a leaked genome is usually
    // dominated on latency, but it can still win on security or noise budget
    // and occupy a front slot that belongs to this tier.
    const lockN = (g: Genome): Genome =>
      g.N === N ? g : new Genome({ N, q: g.q, sigma: g.sigma, Bg: g.Bg });

    /** Mutate, then re-lock N. Use this everywhere inside a tier. */
    const mutateInTier = (g: Genome): Genome => lockN(pm['mutate'](g));

    // Preserves the original semantics: a genome already at this N is left
    // alone; one that is not is rebuilt at this N and mutated. The only change
    // is that the mutation is now followed by a re-lock.
    let population = pm.getPopulation().map((g) =>
      g.N !== N ? mutateInTier(new Genome({ N, q: g.q, sigma: g.sigma, Bg: g.Bg })) : g,
    );

    let bestFront: EvaluatedGenome[] = [];

    for (let gen = 0; gen < this.config.generationsPerTier; gen++) {
      if (this.interrupted) break;

      // Step 1: Security sieve
      const secureGenomes = guillotine(population);

      if (secureGenomes.length === 0) {
        // Reseed if nothing passes security
        pm.seed();
        population = pm.getPopulation().map((g) => {
          if (g.N !== N) {
            return new Genome({ N, q: g.q, sigma: g.sigma, Bg: g.Bg });
          }
          return g;
        });
        continue;
      }

      // Step 2: Pack for FFI
      const batch = this.packSecureGenomes(secureGenomes);

      // Step 3: Deep benchmark
      const benchResult = executeBenchmarkDeep(this.nativeModule, batch, this.config.nttIterations);

      // Step 4: Compute multi-objective fitness
      const evaluated = this.buildEvaluated(secureGenomes, benchResult.scores);

      if (evaluated.length < 2) {
        // Not enough viable genomes, continue with mutation.
        // mutateInTier, not pm.mutate: this path ran every time a generation
        // produced fewer than two viable genomes, and it was the main route by
        // which genomes from the next tier up entered this tier's population.
        population = population.map((g) => mutateInTier(g));
        continue;
      }

      // Step 5: NSGA-II selection
      const selectCount = Math.max(2, Math.ceil(evaluated.length * this.config.eliteRatio));
      const selected = nsga2Select(evaluated, selectCount);

      // Track the Pareto front (rank 1)
      bestFront = evaluated.filter((e) => e.fitness.rank === 1);

      // Step 6: Generate next generation via crossover + mutation
      population = this.breed(selected, pm, N);

      // Convergence tracking
      if (gen % this.config.logInterval === 0 || gen === this.config.generationsPerTier - 1) {
        const entry: ConvergenceEntry = {
          tier: N,
          generation: gen,
          paretoFrontSize: bestFront.length,
          bestLatencyMs: bestFront.length > 0
            ? Math.min(...bestFront.map((e) => e.fitness.latencyMs))
            : Infinity,
          bestSecurityBits: bestFront.length > 0
            ? Math.max(...bestFront.map((e) => e.fitness.securityBits))
            : 0,
          bestNoiseBudgetBits: bestFront.length > 0
            ? Math.max(...bestFront.map((e) => e.fitness.noiseBudgetBits))
            : 0,
          populationSize: population.length,
          secureCount: secureGenomes.length,
        };
        convergence.push(entry);

        if (!this.config.quiet) {
          let logLine =
            `  Gen ${gen} | secure: ${secureGenomes.length} | front: ${bestFront.length} ` +
            `| best latency: ${entry.bestLatencyMs === Infinity ? 'N/A' : entry.bestLatencyMs.toFixed(3) + 'ms'}`;

          // Add ETA after enough generations for stable estimates
          if (gen >= MIN_GENS_FOR_ETA) {
            const elapsedMs = Date.now() - tierStartTime;
            const msPerGen = elapsedMs / (gen + 1);
            const remainingGens = this.config.generationsPerTier - gen - 1;
            const tierEtaMs = msPerGen * remainingGens;

            // Total ETA: remaining current tier + projected remaining tiers
            const totalEtaMs = tierEtaMs + this.computeRemainingTiersEta(
              remainingTiers, completedTierTimings, N, undefined, msPerGen,
            );

            logLine += ` | ETA tier: ${formatDuration(tierEtaMs)} | ETA total: ${formatDuration(totalEtaMs)}`;
          }

          console.log(logLine);
        }
      }
    }

    // Extract Pareto points from the final front
    return bestFront.map((e) => ({
      N: e.genome.N,
      q: e.genome.q,
      sigma: e.genome.sigma,
      Bg: e.genome.Bg,
      latencyMs: e.fitness.latencyMs,
      securityBits: e.fitness.securityBits,
      noiseBudgetBits: e.fitness.noiseBudgetBits,
      generation: this.config.generationsPerTier - 1,
    }));
  }

  /**
   * Project how long a tier will take, based on completed timings or NTT scaling.
   */
  private projectTierDuration(N: number, completedTierTimings: Map<number, number>): number {
    if (completedTierTimings.size === 0) return 0;

    // Use the most recently completed tier as reference
    let refN = 0;
    let refMs = 0;
    for (const [tN, tMs] of completedTierTimings) {
      refN = tN;
      refMs = tMs;
    }
    if (refN === 0) return 0;

    const costRatio = nttCostFactor(N) / nttCostFactor(refN);
    return refMs * costRatio;
  }

  /**
   * Compute estimated remaining time for all future tiers.
   * Uses either measured msPerGen (if provided) or completed tier timing + scaling.
   */
  private computeRemainingTiersEta(
    futureTiers: readonly number[],
    completedTierTimings: Map<number, number>,
    currentN: number,
    currentTierElapsed?: number,
    currentMsPerGen?: number,
  ): number {
    if (futureTiers.length === 0) return 0;

    // Determine the reference ms-per-gen for the current tier
    let refMsPerTier: number;
    if (currentMsPerGen !== undefined) {
      refMsPerTier = currentMsPerGen * this.config.generationsPerTier;
    } else if (currentTierElapsed !== undefined) {
      refMsPerTier = currentTierElapsed;
    } else if (completedTierTimings.has(currentN)) {
      refMsPerTier = completedTierTimings.get(currentN)!;
    } else {
      // Fall back to last completed tier
      let lastN = 0;
      let lastMs = 0;
      for (const [tN, tMs] of completedTierTimings) {
        lastN = tN;
        lastMs = tMs;
      }
      if (lastN === 0) return 0;
      refMsPerTier = lastMs * (nttCostFactor(currentN) / nttCostFactor(lastN));
    }

    const currentCost = nttCostFactor(currentN);
    let totalMs = 0;
    for (const futureN of futureTiers) {
      const costRatio = nttCostFactor(futureN) / currentCost;
      totalMs += refMsPerTier * costRatio;
    }
    return totalMs;
  }

  private packSecureGenomes(secureGenomes: SecureGenome[]): Uint32Array {
    const buffer = new Uint32Array(secureGenomes.length * GENOME_FIELD_COUNT);
    for (let i = 0; i < secureGenomes.length; i++) {
      secureGenomes[i].genome.packInto(buffer, i * GENOME_FIELD_COUNT);
    }
    return buffer;
  }

  private buildEvaluated(secureGenomes: SecureGenome[], scores: Float64Array): EvaluatedGenome[] {
    const evaluated: EvaluatedGenome[] = [];

    for (let i = 0; i < secureGenomes.length; i++) {
      const latencyMs = scores[i];
      if (latencyMs === NOISE_FAILURE_SCORE) continue;

      const fitness: MultiObjectiveFitness = {
        latencyMs,
        securityBits: secureGenomes[i].securityBits,
        noiseBudgetBits: secureGenomes[i].noiseBudgetBits,
        rank: 0,
        crowdingDistance: 0,
      };

      evaluated.push({ genome: secureGenomes[i].genome, fitness });
    }

    return evaluated;
  }

  private breed(selected: EvaluatedGenome[], pm: PopulationManager, N: number): Genome[] {
    const nextGen: Genome[] = [];

    // Keep elites, locked to this tier's N.
    // With the population correctly locked upstream this is a no-op, but an
    // elite is copied forward unchanged, so if a foreign-N genome ever reaches
    // selection it would survive every subsequent generation untouched.
    for (const elite of selected) {
      const g = elite.genome;
      nextGen.push(
        g.N === N ? g : new Genome({ N, q: g.q, sigma: g.sigma, Bg: g.Bg }),
      );
    }

    // Fill with crossover + mutation
    while (nextGen.length < this.config.populationSize) {
      const parentA = selected[Math.floor(Math.random() * selected.length)];
      const parentB = selected[Math.floor(Math.random() * selected.length)];
      const child = pm['crossover'](parentA.genome, parentB.genome);
      const mutated = pm['mutate'](child);
      // Force N to stay in this tier
      if (mutated.N !== N) {
        nextGen.push(new Genome({ N, q: mutated.q, sigma: mutated.sigma, Bg: mutated.Bg }));
      } else {
        nextGen.push(mutated);
      }
    }

    return nextGen;
  }

  private deduplicatePoints(points: ParetoPoint[]): ParetoPoint[] {
    const map = new Map<string, ParetoPoint>();

    for (const p of points) {
      const key = `${p.N}:${p.q}:${p.sigma}:${p.Bg}`;
      const existing = map.get(key);
      if (!existing || p.latencyMs < existing.latencyMs) {
        map.set(key, p);
      }
    }

    return [...map.values()];
  }
}
