/**
 * Structured logger for the FHE parameter evolution simulation.
 *
 * Tracks per-generation stats and maintains an all-time leaderboard
 * of the fastest configurations discovered.
 */

import type { Genome } from './genome.js';

/** Stats collected for a single generation. */
export interface GenerationStats {
  generation: number;
  populationSize: number;
  secureCount: number;
  bestLatencyMs: number;
  avgLatencyMs: number;
  survivalRate: number;
  fastestGenome: Genome;
  timestamp: number;
}

/** An entry on the all-time leaderboard. */
export interface LeaderboardEntry {
  genome: Genome;
  latencyMs: number;
  generation: number;
  timestamp: number;
}

/** Configuration for the logger. */
export interface LoggerConfig {
  /** Number of top entries to keep on the leaderboard. */
  leaderboardSize: number;
  /** If true, suppress console output (useful for testing). */
  silent: boolean;
}

const DEFAULT_LOGGER_CONFIG: LoggerConfig = {
  leaderboardSize: 10,
  silent: false,
};

export class Logger {
  private readonly config: LoggerConfig;
  private readonly leaderboard: LeaderboardEntry[] = [];
  private readonly history: GenerationStats[] = [];

  constructor(config?: Partial<LoggerConfig>) {
    this.config = { ...DEFAULT_LOGGER_CONFIG, ...config };
  }

  /**
   * Log stats for a completed generation.
   */
  logGeneration(stats: GenerationStats): void {
    this.history.push(stats);
    this.updateLeaderboard(stats.fastestGenome, stats.bestLatencyMs, stats.generation);

    if (!this.config.silent) {
      const ts = new Date(stats.timestamp).toISOString();
      console.log(
        `[${ts}] Gen ${stats.generation} | pop: ${stats.populationSize} | secure: ${stats.secureCount} ` +
        `| survival: ${(stats.survivalRate * 100).toFixed(1)}% ` +
        `| best: ${stats.bestLatencyMs.toFixed(3)}ms | avg: ${stats.avgLatencyMs.toFixed(3)}ms`,
      );
      const g = stats.fastestGenome;
      console.log(
        `  └─ fastest: N=${g.N} q=${g.q} σ=${g.sigma} Bg=${g.Bg}`,
      );
    }
  }

  /**
   * Print the all-time leaderboard.
   */
  logLeaderboard(): void {
    if (this.config.silent) return;

    console.log('\n═══ All-Time Fastest Configurations ═══');
    for (let i = 0; i < this.leaderboard.length; i++) {
      const entry = this.leaderboard[i];
      const g = entry.genome;
      console.log(
        `  #${i + 1} | ${entry.latencyMs.toFixed(3)}ms (gen ${entry.generation}) ` +
        `| N=${g.N} q=${g.q} σ=${g.sigma} Bg=${g.Bg}`,
      );
    }
    console.log('');
  }

  /**
   * Get the current leaderboard (sorted by latency ascending — lower is better).
   */
  getLeaderboard(): readonly LeaderboardEntry[] {
    return this.leaderboard;
  }

  /**
   * Get the full generation history.
   */
  getHistory(): readonly GenerationStats[] {
    return this.history;
  }

  /**
   * Update the leaderboard with a new candidate.
   * Only inserts if it beats an existing entry or there's room.
   */
  private updateLeaderboard(genome: Genome, latencyMs: number, generation: number): void {
    const entry: LeaderboardEntry = {
      genome,
      latencyMs,
      generation,
      timestamp: Date.now(),
    };

    // Insert in sorted order (ascending latency — lower is better)
    let inserted = false;
    for (let i = 0; i < this.leaderboard.length; i++) {
      if (latencyMs < this.leaderboard[i].latencyMs) {
        this.leaderboard.splice(i, 0, entry);
        inserted = true;
        break;
      }
    }

    if (!inserted && this.leaderboard.length < this.config.leaderboardSize) {
      this.leaderboard.push(entry);
    }

    // Trim to max size
    if (this.leaderboard.length > this.config.leaderboardSize) {
      this.leaderboard.length = this.config.leaderboardSize;
    }
  }
}
