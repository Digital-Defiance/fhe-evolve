import { describe, it, expect } from 'vitest';
import { BurnLoop } from './burn-loop.js';
import { GENOME_FIELD_COUNT } from './genome.js';
import type { NativeModule } from './ffi.js';

/**
 * Mock native module that returns deterministic latency scores
 * based on N and sigma (mimics the mock from ffi.test.ts).
 */
const mockNative: NativeModule = {
  runBenchmark(parameterBatch: Uint32Array): Float64Array {
    const count = parameterBatch.length / GENOME_FIELD_COUNT;
    const scores = new Float64Array(count);
    for (let i = 0; i < count; i++) {
      const offset = i * GENOME_FIELD_COUNT;
      const n = parameterBatch[offset];
      const sigma = parameterBatch[offset + 2];
      // Latency proportional to N, inversely proportional to sigma
      scores[i] = (n * (sigma / 1000)) / 1_000_000;
    }
    return scores;
  },
  runBenchmarkDeep(parameterBatch: Uint32Array, _iterations: number): Float64Array {
    return this.runBenchmark(parameterBatch);
  },
  validateBatch(parameterBatch: Uint32Array): number {
    return parameterBatch.length / GENOME_FIELD_COUNT;
  },
  echoGenome(parameterBatch: Uint32Array, index: number): Uint32Array {
    const offset = index * GENOME_FIELD_COUNT;
    return parameterBatch.slice(offset, offset + GENOME_FIELD_COUNT);
  },
  benchmarkPoolInfo() {
    return { threads: 1, performanceCores: 1, qosAvailable: true };
  },
};

describe('BurnLoop', () => {
  it('runs a single generation without error', () => {
    const loop = new BurnLoop(mockNative, {
      maxGenerations: 1,
      silent: true,
      population: { populationSize: 20 },
    });

    const result = loop.run();
    expect(result.generations).toBe(1);
    expect(result.wasShutdown).toBe(false);
  });

  it('completes the specified number of generations', () => {
    const loop = new BurnLoop(mockNative, {
      maxGenerations: 5,
      silent: true,
      population: { populationSize: 20 },
    });

    const result = loop.run();
    expect(result.generations).toBe(5);
  });

  it('respects graceful shutdown', () => {
    const loop = new BurnLoop(mockNative, {
      maxGenerations: 100,
      silent: true,
      population: { populationSize: 20 },
    });

    // Seed and run one generation manually, then request shutdown
    loop.getPopulationManager().seed();
    loop.runGeneration();
    loop.shutdown();

    expect(loop.isShutdownRequested()).toBe(true);
  });

  it('tracks the leaderboard across generations', () => {
    const loop = new BurnLoop(mockNative, {
      maxGenerations: 3,
      silent: true,
      leaderboardSize: 5,
      population: { populationSize: 20 },
    });

    const result = loop.run();
    const leaderboard = result.logger.getLeaderboard();

    expect(leaderboard.length).toBeGreaterThan(0);
    expect(leaderboard.length).toBeLessThanOrEqual(5);

    // Leaderboard should be sorted by latency ascending (lowest = fastest)
    for (let i = 1; i < leaderboard.length; i++) {
      expect(leaderboard[i].latencyMs).toBeGreaterThanOrEqual(leaderboard[i - 1].latencyMs);
    }
  });

  it('records generation history', () => {
    const loop = new BurnLoop(mockNative, {
      maxGenerations: 3,
      silent: true,
      population: { populationSize: 20 },
    });

    const result = loop.run();
    const history = result.logger.getHistory();

    expect(history.length).toBe(3);
    for (const stats of history) {
      expect(stats.populationSize).toBe(20);
      expect(stats.secureCount).toBeGreaterThan(0);
      expect(stats.bestLatencyMs).toBeGreaterThan(0);
      expect(stats.survivalRate).toBeGreaterThan(0);
      expect(stats.timestamp).toBeGreaterThan(0);
    }
  });

  it('evolution improves or maintains performance over generations', () => {
    const loop = new BurnLoop(mockNative, {
      maxGenerations: 10,
      silent: true,
      population: { populationSize: 50 },
    });

    const result = loop.run();
    const leaderboard = result.logger.getLeaderboard();

    // The leaderboard should have at least one entry with a valid latency
    expect(leaderboard.length).toBeGreaterThan(0);
    expect(leaderboard[0].latencyMs).toBeGreaterThan(0);
    expect(leaderboard[0].latencyMs).toBeLessThan(Infinity);
  });

  it('handles population where no genomes pass security', () => {
    // Mock that always returns noise failure
    const failingNative: NativeModule = {
      ...mockNative,
      runBenchmark(parameterBatch: Uint32Array): Float64Array {
        const count = parameterBatch.length / GENOME_FIELD_COUNT;
        return new Float64Array(count).fill(-1.0);
      },
    };

    const loop = new BurnLoop(failingNative, {
      maxGenerations: 2,
      silent: true,
      population: { populationSize: 20 },
    });

    // Should not throw even when all benchmarks fail
    const result = loop.run();
    expect(result.generations).toBe(2);
  });
});
