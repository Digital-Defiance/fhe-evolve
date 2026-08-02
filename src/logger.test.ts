import { describe, it, expect } from 'vitest';
import { Logger } from './logger.js';
import { Genome } from './genome.js';

describe('Logger', () => {
  it('records generation stats in history', () => {
    const logger = new Logger({ silent: true });
    const genome = new Genome({ N: 2048, q: 268435456, sigma: 3200, Bg: 1024 });

    logger.logGeneration({
      generation: 0,
      populationSize: 100,
      secureCount: 80,
      bestLatencyMs: 1.5,
      avgLatencyMs: 3.2,
      survivalRate: 0.8,
      fastestGenome: genome,
      timestamp: Date.now(),
    });

    const history = logger.getHistory();
    expect(history.length).toBe(1);
    expect(history[0].generation).toBe(0);
    expect(history[0].bestLatencyMs).toBe(1.5);
  });

  it('maintains leaderboard sorted by latency ascending', () => {
    const logger = new Logger({ silent: true, leaderboardSize: 3 });

    const genomes = [
      new Genome({ N: 4096, q: 536870912, sigma: 3200, Bg: 2048 }),
      new Genome({ N: 2048, q: 268435456, sigma: 3200, Bg: 1024 }),
      new Genome({ N: 1024, q: 134217728, sigma: 3200, Bg: 512 }),
    ];

    // Log in non-sorted order
    logger.logGeneration({
      generation: 0, populationSize: 100, secureCount: 80,
      bestLatencyMs: 5.0, avgLatencyMs: 8.0, survivalRate: 0.8,
      fastestGenome: genomes[0], timestamp: Date.now(),
    });
    logger.logGeneration({
      generation: 1, populationSize: 100, secureCount: 85,
      bestLatencyMs: 2.0, avgLatencyMs: 4.0, survivalRate: 0.85,
      fastestGenome: genomes[1], timestamp: Date.now(),
    });
    logger.logGeneration({
      generation: 2, populationSize: 100, secureCount: 90,
      bestLatencyMs: 8.0, avgLatencyMs: 12.0, survivalRate: 0.9,
      fastestGenome: genomes[2], timestamp: Date.now(),
    });

    const leaderboard = logger.getLeaderboard();
    expect(leaderboard.length).toBe(3);
    expect(leaderboard[0].latencyMs).toBe(2.0);
    expect(leaderboard[1].latencyMs).toBe(5.0);
    expect(leaderboard[2].latencyMs).toBe(8.0);
  });

  it('respects leaderboard size limit', () => {
    const logger = new Logger({ silent: true, leaderboardSize: 2 });
    const genome = new Genome({ N: 2048, q: 268435456, sigma: 3200, Bg: 1024 });

    for (let i = 0; i < 5; i++) {
      logger.logGeneration({
        generation: i, populationSize: 50, secureCount: 40,
        bestLatencyMs: 10 - i, avgLatencyMs: 15 - i, survivalRate: 0.8,
        fastestGenome: genome, timestamp: Date.now(),
      });
    }

    const leaderboard = logger.getLeaderboard();
    expect(leaderboard.length).toBe(2);
    // Best two latencies: 6.0 and 7.0 (from generations 4 and 3)
    expect(leaderboard[0].latencyMs).toBe(6);
    expect(leaderboard[1].latencyMs).toBe(7);
  });
});
