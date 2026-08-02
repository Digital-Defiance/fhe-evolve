/**
 * Main entry point for the FHE Parameter Evolution Simulation.
 *
 * Supports two modes:
 *   - Default: Multi-objective Pareto frontier explorer (NSGA-II)
 *   - --quick: Legacy BurnLoop for rapid single-objective testing
 *
 * Usage:
 *   npm run start          — full Pareto exploration
 *   npm run start:quick    — quick BurnLoop mode
 */

import { loadNativeModule } from './ffi.js';
import { BurnLoop, type BurnLoopConfig } from './burn-loop.js';
import { ParetoExplorer, type ParetoExplorerResult } from './pareto-explorer.js';
import { writeParetoResults } from './pareto-output.js';
import type { ParetoExplorerConfig } from './types.js';

/** Quick mode configuration (legacy BurnLoop). */
const QUICK_CONFIG: Partial<BurnLoopConfig> = {
  maxGenerations: 500,
  leaderboardSize: 10,
  silent: false,
  population: {
    populationSize: 100,
    eliteRatio: 0.05,
    mutationRate: 0.8,
    mutationStrength: 0.2,
  },
};

/** Default Pareto explorer configuration. */
const PARETO_CONFIG: Partial<ParetoExplorerConfig> = {
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

function runQuickMode(): void {
  console.log('═══ FHE Parameter Evolution — Quick Mode ═══');
  console.log('Loading native Rust module...');

  const nativeModule = loadNativeModule();
  console.log('Native module loaded.\n');

  const burnLoop = new BurnLoop(nativeModule, QUICK_CONFIG);

  const handleShutdown = (signal: string) => {
    console.log(`\n[${signal}] Graceful shutdown requested.`);
    burnLoop.shutdown();
  };

  process.on('SIGINT', () => handleShutdown('SIGINT'));
  process.on('SIGTERM', () => handleShutdown('SIGTERM'));

  console.log(`Starting evolution: ${QUICK_CONFIG.maxGenerations} generations, ` +
    `population size ${QUICK_CONFIG.population?.populationSize ?? 100}\n`);

  const result = burnLoop.run();

  console.log(`\n═══ Simulation Complete ═══`);
  console.log(`Generations completed: ${result.generations}`);
  console.log(`Shutdown requested: ${result.wasShutdown}`);

  const leaderboard = result.logger.getLeaderboard();
  if (leaderboard.length > 0) {
    const best = leaderboard[0];
    console.log(`\nAll-time fastest: ${best.latencyMs.toFixed(3)}ms`);
    console.log(`  N=${best.genome.N} q=${best.genome.q} σ=${best.genome.sigma} Bg=${best.genome.Bg}`);
  }
}

async function runParetoMode(): Promise<void> {
  const os = await import('node:os');
  console.log('═══════════════════════════════════════════════════════════');
  console.log('  FHE Parameter Evolution Simulation');
  console.log('  Multi-Objective Pareto Frontier Explorer');
  console.log('═══════════════════════════════════════════════════════════');
  console.log(`  Platform:        ${process.platform} ${process.arch}`);
  console.log(`  Node.js:         ${process.version}`);
  console.log(`  CPUs:            ${os.cpus().length}`);
  console.log(`  Generations/Tier: ${PARETO_CONFIG.generationsPerTier}`);
  console.log(`  Population:      ${PARETO_CONFIG.populationSize}`);
  console.log(`  NTT Iterations:  ${PARETO_CONFIG.nttIterations}`);
  console.log(`  Output:          ${PARETO_CONFIG.outputDir}`);
  console.log('═══════════════════════════════════════════════════════════\n');

  console.log('Loading native Rust module...');
  const nativeModule = loadNativeModule();
  console.log('Native module loaded.\n');

  const explorer = new ParetoExplorer(nativeModule, PARETO_CONFIG);

  // Graceful shutdown: write partial results
  let result: ParetoExplorerResult | null = null;

  const handleShutdown = (signal: string) => {
    console.log(`\n[${signal}] Graceful shutdown requested. Finishing current generation...`);
    explorer.interrupt();
  };

  process.on('SIGINT', () => handleShutdown('SIGINT'));
  process.on('SIGTERM', () => handleShutdown('SIGTERM'));

  const startTime = Date.now();

  try {
    result = explorer.run();
  } catch (err) {
    console.error('Exploration failed:', err);
    process.exit(1);
  }

  // Write output files
  const outputDir = PARETO_CONFIG.outputDir ?? './results';
  console.log(`\nWriting results to ${outputDir}/...`);
  writeParetoResults(result, outputDir);

  // Final summary
  const elapsed = (Date.now() - startTime) / 1000;
  console.log(`\n═══ Exploration Complete ═══`);
  console.log(`Total time: ${elapsed.toFixed(1)}s`);
  console.log(`Pareto-optimal points: ${result.paretoFrontier.length}`);
  console.log(`Interrupted: ${result.wasInterrupted ? 'yes' : 'no'}`);

  if (result.paretoFrontier.length > 0) {
    console.log('\nTop 5 fastest Pareto-optimal configurations:');
    const sorted = [...result.paretoFrontier].sort((a, b) => a.latencyMs - b.latencyMs);
    for (let i = 0; i < Math.min(5, sorted.length); i++) {
      const p = sorted[i];
      console.log(`  #${i + 1} | ${p.latencyMs.toFixed(3)}ms | N=${p.N} q=${p.q} σ=${p.sigma} Bg=${p.Bg} | ${p.securityBits.toFixed(1)} bits`);
    }
  }
}

function main(): void {
  const args = process.argv.slice(2);
  const isQuick = args.includes('--quick');

  if (isQuick) {
    runQuickMode();
  } else {
    runParetoMode().catch((err) => {
      console.error(err);
      process.exit(1);
    });
  }
}

main();
