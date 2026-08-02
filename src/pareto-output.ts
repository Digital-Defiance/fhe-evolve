/**
 * Output writer for Pareto exploration results.
 *
 * Writes JSON and human-readable summaries to the results directory,
 * including comparison against HE Standard 2018 reference values.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { ParetoPoint } from './types.js';
import type { ConvergenceEntry, ParetoExplorerResult } from './pareto-explorer.js';

/** HE Standard 2018 reference table for 128-bit security. */
export const HE_STANDARD_2018 = [
  { N: 1024, maxLogQ: 27, securityBits: 128 },
  { N: 2048, maxLogQ: 54, securityBits: 128 },
  { N: 4096, maxLogQ: 109, securityBits: 128 },
  { N: 8192, maxLogQ: 218, securityBits: 128 },
  { N: 16384, maxLogQ: 438, securityBits: 128 },
] as const;

/**
 * Write all Pareto exploration results to the output directory.
 */
export function writeParetoResults(result: ParetoExplorerResult, outputDir: string): void {
  // Ensure the output directory exists
  mkdirSync(outputDir, { recursive: true });

  // 1. Write merged Pareto frontier
  writeFileSync(
    join(outputDir, 'pareto-frontier.json'),
    JSON.stringify(result.paretoFrontier, null, 2),
  );

  // 2. Write per-tier Pareto fronts
  for (const [N, front] of result.perTierFronts) {
    writeFileSync(
      join(outputDir, `pareto-N${N}.json`),
      JSON.stringify(front, null, 2),
    );
  }

  // 3. Write convergence data
  writeFileSync(
    join(outputDir, 'convergence.json'),
    JSON.stringify(result.convergence, null, 2),
  );

  // 4. Write human-readable summary
  const summary = generateSummary(result);
  writeFileSync(join(outputDir, 'summary.txt'), summary);
}

/**
 * Generate a human-readable summary comparing results against HE Standard 2018.
 */
export function generateSummary(result: ParetoExplorerResult): string {
  const lines: string[] = [];

  lines.push('═══════════════════════════════════════════════════════════════');
  lines.push('  FHE Parameter Evolution — Pareto Frontier Summary');
  lines.push('═══════════════════════════════════════════════════════════════');
  lines.push('');
  lines.push(`Total wall-clock time: ${(result.wallClockMs / 1000).toFixed(1)}s`);
  lines.push(`Interrupted: ${result.wasInterrupted ? 'yes' : 'no'}`);
  lines.push(`Total Pareto-optimal points: ${result.paretoFrontier.length}`);
  lines.push('');

  // Per-tier summary
  lines.push('─── Per-Tier Results ───────────────────────────────────────────');
  lines.push('');

  for (const ref of HE_STANDARD_2018) {
    const tierFront = result.perTierFronts.get(ref.N) ?? [];
    lines.push(`N = ${ref.N} (HE Standard 2018: max log₂(q) = ${ref.maxLogQ}, ${ref.securityBits}-bit security)`);
    lines.push(`  Pareto front size: ${tierFront.length}`);

    if (tierFront.length > 0) {
      const bestLatency = tierFront.reduce((a, b) => a.latencyMs < b.latencyMs ? a : b);
      const bestSecurity = tierFront.reduce((a, b) => a.securityBits > b.securityBits ? a : b);
      const bestNoise = tierFront.reduce((a, b) => a.noiseBudgetBits > b.noiseBudgetBits ? a : b);

      lines.push(`  Best latency:       ${bestLatency.latencyMs.toFixed(3)}ms (q=${bestLatency.q}, σ=${bestLatency.sigma/1000}, security=${bestLatency.securityBits.toFixed(1)} bits)`);
      lines.push(`  Best security:      ${bestSecurity.securityBits.toFixed(1)} bits (q=${bestSecurity.q}, latency=${bestSecurity.latencyMs.toFixed(3)}ms)`);
      lines.push(`  Best noise budget:  ${bestNoise.noiseBudgetBits.toFixed(1)} bits (q=${bestNoise.q})`);

      // Compare discovered max log₂(q) against standard
      const maxLogQ = Math.max(...tierFront.map((p) => Math.log2(p.q)));
      lines.push(`  Max log₂(q) found: ${maxLogQ.toFixed(1)} (standard limit: ${ref.maxLogQ})`);
    } else {
      lines.push('  No valid points found.');
    }

    lines.push('');
  }

  // Convergence summary
  lines.push('─── Convergence ────────────────────────────────────────────────');
  lines.push('');

  const tiers = [...new Set(result.convergence.map((c) => c.tier))];
  for (const tier of tiers) {
    const entries = result.convergence.filter((c) => c.tier === tier);
    if (entries.length > 0) {
      const last = entries[entries.length - 1];
      lines.push(`N=${tier}: ${last.generation + 1} generations, final front size: ${last.paretoFrontSize}`);
    }
  }

  lines.push('');
  lines.push('═══════════════════════════════════════════════════════════════');

  return lines.join('\n');
}
