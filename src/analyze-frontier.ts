/**
 * Pareto Frontier Degeneracy Analysis
 *
 * Reads results/pareto-frontier.json and reports, per N tier and overall:
 *   1. Genome count (raw frontier points)
 *   2. Distinct objective vectors — (latencyMs, securityBits, noiseBudgetBits)
 *      rounded to (4, 1, 1) decimals
 *   3. Distinct (N, q, sigma) triples (Bg ignored)
 *   4. Distinct q values
 *   5. min/max log2(q)
 *   6. min/max sigma (unscaled) and the count of points with sigma < 3.2
 *   7. Count of points satisfying (q - 1) % (2N) == 0 (valid for the
 *      negacyclic ring Z_q[X]/(X^N + 1)) versus those that do not
 *
 * Run: npm run analyze
 */

import { readFileSync, existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import type { ParetoPoint } from './types.js';

// ─── Constants ──────────────────────────────────────────────────────────────

/** HE Standard 2018 / §2.1 minimum noise standard deviation. */
const SIGMA_FLOOR = 3.2;

/** Sigma is stored as an integer scaled by 1000 for FFI packing. */
const SIGMA_SCALE = 1000;

// ─── Types ──────────────────────────────────────────────────────────────────

interface TierStats {
  /** Tier label: the polynomial degree, or 'ALL' for the merged frontier. */
  label: string;
  genomeCount: number;
  distinctObjectiveVectors: number;
  distinctNQSigma: number;
  distinctQ: number;
  minLog2Q: number;
  maxLog2Q: number;
  minSigma: number;
  maxSigma: number;
  sigmaBelowFloor: number;
  negacyclicValid: number;
  negacyclicInvalid: number;
  /** Sorted distinct q values present in the tier. */
  qValues: number[];
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function getProjectRoot(): string {
  return resolve(import.meta.dirname ?? process.cwd(), '..');
}

function loadFrontier(projectRoot: string): ParetoPoint[] | null {
  const filePath = join(projectRoot, 'results', 'pareto-frontier.json');
  if (!existsSync(filePath)) return null;
  return JSON.parse(readFileSync(filePath, 'utf-8')) as ParetoPoint[];
}

/**
 * Objective-space key with the rounding policy defined above. Two genomes
 * sharing this key are indistinguishable to NSGA-II dominance comparison.
 */
function objectiveKey(p: ParetoPoint): string {
  const lat = p.latencyMs.toFixed(4);
  const sec = p.securityBits.toFixed(1);
  const noise = p.noiseBudgetBits.toFixed(1);
  return `${lat}|${sec}|${noise}`;
}

function computeStats(label: string, points: ParetoPoint[]): TierStats {
  const objectives = new Set<string>();
  const nqSigma = new Set<string>();
  const qs = new Set<number>();

  let minLog2Q = Infinity;
  let maxLog2Q = -Infinity;
  let minSigma = Infinity;
  let maxSigma = -Infinity;
  let sigmaBelowFloor = 0;
  let negacyclicValid = 0;

  for (const p of points) {
    objectives.add(objectiveKey(p));
    nqSigma.add(`${p.N}|${p.q}|${p.sigma}`);
    qs.add(p.q);

    const log2q = Math.log2(p.q);
    if (log2q < minLog2Q) minLog2Q = log2q;
    if (log2q > maxLog2Q) maxLog2Q = log2q;

    const sigma = p.sigma / SIGMA_SCALE;
    if (sigma < minSigma) minSigma = sigma;
    if (sigma > maxSigma) maxSigma = sigma;
    if (sigma < SIGMA_FLOOR) sigmaBelowFloor++;

    if ((p.q - 1) % (2 * p.N) === 0) negacyclicValid++;
  }

  return {
    label,
    genomeCount: points.length,
    distinctObjectiveVectors: objectives.size,
    distinctNQSigma: nqSigma.size,
    distinctQ: qs.size,
    minLog2Q,
    maxLog2Q,
    minSigma,
    maxSigma,
    sigmaBelowFloor,
    negacyclicValid,
    negacyclicInvalid: points.length - negacyclicValid,
    qValues: [...qs].sort((a, b) => a - b),
  };
}

function padRight(s: string, len: number): string {
  return s.length >= len ? s : s + ' '.repeat(len - s.length);
}

function padLeft(s: string, len: number): string {
  return s.length >= len ? s : ' '.repeat(len - s.length) + s;
}

function renderTable(headers: string[], rows: string[][]): string {
  const widths = headers.map((h, i) =>
    Math.max(h.length, ...rows.map((r) => r[i].length)),
  );
  const lines: string[] = [];
  lines.push(headers.map((h, i) => padLeft(h, widths[i])).join('  '));
  lines.push(widths.map((w) => '─'.repeat(w)).join('──'));
  for (const row of rows) {
    lines.push(row.map((c, i) => padLeft(c, widths[i])).join('  '));
  }
  return lines.join('\n');
}

function statsRow(s: TierStats): string[] {
  const pct = s.genomeCount > 0
    ? ((s.distinctObjectiveVectors / s.genomeCount) * 100).toFixed(1)
    : '0.0';
  return [
    s.label,
    `${s.genomeCount}`,
    `${s.distinctObjectiveVectors}`,
    `${pct}%`,
    `${s.distinctNQSigma}`,
    `${s.distinctQ}`,
    s.minLog2Q.toFixed(2),
    s.maxLog2Q.toFixed(2),
    s.minSigma.toFixed(3),
    s.maxSigma.toFixed(3),
    `${s.sigmaBelowFloor}`,
    `${s.negacyclicValid}`,
    `${s.negacyclicInvalid}`,
  ];
}

// ─── Main ───────────────────────────────────────────────────────────────────

function main(): void {
  const projectRoot = getProjectRoot();
  const points = loadFrontier(projectRoot);

  if (!points) {
    console.error('No frontier found at ./results/pareto-frontier.json');
    console.error('Run the evolution first: npm run start');
    process.exit(1);
  }

  const tiers = [...new Set(points.map((p) => p.N))].sort((a, b) => a - b);
  const perTier = tiers.map((N) =>
    computeStats(`${N}`, points.filter((p) => p.N === N)),
  );
  const overall = computeStats('ALL', points);

  console.log('═══════════════════════════════════════════════════════════════');
  console.log('  Pareto Frontier Degeneracy Analysis');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');

  const headers = [
    'N', 'genomes', 'distObj', 'distObj%', 'dist(N,q,σ)', 'dist q',
    'min log2q', 'max log2q', 'min σ', 'max σ', 'σ<3.2', 'q≡1(2N)', 'q≢1(2N)',
  ];
  const rows = [...perTier.map(statsRow), statsRow(overall)];
  console.log(renderTable(headers, rows));
  console.log('');

  console.log('─── Distinct moduli per tier ───────────────────────────────────');
  for (const s of perTier) {
    const N = Number(s.label);
    const annotated = s.qValues.map((q) => {
      const k = (q - 1) / N;
      const ok = (q - 1) % (2 * N) === 0 ? '✓' : '✗';
      return `${q} (k=${k}) ${ok}`;
    });
    console.log(`  N = ${padRight(s.label, 5)} ${s.qValues.length} distinct q`);
    for (const a of annotated) console.log(`      ${a}`);
  }
  console.log('');
  console.log('  ✓ = q ≡ 1 (mod 2N), valid for the negacyclic ring Z_q[X]/(X^N+1)');
  console.log('  ✗ = q ≡ 1 (mod N) only, cyclic transform over X^N − 1');
  console.log('');

  console.log('─── Objective-space inflation ──────────────────────────────────');
  for (const s of [...perTier, overall]) {
    const ratio = s.distinctObjectiveVectors > 0
      ? (s.genomeCount / s.distinctObjectiveVectors).toFixed(2)
      : 'n/a';
    console.log(
      `  ${padRight(s.label, 6)} ${padLeft(`${s.genomeCount}`, 5)} genomes → ` +
      `${padLeft(`${s.distinctObjectiveVectors}`, 5)} distinct objective vectors (${ratio}× inflation)`,
    );
  }
  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
}

main();
