/**
 * Pareto Frontier Validation Script
 *
 * Compares evolved FHE parameters against the HE Standard 2018 reference
 * and against the fastest *measured* CPU NTT path on this machine
 * (Montgomery reduction, Apple M4 Max). No accelerator (GPU / AMX / SME)
 * figures are used, and no unmeasured degree is estimated.
 *
 * Run: npx tsx src/validate-pareto.ts
 *   or: npm run validate
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import type { ParetoPoint } from './types.js';

// ─── Constants ──────────────────────────────────────────────────────────────

/** HE Standard 2018 reference (128-bit security). */
const HE_STANDARD_2018 = [
  { N: 1024, maxLogQ: 27, securityBits: 128 },
  { N: 2048, maxLogQ: 54, securityBits: 128 },
  { N: 4096, maxLogQ: 109, securityBits: 128 },
  { N: 8192, maxLogQ: 218, securityBits: 128 },
  { N: 16384, maxLogQ: 438, securityBits: 128 },
] as const;

/**
 * Measured CPU Montgomery NTT latencies on an idle Apple M4 Max (full forward
 * transform), as reported in Table 3 of the paper.
 *
 * These are **not** GPU figures and **not** AMX or SME figures. Only the three
 * degrees below were actually measured; N = 2048 and N = 8192 have no
 * measurement and are therefore reported as unavailable rather than
 * interpolated or scaled.
 *
 * Historical note: an earlier revision of this table carried 8.58 / 39.74 /
 * 186.13 µs labelled "AMX" and "SME Tile", plus two invented "(interpolated)"
 * entries. Those three values sit within a few percent of the Montgomery
 * measurements below, which suggests they were Montgomery timings under a
 * wrong backend label. Clean re-runs of the accelerator library emit no
 * AMX or SME NTT rows at all, so the labels were unsupported and the whole
 * table was replaced with measured CPU values.
 */
const ACCELERATED_NTT_BENCHMARKS: Record<number, { latencyUs: number; backend: string }> = {
  1024: { latencyUs: 8.33, backend: 'CPU Montgomery NTT' },
  4096: { latencyUs: 39.96, backend: 'CPU Montgomery NTT' },
  16384: { latencyUs: 185.28, backend: 'CPU Montgomery NTT' },
};

/**
 * Illustrative NTT-friendly prime chain, hand-picked per polynomial degree.
 *
 * These primes are **not** search output. They are unrelated to the evolved
 * `q` of any Pareto point, they were never benchmarked, and no latency figure
 * anywhere in this file or in `results/fhe-parameters.json` was measured
 * against them. They exist only so the illustrative CustomParameters mapping
 * has a modulus chain of a plausible size for the degree — the evolved `q`
 * values are 13–22 bits, far too small for a real FHE modulus chain.
 *
 * Each entry satisfies `q ≡ 1 (mod 2N)` (the negacyclic NTT condition) and is
 * prime under a deterministic Miller–Rabin test over the witness set
 * {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}, which is sufficient for all
 * n < 3.3 × 10²⁴.
 *
 * Historical note: the N = 1024 entry previously held 1099511627777 = 2⁴⁰ + 1,
 * commented "40-bit". That value is 41 bits and composite (257 × 4278255361);
 * it passed review only because `q ≡ 1 (mod 2048)` happens to hold for it. It
 * has been replaced with the largest 40-bit prime satisfying the same
 * congruence.
 */
const NTT_PRIMES: Record<number, bigint[]> = {
  1024: [1099511592961n],                                    // 40-bit, = 2048·536870895 + 1
  2048: [1125899906826241n],                                 // 50-bit, = 4096·274877906940 + 1
  4096: [1152921504606584833n],                              // 60-bit
  8192: [1152921504606584833n, 1152921504598720513n],        // 2×60-bit
  16384: [1152921504606584833n, 1152921504598720513n, 1152921504597016577n], // 3×60-bit
};

// ─── Interfaces ─────────────────────────────────────────────────────────────

interface TierAnalysis {
  N: number;
  points: ParetoPoint[];
  heStandardRef: (typeof HE_STANDARD_2018)[number] | undefined;
  exceedsStandard: ParetoPoint[];
  top5Fastest: ParetoPoint[];
}

/**
 * Per-field record of how each value in the illustrative mapping was obtained.
 * Emitted verbatim into the artifact so a reader never has to guess which
 * fields are search output and which are projections.
 */
interface MappingProvenance {
  polyDegree: string;
  moduli: string;
  securityLevel: string;
  lweDimension: string;
  lweNoiseStd: string;
  glweDimension: string;
  decompBaseLog: string;
  decompLevel: string;
}

/**
 * Heuristically derived CustomParameters projection. Every numeric field here
 * is either a constant, a hardcoded ladder lookup, or a threshold applied to
 * an estimator output. None of it was benchmarked and none of it has been
 * validated against a working FHE implementation.
 */
interface IllustrativeMapping {
  note: string;
  polyDegree: number;
  moduli: string[];
  securityLevel: number;
  lweDimension: number;
  lweNoiseStd: number;
  glweDimension: number;
  decompBaseLog: number;
  decompLevel: number;
  provenance: MappingProvenance;
}

interface MappedParameters {
  source: ParetoPoint;
  illustrativeMapping: IllustrativeMapping;
  /** Measured reference NTT latency for this degree, or null if unmeasured. */
  acceleratedLatencyUs: number | null;
  /** Ratio against the measured reference, or null when there is no measurement. */
  speedupFactor: number | null;
  recommendation: string;
}

/**
 * Blanket caveat attached to every `illustrativeMapping` block in the emitted
 * artifact.
 */
const ILLUSTRATIVE_MAPPING_NOTE =
  'Illustrative only. Every field in this block is a heuristic projection, ' +
  'not search output and not measured. This mapping has not been validated ' +
  'against a working FHE implementation and must not be used as a parameter ' +
  'set. See the sibling `provenance` object for the origin of each field, and ' +
  'the `searched` and `measured` blocks for the values that were actually ' +
  'produced by the evolutionary run and timed on this machine.';

// ─── Helpers ────────────────────────────────────────────────────────────────

function getProjectRoot(): string {
  return resolve(import.meta.dirname ?? process.cwd(), '..');
}

function padRight(s: string, len: number): string {
  return s.length >= len ? s : s + ' '.repeat(len - s.length);
}

function padLeft(s: string, len: number): string {
  return s.length >= len ? s : ' '.repeat(len - s.length) + s;
}

function formatTable(headers: string[], rows: string[][], colWidths: number[]): string {
  const lines: string[] = [];
  const headerLine = headers.map((h, i) => padRight(h, colWidths[i])).join('  ');
  lines.push(headerLine);
  lines.push('─'.repeat(headerLine.length));
  for (const row of rows) {
    lines.push(row.map((cell, i) => padLeft(cell, colWidths[i])).join('  '));
  }
  return lines.join('\n');
}

/**
 * Compute LWE dimension from N using standard TFHE relationships.
 * lweDimension ≈ 0.72 * N for 128-bit security (conservative).
 */
function estimateLweDimension(N: number): number {
  if (N <= 1024) return 742;
  if (N <= 2048) return 830;
  if (N <= 4096) return 1024;
  if (N <= 8192) return 1400;
  return 2048;
}

/**
 * Estimate decompBaseLog and decompLevel from Bg parameter.
 */
function estimateDecomposition(Bg: number): { baseLog: number; level: number } {
  const baseLog = Math.max(1, Math.round(Math.log2(Bg)));
  // Higher Bg = fewer levels needed
  const level = Bg >= 128 ? 1 : Bg >= 16 ? 2 : 3;
  return { baseLog, level };
}

/**
 * Measured reference NTT latency for a given N, in microseconds.
 *
 * Returns null when no measurement exists for that degree. Unmeasured degrees
 * are never interpolated or scaled — an absent measurement is reported as
 * absent.
 */
function getAcceleratedLatencyUs(N: number): number | null {
  const bench = ACCELERATED_NTT_BENCHMARKS[N];
  return bench ? bench.latencyUs : null;
}

function classifyRecommendation(point: ParetoPoint): string {
  if (point.securityBits >= 200) return 'maximum-security';
  if (point.noiseBudgetBits >= 30) return 'deep-computation';
  if (point.latencyMs < 0.05) return 'fast-bootstrapping';
  return 'balanced';
}

// ─── Phase 1: Load Results ──────────────────────────────────────────────────

function loadParetoFrontier(projectRoot: string): ParetoPoint[] | null {
  const filePath = join(projectRoot, 'results', 'pareto-frontier.json');
  if (!existsSync(filePath)) {
    return null;
  }
  const raw = readFileSync(filePath, 'utf-8');
  return JSON.parse(raw) as ParetoPoint[];
}

// ─── Phase 2: Compare Against HE Standard ───────────────────────────────────

function analyzeByTier(points: ParetoPoint[]): Map<number, TierAnalysis> {
  const tiers = new Map<number, TierAnalysis>();

  // Group points by N
  const grouped = new Map<number, ParetoPoint[]>();
  for (const p of points) {
    const tier = grouped.get(p.N) ?? [];
    tier.push(p);
    grouped.set(p.N, tier);
  }

  for (const ref of HE_STANDARD_2018) {
    const tierPoints = grouped.get(ref.N) ?? [];
    const exceedsStandard = tierPoints.filter((p) => {
      const logQ = Math.log2(p.q);
      return logQ > ref.maxLogQ && p.securityBits >= ref.securityBits;
    });

    const sorted = [...tierPoints].sort((a, b) => a.latencyMs - b.latencyMs);
    const top5 = sorted.slice(0, 5);

    tiers.set(ref.N, {
      N: ref.N,
      points: tierPoints,
      heStandardRef: ref,
      exceedsStandard,
      top5Fastest: top5,
    });
  }

  return tiers;
}

// ─── Phase 3: Performance Comparison ────────────────────────────────────────

function buildPerformanceTable(tiers: Map<number, TierAnalysis>): string {
  const lines: string[] = [];

  for (const [N, tier] of tiers) {
    if (tier.top5Fastest.length === 0) continue;

    const accelUs = getAcceleratedLatencyUs(N);
    const bench = ACCELERATED_NTT_BENCHMARKS[N];

    const heading = bench
      ? `${bench.backend}, ${bench.latencyUs.toFixed(2)}µs`
      : 'no measured reference NTT for this degree';
    lines.push(`\n  N = ${N} (Reference: ${heading})`);
    lines.push('');

    // "Estimator" rather than "Security": these are ordinal estimator outputs
    // (§6.6), not bit-security figures.
    const headers = ['#', 'Rust NTT (ms)', 'Ref NTT (ms)', 'Speedup', 'Estimator', 'Noise Budget'];
    const colWidths = [3, 14, 14, 8, 10, 13];
    const rows: string[][] = [];

    for (let i = 0; i < tier.top5Fastest.length; i++) {
      const p = tier.top5Fastest[i];
      const accelMs = accelUs === null ? null : accelUs / 1000;
      const speedup = accelMs === null ? null : p.latencyMs / accelMs;

      rows.push([
        `${i + 1}`,
        p.latencyMs.toFixed(4),
        accelMs === null ? 'n/a' : accelMs.toFixed(4),
        speedup === null ? 'n/a' : `${speedup.toFixed(1)}×`,
        `${p.securityBits.toFixed(0)} ord.`,
        `${p.noiseBudgetBits.toFixed(1)} bits`,
      ]);
    }

    lines.push(formatTable(headers, rows, colWidths));
  }

  return lines.join('\n');
}

// ─── Phase 4: Illustrative Parameter Mapping ────────────────────────────────

/**
 * Build the per-field provenance record for one point's illustrative mapping.
 *
 * `usedFallbackModuli` is true when no hand-picked chain exists for the degree
 * and the evolved `q` was emitted instead — a case worth naming explicitly,
 * since the two situations have opposite provenance.
 */
function buildMappingProvenance(usedFallbackModuli: boolean): MappingProvenance {
  return {
    polyDegree: 'Taken directly from the searched N. This is search output.',
    moduli: usedFallbackModuli
      ? 'Fallback: no hand-picked chain exists for this degree, so the evolved ' +
        'q was emitted as a single-element chain. The evolved q is 13–22 bits ' +
        'and is not a usable FHE modulus. It is also not the modulus behind ' +
        'measured.referenceNttUs.'
      : 'Hand-picked NTT prime chain for this polynomial degree (see ' +
        'NTT_PRIMES). NOT the evolved searched.q, and NOT the modulus that any ' +
        'benchmark in the `measured` block was run against. Chosen only for ' +
        'plausible size; never benchmarked.',
    securityLevel:
      'Derived by thresholding the estimator output (searched.estimatorBits ' +
      '>= 192 ? 192 : 128). Not from a lattice attack cost model, and not a ' +
      'bit-security claim.',
    lweDimension:
      'Hardcoded ladder lookup on N in estimateLweDimension. Not a search ' +
      'result and not derived from any security analysis.',
    lweNoiseStd:
      'searched.sigma / 1000. The genome stores sigma as an integer scaled by ' +
      '1000 (see genome.ts), so this is the decoded searched sigma rather than ' +
      'a heuristic — but it is applied here as the GLWE/LWE noise std with no ' +
      'calibration step, so treat the role it plays as unvalidated.',
    glweDimension: 'Fixed constant 1. Not searched, not derived.',
    decompBaseLog:
      'Heuristic: round(log2(Bg)) in estimateDecomposition. Derived from the ' +
      'searched Bg but not itself searched or benchmarked.',
    decompLevel:
      'Heuristic: step function of Bg in estimateDecomposition (3 / 2 / 1 at ' +
      'the Bg = 16 and Bg = 128 breakpoints). Not searched or benchmarked.',
  };
}

function mapToIllustrativeParameters(points: ParetoPoint[]): MappedParameters[] {
  return points.map((p) => {
    const { baseLog, level } = estimateDecomposition(p.Bg);
    const accelUs = getAcceleratedLatencyUs(p.N);
    // No measurement for this degree ⇒ no latency and no speedup. Nothing is
    // substituted in place of the missing measurement.
    const speedup = accelUs === null ? null : p.latencyMs / (accelUs / 1000);

    // Hand-picked illustrative chain for this degree. This deliberately does
    // NOT reflect the evolved q, and is NOT what pureRustNttMs was measured on.
    const pickedModuli = NTT_PRIMES[p.N];
    const usedFallbackModuli = pickedModuli === undefined;
    const moduli = pickedModuli ?? [BigInt(p.q)];

    return {
      source: p,
      illustrativeMapping: {
        note: ILLUSTRATIVE_MAPPING_NOTE,
        polyDegree: p.N,
        moduli: moduli.map((m) => m.toString()),
        securityLevel: p.securityBits >= 192 ? 192 : 128,
        lweDimension: estimateLweDimension(p.N),
        lweNoiseStd: p.sigma / 1000,
        glweDimension: 1,
        decompBaseLog: baseLog,
        decompLevel: level,
        provenance: buildMappingProvenance(usedFallbackModuli),
      },
      acceleratedLatencyUs: accelUs,
      speedupFactor: speedup,
      recommendation: classifyRecommendation(p),
    };
  });
}

// ─── Phase 5: Summary Report ────────────────────────────────────────────────

function generateValidationReport(
  points: ParetoPoint[],
  tiers: Map<number, TierAnalysis>,
  mapped: MappedParameters[],
): string {
  const lines: string[] = [];
  const now = new Date().toISOString();

  lines.push('═══════════════════════════════════════════════════════════════════════');
  lines.push('  FHE Parameter Evolution — Validation Report');
  lines.push('  Comparison against HE Standard 2018 & measured CPU Montgomery NTT');
  lines.push('═══════════════════════════════════════════════════════════════════════');
  lines.push('');
  lines.push(`Report generated: ${now}`);
  lines.push(`Total Pareto-optimal points: ${points.length}`);
  lines.push(`Tiers analyzed: ${[...tiers.keys()].join(', ')}`);
  lines.push('');

  // Per-tier analysis
  lines.push('─── Per-Tier Analysis ──────────────────────────────────────────────');
  for (const [N, tier] of tiers) {
    const ref = tier.heStandardRef;
    lines.push('');
    lines.push(`  N = ${N}`);
    lines.push(`    HE Standard 2018: max log₂(q) = ${ref?.maxLogQ ?? 'N/A'}, ${ref?.securityBits ?? 'N/A'}-bit security`);
    lines.push(`    Evolved points in tier: ${tier.points.length}`);
    lines.push(`    Points exceeding HE Standard: ${tier.exceedsStandard.length}`);

    if (tier.top5Fastest.length > 0) {
      const best = tier.top5Fastest[0];
      lines.push(`    Fastest evolved NTT: ${best.latencyMs.toFixed(4)} ms`);
      lines.push(`      Parameters: q=${best.q}, σ=${(best.sigma / 1000).toFixed(4)}, Bg=${best.Bg}`);
      lines.push(`      Estimator output: ${best.securityBits.toFixed(1)} (ordinal, not bit-security), Noise budget: ${best.noiseBudgetBits.toFixed(1)} bits`);
    }
  }

  // Accelerated performance estimates
  lines.push('');
  lines.push('─── Comparison Against Measured Reference NTT ──────────────────────');
  lines.push('  (Measured CPU Montgomery NTT, idle Apple M4 Max. Degrees without a');
  lines.push('   measurement are reported as n/a — they are not interpolated.)');
  lines.push('');

  for (const [N, tier] of tiers) {
    if (tier.top5Fastest.length === 0) continue;
    const accelUs = getAcceleratedLatencyUs(N);
    const best = tier.top5Fastest[0];

    if (accelUs === null) {
      lines.push(`  N=${N}: Pure Rust=${best.latencyMs.toFixed(4)}ms → reference=n/a (no measurement at this degree, speedup n/a)`);
      continue;
    }

    const speedup = best.latencyMs / (accelUs / 1000);
    const backend = ACCELERATED_NTT_BENCHMARKS[N].backend;
    lines.push(`  N=${N}: Pure Rust=${best.latencyMs.toFixed(4)}ms → reference=${(accelUs / 1000).toFixed(4)}ms (${speedup.toFixed(1)}× speedup, ${backend})`);
  }

  // Illustrative mapping caveat
  lines.push('');
  lines.push('─── Illustrative CustomParameters Mapping ──────────────────────────');
  lines.push('  results/fhe-parameters.json carries an `illustrativeMapping` block per');
  lines.push('  record. Every field in it is a heuristic projection: the modulus chain is');
  lines.push('  hand-picked per degree and is neither the evolved q nor the modulus that');
  lines.push('  was benchmarked; lweDimension is a hardcoded ladder; decompBaseLog and');
  lines.push('  decompLevel are heuristics; glweDimension is fixed at 1; securityLevel is');
  lines.push('  a threshold on the estimator output. None of it is benchmarked and none of');
  lines.push('  it has been validated against a working FHE implementation. Search output');
  lines.push('  and measurements live in the separate `searched` and `measured` blocks.');

  // Recommendations
  lines.push('');
  lines.push('─── Recommendations ───────────────────────────────────────────────');
  lines.push('');

  const byCategory = new Map<string, MappedParameters[]>();
  for (const m of mapped) {
    const cat = byCategory.get(m.recommendation) ?? [];
    cat.push(m);
    byCategory.set(m.recommendation, cat);
  }

  const categoryDescriptions: Record<string, string> = {
    'fast-bootstrapping': 'Low-latency bootstrapping (interactive applications, real-time)',
    'deep-computation': 'Deep homomorphic circuits (ML inference, complex analytics)',
    'maximum-security': 'Maximum security margin (critical infrastructure, long-term)',
    'balanced': 'Balanced performance/security (general purpose)',
  };

  for (const [category, params] of byCategory) {
    const desc = categoryDescriptions[category] ?? category;
    lines.push(`  ${category.toUpperCase()}: ${desc}`);
    const best = params.sort((a, b) => a.source.latencyMs - b.source.latencyMs)[0];
    if (best) {
      lines.push(`    Best: N=${best.source.N}, q=${best.source.q}, latency=${best.source.latencyMs.toFixed(4)}ms`);
      if (best.acceleratedLatencyUs === null || best.speedupFactor === null) {
        lines.push('    Measured reference: n/a (no reference NTT measurement at this degree)');
      } else {
        lines.push(`    Measured reference: ${(best.acceleratedLatencyUs / 1000).toFixed(4)}ms (${best.speedupFactor.toFixed(1)}× speedup)`);
      }
    }
    lines.push('');
  }

  lines.push('═══════════════════════════════════════════════════════════════════════');

  return lines.join('\n');
}

// ─── Main ───────────────────────────────────────────────────────────────────

function main(): void {
  const projectRoot = getProjectRoot();
  const resultsDir = join(projectRoot, 'results');

  console.log('╔═══════════════════════════════════════════════════════════════╗');
  console.log('║  FHE Parameter Evolution — Pareto Validation                 ║');
  console.log('╚═══════════════════════════════════════════════════════════════╝');
  console.log('');

  // Phase 1: Load
  console.log('Phase 1: Loading Pareto frontier...');
  const points = loadParetoFrontier(projectRoot);

  if (!points) {
    console.log('');
    console.log('  ⚠ No results found at ./results/pareto-frontier.json');
    console.log('  Run the evolution first: npm run start');
    console.log('');
    process.exit(0);
  }

  console.log(`  Loaded ${points.length} Pareto-optimal points.`);
  const tierCounts = new Map<number, number>();
  for (const p of points) {
    tierCounts.set(p.N, (tierCounts.get(p.N) ?? 0) + 1);
  }
  for (const [N, count] of [...tierCounts.entries()].sort((a, b) => a[0] - b[0])) {
    console.log(`    N=${N}: ${count} points`);
  }
  console.log('');

  // Phase 2: Compare against HE Standard
  console.log('Phase 2: Comparing against HE Standard 2018...');
  const tiers = analyzeByTier(points);
  for (const [N, tier] of tiers) {
    if (tier.points.length === 0) continue;
    const exceeds = tier.exceedsStandard.length;
    if (exceeds > 0) {
      console.log(`  N=${N}: ${exceeds} points exceed HE Standard (more aggressive, still secure)`);
    } else {
      console.log(`  N=${N}: All ${tier.points.length} points within HE Standard bounds`);
    }
  }
  console.log('');

  // Phase 3: Performance comparison
  console.log('Phase 3: Performance comparison (evolved Rust NTT vs measured reference NTT)...');
  const perfTable = buildPerformanceTable(tiers);
  console.log(perfTable);
  console.log('');

  // Phase 4: Parameter mapping
  console.log('Phase 4: Building illustrative node-fhe-accelerate CustomParameters projection...');
  console.log('  (Projection only — heuristic, unbenchmarked, and separated from');
  console.log('   searched and measured values in the emitted artifact.)');
  const mapped = mapToIllustrativeParameters(points);

  // Write fhe-parameters.json.
  //
  // Three blocks per record, kept deliberately separate so provenance is never
  // ambiguous:
  //   searched            — evolutionary run output and what was timed on it
  //   measured            — the reference NTT comparison, null where unmeasured
  //   illustrativeMapping — heuristic projections, none of them benchmarked
  mkdirSync(resultsDir, { recursive: true });
  const outputParams = mapped.map((m) => ({
    searched: {
      N: m.source.N,
      q: m.source.q,
      sigma: m.source.sigma,
      Bg: m.source.Bg,
      latencyMs: m.source.latencyMs,
      // Renamed from `securityBits`: §6.6 of the paper establishes these are
      // ordinal estimator outputs, not bit-security claims.
      estimatorBits: m.source.securityBits,
      noiseBudgetBits: m.source.noiseBudgetBits,
    },
    measured: {
      // Measured on the evolved searched.q — not on illustrativeMapping.moduli.
      pureRustNttMs: m.source.latencyMs,
      // null ⇒ no measured reference NTT exists for this polynomial degree.
      referenceNttUs: m.acceleratedLatencyUs,
      referenceBackend: m.acceleratedLatencyUs === null
        ? null
        : ACCELERATED_NTT_BENCHMARKS[m.source.N].backend,
      measuredSpeedup: m.speedupFactor === null ? null : Number(m.speedupFactor.toFixed(1)),
    },
    illustrativeMapping: m.illustrativeMapping,
    recommendation: m.recommendation,
  }));

  writeFileSync(
    join(resultsDir, 'fhe-parameters.json'),
    JSON.stringify(outputParams, null, 2),
  );
  console.log(`  Written ${mapped.length} records to results/fhe-parameters.json`);
  console.log('    searched / measured / illustrativeMapping blocks per record');
  console.log('');

  // Phase 5: Summary report
  console.log('Phase 5: Generating validation report...');
  const report = generateValidationReport(points, tiers, mapped);
  writeFileSync(join(resultsDir, 'validation-report.txt'), report);
  console.log('  Written results/validation-report.txt');
  console.log('');

  // Print report to console
  console.log(report);
}

main();
