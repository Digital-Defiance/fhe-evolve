#!/usr/bin/env node
/**
 * Smoke-check that benchmark latencies are no longer bimodal (~3.3× E-core mode).
 *
 * Mimics one evolutionary generation: a parallel batch of genomes through
 * runBenchmarkDeep (P-core pool + min-of-k). Passes if the slow cluster is gone
 * without any post-hoc 1.5× filter.
 *
 * Usage:  npm run smoke:bimodality
 * Prefers an idle machine; competing load can inflate spread without the 3.3× gap.
 */
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const require = createRequire(import.meta.url);
const root = join(dirname(fileURLToPath(import.meta.url)), '..');

function loadNative() {
  const paths = [
    join(root, 'native.node'),
    join(root, 'native/target/release/libfhe_evolve_native.dylib'),
  ];
  for (const p of paths) {
    try {
      return require(p);
    } catch {
      /* try next */
    }
  }
  throw new Error('Native module not found. Run `npm run build:native` first.');
}

/** q ≡ 1 (mod 2N) primes drawn from the same style of pool the search uses. */
const GENOMES = [
  // N=1024
  ...[12289, 40961, 65537, 114689, 147457, 163841, 188417, 270337].map((q) => [
    1024, q, 3200, 512,
  ]),
  // N=2048
  ...[12289, 40961, 65537, 114689, 147457, 163841, 270337, 319489].map((q) => [
    2048, q, 3200, 1024,
  ]),
  // N=4096
  ...[40961, 65537, 114689, 147457, 163841, 270337, 786433, 557057].filter(
    (q) => (q - 1) % (2 * 4096) === 0,
  ).map((q) => [4096, q, 3200, 2048]),
];

// Replicate to ~batch size of a generation slice so rayon actually parallelises.
const REPLICAS = 8;
const ITERATIONS = 50; // enough for a median; full run uses 100
const SLOW_RATIO = 2.5; // old artefact sat at ~3.3×
const MID_LO = 1.4;
const MID_HI = 2.5;
const MAX_SLOW_FRAC = 0.02; // allow ≤2% stragglers at ≥2.5×
// Informational only: competing load can widen spread without the 3.3× gap.
const WARN_SPREAD = 0.5;

function main() {
  const native = loadNative();

  if (typeof native.benchmarkPoolInfo === 'function') {
    const info = native.benchmarkPoolInfo();
    console.log('Benchmark pool:', info);
    if (!info.qosAvailable) {
      console.error('FAIL: interactive QoS unavailable — P-core preference will not apply');
      process.exit(1);
    }
    if (info.threads !== info.performanceCores) {
      console.error(
        `FAIL: pool threads (${info.threads}) != performance cores (${info.performanceCores})`,
      );
      process.exit(1);
    }
  } else {
    console.warn('WARN: benchmarkPoolInfo missing — rebuild native after the pinning change');
  }

  const flat = [];
  for (let r = 0; r < REPLICAS; r++) {
    for (const g of GENOMES) flat.push(...g);
  }
  const batch = Uint32Array.from(flat);
  const genomeCount = batch.length / 4;

  console.log(
    `Benchmarking ${genomeCount} genomes × ${ITERATIONS} iters (min-of-k medians)…`,
  );
  const t0 = performance.now();
  const scores = native.runBenchmarkDeep(batch, ITERATIONS);
  const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
  console.log(`Done in ${elapsed}s`);

  const byN = new Map();
  for (let i = 0; i < genomeCount; i++) {
    const n = batch[i * 4];
    const lat = scores[i];
    if (lat < 0) {
      console.error(`FAIL: genome ${i} (N=${n}) returned sentinel ${lat}`);
      process.exit(1);
    }
    if (!byN.has(n)) byN.set(n, []);
    byN.get(n).push(lat);
  }

  let failed = false;
  let warned = false;
  for (const [n, lats] of [...byN.entries()].sort((a, b) => a[0] - b[0])) {
    const mn = Math.min(...lats);
    const mx = Math.max(...lats);
    const ratios = lats.map((l) => l / mn);
    const slow = ratios.filter((r) => r >= SLOW_RATIO).length;
    const mid = ratios.filter((r) => r > MID_LO && r < MID_HI).length;
    const spread = mx / mn - 1;
    const slowFrac = slow / lats.length;
    // The E-core artefact was a clean bimodality: fast mode, empty mid band,
    // then a pile at ~3.3×. Competing load widens the fast mode continuously
    // without that gap — that is not a fail for this smoke check.
    const gapThenSlow = mid === 0 && slow > 0;
    const pass = slowFrac <= MAX_SLOW_FRAC && !gapThenSlow;

    console.log(
      `N=${n}: n=${lats.length} min=${mn.toFixed(6)}ms max/min=${(mx / mn).toFixed(2)}× ` +
        `spread=${(spread * 100).toFixed(1)}% slow(≥${SLOW_RATIO}×)=${slow} mid(${MID_LO}–${MID_HI})=${mid} ` +
        (pass ? 'PASS' : 'FAIL'),
    );
    if (!pass) failed = true;
    if (pass && spread > WARN_SPREAD) {
      console.warn(
        `  WARN: spread ${(spread * 100).toFixed(1)}% with no slow cluster — likely load, re-run idle before the full search`,
      );
      warned = true;
    }
  }

  if (failed) {
    console.error(
      '\nFAIL: ~3.3× E-core cluster still present. ' +
        'Do not treat the flatness claim as unconditioned yet.',
    );
    process.exit(1);
  }

  console.log(
    '\nPASS: no ~3.3× slow cluster (QoS pool + min-of-k).' +
      (warned
        ? ' Prefer an idle machine for the full evolutionary re-run.'
        : ' Safe to start a full evolutionary re-run on an idle machine.'),
  );
}

main();
