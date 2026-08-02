import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, it, expect } from 'vitest';
import { formatDuration } from './pareto-explorer.js';

/** Repo root, for locating the results/ artifacts. */
const projectRoot = new URL('..', import.meta.url).pathname;

describe('formatDuration', () => {
  it('returns seconds for < 60s', () => {
    expect(formatDuration(0)).toBe('0s');
    expect(formatDuration(500)).toBe('1s');
    expect(formatDuration(45_000)).toBe('45s');
    expect(formatDuration(59_499)).toBe('59s');
  });

  it('returns minutes and seconds for 60s-3600s', () => {
    expect(formatDuration(60_000)).toBe('1m');
    expect(formatDuration(90_000)).toBe('1m 30s');
    expect(formatDuration(750_000)).toBe('12m 30s');
    expect(formatDuration(3_599_000)).toBe('59m 59s');
  });

  it('returns hours and minutes for > 3600s', () => {
    expect(formatDuration(3_600_000)).toBe('1h');
    expect(formatDuration(3_600_000 + 45 * 60_000)).toBe('1h 45m');
    expect(formatDuration(3 * 3_600_000 + 45 * 60_000)).toBe('3h 45m');
  });

  it('handles edge cases', () => {
    expect(formatDuration(-1)).toBe('??');
    expect(formatDuration(Infinity)).toBe('??');
    expect(formatDuration(NaN)).toBe('??');
  });
});

describe('tier N-locking invariant', () => {
  /**
   * WHY THIS TEST EXISTS.
   *
   * The design premise is that populations are stratified by polynomial degree:
   * the N=1024 tier searches N=1024 and nothing else. The mutation operator can
   * change N, so every mutation inside a tier has to be followed by re-locking
   * N. Two of the three mutation sites did not do that.
   *
   * The consequence was invisible in every summary the run produced, because
   * both the per-tier fronts and the merged frontier were internally
   * consistent. It only surfaced when the two were cross-checked: the per-tier
   * files summed to 1972 while the merged frontier held 1971, and three tiers
   * held MORE genomes in the merged set than in their own front, which
   * deduplication cannot produce. The actual contamination was:
   *
   *   tier N=1024  front: 423 with N=1024, 16 with N=2048
   *   tier N=2048  front: 370 with N=2048, 17 with N=4096
   *   tier N=4096  front: 354 with N=4096, 14 with N=8192
   *   tier N=8192  front: 377 with N=8192, 13 with N=16384
   *   tier N=16384 front: 388 with N=16384  (clean: no tier above it)
   *
   * Every tier leaked upward into the next. This test asserts the invariant
   * directly against whatever frontier artifacts are present, so a regression
   * fails here rather than being discovered by arithmetic three months later.
   */
  it('every genome in a per-tier front carries that tier N', () => {
    const resultsDir = join(projectRoot, 'results');
    const tiers = [1024, 2048, 4096, 8192, 16384];
    const offenders: string[] = [];

    for (const N of tiers) {
      const file = join(resultsDir, `pareto-N${N}.json`);
      if (!existsSync(file)) continue;
      const points = JSON.parse(readFileSync(file, 'utf8')) as { N: number }[];
      const foreign = points.filter((p) => p.N !== N);
      if (foreign.length > 0) {
        const breakdown = new Map<number, number>();
        for (const p of foreign) breakdown.set(p.N, (breakdown.get(p.N) ?? 0) + 1);
        offenders.push(
          `tier N=${N}: ${foreign.length} foreign genomes ` +
            `(${[...breakdown].map(([n, c]) => `${c}x N=${n}`).join(', ')})`,
        );
      }
    }

    if (offenders.length > 0) {
      throw new Error(
        `Tier stratification is broken. A tier's Pareto front must contain only ` +
          `that tier's polynomial degree, because latency is compared within a ` +
          `tier and roughly doubles per doubling of N.\n  ${offenders.join('\n  ')}\n` +
          `Cause to check: a mutation site that is not followed by re-locking N ` +
          `(see mutateInTier in pareto-explorer.ts).`,
      );
    }
  });

  it('per-tier fronts and the merged frontier reconcile', () => {
    const resultsDir = join(projectRoot, 'results');
    const merged = join(resultsDir, 'pareto-frontier.json');
    if (!existsSync(merged)) return;

    const mergedPoints = JSON.parse(readFileSync(merged, 'utf8')) as { N: number }[];
    const mergedByN = new Map<number, number>();
    for (const p of mergedPoints) mergedByN.set(p.N, (mergedByN.get(p.N) ?? 0) + 1);

    // Deduplication can only remove, so the merged count for a degree can never
    // exceed the number of genomes of that degree across all per-tier fronts.
    const perTierByN = new Map<number, number>();
    for (const N of [1024, 2048, 4096, 8192, 16384]) {
      const file = join(resultsDir, `pareto-N${N}.json`);
      if (!existsSync(file)) continue;
      const points = JSON.parse(readFileSync(file, 'utf8')) as { N: number }[];
      for (const p of points) perTierByN.set(p.N, (perTierByN.get(p.N) ?? 0) + 1);
    }

    for (const [N, mergedCount] of mergedByN) {
      const available = perTierByN.get(N) ?? 0;
      expect(
        mergedCount,
        `merged frontier holds ${mergedCount} genomes at N=${N} but the per-tier ` +
          `fronts only contain ${available}; deduplication cannot create genomes`,
      ).toBeLessThanOrEqual(available);
    }
  });
});
