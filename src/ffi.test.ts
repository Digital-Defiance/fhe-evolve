import { describe, it, expect } from 'vitest';
import { Genome, GENOME_FIELD_COUNT } from './genome.js';
import { PopulationManager } from './population-manager.js';
import {
  validateBatch,
  executeBenchmark,
  NOISE_FAILURE_SCORE,
  type NativeModule,
} from './ffi.js';

describe('FFI Boundary', () => {
  describe('validateBatch', () => {
    it('returns genome count for valid batch', () => {
      const batch = new Uint32Array(8); // 2 genomes × 4 fields
      expect(validateBatch(batch)).toBe(2);
    });

    it('returns 0 for empty batch', () => {
      const batch = new Uint32Array(0);
      expect(validateBatch(batch)).toBe(0);
    });

    it('throws for batch with invalid length', () => {
      const batch = new Uint32Array(5); // not a multiple of 4
      expect(() => validateBatch(batch)).toThrow('not a multiple of 4');
    });

    it('correctly counts genomes for larger batches', () => {
      const batch = new Uint32Array(100 * GENOME_FIELD_COUNT);
      expect(validateBatch(batch)).toBe(100);
    });
  });

  describe('packing roundtrip (Node.js side)', () => {
    it('packs and unpacks a single genome correctly', () => {
      const genome = new Genome({ N: 2048, q: 268435456, sigma: 3200, Bg: 1024 });
      const packed = genome.toUint32Array();

      expect(packed.length).toBe(GENOME_FIELD_COUNT);
      expect(packed[0]).toBe(2048);
      expect(packed[1]).toBe(268435456);
      expect(packed[2]).toBe(3200);
      expect(packed[3]).toBe(1024);

      const restored = Genome.fromUint32Array(packed);
      expect(restored.N).toBe(genome.N);
      expect(restored.q).toBe(genome.q);
      expect(restored.sigma).toBe(genome.sigma);
      expect(restored.Bg).toBe(genome.Bg);
    });

    it('packs multiple genomes into a contiguous flat array', () => {
      const genomes = [
        new Genome({ N: 1024, q: 134217728, sigma: 3200, Bg: 512 }),
        new Genome({ N: 4096, q: 536870912, sigma: 5000, Bg: 2048 }),
        new Genome({ N: 8192, q: 1073741824, sigma: 3200, Bg: 4096 }),
      ];

      const buffer = new Uint32Array(genomes.length * GENOME_FIELD_COUNT);
      genomes.forEach((g, i) => g.packInto(buffer, i * GENOME_FIELD_COUNT));

      // Verify each genome can be read back at its offset
      for (let i = 0; i < genomes.length; i++) {
        const restored = Genome.fromUint32Array(buffer, i * GENOME_FIELD_COUNT);
        expect(restored.N).toBe(genomes[i].N);
        expect(restored.q).toBe(genomes[i].q);
        expect(restored.sigma).toBe(genomes[i].sigma);
        expect(restored.Bg).toBe(genomes[i].Bg);
      }
    });

    it('PopulationManager.packForFFI produces valid batch', () => {
      const pm = new PopulationManager({ populationSize: 10 });
      pm.seed();

      const batch = pm.packForFFI();
      expect(batch.length).toBe(10 * GENOME_FIELD_COUNT);
      expect(validateBatch(batch)).toBe(10);

      // Verify first genome matches population
      const population = pm.getPopulation();
      const first = Genome.fromUint32Array(batch, 0);
      expect(first.N).toBe(population[0].N);
      expect(first.q).toBe(population[0].q);
      expect(first.sigma).toBe(population[0].sigma);
      expect(first.Bg).toBe(population[0].Bg);
    });
  });

  describe('executeBenchmark (with mock native module)', () => {
    /** Mock native module that mimics the Rust stub behavior. */
    const mockNative: NativeModule = {
      runBenchmark(parameterBatch: Uint32Array): Float64Array {
        const count = parameterBatch.length / GENOME_FIELD_COUNT;
        const scores = new Float64Array(count);
        for (let i = 0; i < count; i++) {
          const offset = i * GENOME_FIELD_COUNT;
          const n = parameterBatch[offset];
          const sigma = parameterBatch[offset + 2];
          scores[i] = 1_000_000 / (n * (sigma / 1000));
        }
        return scores;
      },
      runBenchmarkDeep(parameterBatch: Uint32Array, _iterations: number): Float64Array {
        // Same logic as runBenchmark for testing purposes
        return this.runBenchmark(parameterBatch);
      },
      validateBatch(parameterBatch: Uint32Array): number {
        return parameterBatch.length / GENOME_FIELD_COUNT;
      },
      echoGenome(parameterBatch: Uint32Array, index: number): Uint32Array {
        const offset = index * GENOME_FIELD_COUNT;
        return new Uint32Array([
          parameterBatch[offset],
          parameterBatch[offset + 1],
          parameterBatch[offset + 2],
          parameterBatch[offset + 3],
        ]);
      },
      benchmarkPoolInfo() {
        return { threads: 1, performanceCores: 1, qosAvailable: true };
      },
    };

    it('returns correct genome count', () => {
      const genome = new Genome({ N: 2048, q: 268435456, sigma: 3200, Bg: 1024 });
      const batch = genome.toUint32Array();

      const result = executeBenchmark(mockNative, batch);
      expect(result.genomeCount).toBe(1);
    });

    it('returns one score per genome', () => {
      const pm = new PopulationManager({ populationSize: 5 });
      pm.seed();
      const batch = pm.packForFFI();

      const result = executeBenchmark(mockNative, batch);
      expect(result.scores.length).toBe(5);
      expect(result.genomeCount).toBe(5);
    });

    it('scores are positive for valid parameters', () => {
      const genome = new Genome({ N: 1024, q: 134217728, sigma: 3200, Bg: 512 });
      const batch = genome.toUint32Array();

      const result = executeBenchmark(mockNative, batch);
      expect(result.scores[0]).toBeGreaterThan(0);
    });

    it('counts noise failures correctly', () => {
      const failingNative: NativeModule = {
        ...mockNative,
        runBenchmark(parameterBatch: Uint32Array): Float64Array {
          const count = parameterBatch.length / GENOME_FIELD_COUNT;
          const scores = new Float64Array(count);
          // Mark every other genome as a noise failure
          for (let i = 0; i < count; i++) {
            scores[i] = i % 2 === 0 ? 42.0 : NOISE_FAILURE_SCORE;
          }
          return scores;
        },
      };

      const pm = new PopulationManager({ populationSize: 4 });
      pm.seed();
      const batch = pm.packForFFI();

      const result = executeBenchmark(failingNative, batch);
      expect(result.failureCount).toBe(2);
      expect(result.genomeCount).toBe(4);
    });

    it('throws for invalid batch passed to executeBenchmark', () => {
      const badBatch = new Uint32Array(3); // not a multiple of 4
      expect(() => executeBenchmark(mockNative, badBatch)).toThrow();
    });

    it('echoGenome roundtrip matches original packing', () => {
      const genome = new Genome({ N: 4096, q: 536870912, sigma: 5000, Bg: 2048 });
      const batch = genome.toUint32Array();

      const echoed = mockNative.echoGenome(batch, 0);
      expect(echoed[0]).toBe(4096);
      expect(echoed[1]).toBe(536870912);
      expect(echoed[2]).toBe(5000);
      expect(echoed[3]).toBe(2048);
    });
  });
});
