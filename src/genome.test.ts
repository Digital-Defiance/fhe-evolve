import { describe, it, expect } from 'vitest';
import { Genome, GENOME_FIELD_COUNT } from './genome.js';

describe('Genome', () => {
  const params = { N: 4096, q: 536870912, sigma: 3200, Bg: 2048 };

  it('stores FHE parameters correctly', () => {
    const g = new Genome(params);
    expect(g.N).toBe(4096);
    expect(g.q).toBe(536870912);
    expect(g.sigma).toBe(3200);
    expect(g.Bg).toBe(2048);
  });

  it('serializes to Uint32Array', () => {
    const g = new Genome(params);
    const arr = g.toUint32Array();
    expect(arr).toBeInstanceOf(Uint32Array);
    expect(arr.length).toBe(GENOME_FIELD_COUNT);
    expect(arr[0]).toBe(4096);
    expect(arr[1]).toBe(536870912);
    expect(arr[2]).toBe(3200);
    expect(arr[3]).toBe(2048);
  });

  it('packs into buffer at offset', () => {
    const g = new Genome(params);
    const buffer = new Uint32Array(8);
    g.packInto(buffer, 4);
    expect(buffer[4]).toBe(4096);
    expect(buffer[5]).toBe(536870912);
    expect(buffer[6]).toBe(3200);
    expect(buffer[7]).toBe(2048);
  });

  it('deserializes from Uint32Array', () => {
    const g = new Genome(params);
    const arr = g.toUint32Array();
    const restored = Genome.fromUint32Array(arr);
    expect(restored.N).toBe(g.N);
    expect(restored.q).toBe(g.q);
    expect(restored.sigma).toBe(g.sigma);
    expect(restored.Bg).toBe(g.Bg);
  });

  it('deserializes from buffer at offset', () => {
    const buffer = new Uint32Array([0, 0, 8192, 1073741824, 3200, 4096, 0, 0]);
    const g = Genome.fromUint32Array(buffer, 2);
    expect(g.N).toBe(8192);
    expect(g.q).toBe(1073741824);
    expect(g.sigma).toBe(3200);
    expect(g.Bg).toBe(4096);
  });

  it('clones with overrides', () => {
    const g = new Genome(params);
    const cloned = g.clone({ N: 8192 });
    expect(cloned.N).toBe(8192);
    expect(cloned.q).toBe(params.q);
    expect(cloned.sigma).toBe(params.sigma);
    expect(cloned.Bg).toBe(params.Bg);
  });

  it('toJSON returns plain object', () => {
    const g = new Genome(params);
    expect(g.toJSON()).toEqual(params);
  });
});
