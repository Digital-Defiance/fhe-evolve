/**
 * Genome class representing a set of FHE parameters.
 *
 * Each genome encodes:
 *  - N:     Polynomial degree (power of 2)
 *  - q:     Ciphertext modulus
 *  - sigma: Noise standard deviation (stored as integer, scaled x1000)
 *  - Bg:    Gadget decomposition base
 */

import type { FheParams } from './types.js';

/** Number of 32-bit fields per genome in the packed FFI format. */
export const GENOME_FIELD_COUNT = 4;

export class Genome {
  /** Polynomial degree (power of 2: 1024..16384) */
  readonly N: number;
  /** Ciphertext modulus */
  readonly q: number;
  /** Noise standard deviation scaled by 1000 (integer for FFI packing) */
  readonly sigma: number;
  /** Gadget decomposition base */
  readonly Bg: number;

  constructor(params: FheParams) {
    this.N = params.N;
    this.q = params.q;
    this.sigma = params.sigma;
    this.Bg = params.Bg;
  }

  /**
   * Serialize this genome into an existing Uint32Array at the given offset.
   * Layout: [N, q, sigma, Bg] — 4 consecutive uint32 values.
   */
  packInto(buffer: Uint32Array, offset: number): void {
    buffer[offset] = this.N;
    buffer[offset + 1] = this.q;
    buffer[offset + 2] = this.sigma;
    buffer[offset + 3] = this.Bg;
  }

  /**
   * Serialize this genome to a new Uint32Array of length 4.
   */
  toUint32Array(): Uint32Array {
    const buf = new Uint32Array(GENOME_FIELD_COUNT);
    this.packInto(buf, 0);
    return buf;
  }

  /**
   * Deserialize a genome from a Uint32Array at the given offset.
   */
  static fromUint32Array(buffer: Uint32Array, offset = 0): Genome {
    return new Genome({
      N: buffer[offset],
      q: buffer[offset + 1],
      sigma: buffer[offset + 2],
      Bg: buffer[offset + 3],
    });
  }

  /**
   * Create a copy of this genome with optional parameter overrides.
   */
  clone(overrides?: Partial<FheParams>): Genome {
    return new Genome({
      N: overrides?.N ?? this.N,
      q: overrides?.q ?? this.q,
      sigma: overrides?.sigma ?? this.sigma,
      Bg: overrides?.Bg ?? this.Bg,
    });
  }

  /**
   * Returns a plain object representation useful for logging.
   */
  toJSON(): FheParams {
    return { N: this.N, q: this.q, sigma: this.sigma, Bg: this.Bg };
  }
}
