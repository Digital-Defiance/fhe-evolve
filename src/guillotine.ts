/**
 * The Guillotine — security sieve for the FHE parameter evolution simulation.
 *
 * Filters a population of genomes, immediately discarding any parameter set
 * that does not meet the 128-bit security floor. Genomes that survive are
 * annotated with their estimated security level and noise growth budget.
 */

import type { Genome } from './genome.js';
import { estimateSecurity, SECURITY_FLOOR_BITS } from './security-estimator.js';
import type { SecureGenome } from './types.js';

export { SECURITY_FLOOR_BITS };

/**
 * Filter a population of genomes through the security sieve.
 *
 * For each genome:
 *  1. Evaluate LWE security using the lattice estimator
 *  2. Discard genomes below 128-bit security
 *  3. Attach noise budget to surviving genomes
 *
 * @param genomes - Array of genomes to evaluate
 * @returns Array of SecureGenome objects that passed the security threshold
 */
export function guillotine(genomes: readonly Genome[]): SecureGenome[] {
  const survivors: SecureGenome[] = [];

  for (const genome of genomes) {
    const estimate = estimateSecurity(genome.N, genome.q, genome.sigma);

    if (estimate.isSecure) {
      survivors.push({
        genome,
        securityBits: estimate.securityBits,
        noiseBudgetBits: estimate.noiseBudgetBits,
      });
    }
  }

  return survivors;
}

/**
 * Evaluate a single genome's security without filtering.
 * Useful for debugging or inspecting individual parameter sets.
 *
 * @param genome - The genome to evaluate
 * @returns The full security estimate
 */
export function evaluateGenome(genome: Genome) {
  return estimateSecurity(genome.N, genome.q, genome.sigma);
}
