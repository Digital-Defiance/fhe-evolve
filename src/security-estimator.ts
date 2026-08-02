/**
 * Lightweight LWE security estimator calibrated against the Homomorphic Encryption
 * Standard (2018) parameter tables.
 *
 * Uses the primal-uSVP attack cost model with calibration constant derived from
 * known-secure TFHE/BFV parameters. This avoids depending on an external Python
 * lattice-estimator while providing estimates consistent with standard parameter
 * selection guidance.
 *
 * Key relationship (for σ = 3.2, which is standard):
 *   N=1024  → max log₂(q) ≈ 27  for 128-bit security
 *   N=2048  → max log₂(q) ≈ 54  for 128-bit security
 *   N=4096  → max log₂(q) ≈ 109 for 128-bit security
 *   N=8192  → max log₂(q) ≈ 218 for 128-bit security
 *
 * Formula: security_bits ≈ C · N / log₂(q/σ)
 * Where C ≈ 3.166 is calibrated so that N=1024, q=2^27, σ=3.2 → 128 bits.
 *
 * Noise budget: log₂(q/(2σ)) - 1 bits of computation depth available.
 */

import type { SecurityEstimate } from './types.js';

/** Minimum security threshold in bits. */
export const SECURITY_FLOOR_BITS = 128;

/**
 * Calibration constant derived from the Homomorphic Encryption Standard.
 * Computed as: 128 * log₂(2^27 / 3.2) / 1024 ≈ 3.166
 */
const HE_STANDARD_CONSTANT = 3.166;

/**
 * Estimate LWE security level in bits using the primal-uSVP cost model.
 *
 * The formula models the cost of the best known lattice attack (BKZ reduction
 * in the primal attack) against the LWE instance defined by (N, q, σ).
 *
 * Higher N → more secure (larger lattice dimension)
 * Higher q → less secure (easier distinguishing)
 * Higher σ → more secure (more noise to hide the secret)
 *
 * @param N - Polynomial degree / lattice dimension
 * @param q - Ciphertext modulus
 * @param sigmaScaled - Noise standard deviation (scaled by 1000, so 3200 = σ of 3.2)
 * @returns Estimated security level in bits
 */
export function estimateSecurityBits(N: number, q: number, sigmaScaled: number): number {
  if (N <= 0 || q <= 0 || sigmaScaled <= 0) return 0;

  const sigma = sigmaScaled / 1000;
  const qOverSigma = q / sigma;

  if (qOverSigma <= 1) {
    // Modulus smaller than noise — scheme is unusable but trivially "secure"
    // (can't encode anything). Treat as 0 security for practical purposes.
    return 0;
  }

  const log2QOverSigma = Math.log2(qOverSigma);

  if (log2QOverSigma <= 0) return 0;

  return HE_STANDARD_CONSTANT * N / log2QOverSigma;
}

/**
 * Estimate the theoretical noise budget for an FHE parameter set.
 *
 * The noise budget represents how many bits of "computation depth" are available
 * before the ciphertext noise overwhelms the modulus and decryption fails:
 *   budget = log₂(q / (2σ)) - 1
 *
 * @param q - Ciphertext modulus
 * @param sigmaScaled - Noise standard deviation (scaled by 1000)
 * @returns Noise budget in bits (0 if parameters are degenerate)
 */
export function estimateNoiseBudget(q: number, sigmaScaled: number): number {
  if (q <= 0 || sigmaScaled <= 0) return 0;

  const sigma = sigmaScaled / 1000;
  const budget = Math.log2(q / (2 * sigma)) - 1;
  return Math.max(0, budget);
}

/**
 * Full security evaluation of an LWE parameter set.
 *
 * Combines security level estimation and noise budget calculation into a single
 * assessment. Used by the Guillotine to make pass/fail decisions.
 *
 * @param N - Polynomial degree
 * @param q - Ciphertext modulus
 * @param sigmaScaled - Noise standard deviation (scaled by 1000)
 * @returns Complete security estimate with security bits, pass/fail, and noise budget
 */
export function estimateSecurity(N: number, q: number, sigmaScaled: number): SecurityEstimate {
  const securityBits = estimateSecurityBits(N, q, sigmaScaled);
  const noiseBudgetBits = estimateNoiseBudget(q, sigmaScaled);

  return {
    securityBits,
    isSecure: securityBits >= SECURITY_FLOOR_BITS,
    noiseBudgetBits,
  };
}
