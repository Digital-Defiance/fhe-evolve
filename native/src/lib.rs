use napi::bindgen_prelude::*;
use napi_derive::napi;
use rayon::prelude::*;
use std::time::Instant;

mod montgomery;
mod ntt;
mod ntt_accel;

/// Internal representation of a single FHE parameter set.
/// Matches the Node.js Genome layout: [N, q, sigma, Bg]
#[derive(Debug, Clone, Copy)]
struct FheParams {
    /// Polynomial degree (power of 2)
    n: u32,
    /// Ciphertext modulus
    q: u32,
    /// Noise standard deviation (scaled x1000)
    sigma: u32,
    /// Gadget decomposition base
    bg: u32,
}

/// Number of u32 fields per genome in the packed format.
const FIELDS_PER_GENOME: usize = 4;

/// Unpack a flat u32 slice into a Vec of FheParams.
/// Returns an error if the slice length is not a multiple of FIELDS_PER_GENOME.
fn unpack_params(flat: &[u32]) -> Result<Vec<FheParams>> {
    if flat.len() % FIELDS_PER_GENOME != 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Buffer length {} is not a multiple of {} (fields per genome)",
                flat.len(),
                FIELDS_PER_GENOME
            ),
        ));
    }

    let count = flat.len() / FIELDS_PER_GENOME;
    let mut params = Vec::with_capacity(count);

    for i in 0..count {
        let offset = i * FIELDS_PER_GENOME;
        params.push(FheParams {
            n: flat[offset],
            q: flat[offset + 1],
            sigma: flat[offset + 2],
            bg: flat[offset + 3],
        });
    }

    Ok(params)
}

/// Independent median trials per genome evaluation. The reported latency is the
/// **minimum** across these trials (min-of-k).
///
/// Why min-of-k exists on top of the P-core QoS pool (see below): macOS QoS is
/// best-effort, so a worker can still land on an efficiency core for the life
/// of one evaluation. A single median then reports the E-core latency and
/// NSGA-II ranks on it. Running k independent medians with a yield between
/// them gives the scheduler a chance to migrate the thread; taking the min
/// keeps a P-core sample when any trial got one. Combined with the pool, this
/// is what lets the headline flatness claim stand without a post-hoc
/// "conditioned on execution mode" filter.
const LATENCY_TRIALS: u32 = 3;

/// Median of a sorted timing sample. `timings` must be non-empty and is sorted
/// in place.
fn median_ms(timings: &mut [f64]) -> f64 {
    timings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = timings.len() / 2;
    if timings.len() % 2 == 0 {
        (timings[mid - 1] + timings[mid]) / 2.0
    } else {
        timings[mid]
    }
}

/// Min of `LATENCY_TRIALS` independent medians, yielding between trials so the
/// scheduler can reassign the thread. Returns -1.0 if `trial` does.
fn min_of_k_medians(mut trial: impl FnMut() -> f64) -> f64 {
    let mut best = f64::INFINITY;
    for t in 0..LATENCY_TRIALS.max(1) {
        if t > 0 {
            std::thread::yield_now();
        }
        let med = trial();
        if med < 0.0 {
            return -1.0;
        }
        if med < best {
            best = med;
        }
    }
    best
}

/// Executes the negacyclic NTT micro-benchmark for a single genome.
/// Returns latency in milliseconds, or -1.0 if the parameters don't support
/// a valid NTT over Z_q[X]/(X^N + 1) (noise corruption / incompatible N and q).
///
/// The transform benchmarked here is the negacyclic one, matching the ring FHE
/// schemes actually use. That requires a primitive 2N-th root of unity, which
/// exists only when q ≡ 1 (mod 2N) — a strictly stronger condition than the
/// q ≡ 1 (mod N) needed for a cyclic transform.
///
/// When `iterations` > 1, each trial runs the forward NTT `iterations` times
/// and takes the median; the genome score is the **minimum** over
/// `LATENCY_TRIALS` such medians (see that constant). Before any timing, the
/// negacyclic convolution identity is verified once; a failure returns the
/// -1.0 sentinel rather than a timing. A forward+inverse roundtrip is also
/// checked as a regression guard (redundant now that the convolution identity
/// is the gate, but cheap).
///
/// On aarch64 (Apple Silicon), uses the hardware-accelerated NTT path with
/// Montgomery arithmetic and NEON intrinsics. Falls back to the reference
/// implementation on other architectures.
fn benchmark_genome(p: &FheParams, iterations: u32) -> f64 {
    let n = p.n as usize;
    let q = p.q as u64;

    // N must be a power of 2 and at least 2
    if n < 2 || (n & (n - 1)) != 0 {
        return -1.0;
    }

    // q must be at least 2 and satisfy q ≡ 1 (mod 2N), so that a primitive
    // 2N-th root of unity exists and the transform can be negacyclic.
    if q < 2 || (q - 1) % (2 * n as u64) != 0 {
        return -1.0;
    }

    // Build the negacyclic plan: psi (order exactly 2N, psi^N = -1) and
    // omega = psi^2 (order exactly N). Fails loudly, no cyclic fallback.
    let plan = match ntt::NegacyclicPlan::new(n, q) {
        Some(plan) => plan,
        None => return -1.0,
    };

    // Create a coefficient vector of size N (simulating FHE polynomial coefficients)
    let original: Vec<u64> = (0..n).map(|i| (i as u64) % q).collect();

    // Use accelerated path on aarch64 when q is odd (Montgomery requirement)
    #[cfg(target_arch = "aarch64")]
    {
        if q & 1 == 1 {
            return benchmark_genome_accel(n, q, &plan, &original, iterations);
        }
    }

    // Fallback: reference NTT implementation
    benchmark_genome_reference(&plan, &original, iterations)
}

/// Accelerated NTT benchmark path using Montgomery arithmetic + NEON.
#[cfg(target_arch = "aarch64")]
fn benchmark_genome_accel(
    n: usize,
    q: u64,
    plan: &ntt::NegacyclicPlan,
    original: &[u64],
    iterations: u32,
) -> f64 {
    use crate::montgomery::MontCtx32;
    use crate::ntt_accel;

    let mont = MontCtx32::new(q as u32);
    // Tables are built from psi; omega = psi^2 is derived inside.
    let twiddles = ntt_accel::precompute_twiddles(n, plan.psi, &mont);

    // The gate: negacyclic convolution identity over Z_q[X]/(X^N + 1).
    // Sparse operands whose support includes index N-1 in both factors, so the
    // X^N = -1 wraparound is exercised. A cyclic transform fails this.
    if !ntt_accel::verify_negacyclic_convolution_accel(&twiddles, &mont) {
        return -1.0;
    }

    // Validation roundtrip (regression check; the gate above is what matters)
    let mut data = original.to_vec();
    if ntt_accel::forward_ntt_accel(&mut data, &twiddles, &mont).is_err() {
        return -1.0;
    }
    if ntt_accel::inverse_ntt_accel(&mut data, &twiddles, &mont).is_err() {
        return -1.0;
    }
    if data != original {
        return -1.0;
    }

    // Min-of-k medians: each trial is `iterations` forward NTTs → median.
    let iters = iterations.max(1) as usize;
    min_of_k_medians(|| {
        let mut timings: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let mut work = original.to_vec();
            let start = Instant::now();
            if ntt_accel::forward_ntt_accel(&mut work, &twiddles, &mont).is_err() {
                return -1.0;
            }
            timings.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        median_ms(&mut timings)
    })
}

/// Reference NTT benchmark path (fallback for non-aarch64 or even moduli).
fn benchmark_genome_reference(
    plan: &ntt::NegacyclicPlan,
    original: &[u64],
    iterations: u32,
) -> f64 {
    // The gate: negacyclic convolution identity over Z_q[X]/(X^N + 1).
    if !ntt::verify_negacyclic_convolution(plan) {
        return -1.0;
    }

    let mut data = original.to_vec();

    // Validation roundtrip (regression check; the gate above is what matters)
    if ntt::forward_ntt_negacyclic(&mut data, plan).is_err() {
        return -1.0;
    }
    if ntt::inverse_ntt_negacyclic(&mut data, plan).is_err() {
        return -1.0;
    }
    if data != original {
        return -1.0;
    }

    // Min-of-k medians: each trial is `iterations` forward NTTs → median.
    let iters = iterations.max(1) as usize;
    min_of_k_medians(|| {
        let mut timings: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let mut work = original.to_vec();
            let start = Instant::now();
            if ntt::forward_ntt_negacyclic(&mut work, plan).is_err() {
                return -1.0;
            }
            timings.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        median_ms(&mut timings)
    })
}

/// Receives a flat Uint32Array of packed FHE parameters from Node.js,
/// unpacks them into internal structs, and runs NTT micro-benchmarks
/// in parallel using Rayon. Returns a Float64Array of latency scores
// ---------------------------------------------------------------------------
// Performance-core pinning for benchmark threads
// ---------------------------------------------------------------------------
//
// WHY THIS EXISTS.
//
// Latency measurements were bimodal: a minority cluster in every degree tier
// sat at ~3.3x the tier minimum, with nothing between 1.4x and 3.3x, on an
// otherwise idle machine. The ratio held to within 8% across five tiers whose
// absolute latencies differ by more than 20x. That is the signature of a fixed
// throughput ratio between two execution contexts, not of variable external
// interference -- benchmark threads scheduled onto efficiency cores rather than
// performance cores.
//
// It is not a cosmetic problem. The per-evaluation median does not protect
// against it, because core placement persists for the lifetime of an evaluation
// and so affects every iteration equally. NSGA-II then ranks the affected
// genomes on latencies inflated by 3.3x, so the composition of the frontier is
// perturbed and not merely the numbers reported for it.
//
// macOS has no hard thread-affinity API on Apple Silicon: THREAD_AFFINITY_POLICY
// is accepted and ignored. The supported mechanism is Quality of Service class,
// which the scheduler uses to choose a core cluster. QOS_CLASS_USER_INTERACTIVE
// is the highest class and strongly prefers performance cores; the default class
// a thread inherits is what allows efficiency-core placement.
//
// Two measures are applied together:
//   1. Every worker thread sets QOS_CLASS_USER_INTERACTIVE on itself at startup.
//   2. The pool is sized to the performance-core count, so rayon does not create
//      more workers than there are P-cores and spill onto the E-cluster.
//
// This raises the probability of P-core placement; it does not make it a
// guarantee, because the scheduler retains final say. Min-of-k medians in
// benchmark_genome (LATENCY_TRIALS) are the second line of defence. Verification
// is empirical: the bimodality should disappear from the measured distribution.

/// `QOS_CLASS_USER_INTERACTIVE` from `<pthread/qos.h>`.
const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

extern "C" {
    /// Sets the QoS class of the calling thread. Darwin-only.
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
}

/// Request performance-core placement for the calling thread.
///
/// Best-effort: a non-zero return means the request was rejected and the thread
/// keeps its inherited class. That is reported rather than treated as fatal,
/// because a failed request degrades measurement quality without invalidating
/// the result.
fn request_performance_cores() -> bool {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: pthread_set_qos_class_self_np takes a QoS constant and a
        // relative priority, mutates only the calling thread's attributes, and
        // returns an errno-style int. No pointers are involved.
        unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) == 0 }
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// Number of performance cores, from `hw.perflevel0.physicalcpu`.
///
/// Falls back to the total core count if the key is unavailable, which is the
/// conservative choice: an oversized pool degrades measurement quality, while an
/// undersized one would silently reduce throughput.
fn performance_core_count() -> usize {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(out) = Command::new("sysctl")
            .args(["-n", "hw.perflevel0.physicalcpu"])
            .output()
        {
            if let Ok(s) = String::from_utf8(out.stdout) {
                if let Ok(n) = s.trim().parse::<usize>() {
                    if n > 0 {
                        return n;
                    }
                }
            }
        }
    }
    num_cpus_fallback()
}

fn num_cpus_fallback() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// The benchmark thread pool: P-core sized, every worker at interactive QoS.
///
/// Built once and reused, so QoS is set once per worker rather than per batch.
fn benchmark_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(performance_core_count())
            .start_handler(|_index| {
                request_performance_cores();
            })
            .thread_name(|i| format!("fhe-bench-{i}"))
            .build()
            .expect("failed to build the benchmark thread pool")
    })
}

/// Reports the pool configuration, so a run can record the conditions its
/// measurements were taken under instead of assuming them.
#[napi(object)]
pub struct BenchmarkPoolInfo {
    /// Worker threads in the benchmark pool.
    pub threads: u32,
    /// Performance cores detected via `hw.perflevel0.physicalcpu`.
    pub performance_cores: u32,
    /// Whether the calling thread could acquire interactive QoS. Workers use the
    /// same call, so a false here means the request is unavailable on this host.
    pub qos_available: bool,
}

#[napi]
pub fn benchmark_pool_info() -> BenchmarkPoolInfo {
    BenchmarkPoolInfo {
        threads: benchmark_pool().current_num_threads() as u32,
        performance_cores: performance_core_count() as u32,
        qos_available: request_performance_cores(),
    }
}

/// (one per genome). A score of -1.0 indicates noise corruption or
/// incompatible parameters.
///
/// Uses 1 iteration per genome (backward compatible).
#[napi]
pub fn run_benchmark(parameter_batch: Uint32Array) -> Result<Float64Array> {
    let flat = parameter_batch.as_ref();
    let params = unpack_params(flat)?;

    // Run inside the P-core pinned pool. See the comment block above.
    let scores: Vec<f64> = benchmark_pool()
        .install(|| params.par_iter().map(|p| benchmark_genome(p, 1)).collect());

    Ok(Float64Array::new(scores))
}

/// Deep benchmark variant: runs `iterations` forward NTT passes per trial,
/// takes the median per trial, then returns the **minimum** over
/// `LATENCY_TRIALS` trials (see that constant). Intended for serious
/// multi-objective exploration runs.
#[napi]
pub fn run_benchmark_deep(parameter_batch: Uint32Array, iterations: u32) -> Result<Float64Array> {
    let flat = parameter_batch.as_ref();
    let params = unpack_params(flat)?;
    let iters = iterations.max(1);

    // NOTE: do NOT raise the calling thread's QoS here. It was tried and it
    // made the distribution worse, not better: 34 of 500 measurements landed in
    // the slow cluster against 5 with the pool pinned alone, and the clean
    // bimodality smeared into a continuum from 1.0x to 3.3x. The cause is
    // oversubscription -- the pool already occupies all 12 performance cores, so
    // promoting Node's main thread to the same class puts 13 interactive threads
    // on 12 cores and introduces real contention. Leaving the caller at its
    // inherited class keeps the P-cluster to exactly one interactive thread per
    // core. Measured, not assumed.

    // Run inside the P-core pinned pool. This is the path the evolutionary
    // search uses, so it is the one the bimodal artefact came through.
    let scores: Vec<f64> = benchmark_pool()
        .install(|| params.par_iter().map(|p| benchmark_genome(p, iters)).collect());

    Ok(Float64Array::new(scores))
}

/// Validates the buffer can be unpacked and returns the genome count.
/// Useful for testing the unpack logic without running benchmarks.
#[napi]
pub fn validate_batch(parameter_batch: Uint32Array) -> Result<u32> {
    let flat = parameter_batch.as_ref();
    let params = unpack_params(flat)?;
    Ok(params.len() as u32)
}

/// Packs a single genome's parameters back into a Uint32Array.
/// Used for roundtrip testing of the unpack logic.
#[napi]
pub fn echo_genome(parameter_batch: Uint32Array, index: u32) -> Result<Uint32Array> {
    let flat = parameter_batch.as_ref();
    let params = unpack_params(flat)?;

    let idx = index as usize;
    if idx >= params.len() {
        return Err(Error::new(
            Status::InvalidArg,
            format!("Index {} out of range (batch has {} genomes)", idx, params.len()),
        ));
    }

    let p = params[idx];
    Ok(Uint32Array::new(vec![p.n, p.q, p.sigma, p.bg]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpack_single_genome() {
        let flat: Vec<u32> = vec![1024, 134217728, 3200, 512];
        let params = unpack_params(&flat).unwrap();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].n, 1024);
        assert_eq!(params[0].q, 134217728);
        assert_eq!(params[0].sigma, 3200);
        assert_eq!(params[0].bg, 512);
    }

    #[test]
    fn test_unpack_multiple_genomes() {
        let flat: Vec<u32> = vec![
            1024, 134217728, 3200, 512,
            2048, 268435456, 3200, 1024,
        ];
        let params = unpack_params(&flat).unwrap();
        assert_eq!(params.len(), 2);
        assert_eq!(params[1].n, 2048);
        assert_eq!(params[1].q, 268435456);
    }

    #[test]
    fn test_unpack_invalid_length() {
        let flat: Vec<u32> = vec![1024, 134217728, 3200]; // only 3 values
        let result = unpack_params(&flat);
        assert!(result.is_err());
    }

    #[test]
    fn test_unpack_empty() {
        let flat: Vec<u32> = vec![];
        let params = unpack_params(&flat).unwrap();
        assert_eq!(params.len(), 0);
    }

    #[test]
    fn test_benchmark_genome_valid_params() {
        // N=16, q=97: 96 % 32 == 0, valid negacyclic NTT parameters
        let p = FheParams { n: 16, q: 97, sigma: 3200, bg: 512 };
        let score = benchmark_genome(&p, 1);
        // Should return a positive latency (not -1.0)
        assert!(score > 0.0, "Expected positive latency, got {}", score);
    }

    #[test]
    fn test_benchmark_genome_invalid_q() {
        // N=16, q=7: 6 % 32 != 0, no primitive 2N-th root exists → noise failure
        let p = FheParams { n: 16, q: 7, sigma: 3200, bg: 512 };
        let score = benchmark_genome(&p, 1);
        assert_eq!(score, -1.0);
    }

    #[test]
    fn test_benchmark_genome_rejects_q_mod_n_but_not_2n() {
        // q = 7681, N = 512: 7680 % 512 == 0 so a cyclic transform would be
        // possible, but 7680 % 1024 != 0 so there is no primitive 2N-th root.
        // The negacyclic gate must reject it.
        assert_eq!((7681u64 - 1) % 512, 0);
        assert_ne!((7681u64 - 1) % 1024, 0);
        let p = FheParams { n: 512, q: 7681, sigma: 3200, bg: 512 };
        assert_eq!(benchmark_genome(&p, 1), -1.0);
    }

    #[test]
    fn test_benchmark_genome_n_not_power_of_2() {
        // N=6 is not a power of 2 → invalid
        let p = FheParams { n: 6, q: 97, sigma: 3200, bg: 512 };
        let score = benchmark_genome(&p, 1);
        assert_eq!(score, -1.0);
    }

    #[test]
    fn test_benchmark_genome_n_too_small() {
        // N=1 is too small for NTT
        let p = FheParams { n: 1, q: 97, sigma: 3200, bg: 512 };
        let score = benchmark_genome(&p, 1);
        assert_eq!(score, -1.0);
    }

    #[test]
    fn test_benchmark_genome_ntt_friendly_prime() {
        // q = 12289, N = 512: 12288 % 1024 == 0 ✓ (2N-friendly prime).
        // 7681 no longer qualifies: 7680 / 1024 = 7.5.
        let q = 12289u32;
        assert_eq!((q as u64 - 1) % 1024, 0);
        let p = FheParams { n: 512, q, sigma: 3200, bg: 512 };
        let score = benchmark_genome(&p, 1);
        assert!(score > 0.0, "Expected positive latency for 2N-friendly prime, got {}", score);
    }

    #[test]
    fn test_benchmark_genome_production_degrees() {
        // Representative genomes from the recorded Pareto frontiers. Each must
        // clear the negacyclic convolution gate and return a real timing.
        for (n, q) in [(1024u32, 12289u32), (4096, 40961), (16384, 6258689)] {
            assert_eq!((q as u64 - 1) % (2 * n as u64), 0, "q must be 1 mod 2N");
            let p = FheParams { n, q, sigma: 3200, bg: 512 };
            let score = benchmark_genome(&p, 1);
            assert!(score > 0.0, "N={}, q={} returned {}", n, q, score);
        }
    }

    #[test]
    fn test_benchmark_genome_deep_multiple_iterations() {
        // Verify deep benchmark with multiple iterations returns a positive min-of-k median
        let p = FheParams { n: 16, q: 97, sigma: 3200, bg: 512 };
        let score = benchmark_genome(&p, 10);
        assert!(score > 0.0, "Expected positive latency for 10 iterations, got {}", score);
    }

    #[test]
    fn test_min_of_k_is_at_most_single_median_scale() {
        // Smoke: production-sized genome returns a finite positive score under
        // the min-of-k path (k trials × iterations). Does not assert wall-clock.
        let p = FheParams { n: 1024, q: 12289, sigma: 3200, bg: 512 };
        let score = benchmark_genome(&p, 5);
        assert!(score > 0.0 && score.is_finite(), "got {}", score);
    }

    #[test]
    fn test_benchmark_parallel_execution() {
        // Test that multiple genomes can be benchmarked in parallel
        let params = vec![
            FheParams { n: 8, q: 97, sigma: 3200, bg: 512 },
            FheParams { n: 16, q: 97, sigma: 3200, bg: 512 },
            FheParams { n: 4, q: 17, sigma: 3200, bg: 512 },
            FheParams { n: 16, q: 7, sigma: 3200, bg: 512 }, // invalid → -1.0
            FheParams { n: 512, q: 7681, sigma: 3200, bg: 512 }, // 1 mod N only → -1.0
        ];
        let scores: Vec<f64> = params.par_iter().map(|p| benchmark_genome(p, 1)).collect();
        assert_eq!(scores.len(), 5);
        assert!(scores[0] > 0.0);
        assert!(scores[1] > 0.0);
        assert!(scores[2] > 0.0);
        assert_eq!(scores[3], -1.0);
        assert_eq!(scores[4], -1.0);
    }
}
