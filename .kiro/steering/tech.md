# Technology Stack & Environment Constraints

## Hardware Environment (CRITICAL)
This project is built and optimized exclusively for **Apple Silicon (M4 Max)**.
- **CPU/GPU:** M4 Max with 16-core CPU, 40-core GPU, and a 16-core Neural Engine.
- **Memory:** 64GB Unified Memory with massive bandwidth.
- **Storage:** 4TB SSD (extreme I/O throughput).

**Constraint:** Do not suggest x86-specific optimizations (like AVX instructions). All hardware acceleration MUST target Apple Silicon architectures.

## Core Stack
- **Orchestration / Runtime:** Node.js (Latest LTS).
- **Execution / Acceleration:** Rust (Latest stable).
- **FFI Boundary:** `napi-rs` (Strictly use for Node <-> Rust interop).
- **Parallelization:** `rayon` (Rust crate for maximal CPU thread utilization).

## Existing Acceleration Libraries
This project relies on custom, pre-built hardware acceleration libraries. When suggesting cryptographic or mathematical operations, default to these existing primitives:

1.  **`@digitaldefiance/node-rs-accelerate`**
    - High-performance Reed-Solomon error correction and mathematical primitives.
    - Optimized for M1/M2/M3/M4.
    - Used for: Bulk data processing, erasure coding, fast Galois field math.

2.  **`@digitaldefiance/node-fhe-accelerate`**
    - Hardware-accelerated Fully Homomorphic Encryption (FHE) with Zero Knowledge Encryption.
    - M4 optimized lattice cryptography.
    - Used for: All core FHE operations, polynomial multiplications, and parameter benchmarking.

3.  **`@digitaldefiance/node-accelerate`**
    - High-performance Apple Accelerate framework bindings for Node.js.
    - Used for: Matrix operations, vector math, and vDSP/vForce/BLAS routing.

## Development Patterns
- **FFI Optimization:** The Node <-> Rust boundary is the primary bottleneck. Data passed across FFI must be restricted to raw primitive arrays (e.g., `Uint32Array`, `Float64Array`). Do not serialize complex JSON objects across the boundary.
- **Compute Strategy:** Keep state management and asynchronous orchestration in Node.js. Push heavy, synchronous, mathematical batch processing into Rust using Rayon thread pools.