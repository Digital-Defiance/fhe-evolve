# Technical Design: FHE Parameter Evolution Simulation

## Architecture Overview
The system will be split across an orchestration layer (Node.js) and an execution layer (Rust via FFI).

1.  **The Orchestrator (Node.js):** Acts as the asynchronous brain of the genetic algorithm. It handles population generation, mutation, and interfaces with the security estimator.
2.  **The Sieve (Python/WASM Estimator):** A headless wrapper around `lattice-estimator` or a similar tool to evaluate the security and noise of the mutated parameters.
3.  **The Sandbox (Rust):** A highly parallelized execution environment that receives raw parameter structs, initializes the FHE context, and runs micro-benchmarks on the GPU.

## Data Flow
1. Node.js generates Generation $X$.
2. Node.js pipes Generation $X$ through the Security Sieve.
3. Secure genomes are packed into a flat `Uint32Array` and sent across the FFI boundary to Rust.
4. Rust unpacks the array, spins up a Rayon thread pool, and executes the micro-benchmarks.
5. Rust returns a `Float64Array` containing latency scores back to Node.js.
6. Node.js ranks the scores, culls the bottom 95%, and mutates the remaining 5% to create Generation $X+1$.

## Interfaces

### Node-to-Rust FFI Boundary
To minimize serialization overhead, the data contract must be strictly primitive arrays.

```typescript
// Input from Node.js
type ParameterBatch = Uint32Array; // [N, q, σ, Bg, N, q, σ, Bg...]

// Output from Rust
type ExecutionScores = Float64Array; // [latency_ms, latency_ms, ...] (-1.0 for noise failure)
