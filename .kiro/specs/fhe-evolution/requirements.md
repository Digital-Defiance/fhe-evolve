# Requirements Document: FHE Parameter Evolution Simulation

## Introduction
The goal of this project is to build an evolutionary algorithm to discover optimal, hyper-efficient cryptographic parameters for Fully Homomorphic Encryption (FHE). The simulation will use raw GPU compute to test mutated parameter sets against security and noise estimators, isolating the fastest configurations that maintain 128-bit security.

## User Stories & Acceptance Criteria

**User Story 1: Parameter Generation**
As a simulation orchestrator, I want to generate and mutate FHE parameters so that I can explore the cryptographic search space.
* WHEN the simulation initializes, THE SYSTEM SHALL generate a seed population of known-safe FHE parameters (N, q, σ, Bg).
* WHEN a generation completes, THE SYSTEM SHALL mutate the top 5% of passing parameters to create the next generation.

**User Story 2: Security & Noise Estimation (The Guillotine)**
As a security gatekeeper, I want to filter out weak parameters before they are executed on the GPU so that I don't waste compute on insecure configurations.
* WHEN a new parameter set is generated, THE SYSTEM SHALL evaluate it against a 128-bit security floor using a localized lattice estimator.
* WHEN a parameter set fails the security threshold, THE SYSTEM SHALL discard the genome immediately.
* WHEN a parameter set passes the security threshold, THE SYSTEM SHALL calculate its theoretical noise growth budget.

**User Story 3: Micro-Benchmark Execution (The Burn)**
As an execution sandbox, I want to run micro-benchmarks on secure parameter sets using bare-metal hardware acceleration so that I can score their real-world latency.
* WHEN a batch of secure parameters is ready, THE SYSTEM SHALL pass them across the FFI boundary from Node.js to Rust as a flat array of 32-bit integers.
* WHEN the Rust sandbox receives the batch, THE SYSTEM SHALL execute a Number Theoretic Transform (NTT) micro-benchmark using a Rayon thread pool for maximum GPU utilization.
* WHEN the micro-benchmark completes, THE SYSTEM SHALL return an array of execution times (f64 floats) back to Node.js.
* WHEN a parameter set causes data corruption due to noise during execution, THE SYSTEM SHALL return a score of -1.0.
