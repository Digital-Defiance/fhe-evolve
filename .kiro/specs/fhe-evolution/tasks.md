### Kiro `tasks.md` (The "Steps")

```markdown
# Implementation Plan: FHE Parameter Evolution Simulation

- [x] **Task 1: Setup Kiro Steering & Context**
  - Create `.kiro/steering/tech.md` defining the use of Node.js, Rust (with napi-rs for FFI), and Rayon for threading.
  - Link the existing `node-fhe-accelerate` and `node-rs-accelerate` repositories as context.

- [x] **Task 2: Build the Evolutionary Orchestrator (Node.js)**
  - Implement a `Genome` class to represent the $N$, $q$, $\sigma$, and $B_g$ parameters.
  - Implement the `PopulationManager` to handle seeding, mutation logic, and crossover.

- [x] **Task 3: Integrate the Security Sieve**
  - Implement a child process wrapper for a lattice security estimator (or compile a lightweight WASM version).
  - Create the "Guillotine" function to filter out genomes below 128-bit security.

- [x] **Task 4: Define the FFI Boundary**
  - Implement the primitive array packing in Node.js (`Uint32Array`).
  - Implement the primitive array unpacking in Rust (`napi-rs`).

- [x] **Task 5: Build the Execution Sandbox (Rust)**
  - Implement the dynamic parameter initialization in the Rust FHE context.
  - Implement the micro-benchmark loop (e.g., a batch of NTT operations).
  - Implement Rayon threading to execute the batch in parallel.

- [x] **Task 6: The Burn Loop**
  - Connect the Orchestrator, the Sieve, and the Sandbox into a continuous generation loop.
  - Implement logging to track the fastest configurations found over time.

