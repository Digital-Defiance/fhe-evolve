# FHE Parameter Evolution

Multi-objective evolutionary optimization for discovering optimal Fully Homomorphic Encryption parameters on Apple Silicon.

## What This Does

FHE parameter selection traditionally relies on conservative theoretical bounds (HE Standard 2018) that ignore real hardware performance characteristics. This project uses an NSGA-II evolutionary algorithm with hardware-in-the-loop NTT benchmarking to discover Pareto-optimal FHE parameter sets on Apple M4 Max.

The system simultaneously optimizes three objectives:
- **Latency** — NTT execution time (lower is better)
- **Security** — Estimated security level in bits (higher is better)
- **Noise budget** — Available homomorphic computation depth (higher is better)

The result is a Pareto frontier of non-dominated parameter configurations per polynomial degree (N = 1024, 2048, 4096, 8192, 16384).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Node.js Orchestrator                                           │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │ NSGA-II Engine │→│Security Sieve│→│ Population Manager   │  │
│  └───────────────┘  └──────────────┘  └─────────────────────┘  │
│         │                                        │              │
│         ▼                                        ▼              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Rust FFI (napi-rs style raw dylib)                     │    │
│  │  ┌─────────────────┐  ┌───────────────────────────────┐ │    │
│  │  │ Montgomery NTT  │  │ NEON-accelerated butterfly ops│ │    │
│  │  └─────────────────┘  └───────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Pareto Frontier (JSON output)                          │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Build native Rust module (requires Rust toolchain)
npm run build:native

# Run full Pareto exploration (all N tiers, ~hours)
npm run start

# Quick single-objective test (~minutes)
npm run start:quick

# Validate results and generate analysis
npm run validate

# Run tests
npm test
```

## Results

After a run completes, the `results/` directory contains:

| File | Description |
|------|-------------|
| `pareto-frontier.json` | All Pareto-optimal points across all tiers |
| `pareto-N{1024..16384}.json` | Per-tier Pareto fronts |
| `convergence.json` | Per-generation convergence statistics |
| `summary.txt` | Human-readable summary with HE Standard comparison |
| `fhe-parameters.json` | Per-point `searched` / `measured` / `illustrativeMapping` blocks. The `node-fhe-accelerate` CustomParameters mapping is an illustrative projection, not a usable parameter set |
| `validation-report.txt` | Full validation report with performance estimates |

Open `viz/index.html` in a browser to interactively explore the 3D Pareto frontier.

## Methodology

### Multi-Objective Optimization

Standard parameter selection picks a single "safe" configuration from lookup tables. We instead explore the full tradeoff space using NSGA-II (Non-dominated Sorting Genetic Algorithm II):

1. **Population seeding** — Random FHE parameter sets constrained to valid ranges per N tier
2. **Security sieve** — Reject any configuration below 128-bit security (lattice estimator)
3. **Hardware benchmarking** — Run actual NTT transforms via Rust FFI to measure real latency
4. **Non-dominated sorting** — Rank by Pareto dominance across all three objectives
5. **Crowding distance** — Maintain diversity along the frontier
6. **Selection + mutation** — Breed next generation from elite survivors

### Pareto Dominance

Point A dominates point B if A is at least as good as B in all objectives and strictly better in at least one. The Pareto front is the set of all non-dominated points — no single parameter can be improved without degrading another.

### NTT as FHE Proxy

The Number Theoretic Transform is the computational bottleneck of all lattice-based FHE schemes. NTT latency for a given (N, q) pair is a reliable proxy for overall FHE operation performance. We benchmark NTT directly rather than full encryption/decryption cycles to isolate the hardware-sensitive component.

## Hardware

| Component | Specification |
|-----------|--------------|
| Chip | Apple M4 Max |
| CPU | 16 cores (12P + 4E) |
| GPU | 40-core |
| Neural Engine | 16-core |
| Memory | 64GB Unified (546 GB/s bandwidth) |
| Storage | 4TB NVMe SSD |

Hardware-aware parameter selection matters because:
- NTT performance varies significantly with polynomial degree and modulus size
- Apple Silicon's unified memory architecture enables zero-copy FFI between Node.js and Rust
- NEON SIMD and potential AMX/SME acceleration can shift optimal parameter boundaries
- Memory bandwidth characteristics favor different N/q combinations than x86 platforms

## Comparison to HE Standard 2018

The Homomorphic Encryption Standard (2018) provides conservative parameter recommendations for 128-bit security:

| N | Max log₂(q) | Security |
|---|-------------|----------|
| 1024 | 27 | 128-bit |
| 2048 | 54 | 128-bit |
| 4096 | 109 | 128-bit |
| 8192 | 218 | 128-bit |
| 16384 | 438 | 128-bit |

Our evolved parameters are validated against these bounds. Points within the standard's limits maintain guaranteed 128-bit security. Points exceeding the standard's log₂(q) limit may still be secure (the standard is conservative), but are flagged in the validation report.

## Project Structure

```
fhe-evolve/
├── src/
│   ├── main.ts                 # Entry point (Pareto mode + quick mode)
│   ├── pareto-explorer.ts      # NSGA-II multi-tier exploration engine
│   ├── pareto-output.ts        # Result serialization & summary generation
│   ├── validate-pareto.ts      # Post-run validation & parameter mapping
│   ├── nsga2.ts                # Non-dominated sorting + crowding distance
│   ├── population-manager.ts   # Genome breeding, crossover, mutation
│   ├── genome.ts               # FHE parameter genome representation
│   ├── guillotine.ts           # Security sieve (lattice estimator)
│   ├── security-estimator.ts   # Security level estimation
│   ├── burn-loop.ts            # Legacy single-objective evolution loop
│   ├── ffi.ts                  # Rust native module FFI bindings
│   ├── logger.ts               # Leaderboard tracking
│   ├── types.ts                # Shared type definitions
│   └── index.ts                # Library exports
├── native/
│   └── src/
│       ├── lib.rs              # Rust FFI entry point
│       ├── ntt.rs              # Number Theoretic Transform
│       ├── ntt_accel.rs        # NEON-accelerated NTT
│       └── montgomery.rs       # Montgomery modular arithmetic
├── viz/
│   └── index.html              # Interactive 3D Pareto visualization
├── results/                    # Output directory (generated)
├── package.json
├── tsconfig.json
└── vitest.config.ts
```

## License

MIT
