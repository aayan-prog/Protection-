# 🚀 High-Performance Kyber Forward NTT Engine (AVX2-Optimized)

A battle-tested, high-performance implementation of the **Forward Number Theoretic Transform (NTT)** for the Crystals-Kyber Post-Quantum Cryptography algorithm. This engine leverages architecture-specific Intel/AMD **AVX2 SIMD intrinsics** combined with a branchless, vector-aligned **Barrett Reduction** pipeline to achieve blistering speeds.

---

## 🏎️ Performance & Benchmarks

The core layout eliminates dynamic memory allocations and heavy branch prediction overheads by strictly binding computing tasks directly to 256-bit hardware vector registers.

### 📊 Micro-Benchmark Reports (10,000 Iterations)

| Environment / Architecture | Optimization Profile | Execution Time (10k Rounds) | Single NTT Latency | Status |
| :--- | :--- | :--- | :--- | :--- |
| **GitHub Actions CI Runner** | Shared Virtual Core (`+avx2`) | ~23.64 ms | ~2.36 $\mu s$ | Passed ✅ |
| **Google Colab Cloud Core** | Dedicated Server vCPU (`target-cpu=native`) | **11.21 ms** 🚀 | **1.12 $\mu s$** | **Optimal** 🔥 |

---

## 🛠️ Key Architectural Optimizations

* **Zero-Branch Vector Loops:** Eliminated all loop-boundary conditions (`if` guards) within the inner computation layers. This prevents CPU pipeline stalls and minimizes branch misprediction penalties.
* **Vectorized Barrett Reduction:** Integrated `_mm256_mulhi_epi16` and vectorized bit-shifts to compute modular reductions directly on parallel lanes inside `_mm256_storeu_si256` boundaries.
* **Pure Stack Allocation:** The structure enforces a strict `#[repr(align(32))]` layout on the structural `Poly` polynomial array to ensure perfectly aligned, un-faulted AVX2 stream loads.

---

## 📂 Core Architecture Mapping

The code targets specific processing boundaries tailored dynamically according to standard Cooley-Tukey butterfly layers:

1. **Layers 1-3 (SIMD Lanes):** Processed via the high-throughput 256-bit AVX2 execution framework down to a vector width blocksize of 32 elements.
2. **Layers 4-7 (Scalar Fallback):** Handled sequentially by a scalar engine using precise modular parameters to compute standard remaining bounds.

---

## 🚀 Quick Start & Verification

### Prerequisites
* Rust stable toolchain installed.
* An x86_64 target CPU supporting the AVX2 instruction set.

### Run Benchmarks Locally
To compile with full machine-level optimizations and run the stopwatch integration pipeline, use:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release

INITIALIZING HIGH-PERFORMANCE AVX2 NTT ENGINE...
⚡ BENCHMARK COMPLETE: 10,000 NTT Executions took: 11.218216ms
📊 Sample Coefficients Output: [1737, 1737, 2965, 3019, 642]
