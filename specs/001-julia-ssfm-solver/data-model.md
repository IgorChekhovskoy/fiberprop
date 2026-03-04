# Data Model: Julia SSFM Solver Integration

**Branch**: `001-julia-ssfm-solver` | **Date**: 2026-03-04

---

## Entities

### 1. Julia FFTW Plan Cache

Represents pre-computed FFT execution plans, stored in a module-level dictionary in the Julia module. Plans are created once per unique grid shape and reused across all simulation steps.

| Field | Type | Description |
|-------|------|-------------|
| key | `(Int, Int)` | Tuple `(n_cores, M)` identifying the grid shape |
| fwd | `FFTW.cFFTWPlan` | Forward FFT plan for `ComplexF64` arrays of shape `(M, n_cores)` |
| inv | `FFTW.ScaledPlan` | Inverse FFT plan (normalized) for same shape |

**Lifecycle**: Created on first call for a given `(n_cores, M)` shape; persists for the process lifetime. Module `__init__()` pre-warms with typical fiberprop shapes to avoid first-call latency.

**Validation rule**: `n_cores >= 1`, `M >= 2` and `M` is power of 2 (matches existing project convention for FFT-based solvers).

---

### 2. Solver Precomputed Arrays (existing, read-only from Julia perspective)

The Julia functions receive the following pre-computed arrays from the Python `Solver` object. These are already computed by `solver.py` before the step function is called — the Julia module reads them but never modifies them.

| Attribute | Shape | dtype | Description |
|-----------|-------|-------|-------------|
| `solver.gamma_h_half` | `(n_cores,)` | `float64` | Nonlinear coefficient × h/2 per core |
| `solver.g0_h_half` | `(n_cores,)` | `float64` | Gain coefficient × h/2 per core |
| `solver.exp_g0h_half` | `(n_cores,)` | `float64` | `exp(g0 × h/2)` per core |
| `solver.exp_2g0h_half` | `(n_cores,)` | `float64` | `exp(2 × g0 × h/2)` per core |
| `solver.eq.E_sat` | `(n_cores,)` | `float64` | Saturation energy per core |
| `solver.D` | `(n_cores, n_cores, M)` or `(n_cores, n_cores)` | `complex128` | Dispersion/coupling operator |
| `solver.has_beta` | `bool` | — | Whether frequency-dependent dispersion is active |
| `solver.taper` | `(M,)` or `None` | `float64` | Absorbing boundary taper mask |

---

### 3. Simulation State Array `psi`

The primary evolving quantity: the complex optical field across all fiber cores and time grid points.

| Field | Type | Description |
|-------|------|-------------|
| shape | `(n_cores, M)` | n_cores fiber cores, M time grid points |
| dtype | `np.complex128` / `ComplexF64` | Double-precision complex |
| memory layout | C-contiguous (row-major) | NumPy default; requires `permutedims` at Julia boundary |
| mutation | In-place | `ssfm_order2_ndn_julia` modifies `psi` in-place and returns it |

---

### 4. Energy State Array `current_energy`

Per-core optical energy, updated during the nonlinear step for cores with active gain.

| Field | Type | Description |
|-------|------|-------------|
| shape | `(n_cores,)` | One energy value per fiber core |
| dtype | `np.float64` | Real-valued |
| mutation | In-place | Updated during the first nonlinear half-step for gain cores |

---

## State Transitions

### Julia Module Lifecycle

```
[Not loaded]
    │  first call to ssfm_order2_ndn_julia()
    ▼
[Initializing]  ─── juliacall import, Julia runtime start
    │              JULIA_NUM_THREADS must already be set
    ▼
[Julia running]  ─── FiberpropSSFM module loaded
    │              __init__() runs: FFTW plans created for default shapes
    ▼
[Warmed up]  ─── All hot functions compiled for ComplexF64
    │              First real simulation step executes
    ▼
[Steady state]  ─── Subsequent calls use cached plans and compiled specializations
```

### Per-Step Data Flow

```
Python: psi (n_cores, M), current_energy (n_cores,)
    │
    │  juliacall zero-copy pass
    ▼
Julia: PyArray{ComplexF64,2} received
    │
    │  Array() copy → native Julia Matrix{ComplexF64} (M, n_cores)
    ▼
½ Nonlinear step (in-place on Julia array)
    │
    │  (optional) Absorbing boundary (taper applied)
    ▼
Linear step: FFT → multiply D → IFFT (in-place, FFTW plan)
    │
    │  (optional) Absorbing boundary
    ▼
½ Nonlinear step (in-place on Julia array)
    │
    │  (optional) Additive noise
    ▼
    │  psi_py .= result (write-back to Python buffer)
    ▼
Python: psi updated in-place
```

---

## Axis Convention

A critical implementation detail to prevent silent correctness bugs:

| Context | Array shape | Axis 0 | Axis 1 |
|---------|------------|--------|--------|
| NumPy (Python) | `(n_cores, M)` | fiber cores | time grid |
| Julia column-major | `(M, n_cores)` | time grid | fiber cores |
| FFT target axis | Python axis=1 | = Julia dim 1 | ✓ consistent |

The dispersion operator `D` in the `has_beta=True` case has shape `(n_cores, n_cores, M)` in Python. In Julia this must be transposed to `(M, n_cores, n_cores)` for correct einsum-equivalent application.
