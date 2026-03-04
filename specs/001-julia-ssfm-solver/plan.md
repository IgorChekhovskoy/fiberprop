# Implementation Plan: Julia SSFM Solver Integration

**Branch**: `001-julia-ssfm-solver` | **Date**: 2026-03-04 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-julia-ssfm-solver/spec.md`

## Summary

Implement a Julia-backed version of the `ssfm_order2_ndn` (N-D-N split-step) solver and expose it as `ssfm_order2_ndn_julia` in the fiberprop Python package. The implementation uses `juliacall`/`PythonCall.jl` for Python-Julia interop, `FFTW.jl` for FFT operations, and follows the established Python backend patterns (`ssfm_mcf.py` / `ssfm_mcf_pytorch.py`). The Julia method integrates into `solver.py`'s existing method dispatch without requiring any user code changes beyond setting `method="ssfm_order2_ndn_julia"`.

## Technical Context

**Language/Version**: Python 3.10+, Julia 1.9+
**Primary Dependencies**: `juliacall>=0.9`, `juliapkg>=0.1` (Python); `FFTW.jl v1`, `PythonCall.jl v0.9` (Julia)
**Storage**: N/A — no persistent state beyond Julia precompile cache
**Testing**: pytest (existing test suite + new `tests/tests_julia_ssfm.py`)
**Target Platform**: Linux (primary); macOS supported; Julia runs CPU only
**Project Type**: Scientific simulation library
**Performance Goals**: Wall-clock time ≤ Python/Numba baseline for typical simulation sizes; documented benchmark at 1000 steps × 4096 time points × 7 cores
**Constraints**: First-call JIT latency acceptable (< 30s); subsequent calls must not add overhead vs. baseline; zero regression on existing tests

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| **I. Scientific Accuracy** | ✅ PASS | Julia implementation must match Python baseline within 1e-10 relative error; validated by dedicated test suite against known analytical solutions |
| **II. Dual Backend Parity** | ⚠️ EXCEPTION (justified) | See Complexity Tracking below |
| **III. Modular Decomposition** | ✅ PASS | Julia backend lives in `fiberprop/ssfm_julia.py`; Julia source in `julia/`; no circular imports introduced |
| **IV. Test-Driven Validation** | ✅ PASS | `tests/tests_julia_ssfm.py` covers correctness, conservation laws, edge cases, and fallback behavior |
| **V. Reproducibility** | ✅ PASS | Julia CPU path is deterministic; FFTW plans are fixed per shape; random noise uses seeded RNG |

## Project Structure

### Documentation (this feature)

```text
specs/001-julia-ssfm-solver/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── python-api.md    # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
julia/
├── FiberpropSSFM.jl     # Julia module: nonlinear_step!, linear_step!, ssfm_order2_ndn!
├── Project.toml         # Julia dependency declarations (FFTW, PythonCall)
└── Manifest.toml        # Julia lockfile (auto-generated, committed for reproducibility)

fiberprop/
├── ssfm_julia.py        # NEW: Python wrapper; exposes ssfm_order2_ndn_julia,
│                        #      nonlinear_step_julia, linear_step_julia
├── juliapkg.json        # NEW: juliapkg dependency manifest (auto-activates Julia env)
├── ssfm_mcf.py          # MODIFIED: no changes to logic; ssfm_order2_ndn remains
├── solver.py            # MODIFIED: dispatch + calculate_all_dispersion_matrices
└── __init__.py          # MODIFIED: add ssfm_julia import (guarded)

tests/
└── tests_julia_ssfm.py  # NEW: unit + integration tests for Julia backend

requirements.txt         # MODIFIED: add juliacall>=0.9, juliapkg>=0.1
```

**Structure Decision**: Single-project layout matching the existing `fiberprop/` package structure. The `julia/` directory at repo root follows Julia convention for a package-embedded Julia environment. The Python wrapper `ssfm_julia.py` sits alongside `ssfm_mcf.py` and `ssfm_mcf_pytorch.py`, completing the backend trio.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|--------------------------------------|
| Principle II exception: Julia backend does not have a PyTorch equivalent | The Julia backend is an **alternative CPU implementation** of an existing method (`ssfm_order2_ndn`), not a new SSFM feature. Principle II applies to new physical capabilities requiring GPU acceleration. The existing `ssfm_order2_ndn_pytorch` is unaffected and still the GPU path. | A PyTorch-Julia bridge would require copying tensors to NumPy at every call boundary (GPU→CPU→Julia→CPU→GPU), negating all performance benefit. Scope of this feature would double. |

## Phase 0 Artifacts

- **research.md**: [Complete](research.md) — all design decisions resolved, no NEEDS CLARIFICATION remaining

## Phase 1 Design

### Julia Module: `julia/FiberpropSSFM.jl`

The Julia module exposes three functions to the Python bridge:

**`nonlinear_step!(psi, gamma_h, g0_h, exp_g0h, exp_2g0h, E_sat, energy_in, P)`**
- Replicates `ssfm_mcf.nonlinear_step` in Julia
- Handles both gain-free (`g0_h == 0`) and saturated gain cases
- Operates in-place on native Julia `Matrix{ComplexF64}`

**`linear_step!(psi, has_beta, D, plans)`**
- Replicates `ssfm_mcf.linear_step` in Julia
- `has_beta=true` branch: FFT → einsum-equivalent `D` multiply → IFFT using pre-planned FFTW transforms
- `has_beta=false` branch: matrix multiply `D * psi` (coupling-only case)

**`ssfm_order2_ndn!(psi_py, current_energy_py, gamma_h, g0_h, exp_g0h, exp_2g0h, E_sat, D, has_beta, taper, h, tau, damp_length, noise_amplitude)`**
- Composed N-D-N split step
- Receives Python arrays via PythonCall bridge; converts to native Julia at boundary
- Writes result back to Python buffer before returning

**Module `__init__()`**:
- Pre-creates FFTW plans for shapes `[(1,512), (7,1024), (19,2048)]`
- Pre-warms all hot functions via `precompile()` + dummy calls with `ComplexF64` arrays

### Python Wrapper: `fiberprop/ssfm_julia.py`

Pattern: identical to `ssfm_mcf_pytorch.py` — try/except import guard, stub functions if unavailable, full implementation otherwise.

```python
try:
    import juliacall
    # ... load FiberpropSSFM module
    _JULIA_AVAILABLE = True
except ImportError:
    _JULIA_AVAILABLE = False

def _need_julia():
    raise RuntimeError(
        "Julia backend not available: juliacall package not found. "
        "Install with: pip install juliacall juliapkg"
    )

if not _JULIA_AVAILABLE:
    def ssfm_order2_ndn_julia(*args, **kwargs): _need_julia()
    def nonlinear_step_julia(*args, **kwargs): _need_julia()
    def linear_step_julia(*args, **kwargs): _need_julia()
else:
    # ... full implementation calling Julia functions
```

**Lazy initialization**: `juliacall` import deferred to first call via a module-level `_julia_ready` flag. `JULIA_NUM_THREADS` is set via `os.environ.setdefault` to not override user settings.

### Solver Integration: `fiberprop/solver.py`

Two touchpoints:

1. **`calculate_all_dispersion_matrices()`**: Add `ssfm_order2_ndn_julia` to the `ndn` branch (same as `ssfm_order2_ndn`) since it uses the full-step `D` matrix:
   ```python
   if self.com.method in ("ssfm_order2_ndn", "ssfm_order2_ndn_windowed", "ssfm_order2_ndn_julia"):
       if self.D is None:
           self.calculate_D_matrix(self.com.h)
   ```

2. **Main loop dispatch**: Add an `elif` branch in `run_numerical_simulation()`:
   ```python
   elif self.com.method == "ssfm_order2_ndn_julia":
       psi_next = ssfm_order2_ndn_julia(
           psi_next, self.energy[:, n], self,
           self.com.h, tau, self.com.damp_length, self.eq.noise_amplitude,
       )
   ```

### Test Plan: `tests/tests_julia_ssfm.py`

Following **Principle IV**, tests cover:

| Test | What it validates |
|------|------------------|
| `test_julia_linear_only` | Single linear step with known analytical solution (Gaussian pulse, pure dispersion) |
| `test_julia_nonlinear_only` | Single nonlinear step with known analytical phase rotation |
| `test_julia_vs_python_baseline` | Full N-step simulation: Julia vs. Python, relative L2 error < 1e-10 |
| `test_julia_with_gain` | Saturated gain case (`g_0 != 0`): energy conservation verified |
| `test_julia_with_noise` | Noise amplitude > 0: output shape correct, not equal to noiseless |
| `test_julia_with_damping` | Absorbing boundary: energy decreases monotonically |
| `test_julia_fallback_error` | Without juliacall installed: `RuntimeError` raised at call time, not import time |
| `test_julia_single_core` | Edge case: n_cores = 1 |
| `test_julia_no_dispersion` | Edge case: `has_beta = False`, coupling-only linear step |
| `test_julia_multicore` | 7-core MCF: full simulation, energy conserved in lossless regime |

## Phase 1 Artifacts

- [data-model.md](data-model.md) — array shapes, axis conventions, module lifecycle
- [contracts/python-api.md](contracts/python-api.md) — Python API contract for `ssfm_julia.py`
- [quickstart.md](quickstart.md) — user-facing setup and usage guide
