# Quickstart: Julia SSFM Backend

**Feature**: Julia-backed `ssfm_order2_ndn` solver
**Date**: 2026-03-04

---

## Prerequisites

1. **Python dependencies** (add to project `requirements.txt`):
   ```
   juliacall>=0.9
   juliapkg>=0.1
   ```

2. **Julia** (optional — `juliapkg` auto-downloads if not present):
   - For manual control: install Julia 1.9+ from https://julialang.org/downloads/
   - Set `JULIA=/path/to/julia` env var to use a specific binary.

3. **Julia packages** (auto-installed via `juliapkg.json` on first run):
   - `FFTW.jl`
   - `PythonCall.jl`

---

## Benchmark Results (N=100, M=1024, n_cores=1, 3 runs)

| Method | Median (s) | First run (s) | Speedup |
|--------|-----------|--------------|---------|
| `ssfm_order2_ndn` (Python/Numba) | 0.016 | 0.018 | — |
| `ssfm_order2_ndn_julia` (Julia) | 0.032 | 20.142 | 0.51x |

**Notes**:
- First Julia call includes JIT compilation (~20s). Subsequent calls in the same process are fast.
- For single-core small grids the Python/Numba baseline is faster due to Python↔Julia array marshalling overhead.
- Julia advantage is expected to grow with larger grids and multi-core configurations.

---

## First-Time Setup

No manual Julia setup is required. On the first call with `method="ssfm_order2_ndn_julia"`:
1. `juliapkg` downloads/locates Julia
2. Julia packages are installed into the project's Julia environment (`julia/`)
3. The `FiberpropSSFM` module is compiled (JIT warmup: ~5–30s first time, fast on subsequent runs)

**Note**: Set `JULIA_NUM_THREADS` before running to control parallelism:
```bash
export JULIA_NUM_THREADS=4   # or "auto" for all cores
python your_script.py
```

---

## Usage

Change only the `method` parameter — everything else remains identical:

```python
from fiberprop.solver import Solver, ComputationalParameters, EquationParameters
from fiberprop.fiber_geometry import CoreConfig

com = ComputationalParameters(
    N=1000, M=1024,
    L1=0.0, L2=1.0,
    T1=-50.0, T2=50.0,
    method="ssfm_order2_ndn_julia",   # ← only change needed
)

eq = EquationParameters(
    core_configuration=CoreConfig.single,
    beta2=-20.0,
    gamma=1.0,
)

solver = Solver(com=com, eq=eq)
solver.run_numerical_simulation()
```

---

## Switching Between Backends

| `method` | Backend | When to use |
|----------|---------|-------------|
| `"ssfm_order2_ndn"` | Python/Numba (CPU) | Default; always available |
| `"ssfm_order2_ndn_julia"` | Julia (CPU) | Benchmarking; Julia available |
| `"ssfm_order2_ndn"` + `use_torch=True` | PyTorch (GPU) | GPU available |

---

## Troubleshooting

**Julia not found**:
```
RuntimeError: Julia backend not available: juliacall package not found.
Install with: pip install juliacall
```
→ Run `pip install juliacall juliapkg`.

**First call is slow** (~20s):
This is Julia JIT compilation (~20s measured). Subsequent calls in the same process are fast. Re-running the same script starts compilation again unless precompile caches are available.

**Wrong results** (numerical mismatch):
Run with `method="ssfm_order2_ndn"` (Python baseline) and compare. If results differ by more than 1e-10 (relative L2), file a bug with the parameter set.
