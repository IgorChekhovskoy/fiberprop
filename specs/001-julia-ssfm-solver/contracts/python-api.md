# Public API Contract: ssfm_julia.py

**Module**: `fiberprop/ssfm_julia.py`
**Date**: 2026-03-04

This contract defines the public Python interface exposed by the Julia backend wrapper. It must remain compatible with the calling patterns in `solver.py`.

---

## Function: `ssfm_order2_ndn_julia`

```python
def ssfm_order2_ndn_julia(
    psi: np.ndarray,           # shape (n_cores, M), dtype complex128, C-contiguous
    current_energy: np.ndarray, # shape (n_cores,), dtype float64
    solver,                     # Solver instance (read-only access to precomputed arrays)
    h: float,                   # propagation step size [m]
    tau: float,                 # time grid spacing [ps]
    damp_length: float = 0.0,   # absorbing boundary fraction (0 = disabled)
    noise_amplitude: float = 0.0  # additive noise amplitude (0 = disabled)
) -> np.ndarray                 # psi updated in-place; same object returned
```

**Preconditions**:
- `psi.dtype == np.complex128`, `psi.flags['C_CONTIGUOUS'] == True`
- `current_energy.dtype == np.float64`, shape `(psi.shape[0],)`
- `solver.D` is set (not None) before this function is called — guaranteed by `solver.calculate_all_dispersion_matrices()`
- Julia runtime must be initialized (first call triggers lazy initialization)

**Postconditions**:
- `psi` is modified in-place to reflect one N-D-N split-step iteration
- `current_energy` is updated for cores where `solver.eq.g_0 != 0`
- Return value is the same `psi` object (for compatibility with the Python baseline)

**Side effects**:
- On first call: starts Julia runtime, loads `FiberpropSSFM` module, runs JIT warmup (~1–5s)
- Subsequent calls: no side effects beyond mutating `psi` and `current_energy`

**Error conditions**:
- `RuntimeError("Julia backend not available: ...")` if `juliacall` is not installed
- Propagates Julia exceptions as `juliacall.JuliaError` with descriptive message

---

## Function: `nonlinear_step_julia`

```python
def nonlinear_step_julia(
    psi: np.ndarray,           # shape (n_cores, M), dtype complex128, in-place
    gamma_h: np.ndarray,       # shape (n_cores,), dtype float64
    g0_h: np.ndarray,          # shape (n_cores,), dtype float64
    exp_g0h: np.ndarray,       # shape (n_cores,), dtype float64
    exp_2g0h: np.ndarray,      # shape (n_cores,), dtype float64
    E_sat: np.ndarray,         # shape (n_cores,), dtype float64
    energy_in: np.ndarray | None,  # shape (n_cores,), dtype float64; None if no gain
    P: np.ndarray | None = None    # shape (n_cores, M), precomputed |psi|²; None = compute
) -> None                      # in-place mutation, no return value
```

**Contract**: Identical behavior to `ssfm_mcf.nonlinear_step` for the same inputs.

---

## Function: `linear_step_julia`

```python
def linear_step_julia(
    psi: np.ndarray,           # shape (n_cores, M), dtype complex128
    has_beta: bool,            # True = FFT-based dispersion; False = matrix multiply only
    D: np.ndarray              # shape (n_cores, n_cores, M) or (n_cores, n_cores)
) -> np.ndarray                # new array with linear step applied (not in-place)
```

**Contract**: Identical behavior to `ssfm_mcf.linear_step` for the same inputs.

---

## Module-Level Behavior

```python
# Safe to import even without Julia installed:
from fiberprop.ssfm_julia import ssfm_order2_ndn_julia  # succeeds always

# Julia is initialized lazily on first call:
result = ssfm_order2_ndn_julia(...)  # triggers Julia startup if not yet initialized

# If Julia is unavailable, raises at call time, not at import time:
# RuntimeError: "Julia backend not available: juliacall package not found.
#                Install with: pip install juliacall"
```

---

## Compatibility Contract with solver.py

The dispatch entry in `Solver.run_numerical_simulation()` calls:

```python
psi_next = ssfm_order2_ndn_julia(
    psi_next,
    self.energy[:, n],
    self,
    self.com.h,
    tau,
    self.com.damp_length,
    self.eq.noise_amplitude,
)
```

This is identical to the call site for `ssfm_order2_ndn` (Python baseline). No additional parameters or setup are required.
