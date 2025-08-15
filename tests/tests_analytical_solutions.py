"""tests_analytical_solutions.py
==============================================================
Light‑weight convergence tests for **fiberprop** using the *original* analytic
pulses from `pulses.py`:

* **fundamental_soliton**  – classic NLSE soliton (only β₂<0 and Kerr γ≠0).
* **gain_loss_soliton**    – NLSE with small linear gain / loss.

For **each** pulse we perform two sweeps on *nested grids*
---------------------------------------------------------
1. **Temporal sweep** – refine the number of time points
   `M = 2^9 … 2^12` while keeping a very fine longitudinal grid.
   We expect _spectral_ (≈ exponential) convergence.
2. **Longitudinal sweep** – keep the finest `M` **fixed** and successively
   double `N = 50, 100, 200, 400` (thus halving the step `h`).
   The scheme should be **2‑nd order** in `z`.
   The observed order is Ωᵢ = log₂(errᵢ / errᵢ₊₁).

Running & debugging
-------------------
* CI / quick check :  `pytest -q tests_analytical_solutions.py`
* Interactive debug:  `DEBUG=1 python tests_analytical_solutions.py`
  – prints detailed tables, shows error plots (if Matplotlib is available)
    and leaves last `Solver` objects accessible for inspection in PyCharm.
"""
from __future__ import annotations

import os
from math import log2
from pathlib import Path

import numpy as np
import pytest

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

from fiberprop.solver import ComputationalParameters as CP, EquationParameters as EP, Solver, CoreConfig
from fiberprop.pulses import fundamental_soliton as FS, gain_loss_soliton as GLS

DEBUG = True # bool(int(os.getenv("DEBUG", "0")))
HERE  = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def calculate_error(num: np.ndarray, ana: np.ndarray, tau: float) -> float:
    # return np.sqrt(np.sum(np.abs(num - ana) ** 2) * tau)
    return np.max(np.abs(num - ana))


def make_solver(pulse: str, *, M: int, N: int) -> Solver:
    """Return a configured `Solver` instance for given grid sizes."""
    com = CP(N=N, M=M, L1=0.0, L2=1.0, T1=-30.0, T2=30.0, damp_length=0.0, method="ssfm_order2_dnd_compact_windowed")

    if pulse == "fundamental":
        # Classic NLSE – only dispersion & Kerr non‑linearity
        eq = EP(core_configuration=CoreConfig.empty_ring,
                size=3,
                beta2=-1.0,   # β₂ < 0
                gamma=1.0,
                # the rest are zeros by default
                coupling_coefficient=1.0,
                noise_amplitude=0.0)
        p_func   = FS
        p_kwargs = {}           # no extra parameters
    else:  # gain‑loss soliton
        eq = EP(core_configuration=CoreConfig.empty_ring,
                size=3,
                beta2=-1.0,
                gamma=1.0,
                alpha=0.1,
                g_0=0.4,
                E_sat=1.0,
                coupling_coefficient=1.0,
                noise_amplitude=0.0)
        p_func   = GLS
        p_kwargs = {}           # GLS also needs no extras

    sol = Solver(com, eq,
                 pulses=p_func,
                 stored_steps_count=2,
                 pulse_params_list=p_kwargs,
                 num_threads=6,
                 use_torch=False,
                 use_gpu=False,
                 display_debug_info=False)
    return sol

def _dbg_plot(t, num, ana, title):
    """Рисует Re/Im числ. и аналит. решения на общих осях.
       Активируется ТОЛЬКО при DEBUG и наличии matplotlib."""
    if not (DEBUG and HAVE_MPL):
        return
    plt.figure(figsize=(6, 3))
    plt.plot(t, np.abs(ana.real - num.real), 'C0', label='Re error')
    plt.plot(t, np.abs(ana.imag - num.imag), 'C1', label='Im error')
    #plt.plot(t, num.real, 'C0', label='Re num')
    #plt.plot(t, ana.real, 'C1', label='Re ana')
    #plt.plot(t, num.imag, 'C2', label='Im num')
    #plt.plot(t, ana.imag, 'C3', label='Im ana')
    # plt.plot(t, abs(num), 'C4', label='Abs num')
    # plt.plot(t, abs(ana), 'C5', label='Abs ana')
    plt.title(title)
    plt.legend(); plt.tight_layout(); plt.show()

# -----------------------------------------------------------------------------
# Grids (nested)
# -----------------------------------------------------------------------------
Ms = [2 ** k for k in (5, 6, 7, 8, 9, 10, 11, 12)]
Ns = [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]

# -----------------------------------------------------------------------------
# PyTest parametrisations
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("pulse", ["fundamental", "gain_loss"])
def test_temporal_convergence(pulse):
    errs, hs_record = [], []
    Nfine = Ns[-1]
    for M in Ms:
        sol = make_solver(pulse, M=M, N=Nfine)  # very fine z‑grid

        #sol.numerical_solution[0] = np.fft.fft(sol.numerical_solution[0], axis=1)
        #sol.run_numerical_simulation_in_frequency_domain()

        sol.run_numerical_simulation()

        tau = sol.com.tau
        z   = sol.com.L2
        if pulse == "fundamental":
            ana = FS(t=sol.t, z=z, beta2=sol.eq.beta2[0], gamma=sol.eq.gamma[0])
        else:
            ana = GLS(t=sol.t, z=z,
                       beta2=sol.eq.beta2[0], gamma=sol.eq.gamma[0],
                       E_sat=sol.eq.E_sat[0], alpha=sol.eq.alpha[0], g_0=sol.eq.g_0[0])
        # sol.numerical_solution[-1] = np.fft.ifft(sol.numerical_solution[-1], axis=1)
        errs.append(calculate_error(sol.numerical_solution[-1, 0], ana, tau))
        _dbg_plot(sol.t, sol.numerical_solution[-1, 0], ana, f"{pulse}  M={M} N={Nfine}")
    orders = [log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    if DEBUG:
        print(f"\n[pulse={pulse}]  Longitudinal sweep (vary M):")
        for M, e in zip(Ms, errs):
            print(f"  M={M:5d}  err={e:.3e}")
        print("  observed orders:", [f"{o:.2f}" for o in orders])
    # crude spectral check – each refinement should reduce error ≥ 8×
    assert any(r > 4 for r in orders), "temporal error does not decay fast enough"


@pytest.mark.parametrize("pulse", ["fundamental", "gain_loss"])
def test_longitudinal_order(pulse):
    errs = []
    Mfine = Ms[-1]
    for N in Ns:
        sol = make_solver(pulse, M=Mfine, N=N)

        #sol.numerical_solution[0] = np.fft.fft(sol.numerical_solution[0], axis=1)
        #sol.run_numerical_simulation_in_frequency_domain()

        sol.run_numerical_simulation()

        tau = sol.com.tau
        z   = sol.com.L2
        if pulse == "fundamental":
            ana = FS(t=sol.t, z=z, beta2=sol.eq.beta2[0], gamma=sol.eq.gamma[0])
        else:
            ana = GLS(t=sol.t, z=z,
                       beta2=sol.eq.beta2[0], gamma=sol.eq.gamma[0],
                       E_sat=sol.eq.E_sat[0], alpha=sol.eq.alpha[0], g_0=sol.eq.g_0[0])
        # sol.numerical_solution[-1] = np.fft.ifft(sol.numerical_solution[-1], axis=1)
        errs.append(calculate_error(sol.numerical_solution[-1, 0], ana, tau))
        _dbg_plot(sol.t, sol.numerical_solution[-1, 0], ana, f"{pulse}  M={Mfine} N={N}")
    orders = [log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    if DEBUG:
        print(f"\n[pulse={pulse}]  Longitudinal sweep (vary N):")
        for N, e in zip(Ns, errs):
            print(f"  N={N:5d}  h={1/N:7.4f}  err={e:.3e}")
        print("  observed orders:", [f"{o:.2f}" for o in orders])
    assert min(orders) > 1.7, f"observed order < 2 (min={min(orders):.2f})"

# -----------------------------------------------------------------------------
# Stand‑alone run (PyCharm‑friendly)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if not DEBUG:
        print("Set DEBUG=1 for verbose output. Running quick self‑check…")
    test_longitudinal_order("fundamental")
    test_longitudinal_order("gain_loss")
    test_temporal_convergence("fundamental")
    test_temporal_convergence("gain_loss")
    print("✔ All tests passed")
