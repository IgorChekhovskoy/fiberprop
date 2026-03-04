"""
Tests for the Julia SSFM backend (fiberprop/ssfm_julia.py + julia/FiberpropSSFM.jl).

Coverage:
  Phase 3 (US1): correctness vs. Python baseline
  Phase 4 (US2): graceful fallback when juliacall is absent
  Phase 5 (US3): performance smoke test
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

_SSFM_JULIA_PATH = str(Path(__file__).resolve().parent.parent / "fiberprop" / "ssfm_julia.py")

from fiberprop.solver import ComputationalParameters as CP
from fiberprop.solver import CoreConfig, Solver
from fiberprop.solver import EquationParameters as EP
from fiberprop.ssfm_julia import (
    _JULIA_AVAILABLE,
    linear_step_julia,
    nonlinear_step_julia,
)
from fiberprop.ssfm_mcf import linear_step, nonlinear_step

# ── helpers ──────────────────────────────────────────────────────────────────


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 error ||a-b|| / ||b||."""
    denom = np.linalg.norm(b)
    if denom == 0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


def _make_solver(
    method: str,
    *,
    N: int = 20,
    M: int = 256,
    n_cores: int = 1,
    gamma: float = 1.0,
    beta2: float = -1.0,
    g_0: float = 0.0,
    coupling: float = 0.0,
    damp_length: float = 0.0,
    noise_amplitude: float = 0.0,
) -> Solver:
    # CoreConfig.not_set with explicit size lets us choose any n_cores freely.
    com = CP(
        N=N,
        M=M,
        L1=0.0,
        L2=0.5,
        T1=-20.0,
        T2=20.0,
        method=method,
        damp_length=damp_length,
    )
    eq = EP(
        core_configuration=CoreConfig.not_set,
        size=n_cores,
        beta2=beta2,
        gamma=gamma,
        g_0=g_0,
        E_sat=10.0,
        coupling_coefficient=coupling,
        noise_amplitude=noise_amplitude,
    )
    # Provide a sech initial condition so numerical_solution is properly allocated.
    t = np.linspace(-20.0, 20.0, M)
    psi0 = np.tile((1.0 / np.cosh(t)).astype(np.complex128), (n_cores, 1))
    return Solver(com=com, eq=eq, initial_condition=psi0)


def _sech_input(M: int, T1: float, T2: float) -> np.ndarray:
    t = np.linspace(T1, T2, M)
    return (1.0 / np.cosh(t)).astype(np.complex128)


# ── Phase 3: US1 correctness tests ───────────────────────────────────────────


@pytest.mark.skipif(not _JULIA_AVAILABLE, reason="juliacall not installed")
def test_linear_step_julia_vs_python():
    """linear_step_julia must match ssfm_mcf.linear_step to 1e-10 relative L2."""
    rng = np.random.default_rng(0)
    n_cores, M = 3, 128
    psi = (
        rng.standard_normal((n_cores, M)) + 1j * rng.standard_normal((n_cores, M))
    ).astype(np.complex128)

    # Build a diagonal dispersion matrix (has_beta=True, shape (n,n,M))
    D = np.zeros((n_cores, n_cores, M), dtype=np.complex128)
    omega = np.fft.fftfreq(M, d=0.1)
    for i in range(n_cores):
        D[i, i, :] = np.exp(-1j * 0.5 * (-1.0) * omega**2 * 0.01)

    py_result = linear_step(psi.copy(), True, D)
    jl_result = linear_step_julia(psi.copy(), True, D)

    assert _rel_l2(jl_result, py_result) < 1e-10, (
        f"linear_step rel L2 error = {_rel_l2(jl_result, py_result):.2e} (must be < 1e-10)"
    )


@pytest.mark.skipif(not _JULIA_AVAILABLE, reason="juliacall not installed")
def test_linear_step_julia_no_dispersion():
    """has_beta=False branch: coupling-only matrix multiply."""
    rng = np.random.default_rng(1)
    n_cores, M = 2, 64
    psi = (
        rng.standard_normal((n_cores, M)) + 1j * rng.standard_normal((n_cores, M))
    ).astype(np.complex128)
    theta = np.pi / 6
    D = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.complex128,
    )

    py_result = linear_step(psi.copy(), False, D)
    jl_result = linear_step_julia(psi.copy(), False, D)

    assert _rel_l2(jl_result, py_result) < 1e-10


@pytest.mark.skipif(not _JULIA_AVAILABLE, reason="juliacall not installed")
def test_nonlinear_step_julia_vs_python_no_gain():
    """nonlinear_step_julia (gain-free) must match nonlinear_step to 1e-10."""
    rng = np.random.default_rng(2)
    n_cores, M = 3, 64
    psi = (
        rng.standard_normal((n_cores, M)) + 1j * rng.standard_normal((n_cores, M))
    ).astype(np.complex128)

    gamma_h = np.array([0.5, 0.3, 0.7])
    g0_h = np.zeros(n_cores)
    exp_g0h = np.ones(n_cores)
    exp_2g0h = np.ones(n_cores)
    E_sat = np.ones(n_cores) * 10.0

    psi_py = psi.copy()
    nonlinear_step(psi_py, gamma_h, g0_h, exp_g0h, exp_2g0h, E_sat, energy_in=None)

    psi_jl = psi.copy()
    nonlinear_step_julia(
        psi_jl, gamma_h, g0_h, exp_g0h, exp_2g0h, E_sat, energy_in=None
    )

    assert _rel_l2(psi_jl, psi_py) < 1e-10


@pytest.mark.skipif(not _JULIA_AVAILABLE, reason="juliacall not installed")
def test_full_simulation_julia_vs_python():
    """Full ssfm_order2_ndn_julia simulation must match ssfm_order2_ndn to 1e-10."""
    N, M = 10, 128

    solver_py = _make_solver("ssfm_order2_ndn", N=N, M=M)
    solver_jl = _make_solver("ssfm_order2_ndn_julia", N=N, M=M)

    # Set identical initial conditions
    t = np.linspace(-20.0, 20.0, M)
    psi0 = (1.0 / np.cosh(t)).astype(np.complex128).reshape(1, M)
    solver_py.numerical_solution[0][:] = psi0
    solver_jl.numerical_solution[0][:] = psi0

    solver_py.run_numerical_simulation()
    solver_jl.run_numerical_simulation()

    py_final = solver_py.numerical_solution[-1]
    jl_final = solver_jl.numerical_solution[-1]

    err = _rel_l2(jl_final, py_final)
    assert err < 1e-10, f"full simulation rel L2 = {err:.2e} (must be < 1e-10)"


# ── Phase 3: edge-case tests ──────────────────────────────────────────────────


@pytest.mark.skipif(not _JULIA_AVAILABLE, reason="juliacall not installed")
def test_julia_single_core():
    """n_cores=1, no coupling — simulation must match Python baseline."""
    N, M = 5, 64
    solver_py = _make_solver("ssfm_order2_ndn", N=N, M=M, n_cores=1)
    solver_jl = _make_solver("ssfm_order2_ndn_julia", N=N, M=M, n_cores=1)

    t = np.linspace(-20.0, 20.0, M)
    psi0 = (1.0 / np.cosh(t)).astype(np.complex128).reshape(1, M)
    solver_py.numerical_solution[0][:] = psi0
    solver_jl.numerical_solution[0][:] = psi0

    solver_py.run_numerical_simulation()
    solver_jl.run_numerical_simulation()

    err = _rel_l2(solver_jl.numerical_solution[-1], solver_py.numerical_solution[-1])
    assert err < 1e-10, f"single-core rel L2 = {err:.2e}"


@pytest.mark.skipif(not _JULIA_AVAILABLE, reason="juliacall not installed")
def test_julia_no_dispersion():
    """has_beta=False path: beta2=0 so only coupling matrix multiply is used."""
    N, M = 5, 64
    # With beta2=0 the solver sets has_beta=False
    solver_py = _make_solver("ssfm_order2_ndn", N=N, M=M, beta2=0.0, coupling=1.0)
    solver_jl = _make_solver("ssfm_order2_ndn_julia", N=N, M=M, beta2=0.0, coupling=1.0)

    rng = np.random.default_rng(42)
    nc = solver_py.eq.size
    psi0 = (rng.standard_normal((nc, M)) + 1j * rng.standard_normal((nc, M))).astype(
        np.complex128
    )
    solver_py.numerical_solution[0][:] = psi0
    solver_jl.numerical_solution[0][:] = psi0

    solver_py.run_numerical_simulation()
    solver_jl.run_numerical_simulation()

    err = _rel_l2(solver_jl.numerical_solution[-1], solver_py.numerical_solution[-1])
    assert err < 1e-10, f"no-dispersion rel L2 = {err:.2e}"


@pytest.mark.skipif(not _JULIA_AVAILABLE, reason="juliacall not installed")
def test_julia_with_damping():
    """damp_length > 0: output energy must be less than input energy."""
    N, M = 10, 128
    solver = _make_solver("ssfm_order2_ndn_julia", N=N, M=M, damp_length=0.1)

    t = np.linspace(-20.0, 20.0, M)
    psi0 = (1.0 / np.cosh(t)).astype(np.complex128).reshape(1, M)
    solver.numerical_solution[0][:] = psi0

    E_in = np.sum(np.abs(solver.numerical_solution[0]) ** 2) * solver.com.tau
    solver.run_numerical_simulation()
    E_out = np.sum(np.abs(solver.numerical_solution[-1]) ** 2) * solver.com.tau

    assert E_out < E_in, f"damping: E_out={E_out:.4f} should be < E_in={E_in:.4f}"


@pytest.mark.skipif(not _JULIA_AVAILABLE, reason="juliacall not installed")
def test_julia_with_noise():
    """noise_amplitude > 0: output must differ from noiseless run."""
    N, M = 5, 64
    solver_noisy = _make_solver("ssfm_order2_ndn_julia", N=N, M=M, noise_amplitude=0.01)
    solver_noiseless = _make_solver(
        "ssfm_order2_ndn_julia", N=N, M=M, noise_amplitude=0.0
    )

    t = np.linspace(-20.0, 20.0, M)
    psi0 = (1.0 / np.cosh(t)).astype(np.complex128).reshape(1, M)
    solver_noisy.numerical_solution[0][:] = psi0
    solver_noiseless.numerical_solution[0][:] = psi0

    solver_noisy.run_numerical_simulation()
    solver_noiseless.run_numerical_simulation()

    diff = np.linalg.norm(
        solver_noisy.numerical_solution[-1] - solver_noiseless.numerical_solution[-1]
    )
    assert diff > 0, "noisy and noiseless outputs are identical — noise not applied"


# ── Phase 4: US2 fallback tests ───────────────────────────────────────────────


def _load_ssfm_julia_without_juliacall(module_name: str):
    """Load ssfm_julia.py with juliacall mocked as absent."""
    # Remove any cached version
    for key in list(sys.modules.keys()):
        if "ssfm_julia" in key and key != __name__:
            del sys.modules[key]

    builtins_import = __builtins__ if callable(__builtins__) else __builtins__["__import__"]

    def fake_import(name, *args, **kwargs):
        if name == "juliacall":
            raise ImportError("mocked: juliacall not available")
        return builtins_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=fake_import):
        spec = importlib.util.spec_from_file_location(module_name, _SSFM_JULIA_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def test_import_without_julia():
    """Importing ssfm_julia must succeed even when juliacall is absent."""
    mod = _load_ssfm_julia_without_juliacall("ssfm_julia_fresh")
    assert mod._JULIA_AVAILABLE is False


def test_call_without_julia_raises_runtime_error():
    """Calling ssfm_order2_ndn_julia when Julia is absent must raise RuntimeError."""
    mod = _load_ssfm_julia_without_juliacall("ssfm_julia_stub")

    assert mod._JULIA_AVAILABLE is False
    with pytest.raises(RuntimeError) as exc_info:
        mod.ssfm_order2_ndn_julia()
    assert "juliacall" in str(exc_info.value)
    assert "pip install" in str(exc_info.value)


# ── Phase 5: US3 performance smoke test ──────────────────────────────────────


@pytest.mark.skipif(not _JULIA_AVAILABLE, reason="juliacall not installed")
def test_julia_performance_neutral():
    """Julia total wall-clock for small grid must complete within 60 seconds."""
    N, M = 10, 1024
    solver = _make_solver("ssfm_order2_ndn_julia", N=N, M=M)

    t = np.linspace(-20.0, 20.0, M)
    psi0 = (1.0 / np.cosh(t)).astype(np.complex128).reshape(1, M)
    solver.numerical_solution[0][:] = psi0

    t0 = time.perf_counter()
    solver.run_numerical_simulation()
    elapsed = time.perf_counter() - t0

    assert elapsed < 60.0, f"Julia simulation took {elapsed:.1f}s (limit 60s)"
