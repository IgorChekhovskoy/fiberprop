"""
Julia backend wrapper for fiberprop SSFM solver.

Exposes ssfm_order2_ndn_julia, nonlinear_step_julia, linear_step_julia.
Safe to import even when juliacall / Julia are not installed.
"""

import os

__all__ = [
    "ssfm_order2_ndn_julia",
    "nonlinear_step_julia",
    "linear_step_julia",
]

try:
    import juliacall  # noqa: F401 – test import only; real usage deferred to _ensure_julia()
    _JULIA_AVAILABLE = True
except ImportError:
    _JULIA_AVAILABLE = False


def _need_julia():
    raise RuntimeError(
        "Julia backend not available: juliacall package not found. "
        "Install with: pip install juliacall juliapkg"
    )


if not _JULIA_AVAILABLE:
    # ── Stub functions (no-op until juliacall is installed) ──────────────────

    def ssfm_order2_ndn_julia(*args, **kwargs):
        _need_julia()

    def nonlinear_step_julia(*args, **kwargs):
        _need_julia()

    def linear_step_julia(*args, **kwargs):
        _need_julia()

else:
    # ── Full implementation — filled in by T014–T016 ─────────────────────────

    import numpy as np

    _julia_ready: bool = False
    _jl = None  # juliacall.Main namespace, set by _ensure_julia()

    def _ensure_julia():
        """Lazily start the Julia runtime and load FiberpropSSFM on first call."""
        global _julia_ready, _jl
        if _julia_ready:
            return

        # JULIA_NUM_THREADS must be set *before* juliacall is fully initialised.
        os.environ.setdefault("JULIA_NUM_THREADS", str(os.cpu_count() or 1))

        import juliacall

        _jl = juliacall.Main

        # Locate FiberpropSSFM.jl relative to this file: ../../julia/FiberpropSSFM.jl
        _here = os.path.dirname(os.path.abspath(__file__))
        julia_path = os.path.normpath(os.path.join(_here, "..", "julia", "FiberpropSSFM.jl"))
        julia_path_escaped = julia_path.replace("\\", "\\\\")

        _jl.seval(f'include("{julia_path_escaped}")')
        _jl.seval("using .FiberpropSSFM")

        _julia_ready = True

    # ── T010 wrapper ──────────────────────────────────────────────────────────

    def nonlinear_step_julia(
        psi,
        gamma_h,
        g0_h,
        exp_g0h,
        exp_2g0h,
        E_sat,
        energy_in,
        P=None,
    ):
        """
        In-place nonlinear step via Julia.
        Equivalent to ssfm_mcf.nonlinear_step for the same inputs.
        """
        _ensure_julia()
        import numpy as np

        psi_c = np.ascontiguousarray(psi, dtype=np.complex128)
        gamma_h_f = np.ascontiguousarray(gamma_h, dtype=np.float64)
        g0_h_f = np.ascontiguousarray(g0_h, dtype=np.float64)
        exp_g0h_f = np.ascontiguousarray(exp_g0h, dtype=np.float64)
        exp_2g0h_f = np.ascontiguousarray(exp_2g0h, dtype=np.float64)
        E_sat_f = np.ascontiguousarray(E_sat, dtype=np.float64)

        if energy_in is None:
            energy_in_f = np.zeros(psi.shape[0], dtype=np.float64)
        else:
            energy_in_f = np.ascontiguousarray(energy_in, dtype=np.float64)

        if P is None:
            P_f = (psi_c.real ** 2 + psi_c.imag ** 2).astype(np.float64)
        else:
            P_f = np.ascontiguousarray(P, dtype=np.float64)

        _jl.FiberpropSSFM.py_nonlinear_step(
            psi_c, gamma_h_f, g0_h_f, exp_g0h_f, exp_2g0h_f,
            E_sat_f, energy_in_f, P_f,
        )
        psi[:] = psi_c

    # ── T011 wrapper ──────────────────────────────────────────────────────────

    def linear_step_julia(psi, has_beta, D):
        """
        Linear step via Julia.
        Equivalent to ssfm_mcf.linear_step for the same inputs.
        Returns a new array with the result.
        """
        _ensure_julia()
        import numpy as np

        n_cores, M = psi.shape
        psi_c = np.ascontiguousarray(psi, dtype=np.complex128)
        D_c = np.ascontiguousarray(D, dtype=np.complex128)

        plans = _jl.FiberpropSSFM.get_plans(n_cores, M)
        result_jl = _jl.FiberpropSSFM.py_linear_step(psi_c, has_beta, D_c, plans)
        return np.asarray(result_jl)

    # ── T016 wrapper ──────────────────────────────────────────────────────────

    def ssfm_order2_ndn_julia(
        psi,
        current_energy,
        solver,
        h: float,
        tau: float,
        damp_length: float = 0.0,
        noise_amplitude: float = 0.0,
    ):
        """
        Full N-D-N split-step via Julia backend.

        psi is modified in-place and returned (same object).
        """
        _ensure_julia()
        import numpy as np

        psi_c = np.ascontiguousarray(psi, dtype=np.complex128)
        energy_c = np.ascontiguousarray(current_energy, dtype=np.float64)

        gamma_h = np.ascontiguousarray(solver.gamma_h_half, dtype=np.float64)
        g0_h = np.ascontiguousarray(solver.g0_h_half, dtype=np.float64)
        exp_g0h = np.ascontiguousarray(solver.exp_g0h_half, dtype=np.float64)
        exp_2g0h = np.ascontiguousarray(solver.exp_2g0h_half, dtype=np.float64)
        E_sat = np.ascontiguousarray(solver.eq.E_sat, dtype=np.float64)

        D_c = np.ascontiguousarray(solver.D, dtype=np.complex128)
        has_beta = bool(solver.has_beta)

        if solver.taper is not None:
            taper_c = np.ascontiguousarray(solver.taper, dtype=np.float64)
        else:
            taper_c = np.zeros(0, dtype=np.float64)  # empty signals "no taper"

        _jl.FiberpropSSFM.py_ssfm_order2_ndn(
            psi_c,
            energy_c,
            gamma_h,
            g0_h,
            exp_g0h,
            exp_2g0h,
            E_sat,
            D_c,
            has_beta,
            taper_c,
            float(tau),
            float(damp_length),
            float(noise_amplitude),
        )

        psi[:] = psi_c
        current_energy[:] = energy_c
        return psi
