import numpy as np
from numpy.typing import NDArray
from scipy.linalg.lapack import get_lapack_funcs
from tqdm import trange

from fiberprop.ssfm_mcf import apply_absorbing_boundary, nonlinear_step, linear_step, nonlinear_step_windowed

__all__ = [
    "prepare_compact_solver_for_linear_step",
    "linear_step_compact",
    "getF",
    "ssfm_order2_ndn_compact_windowed",
]

Complex = np.complex128


def _build_tridiagonals(a: complex, c_main: complex, M: int
                        ) -> tuple[NDArray[Complex], NDArray[Complex], NDArray[Complex]]:
    dl = np.full(M - 1, a, dtype=Complex)
    du = np.full(M - 1, a, dtype=Complex)
    d  = np.full(M,     c_main, dtype=Complex)
    return dl, d, du


def prepare_compact_solver_for_linear_step(solver, h) -> None:
    if getattr(solver, "_compact_ready", False):
        return

    eq_size = solver.eq.size
    M       = solver.com.M

    # линейные связи ---------------------------------------------------
    lin = solver.linear_coeffs_array.astype(Complex, copy=False)
    nbr_ids, nbr_val = [], []
    for row in lin:
        nz = np.nonzero(row)[0]
        nbr_ids.append(nz.astype(np.intp))
        nbr_val.append(row[nz])

    # коэффициенты компактной схемы -----------------------------------
    tau2  = solver.com.tau**2
    b = 1 * solver.eq.beta2.astype(float)

    cur   = h / tau2
    R     = -0.5j * b * cur
    A     = 1/12 - 0.5*R
    C     = 1 - 2*A
    c_lin = -0.5j * b
    A1    = A + c_lin*cur
    A2    = 1 - 2*A1

    F_side   =  1j * h / 12.0
    F_center =  1j * h / 3.0

    # постоянные диагонали + factorisation + u,v,invAlpha -------------
    gttrf = get_lapack_funcs("gttrf", dtype=Complex)
    gttrs = get_lapack_funcs("gttrs", dtype=Complex)

    dl_f, d_f, du_f, du2_f, ipiv_f = [], [], [], [], []
    u_vec, v_vec, invAlpha = [], [], []

    for a_coef, c_main in zip(A, C):
        dl, d, du = _build_tridiagonals(a_coef, c_main, M)

        # модификации диагонали – см. C++ (dg[0]-=1; dg[-1]-=A*B , B=A)
        d[0]      -= 1.0
        d[-1]     -= a_coef * a_coef   # A*B , B=A

        dl_fac, d_fac, du_fac, du2_fac, piv, info = gttrf(dl, d, du)
        if info != 0:
            raise RuntimeError(f"gttrf info={info}")
        dl_f.append(dl_fac); d_f.append(d_fac)
        du_f.append(du_fac); du2_f.append(du2_fac); ipiv_f.append(piv)

        # u, v, invAlpha ------------------------------------------------
        u = np.zeros(M, dtype=Complex)
        v = np.zeros(M, dtype=Complex)
        u[0] = v[0] = 1.0
        u[-1] = a_coef          # B
        v[-1] = a_coef          # A
        u_sol, info = gttrs(dl_fac, d_fac, du_fac, du2_fac, piv, u.copy())
        if info != 0:
            raise RuntimeError("gttrs(u) failed")
        alpha = 1.0 + v[0]*u_sol[0] + v[-1]*u_sol[-1]
        u_vec.append(u_sol)
        v_vec.append(v)
        invAlpha.append(1.0/alpha)

    # коэффициент −i(α+g₀)/2  (α = loss, g₀ = const gain) -------------
    alpha_arr = getattr(solver.eq, "alpha", np.zeros(eq_size))
    g0_arr    = getattr(solver.eq, "g_0",   np.zeros(eq_size))
    diag_coeff = 0.5j * (alpha_arr + g0_arr)

    # scratch ----------------------------------------------------------
    solver._F_buf   = np.empty((eq_size, M), dtype=Complex)
    solver._rhs_buf = np.empty((eq_size, M), dtype=Complex)

    # cache ------------------------------------------------------------
    solver._dl_fac   = dl_f
    solver._d_fac    = d_f
    solver._du_fac   = du_f
    solver._du2_fac  = du2_f
    solver._ipiv     = ipiv_f
    solver._u_vec    = u_vec
    solver._v_vec    = v_vec
    solver._invAlpha = np.array(invAlpha, dtype=Complex)

    solver._A1 = A1
    solver._A2 = A2
    solver._F_side   = F_side
    solver._F_center = F_center

    solver._nbr_ids  = nbr_ids
    solver._nbr_val  = nbr_val
    solver._diag_coeff = diag_coeff
    solver._compact_ready = True


def getF(U: NDArray[Complex], F: NDArray[Complex], solver) -> None:
    eq, _ = U.shape
    for l in range(eq):
        F[l][:] = solver._diag_coeff[l] * U[l]    # (α+g₀)/2 ·U
        ids = solver._nbr_ids[l]
        coeff = solver._nbr_val[l]
        for k_idx, k in enumerate(ids):
            F[l] += coeff[k_idx] * U[k]           # + Σ C_lk U_k


def _form_U_iter(U: NDArray[Complex], F: NDArray[Complex], solver) -> NDArray[Complex]:
    A1, A2 = solver._A1, solver._A2
    Fs, Fc = solver._F_side, solver._F_center
    U_l  = np.roll(U,  1, axis=1)
    U_r  = np.roll(U, -1, axis=1)
    F_l  = np.roll(F,  1, axis=1)
    F_r  = np.roll(F, -1, axis=1)
    return (A1[:, None]*(U_l+U_r) + A2[:, None]*U
            + Fs*(F_l+F_r) + Fc*F)


def _solve_periodic(rhs: NDArray[Complex], idx: int, solver, gttrs):
    x, info = gttrs(solver._dl_fac[idx], solver._d_fac[idx],
                    solver._du_fac[idx], solver._du2_fac[idx],
                    solver._ipiv[idx], rhs.copy())
    if info != 0:
        raise RuntimeError("gttrs failed")
    beta = solver._v_vec[idx][0]*x[0] + solver._v_vec[idx][-1]*x[-1]
    x += (-solver._invAlpha[idx]*beta) * solver._u_vec[idx]
    return x


def linear_step_compact(psi: NDArray[Complex], solver, h,
                        *, n_iter_max: int = 30, tol: float = 1e-10) -> NDArray[Complex]:
    if not solver._compact_ready:
        raise RuntimeError("call prepare_compact_linear_solver first")

    gttrs = get_lapack_funcs("gttrs", dtype=Complex)
    U_prev = psi.copy()
    U      = psi.copy()
    F      = solver._F_buf

    # неизменяемая в итерациях сумма ----------------------------
    getF(U, F, solver)
    stat_sum = _form_U_iter(U, F, solver)

    for _ in range(n_iter_max):
        getF(U_prev, F, solver)
        rhs = stat_sum + 0.5j * h * F      #   +i·h/2·F

        for l in range(U.shape[0]):
            U[l] = _solve_periodic(rhs[l], l, solver, gttrs)

        if np.max(np.abs(U - U_prev)) < tol:
            break
        U_prev[:] = U

    return U


def ssfm_order2_ndn_compact_windowed(psi, current_energy, solver,
                                     h, tau, window_size,
                                     damp_length=0.0, noise_amplitude=0.0):

    nonlinear_step_windowed(psi, solver.gamma_h_half, solver.g0_h_half,
                            solver.exp_g0h_half, solver.exp_2g0h_half,
                            solver.eq.E_sat, solver.eq.g_0, tau, window_size)

    psi = linear_step_compact(psi, solver, h)

    nonlinear_step_windowed(psi, solver.gamma_h_half, solver.g0_h_half,
                            solver.exp_g0h_half, solver.exp_2g0h_half,
                            solver.eq.E_sat, solver.eq.g_0, tau, window_size)

    # current_energy[:] = np.sum(np.abs(psi)**2, axis=1)*tau

    return psi


def ssfm_order2_dnd_compact_windowed(psi, current_energy, solver,
                                     h, tau, window_size,
                                     damp_length=0.0, noise_amplitude=0.0):

    psi = linear_step_compact(psi, solver, h * 0.5)

    nonlinear_step_windowed(psi, solver.gamma_h, solver.g0_h,
                            solver.exp_g0h, solver.exp_2g0h,
                            solver.eq.E_sat, tau, window_size)

    psi = linear_step_compact(psi, solver, h * 0.5)

    current_energy[:] = np.sum(np.abs(psi)**2, axis=1)*tau

    return psi


def ssfm_order2_dnd_compact_windowed_short(solver, window_size, damp_length=0.0):

    psi = linear_step_compact(solver.numerical_solution[0], solver, solver.com.h * 0.5)

    if damp_length:
        psi = apply_absorbing_boundary(psi, solver=solver)

    for n in trange(solver.com.N):

        nonlinear_step_windowed(psi, solver.gamma_h, solver.g0_h,
                                solver.exp_g0h, solver.exp_2g0h,
                                solver.eq.E_sat, solver.com.tau, window_size,
                                offset_left=solver.com.offset_size)

        psi = linear_step_compact(psi, solver, solver.com.h)

        if damp_length:
            psi = apply_absorbing_boundary(psi, solver=solver)

    psi = linear_step_compact(psi, solver, solver.com.h * 0.5)

    if damp_length:
        psi = apply_absorbing_boundary(psi, solver=solver)

    return psi