import copy
import numpy as np
from numpy import newaxis as _na
from numba import njit

# === Многопоточные FFT (MKL, если есть; иначе SciPy с workers) ===
import os
try:
    # вариант с MKL (если пакет установлен)
    # from numpy.fft import fft as _fft_impl, ifft as _ifft_impl  # Не используй NUMPY.FFT: он поддерживает только 64-битные вычисления!
    from mkl_fft.interfaces.numpy_fft import fft as _fft_impl, ifft as _ifft_impl
    # from mkl_fft import fft as _fft_impl, ifft as _ifft_impl
    def _fft_mt(x, axis):  return _fft_impl(x, axis=axis)
    def _ifft_mt(x, axis): return _ifft_impl(x, axis=axis)
except Exception:
    # чистый SciPy: включаем многопоточность через workers
    from scipy.fft import fft as _fft_impl, ifft as _ifft_impl
    _workers = int(os.environ.get("OMP_NUM_THREADS", "1")) or 1
    def _fft_mt(x, axis):  return _fft_impl(x, axis=axis, workers=_workers)
    def _ifft_mt(x, axis): return _ifft_impl(x, axis=axis, workers=_workers)

from tqdm import trange

from .pulses import *


def get_simpson_integral(arr_func, time_step):
    """ Возвращает величину интеграла по формуле Симпсона """
    n = len(arr_func)

    if n % 2 == 0:
        raise ValueError("Длина массива должна быть нечетной для применения правила Симпсона")

    summ = arr_func[0] + arr_func[-1] + 4 * np.sum(arr_func[1:n - 1:2]) + 2 * np.sum(arr_func[2:n - 2:2])

    return summ * time_step / 3


@njit(inline='always', cache=True)
def get_energy_simpson(arr_func, time_step):
    """ Возвращает величину энергии (интеграл считается по формуле Симпсона) """
    power_arr = np.abs(arr_func)**2
    n = len(arr_func)
    summ = power_arr[n - 2] + 4*power_arr[n-1] + power_arr[0]
    for i in range(1, n - 1, 2):
        summ += power_arr[i - 1] + 4*power_arr[i] + power_arr[i + 1]
    return summ * time_step / 3


@njit(inline='always', cache=True)
def get_energy_rectangles(arr_func, time_step):
    """ Возвращает величину энергии (интеграл считается по формуле левых прямоугольников) """
    return np.sum(np.abs(arr_func)**2) * time_step


@njit(inline='always', cache=True)
def get_rectangles_integral(arr_func, time_step):
    """ Возвращает величину энергии (интеграл считается по формуле левых прямоугольников) """
    return arr_func * time_step


def power_abs2_np(psi: np.ndarray) -> np.ndarray:
    """|psi|^2 поэлементно, без sqrt/abs. Возвращает массив той же формы."""
    return psi.real * psi.real + psi.imag * psi.imag

def energy_from_power_np(P: np.ndarray, tau: float) -> np.ndarray:
    """Интеграл энергии по времени для каждого канала: sum_t P * tau  → (n,)"""
    return P.sum(axis=1) * tau


def nonlinear_step(psi: np.ndarray,
                   gamma_h, g0_h, exp_g0h, exp_2g0h,
                   E_sat: np.ndarray,
                   energy_in: np.ndarray | None = None,
                   *,
                   P: np.ndarray | None = None) -> None:
    """in-place обновление psi (n,M).
       Если P передан — используем его как |psi|^2, иначе посчитаем быстро.
    """
    if P is None:
        P = power_abs2_np(psi)

    no_gain = (g0_h == 0.0)
    if np.any(no_gain):          # без усиления
        psi[no_gain] *= np.exp(1j * gamma_h[no_gain, _na] * P[no_gain])

    gain = ~no_gain
    if np.any(gain):             # с усилением
        Ek   = energy_in[gain, _na]
        Pk   = P[gain]
        esat = E_sat[gain, _na]
        g0h  = g0_h[gain, _na]
        eg1  = exp_g0h[gain, _na]
        eg2  = exp_2g0h[gain, _na]
        gamh = gamma_h[gain, _na]

        E  = np.sqrt((Ek**2 + 2*Ek*esat) * eg2 + esat**2) - esat
        C  = -gamh*Pk*(Ek+esat-esat*np.log(Ek+2*esat)) / (g0h*Ek) + np.angle(psi[gain])
        Pn = Pk*eg1*np.sqrt((Ek+2*esat)/Ek)*np.sqrt(E/(E+2*esat))
        phi= gamh*Pk*(E+esat-esat*np.log(E+2*esat)) / (g0h*Ek) + C
        psi[gain] = np.sqrt(Pn) * np.exp(1j*phi)


def linear_step(psi, has_beta, D):
    """
    Вариант *без копий* и с минимальными проверками.
    psi : (n, M) во временной области
    """
    if has_beta:
        psi_f = _fft_mt(psi, axis=1)

        psi_f = np.einsum('ijk,jk->ik', D, psi_f, optimize=True)

        return _ifft_mt(psi_f, axis=1)
    else:
        # β₁ = β₂ = 0 : оператор не зависит от ω
        # простое перемножение в t-области
        return D @ psi


def ssfm_order2_ndn(psi, current_energy, solver,
                    h, tau, damp_length=0.0, noise_amplitude=0.0):
    """ N–D–N со снятием двойных расчётов мощности. """
    g0 = solver.eq.g_0
    gain = g0 != 0.0

    P = power_abs2_np(psi)

    if gain.any():
        current_energy[gain] = energy_from_power_np(P, tau)[gain]

    # ½ NL
    nonlinear_step(psi,
                   solver.gamma_h_half, solver.g0_h_half,
                   solver.exp_g0h_half, solver.exp_2g0h_half,
                   solver.eq.E_sat, current_energy,
                   P=P)

    if damp_length:
        psi = apply_absorbing_boundary(psi, solver=solver)

    psi = linear_step(psi, solver.has_beta, solver.D)

    if damp_length:
        psi = apply_absorbing_boundary(psi, solver=solver)

    P = power_abs2_np(psi)

    if gain.any():
        current_energy[gain] = energy_from_power_np(P, tau)[gain]

    # ½ NL
    nonlinear_step(psi,
                   solver.gamma_h_half, solver.g0_h_half,
                   solver.exp_g0h_half, solver.exp_2g0h_half,
                   solver.eq.E_sat, current_energy,
                   P=P)

    if damp_length:
        psi = apply_absorbing_boundary(psi, solver=solver)

    if noise_amplitude:
        noise = noise_amplitude * (
            np.random.uniform(-1, 1, psi.shape) +
            1j*np.random.uniform(-1, 1, psi.shape)
        )
        psi += noise

    return psi


def Newton_method(func, func_der, prev_val, epsilon=1e-3):
    """
    Метод Ньютона для отыскания нуля монотонной функции
    """
    new_val = np.infty
    while abs(new_val - prev_val) > epsilon:
        curr_val = copy.deepcopy(new_val)
        new_val = prev_val - func(prev_val) / func_der(prev_val)
        prev_val = curr_val
    return new_val


@njit(inline='always', cache=True)
def nonlinear_step_order1_resonator(psi, gamma, E_sat, g_0, E_total, step):
    """ Нелинейный оператор (Керр и насыщение), метод первого порядка """
    n = len(psi)
    new_psi = np.empty_like(psi)
    for i in range(n):
        local_g = g_0[i] * (2*E_sat[i] + E_total[i]) / (E_sat[i] + E_total[i])
        P_0 = np.abs(psi[i])**2
        P = P_0 * np.exp(local_g * step)
        phi = np.angle(psi[i]) - P_0 * gamma[i]/local_g + P * gamma[i]/local_g
        new_psi[i] = np.sqrt(P) * np.exp(1j * phi)
    return new_psi


@njit(inline='always')
def ssfm_order1_resonator_nocos(psi, energy_forward, energy_backward, D, gamma, E_sat, g_0, h, tau, noise_amplitude=0.0):
    """ Реализация схемы расщепления для резонатора без учёта взаимодействия несущих частот прямой и обратной волн """
    energy_forward = np.copy(energy_forward)  # copy.deepcopy(energy_forward)
    E_total = energy_forward + energy_backward
    new_psi = nonlinear_step_order1_resonator(psi, gamma, E_sat, g_0, E_total, h/2)

    new_psi = linear_step(new_psi, D)

    num, _ = psi.shape
    for i in range(num):
        if g_0[i] != 0.0:
            energy_forward[i] = get_energy_rectangles(new_psi[i], tau)

    E_total = energy_forward + energy_backward
    new_psi = nonlinear_step_order1_resonator(new_psi, gamma, E_sat, g_0, E_total, h/2)

    if noise_amplitude != 0.0:
        current_noise = (np.random.uniform(-noise_amplitude, noise_amplitude, psi.shape) +
                         1j*np.random.uniform(-noise_amplitude, noise_amplitude, psi.shape))
        new_psi += current_noise
    return new_psi


def ssfm_order1_resonator_fullcos(psi_forward, psi_backward, D, gamma, E_sat, g_0, h, tau, noise_amplitude=0.0):
    """ Реализация схемы расщепления для резонатора с учётом взаимодействия несущих частот прямой и обратной волн """
    E_total = get_rectangles_integral(abs(psi_forward) ** 2 + abs(psi_backward) ** 2 +
                                      2 * (psi_forward.conjugate() * psi_backward).real,
                                      tau)
    nonlinear_step_order1_resonator(psi_forward, gamma, E_sat, g_0, E_total, h/2)

    psi_forward = linear_step(psi_forward, D)

    E_total = get_rectangles_integral(abs(psi_forward) ** 2 + abs(psi_backward) ** 2 +
                                      2 * (psi_forward.conjugate() * psi_backward).real,
                                      tau)
    nonlinear_step_order1_resonator(psi_forward, gamma, E_sat, g_0, E_total, h/2)
    if noise_amplitude != 0.0:
        current_noise = (np.random.uniform(-noise_amplitude, noise_amplitude, psi_forward.shape) +
                         1j*np.random.uniform(-noise_amplitude, noise_amplitude, psi_forward.shape))
        psi_forward += current_noise
    return psi_forward


def ssfm_order2_dnd(psi, current_energy, solver,
                    h, tau, damp_length=0.0, noise_amplitude=0.0):
    """ Реализация схемы расщепления """

    psi = linear_step(psi, solver.has_beta, solver.D_half)

    if damp_length:
        psi = apply_absorbing_boundary(psi, solver=solver)

    g0 = solver.eq.g_0
    gain = g0 != 0.0
    if gain.any():
        current_energy[gain] = (np.abs(psi[gain]) ** 2).sum(1) * tau

    nonlinear_step(psi,
                   solver.gamma_h, solver.g0_h,
                   solver.exp_g0h, solver.exp_2g0h,
                   solver.eq.E_sat, current_energy)

    if damp_length:
        psi = apply_absorbing_boundary(psi, solver=solver)

    psi = linear_step(psi, solver.has_beta, solver.D_half)

    if damp_length:
        psi = apply_absorbing_boundary(psi, solver=solver)

    # шум
    if noise_amplitude:
        noise = noise_amplitude * (
                np.random.uniform(-1, 1, psi.shape) +
                1j * np.random.uniform(-1, 1, psi.shape)
        )
        psi += noise

    return psi


def ssfm_order2_dnd_short(solver, damp_length=0.0, disable_progress_bar=False):

    psi = linear_step(solver.numerical_solution[0], solver.has_beta, solver.D_half)

    if damp_length:
        psi = apply_absorbing_boundary(psi, solver=solver)

    g0 = solver.eq.g_0
    gain = g0 != 0.0
    current_energy = np.zeros(psi.shape[0], dtype=solver.dtype)

    for _ in trange(solver.com.N, disable=disable_progress_bar):
        if gain.any():
            current_energy[gain] = (np.abs(psi[gain]) ** 2).sum(1) * solver.com.tau

        nonlinear_step(psi, solver.gamma_h, solver.g0_h,
                                solver.exp_g0h, solver.exp_2g0h,
                                solver.eq.E_sat, current_energy)

        psi = linear_step(psi, solver.has_beta, solver.D)

        if damp_length:
            psi = apply_absorbing_boundary(psi, solver=solver)

    psi = linear_step(psi, solver.has_beta, solver.invD_half)

    if damp_length:
        psi = apply_absorbing_boundary(psi, solver=solver)

    return psi


def nonlinear_step_in_fourier_space(psi: np.ndarray,
                   gamma_h, g0_h, exp_g0h, exp_2g0h, tau,
                   E_sat: np.ndarray,
                   current_energy: np.ndarray) -> None:
    """ in-place в частотной области: считаем |ψ|² один раз. """
    psi_t = _ifft_mt(psi, axis=1)
    P = power_abs2_np(psi_t)

    if (g0_h != 0.0).any():
        current_energy[g0_h != 0.0] = energy_from_power_np(P[g0_h != 0.0], tau)

    nonlinear_step(psi_t, gamma_h, g0_h, exp_g0h, exp_2g0h, E_sat, current_energy, P=P)
    psi[:] = _fft_mt(psi_t, axis=1)


def linear_step_in_fourier_space(psi_f, D):

    return np.einsum('ijk,jk->ik', D, psi_f, optimize=True)


def ssfm_order2_2_in_fourier_space(psi, current_energy, solver, h, tau):

    psi = linear_step_in_fourier_space(psi, solver.D)

    nonlinear_step_in_fourier_space(psi,
                   solver.gamma_h, solver.g0_h,
                   solver.exp_g0h, solver.exp_2g0h, tau,
                   solver.eq.E_sat, current_energy)

    psi = linear_step_in_fourier_space(psi, solver.D)

    return psi


def apply_absorbing_boundary(psi: np.ndarray, *, solver) -> np.ndarray:
    """Использует заранее посчитанный taper из solver."""
    taper = solver.taper
    if taper is None:
        return psi
    psi *= taper          # broadcasting (M,) → (C,M)
    return psi

################################################

# def nonlinear_step_windowed(psi: np.ndarray,
#                             gamma_h, g0_h, exp_g0h, exp_2g0h,
#                             E_sat: np.ndarray,
#                             tau, window):
#     for s in range(0, psi.shape[1], window):
#         e = min(psi.shape[1], s+window)
#         view = psi[:, s:e]
#         energy = (np.abs(view)**2).sum(1)*tau
#         nonlinear_step(view,
#                        gamma_h, g0_h,
#                        exp_g0h, exp_2g0h,
#                        E_sat, energy)


def nonlinear_step_windowed(psi: np.ndarray,
                            gamma_h, g0_h, exp_g0h, exp_2g0h,
                            E_sat: np.ndarray,
                            tau: float,
                            window: int,
                            *,
                            offset_left: int = 0) -> None:
    """
    Kerr+gain in-place по кускам; усиление только внутри [offset_left, M - offset_left).
    Не выделяет константы на каждом шаге — берёт solver.exp1 / solver.g0zeros.
    """
    M = psi.shape[1]
    offset_right = M - offset_left

    for s in range(0, M, window):
        e    = min(M, s + window)
        view = psi[:, s:e]

        l_off = max(0, offset_left  - s)
        r_off = max(0, e - offset_right)
        central_len = view.shape[1] - l_off - r_off

        # левая оффсет-зона (без усиления)
        if l_off:
            sub = view[:, :l_off]
            Psub = power_abs2_np(sub)
            nonlinear_step(sub,
                           gamma_h, g0_h*0,
                           exp_g0h, exp_2g0h,
                           E_sat, energy_in=None,
                           P=Psub)

        # центральная часть (с усилением)
        if central_len > 0:
            sub = view[:, l_off:l_off + central_len]
            Psub = power_abs2_np(sub)
            E_slice = energy_from_power_np(Psub, tau)
            nonlinear_step(sub,
                           gamma_h, g0_h,
                           exp_g0h, exp_2g0h,
                           E_sat, E_slice,
                           P=Psub)

        # правая оффсет-зона (без усиления)
        if r_off:
            sub = view[:, -r_off:]
            Psub = power_abs2_np(sub)
            nonlinear_step(sub,
                           gamma_h, g0_h*0,
                           exp_g0h, exp_2g0h,
                           E_sat, energy_in=None,
                           P=Psub)

def ssfm_order2_ndn_windowed(psi, current_energy, solver,
                            h, tau, window_size,
                            damp_length=0.0, noise_amplitude=0.0):

    nonlinear_step_windowed(psi, solver.gamma_h_half, solver.g0_h_half,
                            solver.exp_g0h_half, solver.exp_2g0h_half,
                            solver.eq.E_sat, tau, window_size,
                            offset_left=solver.com.offset_size)

    psi = linear_step(psi, solver.has_beta, solver.D)

    nonlinear_step_windowed(psi, solver.gamma_h_half, solver.g0_h_half,
                            solver.exp_g0h_half, solver.exp_2g0h_half,
                            solver.eq.E_sat, tau, window_size,
                            offset_left=solver.com.offset_size)

    # current_energy[:] = np.sum(np.abs(psi)**2, axis=1)*tau

    return psi


def ssfm_order2_dnd_windowed_short(solver, window_size, damp_length=0.0, disable_progress_bar=False):

    psi = linear_step(solver.numerical_solution[0], solver.has_beta, solver.D_half)

    if damp_length:
        psi = apply_absorbing_boundary(psi, solver=solver)

    for _ in trange(solver.com.N, disable=disable_progress_bar):

        nonlinear_step_windowed(psi, solver.gamma_h, solver.g0_h,
                                solver.exp_g0h, solver.exp_2g0h,
                                solver.eq.E_sat, solver.com.tau, window_size,
                                offset_left=solver.com.offset_size)

        psi = linear_step(psi, solver.has_beta, solver.D)

        if damp_length:
            psi = apply_absorbing_boundary(psi, solver=solver)

    psi = linear_step(psi, solver.has_beta, solver.invD_half)

    if damp_length:
        psi = apply_absorbing_boundary(psi, solver=solver)

    return psi