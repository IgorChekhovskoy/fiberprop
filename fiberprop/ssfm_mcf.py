import copy
import numpy as np
from numpy import newaxis as _na

from numba import njit
from scipy.fft import fft, ifft
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


def nonlinear_step(psi: np.ndarray,
                   gamma_h, g0_h, exp_g0h, exp_2g0h,
                   E_sat: np.ndarray,
                   g_0:   np.ndarray,
                   energy_in: np.ndarray) -> None:
    """in-place обновление psi (n,M)."""
    P = np.abs(psi)**2
    no_gain = (g_0 == 0.0)

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

    # psi[:, 0] = 0
    # psi[:, -1] = 0


def linear_step(psi, has_beta, D):
    """
    Вариант *без копий* и с минимальными проверками.
    psi : (n, M) во временной области
    """
    if has_beta:
        psi_f = fft(psi, axis=1)

        psi_f = np.einsum('ijk,jk->ik', D, psi_f, optimize=True)

        return ifft(psi_f, axis=1)
    else:
        # β₁ = β₂ = 0 : оператор не зависит от ω
        # простое перемножение в t-области
        return D @ psi


def ssfm_order2(psi, current_energy, solver,
                h, tau, damp_length=0.0, noise_amplitude=0.0):
    """psi.shape=(n,M); current_E=(n,).  Возвращает psi (как в исходнике)."""
    g0   = solver.eq.g_0
    gain = g0 != 0.0
    if gain.any():
        current_energy[gain] = (np.abs(psi[gain]) ** 2).sum(1) * tau

    # ½ NL
    nonlinear_step(psi,
                   solver.gamma_h_half, solver.g0_h_half,
                   solver.exp_g0h_half, solver.exp_2g0h_half,
                   solver.eq.E_sat, g0, current_energy)

    # absorber-1
    if damp_length:
        psi = apply_absorbing_boundary(psi, solver=solver)

    # FFT – L – IFFT
    psi = linear_step(psi, solver.has_beta, solver.D)

    # absorber-2
    if damp_length:
        psi = apply_absorbing_boundary(psi, solver=solver)

    # обновляем энергию ПОСЛЕ absorber-2
    if gain.any():
        current_energy[gain] = (np.abs(psi[gain]) ** 2).sum(1) * tau

    # ½ NL (вторая)
    nonlinear_step(psi,
                   solver.gamma_h_half, solver.g0_h_half,
                   solver.exp_g0h_half, solver.exp_2g0h_half,
                   solver.eq.E_sat, g0, current_energy)

    # absorber-3
    if damp_length:
        psi = apply_absorbing_boundary(psi, solver=solver)

    # шум
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


def ssfm_order2_2(psi, current_energy, solver,
                h, tau, damp_length=0.0, noise_amplitude=0.0):
    """ Реализация схемы расщепления """

    psi = linear_step(psi, solver.has_beta, solver.D)

    num = len(psi)
    for i in range(num):
        if solver.eq.g_0[i] != 0:  # нет усиления
            current_energy[i] = get_energy_rectangles(psi[i], tau)
    nonlinear_step(psi, solver.eq.gamma, solver.eq.E_sat, solver.eq.g_0, current_energy, h)

    psi = linear_step(psi, solver.has_beta, solver.D)

    return psi


def apply_absorbing_boundary(psi: np.ndarray, *, solver) -> np.ndarray:
    """Использует заранее посчитанный taper из solver."""
    taper = solver._taper_np
    if taper is None:
        return psi
    psi *= taper          # broadcasting (M,) → (C,M)
    return psi

