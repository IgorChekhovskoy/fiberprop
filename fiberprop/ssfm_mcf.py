import copy

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
def get_energy_Simpson(arr_func, time_step):
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


def nonlinear_step(psi, gamma, E_sat, g_0, current_energy, step):
    """ Нелинейный оператор (Керр и насыщение) """
    n = len(psi)
    for i in range(n):
        P_k = np.abs(psi[i])**2
        E_k = current_energy[i]
        if g_0[i] == 0:  # нет усиления
            psi[i] = psi[i] * np.exp(1j * gamma[i] * P_k * step)
            continue
        if E_k == 0:
            continue
        e_sat = E_sat[i]
        g0 = g_0[i]
        E = np.sqrt((E_k ** 2 + 2 * E_k * e_sat) * np.exp(2 * g0 * step) + e_sat ** 2) - e_sat
        C = -gamma[i] * P_k * (E_k + e_sat - e_sat * np.log(E_k + 2 * e_sat)) / (g0 * E_k) + np.angle(psi[i])
        P = P_k * np.exp(g0 * step) * np.sqrt((E_k + 2 * e_sat) / E_k) * np.sqrt(E / (E + 2 * e_sat))
        phi = gamma[i] * P_k * (E + e_sat - e_sat * np.log(E + 2 * e_sat)) / (g0 * E_k) + C
        psi[i] = np.sqrt(P) * np.exp(1j * phi)


@njit
def linear_step(psi, Dmat):
    """ Линейный оператор (связи, дисперсия и потери) """
    n = len(psi)
    res_vector = np.zeros_like(psi)
    for i in range(n):
        for j in range(n):
            res_vector[i] += psi[j] * Dmat[i*n + j]
    return res_vector


def coupling_step(psi, Dmat_disp_free):
    """ Линейный оператор для бездисперсионного случая (только связи и потери) """
    res_psi = np.matmul(Dmat_disp_free, psi)
    return res_psi


def ssfm_order2(psi, current_energy, D, gamma, E_sat, g_0, h, tau, damp_length=0.0, noise_amplitude=0.0):
    """ Реализация схемы расщепления """
    num = len(psi)
    for i in range(num):
        if g_0[i] != 0.0:  # нет усиления
            current_energy[i] = get_energy_rectangles(psi[i], tau)
    nonlinear_step(psi, gamma, E_sat, g_0, current_energy, h/2)

    if damp_length != 0.0:
        psi = apply_absorbing_boundary(psi, damping_length=damp_length)

    psi = fft(psi, axis=1)
    psi = linear_step(psi, D)
    psi = ifft(psi, axis=1)

    if damp_length != 0.0:
        psi = apply_absorbing_boundary(psi, damping_length=damp_length)

    for i in range(num):
        if g_0[i] != 0.0:
            current_energy[i] = get_energy_rectangles(psi[i], tau)
    nonlinear_step(psi, gamma, E_sat, g_0, current_energy, h/2)

    if damp_length != 0.0:
        psi = apply_absorbing_boundary(psi, damping_length=damp_length)

    if noise_amplitude != 0.0:
        current_noise = (np.random.uniform(-noise_amplitude, noise_amplitude, psi.shape) +
                         1j*np.random.uniform(-noise_amplitude, noise_amplitude, psi.shape))
        psi += current_noise
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

    new_psi = fft(new_psi, axis=1)
    new_psi = linear_step(new_psi, D)
    new_psi = ifft(new_psi, axis=1)

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

    psi_forward = fft(psi_forward, axis=1)
    psi_forward = linear_step(psi_forward, D)
    psi_forward = ifft(psi_forward, axis=1)

    E_total = get_rectangles_integral(abs(psi_forward) ** 2 + abs(psi_backward) ** 2 +
                                      2 * (psi_forward.conjugate() * psi_backward).real,
                                      tau)
    nonlinear_step_order1_resonator(psi_forward, gamma, E_sat, g_0, E_total, h/2)
    if noise_amplitude != 0.0:
        current_noise = (np.random.uniform(-noise_amplitude, noise_amplitude, psi_forward.shape) +
                         1j*np.random.uniform(-noise_amplitude, noise_amplitude, psi_forward.shape))
        psi_forward += current_noise
    return psi_forward


def ssfm_order2_2(psi, current_energy, D, gamma, E_sat, g_0, h, tau):
    """ Реализация схемы расщепления """
    psi = fft(psi, axis=1)
    psi = linear_step(psi, D)
    psi = ifft(psi, axis=1)

    num = len(psi)
    for i in range(num):
        if g_0[i] != 0:  # нет усиления
            current_energy[i] = get_energy_rectangles(psi[i], tau)
    nonlinear_step(psi, gamma, E_sat, g_0, current_energy, h)

    psi = fft(psi, axis=1)
    psi = linear_step(psi, D)
    psi = ifft(psi, axis=1)
    return psi


def ssfm_order2_dispersion_free(psi, current_energy, D, gamma, E_sat, g_0, h, tau):
    """ Реализация схемы расщепления для случая, когда ДГС отсутствует """
    psi = coupling_step(psi, D)

    num = len(psi)
    for i in range(num):
        if g_0[i] != 0:  # нет усиления
            current_energy[i] = get_energy_rectangles(psi[i], tau)
    nonlinear_step(psi, gamma, E_sat, g_0, current_energy, h)

    psi = coupling_step(psi, D)
    return psi


def apply_absorbing_boundary(psi, damping_length=0.1, damping_factor=1):
    """
    Применяет поглощающие граничные условия путем экспоненциального заглушения краев массива psi.

    Parameters:
    -----------
    psi : np.ndarray
        Волновая функция, к которой применяются граничные условия.
    damping_factor : float
        Фактор демпфирования, определяющий степень поглощения на краях.
    absorption_region : float
        Доля области, на которой применяется поглощение (от 0 до 0.5).

    Returns:
    --------
    psi : np.ndarray
        Волновая функция после применения граничных условий.
    """
    size, M = psi.shape
    taper = np.ones(M)

    # Определяем количество точек, на которые будет распространяться заглушение
    absorption_length = int(M * damping_length)

    # Применяем заглушение к краям области
    for i in range(absorption_length):
        # taper[i] = np.exp(-damping_factor * (absorption_length - i) / absorption_length)
        taper[i] = np.cos(np.pi / 2 * (absorption_length - i) / absorption_length) ** 6
        taper[-i - 1] = taper[i]

    # Применяем заглушение ко всей области
    for k in range(size):
        psi[k] *= taper

    return psi

