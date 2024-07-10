from scipy.fft import fft, ifft
from pulses import *


def get_energy_simpson(arr_func, time_step):
    """ Возвращает величину энергии (интеграл считается по формуле Симпсона) """
    n = len(arr_func)

    if n % 2 == 0:
        raise ValueError("Длина массива должна быть нечетной для применения правила Симпсона")

    summ = arr_func[0] + arr_func[-1] + 4 * np.sum(arr_func[1:n - 1:2]) + 2 * np.sum(arr_func[2:n - 2:2])

    return summ * time_step / 3


def get_energy_rectangles(arr_func, time_step):
    """ Возвращает величину энергии (интеграл считается по формуле левых прямоугольников) """
    return np.sum(np.abs(arr_func)**2) * time_step


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


def linear_step(psi, Dmat):
    """ Линейный оператор (связи, дисперсия и потери) """
    n = len(psi)
    resV = np.zeros_like(psi)
    for i in range(n):
        for j in range(n):
            resV[i] += psi[j] * Dmat[i*n + j]
    return resV


def SSFMOrder2(psi, current_energy, D, gamma, E_sat, g_0, h, tau):
    """ Реализация схемы расщепления """
    num = len(psi)
    for i in range(num):
        if g_0[i] != 0:
            current_energy[i] = get_energy_rectangles(psi[i], tau)
    nonlinear_step(psi, gamma, E_sat, g_0, current_energy, h/2)

    psi = fft(psi, axis=1)
    psi = linear_step(psi, D)
    psi = ifft(psi, axis=1)

    for i in range(num):
        if g_0[i] != 0:
            current_energy[i] = get_energy_rectangles(psi[i], tau)
    nonlinear_step(psi, gamma, E_sat, g_0, current_energy, h/2)
    return psi


def SSFMOrder2_2(psi, current_energy, D, gamma, E_sat, g_0, h, tau):
    """ Реализация схемы расщепления """
    psi = fft(psi, axis=1)
    psi = linear_step(psi, D)
    psi = ifft(psi, axis=1)

    num = len(psi)
    if g_0 != 0:
        for i in range(num):
            current_energy[i] = get_energy_rectangles(psi[i], tau)
    nonlinear_step(psi, gamma, E_sat, g_0, current_energy, h)

    psi = fft(psi, axis=1)
    psi = linear_step(psi, D)
    psi = ifft(psi, axis=1)
    return psi
