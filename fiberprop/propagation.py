from cmath import exp as cexp
from tqdm import trange
from math import pi

from scipy.fft import fft, ifft, fftfreq

from .drawing import *
from .matrices import create_freq_matrix, get_pade_exponential2
from .ssfm_mcf import ssfm_order2, get_energy_rectangles, nonlinear_step, linear_step


def resonator_simulation(pulse, N, equation_number, h, tau, coupling_matrix, beta_2, gamma, E_sat, alpha, g_0,
                         Fresnel_k, omega, delta, Delta, phi, ITER_NUM):
    """ Последовательное моделирование ITER_NUM итераций """

    M = pulse.shape[1]
    w = fftfreq(M, tau) * 2*pi
    Dmat = get_pade_exponential2(create_freq_matrix(coupling_matrix, beta_2, alpha, g_0, w, h))

    current_pulse = np.copy(pulse)
    for _ in trange(ITER_NUM - 1):
        current_pulse = simulate_propagation(current_pulse, N, equation_number, Dmat, gamma, E_sat, g_0, h, tau)
        current_pulse = right_boundary(current_pulse, omega, delta, Delta, phi)
        current_pulse = simulate_propagation(current_pulse, N, equation_number, Dmat, gamma, E_sat, g_0, h, tau)
        current_pulse = left_boundary(current_pulse, Fresnel_k)
    current_pulse = simulate_propagation(current_pulse, N, equation_number, Dmat, gamma, E_sat, g_0, h, tau)
    current_pulse = right_boundary(current_pulse, omega, delta, Delta, phi)
    current_pulse = simulate_propagation(current_pulse, N, equation_number, Dmat, gamma, E_sat, g_0, h, tau)
    current_pulse = output_coupler_condition(current_pulse, Fresnel_k)
    return current_pulse


def output_coupler_condition(pulse, Fresnel_k):
    """ Коэффициент отсечения при выходе из волокна """
    return pulse * (1 - Fresnel_k)


def left_boundary(pulse, Fresnel_k):
    """ Условие на левой границе резонатора """
    return pulse * Fresnel_k


def right_boundary(pulse, omega, delta, Delta, phi):
    """ Условие на правой границе резонатора """
    fourier_pulse = fft(pulse, axis=1)
    fourier_pulse = fourier_pulse * np.exp(-(delta - omega)**2 / (2*Delta**2)) * cexp(1j*phi)
    new_pulse = ifft(fourier_pulse, axis=1)
    return new_pulse


def simulate_propagation(pulse, N, equation_number, h, tau, coupling_matrix, beta_2, gamma, E_sat, alpha, g_0):
    """ Строит решение методом SSFM, строит конечное значение поля в каждой сердцевине """

    M = pulse.shape[1]
    w = fftfreq(M, tau) * 2*pi
    Dmat = get_pade_exponential2(create_freq_matrix(coupling_matrix, beta_2, alpha, g_0, w, h))

    energy = np.array([0] * equation_number, dtype=float)
    for n in range(N):
        pulse = ssfm_order2(pulse, energy, Dmat, gamma, E_sat, g_0, h, tau)
    return pulse


def simulate_propagation_ndn(pulse, N, equation_number, h, tau, coupling_matrix, beta_2, gamma, E_sat, alpha, g_0):
    """ Строит решение методом SSFM 1-го порядка с расщеплением вида nonlinear_step Dispersion nonlinear_step,
     используя объединение соседних половинных шагов """

    M = pulse.shape[1]
    w = fftfreq(M, tau) * 2*pi
    Dmat = get_pade_exponential2(create_freq_matrix(coupling_matrix, beta_2, alpha, g_0, w, h))

    energy = np.array([0] * equation_number, dtype=float)

    equation_number = len(pulse)
    if g_0 != 0:
        for i in range(equation_number):
            energy[i] = get_energy_rectangles(pulse[i], tau)
    nonlinear_step(pulse, gamma, E_sat, g_0, energy, h / 2)

    for n in range(N):
        pulse = fft(pulse, axis=1)
        pulse = linear_step(pulse, Dmat)
        pulse = ifft(pulse, axis=1)

        if g_0 != 0:
            for i in range(equation_number):
                energy[i] = get_energy_rectangles(pulse[i], tau)
        nonlinear_step(pulse, gamma, E_sat, g_0, energy, h)

    if g_0 != 0:
        for i in range(equation_number):
            energy[i] = get_energy_rectangles(pulse[i], tau)
    nonlinear_step(pulse, gamma, E_sat, g_0, energy, -h / 2)

    return pulse


def simulate_propagation_dnd(pulse, N, equation_number, h, tau, coupling_matrix, beta_2, gamma, E_sat, alpha, g_0):
    """ Строит решение методом SSFM 2-го порядка с расщеплением вида Dispersion nonlinear_step Dispersion,
     используя объединение соседних половинных шагов """

    M = pulse.shape[1]
    w = fftfreq(M, tau) * 2*pi
    DmatH = get_pade_exponential2(create_freq_matrix(coupling_matrix, beta_2, alpha, g_0, w, h))
    DmatH2Plus = get_pade_exponential2(create_freq_matrix(coupling_matrix, beta_2, alpha, g_0, w, h / 2))
    DmatH2Minus = get_pade_exponential2(create_freq_matrix(coupling_matrix, beta_2, alpha, g_0, w, -h / 2))

    pulse = fft(pulse, axis=1)
    pulse = linear_step(pulse, DmatH2Plus)
    pulse = ifft(pulse, axis=1)

    energy = np.array([0] * equation_number, dtype=float)

    for n in range(N):
        for i in range(equation_number):
            if g_0[i] != 0:
                energy[i] = get_energy_rectangles(pulse[i], tau)
        nonlinear_step(pulse, gamma, E_sat, g_0, energy, h)

        pulse = fft(pulse, axis=1)
        pulse = linear_step(pulse, DmatH)
        pulse = ifft(pulse, axis=1)

    pulse = fft(pulse, axis=1)
    pulse = linear_step(pulse, DmatH2Minus)
    pulse = ifft(pulse, axis=1)

    return pulse


def make_full(tens):
    """ Добавляет последнюю точку по времени из периодичности условий (для полного поля во всех сердцевинах) """
    equation_number, N_add, M = tens.shape
    new = np.empty((equation_number, N_add, M+1), dtype=complex)
    for j in range(equation_number):
        for k in range(N_add):
            for i in range(M):
                new[j][k][i] = tens[j][k][i]
            new[j][k][M] = tens[j][k][0]
    return new


def make_full_1d(tens, equation_number, M):
    """ Добавляет последнюю точку по времени из периодичности условий (для последней точки по z во всех сердцевинах) """
    new = np.empty((equation_number, M+1), dtype=complex)
    for j in range(equation_number):
        for i in range(M):
            new[j][i] = tens[j][i]
        new[j][M] = tens[j][0]
    return new


