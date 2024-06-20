from cmath import exp as cexp
from tqdm import trange
from SSFM_MCF import *
from matrixes import *


def FullClearPropagation_Simulation(pulse, N, num, Dmat, gamma, E_sat, g_0, h, tau,
                                    Frensel_k, omega, delta, Delta, phi, ITER_NUM):
    """ Последовательное моделирование ITER_NUM итераций """
    current_pulse = np.copy(pulse)
    for i in trange(ITER_NUM - 1):
        current_pulse = SimulateClearPropagation(current_pulse, N, num, Dmat, gamma, E_sat, g_0, h, tau)
        current_pulse = RightBoundary(current_pulse, omega, delta, Delta, phi)
        current_pulse = SimulateClearPropagation(current_pulse, N, num, Dmat, gamma, E_sat, g_0, h, tau)
        current_pulse = LeftBoundary(current_pulse, Frensel_k)
    current_pulse = SimulateClearPropagation(current_pulse, N, num, Dmat, gamma, E_sat, g_0, h, tau)
    current_pulse = RightBoundary(current_pulse, omega, delta, Delta, phi)
    current_pulse = SimulateClearPropagation(current_pulse, N, num, Dmat, gamma, E_sat, g_0, h, tau)
    current_pulse = OutputCouplerCondition(current_pulse, Frensel_k)
    return current_pulse


def OutputCouplerCondition(pulse, Frensel_k):
    """ Коэффициент отсечения при выходе из волокна """
    return pulse * (1 - Frensel_k)


def LeftBoundary(pulse, Frensel_k):
    """ Условие на левой границе резонатора """
    return pulse * Frensel_k


def RightBoundary(pulse, omega, delta, Delta, phi):
    """ Условие на правой границе резонатора """
    fourierPulse = FFTforVector(pulse)
    fourierPulse = fourierPulse * np.exp(-(delta - omega)**2 / (2*Delta**2)) * cexp(1j*phi)
    new_pulse = iFFTforVector(fourierPulse)
    return new_pulse


def SimulateClearPropagation(pulse, N, num, Dmat, gamma, E_sat, g_0, h, tau):
    """ Строит решение методом SSFM, строит конечное значение поля в каждой сердцевине """
    currentEnergy = np.array([0] * num, dtype=float)
    for n in range(N - 1):
        pulse = SSFMOrder2(pulse, currentEnergy, Dmat, gamma, E_sat, g_0, h, tau)
    return pulse


def makeFull(tens):
    """ Добавляет последнюю точку по времени из периодичности условий (для полного поля во всех сердцевинах) """
    num, N, M_ = tens.shape
    M = M_ + 1
    new = np.empty((num, N, M), dtype=complex)
    for j in range(num):
        for k in range(N):
            for i in range(M - 1):
                new[j][k][i] = tens[j][k][i]
            new[j][k][M - 1] = tens[j][k][0]
    return new


def makeFull1D(tens, num, M):
    """ Добавляет последнюю точку по времени из периодичности условий (для последней точки по z во всех сердцевинах) """
    new = np.empty((num, M), dtype=complex)
    for j in range(num):
        for i in range(M - 1):
            new[j][i] = tens[j][i]
        new[j][M - 1] = tens[j][0]
    return new


