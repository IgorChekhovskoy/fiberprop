from cmath import exp as cexp
from tqdm import trange
from SSFM_MCF import *
from matrixes import *
from numpy.fft import fftfreq


def FullClearPropagation_Simulation(pulse, N, num, h, tau, coupling_matrix, beta_2, gamma, E_sat, alpha, g_0,
                                    Fresnel_k, omega, delta, Delta, phi, ITER_NUM):

    """ Последовательное моделирование ITER_NUM итераций """

    M = pulse.shape[1]
    w = fftfreq(M, tau) * 2 * pi
    Dmat = PadeExpForMyFreqMatrix2(CreateMyFreqMatrix(coupling_matrix, beta_2, alpha, g_0, w, h))

    current_pulse = np.copy(pulse)
    for _ in trange(ITER_NUM - 1):
        current_pulse = SimulateClearPropagation(current_pulse, N, num, Dmat, gamma, E_sat, g_0, h, tau)
        current_pulse = RightBoundary(current_pulse, omega, delta, Delta, phi)
        current_pulse = SimulateClearPropagation(current_pulse, N, num, Dmat, gamma, E_sat, g_0, h, tau)
        current_pulse = LeftBoundary(current_pulse, Fresnel_k)
    current_pulse = SimulateClearPropagation(current_pulse, N, num, Dmat, gamma, E_sat, g_0, h, tau)
    current_pulse = RightBoundary(current_pulse, omega, delta, Delta, phi)
    current_pulse = SimulateClearPropagation(current_pulse, N, num, Dmat, gamma, E_sat, g_0, h, tau)
    current_pulse = OutputCouplerCondition(current_pulse, Fresnel_k)
    return current_pulse


def OutputCouplerCondition(pulse, Fresnel_k):
    """ Коэффициент отсечения при выходе из волокна """
    return pulse * (1 - Fresnel_k)


def LeftBoundary(pulse, Fresnel_k):
    """ Условие на левой границе резонатора """
    return pulse * Fresnel_k


def RightBoundary(pulse, omega, delta, Delta, phi):
    """ Условие на правой границе резонатора """
    fourierPulse = FFTforVector(pulse)
    fourierPulse = fourierPulse * np.exp(-(delta - omega)**2 / (2*Delta**2)) * cexp(1j*phi)
    new_pulse = iFFTforVector(fourierPulse)
    return new_pulse


def SimulateClearPropagation(pulse, N, num, h, tau, coupling_matrix, beta_2, gamma, E_sat, alpha, g_0):
    """ Строит решение методом SSFM, строит конечное значение поля в каждой сердцевине """

    M = pulse.shape[1]
    w = fftfreq(M, tau) * 2 * pi
    Dmat = PadeExpForMyFreqMatrix2(CreateMyFreqMatrix(coupling_matrix, beta_2, alpha, g_0, w, h))

    current_energy = np.array([0] * num, dtype=float)
    for n in range(N - 1):
        pulse = SSFMOrder2(pulse, current_energy, Dmat, gamma, E_sat, g_0, h, tau)
    return pulse


def SimulateClearPropagationCompactNDN(pulse, N, num, h, tau, coupling_matrix, beta_2, gamma, E_sat, alpha, g_0):
    """ Строит решение методом SSFM, строит конечное значение поля в каждой сердцевине """

    M = pulse.shape[1]
    w = fftfreq(M, tau) * 2*pi
    Dmat = PadeExpForMyFreqMatrix2(CreateMyFreqMatrix(coupling_matrix, beta_2, alpha, g_0, w, h))

    current_energy = np.array([0] * num, dtype=float)

    num = len(pulse)
    if g_0 != 0:
        for i in range(num):
            current_energy[i] = GetEnergy_Rectangles(pulse[i], tau)
    NonLinear(pulse, gamma, E_sat, g_0, current_energy, h / 2)

    for n in range(N - 1):
        pulse = FFTforVector(pulse)
        pulse = DispAndCoup(pulse, Dmat)
        pulse = iFFTforVector(pulse)

        if g_0 != 0:
            for i in range(num):
                current_energy[i] = GetEnergy_Rectangles(pulse[i], tau)
        NonLinear(pulse, gamma, E_sat, g_0, current_energy, h)

    if g_0 != 0:
        for i in range(num):
            current_energy[i] = GetEnergy_Rectangles(pulse[i], tau)
    NonLinear(pulse, gamma, E_sat, g_0, current_energy, -h / 2)

    return pulse


def SimulateClearPropagationCompactDND(pulse, N, num, h, tau, coupling_matrix, beta_2, gamma, E_sat, alpha, g_0):
    """ Строит решение методом SSFM, строит конечное значение поля в каждой сердцевине """

    M = pulse.shape[1]
    w = fftfreq(M, tau) * 2 * pi
    DmatH = PadeExpForMyFreqMatrix2(CreateMyFreqMatrix(coupling_matrix, beta_2, alpha, g_0, w, h))
    DmatH2Plus = PadeExpForMyFreqMatrix2(CreateMyFreqMatrix(coupling_matrix, beta_2, alpha, g_0, w, h / 2))
    DmatH2Minus = PadeExpForMyFreqMatrix2(CreateMyFreqMatrix(coupling_matrix, beta_2, alpha, g_0, w, -h / 2))

    pulse = FFTforVector(pulse)
    pulse = DispAndCoup(pulse, DmatH2Plus)
    pulse = iFFTforVector(pulse)

    current_energy = np.array([0] * num, dtype=float)

    for n in range(N - 1):
        if g_0 != 0:
            for i in range(num):
                current_energy[i] = GetEnergy_Rectangles(pulse[i], tau)
        NonLinear(pulse, gamma, E_sat, g_0, current_energy, h)

        pulse = FFTforVector(pulse)
        pulse = DispAndCoup(pulse, DmatH)
        pulse = iFFTforVector(pulse)

    pulse = FFTforVector(pulse)
    pulse = DispAndCoup(pulse, DmatH2Minus)
    pulse = iFFTforVector(pulse)

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


