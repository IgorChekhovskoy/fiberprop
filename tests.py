from scipy.fft import fftfreq
from propagation import *
from drawing import *


def test_case1(N, M, num, beta_2, gamma, E_sat, alpha, g_0, a, b, c, d, plot=True):
    """ Строит поле ошибок метода SSFM, начальные данные - солитон, кольцевая структура """
    t = np.linspace(c, d, M)
    h = (b - a) / (N - 1)
    tau = (d - c) / (M - 1)
    w = fftfreq(M - 1, tau) * 2*pi
    coupMatrix = GetRingCouplingMatrix(num)
    D = PadeExpForMyFreqMatrix(CreateMyFreqMatrix(coupMatrix, beta_2, alpha, g_0, w, h))
    numericalSolution = np.zeros((num, N, M-1), dtype=complex)
    currentEnergy = np.array([0] * num, dtype=float)

    # начальные условия
    old = np.zeros((num, M - 1), dtype=complex)
    params_dict = {'beta_2': beta_2, 'gamma': gamma, 'E_sat': E_sat, 'alpha': alpha, 'g_0': g_0}
    for k in range(num):
        old[k] = GainLossSoliton(t=t[:-1], x=0, **params_dict)
        numericalSolution[k][0] = old[k]

    # итерации численного метода
    for n in trange(N - 1):
        new = SSFMOrder2(old, currentEnergy, D, gamma, E_sat, g_0, h, tau)
        for i in range(num):
          numericalSolution[i][n+1] = new[i]
        old = new

    # аналитическое решение
    z = np.linspace(a, b, N)
    analyticalSolution = GetAnalyticalField(params_dict, t[:-1], z, GainLossSoliton)

    # вычисление ошибки
    absolutError = abs(analyticalSolution - numericalSolution[num//2])  # берём одну сердцевину из численного решения
    C_norm = max(absolutError[N-1])
    print('C norm =\t', C_norm)
    L2_norm = GetEnergy_Rectangles(absolutError[N - 1]**2, tau)
    print('L2 norm =\t', L2_norm)

    # вывод графика ошибки
    if plot:
        T_grid, Z_grid = np.meshgrid(t[:-1], z)
        name = 'абсолютная_ошибка-case1'
        plot3D(Z_grid, T_grid, absolutError, name)
    return absolutError


def test_case2(N, M, num, beta_2, gamma, E_sat, alpha, g_0, a, b, c, d, plot=True):
    """ Строит поле мощности решения методом SSFM, начальные данные - не подходящий солитон, кольцевая структура """
    t = np.linspace(c, d, M)
    h = (b - a) / (N - 1)
    tau = (d - c) / (M - 1)
    w = fftfreq(M - 1, tau) * 2*pi
    coupMatrix = GetRingCouplingMatrix(num)
    D = PadeExpForMyFreqMatrix(CreateMyFreqMatrix(coupMatrix, beta_2, alpha, g_0, w, h))
    numericalSolution = np.zeros((num, N, M-1), dtype=complex)
    currentEnergy = np.array([0] * num, dtype=float)

    # начальные условия
    old = np.zeros((num, M - 1), dtype=complex)
    for k in range(num):
        old[k] = ElementarySoliton(t[:-1], 0, beta_2)
        numericalSolution[k][0] = old[k]

    # итерации численного метода
    for n in trange(N - 1):
        new = SSFMOrder2(old, currentEnergy, D, gamma, E_sat, g_0, h, tau)
        for i in range(num):
          numericalSolution[i][n+1] = new[i]
        old = new

    # вывод поля мощности
    if plot:
        z = np.linspace(a, b, N)
        T_grid, Z_grid = np.meshgrid(t[:-1], z)
        name = 'поле_мощности-case2'
        powerField = abs(numericalSolution[num//2])**2
        plot3D(Z_grid, T_grid, powerField, name)
    return numericalSolution


def test_case3(N, M, num, beta_2, gamma, E_sat, alpha, g_0, a, b, c, d, plot=True):
    """ Строит поле ошибок метода SSFM, начальные данные - солитон, кольцевая структура """
    t = np.linspace(c, d, M)
    h = (b - a) / (N - 1)
    tau = (d - c) / (M - 1)
    w = fftfreq(M - 1, tau) * 2*pi
    coupMatrix = GetRingCouplingMatrix(num)
    Dmat = PadeExpForMyFreqMatrix(CreateMyFreqMatrix(coupMatrix, beta_2, alpha, g_0, w, h))

    # начальные условия
    input_pulse = np.zeros((num, M - 1), dtype=complex)
    params_dict = {'beta_2': beta_2, 'gamma': gamma, 'E_sat': E_sat, 'alpha': alpha, 'g_0': g_0}
    for k in range(num):
        input_pulse[k] = GainLossSoliton(t=t[:-1], x=0, **params_dict)

    # итерации численного метода
    output_pulse = SimulateClearPropagation(input_pulse, N, num, Dmat, gamma, E_sat, g_0, h, tau)

    # аналитическое решение
    analyticalOutput = GainLossSoliton(t=t[:-1], x=h*(N-1), **params_dict)

    # вычисление ошибки
    absolutError = abs(analyticalOutput - output_pulse[num//2])  # берём одну сердцевину из численного решения
    C_norm = max(absolutError)
    print('C norm =\t', C_norm)
    L2_norm = GetEnergy_Rectangles(absolutError**2, tau)
    print('L2 norm =\t', L2_norm)

    # вывод графика ошибки
    if plot:
        name = 'абсолютная_ошибка-case3'
        plot2D(t[:-1], absolutError, name)
    return absolutError

