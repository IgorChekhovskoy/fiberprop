from propagation import *
from drawing import *


def test_case1(N, M, num, beta_2, gamma, E_sat, alpha, g_0, L1, L2, T1, T2, plot=True):
    """ Строит поле ошибок метода SSFM, начальные данные - солитон, кольцевая структура """
    t = np.linspace(T1, T2, M)
    h = (L2 - L1) / (N - 1)
    tau = (T2 - T1) / (M - 1)
    w = fftfreq(M - 1, tau) * 2*pi
    coupling_matrix = GetRingCouplingMatrix(num)
    D = PadeExpForMyFreqMatrix(CreateMyFreqMatrix(coupling_matrix, beta_2, alpha, g_0, w, h))
    numericalSolution = np.zeros((num, N, M-1), dtype=complex)
    currentEnergy = np.array([0] * num, dtype=float)

    # начальные условия
    old = np.zeros((num, M - 1), dtype=complex)
    equation_parameters = {'beta_2': beta_2, 'gamma': gamma, 'E_sat': E_sat, 'alpha': alpha, 'g_0': g_0}
    for k in range(num):
        old[k] = GainLossSoliton(t=t[:-1], x=0, **equation_parameters)
        numericalSolution[k][0] = old[k]

    # итерации численного метода
    for n in trange(N - 1):
        new = SSFMOrder2(old, currentEnergy, D, gamma, E_sat, g_0, h, tau)
        for i in range(num):
          numericalSolution[i][n+1] = new[i]
        old = new

    # аналитическое решение
    z = np.linspace(L1, L2, N)
    analyticalSolution = GetAnalyticalField(equation_parameters, t[:-1], z, GainLossSoliton)

    # вычисление ошибки
    absolute_error = abs(analyticalSolution - numericalSolution[num//2])  # берём одну сердцевину из численного решения
    C_norm = max(absolute_error[N-1])
    print('C norm =\t', C_norm)
    L2_norm = GetEnergy_Rectangles(absolute_error[N - 1]**2, tau)
    print('L2 norm =\t', L2_norm)

    # вывод графика ошибки
    if plot:
        T_grid, Z_grid = np.meshgrid(t[:-1], z)
        name = 'абсолютная_ошибка-case1'
        plot3D(Z_grid, T_grid, absolute_error, name)
    return absolute_error


def test_case2(N, M, num, beta_2, gamma, E_sat, alpha, g_0, L1, L2, T1, T2, plot=True):
    """ Строит поле мощности решения методом SSFM, начальные данные - не подходящий солитон, кольцевая структура """
    t = np.linspace(T1, T2, M)
    h = (L2 - L1) / (N - 1)
    tau = (T2 - T1) / (M - 1)
    w = fftfreq(M - 1, tau) * 2*pi
    coupling_matrix = GetRingCouplingMatrix(num)
    D = PadeExpForMyFreqMatrix(CreateMyFreqMatrix(coupling_matrix, beta_2, alpha, g_0, w, h))
    numericalSolution = np.zeros((num, N, M-1), dtype=complex)
    currentEnergy = np.array([0] * num, dtype=float)

    # начальные условия
    old = np.zeros((num, M - 1), dtype=complex)
    for k in range(num):
        old[k] = 2*ElementarySoliton(t[:-1], 0, beta_2)
        numericalSolution[k][0] = old[k]

    # итерации численного метода
    for n in trange(N - 1):
        new = SSFMOrder2(old, currentEnergy, D, gamma, E_sat, g_0, h, tau)
        for i in range(num):
          numericalSolution[i][n+1] = new[i]
        old = new

    # вывод поля мощности
    if plot:
        z = np.linspace(L1, L2, N)
        T_grid, Z_grid = np.meshgrid(t[:-1], z)
        name = 'поле_мощности-case2'
        powerField = abs(numericalSolution[num//2])**2
        plot3D(Z_grid, T_grid, powerField, name)
    return numericalSolution

def test_case2_2(N, M, num, beta_2, gamma, E_sat, alpha, g_0, L1, L2, T1, T2, plot=True):
    """ Строит поле мощности решения методом SSFM, начальные данные - не подходящий солитон, кольцевая структура """
    t = np.linspace(T1, T2, M)
    h = (L2 - L1) / (N - 1)
    tau = (T2 - T1) / (M - 1)
    w = fftfreq(M - 1, tau) * 2*pi
    coupling_matrix = GetRingCouplingMatrix(num)
    D = PadeExpForMyFreqMatrix(CreateMyFreqMatrix(coupling_matrix, beta_2, alpha, g_0, w, h/2))
    numericalSolution = np.zeros((num, N, M-1), dtype=complex)
    currentEnergy = np.array([0] * num, dtype=float)

    # начальные условия
    old = np.zeros((num, M - 1), dtype=complex)
    for k in range(num):
        old[k] = ElementarySoliton(t[:-1], 0, beta_2)
        numericalSolution[k][0] = old[k]

    # итерации численного метода
    for n in trange(N - 1):
        new = SSFMOrder2_2(old, currentEnergy, D, gamma, E_sat, g_0, h, tau)
        for i in range(num):
          numericalSolution[i][n+1] = new[i]
        old = new

    # вывод поля мощности
    if plot:
        z = np.linspace(L1, L2, N)
        T_grid, Z_grid = np.meshgrid(t[:-1], z)
        name = 'поле_мощности-case2'
        powerField = abs(numericalSolution[num//2])**2
        plot3D(Z_grid, T_grid, powerField, name)
    return numericalSolution

def test_case3(N, M, num, beta_2, gamma, E_sat, alpha, g_0, L1, L2, T1, T2, plot=True):
    """ Строит поле ошибок метода SSFM, начальные данные - солитон, кольцевая структура """
    t = np.linspace(T1, T2, M)
    h = (L2 - L1) / (N - 1)
    tau = (T2 - T1) / (M - 1)
    coupling_matrix = GetRingCouplingMatrix(num)

    # начальные условия
    input_pulse = np.zeros((num, M - 1), dtype=complex)
    equation_parameters = {'beta_2': beta_2, 'gamma': gamma, 'E_sat': E_sat, 'alpha': alpha, 'g_0': g_0}
    for k in range(num):
        input_pulse[k] = GainLossSoliton(t=t[:-1], x=0, **equation_parameters)

    # итерации численного метода
    # output_pulse = SimulateClearPropagation(input_pulse, N, num, h, tau, coupling_matrix, **equation_parameters)
    # output_pulse = SimulateClearPropagationCompactNDN(input_pulse, N, num, h, tau, coupling_matrix, **equation_parameters)
    output_pulse = SimulateClearPropagationCompactDND(input_pulse, N, num, h, tau, coupling_matrix, **equation_parameters)

    # аналитическое решение
    analytical_output = GainLossSoliton(t=t[:-1], x=h*(N-1), **equation_parameters)

    # вычисление ошибки
    absolute_error = abs(analytical_output - output_pulse[num//2])  # берём одну сердцевину из численного решения
    C_norm = max(absolute_error)
    print('C norm =\t', C_norm)
    L2_norm = GetEnergy_Rectangles(absolute_error**2, tau)
    print('L2 norm =\t', L2_norm)

    # вывод графика ошибки
    if plot:
        name = 'абсолютная_ошибка-case3'
        plot2D(t[:-1], absolute_error, name)
    return absolute_error


def test_case_MCF_2core(N, M, num, beta_2, gamma, E_sat, alpha, g_0, L1, L2, T1, T2, plot=True):
    """ Строит поле ошибок метода SSFM, начальные данные - солитон, кольцевая структура """
    N = 1000
    M = 2 ** 10 + 1
    num = 2
    beta_2 = -1.0
    gamma = 1.0
    E_sat = 1.0
    alpha = 0.1
    g_0 = 1
    L1, L2 = 0, 5
    T1, T2 = -10, 10

    t = np.linspace(T1, T2, M)
    h = (L2 - L1) / (N - 1)
    tau = (T2 - T1) / (M - 1)
    coupling_matrix = GetRingCouplingMatrix(num)

    # начальные условия
    input_pulse = np.zeros((num, M - 1), dtype=complex)
    equation_parameters = {'beta_2': beta_2, 'gamma': gamma, 'E_sat': E_sat, 'alpha': alpha, 'g_0': g_0}

    for k in range(num):
        input_pulse[k] = fundamental_soliton(t[:-1], 0, beta_2, lamb=1, c=1)

    # итерации численного метода
    output_pulse = SimulateClearPropagation(input_pulse, N, num, h, tau, coupling_matrix, **equation_parameters)
    # output_pulse = SimulateClearPropagationCompactNDN(input_pulse, N, num, h, tau, coupling_matrix, **equation_parameters)
    # output_pulse = SimulateClearPropagationCompactDND(input_pulse, N, num, h, tau, coupling_matrix, **equation_parameters)

    # аналитическое решение
    analytical_output = GainLossSoliton(t=t[:-1], x=h*(N-1), **equation_parameters)

    # вычисление ошибки
    absolute_error = abs(analytical_output - output_pulse[num//2])  # берём одну сердцевину из численного решения
    C_norm = max(absolute_error)
    print('C norm =\t', C_norm)
    L2_norm = GetEnergy_Rectangles(absolute_error**2, tau)
    print('L2 norm =\t', L2_norm)

    # вывод поля мощности
    if plot:
        z = np.linspace(L1, L2, N)
        T_grid, Z_grid = np.meshgrid(t[:-1], z)
        name = 'поле_мощности-case2'
        powerField = abs(numericalSolution[num // 2]) ** 2
        plot3D(Z_grid, T_grid, powerField, name)
    return numericalSolution
