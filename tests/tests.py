from fiberprop.drawing import *
from fiberprop.matrices import get_ring_coupling_matrix
from fiberprop.pulses import fundamental_soliton, gain_loss_soliton
from fiberprop.solver import ComputationalParameters, EquationParameters, Solver, CoreConfig


def test_case1_using_classes(plot=True):
    computational_params = ComputationalParameters(N=100, M=2**10, L1=0, L2=1, T1=-25, T2=25)
    equation_params = EquationParameters(core_configuration=CoreConfig.empty_ring, size=7, beta2=-1.0, gamma=1.0, E_sat=1.0, alpha=0.1, g_0=0.4)

    solver = Solver(computational_params, equation_params, pulses=gain_loss_soliton, use_gpu=True)
    solver.run_test()

    plot2D_plotly(solver.z, solver.absolute_error, names=["error"])
    # plot3D_plotly(solver.t, solver.z, solver.absolute_error, "absolute error")
    # plot3D_plotly(solver.z, solver.t, np.abs(solver.numerical_solution[:, 3, :]) ** 2, "U(z,t)")


def test_case2(plot=True):
    """ Строит поле мощности решения методом SSFM, начальные данные - не подходящий солитон, кольцевая структура """

    N = 100 - 1  # количество шагов, а не точек
    M = 2**10  # количество учитываемых точек по времени
    L1, L2 = 0, 1
    T1, T2 = -25, 25

    num = 7
    beta2 = np.full(num, -1.0, dtype=float)
    gamma = np.full(num, 1.0, dtype=float)
    E_sat = np.full(num, 1.0, dtype=float)
    alpha = np.full(num, 0.1, dtype=float)
    g_0 = np.full(num, 0.4, dtype=float)

    h = (L2 - L1) / N
    tau = (T2 - T1) / M

    t = np.linspace(T1, T2, M, endpoint=False)
    omega = fftfreq(M, tau) * 2*pi
    
    coupling_matrix = get_ring_coupling_matrix(num)
    D = get_pade_exponential2(create_freq_matrix(coupling_matrix, beta2, alpha, g_0, omega, h))
    numerical_solution = np.zeros((num, N+1, M), dtype=complex)
    current_energy = np.array([0] * num, dtype=float)

    # начальные условия
    old = np.zeros((num, M), dtype=complex)
    for k in range(num):
        old[k] = 2 * fundamental_soliton(t, 0, beta2[0])
        numerical_solution[k][0] = old[k]

    # итерации численного метода
    for n in trange(N):
        new = ssfm_order2(old, current_energy, D, gamma, E_sat, g_0, h, tau)
        for i in range(num):
            numerical_solution[i][n+1] = new[i]
        old = new

    # вывод поля мощности
    if plot:
        z = np.linspace(L1, L2, N+1)
        name = 'поле_мощности-case2'
        power_field = abs(numerical_solution[num//2])**2
        plot3D_plotly(t, z, power_field, name)
    return numerical_solution


def test_case3(plot=True):
    """ Строит поле ошибок метода SSFM, начальные данные - солитон, кольцевая структура """

    N = 100 - 1  # количество шагов, а не точек
    M = 2**10  # количество учитываемых точек по времени
    L1, L2 = 0, 1
    T1, T2 = -25, 25

    num = 7
    beta2 = np.full(num, -1.0, dtype=float)
    gamma = np.full(num, 1.0, dtype=float)
    E_sat = np.full(num, 1.0, dtype=float)
    alpha = np.full(num, 0.1, dtype=float)
    g_0 = np.full(num, 0.4, dtype=float)

    h = (L2 - L1) / N
    tau = (T2 - T1) / M

    t = np.linspace(T1, T2, M, endpoint=False)
    
    coupling_matrix = get_ring_coupling_matrix(num)

    # начальные условия
    input_pulse = np.zeros((num, M), dtype=complex)
    scalsr_equation_parameters = {'beta2': beta2[0], 'gamma': gamma[0], 'E_sat': E_sat[0], 'alpha': alpha[0], 'g_0': g_0[0]}
    equation_parameters = {'beta2': beta2, 'gamma': gamma, 'E_sat': E_sat, 'alpha': alpha, 'g_0': g_0}
    for k in range(num):
        input_pulse[k] = gain_loss_soliton(t=t, z=0, **scalsr_equation_parameters)

    # итерации численного метода
    # output_pulse = SimulatePropagation(input_pulse, N, num, h, tau, coupling_matrix, **equation_parameters)
    # output_pulse = SimulatePropagationNDN(input_pulse, N, num, h, tau, coupling_matrix, **equation_parameters)
    output_pulse = simulate_propagation_dnd(input_pulse, N, num, h, tau, coupling_matrix, **equation_parameters)

    # аналитическое решение
    analytical_output = gain_loss_soliton(t=t, z=h*N, **scalsr_equation_parameters)

    # вычисление ошибки
    absolute_error = abs(analytical_output - output_pulse[num//2])  # берём одну сердцевину из численного решения
    C_norm = max(absolute_error)
    print('C norm =\t', C_norm)
    L2_norm = get_energy_rectangles(absolute_error**2, tau)
    print('L2 norm =\t', L2_norm)

    # вывод графика ошибки
    if plot:
        name = 'абсолютная_ошибка-case3'
        plot2D(t, absolute_error, name)
    return absolute_error


def test_case_MCF_2core(N, M, num, beta2, gamma, E_sat, alpha, g_0, L1, L2, T1, T2, plot=True):
    """ Строит поле ошибок метода SSFM, начальные данные - солитон, кольцевая структура """
    N = 1000 - 1  # количество шагов, а не точек
    M = 2**10  # количество учитываемых точек по времени
    num = 2
    beta2 = -1.0
    gamma = 1.0
    E_sat = 1.0
    alpha = 0.1
    g_0 = 1
    L1, L2 = 0, 5
    T1, T2 = -10, 10

    t = np.linspace(T1, T2, M, endpoint=False)
    h = (L2 - L1) / N
    tau = (T2 - T1) / M
    coupling_matrix = get_ring_coupling_matrix(num)

    # начальные условия
    input_pulse = np.zeros((num, M), dtype=complex)
    equation_parameters = {'beta2': beta2, 'gamma': gamma, 'E_sat': E_sat, 'alpha': alpha, 'g_0': g_0}

    for k in range(num):
        input_pulse[k] = fundamental_soliton(t, 0, beta2, lamb=1, c=1)

    # итерации численного метода
    output_pulse = simulate_propagation(input_pulse, N, num, h, tau, coupling_matrix, **equation_parameters)
    # output_pulse = SimulatePropagationCompactNDN(input_pulse, N, num, h, tau, coupling_matrix, **equation_parameters)
    # output_pulse = SimulatePropagationCompactDND(input_pulse, N, num, h, tau, coupling_matrix, **equation_parameters)

    # аналитическое решение
    analytical_output = gain_loss_soliton(t=t, z=h*N, **equation_parameters)

    # вычисление ошибки
    absolute_error = abs(analytical_output - output_pulse[num//2])  # берём одну сердцевину из численного решения
    C_norm = max(absolute_error)
    print('C norm =\t', C_norm)
    L2_norm = get_energy_rectangles(absolute_error**2, tau)
    print('L2 norm =\t', L2_norm)

    # вывод поля мощности
    if plot:
        z = np.linspace(L1, L2, N+1)
        T_grid, Z_grid = np.meshgrid(t, z)
        name = 'поле_мощности-case2'
        power_field = abs(numerical_solution[num // 2]) ** 2
        plot3D(Z_grid, T_grid, power_field, name)
    return numerical_solution


if __name__ == '__main__':
    test_case2()
