from fiberprop.solver import (ComputationalParameters, EquationParameters,
                              Solver, CoreConfig, print_matrix)
from fiberprop.drawing import plot2D_multicore, plot2D_dict
from fiberprop.pulses import laser_pulse, zero_pulse
from scipy.interpolate import make_interp_spline
import numpy as np


def get_dn_of_z(eq, com, z_array,
                dn, dn_step, n_ref, k_coef, J_coef, seed=42):
    np.random.seed(seed)
    steps_num = int((com.L2 - com.L1) / dn_step) + 1
    z_nodes = [com.L1 + i * dn_step for i in range(steps_num)]

    beta1_of_z = np.zeros((eq.size, com.N + 1), dtype=float)
    self_coupling_of_z = np.zeros((eq.size, com.N + 1), dtype=float)
    sum_dni_of_z = np.zeros(com.N + 1, dtype=float)
    for i in range(eq.size - 1):
        dni_values = np.random.uniform(-dn, dn, steps_num)
        dni_of_z = make_interp_spline(z_nodes, dni_values, k=3)(z_array)
        self_coupling_of_z[i] = dni_of_z * k_coef
        beta1_of_z[i] = (dni_of_z + n_ref) / (299792458.0 * 1e-12) * J_coef
        sum_dni_of_z += dni_of_z
    self_coupling_of_z[-1] = -sum_dni_of_z * k_coef
    beta1_of_z[-1] = (n_ref - sum_dni_of_z) / (299792458.0 * 1e-12) * J_coef

    return beta1_of_z, self_coupling_of_z


def simulation_of_propagation_in_mcf():
    """
    Функция реализует моделирование прохождения прямой волны по резонатору Фабри-Перо
    """
    mcf_size = 7

    # параметры вычислительной сетки
    ComputationalParameters.get_info()
    fiber_length = 10  # длина волокна [m]
    my_N = round(fiber_length / 0.25e-2)  # разбиение таково,
    # чтобы пространственный шаг составлял 250 мкм
    my_M = 512
    td_width = 5e3  # ширина временного интервала [ps]
    computational_params = ComputationalParameters(N=my_N, M=my_M, L1=0.0, L2=fiber_length,
                                                   T1=-td_width/2, T2=td_width/2)

    # параметры уравнения
    EquationParameters.get_info()
    dn = 1e-4  # изменение показателя преломления между сердцевинами
    dn_step = 1e-2  # интервал через который меняется показатель преломления [m]
    k_coef = 2*np.pi * 1e6  # волновое число [1/m]
    J_coef = 1e-6  # оценка величины интеграла перекрытия
    c_coef = J_coef * k_coef  # коэффициент связи [1/m]
    n_ref = 15.625/10.8 + 2.7e-3
    eq_params = EquationParameters(core_configuration=CoreConfig.hexagonal, size=mcf_size, ring_count=1,
                                   coupling_coefficient=c_coef, E_sat=0.0, g_0=0.0, alpha=0.0,
                                   beta1=1e-3, beta2=0.0, gamma=50e-3, noise_amplitude=0.0)
    # В данной задаче начальное значение "beta1" не играет роли, поскольку перед каждой итерацией
    # численного метода оно пересчитывается из изменения показателя преломления

    input_pulses = [zero_pulse, zero_pulse,
                    zero_pulse, laser_pulse, zero_pulse,
                    zero_pulse, zero_pulse]
    pulse_width = 600  # ps
    peak_power = 0.07 * 11490  # W
    input_pulses_params = [{}, {},
                           {}, {"peak_power": peak_power, "fwhm": pulse_width}, {},
                           {}, {}]
    solver = Solver(computational_params, eq_params, use_dimensional=True,
                    pulses=input_pulses, pulse_params_list=input_pulses_params,
                    display_debug_info=True, use_gpu=False)
    solver.beta1_of_z, solver.self_coupling_of_z = get_dn_of_z(solver.eq, solver.com, solver.z,
                                                               dn, dn_step, n_ref, k_coef, J_coef)

    print(solver.beta1_of_z[:, 0])
    print(solver.self_coupling_of_z[:, 0])

    # обезразмериваем
    # solver.convert_to_dimensionless(coupling_coefficient=c_coef, beta2=0.0, gamma=0.0)

    solver.run_numerical_simulation()


    # возвращаем размерность
    # solver.convert_to_dimensional(coupling_coefficient=c_coef, beta2=0.0, gamma=0.0,
    #                               print_flag=False)

    visualize_res(solver)  # визуализация результатов


def visualize_res(solver):
    current_solution = np.copy(solver.numerical_solution)
    # интенсивность на входе
    all_magnitudes_by_t = {f'{idx}-core': abs(current_solution[0, idx, :])**2
                           for idx in range(solver.eq.size)}
    plot2D_multicore(solver.t, all_magnitudes_by_t, xlabel='time, ps',
                     ylabel='$|A_n \\text{(t, z=0)}|^2$, W')
    # интенсивность на выходе
    all_magnitudes_by_t = {f'{idx}-core': abs(current_solution[-1, idx, :])**2
                           for idx in range(solver.eq.size)}
    plot2D_multicore(solver.t, all_magnitudes_by_t, xlabel='time, ps',
                     ylabel='$|A_n \\text{(t, z=end)}|^2$, W')
    # интенсивность в пике от z
    all_magnitudes = {f'{idx}-core': abs(current_solution[:, idx, solver.com.M // 2])**2
                      for idx in range(solver.eq.size)}
    plot2D_dict(solver.z, all_magnitudes, xlabel='z, m', ylabel='$|A_n \\text{(t=0, z)}|^2$, W',
                marker_flag=False)
    # средняя мощность от z
    all_mean_powers = {f'{idx}-core': np.sum(abs(current_solution[:, idx, :])**2, axis=1) * solver.com.tau / solver.com.T2 * 4.0
                       for idx in range(solver.eq.size)}
    plot2D_dict(solver.z, all_mean_powers, xlabel='z, m', ylabel='$I_n \\text{(z)}$, W',
                marker_flag=False)


if __name__ == "__main__":
    simulation_of_propagation_in_mcf()

