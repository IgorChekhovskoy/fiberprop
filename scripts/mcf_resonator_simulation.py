from fiberprop.solver import ComputationalParameters, EquationParameters, Solver, CoreConfig, print_matrix

from fiberprop.ssfm_mcf import *
from scipy.fft import fftshift

from tqdm import trange
from fiberprop.drawing import *
from math import pi


def left_condition(pulses, Frensel_refl):
    return pulses * Frensel_refl


def right_condition(omega, pulses, Delta, phi_arr, delta_arr):
    new_pulses = fft(pulses, axis=1)
    for idx, _ in enumerate(new_pulses):
        multiplier = np.exp(1j*phi_arr[idx]) * np.exp(-0.5 * (omega - delta_arr[idx])**2 / Delta**2)
        new_pulses[idx] *= multiplier
    new_pulses = ifft(new_pulses, axis=1)
    return new_pulses


def make_iteration(solver, Delta, phi_arr, delta_arr, Frensel_refl, plot_energy_flag=False, cos_flag='no_cos'):
    """
    Итерация без учёта взаимодействия волн в рамках усиления
    """
    if cos_flag == 'no_cos':
        ref_array = copy.deepcopy(solver.energy)
    elif cos_flag == 'full_cos':
        ref_array = copy.deepcopy(solver.numerical_solution)
    else:
        raise ValueError('cos_flag must be \"no_cos\" or \"full_cos\"')

    solver.numerical_solution[0] = left_condition(solver.numerical_solution[0], Frensel_refl)

    solver.run_resonator_simulation_nocos(ref_array)
    solver.numerical_solution[-1] = right_condition(solver.omega, solver.numerical_solution[-1],
                                                    Delta, phi_arr, delta_arr)

    if cos_flag == 'no_cos':
        ref_array = copy.deepcopy(solver.energy)[:, ::-1]
    else:
        ref_array = copy.deepcopy(solver.numerical_solution)[:, ::-1]
    solver.numerical_solution = solver.numerical_solution[::-1]
    solver.run_resonator_simulation_nocos(ref_array)

    if plot_energy_flag:
        chosen_core = 3
        my_energies = {'last iteration -- backward energy': ref_array[chosen_core, ::-1],
                       'last iteration -- forward energy': solver.energy[chosen_core, ::-1]}
        plot2D_dict(solver.z, my_energies, xlabel='z, m', ylabel='E, pJ', marker_flag=False)

    solver.numerical_solution = solver.numerical_solution[::-1]
    return ref_array


def mcf_resonator_simulation():
    """
    Функция реализует моделирование прохождения прямой волны по резонатору Фабри-Перо
    """
    mcf_size = 7

    # параметры вычислительной сетки
    ComputationalParameters.get_info()
    res_length = 40  # длина резонатора [m]
    my_M = 4 * (mcf_size + 1)
    time_width = 0.5 * my_M / 16  # ширина временного интервала [ps]
    computational_params = ComputationalParameters(N=1000, M=my_M, L1=0.0, L2=res_length,
                                                   T1=-time_width/2, T2=time_width/2)

    # параметры уравнения
    EquationParameters.get_info()
    k = 2*pi * 1e6  # волновое число [1/m]
    c_coef = 1e-7 * k #* 1e-1  # коэффициент связи [1/m]
    g_0 = 10  # коэффициент ненасыщенного усиления [1/m]
    P_sat = 40 * 5e-4  # мощность насыщения [W]
    E_sat = P_sat * time_width  # энергия насыщения [pJ]
    noise_amplitude = 1e-4  # уровень белого равномерного шума, добавляемого на каждом шаге
    equation_params = EquationParameters(core_configuration=CoreConfig.hexagonal, size=mcf_size, ring_number=1,
                                         coupling_coefficient=c_coef, E_sat=E_sat, g_0=g_0, alpha=0.0,
                                         beta2=0.0, gamma=0.0, noise_amplitude=noise_amplitude)

    # параметры граничных условий
    Delta = 6*pi  # полуширина решёток
    phi_arr = np.zeros(equation_params.size, dtype=float)  # фазы отражения
    delta_arr = np.linspace(-24*pi, 24*pi, equation_params.size)  # отстройки пиков отражения от несущей частоты
    # delta_arr = np.array([4, 8, -12, -4, 0, -8, 12], dtype=float) * 2*pi
    Frensel_refl = 0.2

    solver = Solver(computational_params, equation_params, measure_flag=True, pulses=zero_pulse, use_gpu=False)

    # perturbation_scale = 1e-6  # порядок отклонений коэффициента преломления (в сумме у них должен быть ноль)
    # perturbation_array = np.random.uniform(-perturbation_scale, perturbation_scale, equation_params.size)
    # perturbation_array = np.array([-7.54705252e-07, 9.26295029e-07, 9.61144188e-08,
    #                                1.99798946e-07, 6.10540209e-07, 1.65714657e-07,
    #                                -6.18413557e-07], dtype=float)  # случайные числа уроавня 1e-6
    # perturbation_array = np.array([1.000000000000000e-07, -1.816666666666667e-08, -1.766666666666667e-08,
    #                                -1.716666666666667e-08, -1.516666666666667e-08, -1.566666666666667e-08,
    #                                -1.616666666666667e-08], dtype=float)
    perturbation_array = np.zeros(equation_params.size)
    central_idx = equation_params.size//2
    perturbation_array[central_idx] = 0.0
    perturbation_array[central_idx] = - np.sum(perturbation_array)
    solver.set_reflective_index_perturbations(perturbation_array * k)
    print_matrix(solver.linear_coeffs_array, 'Finally coupling matrix')

    #solver.convert_to_dimensionless(coupling_coefficient=c_coef, beta2=0.0, gamma=0.0)  # обезразмериваем

    N_iter = 1000 - 1  # количество итераций в резонаторе
    for _ in trange(N_iter):
        make_iteration(solver, Delta, phi_arr, delta_arr, Frensel_refl)
        #solver.eq.noise_amplitude *= 0.995

    #solver.convert_to_dimensional(coupling_coefficient=c_coef, beta2=0.0, gamma=0.0, print_flag=False)  # возвращаем размерность
    old_solution = copy.deepcopy(solver.numerical_solution)
    old_energy = copy.deepcopy(solver.energy)
    #solver.convert_to_dimensionless(coupling_coefficient=c_coef, beta2=0.0, gamma=0.0, print_flag=False)
    make_iteration(solver, Delta, phi_arr, delta_arr, Frensel_refl, plot_energy_flag=True)
    #solver.convert_to_dimensional(coupling_coefficient=c_coef, beta2=0.0, gamma=0.0)
    current_solution = copy.deepcopy(solver.numerical_solution)
    current_energy = copy.deepcopy(solver.energy)

    chosen_core = 3

    # фаза спектра
    my_spectrum_phase = {f'{N_iter}-iteration': fftshift(np.angle(fft(old_solution[0, chosen_core, :]))),
                         f'{N_iter + 1}-iteration': fftshift(np.angle(fft(current_solution[0, chosen_core, :])))}
    plot2D_dict(fftshift(solver.omega/(2*pi)), my_spectrum_phase,
                xlabel='$\\omega / (2 \\pi)$', ylabel=f'phase($A^-_{chosen_core} (\\omega, z=0)$)', y_min=-pi, y_max=pi)

    # спектр
    my_spectrum_magnitude = {f'{N_iter + 1}-iteration: {idx}-core': fftshift(abs(fft(current_solution[0, idx, :]))) * computational_params.tau/np.sqrt(2*pi)
                             for idx in range(solver.eq.size)}  # /np.sqrt(2*pi)
    plot2D_dict(fftshift(solver.omega/(2*pi)), my_spectrum_magnitude,
                xlabel='$\\omega / (2 \\pi)$', ylabel=f'$|A^-{chosen_core} (\\omega, z=0)|$', y_max=1)
    print(get_energy_rectangles(fftshift(fft(current_solution[0, chosen_core, :])) * computational_params.tau/np.sqrt(2*pi),
                                (solver.omega[1] - solver.omega[0])))
    print(get_energy_rectangles(current_solution[0, chosen_core, :], computational_params.tau))

    # амплитуда
    all_magnitudes_by_t = {f'{N_iter + 1}-iteration: {idx}-core': abs(current_solution[0, idx, :])
                           for idx in range(solver.eq.size)}
    plot2D_multicore(solver.t, all_magnitudes_by_t, xlabel='$time, ps$',
                     ylabel='$|A^-_n (t, z=0)|$')

    all_magnitudes = {f'{N_iter + 1}-iteration: {idx}-core': abs(current_solution[:, idx, solver.com.M//2+1])[::-1]
                      for idx in range(solver.eq.size)}
    plot2D_multicore(solver.z, all_magnitudes, xlabel='z, m', ylabel='$|A^-_n (t=0, z)|$')

    # энергия
    my_energies = {f'{N_iter}-iteration': old_energy[chosen_core, ::-1],
                   f'{N_iter + 1}-iteration': current_energy[chosen_core, ::-1]}
    plot2D_dict(solver.z, my_energies, xlabel='z, m', ylabel='E, pJ', marker_flag=False)

    all_energies = {f'{N_iter + 1}-iteration: {core_idx}-core':
                        [get_energy_rectangles(current_solution[z_idx, core_idx, :], solver.com.tau)
                         for z_idx in range(solver.com.N + 1)]
                    for core_idx in range(solver.eq.size)}
    plot2D_multicore(solver.z, all_energies, xlabel='z, m', ylabel='$\\int |A^-_n (t, z)|^2 dt$')


if __name__ == '__main__':
    mcf_resonator_simulation()
