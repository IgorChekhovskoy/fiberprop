from fiberprop.solver import ComputationalParameters, EquationParameters, Solver, CoreConfig

from fiberprop.propagation import *
from fiberprop.ssfm_mcf import *


def left_condition(pulses, Frensel_refl):
    return pulses * Frensel_refl


def right_condition(solver, pulses, Delta, phi_arr, delta_arr):
    new_pulses = fft(pulses, axis=1)
    for idx, _ in enumerate(new_pulses):
        multiplier = np.exp(1j*phi_arr[idx]) * np.exp(-0.5 * (solver.omega - delta_arr[idx])**2 / Delta**2)
        new_pulses[idx] *= multiplier
    new_pulses = ifft(new_pulses, axis=1)
    return new_pulses


def make_iteration(solver, Delta, phi_arr, delta_arr, Frensel_refl, cos_flag='no_cos'):
    """
    Итерация без учёта взаимодействия волн в рамках усиления
    """
    if cos_flag == 'no_cos':
        ref_array = copy.deepcopy(solver.energy)
    elif cos_flag == 'full_cos':
        ref_array = copy.deepcopy(solver.numerical_solution)
    else:
        raise ValueError('cos_flag must be \"no_cos\" or \"full_cos\"')
    solver.run_resonator_simulation_nocos(ref_array, print_modulus=False, print_interval=10)

    solver.numerical_solution[-1] = right_condition(solver, solver.numerical_solution[-1], Delta, phi_arr, delta_arr)

    if cos_flag == 'no_cos':
        ref_array = copy.deepcopy(solver.energy)[:, ::-1]
    else:
        ref_array = copy.deepcopy(solver.numerical_solution)[:, ::-1]
    solver.numerical_solution = solver.numerical_solution[::-1]
    solver.run_resonator_simulation_nocos(ref_array, print_modulus=False, print_interval=10)

    solver.numerical_solution[-1] = left_condition(solver.numerical_solution[-1], Frensel_refl)
    solver.numerical_solution = solver.numerical_solution[::-1]


def mcf_resonator_simulation():
    """
    Функция реализует моделирование прохода прямой волны по резонатору Фабри-Перо
    """
    # параметры вычислительной сетки
    ComputationalParameters.get_info()
    res_length = 40  # длина резонатора [m]
    time_width = 60  # ширина временного интервала [ps]
    computational_params = ComputationalParameters(N=500, M=2**10, L1=0.0, L2=res_length,
                                                   T1=-time_width/2, T2=time_width/2)

    # параметры уравнения
    EquationParameters.get_info()
    k = 2*pi * 1e6 * 1e-2  # волновое число [1/cm]
    c_coef = 1e-7 * k  # коэффициент связи [1/cm]
    g_0 = 10  # коэффициент ненасыщенного усиления [1/m]
    P_sat = 40 * 5e-4  # мощность насыщения [W]
    E_sat = P_sat * time_width  # энергия насыщения [pJ]
    noise_amplitude = 1e-2 / computational_params.N  # уровень белого равномерного шума, добавляемого на каждом шаге
    equation_params = EquationParameters(core_configuration=CoreConfig.hexagonal, size=7, ring_number=1,
                                         coupling_coefficient=c_coef, E_sat=E_sat, g_0=g_0, alpha=0.0,
                                         beta2=0.0, gamma=0.0, noise_amplitude=noise_amplitude)
    # TODO: не работает, потому что в случае beta2 = 0 матрица D является квадратной, а должна быть одномерным массивом
    #  (надо обработать)

    # параметры граничных условий
    Delta = 6*pi  # полуширина решёток
    phi_arr = np.zeros(equation_params.size, dtype=float)  # фазы отражения
    delta_arr = np.linspace(-24*pi, 24*pi, equation_params.size)  # отстройки пиков отражения от несущей частоты
    Frensel_refl = 0.2

    solver = Solver(computational_params, equation_params, measure_flag=True, pulses=zero_pulse, use_gpu=False)

    perturbation_scale = 1e-6  # порядок отклонений коэффициента преломления (в сумме у них должен быть ноль)
    perturbation_array = np.random.uniform(-perturbation_scale, perturbation_scale, equation_params.size)
    central_idx = equation_params.size//2
    perturbation_array[central_idx] = 0.0
    perturbation_array[central_idx] = -1.0 * np.sum(perturbation_array)
    solver.set_reflective_index_perturbations(perturbation_array * k)

    solver.convert_to_dimensionless(coupling_coefficient=c_coef, beta2=0.0, gamma=0.0)  # обезразмериваем

    N_iter = 1000 - 1  # количество итераций в резонаторе
    for _ in trange(N_iter):
        make_iteration(solver, Delta, phi_arr, delta_arr, Frensel_refl)

    solver.convert_to_dimensional(coupling_coefficient=c_coef, beta2=0.0, gamma=0.0)  # возвращаем размерность
    old_solution = copy.deepcopy(solver.numerical_solution)
    old_energy = copy.deepcopy(solver.energy)
    solver.convert_to_dimensionless(coupling_coefficient=c_coef, beta2=0.0, gamma=0.0)
    make_iteration(solver, Delta, phi_arr, delta_arr, Frensel_refl)
    solver.convert_to_dimensional(coupling_coefficient=c_coef, beta2=0.0, gamma=0.0)
    current_solution = copy.deepcopy(solver.numerical_solution)
    current_energy = copy.deepcopy(solver.energy)

    chosen_core = 2
    my_energies = {f'{N_iter}-iteration': old_energy[chosen_core, :],
                   f'{N_iter+1}-iteration': current_energy[chosen_core, :]}
    plot2D_dict(solver.z, my_energies, xlabel='z, m', ylabel='E, pJ')

    my_spectrum_phase = {f'{N_iter}-iteration': np.angle(fft(old_solution[0, chosen_core, :])),
                         f'{N_iter + 1}-iteration': np.angle(fft(current_solution[0, chosen_core, :]))}
    plot2D_dict(solver.omega/(2*pi), my_spectrum_phase, xlabel='$\\omega / (2 \\pi)$', ylabel='phase($A^-_2 (\\omega, z=0)$)')

    my_spectrum_magnitude = {f'{N_iter + 1}-iteration: {idx}-core': abs(fft(current_solution[0, chosen_core, :]))
                             for idx in range(solver.eq.size)}
    plot2D_dict(solver.omega/(2*pi), my_spectrum_magnitude, xlabel='$\\omega / (2 \\pi)$', ylabel='$|A^-_2 (\\omega, z=0)|$')

    all_spectrum_magnitudes = {f'{N_iter}-iteration': abs(fft(old_solution[0, chosen_core, :])),
                         f'{N_iter + 1}-iteration': abs(fft(current_solution[0, chosen_core, :]))}
    plot2D_multicore(solver.omega/(2*pi), all_spectrum_magnitudes, xlabel='$\\omega / (2 \\pi)$',
                ylabel='$|A^-_n (\\omega, z=0)|$')

    all_magnitudes = {f'{N_iter + 1}-iteration: {idx}-core': abs(current_solution[:, idx, solver.com.M//2+1])**2
                  for idx in range(solver.eq.size)}
    plot2D_multicore(solver.z, all_magnitudes, xlabel='z, m', ylabel='$|A^-_n (t=0, z)|$')


if __name__ == '__main__':
    mcf_resonator_simulation()
