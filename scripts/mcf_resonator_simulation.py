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


def make_iteration(solver, Delta, phi_arr, delta_arr, Frensel_refl):
    """
    Итерация без учёта взаимодействия волн в рамках усиления
    """
    # TODO: распространение, правое условие, распространение, левон условие
    #  (надо добавить изменения в нелинейность и в усиление, сумму прямой и обратной волн)
    pass


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

    N_iter = 1000  # количество итераций в резонаторе
    counter = 0
    while counter < N_iter:
        make_iteration(solver, Delta, phi_arr, delta_arr, Frensel_refl)

    solver.convert_to_dimensional(coupling_coefficient=c_coef, beta2=0.0, gamma=0.0)  # возвращаем размерность

    # TODO: графики строятся пока непоказательные (просто какие-то)
    energies = [solver.energy[i, :] for i in range(solver.eq.size)]
    names = [f'$E_{{{i}}}$' for i in range(solver.eq.size)]
    plot2D_plotly(solver.z, energies, title_text='Динамика энергии',
                  names=names, x_axis_label='z [m]', y_axis_label='energy [pJ]')

    peak_powers = [solver.peak_power[i, :] for i in range(solver.eq.size)]
    names = [f'$P_{{{i}}}$' for i in range(solver.eq.size)]
    plot2D_plotly(solver.z, peak_powers, title_text='Динамика пиковой мощности',
                  names=names, x_axis_label='z [m]', y_axis_label='peak power [W]')

    plot2D_plotly(solver.t, [np.abs(solver.numerical_solution[0][3]) ** 2,
                             np.abs(solver.numerical_solution[solver.com.N][3]) ** 2],
                  title_text='Сравнение профиля импульса в начале и в конце',
                  names=[f"$|U_3(z=0,t)|^2$", f"$|U_3(z=L,t)|^2$"], x_axis_label='t [ps]', y_axis_label='power [W]')


if __name__ == '__main__':
    mcf_resonator_simulation()
