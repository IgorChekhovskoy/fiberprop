from mcf_resonator_simulation import *
import matplotlib.animation as animation


def mcf_resonator_EvoVideo():
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
    c_coef = 1e-7 * k  # коэффициент связи [1/m]
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
    # delta_arr = np.array([4, 8, -12, -4, 0, -8, 12], dtype=float) * 2 * pi
    Frensel_refl = 0.2

    solver = Solver(computational_params, equation_params, measure_flag=True, pulses=zero_pulse, use_gpu=False)

    # perturbation_scale = 1e-6  # порядок отклонений коэффициента преломления (в сумме у них должен быть ноль)
    # perturbation_array = np.random.uniform(-perturbation_scale, perturbation_scale, equation_params.size)
    # perturbation_array = np.array([-7.54705252e-07, 9.26295029e-07, 9.61144188e-08,
    #                                1.99798946e-07, 6.10540209e-07, 1.65714657e-07,
    #                                -6.18413557e-07], dtype=float)  # c_coef / (k * 1e-7)
    perturbation_array = np.zeros(equation_params.size)
    central_idx = equation_params.size//2
    perturbation_array[central_idx] = 0.0
    perturbation_array[central_idx] = - np.sum(perturbation_array)
    solver.set_reflective_index_perturbations(perturbation_array * k)
    print_matrix(solver.linear_coeffs_array, 'Finally coupling matrix')

    N_iter = 200  # количество итераций в резонаторе
    chosen_core = 3  # ядро, которое выводится на некоторых графиках отдельно от оставльных
    chosen_EVO = animate_energy_z  # animate_energy_z, animate_spectrum, animate_magnitude
    fig, ax = get_artist(chosen_EVO, solver.eq.size, y_min=0, y_max=10)

    video_saving_path = 'C:\\Users\\Georgiy\\Desktop\\'
    output_file_name = 'evolution_' + chosen_EVO.__name__ + '.gif'
    my_fps = 4
    anim_func = lambda iter: chosen_EVO(fig, ax, solver, Delta, phi_arr, delta_arr, Frensel_refl,
                                        iter, chosen_core)
    anim = animation.FuncAnimation(fig, anim_func, interval=my_fps*1000, frames=N_iter)
    my_writer = animation.PillowWriter(fps=my_fps)
    anim.save(video_saving_path + output_file_name, writer=my_writer,
              progress_callback=lambda i, n: print(f'{round(100*i/N_iter, 2)}%'))


def get_artist(animation_func, figures_num, y_min=0, y_max=10):
    func_name = animation_func.__name__
    if func_name == 'animate_phase':
        return create_single_ax(xlabel='$\\omega / (2 \\pi)$', ylabel='phase($A^-_2 (\\omega, z=0)$)',
                                y_min=y_min, y_max=y_max)
    elif func_name == 'animate_spectrum':
        return create_single_ax(xlabel='$\\omega / (2 \\pi)$', ylabel='$|A^-_2 (\\omega, z=0)|$',
                                y_min=y_min, y_max=y_max)
    elif func_name == 'animate_magnitude':
        return create_multicore_axs(figures_num, xlabel='$time, ps$', ylabel='$|A^-_n (t, z=0)|$',
                                    y_min=y_min, y_max=y_max)
    elif func_name == 'animate_energy_z':
        return create_single_ax(xlabel='z, m', ylabel='E, pJ', y_min=y_min, y_max=y_max)
    elif func_name == 'animate_energy_multicore':
        return create_multicore_axs(figures_num, xlabel='z, m', ylabel='$\\int |A^-_n (t, z)|^2 dt$',
                                    y_min=y_min, y_max=y_max)
    else:
        raise ValueError('Unsupportable animation function')


def animate_phase(fig, ax, solver, Delta, phi_arr, delta_arr, Frensel_refl,
                  current_iter, chosen_core):
    y_lims = ax.get_ylim()
    ax.clear()
    ax.set_ylim(y_lims)
    old_energy = make_iteration(solver, Delta, phi_arr, delta_arr, Frensel_refl)
    # фаза спектра
    my_spectrum_phase = {f'{current_iter+1}-iteration':
                             fftshift(np.angle(fft(solver.numerical_solution[0, chosen_core, :])))}
    plot2D_dict_ForVideo(fig, ax, fftshift(solver.omega / (2 * pi)), my_spectrum_phase)


def animate_spectrum(fig, ax, solver, Delta, phi_arr, delta_arr, Frensel_refl,
                     current_iter, chosen_core):
    y_lims = ax.get_ylim()
    ax.clear()
    ax.set_ylim(y_lims)
    ax.set_title(f'{current_iter+1}-iteration')
    old_energy = make_iteration(solver, Delta, phi_arr, delta_arr, Frensel_refl)
    # спектр
    my_spectrum_magnitude = {f'{idx}-core':
                                 fftshift(abs(fft(solver.numerical_solution[0, idx, :]))) * solver.com.tau / np.sqrt(2 * pi)
                             for idx in range(solver.eq.size)}
    plot2D_dict_ForVideo(fig, ax, fftshift(solver.omega / (2 * pi)), my_spectrum_magnitude)


def animate_magnitude(fig, axs, solver, Delta, phi_arr, delta_arr, Frensel_refl,
                      current_iter, chosen_core):
    for ax in axs:
        y_lims = ax.get_ylim()
        ax.clear()
        ax.set_ylim(y_lims)
    old_energy = make_iteration(solver, Delta, phi_arr, delta_arr, Frensel_refl)
    # амплитуда
    all_magnitudes_by_t = {f'{current_iter+1}-iteration: {idx}-core': abs(solver.numerical_solution[0, idx, :])
                           for idx in range(solver.eq.size)}
    plot2D_multicore_ForVideo(fig, axs, solver.t, all_magnitudes_by_t)


def animate_energy_z(fig, ax, solver, Delta, phi_arr, delta_arr, Frensel_refl,
                     current_iter, chosen_core):
    y_lims = ax.get_ylim()
    ax.clear()
    ax.set_ylim(y_lims)
    ax.set_title(f'{current_iter + 1}-iteration')
    old_energy = make_iteration(solver, Delta, phi_arr, delta_arr, Frensel_refl)
    # энергия
    my_energies = {f'backward energy': old_energy[chosen_core, ::-1],
                   f'forward energy': solver.energy[chosen_core, ::-1]}
    plot2D_dict_ForVideo(fig, ax, solver.z, my_energies, marker_flag=False)


def animate_energy_multicore(fig, axs, solver, Delta, phi_arr, delta_arr, Frensel_refl,
                             current_iter, chosen_core):
    for ax in axs:
        y_lims = ax.get_ylim()
        ax.clear()
        ax.set_ylim(y_lims)
    old_energy = make_iteration(solver, Delta, phi_arr, delta_arr, Frensel_refl)
    # энергия
    all_energies = {f'{current_iter+1}-iteration: {core_idx}-core':
                        [get_energy_rectangles(solver.numerical_solution[z_idx, core_idx, :], solver.com.tau)
                         for z_idx in range(solver.com.N + 1)]
                    for core_idx in range(solver.eq.size)}
    plot2D_multicore_ForVideo(fig, axs, solver.z, all_energies)


if __name__ == '__main__':
    mcf_resonator_EvoVideo()

