from fiberprop.solver import (ComputationalParameters, EquationParameters,
                              Solver, CoreConfig)
from fiberprop.drawing import plot2D_multicore, plot2D_dict
# from scripts.mcf_reservoir_computing.mcf_reservoir_computing import compute_characteristic_lengths
from fiberprop.pulses import laser_pulse, zero_pulse
from fiberprop.ssfm_mcf import get_energy_rectangles
from scipy.interpolate import make_interp_spline
from scipy.fft import fftshift, fft
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import numpy as np
import os


def compute_characteristic_lengths(beta2_ps2_m: float,
                                   gamma_1_w_m: float,
                                   coupling_coefficient: float,
                                   data_in: NDArray[np.complex128],
                                   time_step_ps: float,
                                   *,
                                   use_fwhm: bool = False,
                                   central_core_ind: int = 0,
                                   g0_array=(),
                                   psat_array=(),
                                   display_debug_info: bool = False):
    """
    Return (L_D, L_NL, L_coupling, L_gain)  using either
    • integral definitions  (default)  or
    • peak-power/FWHM shortcut   (set use_fwhm=True).

    Parameters are the same as the old version, plus
    `use_fwhm` – choose old behaviour.
    """

    # ------------------------------------------------------------------
    # common quantities
    # ------------------------------------------------------------------
    q = data_in[central_core_ind]
    power = np.abs(q) ** 2  # W
    tau = time_step_ps  # ps
    L_coupling = np.pi / (2 * coupling_coefficient) if coupling_coefficient else np.inf

    if use_fwhm:
        # ==============================================================
        # 1. old FWHM / peak-power approach
        # ==============================================================
        P_peak = power.max() if power.size else 0.0
        idx_peak = power.argmax()

        half = 0.5 * P_peak
        above = power >= half
        # boundaries
        l = idx_peak
        while l > 0 and above[l]:
            l -= 1
        r = idx_peak
        while r < power.size - 1 and above[r]:
            r += 1

        tau_fwhm_ps = (r - l) * tau
        T0_ps = tau_fwhm_ps / 1.763 if tau_fwhm_ps > 0 else np.inf

        L_D = T0_ps ** 2 / abs(beta2_ps2_m) if beta2_ps2_m else np.inf
        L_NL = 1.0 / (gamma_1_w_m * P_peak) if P_peak else np.inf

    else:
        # ==============================================================
        # 2. integral definitions  (preferred / default)
        # ==============================================================
        # ∫|q|² dt
        energy = power.sum() * tau  # W·ps
        if energy == 0:
            return np.inf, np.inf, L_coupling, np.inf

        # ∫|∂_t q|² dt
        dqdt = np.gradient(q, tau)  # √W / ps
        disp_int = (np.abs(dqdt) ** 2).sum() * tau  # W / ps

        # ∫|q|⁴ dt
        quartic = (power ** 2).sum() * tau  # W²·ps

        L_D = 2 * energy / (abs(beta2_ps2_m) * disp_int) if disp_int and beta2_ps2_m else np.inf
        L_NL = energy / (gamma_1_w_m * quartic) if quartic else np.inf

    g = np.asarray(g0_array, float).ravel()
    m = np.isfinite(g) & (g > 1e-12)
    L_gain = 1.0 / np.max(g[m]) if np.any(m) else np.inf

    if display_debug_info:
        print(f"\nIntegral method = {not use_fwhm}")
        print(f"L_D        : {L_D:.4g} m")
        print(f"L_NL       : {L_NL:.4g} m")
        print(f"L_coupling : {L_coupling:.4g} m")
        print(f"L_gain : {L_gain:.4g} m\n")

    return L_D, L_NL, L_coupling, L_gain


def get_dn_of_z(eq, com, z_array,
                dn, dn_step, n_ref, k_coef, J_coef, seed=42):
    np.random.seed(seed)
    steps_num = int((com.L2 - com.L1) / dn_step) + 1
    if steps_num < 5:
        steps_num = 5
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


def simulation_of_propagation_in_mcf(spatial_step=0.25e-3, dn=4e-6, dn_step=1e-4, J_coef=9e-7,
                                     beta2_coef=0.0, gamma_coef=5e-3, path=''):
    mcf_size = 7

    # параметры вычислительной сетки
    ComputationalParameters.get_info()
    fiber_length = 200.0  # длина волокна [m]
    my_N = round(fiber_length / spatial_step)  # разбиение таково,
    # чтобы пространственный шаг составлял spatial_step м
    my_M = 512
    td_width = 5e3  # ширина временного интервала [ps]
    computational_params = ComputationalParameters(N=my_N, M=my_M, L1=0.0, L2=fiber_length,
                                                   T1=-td_width/2, T2=td_width/2, 
                                                   method="ssfm_order2_ndn_by_julia")
    # "ssfm_order2_ndn_by_julia", "ssfm_order2_ndn"

    # параметры уравнения
    EquationParameters.get_info()
    dn = dn  # изменение показателя преломления между сердцевинами
    dn_step = dn_step  # интервал через который меняется показатель преломления [m]
    k_coef = 2*np.pi * 1e6  # волновое число [1/m]
    J_coef = J_coef  # оценка величины интеграла перекрытия
    c_coef = J_coef * k_coef  # коэффициент связи [1/m]
    n_ref = 15.625/10.8 + 2.7e-3
    eq_params = EquationParameters(core_configuration=CoreConfig.hexagonal, size=mcf_size, ring_count=1,
                                   coupling_coefficient=c_coef, E_sat=0.0, g_0=0.0, alpha=0.0,
                                   beta1=1e-3, beta2=beta2_coef, gamma=gamma_coef, noise_amplitude=0.0)
    # В данной задаче начальное значение "beta1" не играет роли, поскольку перед каждой итерацией
    # численного метода оно пересчитывается из изменения показателя преломления

    input_pulses = [zero_pulse, zero_pulse,
                    zero_pulse, laser_pulse, zero_pulse,
                    zero_pulse, zero_pulse]
    pulse_width = 600  # ps
    peak_power = 5e3  # W
    input_pulses_params = [{}, {},
                           {}, {"peak_power": peak_power, "fwhm": pulse_width}, {},
                           {}, {}]
    solver = Solver(computational_params, eq_params, use_dimensional=True,
                    pulses=input_pulses, pulse_params_list=input_pulses_params,
                    stored_steps_count=2, display_debug_info=True, use_gpu=False)
    solver.beta1_of_z, solver.self_coupling_of_z = get_dn_of_z(solver.eq, solver.com, solver.z,
                                                               dn, dn_step, n_ref, k_coef, J_coef)

    compute_characteristic_lengths(beta2_coef, gamma_coef, c_coef,
                                   solver.numerical_solution[0], computational_params.tau,
                                   use_fwhm=True, central_core_ind=3, display_debug_info=True)

    solver.run_numerical_simulation()

    # Сейчас построение графиков производится отдельно для экономии оперативной памяти
    # поэтому в visualize_res сохраняются только данные для построения этих графиков
    visualize_res(solver, path)  # визуализация результатов


def save_1d_arrays(axes_dict, arrs_dict, file_name, perm='.txt'):
    """ Функция сохраняет одномерные массивы данных в виде столбцов. В качестве аргумента arrs_dict предполагается
    словарь, в котором ключ - название графика, а значение - массив. Аналогично значения осей координат передаются
    в аргументе axes_dict. """
    print(file_name + ':\tsaving...')
    output_file = open(file_name + perm, 'wt')

    axes_num = len(axes_dict)
    variables_num = len(arrs_dict)
    names = list(axes_dict.keys()) + list(arrs_dict.keys())
    separator = ', '
    names_row = separator.join(names)
    output_file.write(names_row)
    output_file.write('\n')

    points_num = len(arrs_dict[names[axes_num+0]])
    for i in range(points_num):
        nums = ([f'{axes_dict[names[j]][i]:.5e}' for j in range(axes_num)] +
                [f'{arrs_dict[names[axes_num + j]][i]:.5e}' for j in range(variables_num)])
        num_separator = '\t'
        nums_row = num_separator.join(nums)
        output_file.write(nums_row)
        output_file.write('\n')

    output_file.close()
    print(file_name + ':\tsaved')
    return


def read_1d_arrays(full_file_name):
    """ Функция считывает файл с данными-колонками, где в первой строке названия столбцов. Возвращает словарь. """
    input_file = open(full_file_name, 'r')
    print(full_file_name + '\treading...')

    names = input_file.readline()[:-1]
    keys_sep = ', '
    keys = names.split(keys_sep)
    rotated_arrays = [np.array(line.split('\t'), dtype=float) for line in input_file]
    reversed_vals = np.transpose(rotated_arrays)
    res_dict = dict(zip(keys, reversed_vals))

    input_file.close()
    print(full_file_name + '\tread')
    return res_dict


def visualize_res(solver, path):
    current_solution = solver.numerical_solution
    # интенсивность на входе
    all_magnitudes_by_t = {f'{idx}-core': abs(current_solution[0, idx, :])**2
                           for idx in range(solver.eq.size)}
    plot2D_multicore(solver.t*1e-3, all_magnitudes_by_t, xlabel='time, ns',
                     ylabel='$|A_n \\text{(t, z=0)}|^2$, W', y_max=np.max(abs(current_solution[0, 3, :])**2)*1.1,
                     path=path, name="интенсивность_вход")

    # интенсивность на выходе
    all_magnitudes_by_t = {f'{idx}-core': abs(current_solution[-1, idx, :])**2
                           for idx in range(solver.eq.size)}
    plot2D_multicore(solver.t*1e-3, all_magnitudes_by_t, xlabel='time, ns',
                     ylabel='$|A_n \\text{(t, z=end)}|^2$, W', y_max=np.max(abs(current_solution[-1, 3, :])**2)*1.1,
                     path=path, name="интенсивность_выход")
    print(solver.com.tau)

    # спектр на входе
    freqs = fftshift(solver.omega / (2 * np.pi)) + 299792458.0 / 1050.0 * 1e-3  # THz
    wlengths = 299792458.0 / freqs * 1e-3  # nm
    my_spectrum_magnitude = {
        f'{idx}-core': fftshift(abs(fft(current_solution[0, idx, :]))) * solver.com.tau / np.sqrt(2 * np.pi)
        for idx in range(solver.eq.size)}
    plot2D_multicore(wlengths, my_spectrum_magnitude, xlabel='$\\lambda, nm$',
                     ylabel='$|\\hat{A_n} \\text{(t, z=0)}|$, $\\sqrt{J/nm}$',
                     y_max=np.max(abs(current_solution[0, 3, :])**2) * 1.1,
                     y_logscale=True, path=path, name="спектр_вход")

    # спектр на выходе
    my_spectrum_magnitude = {
        f'{idx}-core': fftshift(abs(fft(current_solution[-1, idx, :]))) * solver.com.tau / np.sqrt(2 * np.pi)
        for idx in range(solver.eq.size)}
    plot2D_multicore(wlengths, my_spectrum_magnitude, xlabel='$\\lambda, nm$',
                     ylabel='$|\\hat{A_n} \\text{(t, z=end)}|$, $\\sqrt{J/nm}$',
                     y_max=np.max(abs(current_solution[0, 3, :])**2) * 1.1,
                     y_logscale=True, path=path, name="спектр_выход")

    # доля по мощности в пике от z
    maximal_initial_power = abs(current_solution[0, 3, solver.com.M//2])**2
    all_magnitudes = {f'{idx}-core': solver.peak_power[idx, :]*100.0 / maximal_initial_power
                      for idx in range(solver.eq.size)}
    # plot2D_dict(solver.z, all_magnitudes, xlabel='z, m', ylabel='$P_{out} / P_{in}$, %',
    #             title='average by peak power', marker_flag=False, y_max=100,
    #             path=path, name="распределение_пиковой_мощности")
    axes_dict = {'z': solver.z}
    all_magnitudes_file_name = path + '\\распределение_пиковой_мощности'
    save_1d_arrays(axes_dict, all_magnitudes, all_magnitudes_file_name)

    # мощность в пике от z
    all_magnitudes = {f'{idx}-core': solver.peak_power[idx, :]
                      for idx in range(solver.eq.size)}
    # plot2D_dict(solver.z, all_magnitudes, xlabel='z, m', ylabel='$P_{out}$, W',
    #             title='by peak power', marker_flag=False, y_max=all_magnitudes['3-core'][0]*1.05,
    #             path=path, name="пиковая_мощность")
    axes_dict = {'z': solver.z}
    all_magnitudes_file_name = path + '\\пиковая_мощность'
    save_1d_arrays(axes_dict, all_magnitudes, all_magnitudes_file_name)

    # доля по средней мощности от z
    maximal_energy = solver.energy[3, 0]
    all_mean_powers = {
        f'{idx}-core': solver.energy[idx]*100.0 / maximal_energy
        for idx in range(solver.eq.size)}
    # plot2D_dict(solver.z, all_mean_powers, xlabel='z, m', ylabel='$P_{out} / P_{in}$, % ',
    #             title='average by mean power', marker_flag=False, y_max=100,
    #             path=path, name="распределение_средней_мощности")
    axes_dict = {'z': solver.z}
    all_mean_file_name = path + '\\распределение_средней_мощности'
    save_1d_arrays(axes_dict, all_mean_powers, all_mean_file_name)

    # энергия от z
    all_mean_powers = {
        f'{idx}-core': solver.energy[idx]
        for idx in range(solver.eq.size)}
    # plot2D_dict(solver.z, all_mean_powers, xlabel='z, m', ylabel='$P_{out}$, J',
    #             title='by mean power', marker_flag=False, y_max=all_mean_powers['3-core'][0],
    #             path=path, name="средняя_мощность")
    axes_dict = {'z': solver.z}
    all_mean_file_name = path + '\\средняя_мощность'
    save_1d_arrays(axes_dict, all_mean_powers, all_mean_file_name)


def plot_triple_picture(x_arr, dict_list):
    ncols = 3
    nrows = 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * 3, 4), frameon=True)

    axs = axs.flatten()
    for ax in axs:
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.set(facecolor='w')
        ax.set_xlabel('z (м)')

    colors = ['orange', 'yellow', 'darkviolet', 'blue', 'green', 'deepskyblue', 'brown']
    max_y = 625.0
    for i, data_dict in enumerate(dict_list):
        max_val = max(data_dict[key][0] for key in data_dict)
        scale_coef = max_y / max_val
        axs[i].set_xticks(np.arange(0, 201, 100))
        axs[i].set_yticks(np.arange(0, 601, 200))
        axs[i].set_ylim(0.0, max_y)
        axs[i].set_xlim(-1, 200)
        for j, line_label in enumerate(data_dict):
            color = colors[j]
            axs[i].plot(x_arr, data_dict[line_label]*scale_coef,
                        color=color, linestyle='-', linewidth=2, alpha=0.85)

    plt.show()
    print('plot_triple: \tDone')


def plot_duple_picture(x_arr, dict_list):
    ncols = 2
    nrows = 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * 3, 4), frameon=True)

    axs = axs.flatten()
    for ax in axs:
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.set(facecolor='w')
        ax.set_xlabel('z (м)')

    colors = ['orange', 'yellow', 'darkviolet', 'blue', 'green', 'deepskyblue', 'brown']
    max_y = 625.0
    for i, data_dict in enumerate(dict_list):
        max_val = max(data_dict[key][0] for key in data_dict)
        scale_coef = max_y / max_val
        axs[i].set_xticks(np.arange(0, 201, 50))
        axs[i].set_yticks(np.arange(0, 601, 200))
        axs[i].set_ylim(0.0, max_y)
        axs[i].set_xlim(-1, 200)
        for j, line_label in enumerate(data_dict):
            color = colors[j]
            axs[i].plot(x_arr, data_dict[line_label]*scale_coef,
                        color=color, linestyle='-', linewidth=2, alpha=0.85)

    plt.show()
    print('plot_duple: \tDone')


if __name__ == "__main__":
    ###
    # величина шага по z "spatial_step" в [m],
    # вариация показателя преломления "dn" в абсолютных величинах
    # расстояние вдоль z "dn_step", через которое меняется показатель преломления, в [m]
    # оценка интеграла перекрытиz "J_coef" в абсолютных величинах
    # параметр дисперсии группвых скоростей "beta2_coef" в [ps^2 / m]
    # коэффициент керровской нелинейности "gamma_coef" в [1/W/m]
    ###

    def save_readme(path, dict_info):
        with open(path + '\\readme.txt', 'w', encoding='utf-8') as file:
            for key in dict_info:
                val = dict_info[key]
                file.write(key + ":\t\t\t" + str(val) + "\n")
    dir_path = os.path.dirname(os.path.abspath(__file__))

    spatial_step = 5e-4  # [m]
    curr_path = dir_path + "\\propagation_results"
    params_dict = {'spatial_step': spatial_step, 'dn': 1e-3,
                   'dn_step': 0.005, 'J_coef': 1e-4,
                   'beta2_coef': 0.0, 'gamma_coef': 5e-3,
                   'path': curr_path}

    dicts_list = [params_dict]

    for params_dict in dicts_list:
        curr_path = params_dict['path']
        os.makedirs(curr_path, exist_ok=True)
        simulation_of_propagation_in_mcf(**params_dict)
        save_readme(curr_path, params_dict)


    ### Ниже код для построения графиков, как в отчёте у Алёны Колесниковой

    # x_arr = None
    # dict_list = []
    # for i in range(3):
    #     curr_path = dir_path + f"\\data_{int(i+1)}\\распределение_средней_мощности.txt"
    #     curr_dict = read_1d_arrays(curr_path)
    #     x_arr = curr_dict['z']
    #     del curr_dict['z']
    #     dict_list.append(curr_dict)
    # plot_triple_picture(x_arr, dict_list)

    # idxs_list = [41, 6]
    # dict_list_new = []
    # for i in idxs_list:
    #     curr_path = dir_path + f"\\data_{int(i)}\\распределение_средней_мощности.txt"
    #     curr_dict = read_1d_arrays(curr_path)
    #     x_arr = curr_dict['z']
    #     del curr_dict['z']
    #     dict_list_new.append(curr_dict)
    # plot_duple_picture(x_arr, dict_list_new[::-1])
