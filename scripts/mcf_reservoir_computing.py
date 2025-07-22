from fiberprop.solver import ComputationalParameters, EquationParameters, Solver, print_matrix
from fiberprop.fiber import Fiber, FiberMaterial, CoreConfig
from fiberprop.fiber_geometry import get_core_count
from fiberprop.light import Light
from fiberprop.base_functions import get_coupling_coefficients
from fiberprop.drawing import *
from time import time
from tqdm import trange
import numpy as np


def mackey_glass(t_size, tau=17, n=10, beta=2, gamma=1, initial_condition=1.2):
    t = np.zeros(t_size)
    t[0] = initial_condition
    for i in range(1, t_size):
        if i - tau < 0:
            t[i] = t[i-1] + (beta * t[i-1]**n) / (1 + t[i-1]**n) - gamma * t[i-1]
        else:
            t[i] = t[i-1] + (beta * t[i-1]**n) / (1 + t[i-1]**n) - gamma * t[i-1] + (beta * t[i-int(tau)]**n) / (1 + t[i-int(tau)]**n) - gamma * t[i-int(tau)]
    return t


def create_mask(mask_size: int, seed: int | None = None) -> np.ndarray:
    """
    Генерирует одну и ту же случайную маску длиной *mask_size*
    для всех сердцевин.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, mask_size)


def mackey_glass_masked(core_count: int,
                        mackey_glass_symbol_count: int,
                        mask_size: int,
                        seed: int | None = None,
                        **mg_params) -> np.ndarray:
    """
    Формирует входной сигнал для резервуарных вычислений.

    1. Строим ряд Макея-Гласса длиной *mackey_glass_symbol_count*.
    2. Создаём одну маску длиной *mask_size* (одинакова для всех C).
    3. Для каждого символа sᵢ берём поэлементное произведение
       sᵢ × mask и конкатенируем. Итоговая длина:
       *mackey_glass_symbol_count × mask_size*.
    4. Дублируем получившийся вектор для всех *core_count* сердцевин.

    Возвращает ndarray формы (C, mackey_glass_symbol_count·mask_size).
    """
    # 1. Ряд Макея-Гласса
    mg_series = mackey_glass(mackey_glass_symbol_count, **mg_params)   # (S,)

    # 2. Маска
    mask = create_mask(mask_size, seed)                                # (M,)

    # 3. Поэлементное произведение каждого символа и маски,
    #    затем склейка: kron даёт нужный порядок [s0·mask, s1·mask, …]
    pattern = np.kron(mg_series, mask)                                 # (S*M,)

    # 4. Дублируем для всех сердцевин
    initial_conditions = np.tile(pattern, (core_count, 1))                # (C, S*M)

    return initial_conditions


import numpy as np
from numpy.typing import NDArray


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
    q      = data_in[central_core_ind]
    power  = np.abs(q)**2                           # W
    tau    = time_step_ps                          # ps
    L_coup = np.pi / (2*coupling_coefficient) if coupling_coefficient else np.inf

    if use_fwhm:
        # ==============================================================
        # 1. old FWHM / peak-power approach
        # ==============================================================
        P_peak   = power.max() if power.size else 0.0
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

        L_D  = T0_ps**2 / abs(beta2_ps2_m) if beta2_ps2_m else np.inf
        L_NL = 1.0 / (gamma_1_w_m * P_peak) if P_peak else np.inf

    else:
        # ==============================================================
        # 2. integral definitions  (preferred / default)
        # ==============================================================
        # ∫|q|² dt
        energy = power.sum() * tau                            # W·ps
        if energy == 0:
            return np.inf, np.inf, L_coup

        # ∫|∂_t q|² dt
        dqdt   = np.gradient(q, tau)                          # √W / ps
        disp_int = (np.abs(dqdt)**2).sum() * tau              # W / ps

        # ∫|q|⁴ dt
        quartic = (power**2).sum() * tau                      # W²·ps

        L_D  = 2 * energy / (abs(beta2_ps2_m) * disp_int) if disp_int and beta2_ps2_m else np.inf
        L_NL = energy / (gamma_1_w_m * quartic)      if quartic  else np.inf

    L_gain = 1 / np.max(g0_array)

    if display_debug_info:
        print(f"\nIntegral method = {not use_fwhm}")
        print(f"L_D        : {L_D:.4g} m")
        print(f"L_NL       : {L_NL:.4g} m")
        print(f"L_coupling : {L_coup:.4g} m")
        print(f"L_gain : {L_gain:.4g} m\n")

    return L_D, L_NL, L_coup, L_gain


def mcf_nn_reservoir_computing(
        data_in=None,                       # ndarray (C, M_in)
        fiber_length_m=5.0,                 # длина MCF, m
        window_size=1000,                   # размер окна в отчетах (количество time_step_ps), по нему будет выбираться длина воздушного плеча
        time_step_ps=0.1,                   # шаг по времени, ps
        step_number_per_dimensionless_distance=500,
        layer_count=1.0,
        layer_radii_array=(1,),             # радиусы колец, µm
        g0_array=(),
        psat_array=(),
        use_gpu=False,
        display_debug_info=False,
        display_plots=False,
        save_gif=False
):
    """
        Численно моделирует единичный *пробег* комплексного сигнала по
        многоядерному волокну (MCF) и воздушному плечу обратной связи.

        Алгоритм строит полное комплексное поле **U(z,t)** для всех *C*
        сердцевин, интегрируя систему линейно-связанных NLSE
        с помощью метода расщепления по физическим процессам (SSFM).
        По завершении возвращается комплексное поле на выходе MCF и
        требуемая длина воздушного плеча задержки.

        ----------
        Параметры
        ----------
        data_in : ndarray, shape = (C, M)
            **Начальное условие** – комплексная огибающая сигналов (√W)
            в *C* сердцевинах (комплексные величины).
            *C* — количество сердцевин, *M* — размер временной сетки.
        fiber_length_m : float, default 5.0
            Длина моделируемого участка многоядерного волокна, м.
        time_step_ps : float, default 0.1
            Шаг временной сетки Δt в пикосекундах (ps).
            Общая длительность окна 2 T = *M* Δt.
        step_number_per_dimensionless_distance : int, default 500
            Число продольных шагов интегрирования SSFM на единицу
            безразмерной длины (см. *length_scale* ниже).
        layer_count : float, default 1
            Число кольцевых слоёв вокруг центральной сердцевины
            (0 → одиночное ядро).
        layer_radii_array : tuple[float, …], default (1.,)
            Радиусы слоёв (микроны), начиная с центрального (0 µm).
            Длина = ``layer_count + 1``.
        g0_array : array-like, default ()
            Коэффициенты малого-сигнала g₀ [1/м] для *C* сердцевин.
            Нулевой массив → усиление отключено.
        psat_array : array-like, default ()
            Мощность насыщения P_sat, Вт, для *C* сердцевин
            (используется как E_sat = 2 T P_sat).
        use_gpu : bool, default False
            True → основное ядро SSFM выполняется на GPU (PyTorch-CUDA),
            иначе – NumPy/CPU.
        display_debug_info : bool, default False
            Печатает расчётные коэффициенты, характерные длины,
            задержки и прочие служебные данные.
        display_plots : bool, default False
            Визуализация хода интегрирования и итоговых спектров
            средствами plotly (2D/3D).
        save_gif : bool, default False
            При включённой отрисовке modulus-кадров сохраняет анимацию
            эволюции поля в GIF-файл в рабочем каталоге.

        ----------
        Возвращает
        ----------
        data_out : ndarray, shape = (C, M)
            Комплексное поле после прохождения MCF
            *и* фазового сдвига в воздушном плече:
            ``U_out = U(L, t) · exp(+j β₁,air · L_air)``.
        feedback_length_m : float
            Рассчитанная длина воздушного плеча петли обратной связи.

        ----------
        Исключения
        ----------
        ValueError
            • `data_in is None` или некорректной формы
            • Размерности массивов g₀/E_sat не совпадают с *C*.
        AssertionError
            Возникает, если вычисленная длина воздушного плеча
            меньше самой секции MCF (нарушена длительность окна).

        ----------
        Примечания
        ----------
        * Характерное время **T₀** вычисляется по FWHM импульса
          центральной сердцевины, далее строятся длины
          L_D, L_NL, L_coup.  Минимальная из них задаёт
          *length_scale*, что, в сочетании с
          *step_number_per_dimensionless_distance*, определяет
          общее число продольных шагов `N` для интегратора.
        * В GPU-режиме копируются на CPU только
          ``stored_steps_count`` снимков поля – экономия VRAM.
        * Подробнее о применении delay-based reservoir computing
          с многоядерным волокном см.:
          S. Honardoost *et al.*, *Opt. Express* 26 (2018) 11072-11090;
          L. Duport *et al.*, *IEEE Photon. Tech. Lett.* 31 (2019) 890-893.

        ----------
        Пример
        ----------
        >>> M = 8192                       # точек на окно
        >>> C = 7                          # сердцевин
        >>> u0 = np.random.randn(C, M) * .01
        >>> u_out, L_air = mcf_nn_reservoir_computing(
        ...     data_in=u0,
        ...     fiber_length_m=1.0,
        ...     time_step_ps=25,           # 40 GHz
        ...     step_number_per_dimensionless_distance=300,
        ...     layer_count=1,
        ...     layer_radii_array=(0., 34.6),
        ...     display_debug_info=True
        ... )
        """

    # ─── входные данные ──────────────────────────────────────────
    if data_in is None:
        raise ValueError('Массив data_in размера (C×M) должен быть задан')
    eq_size, M = data_in.shape

    core_configuration = CoreConfig.hexagonal
    light = Light(lambda0=1.55)                                 # µm

    # ─── волокно и линейка ──────────────────────────────────────
    fiber = Fiber(core_configuration=core_configuration,
                  ring_count=layer_count,
                  core_radius=2.95,
                  cladding_diameter=125.0,
                  n2=3.2,
                  distance_to_fiber_center=layer_radii_array,
                  NA=0.125,
                  core_material=FiberMaterial.SIO2_AND_GEO2_ALLOY,
                  material_concentration=0.038)

    fiber.set_refractive_indexes_by_lambda(light.lambda0)

    central_core_ind = int(np.floor(eq_size / 2)) if eq_size > 1 else 0

    coupling_matrix = get_coupling_coefficients(fiber, light, eps=2e-4, display_debug_info=display_debug_info)
    coupling_coefficient = coupling_matrix[central_core_ind - 1][central_core_ind] if eq_size > 1 else 139.55
    gamma = fiber.get_gamma(light, eps=1e-3)
    beta1 = fiber.get_beta1(light)                     # [ps/m]
    beta2 = fiber.get_beta2(light) * 1e-3             # [ps²/m]

    if display_debug_info:
        print("coupling_coefficient =", coupling_coefficient)
        print("gamma =", gamma)
        print("beta1 =", beta1)
        print("beta2 =", beta2)

    T = time_step_ps * M / 2

    # ─── буфер задержки ──────────────────────────────────────────
    fiber_propagation_time = fiber_length_m * beta1                           # [ps]

    feedback_loop_propagation_time = window_size * time_step_ps - fiber_propagation_time
    beta1_air = 1 / light.c_light * 1e+12
    feedback_length_m = feedback_loop_propagation_time / beta1_air  # длина воздушного плеча, m

    feedback_coeff = kappa * np.exp(1j * beta1_air * feedback_length_m)

    if display_debug_info:
        print()
        print("beta1_air =", beta1_air)
        print("fiber_length_m =", fiber_length_m)
        print("feedback_length_m =", feedback_length_m)

    assert feedback_length_m > fiber_length_m

    L_D, L_NL, L_coupling, L_gain = compute_characteristic_lengths(beta2_ps2_m=beta2,
                                         gamma_1_w_m=gamma,
                                         coupling_coefficient=coupling_coefficient,
                                         data_in=data_in,
                                         time_step_ps=time_step_ps,
                                         central_core_ind=central_core_ind,
                                         g0_array=g0_array,
                                         psat_array=psat_array,
                                         display_debug_info=display_debug_info)

    # ─── масштабы и временное окно ──────────────────────────────
    time_scale = np.sqrt(0.5 * abs(beta2) / coupling_coefficient)                        # [ps]
    length_scale = np.min([L_D, L_NL, L_coupling, L_gain])  # [m]

    fiber_length_dimensionless = fiber_length_m / length_scale
    n_z = step_number_per_dimensionless_distance * int(round(fiber_length_dimensionless))

    esat_array = np.asarray(psat_array) * window_size

    if display_debug_info:
        print("data_in.shape=", data_in.shape)
        print("data_in size =", data_in.shape[1] * time_step_ps, "ps")
        print("fiber_propagation_time =", fiber_propagation_time, "ps")
        print(f'feedback_loop_propagation_time={feedback_loop_propagation_time:.1f} ps')
        print("fiber_length_dimensionless =", fiber_length_dimensionless)
        print("length_scale =", length_scale)
        print("n_z =", n_z)
        print("esat = ", esat_array)
        print("\nwindow_size * time_step_ps =", window_size * time_step_ps)

    # ─── Solver и параметры уравнения ────────────────────────────

    offset_part = 0.2
    offset_size = int(offset_part * data_in.shape[1])
    initial_data = np.zeros((data_in.shape[0], data_in.shape[1] * 2 + offset_size))

    initial_data[:, offset_size:data_in.shape[1] + offset_size] = data_in
    T_new = T * (1 + offset_part / 2)
    
    comp = ComputationalParameters(N=n_z, M=initial_data.shape[1],
                                   L1=0.0, L2=fiber_length_m,
                                   T1=-T_new, T2=T_new,
                                   method="ssfm_order2_dnd_windowed",
                                   # damp_length=offset_part * 0.5,
                                   window_size=window_size, offset_size=offset_size)
    print("damp_length=", comp.damp_length)
    eq = EquationParameters(core_configuration=core_configuration, size=eq_size,
                            ring_count=layer_count,
                            coupling_coefficient=coupling_coefficient, beta1=0,
                            beta2=beta2, gamma=gamma,
                            E_sat=esat_array, alpha=0.0, g_0=g0_array,
                            display_debug_info=display_debug_info)

    solver = Solver(comp, eq,
                    initial_condition=initial_data,
                    stored_steps_count=2, #None if display_debug_info else 2,
                    use_dimensional=True,
                    use_gpu=use_gpu,
                    use_torch=use_gpu,
                    display_debug_info=display_debug_info)

    solver.linear_coeffs_array = coupling_matrix

    iteration_count = int(np.ceil(M / window_size))

    if display_debug_info:
        print("\niteration_count =", iteration_count)

    for iteration_index in range(iteration_count):

        if display_debug_info:
            print("\niteration", iteration_index + 1, "of", iteration_count)

        solver.run_numerical_simulation(
                                        # draw_modulus=display_debug_info,
                                        draw_interval=10,
                                        save_gif=save_gif,
                                        yscale="linear")

        if display_plots:
            number_of_points_for_display = np.min([5000, solver.com.M])

            step = int(solver.com.M / number_of_points_for_display)

            # energies = [solver.energy[i, :] for i in range(solver.eq.size)]
            # names = [f'$E_{{{i}}}$' for i in range(solver.eq.size)]
            # plot2D_plotly(solver.z, energies, names=names, x_axis_label='z [m]', y_axis_label='energy [pJ]')
            #
            # peak_powers = [solver.peak_power[i, :] for i in range(solver.eq.size)]
            # names = [f'$P_{{{i}}}$' for i in range(solver.eq.size)]
            # plot2D_plotly(solver.z, peak_powers, names=names, x_axis_label='z [m]', y_axis_label='peak power [W]')

            # phase = [solver.phase_by_z[i, :] for i in range(solver.eq.size)]
            # names = [f'$phi_{{{i}}}$' for i in range(solver.eq.size)]
            # plot2D_plotly(solver.z, phase, names=names, x_axis_label='z [m]', y_axis_label='phase [rad]')

            plot2D_plotly(solver.t[::step] * 1e-3, [np.abs(solver.numerical_solution[0][central_core_ind][::step]) ** 2,
                                     np.abs(solver.numerical_solution[-1][central_core_ind][::step]) ** 2],
                          names=[f"$|U_{central_core_ind}(z=0,t)|^2$", f"$|U_{central_core_ind}(z=L,t)|^2$"], x_axis_label='t [ns]', y_axis_label='power [W]')

            # plot3D_plotly(solver.t[::step], solver.z, np.abs(solver.numerical_solution[central_core_ind][::step]) ** 2, f"$|U_{central_core_ind}(z,t)|^2$")

            solver.numerical_solution[0] = initial_data + np.roll(solver.numerical_solution[-1], window_size, axis=1) * feedback_coeff

    # plot2D_plotly(solver.t * 1e-3, [np.abs(solver.numerical_solution[0][central_core_ind]) ** 2,
    #                                 np.abs(solver.numerical_solution[-1][central_core_ind]) ** 2],
    #               names=[f"$|U_{central_core_ind}(z=0,t)|^2$", f"$|U_{central_core_ind}(z=L,t)|^2$"], x_axis_label='t [ns]', y_axis_label='power [W]')

    return solver.numerical_solution[-1][:, offset_size:offset_size+data_in.shape[1]], feedback_length_m


def mcf_nn_reservoir_computing_temporal_evolution(
        data_in=None,  # ndarray (C, M_in)
        fiber_length_m=5.0,  # длина MCF, m
        time_step_ps=0.1,  # шаг по времени, ps
        feedback_coeff=1.0,
        step_number_per_dimensionless_distance=500,
        layer_count=1.0,
        layer_radii_array=(1,),  # радиусы колец, µm
        g0_array=(),
        psat_array=(),
        use_gpu=False,
        display_debug_info=False,
        display_plots=False,
        save_gif=False,
        upsampling=1
):
    """
        Численно моделирует единичный *пробег* комплексного сигнала по
        многоядерному волокну (MCF) и воздушному плечу обратной связи.

        Алгоритм строит полное комплексное поле **U(z,t)** для всех *C*
        сердцевин, интегрируя систему линейно-связанных NLSE
        с помощью метода расщепления по физическим процессам (SSFM).
        По завершении возвращается комплексное поле на выходе MCF и
        требуемая длина воздушного плеча задержки.

        ----------
        Параметры
        ----------
        data_in : ndarray, shape = (C, M)
            **Начальное условие** – комплексная огибающая сигналов (√W)
            в *C* сердцевинах (комплексные величины).
            *C* — количество сердцевин, *M* — размер временной сетки.
        fiber_length_m : float, default 5.0
            Длина моделируемого участка многоядерного волокна, м.
        time_step_ps : float, default 0.1
            Шаг временной сетки Δt в пикосекундах (ps).
            Общая длительность окна 2 T = *M* Δt.
        step_number_per_dimensionless_distance : int, default 500
            Число продольных шагов интегрирования SSFM на единицу
            безразмерной длины (см. *length_scale* ниже).
        layer_count : float, default 1
            Число кольцевых слоёв вокруг центральной сердцевины
            (0 → одиночное ядро).
        layer_radii_array : tuple[float, …], default (1.,)
            Радиусы слоёв (микроны), начиная с центрального (0 µm).
            Длина = ``layer_count + 1``.
        g0_array : array-like, default ()
            Коэффициенты малого-сигнала g₀ [1/м] для *C* сердцевин.
            Нулевой массив → усиление отключено.
        psat_array : array-like, default ()
            Мощность насыщения P_sat, Вт, для *C* сердцевин
            (используется как E_sat = 2 T P_sat).
        use_gpu : bool, default False
            True → основное ядро SSFM выполняется на GPU (PyTorch-CUDA),
            иначе – NumPy/CPU.
        display_debug_info : bool, default False
            Печатает расчётные коэффициенты, характерные длины,
            задержки и прочие служебные данные.
        display_plots : bool, default False
            Визуализация хода интегрирования и итоговых спектров
            средствами plotly (2D/3D).
        save_gif : bool, default False
            При включённой отрисовке modulus-кадров сохраняет анимацию
            эволюции поля в GIF-файл в рабочем каталоге.

        ----------
        Возвращает
        ----------
        data_out : ndarray, shape = (C, M)
            Комплексное поле после прохождения MCF
            *и* фазового сдвига в воздушном плече:
            ``U_out = U(L, t) · exp(+j β₁,air · L_air)``.
        feedback_length_m : float
            Рассчитанная длина воздушного плеча петли обратной связи.

        ----------
        Исключения
        ----------
        ValueError
            • `data_in is None` или некорректной формы
            • Размерности массивов g₀/E_sat не совпадают с *C*.
        AssertionError
            Возникает, если вычисленная длина воздушного плеча
            меньше самой секции MCF (нарушена длительность окна).

        ----------
        Примечания
        ----------
        * Характерное время **T₀** вычисляется по FWHM импульса
          центральной сердцевины, далее строятся длины
          L_D, L_NL, L_coup.  Минимальная из них задаёт
          *length_scale*, что, в сочетании с
          *step_number_per_dimensionless_distance*, определяет
          общее число продольных шагов `N` для интегратора.
        * В GPU-режиме копируются на CPU только
          ``stored_steps_count`` снимков поля – экономия VRAM.
        * Подробнее о применении delay-based reservoir computing
          с многоядерным волокном см.:
          S. Honardoost *et al.*, *Opt. Express* 26 (2018) 11072-11090;
          L. Duport *et al.*, *IEEE Photon. Tech. Lett.* 31 (2019) 890-893.

        ----------
        Пример
        ----------
        >>> M = 8192                       # точек на окно
        >>> C = 7                          # сердцевин
        >>> u0 = np.random.randn(C, M) * .01
        >>> u_out, L_air = mcf_nn_reservoir_computing(
        ...     data_in=u0,
        ...     fiber_length_m=1.0,
        ...     time_step_ps=25,           # 40 GHz
        ...     step_number_per_dimensionless_distance=300,
        ...     layer_count=1,
        ...     layer_radii_array=(0., 34.6),
        ...     display_debug_info=True
        ... )
        """

    # ─── входные данные ──────────────────────────────────────────
    if data_in is None:
        raise ValueError('Массив data_in размера (C×M) должен быть задан')

    if not isinstance(upsampling, int) or upsampling < 1:
        raise ValueError('upsampling должен быть натуральным числом ≥ 1')

    eq_size, M_orig = data_in.shape
    M = M_orig * upsampling

    data_in = np.repeat(data_in, upsampling, axis=1)
    time_step_ps /= upsampling

    core_configuration = CoreConfig.hexagonal
    light = Light(lambda0=1.55)  # µm

    # ─── волокно и линейка ──────────────────────────────────────
    fiber = Fiber(core_configuration=core_configuration,
                  ring_count=layer_count,
                  core_radius=2.95,
                  cladding_diameter=125.0,
                  n2=3.2,
                  distance_to_fiber_center=layer_radii_array,
                  NA=0.125,
                  core_material=FiberMaterial.SIO2_AND_GEO2_ALLOY,
                  material_concentration=0.038)

    fiber.set_refractive_indexes_by_lambda(light.lambda0)

    central_core_ind = int(np.floor(eq_size / 2)) if eq_size > 1 else 0

    coupling_matrix = get_coupling_coefficients(fiber, light, eps=2e-4, display_debug_info=display_debug_info)
    coupling_coefficient = coupling_matrix[central_core_ind - 1][central_core_ind] if eq_size > 1 else 139.55
    gamma = fiber.get_gamma(light, eps=1e-3)
    beta1 = fiber.get_beta1(light)  # [ps/m]
    beta2 = fiber.get_beta2(light) * 1e-3  # [ps²/m]

    if display_debug_info:
        print("coupling_coefficient =", coupling_coefficient)
        print("gamma =", gamma)
        print("beta1 =", beta1)
        print("beta2 =", beta2)

    T = time_step_ps * M / 2

    # ─── буфер задержки ──────────────────────────────────────────
    fiber_propagation_time = fiber_length_m * beta1  # [ps]

    feedback_loop_propagation_time = 2 * T - fiber_propagation_time
    beta1_air = 1 / light.c_light * 1e+12
    feedback_length_m = feedback_loop_propagation_time / beta1_air  # длина воздушного плеча, m

    if display_debug_info:
        print()
        print("beta1_air =", beta1_air)
        print("fiber_length_m =", fiber_length_m)
        print("feedback_length_m =", feedback_length_m)

    assert feedback_length_m > fiber_length_m

    ################################################

    t_axis = (np.arange(data_in.shape[1]) - data_in.shape[1] / 2) * time_step_ps  # [ps]

    # берём, скажем, центральную сердцевину:
    u0 = data_in[central_core_ind]

    # первая производная dU/dt (комплексная)
    du_dt = np.gradient(u0, time_step_ps)  # [√W / ps]

    plt.figure(figsize=(9, 4))

    # ── |u|² ----------------------------------------------------------------
    plt.subplot(1, 2, 1)
    plt.plot(t_axis, np.abs(u0) ** 2)
    plt.title("Модуль² входного поля")
    plt.xlabel("t  [ps]")
    plt.ylabel("|u|²  [W]")

    # ── dRe(u)/dt  -----------------------------------------------------------
    plt.subplot(1, 2, 2)
    plt.plot(t_axis, du_dt.real, label="Re du/dt")
    plt.plot(t_axis, du_dt.imag, "--", label="Im du/dt")
    plt.title("Первая производная  du/dt")
    plt.xlabel("t  [ps]")
    plt.legend()

    plt.tight_layout()
    plt.show()

    #############################################

    L_D, L_NL, L_coupling = compute_characteristic_lengths(beta2_ps2_m=beta2,
                                                           gamma_1_w_m=gamma,
                                                           coupling_coefficient=coupling_coefficient,
                                                           data_in=data_in,
                                                           time_step_ps=time_step_ps,
                                                           central_core_ind=central_core_ind,
                                                           g0_array=g0_array,
                                                           psat_array=psat_array,
                                                           display_debug_info=display_debug_info)

    # ─── масштабы и временное окно ──────────────────────────────
    time_scale = np.sqrt(0.5 * abs(beta2) / coupling_coefficient)  # [ps]
    length_scale = np.min([L_D, L_NL, L_coupling])  # [m]

    fiber_length_dimensionless = fiber_length_m / length_scale
    n_z = step_number_per_dimensionless_distance * int(round(fiber_length_dimensionless))

    esat_array = np.asarray(psat_array) * 2 * T

    if display_debug_info:
        print("data_in.shape=", data_in.shape)
        print("data_in size =", data_in.shape[1] * time_step_ps, "ps")
        print("fiber_propagation_time =", fiber_propagation_time, "ps")
        print(f'feedback_loop_propagation_time={feedback_loop_propagation_time:.1f} ps')
        print("fiber_length_dimensionless =", fiber_length_dimensionless)
        print("length_scale =", length_scale)
        print("n_z =", n_z)
        print("esat = ", esat_array)

    # ─── Solver и параметры уравнения ────────────────────────────

    comp = ComputationalParameters(N=n_z, M=data_in.shape[1],
                                   L1=0.0, L2=fiber_length_m,
                                   T1=-T, T2=T)

    eq = EquationParameters(core_configuration=core_configuration, size=eq_size,
                            ring_count=layer_count,
                            coupling_coefficient=coupling_coefficient, beta1=0,
                            beta2=beta2, gamma=gamma,
                            E_sat=esat_array, alpha=0.0, g_0=g0_array,
                            display_debug_info=display_debug_info)

    solver = Solver(comp, eq,
                    initial_condition=None, # !!!!!!!!!!!!!!!!!!!!!
                    # stored_steps_count=None if display_debug_info else 2,
                    use_dimensional=True,
                    use_gpu=use_gpu,
                    use_torch=use_gpu,
                    display_debug_info=display_debug_info)

    solver.linear_coeffs_array = coupling_matrix

    ############################################

    dz = solver.com.h  # шаг по z
    vmax = abs(solver.eq.beta1).max()  # макс. скорость характеристики
    dt_cfl = 0.8 * dz * vmax  # безопасность 0.8
    rhoA = (4 * vmax) / (3 * dz)
    dt_rk4 = 2.78 / rhoA

    if display_debug_info:
        print("\ntau =", solver.com.tau, "ps")
        print("h =", solver.com.h, "m")
        print("beta1*h =", beta1 * solver.com.h, "ps")


    solver.feedback_coefficient = feedback_coeff
    solver.boundary_condition = data_in

    solver.run_numerical_simulation_time(draw_modulus=display_debug_info,
                                    draw_interval=10,
                                    save_gif=save_gif,
                                    yscale="linear")

    #############################################

    result = solver.numerical_solution[-1][:, ::upsampling] # TODO сделать правильно: снять поле с правой границы
    result *= np.exp(1j * beta1_air * feedback_length_m)

    if display_plots:
        energies = [solver.energy[i, :] for i in range(solver.eq.size)]
        names = [f'$E_{{{i}}}$' for i in range(solver.eq.size)]
        plot2D_plotly(solver.z, energies, names=names, x_axis_label='z [m]', y_axis_label='energy [pJ]')

        peak_powers = [solver.peak_power[i, :] for i in range(solver.eq.size)]
        names = [f'$P_{{{i}}}$' for i in range(solver.eq.size)]
        plot2D_plotly(solver.z, peak_powers, names=names, x_axis_label='z [m]', y_axis_label='peak power [W]')

        plot2D_plotly(solver.t, [np.abs(solver.numerical_solution[0][central_core_ind]) ** 2,
                                 np.abs(solver.numerical_solution[-1][central_core_ind]) ** 2],
                      names=[f"$|U_3(z=0,t)|^2$", f"$|U_3(z=L,t)|^2$"], x_axis_label='t [ps]', y_axis_label='power [W]',
                      linewidth=0.5)

        # plot3D_plotly(solver.t, solver.z, np.abs(solver.numerical_solution[central_core_ind]) ** 2, f"$|U_3(z,t)|^2$")

    return result, feedback_length_m


if __name__ == '__main__':

    layer_count = 1.1
    core_configuration = CoreConfig.hexagonal
    core_count = get_core_count(core_configuration=core_configuration, ring_count=layer_count)

    # layer 0 - центральная сердцевина
    # layer 1 - первый круг из 6 сердцевин
    # layer 2 - второй "круг" из 12 сердцевин, расстояние от которых до центра разное
    # ...
    layer_radii_array = np.zeros(int(layer_count) + 1)
    for i in range(int(layer_count) + 1):
        if i == 0:
            layer_radii_array[i] = 0 # [mkm]
        if i == 1:
            layer_radii_array[i] = 17.3 # [mkm]
        if i == 2:
            layer_radii_array[i] = 50 # [mkm]

        # layer_radii_array[i] = 17.3 * (i * 1.5) # [mkm]

    g0_array = np.zeros(core_count)
    for i in range(core_count):
        g0_array[i] = 10.0 # [1/m]
    
    psat_array = np.zeros(core_count)
    for i in range(core_count):
        psat_array[i] = 40 * 5e-4 # мощность насыщения [W]


    modulation_frequency_ghz = 40 # GHz
    mackey_glass_symbol_count = 2**12
    mask_size = modulation_frequency_ghz

    time_step_ps = 1 / modulation_frequency_ghz * 1e+3 # длина по времени одного отсчета
    window_size = modulation_frequency_ghz * 30 # в количестве отсчетов

    mg_params = {
        'tau': 17,
        'n': 10,
        'beta': 2,
        'gamma': 1,
        'initial_condition': 1.2
    }
    seed = 42
    data_in = mackey_glass_masked(core_count, mackey_glass_symbol_count, mask_size, seed, **mg_params) # √W

    required_avg_power_w = 1.0  # ← задайте требуемую среднюю мощность, W
    current_avg_power = np.mean(np.abs(data_in) ** 2)  # ⟨|U|²⟩
    scale_factor = np.sqrt(required_avg_power_w / current_avg_power)
    data_in *= scale_factor

    kappa = 0.9 # коэффициент обратной связи

    # data_out, feedback_length_m = mcf_nn_reservoir_computing_temporal_evolution(
    #     data_in=data_in,  # ndarray (C, M_in)
    #     fiber_length_m=1,  # физическая длина MCF, m
    #     time_step_ps=1/modulation_frequency_ghz*1e+3,  # шаг сетки t, ps
    #     feedback_coeff=0.9, # необходимо передать, так как поле добавляется непрерывно,
    #                         # а внутри считается производная по z
    #     step_number_per_dimensionless_distance=20,
    #     layer_count=layer_count,
    #     layer_radii_array=layer_radii_array,  # радиусы колец, µm
    #     g0_array=g0_array,
    #     psat_array=psat_array,
    #     use_gpu=False,
    #     display_debug_info=True,
    #     display_plots=True,
    #     save_gif=False,
    #     upsampling = 10000
    # )

    data_out, feedback_length_m = mcf_nn_reservoir_computing(
        data_in=data_in,  # ndarray (C, M_in)
        fiber_length_m=0.1,  # физическая длина MCF, m
        window_size=window_size,
        time_step_ps=time_step_ps,  # шаг сетки t, ps
        step_number_per_dimensionless_distance=20, # ставишь меньше 20 - проверь, что поле не улетело в космос от переусиления
        layer_count=layer_count,
        layer_radii_array=layer_radii_array,  # радиусы колец, µm
        g0_array=g0_array,
        psat_array=psat_array,
        use_gpu=False,
        display_debug_info=True,
        display_plots=True,
        save_gif=False
    )

    central_core_ind = 3
    t = np.arange(data_in.shape[1])
    plot2D_plotly(t, [np.abs(data_in[central_core_ind]) ** 2,
                                            np.abs(data_out[central_core_ind]) ** 2],
                  names=[f"$|U_{central_core_ind}(z=0,t)|^2$", f"$|U_{central_core_ind}(z=L,t)|^2$"],
                  x_axis_label='t [ns]', y_axis_label='power [W]')
