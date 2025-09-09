from __future__ import annotations
import os
from datetime import datetime
from typing import Optional, Union

from fiberprop.solver import ComputationalParameters, EquationParameters, Solver
from fiberprop.fiber import Fiber, FiberMaterial, CoreConfig
from fiberprop.fiber_geometry import get_core_count
from fiberprop.light import Light
from fiberprop.base_functions import get_coupling_coefficients
from fiberprop.drawing import *
from fiberprop.threading_control import physical_cpu_count, temporary_thread_limits

from time import time
import numpy as np #; np.show_config()
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from pathlib import Path
STYLE_PATH = Path(__file__).with_name("styles") / "mcf.mplstyle"
plt.style.use(str(STYLE_PATH))

MM = 1/25.4
COL1, COL15, COL2 = 89*MM, 136*MM, 183*MM   # Nature: 1, 1.5, 2 колонки


def mackey_glass(t_size, tau=17.0, n=10, beta=0.2, gamma=0.1, initial_condition=1.2, dt=1.0):
    delay = tau / dt
    k = int(np.floor(delay))
    frac = delay - k

    x = np.empty(t_size, dtype=float)
    start_index = k + 1 if frac == 0.0 else k + 2
    if start_index > t_size:
        start_index = t_size
    x[:start_index] = initial_condition

    for i in range(start_index, t_size):
        j = (i - 1) - k
        if frac == 0.0:
            x_tau = x[j]
        else:
            x_tau = (1.0 - frac) * x[j - 1] + frac * x[j]
        dxdt = beta * x_tau / (1.0 + x_tau ** n) - gamma * x[i - 1]
        x[i] = x[i - 1] + dt * dxdt
    return x


def _normalize_zero_mean_unit_std(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    m = float(np.mean(x))
    s = float(np.std(x))
    if s < eps:
        return np.zeros_like(x)
    return (x - m) / s


def create_mask(mask_size: int, rng: np.random.Generator, kind: str = "rademacher") -> np.ndarray:
    if kind == "rademacher":
        # {−1, +1}
        return rng.choice(np.array([-1.0, 1.0], dtype=float), size=mask_size)
    elif kind == "uniform":
        m = rng.uniform(-1.0, 1.0, size=mask_size)
        return m - float(m.mean())  # зануляем среднее
    elif kind == "gaussian":
        return rng.normal(0.0, 1.0, size=mask_size)
    else:
        # по умолчанию rademacher
        return rng.choice(np.array([-1.0, 1.0], dtype=float), size=mask_size)


def mackey_glass_masked(core_count: int,
                        mackey_glass_symbol_count: int,
                        mask_size: int,
                        seed: int | None = None,
                        gain_in: float = 1.0,
                        warmup: int | None = None,
                        mask_kind: str = "rademacher",
                        **mg_params) -> np.ndarray:
    """
    Генерирует вход для резервуара (C, S*M):
      1) ряд MG длиной S + warmup;
      2) выбрасываем warmup для выхода на аттрактор;
      3) нормировка (0-среднее, 1-std);
      4) независимые маски на каждый core (нулевое среднее);
      5) time-multiplexing: kron(MG, mask_c) для каждого core;
      6) масштабирование gain_in.
    """
    rng = np.random.default_rng(seed)

    # 1) MG (длиннее на разгон)
    S = int(mackey_glass_symbol_count)
    if warmup is None:
        tau = mg_params.get("tau", 17)
        dt = mg_params.get("dt", 1.0)
        warmup = max(1000, int(10 * tau / dt))
    x_full = mackey_glass(S + warmup, **mg_params)

    # 2) выбрасываем разгон
    x = x_full[warmup:]

    # 3) нормировка
    x = _normalize_zero_mean_unit_std(x)

    # 4) маски per-core
    masks = np.empty((core_count, mask_size), dtype=float)
    for c in range(core_count):
        masks[c] = create_mask(mask_size, rng, kind=mask_kind)

    # 5) time-multiplexing
    out = np.empty((core_count, S * mask_size), dtype=float)
    for c in range(core_count):
        out[c] = np.kron(x, masks[c])

    # 6) масштаб под тракт
    out *= gain_in
    return out


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
    L_coup = np.pi / (2 * coupling_coefficient) if coupling_coefficient else np.inf

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
            return np.inf, np.inf, L_coup, np.inf

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
        print(f"L_coupling : {L_coup:.4g} m")
        print(f"L_gain : {L_gain:.4g} m\n")

    return L_D, L_NL, L_coup, L_gain


# === NEW: FFT-friendly padding helper ===
def _fft_padding_params(M: int, min_fraction: float = 0.1) -> Tuple[int, float, int]:
    """
    Возвращает (offset_size0, offset_part, target_len) для добивки нулями:
      - offset_size0: сколько нулей добавить в начало,
      - offset_part: доля (offset_size0 / M) — всегда >= min_fraction,
      - target_len: итоговая длина по времени после добивки (= next_fast_len(M + ceil(min_fraction*M))).
    """
    from scipy.fft import next_fast_len as _next_fast_len

    min_off = max(1, int(np.ceil(min_fraction * float(M))))
    target_len = int(_next_fast_len(int(M + min_off)))
    offset_size0 = int(target_len - M)
    # гарантируем минимум 10% и согласуем с выбранным быстрым размером
    offset_part = max(float(min_fraction), float(offset_size0) / float(M))
    return offset_size0, offset_part, target_len


def mcf_nn_reservoir_computing(
        data_in=None,  # ndarray (C, M_in)
        fiber_length_m=5.0,  # длина MCF, m
        window_size=1000,
        time_step_ps=0.1,  # шаг по времени, ps
        step_number_per_dimensionless_distance=500,
        upsampling=1,
        layer_count=1.0,
        layer_radii_array=(1,),  # радиусы колец, µm
        g0_array=(),
        psat_array=(),
        kappa=0.9,
        use_gpu=False,
        use_torch=False,
        num_threads: int | str | None = "default",
        display_debug_info=False,
        display_debug_plots=False,
        save_gif=False,
        max_hours_total: Optional[float] = None,
        precision: Optional[str] = 'float64',
        use_dispersion: bool = True,
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
        window_size : int, default 1000
            Размер окна в отчетах (количество time_step_ps), по нему будет выбираться длина воздушного плеча
        time_step_ps : float, default 0.1
            Длительность одного шага по маске в пикосекундах (ps).
            Общая длительность окна 2 T = *M* Δt.
        step_number_per_dimensionless_distance : int, default 500
            Число продольных шагов интегрирования SSFM на единицу
            безразмерной длины (см. *length_scale* ниже).
        upsampling : int, default 1
            Число точек на один шаг маски (апсемплинг по времени).
        layer_count : float, default 1.0
            Количество колец вокруг центральной сердцевины (может быть дробным).
        layer_radii_array : tuple, default (1,)
            Радиусы слоёв (микроны), начиная с центрального (0 µm).
            Длина = ``layer_count + 1``.
        g0_array : array-like, default ()
            Коэффициенты малого-сигнала g₀ [1/м] для *C* сердцевин.
            Нулевой массив → усиление отключено.
        psat_array : array-like, default ()
            Мощность насыщения P_sat, Вт, для *C* сердцевин
            (используется как E_sat = 2 T P_sat).
        kappa : float, default 0.9
            Коэффициент обратной связи
        use_gpu : bool, default False
            True → основное ядро SSFM выполняется на GPU (PyTorch-CUDA),
            иначе – NumPy/CPU.
        display_debug_info : bool, default False
            Печатает расчётные коэффициенты, характерные длины,
            задержки и прочие служебные данные.
        display_debug_plots : bool, default False
            Визуализация хода интегрирования и итоговых спектров
            средствами plotly (2D/3D).
        save_gif : bool, default False
            При включённой отрисовке modulus-кадров сохраняет анимацию
            эволюции поля в GIF-файл в рабочем каталоге.

        ----------
        Возвращает
        ----------
        словарь:
            "data_out": ndarray (C, M) — комплексное поле после прохождения MCF,
            "params": dict — рассчитанные параметры волокна и системы обратной связи.

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

        * при display_debug_info=True на каждой итерации печатается
        массив NRMSE по окнам задержки между соседними итерациями:
          "NRMSE per window (iter k vs k-1): [ ... ]"
        """

    # ─── входные данные ──────────────────────────────────────────
    if data_in is None:
        raise ValueError('Массив data_in размера (C×M) должен быть задан')
    eq_size, M = data_in.shape

    dtype = np.complex64 if precision == "float32" else np.complex128

    if window_size <= 0 or window_size > M:
        raise ValueError("window_size должен быть >= 1")

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

    coupling_matrix = get_coupling_coefficients(fiber, light, eps=2e-4, display_debug_plots=display_debug_plots)
    coupling_coefficient = coupling_matrix[central_core_ind - 1][central_core_ind] if eq_size > 1 else 1e-10
    max_val = np.max(np.abs(coupling_matrix))
    threshold = max_val * 1e-2
    coupling_matrix = np.where(np.abs(coupling_matrix) > threshold, coupling_matrix, 0)

    gamma = fiber.get_gamma(light, eps=1e-3)
    beta1 = fiber.get_beta1(light)  # [ps/m]
    beta2 = fiber.get_beta2(light) * 1e-3  # [ps²/m]

    if display_debug_info:
        print("coupling_coefficient =", coupling_coefficient)
        print("gamma =", gamma)
        print("beta1 =", beta1)
        print("beta2 =", beta2)

    # ─── буфер задержки ──────────────────────────────────────────
    fiber_propagation_time = fiber_length_m * beta1  # [ps]
    beta1_air = 1 / light.c_light * 1e+12
    feedback_loop_propagation_time = window_size * time_step_ps - fiber_propagation_time
    feedback_length_m = feedback_loop_propagation_time / beta1_air  # длина воздушного плеча, m
    feedback_coeff = kappa * np.exp(1j * beta1_air * feedback_length_m)

    if feedback_length_m < fiber_length_m:
        print("feedback_length_m < fiber_length_m")
        raise RuntimeError("feedback_length_m < fiber_length_m")

    # ─── характерные длины ───────────────────────────────────────
    L_D, L_NL, L_coupling, L_gain = compute_characteristic_lengths(
        beta2_ps2_m=beta2,
        gamma_1_w_m=gamma,
        coupling_coefficient=coupling_coefficient,
        data_in=data_in,
        time_step_ps=time_step_ps,
        central_core_ind=central_core_ind,
        g0_array=g0_array,
        psat_array=psat_array,
        display_debug_info=display_debug_info
    )
    time_scale = np.sqrt(0.5 * abs(beta2) / coupling_coefficient) if beta2 != 0 else 0.0  # [ps]
    length_scale = np.min([L_D, L_NL, L_coupling, L_gain])  # [m]

    # ─── масштабы и временное окно ──────────────────────────────
    fiber_length_dimensionless = fiber_length_m / length_scale
    n_z = max(int(round(step_number_per_dimensionless_distance * fiber_length_dimensionless)), 1)
    esat_array = np.asarray(psat_array) * window_size * time_step_ps

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

    # --- режим без дисперсии (beta2=0) с оконным стримингом ---
    if not use_dispersion:

        T_half_stream = (window_size * time_step_ps) / 2.0

        comp = ComputationalParameters(N=n_z, M=window_size,
                                       L1=0.0, L2=fiber_length_m,
                                       T1=-T_half_stream, T2=T_half_stream,
                                       method="ssfm_order2_dnd_short",
                                       # window_size=window_size,
                                       offset_size=0)

        eq = EquationParameters(core_configuration=core_configuration, size=eq_size,
                                ring_count=layer_count,
                                coupling_matrix=coupling_matrix,
                                beta1=0, beta2=0.0, gamma=gamma,
                                E_sat=esat_array, alpha=0.0, g_0=g0_array,
                                display_debug_info=display_debug_info)

        solver = Solver(comp, eq,
                        initial_condition=data_in[:, 0:window_size],
                        stored_steps_count=2,
                        use_dimensional=True,
                        use_gpu=use_gpu,
                        use_torch=use_torch,
                        precision=precision,
                        display_debug_info=display_debug_info)

        data_out_stream = np.zeros((eq_size, M), dtype=dtype)

        t_stream_ps = np.arange(M) * time_step_ps  # [ps]
        omega_stream = 2 * np.pi * np.fft.fftfreq(M, d=time_step_ps)  # [rad/ps]

        # ВАЖНО: буфер предыдущего окна (для feedback) — определён заранее
        fb_buf = np.zeros((eq_size, window_size), dtype=dtype)

        n_batches = int(np.ceil(M / window_size))
        if display_debug_info:
            print("n_batches =", n_batches)

        t1 = time()

        for batch_index in range(n_batches):

            if display_debug_info:
                print("batch_index =", batch_index)

            if batch_index > 0:
                s = batch_index * window_size
                e = min(s + window_size, M)
                seg = data_in[:, s:e]
                if seg.shape[1] < window_size:
                    tmp = np.zeros((eq_size, window_size), dtype=dtype)
                    tmp[:, :seg.shape[1]] = seg
                    seg = tmp
                solver.numerical_solution[0] = feedback_coeff * fb_buf + seg

            dt_b = solver.run_numerical_simulation(draw_interval=10, save_gif=save_gif, yscale="linear")

            if (batch_index == 0) and (max_hours_total is not None):
                est_total_sec = dt_b * n_batches
                if est_total_sec > float(max_hours_total) * 3600.0:
                    print(
                        f"TIME_LIMIT_EXCEEDED: est_total_hours={est_total_sec / 3600:.3f} > "
                        f"max_hours_total={float(max_hours_total):.3f}; "
                        f"windows={n_batches}, dt_first={dt_b:.3f}s"
                    )
                    raise RuntimeError(
                        f"TIME_LIMIT_EXCEEDED: est_total_hours={est_total_sec / 3600:.3f} > "
                        f"max_hours_total={float(max_hours_total):.3f}; "
                        f"windows={n_batches}, dt_first={dt_b:.3f}s"
                    )

            fb_buf = solver.numerical_solution[-1]
            s = batch_index * window_size
            e = min(s + window_size, M)
            data_out_stream[:, s:e] = fb_buf[:, :e - s]

            if display_debug_plots:

                number_of_points_for_display = 1000 # solver.com.M  # np.min([5000, solver.com.M])
                step = int(solver.com.M / number_of_points_for_display)

                # plot2D_plotly(
                #     solver.t[::step] * 1e-3,
                #     [np.abs(solver.numerical_solution[0][central_core_ind][::step]) ** 2,
                #      np.abs(solver.numerical_solution[-1][central_core_ind][::step]) ** 2],
                #     names=[f"$|U_{central_core_ind}(z=0,t)|^2$", f"$|U_{central_core_ind}(z=L,t)|^2$"],
                #     x_axis_label='t [ns]', y_axis_label='power [W]'
                # )

                # plot2D_plotly(
                #     t_stream_ps * 1e-3,
                #     [np.abs(data_out_stream[central_core_ind][::step]) ** 2],
                #     names=[f"$|U_{central_core_ind}(z=L,t)|^2$"],
                #     x_axis_label='t [ns]', y_axis_label='power [W]'
                # )

        if display_debug_info:
            print("Elapsed time =", time() - t1)

        if display_debug_plots:
            plot2D_plotly(
                np.fft.fftshift(omega_stream),
                np.abs(np.fft.fftshift(np.fft.fft(data_out_stream[central_core_ind]))) ** 2,
                names=[rf"$|U_{central_core_ind}(z=L,\omega)|^2$"],
                x_axis_label=r'$\omega, \text{rad/ps}$',
                y_axis_label='spectrum intensity [W]', yscale="log", title_text="Spectrum"
            )

        if display_debug_plots:
            plot2D_plotly(
                t_stream_ps * 1e-3,
                np.abs(data_out_stream[central_core_ind]) ** 2,
                names=[f"$|U_{central_core_ind}(z=0,t)|^2$", f"$|U_{central_core_ind}(z=L,t)|^2$"],
                x_axis_label='t [ns]', y_axis_label='power [W]'
            )

        params = {
            "gamma": float(gamma),
            "beta1": float(beta1),
            "beta2": 0.0,
            "beta1_air": float(beta1_air),
            "feedback_length_m": float(feedback_length_m),
            "feedback_loop_propagation_time_ps": float(feedback_loop_propagation_time),
            "fiber_propagation_time_ps": float(fiber_propagation_time),
            # характерные длины и масштабы:
            "L_D": float(L_D), "L_NL": float(L_NL),
            "L_coupling": float(L_coupling), "L_gain": float(L_gain),
            "time_scale_ps": float(time_scale),
            "length_scale_m": float(length_scale),
            "fiber_length_dimensionless": float(fiber_length_dimensionless),
            "n_z": int(n_z),
            # расчётные настройки окна:
            "window_size": int(window_size),
            "upsampling": int(upsampling),
            "step_number_per_dimensionless_distance": int(step_number_per_dimensionless_distance),
            # итерационная информация (стриминг по окнам):
            "iteration_count": int(n_batches),
            "offset_size": 0,
            "offset_part": 0.0,
        }
        return {
            "data_out": data_out_stream,
            "params": params,
        }

    # =============================== beta2 != 0 ==========================================
    else:
        # ─── Solver и параметры уравнения ────────────────────────────

        # Выбираем размер «обрезки»/добивки: теперь подгоняем длину под быстрый FFT
        # и при этом offset_part гарантированно >= 0.1.
        offset_size0, offset_part, _target_len = _fft_padding_params(M, min_fraction=0.1)
        initial_data = np.zeros((data_in.shape[0], (M + offset_size0) * 2), dtype=dtype)
        initial_data[:, offset_size0:M + offset_size0] = data_in

        # upsampling
        if upsampling != 1:
            initial_data = np.repeat(initial_data, upsampling, axis=1)

        M_final = initial_data.shape[1]
        offset_size = offset_size0 * upsampling

        T_half = (M_final * time_step_ps) / (2.0 * upsampling)

        comp = ComputationalParameters(N=n_z, M=M_final,
                                       L1=0.0, L2=fiber_length_m,
                                       T1=-T_half, T2=T_half,
                                       method="ssfm_order2_dnd_windowed_short",  # "ssfm_order2_dnd_compact_windowed",
                                       # damp_length=offset_part * 0.5,
                                       window_size=window_size * upsampling,
                                       offset_size=offset_size)

        eq = EquationParameters(core_configuration=core_configuration, size=eq_size,
                                ring_count=layer_count,
                                coupling_matrix=coupling_matrix,
                                beta1=0,
                                beta2=beta2, gamma=gamma,
                                E_sat=esat_array, alpha=0.0, g_0=g0_array,
                                display_debug_info=display_debug_info)

        solver = Solver(comp, eq,
                        initial_condition=initial_data,
                        stored_steps_count=2,  # None if display_debug_info else 2,
                        use_dimensional=True,
                        use_gpu=use_gpu,
                        use_torch=use_torch,
                        precision=precision,
                        num_threads=num_threads,
                        display_debug_info=display_debug_info)

        if display_debug_plots:
            number_of_points_for_display = 10000 #solver.com.M  # np.min([5000, solver.com.M])
            step = int(solver.com.M / number_of_points_for_display)

        iteration_count = int(np.ceil(M / window_size))

        if display_debug_info:
            print("\niteration_count =", iteration_count)

        # ── подготовка метрики сходимости по окнам ───────────────────
        w_mask = int(window_size * upsampling)  # ширина окна в отсчётах solver'а
        main_start = int(offset_size)  # начало «полезной» области
        main_len = int(M * upsampling)  # длина «полезной» области
        main_end = main_start + main_len
        _prev_int_main = None  # интенсивности |U|^2 из пред. итерации (C, main_len)
        _last_max_nrmse = 0.0

        t_stream_ps = np.arange(M_final) * time_step_ps / upsampling  # [ps]
        omega_stream = 2 * np.pi * np.fft.fftfreq(M_final, d=time_step_ps / upsampling)  # [rad/ps]

        t1 = time()

        for iteration_index in range(iteration_count + 1):  # одна доп. итерация для окончательного установления
            if display_debug_info:
                print("\niteration", iteration_index, "of", iteration_count)

            dt_b = solver.run_numerical_simulation(
                # draw_modulus=display_debug_info,
                draw_interval=10,
                save_gif=save_gif,
                yscale="linear"
            )

            # — оценка общего времени на итерации 1 (после прогрева) —
            if (iteration_index == 0) and (max_hours_total is not None):
                est_total_sec = dt_b * (iteration_count + 1)
                if est_total_sec > float(max_hours_total) * 3600.0:
                    print(
                        f"TIME_LIMIT_EXCEEDED: est_total_hours={est_total_sec / 3600:.3f} > "
                        f"max_hours_total={float(max_hours_total):.3f}; "
                        f"iter_count={iteration_count + 1}, dt1={dt_b:.3f}s"
                    )
                    raise RuntimeError(
                        f"TIME_LIMIT_EXCEEDED: est_total_hours={est_total_sec / 3600:.3f} > "
                        f"max_hours_total={float(max_hours_total):.3f}; "
                        f"iter_count={iteration_count + 1}, dt1={dt_b:.3f}s"
                    )

            # ── метрика сходимости: NRMSE по окнам (между итерациями) ──
            cur_main = solver.numerical_solution[-1][:, main_start:main_end]  # (C, main_len)
            cur_int = (np.abs(cur_main) ** 2).astype(np.float64, copy=False)  # (C, main_len)

            if _prev_int_main is not None:
                errs = []
                Lm = cur_int.shape[1]
                for k in range(iteration_count):
                    s = k * w_mask
                    if s >= Lm:
                        break
                    e = min(s + w_mask, Lm)
                    a = _prev_int_main[:, s:e].ravel()
                    b = cur_int[:, s:e].ravel()
                    denom = (a.max() - a.min()) or 1.0
                    errs.append(float(np.sqrt(np.mean((a - b) ** 2)) / denom))
                if errs:
                    _last_max_nrmse = float(np.max(errs))
                if display_debug_info:
                    print(f"NRMSE per window (iter {iteration_index} vs {iteration_index - 1}): "
                          f"{[f'{e:.1e}' for e in np.asarray(errs, float)]}")

            _prev_int_main = cur_int  # обновляем эталон для следующей итерации

            # if display_debug_plots:
            #     plot2D_plotly(
            #         t_stream_ps[::step] * 1e-3,
            #         [np.abs(solver.numerical_solution[0][central_core_ind][::step]) ** 2,
            #          np.abs(solver.numerical_solution[-1][central_core_ind][::step]) ** 2],
            #         names=[f"$|U_{central_core_ind}(z=0,t)|^2$", f"$|U_{central_core_ind}(z=L,t)|^2$"],
            #         x_axis_label='t [ns]', y_axis_label='power [W]'
            #     )

            solver.numerical_solution[0] = initial_data + np.roll(solver.numerical_solution[-1],
                                                                  window_size * upsampling, axis=1) * feedback_coeff

        if display_debug_info:
            print("Elapsed time =", time() - t1)

        # — проверка сходимости по последней паре итераций —
        if _last_max_nrmse > 5e-2:
            print(
                f"ITERATION_NOT_CONVERGED: max_nrmse_last={_last_max_nrmse:.6g} > 5e-2"
            )
            raise RuntimeError(
                f"ITERATION_NOT_CONVERGED: max_nrmse_last={_last_max_nrmse:.6g} > 5e-2"
            )

        if display_debug_plots:
            plot2D_plotly(
                np.fft.fftshift(omega_stream),
                np.abs(np.fft.fftshift(np.fft.fft(solver.numerical_solution[-1][central_core_ind]))) ** 2,
                names=[rf"$|U_{central_core_ind}(z=L,\omega)|^2$"],
                x_axis_label=r'$\omega, \text{rad/ps}$',
                y_axis_label='spectrum intensity [W]', yscale="log", title_text="Spectrum"
            )

        if display_debug_plots:
            plot2D_plotly(
                t_stream_ps * 1e-3,
                np.abs(solver.numerical_solution[-1][central_core_ind][
                       offset_size:offset_size + data_in.shape[1] * upsampling]) ** 2,
                names=[f"$|U_{central_core_ind}(z=0,t)|^2$", f"$|U_{central_core_ind}(z=L,t)|^2$"],
                x_axis_label='t [ns]', y_axis_label='power [W]'
            )

        return {
            "data_out": (solver.numerical_solution[-1][:,
                         offset_size:offset_size + data_in.shape[1] * upsampling:upsampling]),
            "params": {
                "gamma": float(gamma),
                "beta1": float(beta1),
                "beta2": float(beta2),
                "beta1_air": float(beta1_air),
                "feedback_length_m": float(feedback_length_m),
                "feedback_loop_propagation_time_ps": float(feedback_loop_propagation_time),
                "fiber_propagation_time_ps": float(fiber_propagation_time),
                # характерные длины и масштабы:
                "L_D": float(L_D), "L_NL": float(L_NL),
                "L_coupling": float(L_coupling), "L_gain": float(L_gain),
                "time_scale_ps": float(time_scale),
                "length_scale_m": float(length_scale),
                "fiber_length_dimensionless": float(fiber_length_dimensionless),
                "n_z": int(n_z),
                # расчётные настройки окна:
                "window_size": int(window_size),
                "upsampling": int(upsampling),
                "step_number_per_dimensionless_distance": int(step_number_per_dimensionless_distance),
                # итерационная информация (оконная итерация по задержке):
                "iteration_count": int(iteration_count + 1),  # с учётом финальной
                "offset_size": int(offset_size),
                "offset_part": float(offset_part),
            },
        }


# if __name__ == '__main__':
#
#     layer_count = 1.1
#     core_configuration = CoreConfig.hexagonal
#     core_count = get_core_count(core_configuration=core_configuration, ring_count=layer_count)
#
#     # layer 0 - центральная сердцевина
#     # layer 1 - первый круг из 6 сердцевин
#     # layer 2 - второй "круг" из 12 сердцевин, расстояние от которых до центра разное
#     # ...
#     layer_radii_array = np.zeros(int(layer_count) + 1)
#     for i in range(int(layer_count) + 1):
#         if i == 0:
#             layer_radii_array[i] = 0 # [mkm]
#         if i == 1:
#             layer_radii_array[i] = 17.3 # [mkm]
#         if i == 2:
#             layer_radii_array[i] = 50 # [mkm]
#
#         # layer_radii_array[i] = 17.3 * (i * 1.5) # [mkm]
#
#     g0_array = np.zeros(core_count)
#     for i in range(core_count):
#         g0_array[i] = 10.0 # [1/m]
#
#     psat_array = np.zeros(core_count)
#     for i in range(core_count):
#         psat_array[i] = 40 * 5e-4 # мощность насыщения [W]
#
#
#     modulation_frequency_ghz = 40 # GHz
#     mackey_glass_symbol_count = 2**9
#     mask_size = modulation_frequency_ghz
#
#     time_step_ps = 1 / modulation_frequency_ghz * 1e+3 # длина по времени одного отсчета
#     window_size = modulation_frequency_ghz * 30 # в количестве отсчетов
#
#     mg_params = {
#         'tau': 17,
#         'n': 10,
#         'beta': 2,
#         'gamma': 1,
#         'initial_condition': 1.2
#     }
#     seed = 42
#     data_in = mackey_glass_masked(core_count, mackey_glass_symbol_count, mask_size, seed, **mg_params) # √W
#
#     required_avg_power_w = 1.0  # ← задайте требуемую среднюю мощность, W
#     current_avg_power = np.mean(np.abs(data_in) ** 2)  # ⟨|U|²⟩
#     scale_factor = np.sqrt(required_avg_power_w / current_avg_power)
#     data_in *= scale_factor
#
#     kappa = 0.9 # коэффициент обратной связи
#
#     t_start = time()
#
#     # data_out, feedback_length_m = mcf_nn_reservoir_computing_temporal_evolution(
#     #     data_in=data_in,  # ndarray (C, M_in)
#     #     fiber_length_m=0.1,  # физическая длина MCF, m
#     #     time_step_ps=1/modulation_frequency_ghz*1e+3,  # шаг сетки t, ps
#     #     kappa=kappa, # необходимо передать, так как поле добавляется непрерывно,
#     #                         # а внутри считается производная по z
#     #     step_number_per_dimensionless_distance=20,
#     #     layer_count=layer_count,
#     #     layer_radii_array=layer_radii_array,  # радиусы колец, µm
#     #     g0_array=g0_array,
#     #     psat_array=psat_array,
#     #     use_gpu=False,
#     #     display_debug_info=True,
#     #     display_debug_plots=True,
#     #     save_gif=False,
#     #     upsampling = 2000
#     # )
#
#     data_out, feedback_length_m = mcf_nn_reservoir_computing(
#         data_in=data_in,  # ndarray (C, M_in)
#         fiber_length_m=0.1,  # физическая длина MCF, m
#         window_size=window_size,
#         time_step_ps=time_step_ps,  # шаг сетки t, ps
#         step_number_per_dimensionless_distance=20, # ставишь меньше 20 - проверь, что поле не улетело в космос от переусиления
#         upsampling=2,
#         layer_count=layer_count,
#         layer_radii_array=layer_radii_array,  # радиусы колец, µm
#         g0_array=g0_array,
#         psat_array=psat_array,
#         kappa=kappa,
#         use_gpu=False,
#         display_debug_info=True,
#         display_debug_plots=True,
#         save_gif=False
#     )
#
#     print("Total elapsed time =", (time() - t_start) / 60, "min.")
#
#     central_core_ind = 3
#     t = np.arange(data_in.shape[1])
#     plot2D_plotly(t, [np.abs(data_in[central_core_ind]) ** 2,
#                                             np.abs(data_out[central_core_ind]) ** 2],
#                   names=[f"$|U_{central_core_ind}(z=0,t)|^2$", f"$|U_{central_core_ind}(z=L,t)|^2$"],
#                   x_axis_label='t [ns]', y_axis_label='power [W]')


######################################################################

import json, hashlib
import dataclasses as _dc
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Tuple, Literal, Optional

# =========================
# Утилиты, метрики, кэш
# =========================

CACHE_DIR = Path("mcf_rc_cache")


def json_dumps_compact(obj):
    """
    Компактная JSON-сериализация с безопасным приведением numpy-типов:
    - numpy scalars (np.int64, np.float64, np.bool_) -> обычные int/float/bool через .item()
    - numpy.ndarray -> list через .tolist()
    - коллекции и словари обходим рекурсивно
    """
    def _to_jsonable(x):
        # numpy скаляры → базовые типы
        if isinstance(x, np.generic):
            return x.item()
        # numpy массивы → списки
        if isinstance(x, np.ndarray):
            return x.tolist()
        # словари → рекурсивно
        if isinstance(x, dict):
            # ключи на всякий случай приводим к строке (если вдруг попались не-строки)
            return {str(k): _to_jsonable(v) for k, v in x.items()}
        # последовательности/множества → рекурсивно в список
        if isinstance(x, (list, tuple, set)):
            return [_to_jsonable(v) for v in x]
        # остальное — как есть
        return x

    return json.dumps(_to_jsonable(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_of_json(obj: Any) -> str:
    s = json_dumps_compact(obj).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    NRMSE = RMSE / std(y_true). Возвращает NaN для пустых входов без генерации ворнингов NumPy.
    Если std≈0: RMSE≈0 → 0.0, иначе → +inf.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    n = y_true.size
    if n == 0:
        return float('nan')  # валидно, и без предупреждений

    # На всякий случай подровняем длины (если расходятся).
    if y_pred.size != n:
        m = min(n, y_pred.size)
        if m == 0:
            return float('nan')
        y_true = y_true[:m]
        y_pred = y_pred[:m]

    diff = y_true - y_pred
    rmse = float(np.sqrt((diff * diff).mean()))  # безопасно: массив гарантированно непустой

    s = float(np.std(y_true))  # безопасно: массив гарантированно непустой
    if not np.isfinite(s) or s < eps:
        return 0.0 if rmse < eps else float('inf')

    return rmse / s


def train_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1e-6, add_bias: bool = True):
    if add_bias:
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    else:
        Xb = X

    # Убедимся, что используем float64
    Xb = Xb.astype(np.float64)
    y = y.astype(np.float64)

    # Вычисляем матрицу системы
    A = Xb.T @ Xb
    n_features = A.shape[0]

    # Добавляем регуляризацию
    A_reg = A + alpha * np.eye(n_features)

    # Вычисляем число обусловленности
    cond_number = np.linalg.cond(A_reg)
    print(f"Alpha: {alpha}, Condition number: {cond_number}")

    # Если число обусловленности слишком высокое, увеличиваем alpha
    if cond_number > 1e15:
        print("Condition number too high, increasing alpha...")
        # Находим оптимальное alpha через итерации
        for new_alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e+1, 1e+2, 1e+3]:
            A_reg_new = A + new_alpha * np.eye(n_features)
            new_cond = np.linalg.cond(A_reg_new)
            print(f"Trying alpha={new_alpha}, condition={new_cond}")
            if new_cond < 1e15:
                alpha = new_alpha
                A_reg = A_reg_new
                print(f"Using alpha={alpha}")
                break

    # Используем псевдообратную матрицу для устойчивого решения
    try:
        # Пробуем решить систему
        W = np.linalg.solve(A_reg, Xb.T @ y)
    except np.linalg.LinAlgError:
        # Если не получается, используем SVD-решение
        print("Using SVD solution due to numerical issues")
        U, s, Vt = np.linalg.svd(A_reg, full_matrices=False)
        # Отсекаем очень малые сингулярные значения
        s_inv = np.zeros_like(s)
        threshold = np.finfo(np.float64).eps * max(U.shape) * np.max(s)
        for i in range(len(s)):
            if s[i] > threshold:
                s_inv[i] = 1.0 / s[i]
        W = Vt.T @ np.diag(s_inv) @ U.T @ Xb.T @ y

    return W, alpha


def apply_readout(X: np.ndarray, W: np.ndarray, add_bias: bool = True) -> np.ndarray:
    if add_bias:
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    else:
        Xb = X
    return Xb @ W


# =========================
# Конфиги
# =========================

MaskVariant = Literal["temporal_same_all_cores", "temporal_unique_per_core", "spatial_only"]


# "temporal_same_all_cores"     → одна и та же временная маска для всех ядер
# "temporal_unique_per_core"    → своя временная маска для каждого ядра
# "spatial_only"                → без временной маски; только разные постоянные веса по ядрам

@dataclass
class MGConfig:
    """
    Конфиг генерации ряда Маккея–Гласса (MG).

    Поля:
      t_size : int
          Длина используемого ряда после warmup.
      tau, n, beta, gamma, initial_condition, dt : как в mackey_glass(...)
      warmup : int, default=300
          Сколько начальных точек MG отбросить перед нормировкой и использованием.
    """
    t_size: int
    tau: float = 17.0
    n: int = 10
    beta: float = 0.2
    gamma: float = 0.1
    initial_condition: float = 1.2
    dt: float = 1.0
    warmup: int = 300


@dataclass
class MaskConfig:
    """
    Настройки маскирования/масштаба входа.

    Поля:
      mask_size : int
          Число виртуальных узлов на символ (длина маски).
      mask_kind : str, default="uniform"
          Тип маски:
            - "uniform"    → значения равномерно из [-1, 1] (с нулевым средним)
            - "rademacher" → значения в {-1, +1}
            - "gaussian"   → значения ~ N(0,1)
      seed : int | None
          Сид для воспроизводимости масок/весов.
      gain_in : float
          Глобальный масштаб амплитуды входа после маскирования.
    """
    mask_size: int
    mask_kind: str = "uniform"
    seed: Optional[int] = 42
    gain_in: float = 1.0


@dataclass
class ReservoirConfig:
    fiber_length_m: float
    time_step_ps: float  # сигнал - кусочно-постоянная функция, time_step_ps - длительность одного элемента этого сигнала
    step_number_per_dimensionless_distance: int = 20
    upsampling: int = 2
    layer_count: float = 1.0
    layer_radii_array: Tuple[float, ...] = (0.0, 17.3)
    g0_array: Tuple[float, ...] = (10.0,)
    psat_array: Tuple[float, ...] = (0.02,)
    kappa: float = 0.9
    use_gpu: bool = False
    use_torch: bool = False
    num_threads: int | str | None = "default"
    display_debug_info: bool = False
    display_debug_plots: bool = False
    save_figs: bool = False  # Сохраняем все графики. Формат задается в файле конфигурации ./styles/mcf.mplstyle
    save_gif: bool = False
    delay_factor_in_symbols: int | None = None  # число символов в петле обратной связи
    delay_additional_in_mask_steps: int = 0  # дополнительный фазовый сдвиг в шагах маски в петле обратной связи (0..mask_size-1); для spatial_only эффекта не даст
    window_size: Optional[
        int] = None  # Пользователь НЕ задаёт руками; вычисляется из delay_factor_in_symbols/phase внутри запуска
    max_hours_total: Optional[float] = None  # ОГРАНИЧЕНИЕ: оценка общего времени прогона (часы)
    precision: Optional[str] = 'float64' # Размер данных в вычислениях ('float64', 'float32')
    use_dispersion: Optional[bool] = True  # Включать ли слагаемое с дисперсией в модель

@dataclass
class TrainingConfig:
    """
    Настройки обучения рид-аута.

      feature_mode : {"intensity","realimag"}, default="intensity"
          Признаки из выхода резервуара:
            - "intensity" → |U_c(t)|^2 для всех ядер c
            - "realimag"  → concat([Re U_c(t), Im U_c(t)])
      washout : int | None
          Сколько первых состояний резервуара отбросить. None → авто по κ и окну задержки.
      taps : int >= 1
          Сколько последних символов включать в признак рид-аута.
          taps=1 → только текущий символ (история отключена).
      ridge_alpha : float
          L2-регуляризация в ridge-регрессии.
      target_shift : int
          На сколько шагов вперёд предсказываем MG: 1 → one-step-ahead, k → k-step.
      train_frac, val_frac : float
          Доли данных (после washout) на обучение и валидацию; тест = 1 - train - val.
    """
    feature_mode: Literal["intensity", "realimag"] = "intensity"
    washout: Optional[int] = None
    taps: int = 1
    ridge_alpha: float = 1e-6
    target_shift: int = 1
    train_frac: float = 0.6
    val_frac: float = 0.2

@dataclass
class ExperimentConfig:
    core_count: int
    mg: MGConfig
    mask: MaskConfig
    reservoir: ReservoirConfig
    training: TrainingConfig
    variant: MaskVariant

# =========================
# Графики
# =========================

def _plot_temporal_masks(ax, masks: np.ndarray, mask_kind: str):
    """Heatmap временных масок: ось X — индекс внутри символа, ось Y — ядро."""
    C, M = masks.shape
    im = ax.imshow(masks, aspect="auto", interpolation="nearest",
                   extent=[0, M, C, 0], cmap="coolwarm")
    ax.set_title(f"Temporal masks (size={M})", loc="left")
    ax.set_xlabel("mask element index")
    ax.set_ylabel("core index")
    return im

def _plot_spatial_weights(ax, weights: np.ndarray, title: str = "Spatial weights by cores"):
    """Bar-чарт по ядрам."""
    C = weights.shape[0]
    ax.bar(np.arange(C), weights, width=0.7)
    ax.set_title(title, loc="left")
    ax.set_xlabel("core index")
    ax.set_ylabel("weight")

def _reconstruct_masks_or_weights(core_count: int,
                                  variant: str,
                                  mask_size: int,
                                  mask_kind: str,
                                  seed: int | None) -> dict:
    """
    Возвращает один из словарей:
      • {'type':'temporal', 'masks': (C,M)}        — для temporal_* (без каких-либо 'weights')
      • {'type':'spatial',  'weights': (C,)}       — для spatial_only
    """
    rng = np.random.default_rng(seed)

    if variant == "temporal_unique_per_core":
        masks = np.empty((core_count, mask_size), dtype=float)
        for c in range(core_count):
            masks[c] = create_mask(mask_size, rng, kind=mask_kind)
        # НИКАКИХ spatial weights для temporal-режимов
        return {"type": "temporal", "masks": masks}

    if variant == "temporal_same_all_cores":
        mask = create_mask(mask_size, rng, kind=mask_kind)
        masks = np.tile(mask, (core_count, 1))
        # НИКАКИХ spatial weights для temporal-режимов
        return {"type": "temporal", "masks": masks}

    if variant == "spatial_only":
        # В spatial_only подаём постоянные веса на ядра
        weights = rng.uniform(-1.0, 1.0, size=core_count)
        return {"type": "spatial", "weights": weights}

    raise ValueError(f"Unknown variant: {variant}")


def _default_fig_path(basename: str) -> Path:
    fmt = str(plt.rcParams.get("savefig.format", "png")).lower()
    out_dir = Path(__file__).parent  # сохраняем рядом со скриптом
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{basename}.{fmt}"
    if p.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = out_dir / f"{basename}_{ts}.{fmt}"
    return p

def _maybe_savefig(fig,
                   basename: str,
                   explicit_path: Optional[Union[str, Path]] = None,
                   enabled: Optional[bool] = None) -> Optional[Path]:
    """
    explicit_path задан → сохраняем туда (игнорируем enabled).
    explicit_path не задан → сохраняем в <папка скрипта>/<basename>.<fmt>, только если enabled=True.
    Параметры сохранения полностью из rcParams (формат, dpi, bbox, прозрачность и т.д.).
    """
    try:
        if explicit_path is not None:
            fig.savefig(explicit_path)
            return Path(explicit_path)
        if enabled:
            p = _default_fig_path(basename)
            fig.savefig(p)
            return p
    except Exception as e:
        print(f"[warn] savefig failed: {e}")
    return None


def debug_plot_input_overview(cfg, mg_series_used: np.ndarray):
    """
    Рисует:
      1) Полный ряд MG (нормированный так же, как используется в расчёте),
         с пометками: warmup, shift, washout, train/val/test.
      2) Маски (по времени) для каждого ядра ИЛИ пространственные веса (bar),
         в зависимости от варианта.
    """
    # --- восстановим полный MG, чтобы показать warmup слева
    warmup = cfg.mg.warmup
    x_full = mackey_glass(cfg.mg.t_size + warmup,
                          tau=cfg.mg.tau, n=cfg.mg.n, beta=cfg.mg.beta, gamma=cfg.mg.gamma,
                          initial_condition=cfg.mg.initial_condition, dt=cfg.mg.dt)

    # Нормируем ровно как в коде: по куску после warmup
    x_used = x_full[warmup:].astype(float)
    mu, sigma = float(np.mean(x_used)), float(np.std(x_used) + 1e-12)
    x_full_norm = (x_full - mu) / sigma

    # Проверим согласованность длины
    S = cfg.mg.t_size
    assert mg_series_used.shape[0] == S, "mg_series_used длиной не равно t_size"

    # --- индексы сегментов на оси полного ряда
    shift_syms = int(cfg.training.target_shift)  # shift у нас в символах — ок

    # 1) берём washout в отсчётах (если авто — тем же способом, что и в пайплайне)
    if cfg.training.washout is None:
        w_samples = auto_washout_samples(cfg.reservoir, eps=1e-3, min_loops=1, max_loops=3)
    else:
        w_samples = int(cfg.training.washout)

    # 2) переводим washout в символы: делим на «эффективный размер маски»
    #    (для temporal_* это mask_size; для spatial_only — 1)
    M_eff = cfg.mask.mask_size if cfg.variant.startswith("temporal_") else 1
    w_syms = int(np.ceil(w_samples / max(1, M_eff)))  # теперь w_syms в символах

    # 3) дальше считаем разметку ТОЛЬКО в символах
    N_eff = S - shift_syms - w_syms
    if N_eff <= 10:
        print("N_eff =", N_eff, " S =", S, " shift_syms =", shift_syms, " w_syms =", w_syms)
        raise ValueError("Слишком мало символов после shift+washout")

    i_warmup_L = 0
    i_warmup_R = cfg.mg.warmup
    i_shift_L = i_warmup_R
    i_shift_R = i_warmup_R + shift_syms
    i_wash_L = i_shift_R
    i_wash_R = i_shift_R + w_syms
    i_tr_L = i_wash_R
    n_train = int(N_eff * cfg.training.train_frac)
    n_val = int(N_eff * cfg.training.val_frac)
    i_tr_R = i_tr_L + n_train
    i_va_L = i_tr_R
    i_va_R = i_va_L + n_val
    i_te_L = i_va_R
    i_te_R = i_te_L + (N_eff - n_train - n_val)

    # --- рисуем
    fig = plt.figure(figsize=(COL2, COL2*0.62))  # соотношение ~3:2
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.6], hspace=0.35)

    # (1) полный MG с разметкой
    ax1 = fig.add_subplot(gs[0, 0])
    t_full = np.arange(x_full_norm.shape[0])
    ax1.plot(t_full, x_full_norm, label="Mackey-Glass")
    ax1.set_xlim(t_full[0], t_full[-1])  # жёстко фиксируем границы
    ax1.margins(x=0.0)  # убираем автополя по X

    def span_if(a, b, color, label, alpha=0.18):
        """Добавить заливку, только если пересекается с видимым диапазоном и длина ≥ 2."""
        a, b = int(a), int(b)
        if b - a < 2:
            return False
        a_plot = max(a, 0)
        b_plot = min(b, int(t_full[-1]) + 1)
        if b_plot - a_plot < 2:
            return False
        ax1.axvspan(a_plot, b_plot, color=color, alpha=alpha, label=label)
        return True

    shown = []
    if span_if(i_warmup_L, i_warmup_R, "#888888", "warmup"):          shown.append("warmup")
    if span_if(i_shift_L,  i_shift_R,  "#1f77b4", "target shift"):    shown.append("target shift")
    if span_if(i_wash_L,   i_wash_R,   "#ff7f0e", "washout"):         shown.append("washout")
    if span_if(i_tr_L,     i_tr_R,     "#2ca02c", "train"):           shown.append("train")
    if span_if(i_va_L,     i_va_R,     "#9467bd", "val"):             shown.append("val")
    if span_if(i_te_L,     i_te_R,     "#d62728", "test"):            shown.append("test")

    ax1.set_title("Mackey-Glass series: warmup/shift/washout/train/val/test", loc="left")
    ax1.set_xlabel("symbol index")
    ax1.set_ylabel("normalized amplitude")

    # Легенда: только из реально добавленных элементов
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        leg = ax1.legend(uniq.values(), uniq.keys(),
                         ncols=3,
                         bbox_to_anchor=(0.98, 0.98),
                         frameon=True)
        leg.get_frame().set_facecolor((1, 1, 1, 0.6))
        leg.get_frame().set_edgecolor((0, 0, 0, 0.3))

    # (2) маски / пространственные веса — БЕЗ лишних расчётов для temporal_*
    masks_info = _reconstruct_masks_or_weights(core_count=cfg.core_count,
                                               variant=cfg.variant,
                                               mask_size=cfg.mask.mask_size,
                                               mask_kind=cfg.mask.mask_kind,
                                               seed=cfg.mask.seed)

    if masks_info["type"] == "temporal":
        # только heatmap масок
        ax2 = fig.add_subplot(gs[1, 0])
        im = _plot_temporal_masks(ax2, masks_info["masks"], cfg.mask.mask_kind)
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("mask value")
    else:
        # только пространственные веса (spatial_only)
        ax2 = fig.add_subplot(gs[1, 0])
        _plot_spatial_weights(ax2, masks_info["weights"], title="Spatial weights by cores")

    _maybe_savefig(fig, f"input_overview_{cfg.variant}_C{cfg.core_count}",
                   enabled=getattr(cfg.reservoir, "save_figs", False))
    plt.show()


def debug_plot_mg_attractor(cfg,
                            mg_series_used: np.ndarray,
                            title: str = "Mackey-Glass attractor (delay embedding)"):
    """
    3D-визуализация аттрактора Mackey–Glass по delay-embedding: (x(t), x(t-τ), x(t-2τ)).
    Сегменты shift/washout/train/val/test рисуются только если их длина ≥ 2.
    График оформлен под публикации: без сетки/панелей, ortho-проекция, equal-aspect.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (регистрация 3D)
    from matplotlib.ticker import MaxNLocator
    import matplotlib.patheffects as pe

    x1d = np.asarray(mg_series_used, dtype=float).ravel()
    S = x1d.size
    tau_samples = max(1, int(round(float(cfg.mg.tau) / float(cfg.mg.dt))))
    off = 2 * tau_samples
    if S <= off + 1:
        print(f"debug_plot_mg_attractor: серия короче 2τ (S={S}, 2τ={off}) — пропуск.")
        return

    # delay-вложение
    X = x1d[off:]                         # x(t)
    Y = x1d[tau_samples:-tau_samples]     # x(t-τ)
    Z = x1d[:-off]                        # x(t-2τ)
    L = X.shape[0]

    # границы сегментов в ИНДЕКСАХ mg_series_used (0..S)
    shift_syms = int(getattr(cfg.training, "target_shift", 0))
    if cfg.training.washout is None:
        w_samples = auto_washout_samples(cfg.reservoir, eps=1e-3, min_loops=1, max_loops=3)
    else:
        w_samples = int(cfg.training.washout)
    M_eff = cfg.mask.mask_size if str(cfg.variant).startswith("temporal_") else 1
    w_syms = int(np.ceil(w_samples / max(1, M_eff)))
    N_eff = S - shift_syms - w_syms
    n_train = int(N_eff * cfg.training.train_frac) if N_eff > 0 else 0
    n_val   = int(N_eff * cfg.training.val_frac) if N_eff > 0 else 0
    n_test  = max(0, N_eff - n_train - n_val)

    i_shift = (0, max(0, shift_syms))
    i_wash  = (i_shift[1], i_shift[1] + max(0, w_syms))
    i_tr    = (i_wash[1],  i_wash[1]  + max(0, n_train))
    i_va    = (i_tr[1],    i_tr[1]    + max(0, n_val))
    i_te    = (i_va[1],    min(S, i_va[1] + max(0, n_test)))

    segments = [
        ("target shift", "#1f77b4", i_shift),
        ("washout",      "#ff7f0e", i_wash),
        ("train",        "#2ca02c", i_tr),
        ("val",          "#9467bd", i_va),
        ("test",         "#d62728", i_te),
    ]

    # фигура
    fig_w = globals().get("COL2", 6)  # если у тебя есть константа COL2 — используем её
    fig = plt.figure(figsize=(fig_w*0.62, fig_w*0.62))
    ax = fig.add_subplot(111, projection="3d")

    # — оформление под публикации —
    ax.set_proj_type('ortho')  # без перспективных искажений (ortho).  :contentReference[oaicite:4]{index=4}
    # equal-aspect в 3D: подгоняем коробку под диапазоны данных.  :contentReference[oaicite:5]{index=5}
    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
    ax.view_init(elev=20, azim=-15)
    ax.grid(False)  # сетка off (чистая схема).  :contentReference[oaicite:6]{index=6}
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False            # выключить заливку «панелей»  :contentReference[oaicite:7]{index=7}
        axis.pane.set_edgecolor((0, 0, 0, 0))  # скрыть края панелей

    # компактные тики (не больше 4 по каждой оси)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))

    ax.set_title(title, loc="left")
    ax.set_xlabel("x(t)")
    ax.set_ylabel("x(t − τ)")
    ax.set_zlabel("x(t − 2τ)")

    base_lw = float(plt.rcParams.get("lines.linewidth", 1.0))

    def plot_segment(name, color, bounds, min_len: int = 2):
        a, b = int(bounds[0]), int(bounds[1])
        aa, bb = max(a, off), min(b, S)
        if bb - aa >= min_len:
            lo = max(0, min(aa - off, L))
            hi = max(0, min(bb - off, L))
            if hi - lo >= min_len:
                (line,) = ax.plot(X[lo:hi], Y[lo:hi], Z[lo:hi], color=color, label=name)
                # белое «хало» вокруг линии для читаемости на типографском фоне  :contentReference[oaicite:8]{index=8}
                line.set_path_effects([
                    pe.Stroke(linewidth=base_lw*1.8, foreground='white'),
                    pe.Normal()
                ])
                return True
        return False

    for name, color, bounds in segments:
        plot_segment(name, color, bounds)

    # Легенда (только если есть что показывать), полупрозрачный фон
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        leg = ax.legend(uniq.values(), uniq.keys(),
                        bbox_to_anchor=(0.02, 0.98), loc="upper left",
                        frameon=True, title="Segments:")
        leg.get_frame().set_facecolor((1, 1, 1, 0.6))
        leg.get_frame().set_edgecolor((0, 0, 0, 0.3))

    # единое сохранение (формат/параметры — из rcParams, как и раньше)
    _maybe_savefig(fig,
                   f"mg_attractor_{cfg.variant}_C{cfg.core_count}",
                   enabled=getattr(cfg.reservoir, "save_figs", False))

    plt.show()


def debug_plot_post_training_comparison(cfg,
                                        y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        title: str = "Comparison: truth vs prediction",
                                        n_show: int = 2000,
                                        start: int = 0) -> float:
    """
    Рисует сравнение на тесте и возвращает NRMSE по видимому окну.
    Устойчиво работает при коротких окнах: если точек < 2, вместо линии рисуется маркер,
    легенда и NRMSE подавляются, а set_xlim не вызывается (чтобы не ловить warning).
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    N = min(y_true.shape[0], y_pred.shape[0])
    if N == 0:
        print("debug_plot_post_training_comparison: пустые входы — нечего рисовать.")
        return float('nan')

    # окно отображения
    start = max(0, min(int(start), N - 1))
    end = min(start + int(n_show), N)
    x = np.arange(start, end)
    Lvis = int(end - start)

    # <<< НОВОЕ: если окно < 2, но данных >= 2 — показываем хвост из 2 точек >>>
    if Lvis < 2 and N >= 2:
        start, end = max(0, N - 2), N
        x = np.arange(start, end)
        Lvis = end - start

    # NRMSE только если точек >= 2 и std > 0
    if Lvis >= 2:
        denom = float(np.std(y_true[start:end]))
        err = (np.sqrt(np.mean((y_true[start:end] - y_pred[start:end]) ** 2)) /
               (denom + 1e-12)) if denom > 0.0 else float('nan')
    else:
        err = float('nan')

    fig, ax = plt.subplots(figsize=(COL2, COL2*0.33), constrained_layout=True)

    if Lvis >= 2:
        ax.plot(x, y_true[start:end], label="ground truth")
        ax.plot(x, y_pred[start:end], label="prediction")
        ax.set_xlim(x[0], x[-1])  # безопасно только при >= 2 точках
    else:
        # одна точка — аккуратные маркеры, без легенды
        ax.plot(x, y_true[start:end], ls="none", marker="o", ms=3)
        ax.plot(x, y_pred[start:end], ls="none", marker="o", ms=3)

    ax.set_title(
        f"{title}" + (f"   •   NRMSE={err:.4f}" if np.isfinite(err) else "   •   NRMSE=—"),
        loc="left"
    )
    ax.set_xlabel("symbol index")
    ax.set_ylabel("MG series value")
    ax.margins(x=0.0)

    # Легенда — только если были линии
    if Lvis >= 2:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True)
            leg.get_frame().set_facecolor((1, 1, 1, 0.6))  # полупрозрачный фон
            leg.get_frame().set_edgecolor((0, 0, 0, 0.3))

    _maybe_savefig(fig, f"readout_train_val_test_{cfg.variant}_C{cfg.core_count}",
                   enabled=getattr(cfg.reservoir, "save_figs", False))
    plt.show()

    return err


def debug_plot_readout_train_val_test(res: dict,
                                      title: str = "Mackey-Glass series: prediction") -> dict:
    """
    Склейка train→val→test: истина и прогноз. Каждая зона и подпись добавляется
    только если длина сегмента ≥ 2. Метрики для сегментов короче 2 точек → NaN.
    Легенда с полупрозрачным фоном у кромки.
    """
    # --- извлечение (оставляем вашу схему имён) ---
    W = res["W_out"]
    Xtr, ytr = res["X_train"], res["y_train"].reshape(-1, 1)
    Xva, yva = res["X_val"],   res["y_val"].reshape(-1, 1)
    Xte, yte = res["X_test"],  res["y_test"].reshape(-1, 1)

    # предсказания
    ytr_hat = apply_readout(Xtr, W)
    yva_hat = apply_readout(Xva, W) if Xva.size else np.zeros_like(yva)
    yte_hat = apply_readout(Xte, W)

    # склейка
    y_true = np.concatenate([ytr, yva, yte], axis=0).ravel()
    y_pred = np.concatenate([ytr_hat, yva_hat, yte_hat], axis=0).ravel()

    n_tr, n_va, n_te = len(ytr), len(yva), len(yte)
    b_tr = (0, n_tr)
    b_va = (n_tr, n_tr + n_va)
    b_te = (n_tr + n_va, n_tr + n_va + n_te)

    # безопасный NRMSE: только если длина ≥ 2 и std > 0
    def _nrmse_safe(y, yhat):
        y = np.asarray(y).ravel(); yhat = np.asarray(yhat).ravel()
        if y.size < 2 or yhat.size < 2:
            return float('nan')
        s = float(np.std(y))
        if s == 0.0:
            return float('nan')
        return float(np.sqrt(np.mean((yhat - y) ** 2)) / (s + 1e-12))

    m = {
        "nrmse_train": _nrmse_safe(ytr, ytr_hat),
        "nrmse_val":   _nrmse_safe(yva, yva_hat),
        "nrmse_test":  _nrmse_safe(yte, yte_hat),
    }

    # --- рисуем ---
    fig, ax = plt.subplots(figsize=(COL2, COL2*0.33), constrained_layout=True)
    x = np.arange(y_true.shape[0])

    ax.plot(x, y_true, label="ground truth")
    ax.plot(x, y_pred, label="prediction")

    # зоны (рисуем и подписываем только если длина ≥ 2)
    def span_if(bounds, color, label):
        lo, hi = int(bounds[0]), int(bounds[1])
        if hi - lo >= 2:
            ax.axvspan(lo, hi, color=color, alpha=0.18, label=label)
            return True
        return False

    shown_any = False
    if span_if(b_tr, "#2ca02c", f"train  (NRMSE={m['nrmse_train']:.4f})") and np.isfinite(m["nrmse_train"]):
        shown_any = True
    if n_va >= 2 and np.isfinite(m["nrmse_val"]):
        shown_any |= span_if(b_va, "#9467bd", f"val    (NRMSE={m['nrmse_val']:.4f})")
    if n_te >= 2 and np.isfinite(m["nrmse_test"]):
        shown_any |= span_if(b_te, "#d62728", f"test   (NRMSE={m['nrmse_test']:.4f})")

    # разделители — только если есть правая граница предыдущего сегмента
    if n_tr >= 1: ax.axvline(b_tr[1], color="k", lw=1, alpha=0.6)
    if n_va >= 1: ax.axvline(b_va[1], color="k", lw=1, alpha=0.6)

    ax.set_title(title, loc="left")
    ax.set_xlabel("symbol index")
    ax.set_ylabel("MG series value")
    ax.margins(x=0.0)

    # легенда без дублей; только если что-то добавили
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        leg = ax.legend(uniq.values(), uniq.keys(),
                        ncol=2,
                        # loc="upper right",
                        bbox_to_anchor=(0.98, 0.98),
                        frameon=True)
        leg.get_frame().set_facecolor((1, 1, 1, 0.6))
        leg.get_frame().set_edgecolor((0, 0, 0, 0.3))

    cfg = res.get("cfg")
    _maybe_savefig(fig, f"readout_concat_{cfg.variant}_C{cfg.core_count}",
                   enabled=getattr(cfg.reservoir, "save_figs", False))
    plt.show()

    return m


# =========================
# Генерация входа (3 варианта)
# =========================

def _mg_series_from_cfg(cfg: MGConfig) -> np.ndarray:
    x_full = mackey_glass(cfg.t_size + cfg.warmup,
                          tau=cfg.tau, n=cfg.n, beta=cfg.beta, gamma=cfg.gamma,
                          initial_condition=cfg.initial_condition, dt=cfg.dt)
    x_used = x_full[cfg.warmup:]
    return _normalize_zero_mean_unit_std(x_used)

def generate_input_temporal_unique_per_core(core_count: int, mg_cfg: MGConfig, mask_cfg: MaskConfig) -> Tuple[np.ndarray, np.ndarray]:
    mg_series = _mg_series_from_cfg(mg_cfg)
    data_in = mackey_glass_masked(core_count=core_count,
                                  mackey_glass_symbol_count=mg_cfg.t_size,
                                  mask_size=mask_cfg.mask_size,
                                  seed=mask_cfg.seed,
                                  gain_in=mask_cfg.gain_in,
                                  warmup=mg_cfg.warmup,
                                  mask_kind=mask_cfg.mask_kind,
                                  tau=mg_cfg.tau, n=mg_cfg.n, beta=mg_cfg.beta, gamma=mg_cfg.gamma,
                                  initial_condition=mg_cfg.initial_condition, dt=mg_cfg.dt)
    return data_in, mg_series

def generate_input_temporal_same_all_cores(core_count: int, mg_cfg: MGConfig, mask_cfg: MaskConfig) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(mask_cfg.seed)
    mg_series = _mg_series_from_cfg(mg_cfg)
    mask = create_mask(mask_cfg.mask_size, rng, kind=mask_cfg.mask_kind)
    pattern = np.kron(mg_series, mask) * mask_cfg.gain_in
    data_in = np.tile(pattern, (core_count, 1))
    return data_in, mg_series

def generate_input_spatial_only(core_count: int, mg_cfg: MGConfig, mask_cfg: MaskConfig) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(mask_cfg.seed)
    mg_series = _mg_series_from_cfg(mg_cfg)
    weights = rng.uniform(-1.0, 1.0, size=core_count)
    data_in = (weights[:, None] * mg_series[None, :]) * mask_cfg.gain_in
    return data_in, mg_series

# =========================
# Ключ для кэша и запуск MCF
# =========================

# поля, НЕ влияющие на физику, которые не должны входить в ключ
_VOLATILE_FIELDS = {
    ("reservoir", "use_gpu"),
    ("reservoir", "use_torch"),
    ("reservoir", "num_threads"),
    ("reservoir", "display_debug_info"),
    ("reservoir", "display_debug_plots"),
    ("reservoir", "save_figs"),
    ("reservoir", "save_gif"),
    ("reservoir", "max_hours_total"),
    ("training",),
}

# сколько знаков оставлять у float в ключе
_DEFAULT_FLOAT_DIGITS = 3

# при желании можно задать «пер-поле» точность:
_FIELD_DIGITS = {
    ("reservoir", "time_step_ps"): 3,
    ("reservoir", "fiber_length_m"): 3,
    ("reservoir", "psat_array"): 5,
    ("reservoir", "g0_array"): 3,
    ("reservoir", "kappa"): 3,
    ("mask", "gain_in"): 3,
}


def _quantize_for_hash(obj, path=()) -> Any:
    """Рекурсивно: округляет float, приводит np-числа к python-типа, массивы к спискам;
       выбрасывает «летучие» поля; возвращает детерминированную структуру для JSON/hashing."""

    # отфильтруем летучие поля на уровне dict
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            p = path + (k,)
            if p in _VOLATILE_FIELDS:
                continue
            out[k] = _quantize_for_hash(v, p)
        return out

    # списки/кортежи
    if isinstance(obj, (list, tuple)):
        return [_quantize_for_hash(v, path) for v in obj]

    # numpy-скаляры → python-скаляры
    if isinstance(obj, (np.generic,)):
        obj = obj.item()

    # float → округлить
    if isinstance(obj, float):
        nd = _FIELD_DIGITS.get(path, _DEFAULT_FLOAT_DIGITS)
        return float(round(obj, nd))

    # numpy-массивы → к спискам + рекурсивная обработка
    if isinstance(obj, np.ndarray):
        return _quantize_for_hash(obj.tolist(), path)

    # остальные типы (int/str/bool/None)
    return obj


def _params_for_cache(core_count: int, mg_cfg: MGConfig, mask_cfg: MaskConfig,
                      reservoir_cfg: ReservoirConfig, variant: MaskVariant) -> Dict[str, Any]:
    d = dict(
        version="v1.1",
        variant=variant,
        core_count=int(core_count),
        mg=asdict(mg_cfg),
        mask=asdict(mask_cfg),
        reservoir={
            **asdict(reservoir_cfg),
            "layer_radii_array": list(reservoir_cfg.layer_radii_array),
            "g0_array": list(reservoir_cfg.g0_array),
            "psat_array": list(reservoir_cfg.psat_array),
            "precision": str(reservoir_cfg.precision),
            "use_dispersion": bool(getattr(reservoir_cfg, "use_dispersion", True)),
        },
    )
    # ВАЖНО: возвращаем «сырой» словарь для записи в артефакты,
    # а для ключа кэша используем _quantize_for_hash(d) внутри _cache_path().
    return d


def _cache_path(params_dict: Dict[str, Any]) -> Path:
    # Ключ строим НЕ по «сырому», а по канонизированному словарю:
    key = sha256_of_json(_quantize_for_hash(params_dict))
    return CACHE_DIR / f"{key}.npz"


def reconstruct_data_in_from_compact(core_count: int,
                                     variant: str,
                                     mg_series: np.ndarray,
                                     mask_cfg: MaskConfig,
                                     *,
                                     mask_time: Optional[np.ndarray] = None,
                                     masks_time: Optional[np.ndarray] = None,
                                     core_weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Восстановление data_in по компактному представлению:
      • mg_series — ряд Маккея–Гласса длиной S (после нормировки и warmup);
      • для temporal_same_all_cores: mask_time формы (M,);
      • для temporal_unique_per_core: masks_time формы (C,M);
      • для spatial_only: core_weights формы (C,).

    Возвращает массив data_in формы:
      • temporal_*  → (C, S*M)
      • spatial_only → (C, S)
    """
    S = int(mg_series.shape[0])
    C = int(core_count)

    if variant == "temporal_same_all_cores":
        if mask_time is None:
            rng = np.random.default_rng(mask_cfg.seed)
            mask_time = create_mask(mask_cfg.mask_size, rng, kind=mask_cfg.mask_kind)  # (M,)
        pattern = np.kron(mg_series, mask_time) * float(mask_cfg.gain_in)  # (S*M,)
        data_in = np.tile(pattern, (C, 1))  # (C,S*M)
        return data_in

    if variant == "temporal_unique_per_core":
        if masks_time is None:
            rng = np.random.default_rng(mask_cfg.seed)
            masks_time = np.empty((C, mask_cfg.mask_size), dtype=float)
            for c in range(C):
                masks_time[c] = create_mask(mask_cfg.mask_size, rng, kind=mask_cfg.mask_kind)
        out = np.empty((C, S * mask_cfg.mask_size), dtype=float)
        for c in range(C):
            out[c] = np.kron(mg_series, masks_time[c])
        out *= float(mask_cfg.gain_in)
        return out

    if variant == "spatial_only":
        if core_weights is None:
            rng = np.random.default_rng(mask_cfg.seed)
            core_weights = rng.uniform(-1.0, 1.0, size=C)
        return (np.asarray(core_weights, dtype=float).reshape(C, 1) * mg_series.reshape(1, S)) * float(mask_cfg.gain_in)

    raise ValueError(f"Unknown variant: {variant}")


def run_mcf_with_cache(data_in: np.ndarray,
                       params_dict: Dict[str, Any],
                       force_rerun: bool = False,
                       save_cache: bool = False,
                       cache_bits: int = 64) -> Tuple[np.ndarray, float, str]:
    """
    Возвращает (data_out[C,M], params, cache_key).

    КЭШ (НОВЫЙ формат):
      • params_json (строка JSON в uint8);
      • cache_bits (int: 16/32/64);
      • mg_series (float{16,32,64});
      • mask_time (если temporal_same_all_cores) ИЛИ
        masks_time (если temporal_unique_per_core) ИЛИ
        core_weights (если spatial_only);
      • data_out — комплекс:
          - при 64 бит: data_out_c64  (complex128);
          - при 32 бит: data_out_c32  (complex64);
          - при 16 бит: data_out_re_f16, data_out_im_f16 (float16);
      • params (dict).
    """

    def _pack_complex(arr: np.ndarray, bits: int) -> Dict[str, np.ndarray]:
        if bits == 64:
            return {"data_out_c64": np.asarray(arr, dtype=np.complex128)}
        if bits == 32:
            return {"data_out_c32": np.asarray(arr, dtype=np.complex64)}
        if bits == 16:
            a = np.asarray(arr)
            return {
                "data_out_re_f16": a.real.astype(np.float16, copy=False),
                "data_out_im_f16": a.imag.astype(np.float16, copy=False),
            }
        raise ValueError("cache_bits must be one of {16,32,64}")

    def _unpack_complex(z: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, Dict[str, Any]]:
        if "data_out_c64" in z.files:
            data_out = z["data_out_c64"]
        elif "data_out_c32" in z.files:
            data_out = z["data_out_c32"]
        elif "data_out_re_f16" in z.files and "data_out_im_f16" in z.files:
            data_out = z["data_out_re_f16"].astype(np.float32, copy=False) \
                       + 1j * z["data_out_im_f16"].astype(np.float32, copy=False)
        else:
            raise KeyError("cache archive: missing data_out_* arrays (c64/c32 or re_f16+im_f16)")

        if "fiber_params_json" not in z.files:
            params = {}
        else:
            try:
                params = json.loads(z["fiber_params_json"].tobytes().decode("utf-8"))
            except Exception:
                params = {}

        return data_out, params

    def _real_dtype(bits: int) -> np.dtype:
        if bits == 64:
            return np.float64
        if bits == 32:
            return np.float32
        if bits == 16:
            return np.float16
        raise ValueError("cache_bits must be one of {16,32,64}")

    p = _cache_path(params_dict)
    key = p.stem

    # --- ЕДИНООБРАЗНЫЙ ПУТЬ К КЭШУ: (корень проекта)/scripts/mcf_rc_cache ---
    from pathlib import Path as _Path
    _base_dir = _Path(__file__).parent.resolve()  # ./scripts
    _p_scripts = (_base_dir / "mcf_rc_cache" / p.name)  # ./scripts/mcf_rc_cache/<hash>.npz
    _p_scripts.parent.mkdir(parents=True, exist_ok=True)

    if _p_scripts.exists() and not force_rerun:
        z = np.load(_p_scripts, allow_pickle=False)

        if "mg_series" in z.files:
            mg_series = z["mg_series"]
            variant = params_dict["variant"]
            mask_cfg = MaskConfig(**params_dict["mask"])
            C = int(params_dict["core_count"])

            mask_time = z["mask_time"] if "mask_time" in z.files else None
            masks_time = z["masks_time"] if "masks_time" in z.files else None
            core_weights = z["core_weights"] if "core_weights" in z.files else None

            _ = reconstruct_data_in_from_compact(
                core_count=C,
                variant=variant,
                mg_series=mg_series.astype(float, copy=False),
                mask_cfg=mask_cfg,
                mask_time=mask_time,
                masks_time=masks_time,
                core_weights=core_weights
            )
            data_out, params = _unpack_complex(z)
            return data_out, params, key

        data_out, params = _unpack_complex(z)
        return data_out, params, key

    rc = params_dict["reservoir"]
    res = mcf_nn_reservoir_computing(
        data_in=np.array(data_in, dtype=np.complex128),
        fiber_length_m=rc["fiber_length_m"],
        window_size=rc["window_size"],
        time_step_ps=rc["time_step_ps"],
        step_number_per_dimensionless_distance=int(rc["step_number_per_dimensionless_distance"]),
        upsampling=int(rc["upsampling"]),
        layer_count=rc["layer_count"],
        layer_radii_array=tuple(rc["layer_radii_array"]),
        g0_array=tuple(rc["g0_array"]),
        psat_array=tuple(rc["psat_array"]),
        kappa=rc["kappa"],
        use_gpu=bool(rc["use_gpu"]),
        use_torch=bool(rc["use_torch"]),
        num_threads=rc["num_threads"],
        display_debug_info=bool(rc["display_debug_info"]),
        display_debug_plots=False, # bool(rc["display_debug_plots"]),
        save_gif=bool(rc["save_gif"]),
        max_hours_total=rc.get("max_hours_total", None),
        precision=rc.get("precision", None),
        use_dispersion=bool(rc.get("use_dispersion", True)),
    )

    data_out = res["data_out"]
    params_out = res.get("params", {})
    fiber_params_json = json.dumps(params_out, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    C = int(params_dict["core_count"])
    mg_cfg = MGConfig(**params_dict["mg"])
    mask_cfg = MaskConfig(**params_dict["mask"])
    variant = params_dict["variant"]

    mg_series = _mg_series_from_cfg(mg_cfg).astype(_real_dtype(cache_bits), copy=False)

    mask_time = None
    masks_time = None
    core_weights = None

    rng = np.random.default_rng(mask_cfg.seed)
    if variant == "temporal_same_all_cores":
        mask_time = create_mask(mask_cfg.mask_size, rng, kind=mask_cfg.mask_kind).astype(_real_dtype(cache_bits),
                                                                                         copy=False)
    elif variant == "temporal_unique_per_core":
        masks_time = np.empty((C, mask_cfg.mask_size), dtype=_real_dtype(cache_bits))
        for c in range(C):
            masks_time[c] = create_mask(mask_cfg.mask_size, rng, kind=mask_cfg.mask_kind).astype(
                _real_dtype(cache_bits),
                copy=False)
    elif variant == "spatial_only":
        core_weights = rng.uniform(-1.0, 1.0, size=C).astype(_real_dtype(cache_bits), copy=False)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if save_cache:
        np.savez_compressed(
            _p_scripts,
            params_json=np.frombuffer(json_dumps_compact(params_dict).encode("utf-8"), dtype=np.uint8),
            cache_bits=np.asarray(int(cache_bits), dtype=np.int32),
            mg_series=mg_series,
            **({"mask_time": mask_time} if mask_time is not None else {}),
            **({"masks_time": masks_time} if masks_time is not None else {}),
            **({"core_weights": core_weights} if core_weights is not None else {}),
            **_pack_complex(data_out, cache_bits),
            **({"fiber_params_json": np.frombuffer(fiber_params_json,
                                                   dtype=np.uint8)} if fiber_params_json is not None else {}),
        )
    return data_out, params_out, key


# =========================
# Признаки/таргет/сплиты + free-running
# =========================

def make_states(data_out: np.ndarray,
                feature_mode: Literal["intensity", "realimag"] = "intensity") -> np.ndarray:
    if feature_mode == "intensity":
        X = (np.abs(data_out) ** 2).T  # (T,C)
    elif feature_mode == "realimag":
        X = np.hstack([data_out.real.T, data_out.imag.T])  # (T,2C)
    else:
        raise ValueError("feature_mode ∈ {'intensity','realimag'}")
    return np.asarray(X, dtype=np.float64)

def split_train_val_test(N: int, train_frac: float, val_frac: float):
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)

    # аккуратно подрезаем под доступные N, чтобы не получить отрицательный test
    n_train = max(0, min(n_train, N))
    n_val = max(0, min(n_val, N - n_train))

    # test может быть нулевым — это ок
    # Возвращаем (slice_trainval, slice_train, slice_val, slice_test)
    return (
        slice(0, n_train + n_val),
        slice(0, n_train),
        slice(n_train, n_train + n_val),
        slice(n_train + n_val, N),
    )


def pseudo_free_run(y_seed: np.ndarray,
                    X_seq: np.ndarray,
                    W_out: np.ndarray,
                    horizon: int,
                    add_bias: bool = True) -> np.ndarray:
    """
    Псевдо free-running на фиксированной последовательности состояний X_seq:
    ŷ(t+1)=W·x(t), затем просто сдвигаем по готовым x(t).
    Это НЕ замыкает выход в резервуар (для настоящего free-run нужен t-marching).
    """
    preds = []
    T = X_seq.shape[0]
    start = int(y_seed.shape[0])  # обычно 1–5 стартовых точек
    t = start - 1
    while len(preds) < horizon and t < T - 1:
        yhat = float(apply_readout(X_seq[t:t + 1], W_out, add_bias=add_bias).ravel()[0])
        preds.append(yhat)
        t += 1
    return np.asarray(preds, dtype=np.float64).reshape(-1, 1)


# =========================
# Функции для оценок размеров данных для обучения
# =========================

def compute_window_size_samples(cfg: ExperimentConfig) -> int:
    """
    Возвращает window_size в ОТСЧЁТАХ:
      window_size = delay_factor_in_symbols * M_eff + phase,
    где M_eff = mask_size для temporal_* и 1 для spatial_only,
        phase = delay_additional_in_mask_steps (0..mask_size-1).
    """
    M_eff = cfg.mask.mask_size if cfg.variant.startswith("temporal_") else 1
    phase = int(getattr(cfg.reservoir, "delay_additional_in_mask_steps", 0)) % max(1, int(M_eff))
    delay_syms = int(getattr(cfg.reservoir, "delay_factor_in_symbols", 30))
    return delay_syms * int(M_eff) + phase


def auto_washout_samples(res_cfg: ReservoirConfig,
                         eps: float = 1e-3,
                         min_loops: int = 1,
                         max_loops: int = 3) -> int:
    """
    Возвращает washout в отсчётах, исходя из окна задержки и коэффициента обратной связи κ.

    Модель: затухание ~ κ^(N_loops). Ищем N_loops так, чтобы κ^(N_loops) ≤ eps.
    Затем washout = N_loops * window_size.

    Параметры:
      eps       – целевой уровень подавления транзиента (10^-3 по умолчанию);
      min_loops – нижняя граница числа оборотов (обычно 1);
      max_loops – верхняя граница (обычно 3; больше редко нужно).
    """
    delay = int(res_cfg.window_size)  # число отсчётов в одном обходе петли
    kappa = float(res_cfg.kappa)
    # защитим расчёт для крайностей κ
    k_eff = min(0.99, max(1e-6, kappa))  # κ ∈ (0, 0.99]
    # требуемое число оборотов без клипов
    n_loops_ideal = np.log(eps) / np.log(k_eff)  # оба логарифма < 0, отношение > 0
    n_loops = int(np.ceil(np.clip(n_loops_ideal, min_loops, max_loops)))
    return max(delay * n_loops, delay)  # минимум один виток


def estimate_required_t_size_fast(cfg) -> int:
    """
    Быстрая оценка общего числа символов (t_size) для символьно-уровневого обучения.
    Использует конфиг эксперимента:
      - cfg.core_count, cfg.variant, cfg.mask.mask_size
      - cfg.training.feature_mode, cfg.training.taps
      - cfg.training.train_frac, cfg.training.val_frac
      - cfg.training.washout, cfg.training.target_shift

    Правило 'fast':
      Train  >= max(5·D, 1000),
      Val    >= max(2·D,  500)  (если val_frac > 0),
      Test   >= max(2·D,  500),
    где D = C · M_eff · taps · (1 или 2).
    Возвращается t_size уже с добавленными washout и target_shift.
    """
    C = int(cfg.core_count)
    variant = str(cfg.variant)
    mask_size = int(cfg.mask.mask_size)
    # эффективные узлы на символ
    M_eff = mask_size if variant.startswith("temporal_") else 1

    # множитель признаков: intensity=1, field=2 (Re+Im)
    feat_mode = getattr(cfg.training, "feature_mode", "intensity").lower()
    feat_mul = 2 if feat_mode in ("field", "realimag") else 1

    # taps из конфига (дефолт 1)
    taps = int(getattr(cfg.training, "taps", 1))

    # размерность признаков на символ
    D = C * M_eff * taps * feat_mul

    # доли сплита
    train_frac = float(cfg.training.train_frac)
    val_frac = float(getattr(cfg.training, "val_frac", 0.0))
    test_frac = 1.0 - train_frac - val_frac
    if test_frac < -1e-16:
        raise ValueError("train_frac + val_frac must be <= 1.0")

    # минимальные требования в символах (fast-политика)
    req_train = max(5 * D, 1000)
    req_val = max(2 * D, 500) if val_frac > 0.0 else 0
    req_test = max(2 * D, 500) if test_frac > 0.0 else 0

    # общее число символов до добавления служебных хвостов
    S_needed = max(
        np.ceil(req_train / train_frac),
        np.ceil(req_val / val_frac) if val_frac > 0.0 else 0,
        np.ceil(req_test / test_frac) if test_frac > 0.0 else 0,
    )

    # служебные хвосты
    washout = getattr(cfg.training, "washout", None)
    if washout is None:
        washout = 300  # безопасный дефолт
    washout = int(washout)
    target_shift = int(getattr(cfg.training, "target_shift", 0))

    S_total = int(S_needed + washout + max(target_shift, 0))
    return S_total

def learning_curve_for_result_plotly(
    result: dict,
    *,
    add_bias: bool = True,
) -> dict:
    """
    Строит лёрнинг-кривую (validation NRMSE vs S_train) на УЖЕ посчитанных состояниях,
    без пересчёта физики. Ожидает в result поля:
      - "X_train", "y_train", "X_val", "y_val".
    Внутри:
      • берёт сетку S_train (6 точек от ~10% до 100% train),
      • в каждой точке подбирает alpha риджа по лог-сетке через ОДНУ SVD,
      • возвращает список метрик + сохраняет интерактивный HTML-график (Plotly).

    Возвращает:
      {
        "curve": [{"S_train", "alpha_best", "nrmse_train", "nrmse_val"}, ...],
        "plot_saved_to": "<путь к .html>" | None
      }
    """

    # ---- входные массивы ----
    Xtr_full = result["X_train"]
    ytr_full = result["y_train"]
    Xva = result["X_val"]
    yva = result["y_val"]

    n_train = Xtr_full.shape[0]
    # Сетка S_train: 6 точек от ~10% до 100% train
    lo = max(5, n_train // 10)
    train_sizes_syms = sorted(set(np.linspace(lo, n_train, num=100, dtype=int).tolist()))

    # Сетка alpha по умолчанию
    alphas = np.logspace(-6, 2, 41)

    curve = []
    for S in train_sizes_syms:
        if S < 2:
            continue
        S_use = min(S, n_train)
        Xtr = Xtr_full[:S_use, :]
        ytr = ytr_full[:S_use, :]

        if add_bias:
            Xb_tr = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))])
            Xb_va = np.hstack([Xva, np.ones((Xva.shape[0], 1))])
        else:
            Xb_tr, Xb_va = Xtr, Xva

        # --- одна SVD на весь путь по alpha ---
        U, Svd, Vt = np.linalg.svd(Xb_tr, full_matrices=False)
        Ut_y = U.T @ ytr

        best_alpha, best_W, best_val = None, None, np.inf
        for a in alphas:
            shrink = Svd / (Svd * Svd + a)
            W = (Vt.T * shrink) @ Ut_y
            val = nrmse(yva, Xb_va @ W)
            if val < best_val:
                best_val, best_alpha, best_W = val, float(a), W
        print("best_alpha =", best_alpha)
        nrmse_tr = nrmse(ytr, Xb_tr @ best_W)

        curve.append({
            "S_train": int(S_use),
            "alpha_best": float(best_alpha),
            "nrmse_train": float(nrmse_tr),
            "nrmse_val": float(best_val),
        })

    # ---- график (Plotly) ----
    plot_saved_to = None
    try:
        import plotly.graph_objects as go

        Ss = [d["S_train"] for d in curve]
        vals = [d["nrmse_val"] for d in curve]
        trs  = [d["nrmse_train"] for d in curve]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Ss, y=vals, mode="lines+markers", name="val NRMSE"))
        fig.add_trace(go.Scatter(x=Ss, y=trs,  mode="lines+markers", name="train NRMSE", line=dict(dash="dash")))
        fig.update_layout(
            title="Learning curve (validation NRMSE vs S_train)",
            xaxis_title="Число обучающих символов S_train",
            yaxis_title="NRMSE",
            template="plotly_white",
            legend=dict(x=0.02, y=0.98),
        )

        # Автогенерация имени HTML рядом с экспериментом.
        # Пытаемся использовать несколько полей из result["params"], если есть
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "lc"
        try:
            p = result.get("params", {})
            ms = p.get("mask_size", "M")
            df = p.get("delay_factor_in_symbols", "D")
            up = p.get("upsampling", "U")
            tag = f"lc_m{ms}_d{df}_u{up}"
        except Exception:
            pass
        plot_saved_to = f"{tag}_{stamp}.html"
        # fig.write_html(plot_saved_to, include_plotlyjs="cdn")
        # Покажем интерактивно
        fig.show()
    except Exception as _e:
        # не роняем эксперимент из-за графика
        result["_learning_curve_plot_error"] = str(_e)

    return {"curve": curve, "plot_saved_to": plot_saved_to}



# =========================
# Сценарии запуска
# =========================

def run_temporal_unique_per_core(base_cfg: ExperimentConfig,
                                 n_trials_opt: int = 0,
                                 free_run_horizon: int = 0,
                                 force_rerun=False,
                                 save_cache=False) -> Dict[str, Any]:
    cfg = base_cfg
    cfg.variant = "temporal_unique_per_core"
    if n_trials_opt > 0:
        return optimize_hyperparams(cfg, n_trials=n_trials_opt,
                                    n_jobs=physical_cpu_count(),
                                    free_run_horizon=free_run_horizon, force_rerun=force_rerun, save_cache=save_cache)
    else:
        return run_single_experiment(cfg, free_run_horizon=free_run_horizon, force_rerun=force_rerun, save_cache=save_cache)


def run_temporal_same_all_cores(base_cfg: ExperimentConfig,
                                n_trials_opt: int = 0,
                                free_run_horizon: int = 0,
                                force_rerun=False,
                                save_cache=False) -> Dict[str, Any]:
    cfg = base_cfg
    cfg.variant = "temporal_same_all_cores"
    if n_trials_opt > 0:
        return optimize_hyperparams(cfg, n_trials=n_trials_opt,
                                    n_jobs=physical_cpu_count(),
                                    free_run_horizon=free_run_horizon, force_rerun=force_rerun, save_cache=save_cache)
    else:
        return run_single_experiment(cfg, free_run_horizon=free_run_horizon, force_rerun=force_rerun, save_cache=save_cache)


def run_spatial_only(base_cfg: ExperimentConfig,
                     n_trials_opt: int = 0,
                     free_run_horizon: int = 0,
                     force_rerun=False,
                     save_cache=False) -> Dict[str, Any]:
    cfg = base_cfg
    cfg.variant = "spatial_only"
    if n_trials_opt > 0:
        return optimize_hyperparams(cfg, n_trials=n_trials_opt,
                                    n_jobs=physical_cpu_count(),
                                    free_run_horizon=free_run_horizon, force_rerun=force_rerun, save_cache=save_cache)
    else:
        return run_single_experiment(cfg, free_run_horizon=free_run_horizon, force_rerun=force_rerun, save_cache=save_cache)


# =========================
# Полный пайплайн одного прогона
# =========================

def run_single_experiment(cfg: ExperimentConfig,
                          free_run_horizon: int = 0,
                          force_rerun: bool = False,
                          save_cache: bool = False,
                          do_learning_curve: bool = False) -> Dict[str, Any]:
    """
    1) генерируем вход/ряд MG,
    2) считаем/читаем из кэша MCF,
    3) формируем X,y (символьная сетка, flatten по узлам×ядрам, опционально taps),
       сплиты, обучаем ridge, метрики,
    4) по желанию — pseudo free-run и сохранение артефактов.
    """

    # вход
    if cfg.variant == "temporal_unique_per_core":
        data_in, mg_series = generate_input_temporal_unique_per_core(cfg.core_count, cfg.mg, cfg.mask)
    elif cfg.variant == "temporal_same_all_cores":
        data_in, mg_series = generate_input_temporal_same_all_cores(cfg.core_count, cfg.mg, cfg.mask)
    elif cfg.variant == "spatial_only":
        data_in, mg_series = generate_input_spatial_only(cfg.core_count, cfg.mg, cfg.mask)
    else:
        raise ValueError(f"Unknown variant: {cfg.variant}")

    # 1) window_size из delay_factor/M_eff/phase — вычисляем ОДИН РАЗ и кладём в cfg
    ws = compute_window_size_samples(cfg)
    cfg_with_ws = _dc.replace(cfg, reservoir=_dc.replace(cfg.reservoir, window_size=ws))

    # 2) рисовалка уже с заполненным window_size
    if cfg.reservoir.display_debug_plots:
        debug_plot_input_overview(cfg_with_ws, mg_series_used=mg_series)
        debug_plot_mg_attractor(cfg_with_ws, mg_series_used=mg_series)

    # ключ кэша
    params_dict = _params_for_cache(cfg_with_ws.core_count, cfg_with_ws.mg, cfg_with_ws.mask,
                                    cfg_with_ws.reservoir, cfg_with_ws.variant)
    # основной запуск с кэшированием
    data_out, params_out, cache_key = run_mcf_with_cache(data_in,
                                                         params_dict,
                                                         force_rerun=force_rerun,
                                                         save_cache=save_cache,
                                                         cache_bits=64)

    # ── признаки/таргеты: ПЕРЕХОД НА СИМВОЛЫ + taps
    X_full = make_states(data_out, feature_mode=cfg.training.feature_mode)  # (T,D)

    # эффективные «узлы на символ»
    M_eff = cfg.mask.mask_size if cfg.variant.startswith("temporal_") else 1

    T, D = X_full.shape
    S = T // M_eff
    if S < 10:
        raise ValueError("Слишком мало символов: увеличьте длину MG или уменьшите mask_size")

    # (S, M_eff, D) → (S, M_eff*D)   — flatten по узлам×ядрам
    X_blocks = X_full[:S * M_eff].reshape(S, M_eff, D)
    X_sym = X_blocks.reshape(S, M_eff * D)  # (S, F)

    # taps по символам: concat [x_s, x_{s-1}, ..., x_{s-(taps-1)}]
    def _make_tapped(Xs: np.ndarray, taps_: int) -> np.ndarray:
        if taps_ <= 1:
            return Xs
        S_, F = Xs.shape
        L = S_ - taps_ + 1
        if L <= 0:
            raise ValueError("taps слишком велик для числа символов")
        Xt = np.empty((L, F * taps_), dtype=Xs.dtype)
        for k in range(taps_):
            Xt[:, k * F:(k + 1) * F] = Xs[(taps_ - 1 - k):(taps_ - 1 - k + L), :]
        return Xt

    X_tapped = _make_tapped(X_sym, int(cfg.training.taps))  # (S - taps + 1, F*taps)
    y_sym = mg_series.reshape(-1, 1)  # (S,1)

    # сдвиг таргета на target_shift СИМВОЛОВ
    shift_syms = int(cfg.training.target_shift)
    if (cfg.training.taps - 1 + shift_syms) >= y_sym.shape[0]:
        raise ValueError("target_shift или taps слишком велики для длины ряда")
    y_aligned = y_sym[(cfg.training.taps - 1 + shift_syms):, :]  # (S - (taps-1) - shift, 1)
    X_aligned = X_tapped[:y_aligned.shape[0], :]

    # washout: авто в СЭМПЛАХ → переводим в СИМВОЛЫ
    if cfg.training.washout is None:
        w_samples = auto_washout_samples(cfg_with_ws.reservoir, eps=1e-3, min_loops=1, max_loops=3)
    else:
        w_samples = int(cfg.training.washout)
    w_syms = int(np.ceil(w_samples / max(1, M_eff)))
    if w_syms >= X_aligned.shape[0] - 10:
        raise ValueError("washout (в символах) слишком большой относительно длины данных")

    Xw, yw = X_aligned[w_syms:, :], y_aligned[w_syms:, :]

    # сплиты
    sl_trainval, sl_train, sl_val, sl_test = split_train_val_test(
        Xw.shape[0], cfg.training.train_frac, cfg.training.val_frac
    )
    Xtr, ytr = Xw[sl_train], yw[sl_train]
    Xva, yva = Xw[sl_val], yw[sl_val]
    Xte, yte = Xw[sl_test], yw[sl_test]

    # обучение рид-аута
    W, ridge_alpha = train_ridge(Xtr, ytr, alpha=cfg.training.ridge_alpha, add_bias=True)

    # метрики
    ytr_hat = apply_readout(Xtr, W)
    yva_hat = apply_readout(Xva, W)
    yte_hat = apply_readout(Xte, W)
    metrics = dict(
        nrmse_train=nrmse(ytr, ytr_hat),
        nrmse_val=nrmse(yva, yva_hat),
        nrmse_test=nrmse(yte, yte_hat),
        ridge_alpha=ridge_alpha,
        T_total=int(X_full.shape[0]),
        features_dim=int(Xw.shape[1]),
        taps=int(cfg.training.taps),
    )

    # pseudo free-run (по тестовым состояниям; это не замкнутый контур)
    y_free = None
    if free_run_horizon > 0:
        seed_len = min(5, yte.shape[0])
        y_seed = yte[:seed_len]
        y_free = pseudo_free_run(y_seed, Xte, W, horizon=free_run_horizon, add_bias=True)

    result = dict(
        cfg=cfg,
        params_json=json_dumps_compact(params_dict),
        metrics=metrics,
        W_out=W,
        X_train=Xtr, y_train=ytr,
        X_val=Xva, y_val=yva, y_val_hat=yva_hat,
        X_test=Xte, y_test=yte, y_test_hat=yte_hat,
        data_in=data_in,
        data_out=data_out,
        mg_series=mg_series,
        cache_key=cache_key,
        y_free_run=y_free,
        # NEW: пробрасываем физические/служебные параметры для Optuna Dashboard
        fiber_params=params_out
    )

    if cfg.reservoir.display_debug_plots:
        # Выбираем первую доступную пару (предпочтительно с длиной ≥2)
        pairs = [
            ("test", yte, yte_hat),
            ("val", yva, yva_hat),
            ("train", ytr, ytr_hat),
        ]
        chosen = next(((name, y, yhat) for name, y, yhat in pairs
                       if y is not None and yhat is not None and min(y.shape[0], yhat.shape[0]) >= 2), None)
        if chosen is None:
            chosen = next(((name, y, yhat) for name, y, yhat in pairs
                           if y is not None and yhat is not None and min(y.shape[0], yhat.shape[0]) >= 1), None)

        if chosen is not None:
            name, y_sel, yhat_sel = chosen
            title_map = {
                "test": "MCF-RC: test symbols",
                "val": "MCF-RC: validation symbols",
                "train": "MCF-RC: training symbols",
            }
            n = min(y_sel.shape[0], yhat_sel.shape[0])
            debug_plot_post_training_comparison(
                cfg_with_ws,
                y_true=y_sel[:n].ravel(),
                y_pred=yhat_sel[:n].ravel(),
                title=title_map.get(name, "MCF-RC: последовательность"),
            )
        else:
            print("Нет данных ни для test, ни для val, ни для train — пропускаю сравнение.")

        debug_plot_readout_train_val_test(result, title="Mackey-Glass series: training and prediction")

    if do_learning_curve:
        # проверим, что нужные поля присутствуют
        missing = [k for k in ("X_train", "y_train", "X_val", "y_val") if k not in result]
        if missing:
            raise KeyError(f"Для лёрнинг-кривой не хватает полей в result: {missing}")
        lc = learning_curve_for_result_plotly(result, add_bias=True)
        result["learning_curve"] = lc["curve"]
        result["learning_curve_plot_path"] = lc["plot_saved_to"]

    return result


# =========================
# Оптимизация (только Optuna)
# =========================

def optimize_hyperparams(base_cfg: ExperimentConfig,
                         n_trials: int,
                         n_jobs: Optional[int] = None,
                         free_run_horizon: int = 0,
                         force_rerun: bool = False,
                         save_cache: bool = False) -> Dict[str, Any]:
    """
    Подбор гиперпараметров с Optuna.

    Что тюним всегда: kappa, g0, psat, fiber_length, gain_in (лог-шкала).
    Опционально: mask_size и delay_factor_in_symbols (а НЕ прямой window_size).
    Внутри trial все вычисления — в 1 поток (temporary_thread_limits(1)), чтобы не было nested parallelism.
    """
    import optuna
    import warnings
    try:
        # публичного алиаса у класса нет, поэтому берем из _experimental
        from optuna._experimental import ExperimentalWarning  # noqa: WPS433
    except Exception:
        class ExperimentalWarning(Warning):  # fallback на всякий случай
            pass
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    from pathlib import Path
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    from optuna.trial import TrialState
    from optuna.exceptions import TrialPruned

    # --- надёжное хранилище без конфликтов SQLite (журнал рядом со скриптом) ---
    base_dir = Path(__file__).parent.resolve()
    journal_path = (base_dir / "mcf_optuna.journal").resolve()
    # На Windows избегаем symlink-локов: используем OpenLock (без прав на symlink).
    lock_obj = JournalFileOpenLock(str(journal_path)) if os.name == "nt" else None
    storage = JournalStorage(JournalFileBackend(str(journal_path), lock_obj=lock_obj))

    # --- на всякий случай создадим каталог для кэша npz (если кэшер туда пишет) ---
    try:
        (base_dir / "mcf_rc_cache").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # --- определим n_jobs по умолчанию ---
    if n_jobs is None:
        if os.getenv("MCF_BASH", ""):
            # в распределённом bash-режиме один поток на процесс
            n_jobs = 1
        else:
            # локальный режим: параллелим в одном процессе
            n_jobs = max(1, physical_cpu_count())
    # ваш спец-кейс для ПК НГУ
    if n_jobs == 6:
        n_jobs = 5

    print("n_trials =", n_trials)
    print("n_jobs   =", n_jobs)

    best = dict(score=float("inf"), res=None, params=None)

    def _round(x: float, digit=3) -> float:
        return float(round(float(x), digit))

    def _flatten(prefix, obj):
        """Плоское представление словаря для user_attrs: ('a.b', value)."""
        import numbers
        if obj is None:
            yield (prefix, None); return
        if isinstance(obj, dict):
            for k, v in obj.items():
                newp = f"{prefix}.{k}" if prefix else str(k)
                yield from _flatten(newp, v)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                newp = f"{prefix}[{i}]"
                yield from _flatten(newp, v)
        elif isinstance(obj, (numbers.Number, str, bool)):
            yield (prefix, obj)
        else:
            yield (prefix, str(obj))

    def objective(trial: "optuna.trial.Trial") -> float:
        with temporary_thread_limits(1):
            # --- 1) основные гиперпараметры резерваура и усиления
            kappa = _round(trial.suggest_float("kappa", 0.1, 0.99)) #, step=1e-3)
            g0 = _round(trial.suggest_float("g0", 0.01, 20.0, log=True))
            psat = _round(trial.suggest_float("psat", 0.00001, 0.1, log=True), digit=5)
            fiber_length = _round(trial.suggest_float("fiber_length", 0.2, 3, log=True))
            gain_in = _round(trial.suggest_float("gain_in", 1, 1e+2, log=True))
            mask_size = trial.suggest_int("mask_size", 30, 400)
            delay_factor = trial.suggest_int("delay_factor_in_symbols", 1, 20)

            # --- 3) конфиг trial'а (прокидываем delay_factor_in_symbols, а не window_size)
            cfg = ExperimentConfig(
                core_count=base_cfg.core_count,
                mg=base_cfg.mg,  # длину ряда не меняем в оптимизации
                mask=MaskConfig(mask_size=mask_size,
                                mask_kind=base_cfg.mask.mask_kind,
                                seed=base_cfg.mask.seed,
                                gain_in=gain_in),
                reservoir=ReservoirConfig(
                    fiber_length_m=fiber_length,
                    time_step_ps=base_cfg.reservoir.time_step_ps,
                    step_number_per_dimensionless_distance=base_cfg.reservoir.step_number_per_dimensionless_distance,
                    upsampling=base_cfg.reservoir.upsampling,
                    layer_count=base_cfg.reservoir.layer_count,
                    layer_radii_array=base_cfg.reservoir.layer_radii_array,
                    g0_array=tuple([g0] * base_cfg.core_count),
                    psat_array=tuple([psat] * base_cfg.core_count),
                    kappa=kappa,
                    use_gpu=base_cfg.reservoir.use_gpu,
                    use_torch=base_cfg.reservoir.use_torch,
                    num_threads=1,
                    display_debug_info=False,
                    display_debug_plots=False,
                    save_figs=False,
                    save_gif=False,
                    delay_factor_in_symbols=int(delay_factor),
                    delay_additional_in_mask_steps=base_cfg.reservoir.delay_additional_in_mask_steps,
                    max_hours_total=base_cfg.reservoir.max_hours_total,
                    precision=base_cfg.reservoir.precision,
                    use_dispersion=base_cfg.reservoir.use_dispersion,
                ),
                training=base_cfg.training,
                variant=base_cfg.variant
            )

            # --- 4) запуск и метрика (плохие конфиги помечаем PRUNED, чтобы НЕ шли в COMPLETE)
            try:
                res = run_single_experiment(cfg, free_run_horizon=free_run_horizon,
                                            force_rerun=force_rerun, save_cache=save_cache)
            except (AssertionError, ValueError, RuntimeError) as e:
                msg = str(e).lower()
                if ("feedback_length" in msg and "fiber" in msg) or "invalid_config" in msg:
                    trial.set_user_attr("skip_reason", "feedback_length_le_fiber_length")
                    print(cfg, "skip_reason", "feedback_length_le_fiber_length")
                    raise TrialPruned("invalid_config: feedback_length <= fiber_length")
                if "washout" in msg or "wash-out" in msg:
                    trial.set_user_attr("skip_reason", "washout_too_small")
                    print(cfg, "skip_reason: washout too small")
                    raise TrialPruned("skip_reason: washout too small")
                if "time_limit_exceeded" in msg:
                    trial.set_user_attr("skip_reason", "time_budget")
                    print(cfg, "skip_reason: time budget exceeded")
                    raise TrialPruned("time budget exceeded")

                if "iteration_not_converged" in msg:
                    trial.set_user_attr("skip_reason", "no_convergence")
                    print(cfg, "skip_reason: no convergence")
                    raise TrialPruned("no convergence")
                raise

            score = min(float(res["metrics"]["nrmse_val"]), 1e+1)
            if not np.isfinite(score):
                trial.set_user_attr("skip_reason", f"non-finite score {score}")
                print("skip_reason", "non-finite score")
                raise TrialPruned("skip_reason: non-finite score")

            # === МИНИ-ЛОГИРОВАНИЕ ДЛЯ DASHBOARD ===
            # метрики + лучшая альфа
            trial.set_user_attr("nrmse_train", float(res["metrics"]["nrmse_train"]))
            trial.set_user_attr("nrmse_val", float(res["metrics"]["nrmse_val"]))
            trial.set_user_attr("ridge_alpha", float(res["metrics"]["ridge_alpha"]))

            # физика/характерные длины/прочее из симулятора
            fiber_params = res.get("fiber_params", {}) or {}
            for k, v in _flatten("fiber", fiber_params):
                trial.set_user_attr(k, v)

            # полезные управляемые параметры (для быстрых фильтров в UI)
            trial.set_user_attr("cfg.mask.mask_size", int(cfg.mask.mask_size))
            trial.set_user_attr("cfg.mask.gain_in", float(cfg.mask.gain_in))
            trial.set_user_attr("cfg.reservoir.kappa", float(cfg.reservoir.kappa))
            trial.set_user_attr("cfg.reservoir.upsampling", int(cfg.reservoir.upsampling))
            trial.set_user_attr("cfg.reservoir.delay_factor_in_symbols",
                                int(cfg.reservoir.delay_factor_in_symbols))

            nonlocal best
            if score < best["score"]:
                best["score"] = score
                best["res"] = res
                best["params"] = cfg

            return score

    # --- имя исследования ---
    env_name = os.getenv("MCF_STUDY_NAME")
    if env_name:
        study_name = env_name
    else:
        variant_str = str(base_cfg.variant)
        core_count = int(base_cfg.core_count)
        ts = datetime.now().strftime('%Y%m%d-%H%M')
        study_name = f"mcf_rc_{variant_str}_C{core_count}_{ts}"

    sampler = optuna.samplers.TPESampler(
        # seed=seed,
        constant_liar=True,  # рекомендовано для distributed
        multivariate=True,   # если есть взаимозависимости гиперпараметров
        group=True           # если есть условные/групповые параметры
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=sampler
    )

    print(f"[optimize] Target COMPLETE trials = {n_trials}")

    # --- ДВА РЕЖИМА: распределённый (MCF_BASH) и локальный ---
    if os.getenv("MCF_BASH", ""):
        # РАСПРЕДЕЛЁННЫЙ: ручной цикл ask/tell с бюджетом по COMPLETE
        while True:
            complete_cnt = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
            if complete_cnt >= n_trials:
                break

            trial = study.ask()  # sampler задан в create_study
            try:
                value = objective(trial)
                study.tell(trial, value)  # COMPLETE
            except TrialPruned:
                study.tell(trial, state=TrialState.PRUNED)
            except (AssertionError, ValueError, RuntimeError) as _e:
                # Любая иная ошибка — явно FAIL (для наглядности в дашборде и исключаем из COMPLETE-бюджета)
                study.tell(trial, state=TrialState.FAIL)
                continue
            except Exception as _e:
                # На всякий случай всё прочее тоже считаем FAIL
                study.tell(trial, state=TrialState.FAIL)
                continue
    else:
        # --- локальный режим: стандартный Optuna с многопоточностью n_jobs ---
        print("[mode] Local study.optimize (no MCF_BASH)")
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
            catch=(AssertionError, ValueError, RuntimeError, Exception)
        )

    best_trial = study.best_trial

    return dict(
        best_cfg=best["res"]["cfg"],
        best_metrics=best["res"]["metrics"],
        best_trial_params=best["params"],
        best_val_nrmse=float(best["score"]),
        optuna_best_params=best_trial.params,
        optuna_best_value=float(best_trial.value),
        study_name=study_name,
    )


# =========================
# Пример точки входа
# =========================

if __name__ == "__main__":

    # Базовые параметры
    force_rerun = False  # игнорировать кэш и считать заново
    save_cache = True
    layer_count = 1
    core_configuration = CoreConfig.hexagonal
    core_count = get_core_count(core_configuration=core_configuration, ring_count=layer_count)

    # layer 0 - центральная сердцевина
    # layer 1 - первый круг из 6 сердцевин
    # layer 2 - второй "круг" из 12 сердцевин, расстояние от которых до центра разное
    # ...
    layer_radii_array = np.zeros(int(layer_count) + 1)
    for i in range(int(layer_count) + 1):
        if i == 0:
            layer_radii_array[i] = 0  # [mkm]
        if i == 1:
            layer_radii_array[i] = 30 # 17.3 * 1  # 17.3 # [mkm]
        if i == 2:
            layer_radii_array[i] = 17.3 * 2  # [mkm]
        if i == 3:
            layer_radii_array[i] = 17.3 * 3  # [mkm]

        # layer_radii_array[i] = 17.3 * (i * 1.5) # [mkm]

    temporal_mask_modulation_frequency_ghz = 40  # GHz

    variant = "temporal_unique_per_core"  # "spatial_only" "temporal_same_all_cores" "temporal_unique_per_core"

    if variant == "temporal_same_all_cores" or variant == "temporal_unique_per_core":
        temporal_mask_size = 121
    else:
        temporal_mask_size = 1

    mg_cfg = MGConfig(t_size=2 ** 10, tau=17, n=10, beta=0.2, gamma=0.1, initial_condition=1.2, dt=1.0)

    mask_cfg = MaskConfig(mask_size=temporal_mask_size, mask_kind="uniform", seed=42, gain_in=17.425874971099564)

    reservoir_cfg = ReservoirConfig(
        fiber_length_m=0.6571013875307805,  # 0.1
        time_step_ps=1.0 / temporal_mask_modulation_frequency_ghz * 1e+3,
        step_number_per_dimensionless_distance=20,
        upsampling=2,
        delay_factor_in_symbols=6,
        delay_additional_in_mask_steps=0,
        layer_count=layer_count,
        layer_radii_array=layer_radii_array,
        g0_array=tuple([0.15163556871712178] * core_count),  # 10
        psat_array=tuple([1.0273745093098033e-05] * core_count),  # 0.02
        kappa=0.455,  # 0.9
        use_gpu=False,
        use_torch=False,
        num_threads=8,
        display_debug_plots=True,
        display_debug_info=True,
        save_figs=True,
        max_hours_total=72,
        precision='float64',
        use_dispersion=True,
    )

    training_cfg = TrainingConfig(feature_mode="intensity", taps=1, ridge_alpha=1e-4,  # washout=100,
                                  target_shift=1,
                                  train_frac=0.8, val_frac=0.2)  # для одиночного запуска

    base_cfg = ExperimentConfig(core_count=core_count, mg=mg_cfg, mask=mask_cfg,
                                reservoir=reservoir_cfg, training=training_cfg,
                                variant=variant)

    t_size = estimate_required_t_size_fast(base_cfg)
    print("Estimated required symbol count =", t_size)
    mg_cfg.t_size = 5000 # np.min([int(np.ceil(t_size / 500.0)) * 500, 5000])

    # Пример: одиночный прогон с сохранением артефактов и pseudo free-run
    if variant == "spatial_only":
        res = run_spatial_only(base_cfg, n_trials_opt=0, free_run_horizon=0, force_rerun=force_rerun, save_cache=save_cache)
    elif variant == "temporal_same_all_cores":
        res = run_temporal_same_all_cores(base_cfg, n_trials_opt=0, free_run_horizon=0, force_rerun=force_rerun, save_cache=save_cache)
    elif variant == "temporal_unique_per_core":
        res = run_temporal_unique_per_core(base_cfg, n_trials_opt=0, free_run_horizon=0, force_rerun=force_rerun, save_cache=save_cache)

    print("Val/Test NRMSE:", res["metrics"]["nrmse_val"], res["metrics"]["nrmse_test"])

    # === Optuna: поиск по κ, g0, Psat, L_fiber и gain_in (лог по мощности) ===
    # Вариант «temporal_same_all_cores» для одной сердцевины;
    # окно задержки и размер маски тоже можно подстроить.
    # Для запуска optuna dashboard установи
    # pip install optuna-dashboard
    # и выполни в отдельной консоли после запуска расчета (путь укажи свой до папки проекта)
    # optuna-dashboard "C:/Users/Igor/YandexDisk/Code/Photonics/fiberprop/scripts/mcf_optuna.journal" --port 8080
    # Открой в браузере http://127.0.0.1:8080/dashboard/
    # На линуксе порт 18080, поэтому
    # открой в браузере http://127.0.0.1:18080/dashboard

    # base_cfg.training.train_frac = 0.8
    # base_cfg.training.val_frac = 0.2
    # res = run_temporal_same_all_cores(
    #     base_cfg,
    #     n_trials_opt=10000,  # сколько попробовать конфигураций
    #     free_run_horizon=0,  # без свободного прогона после обучения
    #     force_rerun=False,  # используй кэш, если ключ совпал
    #     save_cache=False,
    # )
    # print("Лучший результат Optuna:\n", res)
