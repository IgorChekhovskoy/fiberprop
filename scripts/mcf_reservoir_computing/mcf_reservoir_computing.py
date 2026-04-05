from __future__ import annotations
import os
from copy import deepcopy
from datetime import datetime
from typing import Optional, Union

from fiberprop.solver import ComputationalParameters, EquationParameters, Solver
from fiberprop.fiber import Fiber, FiberMaterial, CoreConfig
from fiberprop.fiber_geometry import get_core_count
from fiberprop.light import Light
from fiberprop.fiber_base_functions import get_coupling_coefficients_2d
from fiberprop.drawing import *
from fiberprop.parallel_runtime import physical_cpu_count, temporary_thread_limits

from time import time
import numpy as np  # ; np.show_config()
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from pathlib import Path
import json

from scripts.mcf_reservoir_computing.mcf_auxiliary_functions import calc_p_pump_per_core_from_slm, \
    calc_g0_psat_from_p_pump_josab, nrmse, sha256_of_json, CACHE_DIR, json_dumps_compact, \
    learning_curve_for_result_plotly, plot_laser_gain_curve_from_params, MM, COL1, COL15, COL2, apply_readout, \
    ridge_alpha_sweep_diagnostics, ridge_alpha_selection_compare_diagnostics

STYLE_PATH = Path(__file__).with_name("styles") / "mcf.mplstyle"
plt.style.use(str(STYLE_PATH))


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
        if mask_size == 1:
            masks[c] = 1
        else:
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


def generate_ase_noise(
        *,
        size: tuple[int, int],
        dt_ps: float,
        lambda0_um: float,
        g0_array: tuple[float, ...] | np.ndarray,
        fiber_length_m: float,
        nf_db: float = 5.0,
        optical_bw_hz: float = 12.5e9,
        n_polarizations: int = 1,
        seed: int | None = None,
        step_index: int = 0,
        dtype: np.dtype = np.complex128,
) -> np.ndarray:
    """
    Генерация комплексного ASE шума (additive complex AWGN) в огибающей поля (√W).

    Модель (в духе стандартных оценок для оптического усилителя):
        S_ASE = n_pol * n_sp * h * nu * (G - 1)    [W/Hz],
    где nu = c / lambda0.

    Для связи noise figure и параметра спонтанной эмиссии используем точную формулу
    для конечного усиления G:
        NF_lin = (2 * n_sp * (G - 1) + 1) / G,
    откуда
        n_sp = (NF_lin * G - 1) / (2 * (G - 1)).

    В дискретном времени задаём суммарную мощность шума в полосе:
        P_noise = S_ASE * B_eff,   B_eff = min(optical_bw_hz, 1/dt),
    и генерируем комплексный гауссов шум так, что E|n|^2 = P_noise.

    Для воспроизводимости используется SeedSequence([seed, step_index]): фиксированный seed ⇒ разные шумы на разных
    оборотах (step_index), но полностью повторяемые при повторном запуске.
    """
    # --- проверки ---
    dt_s = float(dt_ps) * 1e-12
    if dt_s <= 0.0:
        raise ValueError("dt_ps must be > 0")

    C, T = int(size[0]), int(size[1])
    if C <= 0 or T <= 0:
        return np.zeros((max(C, 0), max(T, 0)), dtype=dtype)

    if float(optical_bw_hz) <= 0.0:
        return np.zeros((C, T), dtype=dtype)

    # --- константы ---
    h = 6.62607015e-34  # J*s (точное значение в СИ)
    c = 299792458.0  # m/s

    # --- параметры усиления ---
    g0 = np.asarray(g0_array, dtype=float).ravel()
    if g0.size == 0:
        return np.zeros((C, T), dtype=dtype)

    if g0.size not in (1, C):
        raise ValueError(f"g0_array size mismatch: got {g0.size}, expected 1 or {C}")

    if g0.size == 1 and C > 1:
        g0 = np.full(C, float(g0[0]), dtype=float)

    # Оценка малого-сигнала: G = exp(g0 * L), отрицательный g0 не даёт ASE (обнуляем по (G-1)+).
    G = np.exp(np.maximum(g0, 0.0) * float(fiber_length_m))
    Gm1 = np.maximum(G - 1.0, 0.0)

    # --- n_sp и PSD ASE ---
    nf_lin = 10.0 ** (float(nf_db) / 10.0)

    n_sp = np.zeros_like(G, dtype=float)
    gain_mask = Gm1 > 0.0
    if np.any(gain_mask):
        n_sp[gain_mask] = (nf_lin * G[gain_mask] - 1.0) / (2.0 * Gm1[gain_mask])

        # Физический минимум n_sp = 1 (полная инверсия).
        n_sp[gain_mask] = np.maximum(n_sp[gain_mask], 1.0)

    nu_hz = c / (float(lambda0_um) * 1e-6)
    s_ase_w_per_hz = float(n_polarizations) * n_sp * h * nu_hz * Gm1  # (C,)

    # --- мощность в полосе, которую реально может представить дискретизация ---
    b_sim_hz = 1.0 / dt_s
    b_eff_hz = min(float(optical_bw_hz), b_sim_hz)
    p_noise_w = s_ase_w_per_hz * b_eff_hz  # (C,)

    # --- воспроизводимый RNG: seed + индекс оборота ---
    ss = np.random.SeedSequence([int(seed) if seed is not None else 0, int(step_index)])
    rng = np.random.default_rng(ss)

    re = rng.standard_normal((C, T))
    im = rng.standard_normal((C, T))
    noise = (re + 1j * im) * np.sqrt(0.5)  # E|noise|^2 = 1
    noise = noise * np.sqrt(p_noise_w).reshape(C, 1)
    return np.asarray(noise, dtype=dtype)


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
        delta_phase=0.0,
        use_gpu=False,
        use_torch=False,
        num_threads: int | str | None = "default",
        display_debug_info=False,
        display_debug_plots=False,
        save_figs=False,
        save_gif=False,
        max_hours_total: Optional[float] = None,
        precision: Optional[str] = 'float64',
        use_dispersion: bool = True,
        ase_noise_seed: int | None = None,
        ase_noise_nf_db: float = 5.0,
        ase_noise_optical_bw_hz: float = 12.5e9,
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
            Коэффициенты малого-сигнала g₀ [1/m] для *C* сердцевин.
            Нулевой массив → усиление отключено.
        psat_array : array-like, default ()
            Мощность насыщения P_sat, [W], для *C* сердцевин
            (используется как E_sat = 2 T P_sat).
        kappa : float, default 0.9
            Модуль коэффициента обратной связи
        delta_phase : float, default 0
            Фаза в коэффициенте обратной связи
        use_gpu : bool, default False
            True → основное ядро SSFM выполняется на GPU (PyTorch-CUDA),
            иначе – NumPy/CPU.
        display_debug_info : bool, default False
            Печатает расчётные коэффициенты, характерные длины,
            задержки и прочие служебные данные.
        display_debug_plots : bool, default False
            Визуализация хода интегрирования и итоговых спектров
            средствами plotly (2D/3D).
        save_figs  bool, default False
            Сохранять графики
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

    data_in_mod = data_in

    core_configuration = CoreConfig.hexagonal
    light = Light(lambda0=1.55)  # µm

    # ─── волокно и линейка ──────────────────────────────────────
    fiber = Fiber(core_configuration=core_configuration,
                  ring_count=layer_count,
                  core_radius=2.95,
                  cladding_diameter=125.0 * 2,
                  n2=3.2,
                  distance_to_fiber_center=layer_radii_array,
                  NA=0.125,
                  core_material=FiberMaterial.SIO2_AND_GEO2_ALLOY,
                  material_concentration=0.038)
    fiber.set_refractive_indexes_by_lambda(light.lambda0)

    central_core_ind = int(np.floor(eq_size / 2)) if eq_size > 1 else 0

    save_scheme_path = None
    if save_figs:
        save_scheme_path = str(Path(__file__).parent / "fig_mcf_scheme")

    import hashlib
    d = os.path.join(os.path.dirname(__file__), ".ccache")
    os.makedirs(d, exist_ok=True)
    h = hashlib.sha1(repr((getattr(fiber, "core_configuration", None), getattr(fiber, "ring_count", None),
                           getattr(fiber, "core_radius", None), getattr(fiber, "cladding_diameter", None),
                           getattr(fiber, "n2", None), tuple(np.asarray(getattr(fiber, "distance_to_fiber_center", ()))
                                                             .astype(float).round(9).tolist()),
                           getattr(fiber, "NA", None), getattr(fiber, "core_material", None),
                           getattr(fiber, "material_concentration", None),
                           getattr(light, "lambda0", None))).encode()).hexdigest()
    p = os.path.join(d, f"coupling_{h}.npy")
    if os.path.exists(p):
        coupling_matrix = np.load(p)
    else:
        coupling_matrix, _ = get_coupling_coefficients_2d(
            fiber, light, eps=2e-6, display_debug_plots=display_debug_plots, save_debug_plot_path=save_scheme_path
        )
        np.save(p, np.asarray(coupling_matrix))

    # coupling_matrix = np.zeros((core_count, core_count))

    coupling_coefficient = coupling_matrix[central_core_ind - 1][central_core_ind] if eq_size > 1 else 1e-10
    max_val = np.max(np.abs(coupling_matrix))
    threshold = max_val * 1e-3
    coupling_matrix = np.where(np.abs(coupling_matrix) > threshold, coupling_matrix, 0)

    gamma = fiber.get_gamma(light, eps=1e-3)  # [1/(W·m)]
    beta1 = fiber.get_beta1(light)  # [ps/m]
    beta2 = fiber.get_beta2(light)  # [ps²/m]

    if display_debug_info:
        print("coupling_coefficient =", coupling_coefficient)
        print("gamma =", gamma, "1/(W·m)")
        print("beta1 =", beta1, "ps/m")
        print("beta2 =", beta2, "ps²/m")

    # ─── буфер задержки ──────────────────────────────────────────
    fiber_propagation_time = fiber_length_m * beta1  # [ps]
    beta1_air = 1 / light.c_light * 1e+12
    feedback_loop_propagation_time = window_size * time_step_ps - fiber_propagation_time
    feedback_length_m = feedback_loop_propagation_time / beta1_air  # длина воздушного плеча, m

    tau_ps = beta1_air * feedback_length_m  # [ps]
    omega0_rad_per_ps = 2 * np.pi * light.c_light * 1e-12 / (light.lambda0 * 1e-6)
    phase_air = omega0_rad_per_ps * tau_ps
    phase_fiber = 0  #
    feedback_coeff = kappa * np.exp(1j * delta_phase)

    if feedback_length_m < fiber_length_m:
        # print("feedback_length_m < fiber_length_m", feedback_length_m, fiber_length_m)
        raise RuntimeError("feedback_length_m < fiber_length_m")

    # ─── характерные длины ───────────────────────────────────────
    L_D, L_NL, L_coupling, L_gain = compute_characteristic_lengths(
        beta2_ps2_m=beta2,
        gamma_1_w_m=gamma,
        coupling_coefficient=coupling_coefficient,
        data_in=data_in_mod,
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
    window_duration_ps = window_size * time_step_ps
    psat_array_w = np.asarray(psat_array)
    esat_array = psat_array_w * window_duration_ps  # [pJ]
    mean_power_per_core_w = np.mean(np.abs(data_in_mod) ** 2, axis=1).astype(float, copy=False)
    energy_per_window = mean_power_per_core_w * window_duration_ps
    ase_signal_power_sum = 0.0
    ase_noise_power_sum = 0.0

    if display_debug_info:
        print("data_in.shape=", data_in_mod.shape)
        print("data_in size =", data_in_mod.shape[1] * time_step_ps, "ps")
        print("fiber_propagation_time =", fiber_propagation_time, "ps")
        print(f'feedback_loop_propagation_time={feedback_loop_propagation_time:.1f} ps')
        print("fiber_length_dimensionless =", fiber_length_dimensionless)
        print("length_scale =", length_scale)
        print("n_z =", n_z)
        print("g0 =", g0_array)
        print("esat = ", esat_array, "pJ")
        print("\nwindow_size * time_step_ps =", window_duration_ps)
        if len(psat_array):
            print("P_sat [W] =", psat_array_w)
            print("E_sat [pJ] =", esat_array)
            print("mean_power_per_core [W] =", mean_power_per_core_w)
            print("Energy_per_window [pJ] =", energy_per_window)

    # --- режим без дисперсии (beta2=0) с оконным стримингом ---
    if not use_dispersion:

        T_half_stream = window_duration_ps / 2.0

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

        # начальные данные окна: уже с занулённой 0-й сердцевиной при необходимости
        ic0 = data_in_mod[:, 0:window_size]

        solver = Solver(comp, eq,
                        initial_condition=ic0,
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
                seg = data_in_mod[:, s:e]
                if seg.shape[1] < window_size:
                    tmp = np.zeros((eq_size, window_size), dtype=dtype)
                    tmp[:, :seg.shape[1]] = seg
                    seg = tmp
                solver.numerical_solution[0] = feedback_coeff * fb_buf + seg

            if ase_noise_seed is not None:
                sig_power = float(np.mean(np.abs(solver.numerical_solution[0]) ** 2))
                noise = generate_ase_noise(
                    size=solver.numerical_solution[0].shape,
                    dt_ps=time_step_ps,
                    lambda0_um=light.lambda0,
                    g0_array=g0_array,
                    fiber_length_m=fiber_length_m,
                    nf_db=ase_noise_nf_db,
                    optical_bw_hz=ase_noise_optical_bw_hz,
                    seed=ase_noise_seed,
                    step_index=batch_index,
                    dtype=dtype,
                )

                ase_signal_power_sum += sig_power
                ase_noise_power_sum += float(np.mean(np.abs(noise) ** 2))
                solver.numerical_solution[0] = solver.numerical_solution[0] + noise

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
            data_out_stream[:, s:e] = fb_buf[:, :e - s] * np.exp(
                1j * phase_fiber)  # добавляем набег фазы самого поля E = A * F * exp(i*beta*z)

            if display_debug_plots:
                number_of_points_for_display = solver.com.M  # np.min([5000, solver.com.M])
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

        ase_osnr_db = None
        if ase_noise_seed is not None and ase_noise_power_sum > 0.0:
            ase_osnr_db = 10.0 * np.log10(ase_signal_power_sum / ase_noise_power_sum)

        params = {
            "gamma": float(gamma),
            "beta1": float(beta1),
            "beta2": 0.0,
            "beta1_air": float(beta1_air),
            "coupling_coefficient": float(coupling_coefficient),
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
            # характеристики сигнала
            "E_sat": esat_array.tolist(),
            "mean_power_per_core:": mean_power_per_core_w.tolist(),
            "Energy_per_window": energy_per_window.tolist(),
            "ase_noise": {
                "seed": int(ase_noise_seed) if ase_noise_seed is not None else None,
                "nf_db": float(ase_noise_nf_db) if ase_noise_seed is not None else None,
                "optical_bw_hz": float(ase_noise_optical_bw_hz) if ase_noise_seed is not None else None,
                "osnr_db": float(ase_osnr_db) if ase_osnr_db is not None else None,
            },
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
        initial_data = np.zeros((data_in_mod.shape[0], (M + offset_size0) * 2), dtype=dtype)
        initial_data[:, offset_size0:M + offset_size0] = data_in_mod

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
            number_of_points_for_display = 10000  # solver.com.M  # np.min([5000, solver.com.M])
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
        _prev_max_nrmse = 1.0
        _last_max_nrmse = 1.0

        t_stream_ps = np.arange(M_final) * time_step_ps / upsampling  # [ps]
        omega_stream = 2 * np.pi * np.fft.fftfreq(M_final, d=time_step_ps / upsampling)  # [rad/ps]

        solver.collapse_if_possible()
        if solver.is_collapsed:
            initial_data = deepcopy(solver.numerical_solution[0])

        t1 = time()

        for iteration_index in range(iteration_count + 1):  # одна доп. итерация для окончательного установления
            # if display_debug_info:
            print("\nC =", coupling_coefficient, ", L_coupling =", L_coupling, ", layer_radii_array =",
                  layer_radii_array[-1], ", iteration", iteration_index, "of", iteration_count)

            if ase_noise_seed is not None:
                sig_power = float(np.mean(np.abs(solver.numerical_solution[0]) ** 2))
                noise = generate_ase_noise(
                    size=solver.numerical_solution[0].shape,
                    dt_ps=time_step_ps / upsampling,
                    lambda0_um=light.lambda0,
                    g0_array=g0_array,
                    fiber_length_m=fiber_length_m,
                    nf_db=ase_noise_nf_db,
                    optical_bw_hz=ase_noise_optical_bw_hz,
                    seed=ase_noise_seed,
                    step_index=iteration_index,
                    dtype=dtype,
                )

                ase_signal_power_sum += sig_power
                ase_noise_power_sum += float(np.mean(np.abs(noise) ** 2))
                solver.numerical_solution[0] = solver.numerical_solution[0] + noise

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
                    _prev_max_nrmse = _last_max_nrmse
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
            #         x_axis_label='t [ns]', y_axis_label='power [W]', yscale='log',
            #     )

            solver.numerical_solution[0] = initial_data + np.roll(solver.numerical_solution[-1],
                                                                  window_size * upsampling, axis=1) * feedback_coeff

            # Если две итерации подряд финальное поле не меняется по сравнению с предыдущим, выходим
            if _prev_max_nrmse < 1e-15 and _last_max_nrmse < 1e-15:
                # if display_debug_info:
                print("Iterations stopped after", iteration_index, "of ", iteration_count, "iterations.")
                break

        # добавляем набег фазы самого поля E = A * F * exp(i*beta*z)
        solver.numerical_solution[-1] *= np.exp(1j * phase_fiber)

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

        solver.restore_full_system()

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
                       offset_size:offset_size + data_in_mod.shape[1] * upsampling]) ** 2,
                names=[f"$|U_{central_core_ind}(z=0,t)|^2$", f"$|U_{central_core_ind}(z=L,t)|^2$"],
                x_axis_label='t [ns]', y_axis_label='power [W]'
            )

        ase_osnr_db = None
        if ase_noise_seed is not None and ase_noise_power_sum > 0.0:
            ase_osnr_db = 10.0 * np.log10(ase_signal_power_sum / ase_noise_power_sum)

        return {
            "data_out": (solver.numerical_solution[-1][:,
                         offset_size:offset_size + data_in_mod.shape[1] * upsampling:upsampling]),
            "params": {
                "gamma": float(gamma),
                "beta1": float(beta1),
                "beta2": float(beta2),
                "beta1_air": float(beta1_air),
                "coupling_coefficient": float(coupling_coefficient),
                "feedback_length_m": float(feedback_length_m),
                "feedback_loop_propagation_time_ps": float(feedback_loop_propagation_time),
                "fiber_propagation_time_ps": float(fiber_propagation_time),
                "layer_radii_array": layer_radii_array,
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
                # характеристики сигнала
                "E_sat": esat_array.tolist(),
                "mean_power_per_core:": mean_power_per_core_w.tolist(),
                "Energy_per_window": energy_per_window.tolist(),
                "ase_noise": {
                    "seed": int(ase_noise_seed) if ase_noise_seed is not None else None,
                    "nf_db": float(ase_noise_nf_db) if ase_noise_seed is not None else None,
                    "optical_bw_hz": float(ase_noise_optical_bw_hz) if ase_noise_seed is not None else None,
                    "osnr_db": float(ase_osnr_db) if ase_osnr_db is not None else None,
                },
            },
        }


######################################################################

import dataclasses as _dc
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Literal, Optional

# =========================
# Утилиты, метрики, кэш
# =========================


def train_ridge(
        X: np.ndarray,
        y: np.ndarray,
        alpha: float | str | None,
        add_bias: bool = True,
        *,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
):
    if add_bias:
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    else:
        Xb = X

    # Убедимся, что используем float64
    Xb = Xb.astype(np.float64)
    y = y.astype(np.float64)

    # Решаем ridge устойчиво через SVD, без формирования X^T X (normal equations).
    # Xb = U diag(s) V^T
    # W = V diag(s / (s^2 + alpha)) U^T y
    U, s, Vt = np.linalg.svd(Xb, full_matrices=False)

    # Диагностика обусловленности
    # cond(Xb) = s_max / s_min, cond_ridge ~ (s_max^2 + alpha) / (s_min^2 + alpha)
    s_max = float(s[0]) if s.size > 0 else 0.0
    s_min = float(s[-1]) if s.size > 0 else 0.0

    if s_min == 0.0:
        cond_x = np.inf
    else:
        cond_x = s_max / s_min

    s_max2 = s_max * s_max
    s_min2 = s_min * s_min

    # -------------------------------------------------------------------------
    # Auto alpha selection: val_max_plateau_0.2 (block-wise worst-case) + cond cap.
    #
    # Summary:
    #   - Split val into sequential blocks (time order preserved).
    #   - For each alpha: compute block-wise MSE, take max over blocks, then take sqrt and normalize:
    #         e_max(alpha) = sqrt( max_b MSE_b(alpha) / Var_global(val) )
    #     (i.e., worst-case NRMSE with *global* sigma, but aggregated via MSE then sqrt).
    #   - Select the largest alpha satisfying:
    #         e_max(alpha) <= (1 + max_rtol) * min_alpha e_max(alpha)
    #     ("right edge of plateau" for stability).
    #   - Enforce numerical stability by requiring cond_ridge(alpha) <= max_cond when possible,
    #     and finally alpha = max(alpha, alpha_needed_for_max_cond).
    #
    # References (high-level motivation):
    #   - Blocked CV for dependent (temporal/spatial) data:
    #       Roberts et al., Ecography 2017: https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881
    #       Bergmeir, Hyndman, Koo 2018 (time-series CV): https://robjhyndman.com/papers/cv-wp.pdf
    #   - Train-only ridge parameter selection (fallback without val): GCV
    #       Golub, Heath, Wahba 1979: https://pages.stat.wisc.edu/~wahba/ftp1/oldie/golub.heath.wahba.pdf
    #   - Worst-case / stability-aware hyperparameter selection (general context):
    #       Pfohl et al. 2022 (worst-case model selection): https://pmc.ncbi.nlm.nih.gov/articles/PMC8885701/
    #       Cory-Wright & Gómez 2025 (stability-regularized CV): https://optimization-online.org/wp-content/uploads/2025/05/Stability_Adjusted_Cross_Validation_for_NeurIPS_2025.pdf
    # -------------------------------------------------------------------------

    max_cond = 1e5
    val_blocks = 8
    max_rtol = 0.2
    n_alpha = 61
    auto_min_decades = 12.0
    auto_lo_factor_vs_smax2 = 1e-12
    auto_hi_factor_vs_smax2 = 1e3

    alpha_is_auto = False
    if alpha is None:
        alpha_is_auto = True
    elif isinstance(alpha, str):
        alpha_is_auto = (alpha.strip().lower() == "auto")
    else:
        try:
            a0 = float(alpha)
            alpha_is_auto = (not np.isfinite(a0)) or (a0 <= 0.0)
        except Exception:
            alpha_is_auto = True

    s = np.asarray(s, dtype=np.float64)
    s2 = s * s
    Ut_y_sel = U.T @ y  # (M, T)

    def _cond_ridge(a: float) -> float:
        a = float(a)
        if not np.isfinite(a) or a < 0.0:
            a = 0.0
        denom = s_min2 + a
        if denom <= 0.0:
            denom = np.finfo(np.float64).tiny
        return float((s_max2 + a) / denom)

    def _alpha_needed_for_max_cond(target_cond: float) -> float:
        """
        Minimal alpha so that cond_ridge(alpha) <= target_cond.
        Solve (s_max^2 + a)/(s_min^2 + a) <= k (monotone in a).
        """
        k = float(target_cond)
        if not np.isfinite(k) or k <= 1.0:
            return 0.0
        if s_max2 <= 0.0:
            return 0.0

        if s_min2 <= 0.0:
            # (s_max^2 + a)/a <= k  ->  a >= s_max^2/(k-1)
            return float(s_max2 / (k - 1.0))

        num = s_max2 - k * s_min2
        if num <= 0.0:
            return 0.0
        return float(num / (k - 1.0))

    alpha_cond = _alpha_needed_for_max_cond(max_cond)

    def _compute_W_for_alpha(a: float) -> np.ndarray:
        a = float(a)
        denom = s2 + a
        denom[denom == 0.0] = np.finfo(np.float64).tiny
        filt = (s / denom)[:, None]  # (M, 1)
        return Vt.T @ (filt * Ut_y_sel)  # (F(+1), T)

    def _alpha_grid_auto() -> np.ndarray:
        if s_max2 > 0.0:
            lo = max(float(s_max2) * float(auto_lo_factor_vs_smax2), np.finfo(np.float64).tiny)
            hi = float(s_max2) * float(auto_hi_factor_vs_smax2)
            if hi <= lo:
                hi = lo * 1e6
            decades = float(np.log10(hi) - np.log10(lo))
            if decades < float(auto_min_decades):
                hi = lo * (10.0 ** float(auto_min_decades))
        else:
            lo, hi = 1e-12, 1e3

        if np.isfinite(alpha_cond) and alpha_cond > 0.0:
            lo = min(lo, float(alpha_cond))
            hi = max(hi, float(alpha_cond))

        grid = np.logspace(np.log10(lo), np.log10(hi), int(max(2, n_alpha))).astype(np.float64)
        return np.unique(grid)

    def _block_slices(n: int, k: int) -> list[slice]:
        n = int(n)
        k = int(k)
        if n <= 0 or k <= 1:
            return [slice(0, n)]
        k = max(2, min(k, n))
        edges = np.linspace(0, n, num=k + 1, dtype=int)
        out = []
        for i in range(k):
            a, b = int(edges[i]), int(edges[i + 1])
            if b - a >= 2:
                out.append(slice(a, b))
        return out if out else [slice(0, n)]

    def _gcv_score(a: float) -> float:
        """
        Train-only fallback: GCV(alpha) ~ MSE(alpha) / (1 - df(alpha)/n)^2,
        df(alpha) = sum s^2/(s^2+alpha).
        """
        W_loc = _compute_W_for_alpha(a)
        y_hat = apply_readout(X, W_loc, add_bias=add_bias)
        r = (y_hat - y).astype(np.float64, copy=False)
        mse = float(np.mean(r * r))

        n = float(max(1, X.shape[0]))
        df = float(np.sum(s2 / (s2 + float(a))))
        denom = 1.0 - df / n
        denom = float(np.clip(denom, 1e-12, 1.0))
        return mse / (denom * denom)

    def _val_emax_from_blocks_mse(a: float, blocks: list[slice], var_global: float) -> float:
        W_loc = _compute_W_for_alpha(a)
        y_hat = apply_readout(X_val, W_loc, add_bias=add_bias)

        y_true = np.asarray(y_val, dtype=np.float64)
        y_pred = np.asarray(y_hat, dtype=np.float64)

        n = min(int(y_true.shape[0]), int(y_pred.shape[0]))
        if n < 2 or not np.isfinite(var_global) or var_global <= 0.0:
            return float("nan")

        y_true = y_true[:n]
        y_pred = y_pred[:n]

        max_mse = -np.inf
        for sl in blocks:
            a0, b0 = int(sl.start or 0), int(sl.stop or 0)
            a0 = max(0, min(a0, n))
            b0 = max(0, min(b0, n))
            if b0 - a0 < 2:
                continue

            d = y_true[a0:b0] - y_pred[a0:b0]
            mse_b = float(np.mean(d * d))
            max_mse = max(max_mse, mse_b)

        if not np.isfinite(max_mse) or max_mse < 0.0:
            return float("nan")

        # requested aggregation: max over block MSE, then sqrt
        return float(np.sqrt(max_mse / var_global))

    # ---- choose alpha ----
    if alpha_is_auto:
        use_val = (
            X_val is not None and y_val is not None and
            isinstance(X_val, np.ndarray) and isinstance(y_val, np.ndarray) and
            X_val.ndim == 2 and y_val.ndim == 2 and
            X_val.shape[0] == y_val.shape[0] and X_val.shape[0] >= 8
        )

        grid = _alpha_grid_auto()

        if use_val:
            blocks = _block_slices(int(X_val.shape[0]), int(val_blocks))

            yv = np.asarray(y_val, dtype=np.float64).reshape(-1)
            if yv.size < 2:
                var_global = float("nan")
            else:
                var_global = float(np.var(yv))

            emax = np.array([_val_emax_from_blocks_mse(a, blocks, var_global) for a in grid], dtype=np.float64)
            m = np.isfinite(emax)

            if not np.any(m):
                alpha = float(alpha_cond) if np.isfinite(alpha_cond) and alpha_cond > 0.0 else float(grid[0])
            else:
                e_min = float(np.min(emax[m]))
                thr = (1.0 + float(max_rtol)) * e_min
                ok = m & (emax <= thr)

                if not np.any(ok):
                    alpha = float(grid[int(np.nanargmin(emax))])
                else:
                    # prefer alphas that already satisfy the cond cap (alpha >= alpha_cond), if any in plateau
                    if np.isfinite(alpha_cond) and alpha_cond > 0.0:
                        ok2 = ok & (grid >= float(alpha_cond))
                        alpha = float(np.max(grid[ok2])) if np.any(ok2) else float(np.max(grid[ok]))
                    else:
                        alpha = float(np.max(grid[ok]))
        else:
            # fallback: train-only GCV (no val supplied)
            gcv = np.array([_gcv_score(a) for a in grid], dtype=np.float64)
            m = np.isfinite(gcv)
            if np.any(m):
                alpha = float(grid[int(np.nanargmin(gcv))])
            else:
                alpha = float(alpha_cond) if np.isfinite(alpha_cond) and alpha_cond > 0.0 else float(grid[0])
    else:
        alpha = float(alpha)
        if not np.isfinite(alpha) or alpha < 0.0:
            alpha = 0.0

    # final stability clamp by cond cap
    if np.isfinite(alpha_cond) and alpha_cond > 0.0:
        alpha = float(max(alpha, alpha_cond))

    cond_ridge = _cond_ridge(alpha)
    # print(f"[train_ridge] alpha={alpha:.6g}  cond_ridge={cond_ridge:.6g}")

    # -------------------------------------------------------------------------
    # Конец блока подбора alpha
    # -------------------------------------------------------------------------

    # Считаем W (НЕ ТРОГАЕМ)
    Ut_y = U.T @ y  # (M, T)
    filt = (s / (s * s + alpha))[:, None]  # (M, 1)
    W = Vt.T @ (filt * Ut_y)  # (F(+1), T)

    return W, alpha


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
    g0_array: Tuple[float, ...] = (10.0,)  # [1/m]
    psat_array: Tuple[float, ...] = (0.02,)  # [W]
    kappa: float = 0.9
    delta_phase: float = 0
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
    precision: Optional[str] = 'float64'  # Размер данных в вычислениях ('float64', 'float32')
    use_dispersion: Optional[bool] = True  # Включать ли слагаемое с дисперсией в модель
    ase_noise_seed: Optional[int] = None  # Включение ASE шума в модель, если установлен seed для генерации шума
    ase_noise_nf_db: Optional[float] = None  # Noise figure (dB) для ASE шума
    ase_noise_optical_bw_hz: Optional[float] = None
    ase_noise_compute_effective_osnr: Optional[bool] = False  # честный effective output OSNR по паре clean/noisy прогонов

@dataclass
class TrainingConfig:
    """
    Настройки обучения рид-аута.

      feature_mode : {"intensity","realimag"}, default="intensity"
          Признаки из выхода резервуара:
            - "intensity" → |U_c(t)|^2 для всех ядер c
            - "realimag"  → concat([Re U_c(t), Im U_c(t)])
      washout : int | None
          Сколько первых состояний резервуара отбросить при вычислении ошибки на обучающей выборке. None → авто по κ и окну задержки.
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
    ridge_alpha: float = None
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

    Для spatial_only в идеальном (lossless) случае нормируем weights так, чтобы sum(weights**2) = 1.
    Тогда суммарная мощность входа не зависит от core_count, а gain_in задаёт общий масштаб поля.
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
        # В spatial_only подаём постоянные веса на ядра (коэффициенты по полю).
        # Идеальный SLM: сохраняем суммарную мощность -> нормировка по L2.
        weights = rng.uniform(0.0, 1.0, size=core_count)
        w_norm = float(np.linalg.norm(weights))
        if w_norm > 0.0:
            weights = weights / w_norm
        else:
            weights = np.zeros(core_count, dtype=float)
            weights[0] = 1.0
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


def get_washout_samples(cfg: ExperimentConfig) -> int:
    """
    ЕДИНЫЙ механизм получения washout в ОТСЧЁТАХ (samples).

    Важно:
      • Если cfg.training.washout задан (int > 0), он трактуется КАК ЧИСЛО СИМВОЛОВ.
        Тогда переводим в отсчёты: washout_samples = washout_symbols * M_eff,
        где M_eff = mask_size для temporal_* и 1 для spatial_only.
      • Если cfg.training.washout == None или <= 0 — автооценка по петле задержки и κ
        (несколько витков delay до затухания транзиента).

    Требование: cfg.reservoir.window_size должен быть уже проставлен (см. run_single_experiment).
    """
    # если явно задано – это ЧИСЛО СИМВОЛОВ
    w_syms = getattr(cfg.training, "washout", None)
    if w_syms is not None and float(w_syms) > 0:
        M_eff = cfg.mask.mask_size if str(cfg.variant).startswith("temporal_") else 1
        return int(w_syms) * int(M_eff)

    # авто: минимум один виток задержки, максимум три (обычно достаточно)
    return int(auto_washout_samples(cfg.reservoir, eps=1e-3, min_loops=1, max_loops=3))


def debug_plot_input_overview(cfg, mg_series_used: np.ndarray):
    """
    Рисует:
      1) Полный ряд MG (нормированный так же, как используется в расчёте),
         с пометками: warmup, shift, washout, train/val/test.
      2) Маски (по времени) для каждого ядра ИЛИ пространственные веса (bar),
         в зависимости от варианта.
    """
    # --- восстановим полный MG, чтобы показать warmup слева
    warmup = getattr(cfg.mg, "warmup", 0)
    x_full = mackey_glass(cfg.mg.t_size + warmup,
                          tau=cfg.mg.tau, n=cfg.mg.n, beta=cfg.mg.beta, gamma=cfg.mg.gamma,
                          initial_condition=cfg.mg.initial_condition, dt=cfg.mg.dt)

    # Нормируем ровно как в коде: по куску после warmup
    x_used = x_full[warmup:].astype(float)
    mu, sigma = float(np.mean(x_used)), float(np.std(x_used) + 1e-12)
    x_full_norm = (x_full - mu) / sigma

    # Проверим согласованность длины
    S = int(cfg.mg.t_size)
    assert mg_series_used.shape[0] == S, "mg_series_used длиной не равно t_size"

    # Проверка: mg_series_used должен совпадать с куском полного ряда после warmup (это те же данные).
    x_used_norm = x_full_norm[int(warmup):int(warmup) + S]
    diff = float(np.max(np.abs(x_used_norm.reshape(-1) - mg_series_used.reshape(-1)))) if S > 0 else 0.0
    if diff > 1e-6:
        print(f"[WARN] debug_plot_input_overview: mg_series_used != x_full_norm[warmup:] (max|diff|={diff:.3e})")

    # --- индексы сегментов на оси полного ряда (в символах)
    taps = int(getattr(cfg.training, "taps", 1) or 1)
    taps_drop = max(0, int(taps) - 1)  # столько символов теряем из-за tapped-state (taps-1)
    shift_syms = int(getattr(cfg.training, "target_shift", 0) or 0)

    # ЕДИНО: washout → в отсчётах → в символы
    M_eff = cfg.mask.mask_size if str(cfg.variant).startswith("temporal_") else 1
    w_samples = get_washout_samples(cfg)
    w_syms = int(np.ceil(int(w_samples) / max(1, int(M_eff))))

    # Эффективный хвост под train/val/test (точно как в run_single_experiment):
    #   y_aligned = y[(taps-1 + shift):], затем washout в символах
    align_drop = taps_drop + shift_syms
    N_eff = max(0, int(S) - int(align_drop) - int(w_syms))
    train_frac = float(getattr(cfg.training, "train_frac", 0.6))
    val_frac = float(getattr(cfg.training, "val_frac", 0.2))
    sl_train, sl_val, sl_test = split_train_val_test(N_eff, train_frac, val_frac)
    n_train = int(sl_train.stop - sl_train.start)
    n_val = int(sl_val.stop - sl_val.start)
    n_test = int(sl_test.stop - sl_test.start)

    # Границы по полной оси (0..S+warmup) в символах
    i_warmup_L = 0
    i_warmup_R = int(warmup)

    # tapped-state: первые (taps-1) символов после warmup не участвуют в обучении (из-за формирования лагов)
    i_taps_L = i_warmup_R
    i_taps_R = i_taps_L + int(taps_drop)

    # target shift: сдвиг таргета относительно состояния
    i_shift_L = i_taps_R
    i_shift_R = i_shift_L + int(shift_syms)

    i_wash_L = i_shift_R
    i_wash_R = i_wash_L + int(w_syms)

    i_tr_L = i_wash_R
    i_tr_R = i_tr_L + int(n_train)
    i_va_L = i_tr_R
    i_va_R = i_va_L + int(n_val)
    i_te_L = i_va_R
    i_te_R = i_te_L + int(n_test)

    # --- рисуем
    fig = plt.figure(figsize=(COL2, COL2 * 0.62))  # соотношение ~3:2
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.6], hspace=0.35)

    # (1) полный MG с разметкой
    ax1 = fig.add_subplot(gs[0, 0])
    t_full = np.arange(x_full_norm.shape[0])
    ax1.plot(t_full, x_full_norm, label="Mackey-Glass series")  # ← подпись линии
    ax1.set_xlim(t_full[0], t_full[-1])
    ax1.margins(x=0.0)

    # Явная граница warmup (на случай, если полоса слишком узкая и визуально теряется)
    if int(warmup) > 0:
        ax1.axvline(i_warmup_R, color="k", lw=1, alpha=0.25)

    def span_if(a, b, color, label, alpha=0.18):
        # рисуем даже для интервала шириной 1 символ (раньше было ≥2)
        a, b = int(a), int(b)
        if b - a < 1:
            return False
        a_plot = max(a, 0)
        b_plot = min(b, int(t_full[-1]) + 1)
        if b_plot - a_plot < 1:
            return False
        ax1.axvspan(a_plot, b_plot, color=color, alpha=alpha, label=label)
        return True

    shown = []
    if span_if(i_warmup_L, i_warmup_R, "#888888", "warmup"):
        shown.append("warmup")
    if taps_drop > 0 and span_if(i_taps_L, i_taps_R, "#17becf", f"taps-1={taps_drop}"):
        shown.append("taps")
    if span_if(i_shift_L, i_shift_R, "#1f77b4", "target shift"):
        shown.append("target shift")
    if w_syms > 0 and span_if(i_wash_L, i_wash_R, "#ff7f0e", "washout"):
        shown.append("washout")
    if span_if(i_tr_L, i_tr_R, "#2ca02c", "train"):
        shown.append("train")
    if span_if(i_va_L, i_va_R, "#9467bd", "val"):
        shown.append("val")
    if span_if(i_te_L, i_te_R, "#d62728", "test"):
        shown.append("test")

    title_suffix = ("/".join(shown)) if shown else ""
    ax1.set_title(f"Mackey-Glass series{': ' + title_suffix if title_suffix else ''}", loc="left")
    ax1.set_xlabel("symbol index")
    ax1.set_ylabel("normalized amplitude")

    # легенда (линия + зоны)
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        leg = ax1.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98),
                         frameon=True)
        leg.get_frame().set_facecolor((1, 1, 1, 0.6))
        leg.get_frame().set_edgecolor((0, 0, 0, 0.3))

    # (2) маски / пространственные веса
    masks_info = _reconstruct_masks_or_weights(core_count=cfg.core_count,
                                               variant=cfg.variant,
                                               mask_size=cfg.mask.mask_size,
                                               mask_kind=cfg.mask.mask_kind,
                                               seed=cfg.mask.seed)

    if masks_info["type"] == "temporal":
        ax2 = fig.add_subplot(gs[1, 0])
        im = _plot_temporal_masks(ax2, masks_info["masks"], cfg.mask.mask_kind)
        cbar = fig.colorbar(im, ax=ax2, fraction=0.46 / 10, pad=0.04)
        cbar.set_label("mask value")
    else:
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
    X = x1d[off:]  # x(t)
    Y = x1d[tau_samples:-tau_samples]  # x(t-τ)
    Z = x1d[:-off]  # x(t-2τ)
    L = X.shape[0]

    # границы сегментов в ИНДЕКСАХ mg_series_used (0..S)
    shift_syms = int(getattr(cfg.training, "target_shift", 0))

    # ЕДИНО: washout в отсчётах → в символы
    w_samples = get_washout_samples(cfg)
    M_eff = cfg.mask.mask_size if str(cfg.variant).startswith("temporal_") else 1
    w_syms = int(np.ceil(w_samples / max(1, int(M_eff))))

    N_eff = S - shift_syms - w_syms
    n_train = int(N_eff * cfg.training.train_frac) if N_eff > 0 else 0
    n_val = int(N_eff * cfg.training.val_frac) if N_eff > 0 else 0
    n_test = max(0, N_eff - n_train - n_val)

    i_shift = (0, max(0, shift_syms))
    i_wash = (i_shift[1], i_shift[1] + max(0, w_syms))
    i_tr = (i_wash[1], i_wash[1] + max(0, n_train))
    i_va = (i_tr[1], i_tr[1] + max(0, n_val))
    i_te = (i_va[1], min(S, i_va[1] + max(0, n_test)))

    fig = plt.figure(figsize=(COL2 * 0.62, COL2 * 0.62), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    ax.set_proj_type('ortho')
    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
    ax.view_init(elev=20, azim=-15)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0, 0, 0, 0))

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
                line.set_path_effects([
                    pe.Stroke(linewidth=base_lw * 1.8, foreground='white'),
                    pe.Normal()
                ])
                return True
        return False

    segments = [
        ("target shift", "#1f77b4", i_shift),
        ("washout", "#ff7f0e", i_wash),
        ("train", "#2ca02c", i_tr),
        ("val", "#9467bd", i_va),
        ("test", "#d62728", i_te),
    ]
    for name, color, bounds in segments:
        plot_segment(name, color, bounds)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        leg = ax.legend(uniq.values(), uniq.keys(),
                        bbox_to_anchor=(0.02, 0.98), loc="upper left",
                        frameon=True, title="Segments:")
        leg.get_frame().set_facecolor((1, 1, 1, 0.6))
        leg.get_frame().set_edgecolor((0, 0, 0, 0.3))

    if getattr(cfg.reservoir, "save_figs", False):
        p = _default_fig_path(f"mg_attractor_{cfg.variant}_C{cfg.core_count}")
        try:
            fig.savefig(p, bbox_inches="tight", pad_inches=0.04)
        except Exception as e:
            print(f"[warn] savefig failed: {e}")

    plt.show()


def debug_plot_post_training_comparison(cfg,
                                        y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        title: str = "Comparison: truth vs prediction",
                                        n_show: int = 2000,
                                        start: int = 0) -> float:
    """
    Рисует сравнение на тесте и возвращает NRMSE по ВСЕМУ переданному сегменту.

    Важно: отображаемое окно (start/n_show) влияет только на картинку, а не на метрику.
    Это нужно, чтобы NRMSE здесь совпадал с тем, что рисуется/считается по train/val/test в других местах.

    Устойчиво работает при коротких окнах: если точек < 2, вместо линии рисуется маркер,
    легенда и NRMSE подавляются, а set_xlim не вызывается (чтобы не ловить warning).
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    N = min(y_true.shape[0], y_pred.shape[0])
    if N == 0:
        fig, ax = plt.subplots(figsize=(COL2, COL2 * 0.33), constrained_layout=True)
        ax.set_title(f"{title}   •   NRMSE=—", loc="left")
        ax.set_xlabel("symbol index")
        ax.set_ylabel("MG series value")
        _maybe_savefig(fig, f"post_training_comparison_{cfg.variant}_C{cfg.core_count}",
                       enabled=getattr(cfg.reservoir, "save_figs", False))
        plt.show()
        return float('nan')

    y_true = y_true[:N]
    y_pred = y_pred[:N]

    # NRMSE по ПОЛНОМУ сегменту (как в result['metrics']).
    err = float('nan') if N < 2 else float(nrmse(y_true, y_pred))

    # окно визуализации
    start = int(max(0, start))
    end = int(min(N, start + int(max(1, n_show))))
    x = np.arange(start, end)
    Lvis = int(end - start)

    # <<< НОВОЕ: если окно < 2, но данных >= 2 — показываем хвост из 2 точек >>>
    if Lvis < 2 and N >= 2:
        start, end = max(0, N - 2), N
        x = np.arange(start, end)
        Lvis = end - start

    fig, ax = plt.subplots(figsize=(COL2, COL2 * 0.33), constrained_layout=True)

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

    _maybe_savefig(fig, f"post_training_comparison_{cfg.variant}_C{cfg.core_count}",
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
    Xva, yva = res["X_val"], res["y_val"].reshape(-1, 1)
    Xte, yte = res["X_test"], res["y_test"].reshape(-1, 1)

    # предсказания
    ytr_hat = apply_readout(Xtr, W)
    yva_hat = apply_readout(Xva, W) if Xva.size else np.zeros_like(yva)
    yte_hat = apply_readout(Xte, W) if Xte.size else np.zeros_like(yte)

    def _nrmse_safe(y, yhat):
        y = np.asarray(y).ravel();
        yhat = np.asarray(yhat).ravel()
        if y.size < 2 or yhat.size < 2:
            return float('nan')
        # используем тот же nrmse(), что и result["metrics"]
        return float(nrmse(y, yhat))

    m = {
        "nrmse_train": _nrmse_safe(ytr, ytr_hat),
        "nrmse_val": _nrmse_safe(yva, yva_hat),
        "nrmse_test": _nrmse_safe(yte, yte_hat),
    }

    # конкатенация
    y_all = np.concatenate([ytr, yva, yte], axis=0).reshape(-1)
    yhat_all = np.concatenate([ytr_hat, yva_hat, yte_hat], axis=0).reshape(-1)
    N = y_all.size

    n_tr = int(ytr.shape[0])
    n_va = int(yva.shape[0])
    n_te = int(yte.shape[0])

    b_tr = (0, n_tr)
    b_va = (n_tr, n_tr + n_va)
    b_te = (n_tr + n_va, n_tr + n_va + n_te)

    fig, ax = plt.subplots(figsize=(COL2, COL2 * 0.33), constrained_layout=True)
    x = np.arange(N)

    ax.plot(x, y_all, label="ground truth")
    ax.plot(x, yhat_all, label="prediction")

    def span_if(bounds, color, label):
        lo, hi = int(bounds[0]), int(bounds[1])
        if hi - lo >= 2:
            ax.axvspan(lo, hi, color=color, alpha=0.18, label=label)
            return True
        return False

    def _fmt_nrmse(v: float) -> str:
        return f"{v:.4f}" if np.isfinite(v) else "—"

    span_if(b_tr, "#2ca02c", f"train  (NRMSE={_fmt_nrmse(m['nrmse_train'])})")
    span_if(b_va, "#9467bd", f"val    (NRMSE={_fmt_nrmse(m['nrmse_val'])})")
    span_if(b_te, "#d62728", f"test   (NRMSE={_fmt_nrmse(m['nrmse_test'])})")

    # разделители — только если есть правая граница предыдущего сегмента
    if n_tr >= 1: ax.axvline(b_tr[1], color="k", lw=1, alpha=0.6)
    if n_va >= 1: ax.axvline(b_va[1], color="k", lw=1, alpha=0.6)

    ax.set_title(title, loc="left")
    ax.set_xlabel("symbol index")
    ax.set_ylabel("MG series value")
    ax.margins(x=0.0)

    # Легенда
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98),
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


def generate_input_temporal_unique_per_core(core_count: int, mg_cfg: MGConfig, mask_cfg: MaskConfig) -> Tuple[
    np.ndarray, np.ndarray]:
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


def generate_input_temporal_same_all_cores(core_count: int, mg_cfg: MGConfig, mask_cfg: MaskConfig) -> Tuple[
    np.ndarray, np.ndarray]:
    rng = np.random.default_rng(mask_cfg.seed)
    mg_series = _mg_series_from_cfg(mg_cfg)
    if mask_cfg.mask_size == 1:
        mask = 1
    else:
        mask = create_mask(mask_cfg.mask_size, rng, kind=mask_cfg.mask_kind)
    pattern = np.kron(mg_series, mask) * mask_cfg.gain_in
    data_in = np.tile(pattern, (core_count, 1))
    return data_in, mg_series


def generate_input_spatial_only(core_count: int, mg_cfg: MGConfig, mask_cfg: MaskConfig) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Spatial-only вход: один и тот же временной сигнал подаётся на все ядра, но с разными постоянными весами.

    Идеальный (lossless) SLM моделируем как перераспределение мощности между ядрами при сохранении суммарной:
        sum_c |u_c(t)|^2 = gain_in^2 * |s(t)|^2

    Для этого weights нормируются так, что sum(weights**2) = 1 (weights — коэффициенты по полю/амплитуде).
    """
    rng = np.random.default_rng(mask_cfg.seed)
    mg_series = _mg_series_from_cfg(mg_cfg)

    # Коэффициенты по полю (амплитуде). Оставляю ваш диапазон [0, 1], меняю только нормировку.
    weights = rng.uniform(0.0, 1.0, size=core_count)
    w_norm = float(np.linalg.norm(weights))
    if w_norm > 0.0:
        weights = weights / w_norm
    else:
        weights = np.zeros(core_count, dtype=float)
        weights[0] = 1.0

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
    ("reservoir", "ase_noise_compute_effective_osnr"),
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
    ("reservoir", "delta_phase"): 3,
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
    _base_dir = Path(__file__).parent.resolve()  # ./scripts
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
        delta_phase=rc["delta_phase"],
        use_gpu=bool(rc["use_gpu"]),
        use_torch=bool(rc["use_torch"]),
        num_threads=rc["num_threads"],
        display_debug_info=bool(rc["display_debug_info"]),
        display_debug_plots=bool(rc["display_debug_plots"]),
        save_figs=bool(rc["save_figs"]),
        save_gif=bool(rc["save_gif"]),
        max_hours_total=rc.get("max_hours_total", None),
        precision=rc.get("precision", None),
        use_dispersion=bool(rc.get("use_dispersion", True)),
        ase_noise_seed=rc.get("ase_noise_seed", None),
        ase_noise_nf_db=rc.get("ase_noise_nf_db", 5.0),
        ase_noise_optical_bw_hz=rc.get("ase_noise_optical_bw_hz", 12.5e9),
    )

    data_out = res["data_out"]
    params_out = res.get("params", {})
    if rc["display_debug_info"]:
        print(params_out)
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

def make_states(
        data_out: np.ndarray,
        mask_cfg,
        feature_mode: Literal["intensity", "realimag"] = "intensity",
        *,
        phase_shift: int = 0,
) -> Tuple[np.ndarray, int, int]:
    """
    Собирает матрицу признаков для time-multiplexed RC по символам.

    Parameters
    ----------
    data_out : (C, T) complex
        Выход резервара во времени (после нелинейности). T = N * M.
    mask_cfg : MaskConfig
        Должен содержать поле mask_size (M).
    feature_mode : "intensity", "realimag"
    phase_shift : int, default 0
        Циклический сдвиг по времени в узлах маски (компенсация фазы задержки
        относительно маски, если нужно): 0..M-1.

    Returns
    -------
    X : (N, F)
        Матрица признаков по символам. F = M*C при "intensity", либо F = 2*M*C при "realimag".
    N : int
        Число полных символов, попавших в X.
    M : int
        Размер маски (число виртуальных узлов на символ).
    """
    if data_out.ndim != 2:
        raise ValueError("data_out должен быть формы (C, T)")
    C, T = data_out.shape
    M = int(mask_cfg.mask_size)
    if M <= 0:
        raise ValueError("mask_size должен быть положительным целым числом")

    if M > 1:
        r = int(phase_shift) % M
        if r:
            data_out = np.roll(data_out, -r, axis=1)

    # Обрезаем хвост до кратности M
    if M > 1:
        N = T // M
        if N == 0:
            raise ValueError("недостаточно тактов для одного символа (T < M)")
        T_eff = N * M
        if T_eff != T:
            data_out = data_out[:, :T_eff]
            T = T_eff
    else:
        # Без временной маски трактуем каждый такт как символ (M=1)
        N = T

    if feature_mode == "intensity":
        feats = np.abs(data_out) ** 2  # (C, T)
        C_feat = C
    elif feature_mode == "realimag":
        feats = np.vstack([data_out.real, data_out.imag])  # (2C, T)
        C_feat = 2 * C
    else:
        raise ValueError(f"unknown feature_mode={feature_mode!r}")

    # Укладка по символам: (C_feat, N*M) -> (C_feat, N, M) -> (N, C_feat, M) -> (N, C_feat*M)
    feats_blk = feats.reshape(C_feat, N, M)  # (C_feat, N, M)
    X = np.transpose(feats_blk, (1, 0, 2)).reshape(N, C_feat * M)
    return X, N, M


def split_train_val_test(xw_len: int,
                         train_frac: float,
                         val_frac: float,
                         *,
                         data_out: Optional[np.ndarray] = None,
                         settle: bool = True,
                         tail_frac: float = 0.15,
                         win_symbols: int = 200,
                         tol_rel: float = 0.03,
                         tol_abs: float = 0.0,
                         k_sigma: float = 3.0,
                         align_to: Optional[int] = None) -> tuple[slice, slice, slice]:
    """
    Возвращает срезы (train, val, test), которые применяются к Xw (т.е. по оси времени Xw).

    1) Если data_out is None, то режем Xw длины xw_len по train_frac/val_frac.

    2) Если передан data_out (сырые данные из run_mcf_with_cache), то по мощности |E|^2:
       - оцениваем "конечную" среднюю мощность по хвосту сигнала (tail),
       - находим самый ранний момент i (в шкале data_out), после которого одновременно:
           (a) локальная средняя мощность на окне близка к конечной средней,
           (b) средняя мощность по всей оставшейся части [i:конец] близка к конечной средней,
         что защищает от случая "ранняя стационарность → потом переход в другой режим",
       - переводим найденный i в индекс Xw через масштабирование длинами (data_out vs Xw),
       - и только затем режем оставшуюся часть Xw по train_frac/val_frac.

    Параметры подбора точки установления (для data_out):
      settle: включить/выключить поиск установления.
      tail_frac: доля конца сигнала для оценки стационарной средней мощности.
      win_symbols: размер окна (в единицах Xw); в шкалу data_out переводится пропорционально длинам.
      tol_rel/tol_abs: допуски на отличие средней мощности от конечной.
      k_sigma: допуск через дисперсию хвоста (k_sigma * std / sqrt(win)).
      align_to: если задано, найденный старт в Xw делается кратным align_to (например mask_size или иной блок).
    """
    if train_frac + val_frac > 1.0:
        raise ValueError("train_frac + val_frac must be <= 1.0")

    if not isinstance(xw_len, (int, np.integer)):
        raise TypeError("xw_len must be an integer (e.g., Xw.shape[0])")

    T_xw = int(xw_len)
    if T_xw < 0:
        raise ValueError("xw_len must be non-negative")
    if T_xw == 0:
        return slice(0, 0), slice(0, 0), slice(0, 0)

    # --- Case A: classic behaviour (no data_out) -> split Xw directly
    if data_out is None or not settle:
        train_N = round(T_xw * train_frac)
        val_N = round(T_xw * val_frac)
        test_N = T_xw - train_N - val_N

        i0 = 0
        i1 = i0 + train_N
        i2 = i1 + val_N
        i3 = i2 + test_N
        return slice(i0, i1), slice(i1, i2), slice(i2, i3)

    # --- Case B: data_out provided -> find "settled" start point using power statistics (in data_out),
    #             then map it to Xw indices
    data = np.asarray(data_out)
    if data.ndim < 1:
        raise ValueError(f"Unsupported data_out ndim={data.ndim}; expected array with time as the last axis.")

    power = np.abs(data) ** 2
    if power.ndim > 1:
        # Усредняем по всем осям, кроме последней (время): "исходя из всех данных о мощности"
        power_t = np.mean(power, axis=tuple(range(power.ndim - 1)))
    else:
        power_t = power

    power_t = power_t.astype(np.float64, copy=False)
    T_pow = int(power_t.shape[0])
    if T_pow <= 0:
        # Нечего анализировать по мощности -> режем как обычно
        train_N = round(T_xw * train_frac)
        val_N = round(T_xw * val_frac)
        test_N = T_xw - train_N - val_N
        return slice(0, train_N), slice(train_N, train_N + val_N), slice(train_N + val_N, T_xw)

    start_xw = 0

    if T_pow >= 10:
        tail_len = max(1, int(round(T_pow * tail_frac)))
        tail_start = max(0, T_pow - tail_len)
        tail = power_t[tail_start:]
        mu_tail = float(np.mean(tail))
        std_tail = float(np.std(tail))

        # Перевод "окна в Xw" -> "окно в data_out" по пропорции длин:
        # samples_per_xw ~ T_pow / T_xw
        samples_per_xw = float(T_pow) / max(1.0, float(T_xw))

        win_xw = int(max(1, win_symbols))
        if align_to is not None and align_to > 0:
            win_xw = int(max(1, win_xw * int(align_to)))

        win = int(max(1, round(win_xw * samples_per_xw)))
        win = min(win, T_pow)

        tol = max(
            float(tol_abs),
            float(tol_rel) * max(abs(mu_tail), 1e-30),
            float(k_sigma) * std_tail / max(float(np.sqrt(win)), 1.0),
        )

        # prefix sums for fast means
        cs = np.cumsum(np.concatenate(([0.0], power_t)))

        # mean on [i:i+win)
        mu_local = (cs[win:] - cs[:-win]) / float(win)  # len = T_pow - win + 1

        # mean on [i:T_pow)
        idx = np.arange(T_pow, dtype=np.int64)
        mu_from = (cs[-1] - cs[:-1]) / (T_pow - idx).astype(np.float64)  # len = T_pow

        L = int(mu_local.shape[0])
        ok_local = np.abs(mu_local - mu_tail) <= tol
        ok_from = np.abs(mu_from[:L] - mu_tail) <= tol
        ok = ok_local & ok_from

        if np.any(ok):
            start_pow = int(np.argmax(ok))  # first True, in data_out index space

            # Map data_out index -> Xw index by length proportion (robust when sizes differ)
            start_xw = int(round(start_pow * float(T_xw) / max(1.0, float(T_pow))))

            # Align start to block boundary in Xw space if requested (e.g., mask_size)
            if align_to is not None and align_to > 1:
                start_xw = int(((start_xw + align_to - 1) // align_to) * align_to)

            start_xw = max(0, min(start_xw, T_xw - 1))

    # Split the remaining Xw samples [start_xw:T_xw)
    T_eff = max(0, T_xw - start_xw)
    train_N = round(T_eff * train_frac)
    val_N = round(T_eff * val_frac)
    test_N = T_eff - train_N - val_N

    i0 = start_xw
    i1 = i0 + train_N
    i2 = i1 + val_N
    i3 = i2 + test_N

    return slice(i0, i1), slice(i1, i2), slice(i2, i3)


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


def _compute_effective_output_osnr_on_arrays(
        data_out_clean: np.ndarray,
        data_out_noisy: np.ndarray,
        eps: float = 1e-30,
) -> dict:
    """
    Effective output OSNR по паре массивов:
      clean  - прогон без шума
      noisy  - прогон с шумом

    OSNR_eff,out = 10 log10( <|U_clean|^2> / <|U_noisy - U_clean|^2> )
    """
    u_clean = np.asarray(data_out_clean)
    u_noisy = np.asarray(data_out_noisy)

    if u_clean.shape != u_noisy.shape:
        raise ValueError(f"Shape mismatch: clean={u_clean.shape}, noisy={u_noisy.shape}")

    if u_clean.size == 0:
        return {
            "p_signal_w": float("nan"),
            "p_noise_eff_w": float("nan"),
            "osnr_eff_out_db": float("nan"),
            "n_time_samples": 0,
        }

    p_sig = float(np.mean(np.abs(u_clean) ** 2))
    p_noise_eff = float(np.mean(np.abs(u_noisy - u_clean) ** 2))

    if p_sig <= eps:
        raise ValueError("Signal power is too small or zero in clean run")

    if p_noise_eff <= eps:
        osnr_db = float("inf")
    else:
        osnr_db = float(10.0 * np.log10(p_sig / p_noise_eff))

    return {
        "p_signal_w": p_sig,
        "p_noise_eff_w": p_noise_eff,
        "osnr_eff_out_db": osnr_db,
        "n_time_samples": int(u_clean.shape[-1]),
    }


def _get_effective_osnr_time_slices(
        cfg: ExperimentConfig,
        data_out: np.ndarray,
) -> dict:
    """
    Возвращает временные срезы data_out, соответствующие областям:
      full, train, val, test, val_test

    Срезы строятся честно в тех же координатах, что и обучение readout:
      make_states -> taps -> target_shift -> washout -> split_train_val_test
    """
    if data_out.ndim != 2:
        raise ValueError("data_out должен быть формы (C, T)")

    cfg_with_ws = _dc.replace(
        cfg,
        reservoir=_dc.replace(cfg.reservoir, window_size=compute_window_size_samples(cfg))
    )

    T = int(data_out.shape[1])
    M_eff = int(cfg.mask.mask_size) if int(cfg.mask.mask_size) > 0 else 1

    if M_eff > 1:
        S = T // M_eff
        if S == 0:
            raise ValueError("Недостаточно временных отсчётов для одного символа")
    else:
        S = T

    taps = int(cfg.training.taps)
    shift_syms = int(cfg.training.target_shift)

    aligned_len = S - (taps - 1 + shift_syms)
    if aligned_len < 0:
        raise ValueError("target_shift или taps слишком велики для длины data_out")

    w_samples = get_washout_samples(cfg_with_ws)
    w_syms = int(np.ceil(w_samples / max(1, M_eff)))

    xw_len = aligned_len - w_syms
    if xw_len < 0:
        raise ValueError("washout слишком велик для доступной длины после taps/shift")

    sl_train, sl_val, sl_test = split_train_val_test(
        xw_len,
        cfg.training.train_frac,
        cfg.training.val_frac,
        data_out=data_out,
    )

    sym_offset = w_syms + taps - 1

    def _to_time_slice(sl: slice) -> slice:
        a = sym_offset + int(sl.start or 0)
        b = sym_offset + int(sl.stop or 0)

        a = max(0, min(a * M_eff, T))
        b = max(0, min(b * M_eff, T))
        return slice(a, b)

    time_train = _to_time_slice(sl_train)
    time_val = _to_time_slice(sl_val)
    time_test = _to_time_slice(sl_test)

    if (time_val.stop - time_val.start) > 0:
        val_test_start = time_val.start
    else:
        val_test_start = time_test.start

    if (time_test.stop - time_test.start) > 0:
        val_test_stop = time_test.stop
    else:
        val_test_stop = time_val.stop

    return {
        "full": slice(0, T),
        "train": time_train,
        "val": time_val,
        "test": time_test,
        "val_test": slice(val_test_start, val_test_stop),
    }


def compute_effective_output_osnr_from_results(
        clean_result: dict,
        noisy_result: dict,
        eps: float = 1e-30,
) -> dict:
    """
    Считает честный effective output OSNR по одной и той же конфигурации
    для пары прогонов clean/noisy.

    OSNR считается на raw data_out, но отдельно по тем временным кускам,
    которые соответствуют train / val / test после taps / shift / washout.
    """
    cfg = clean_result["cfg"]

    data_out_clean = np.asarray(clean_result["data_out"])
    data_out_noisy = np.asarray(noisy_result["data_out"])

    if data_out_clean.shape != data_out_noisy.shape:
        raise ValueError(
            f"Shape mismatch: clean={data_out_clean.shape}, noisy={data_out_noisy.shape}"
        )

    scopes = _get_effective_osnr_time_slices(cfg, data_out_clean)

    out = {}
    for name, sl in scopes.items():
        out[name] = _compute_effective_output_osnr_on_arrays(
            data_out_clean[:, sl],
            data_out_noisy[:, sl],
            eps=eps,
        )
        out[name]["time_slice"] = (int(sl.start), int(sl.stop))

    return out


# =========================
# Сценарии запуска
# =========================

def run_experiments(base_cfg: ExperimentConfig,
                    n_trials_opt: int = 0,
                    free_run_horizon: int = 0,
                    force_rerun: bool = False,
                    save_cache: bool = False,
                    *,
                    param_space: Optional[dict] = None) -> Dict[str, Any]:
    """
    ЕДИНАЯ точка входа.
    - Использует base_cfg.variant как источник истины (ничего не переопределяет).
    - Если n_trials_opt > 0 → Optuna-оптимизация поверх run_single_experiment.
    - Иначе → один прогон run_single_experiment.
    """
    cfg = base_cfg  # без копирования: вызывающий сам управляет копией/базой

    if n_trials_opt and n_trials_opt > 0:
        return optimize_hyperparams(cfg, n_trials=n_trials_opt,
                                    n_jobs=physical_cpu_count(),
                                    free_run_horizon=free_run_horizon,
                                    force_rerun=force_rerun,
                                    save_cache=save_cache,
                                    param_space=param_space)

    res_noisy = run_single_experiment(cfg,
                                      free_run_horizon=free_run_horizon,
                                      force_rerun=force_rerun,
                                      save_cache=save_cache)

    if bool(getattr(cfg.reservoir, "ase_noise_compute_effective_osnr", False)) and cfg.reservoir.ase_noise_seed is not None:
        cfg_clean = _dc.replace(
            cfg,
            reservoir=_dc.replace(
                cfg.reservoir,
                ase_noise_seed=None,
                ase_noise_compute_effective_osnr=False,
            ),
        )

        res_clean = run_single_experiment(cfg_clean,
                                          free_run_horizon=free_run_horizon,
                                          force_rerun=force_rerun,
                                          save_cache=save_cache)

        eff = compute_effective_output_osnr_from_results(res_clean, res_noisy)

        res_noisy["effective_output_osnr"] = eff
        res_noisy["metrics"]["effective_output_osnr_full_db"] = eff["full"]["osnr_eff_out_db"]
        res_noisy["metrics"]["effective_output_osnr_train_db"] = eff["train"]["osnr_eff_out_db"]
        res_noisy["metrics"]["effective_output_osnr_val_db"] = eff["val"]["osnr_eff_out_db"]
        res_noisy["metrics"]["effective_output_osnr_test_db"] = eff["test"]["osnr_eff_out_db"]
        res_noisy["metrics"]["effective_output_osnr_val_test_db"] = eff["val_test"]["osnr_eff_out_db"]

    return res_noisy


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
    3) формируем X,y (символьно, с учётом taps/shift/washout),
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
    X_sym, S, M_eff = make_states(data_out, cfg.mask, feature_mode=cfg.training.feature_mode)  # (S,F)
    if S < 10:
        raise ValueError("Слишком мало символов: увеличьте длину MG или уменьшите mask_size")

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

    # === ЕДИНЫЙ washout: считаем в отсчётах и переводим в СИМВОЛЫ, затем отбрасываем первые w_syms ===
    w_samples = get_washout_samples(cfg_with_ws)
    w_syms = int(np.ceil(w_samples / max(1, int(M_eff))))
    if w_syms > 0:
        if X_aligned.shape[0] <= w_syms or y_aligned.shape[0] <= w_syms:
            raise ValueError("washout слишком велик для доступной длины после taps/shift")
        Xw = X_aligned[w_syms:, :]
        yw = y_aligned[w_syms:, :]
    else:
        Xw, yw = X_aligned, y_aligned

    # сплиты
    sl_train, sl_val, sl_test = split_train_val_test(
        Xw.shape[0], cfg.training.train_frac, cfg.training.val_frac, data_out=data_out
    )
    Xtr, ytr = Xw[sl_train], yw[sl_train]
    Xva, yva = Xw[sl_val], yw[sl_val]
    Xte, yte = Xw[sl_test], yw[sl_test]

    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    Xtr = (Xtr - mu) / sigma
    Xva = (Xva - mu) / sigma
    Xte = (Xte - mu) / sigma

    # обучение рид-аута
    W, ridge_alpha = train_ridge(Xtr, ytr, alpha=cfg.training.ridge_alpha, add_bias=True, X_val=Xva, y_val=yva)

    # метрики
    ytr_hat = apply_readout(Xtr, W)
    yva_hat = apply_readout(Xva, W)
    yte_hat = apply_readout(Xte, W)
    metrics = dict(
        nrmse_train=nrmse(ytr, ytr_hat),
        nrmse_val=nrmse(yva, yva_hat),
        nrmse_test=nrmse(yte, yte_hat),
        ridge_alpha=ridge_alpha,
        T_total=int(X_sym.shape[0]),
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

def _round(x, digit=13) -> float:
    return float(round(float(x), digit))


def optimize_hyperparams(base_cfg: ExperimentConfig,
                         n_trials: int,
                         n_jobs: Optional[int] = None,
                         free_run_horizon: int = 0,
                         force_rerun: bool = False,
                         save_cache: bool = False,
                         *,
                         param_space: Optional[dict] = None) -> Dict[str, Any]:
    """
    Подбор гиперпараметров с Optuna.

    Управление поисковым пространством из main:
      param_space: dict | None
        Ключи: "kappa", "delta_phase", "fiber_length_m", "gain_in",
               "mask_size", "delay_factor_in_symbols", "time_step_ps",
               и ЛИБО ("g0", "psat"), ЛИБО pump-параметры:
                 - "ppump" (как число или список per-core),
                 - ИЛИ "p_pump_total" + "pump_slm_w" (pump_slm_w как список per-core).
        Значение для ключа:
          • число → фиксировать (не оптимизировать);
          • {"low":a,"high":b,"log":bool?,"step":s?,"sig":int?,"kind":"float"/"int"} → suggest_*;
          • {"choices":[...]} или список/кортеж длиной >1 → categorical;
          • (a, b) → suggest_float(a..b).

    Всегда в 1 поток внутри trial (temporary_thread_limits(1)).

    Важно (кластер / много процессов):
      Печать "GLOBAL BEST" синхронизируется через lock-файл на общей ФС.
    Прогресс:
      MCF_OPTUNA_PROGRESS_EVERY (default: 1000) — печатает
        [optuna][PROGRESS] trial=<n> ts=<iso8601>
    """
    import optuna
    import warnings
    try:
        from optuna._experimental import ExperimentalWarning  # noqa: WPS433
    except Exception:
        class ExperimentalWarning(Warning):
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
    lock_obj = JournalFileOpenLock(str(journal_path)) if os.name == "nt" else None
    storage = JournalStorage(JournalFileBackend(str(journal_path), lock_obj=lock_obj))

    try:
        (base_dir / "mcf_rc_cache").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    if n_jobs is None:
        if os.getenv("MCF_BASH", ""):
            n_jobs = 1
        else:
            n_jobs = max(1, physical_cpu_count())
    if n_jobs == 6:
        n_jobs = 5

    print("n_trials =", n_trials)
    print("n_jobs   =", n_jobs)

    best = dict(score=float("inf"), res=None, params=None)

    progress_every = int(os.getenv("MCF_OPTUNA_PROGRESS_EVERY", "1000") or "1000")
    if progress_every <= 0:
        progress_every = 1000
    # --- "GLOBAL BEST" печатаем только один раз на улучшение (межпроцессная синхронизация) ---
    best_state_path = base_dir / ".optuna_global_best.json"
    best_lock_path = base_dir / ".optuna_global_best.lock"
    last_best_printed_local = {"trial_number": -1}

    def _print_best_trial(bt: "optuna.trial.FrozenTrial") -> None:
        print(f"[optuna][GLOBAL BEST] trial={bt.number} value={bt.value}", flush=True)

        best_params_main = bt.user_attrs.get("best_params_main")
        if best_params_main:
            print(best_params_main, flush=True)

        print("[params]", flush=True)
        for k in sorted(bt.params.keys()):
            print(f"  {k} = {bt.params[k]}", flush=True)

        print("[user_attrs]", flush=True)
        if bt.user_attrs:
            for k in sorted(bt.user_attrs.keys()):
                if k == "best_params_main":
                    continue
                print(f"  {k} = {bt.user_attrs[k]}", flush=True)
        else:
            print("  <empty>", flush=True)

        print("-" * 80, flush=True)

    def _print_progress(trial_number: int) -> None:
        if trial_number <= 0:
            return
        if (trial_number % int(progress_every)) != 0:
            return
        print(
            f"[optuna][PROGRESS] trial={trial_number} ts={datetime.now().isoformat(timespec='seconds')}",
            flush=True,
        )

    def _print_global_best_if_updated(study: "optuna.study.Study") -> None:
        """
        Печатает best_trial только тогда, когда глобальный best действительно обновился.

        В multi-process режиме (кластер) синхронизируемся через lock-файл на общей ФС.
        """
        try:
            bt = study.best_trial
        except Exception:
            return

        best_no = int(getattr(bt, "number", -1))
        if best_no < 0:
            return

        if os.name == "posix":
            try:
                import fcntl  # noqa: WPS433
            except Exception:
                fcntl = None

            if fcntl is not None:
                best_lock_path.parent.mkdir(parents=True, exist_ok=True)
                with open(best_lock_path, "a+", encoding="utf-8") as lf:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX)

                    prev_no = -1
                    if best_state_path.exists():
                        try:
                            import json  # noqa: WPS433
                            with open(best_state_path, "r", encoding="utf-8") as sf:
                                prev_no = int(json.load(sf).get("best_trial_number", -1))
                        except Exception:
                            prev_no = -1

                    if best_no <= prev_no:
                        return

                    try:
                        import json  # noqa: WPS433
                        with open(best_state_path, "w", encoding="utf-8") as sf:
                            json.dump(
                                {
                                    "best_trial_number": best_no,
                                    "best_value": float(bt.value) if bt.value is not None else None,
                                    "ts": datetime.now().isoformat(timespec="seconds"),
                                },
                                sf,
                                ensure_ascii=False,
                            )
                    except Exception:
                        pass

                    _print_best_trial(bt)
                    return

        if best_no <= int(last_best_printed_local["trial_number"]):
            return
        last_best_printed_local["trial_number"] = best_no
        _print_best_trial(bt)

    def _as_num(x):
        import numbers
        return isinstance(x, numbers.Number)

    DEFAULT_SIG = 5
    _log_grid_cache = {}

    def _get_sig(name: str, idx: Optional[int] = None) -> int:
        if not param_space or name not in param_space:
            return DEFAULT_SIG

        spec = param_space.get(name)

        if idx is not None and isinstance(spec, (list, tuple)):
            if 0 <= idx < len(spec) and isinstance(spec[idx], dict) and "sig" in spec[idx]:
                try:
                    return int(spec[idx]["sig"])
                except Exception:
                    return DEFAULT_SIG

        if isinstance(spec, dict) and "sig" in spec:
            try:
                return int(spec["sig"])
            except Exception:
                return DEFAULT_SIG

        return DEFAULT_SIG

    def _format_number_for_main(x, sig: int = DEFAULT_SIG) -> str:
        if isinstance(x, (bool, np.bool_)):
            return "True" if x else "False"

        if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
            return str(int(x))

        if isinstance(x, (float, np.floating)):
            xf = float(x)
            if not np.isfinite(xf):
                return repr(xf)
            xr = _round_sig(xf, sig)
            return repr(xr)

        return repr(x)

    def _format_tuple_for_main(name: str, values) -> str:
        parts = []
        for i, v in enumerate(values):
            parts.append(_format_number_for_main(v, sig=_get_sig(name, idx=i)))
        if len(parts) == 1:
            return f"({parts[0]},)"
        return "(" + ", ".join(parts) + ")"

    def _variant_comment_label(variant: str) -> str:
        if variant == "spatial_only":
            return "spatial mask"
        if variant == "temporal_same_all_cores":
            return "equal temporal mask"
        if variant == "temporal_unique_per_core":
            return "unique temporal mask"
        return variant

    def _build_best_params_main(cfg: ExperimentConfig,
                                metrics: dict,
                                p_pump_per_core=None) -> str:
        nrmse_val = float(metrics["nrmse_val"])
        alpha = float(metrics["ridge_alpha"])

        params_lines = [
            f'# best {cfg.core_count}-core with {_variant_comment_label(cfg.variant)} '
            f'("nrmse_val": {_format_number_for_main(nrmse_val, sig=5)}), '
            f'alpha={_format_number_for_main(alpha, sig=5)}',
            "params = {",
            f'        "layer_count": {_format_number_for_main(cfg.reservoir.layer_count, sig=5)},',
            f'        "variant": {repr(cfg.variant)},',
            f'        "temporal_modulation_frequency_ghz": '
            f'{_format_number_for_main(1e3 / float(cfg.reservoir.time_step_ps), sig=_get_sig("time_step_ps"))},',
            f'        "delay_factor_in_symbols": '
            f'{_format_number_for_main(cfg.reservoir.delay_factor_in_symbols, sig=_get_sig("delay_factor_in_symbols"))},',
            f'        "delta_phase": {_format_number_for_main(cfg.reservoir.delta_phase, sig=_get_sig("delta_phase"))},',
            f'        "fiber_length_m": {_format_number_for_main(cfg.reservoir.fiber_length_m, sig=_get_sig("fiber_length_m"))},',
        ]

        if p_pump_per_core is not None:
            params_lines.append(f'        "ppump": {_format_tuple_for_main("ppump", p_pump_per_core)},')
        else:
            if len(cfg.reservoir.g0_array) == 1:
                params_lines.append(
                    f'        "g0": {_format_number_for_main(cfg.reservoir.g0_array[0], sig=_get_sig("g0", idx=0))},'
                )
            else:
                params_lines.append(f'        "g0": {_format_tuple_for_main("g0", cfg.reservoir.g0_array)},')

            if len(cfg.reservoir.psat_array) == 1:
                params_lines.append(
                    f'        "psat": {_format_number_for_main(cfg.reservoir.psat_array[0], sig=_get_sig("psat", idx=0))},'
                )
            else:
                params_lines.append(f'        "psat": {_format_tuple_for_main("psat", cfg.reservoir.psat_array)},')

        params_lines.extend(
            [
                f'        "gain_in": {_format_number_for_main(cfg.mask.gain_in, sig=_get_sig("gain_in"))},',
                f'        "kappa": {_format_number_for_main(cfg.reservoir.kappa, sig=_get_sig("kappa"))},',
                f'        "mask_size": {_format_number_for_main(cfg.mask.mask_size, sig=_get_sig("mask_size"))},',
                "    }",
            ]
        )

        return "\n".join(params_lines)

    def _round_sig(x: float, sig: int) -> float:
        if float(x) == 0.0:
            return 0.0
        return float(f"{float(x):.{int(sig)}g}")

    def _step_from_sig(low_val: float, high_val: float, sig: int) -> float:
        s = max(abs(float(low_val)), abs(float(high_val)))
        if not np.isfinite(s) or s == 0.0:
            return 10.0 ** (-int(sig) + 1)
        order = int(np.floor(np.log10(s)))
        return 10.0 ** (order - int(sig) + 1)

    def _suggest(
            name: str,
            trial,
            default_kind: str = "float",
            default_low=None,
            default_high=None,
            *,
            search_space: dict | None = None,
            default_log: bool = False,
            default_step=None,
    ) -> object:
        """
        Унифицированный helper для предложений гиперпараметров.
        - Если во внешнем search_space указан фикс (int/float/str/bool) или список из 1 элемента — возвращает его.
        - Если список длиной >1 — трактуется как categorical.
        - Если dict — читаются поля: kind, low, high, step, log, choices.
        - Если параметр не описан в search_space — используются default_*.
        """
        spec = None
        if search_space and name in search_space:
            spec = search_space[name]

        if isinstance(spec, (int, float, str, bool)):
            return spec
        if isinstance(spec, (list, tuple)):
            if len(spec) == 1:
                return spec[0]
            return trial.suggest_categorical(name, list(spec))

        if isinstance(spec, dict):
            kind = spec.get("kind", default_kind)
            low = spec.get("low", default_low)
            high = spec.get("high", default_high)
            step = spec.get("step", None)
            log = bool(spec.get("log", False))
            choices = spec.get("choices", None)
            sig = spec.get("sig", None)
        else:
            kind = default_kind
            low = default_low
            high = default_high
            step = default_step
            log = bool(default_log)
            choices = None
            sig = None

        if choices is not None:
            return trial.suggest_categorical(name, list(choices))

        if low is None or high is None:
            raise ValueError(f"{name}: missing low/high for {kind}")

        if kind == "int":
            if step is None:
                return trial.suggest_int(name, int(low), int(high), log=log)

            s = int(step)
            if s <= 0:
                s = 1
            if log and s != 1:
                s = 1
            return trial.suggest_int(name, int(low), int(high), step=s, log=log)

        if kind == "float":
            low_f = float(low)
            high_f = float(high)
            if low_f == high_f:
                return low_f

            try:
                sig_i = int(sig) if sig is not None else DEFAULT_SIG
            except Exception:
                sig_i = DEFAULT_SIG

            if log:
                if low_f <= 0.0 or high_f <= 0.0:
                    raise ValueError(
                        f"{name}: log-parameter requires low>0 and high>0, got low={low_f}, high={high_f}"
                    )
                if low_f > high_f:
                    low_f, high_f = high_f, low_f

                raw = float(trial.suggest_float(name, low_f, high_f, log=True))

                rel = 10.0 ** (-int(sig_i) + 1)
                ratio = 1.0 + rel
                if ratio <= 1.0:
                    ratio = 1.0 + 1e-12

                try:
                    n = int(round(np.log(raw / low_f) / np.log(ratio)))
                except Exception:
                    n = 0

                v = low_f * (ratio ** n)
                v = _round_sig(v, sig_i)

                if v < low_f:
                    v = low_f
                if v > high_f:
                    v = high_f
                return v

            if step is None:
                step = _step_from_sig(low_f, high_f, sig_i)

            s = float(step) if step is not None else None
            if s is None or (not np.isfinite(s)) or s <= 0:
                return trial.suggest_float(name, low_f, high_f, log=False)
            return trial.suggest_float(name, low_f, high_f, step=s, log=False)

        return trial.suggest_categorical(name, list(choices) if choices is not None else [])

    def objective(trial: "optuna.trial.Trial") -> float:
        with temporary_thread_limits(1):
            kappa = (_suggest("kappa", trial, default_kind="float", default_low=0.1, default_high=0.99,
                              search_space=param_space))
            delta_phase = (_suggest("delta_phase", trial, default_kind="float", default_low=0.0,
                                    default_high=2 * np.pi, search_space=param_space))

            fiber_length = (_suggest("fiber_length_m", trial, default_kind="float",
                                     default_low=0.2, default_high=3.0, default_log=True, search_space=param_space))

            def _suggest_per_core_list(name: str,
                                       *,
                                       default_kind: str,
                                       default_low,
                                       default_high,
                                       default_log: bool = False) -> list:
                """
                Если param_space[name] = list/tuple, то это per-core спецификации.
                Для каждого ядра создаём локальный search_space, где ключ name_i -> spec_i,
                иначе Optuna не увидит bounds (потому что в исходном param_space нет name_i).
                """
                if not param_space or name not in param_space:
                    v = _suggest(name, trial, default_kind=default_kind, default_low=default_low,
                                 default_high=default_high, default_log=default_log, search_space=param_space)
                    return [v] * base_cfg.core_count

                spec = param_space.get(name)

                if isinstance(spec, (list, tuple)):
                    out = []
                    for i in range(base_cfg.core_count):
                        spec_i = spec[i] if i < len(spec) else spec[-1]
                        local_space = dict(param_space)
                        local_space[f"{name}_{i}"] = spec_i
                        v = _suggest(f"{name}_{i}", trial, default_kind=default_kind, default_low=default_low,
                                     default_high=default_high, default_log=default_log, search_space=local_space)
                        out.append(v)
                    return out

                v = _suggest(name, trial, default_kind=default_kind, default_low=default_low,
                             default_high=default_high, default_log=default_log, search_space=param_space)
                return [v] * base_cfg.core_count

            # --- g0/psat либо напрямую, либо через pump -> (g0, psat) ---
            g0_array = []
            psat_array = []
            p_pump_per_core = None

            has_p_pump = bool(param_space and ("ppump" in param_space))
            has_p_pump_total = bool(
                param_space and (("p_pump_total" in param_space) or ("p_pump_total_w" in param_space))
            )
            has_pump_slm_w = bool(param_space and ("pump_slm_w" in param_space))

            if has_p_pump_total and has_pump_slm_w:
                p_total_name = "p_pump_total_w" if ("p_pump_total_w" in param_space) else "p_pump_total"
                p_pump_total = _suggest(p_total_name, trial, default_kind="float",
                                        default_low=0.01, default_high=1.0, default_log=True, search_space=param_space)
                pump_slm_w = _suggest_per_core_list("pump_slm_w", default_kind="float",
                                                    default_low=-3.0, default_high=3.0, default_log=False)
                p_pump_per_core = calc_p_pump_per_core_from_slm(float(p_pump_total), pump_slm_w)

                g0_t, psat_t = calc_g0_psat_from_p_pump_josab(p_pump_per_core, float(fiber_length), alpha=0)
                g0_array = list(g0_t)
                psat_array = list(psat_t)

            elif has_p_pump:
                p_pump_per_core = _suggest_per_core_list("ppump", default_kind="float",
                                                         default_low=0.01, default_high=1.0, default_log=True)

                g0_t, psat_t = calc_g0_psat_from_p_pump_josab(
                    tuple(float(x) for x in p_pump_per_core), float(fiber_length), alpha=0
                )
                g0_array = list(g0_t)
                psat_array = list(psat_t)

            else:
                g0_array = _suggest_per_core_list("g0", default_kind="float",
                                                  default_low=0.01, default_high=20.0, default_log=True)
                psat_array = _suggest_per_core_list("psat", default_kind="float",
                                                    default_low=1e-5, default_high=0.1, default_log=True)

            gain_in = (_suggest("gain_in", trial, default_kind="float",
                                default_low=1.0, default_high=1e2, default_log=True, search_space=param_space))
            mask_size = _suggest("mask_size", trial, default_kind="int", default_low=30, default_high=400,
                                 search_space=param_space)
            delay_factor = _suggest("delay_factor_in_symbols", trial, default_kind="int", default_low=1,
                                    default_high=20, search_space=param_space)
            time_step_ps = _suggest("time_step_ps", trial, default_kind="float",
                                    default_low=base_cfg.reservoir.time_step_ps,
                                    default_high=base_cfg.reservoir.time_step_ps,
                                    search_space=param_space)

            cfg = ExperimentConfig(
                core_count=base_cfg.core_count,
                mg=base_cfg.mg,
                mask=MaskConfig(mask_size=int(mask_size),
                                mask_kind=base_cfg.mask.mask_kind,
                                seed=base_cfg.mask.seed,
                                gain_in=gain_in),
                reservoir=ReservoirConfig(
                    fiber_length_m=fiber_length,
                    time_step_ps=time_step_ps,
                    step_number_per_dimensionless_distance=base_cfg.reservoir.step_number_per_dimensionless_distance,
                    upsampling=base_cfg.reservoir.upsampling,
                    layer_count=base_cfg.reservoir.layer_count,
                    layer_radii_array=base_cfg.reservoir.layer_radii_array,
                    g0_array=tuple(g0_array),
                    psat_array=tuple(psat_array),
                    kappa=kappa,
                    delta_phase=delta_phase,
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
                    ase_noise_seed=base_cfg.reservoir.ase_noise_seed,
                    ase_noise_nf_db=base_cfg.reservoir.ase_noise_nf_db,
                    ase_noise_optical_bw_hz=base_cfg.reservoir.ase_noise_optical_bw_hz,
                    ase_noise_compute_effective_osnr=base_cfg.reservoir.ase_noise_compute_effective_osnr,
                ),
                training=base_cfg.training,
                variant=base_cfg.variant
            )

            try:
                res = run_experiments(cfg,
                                      n_trials_opt=0,
                                      free_run_horizon=free_run_horizon,
                                      force_rerun=force_rerun,
                                      save_cache=save_cache)
            except (AssertionError, ValueError, RuntimeError) as e:
                msg = str(e).lower()
                if ("feedback_length" in msg and "fiber" in msg) or "invalid_config" in msg:
                    trial.set_user_attr("skip_reason", "feedback_length_le_fiber_length")
                    raise TrialPruned("invalid_config: feedback_length <= fiber_length")
                if "washout" in msg or "wash-out" in msg:
                    trial.set_user_attr("skip_reason", "washout_too_small")
                    raise TrialPruned("skip_reason: washout too small")
                if "time_limit_exceeded" in msg:
                    trial.set_user_attr("skip_reason", "time_budget")
                    raise TrialPruned("time budget exceeded")
                if "iteration_not_converged" in msg:
                    trial.set_user_attr("skip_reason", "no_convergence")
                    raise TrialPruned("no convergence")
                raise

            score = min(float(res["metrics"]["nrmse_val"]), 2)  # для оптимизации нельзя использовать nrmse_test!
            if not np.isfinite(score):
                trial.set_user_attr("skip_reason", f"non-finite score {score}")
                raise TrialPruned("skip_reason: non-finite score")

            trial.set_user_attr(
                "best_params_main",
                _build_best_params_main(cfg, res["metrics"], p_pump_per_core=p_pump_per_core)
            )

            trial.set_user_attr(
                "mcf",
                {
                    "nrmse_train": float(res["metrics"]["nrmse_train"]),
                    "nrmse_val": float(res["metrics"]["nrmse_val"]),
                    "nrmse_test": float(res["metrics"]["nrmse_test"]),
                    "ridge_alpha": float(res["metrics"]["ridge_alpha"]),
                    "effective_output_osnr_full_db": (
                        float(res["metrics"]["effective_output_osnr_full_db"])
                        if "effective_output_osnr_full_db" in res["metrics"] else None
                    ),
                    "effective_output_osnr_train_db": (
                        float(res["metrics"]["effective_output_osnr_train_db"])
                        if "effective_output_osnr_train_db" in res["metrics"] else None
                    ),
                    "effective_output_osnr_val_db": (
                        float(res["metrics"]["effective_output_osnr_val_db"])
                        if "effective_output_osnr_val_db" in res["metrics"] else None
                    ),
                    "cfg.mask.mask_size": int(cfg.mask.mask_size),
                    "cfg.mask.gain_in": float(cfg.mask.gain_in),
                    "cfg.reservoir.kappa": float(cfg.reservoir.kappa),
                    "cfg.reservoir.delta_phase": float(cfg.reservoir.delta_phase),
                    "cfg.reservoir.upsampling": int(cfg.reservoir.upsampling),
                    "cfg.reservoir.delay_factor_in_symbols": int(cfg.reservoir.delay_factor_in_symbols),
                    "cfg.reservoir.time_step_ps": float(cfg.reservoir.time_step_ps),
                    "cfg.reservoir.fiber_length_m": float(cfg.reservoir.fiber_length_m),
                },
            )

            nonlocal best
            if score < best["score"]:
                best["score"] = score
                best["res"] = res
                best["params"] = cfg

            return score

    env_name = os.getenv("MCF_STUDY_NAME")
    if env_name:
        study_name = env_name
    else:
        variant_str = str(base_cfg.variant)
        core_count = int(base_cfg.core_count)
        ts = datetime.now().strftime('%Y%m%d-%H%M')
        study_name = f"mcf_rc_{variant_str}_C{core_count}_{ts}"

    sampler = optuna.samplers.TPESampler(constant_liar=True, multivariate=True, group=True, seed=base_cfg.mask.seed)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=sampler
    )

    try:
        study_id = storage.get_study_id_from_name(study_name)
    except Exception:
        study_id = None
    print(f"[optimize] Target COMPLETE trials = {n_trials}")

    def _optuna_global_best_callback(study_: "optuna.study.Study", frozen_trial: "optuna.trial.FrozenTrial") -> None:
        tn = int(getattr(frozen_trial, "number", -1))
        if tn >= 0:
            _print_progress(tn)
        if getattr(frozen_trial, "state", None) != TrialState.COMPLETE:
            return
        _print_global_best_if_updated(study_)

    if os.getenv("MCF_BASH", ""):
        while True:
            if study_id is not None:
                complete_cnt = int(storage.get_n_trials(study_id, state=(TrialState.COMPLETE,)))
            else:
                complete_cnt = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
            if complete_cnt >= n_trials:
                break
            trial = study.ask()
            try:
                value = objective(trial)
                study.tell(trial, value)
                _print_global_best_if_updated(study)
                _print_progress(int(getattr(trial, "number", -1)))
            except TrialPruned:
                study.tell(trial, state=TrialState.PRUNED)
                _print_progress(int(getattr(trial, "number", -1)))
            except (AssertionError, ValueError, RuntimeError) as _e:
                study.tell(trial, state=TrialState.FAIL)
                _print_progress(int(getattr(trial, "number", -1)))
                continue
            except Exception as _e:
                study.tell(trial, state=TrialState.FAIL)
                _print_progress(int(getattr(trial, "number", -1)))
                continue
    else:
        print("[mode] Local study.optimize (no MCF_BASH)")
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
            catch=(AssertionError, ValueError, RuntimeError, Exception),
            callbacks=[_optuna_global_best_callback],
        )

    best_trial = study.best_trial

    return dict(
        best_cfg=best["res"]["cfg"],
        best_metrics=best["res"]["metrics"],
        best_trial_params=best["params"],
        best_val_nrmse=float(best["score"]),
        fiber_params=best["res"]["fiber_params"],
        optuna_best_params=best_trial.params,
        optuna_best_value=float(best_trial.value),
        study_name=study_name,
    )


def _update_plots_and_save(results_stream: list[dict],
                           base_cfg: ExperimentConfig,
                           *,
                           logx: bool = False,
                           logy: bool = False,
                           ts: str | None = None) -> dict:
    """
    Рисует три графика (NRMSE vs coupling / L_coupling / radius) и пишет CSV.
    Поведение и стиль согласованы с функциями отрисовки в модуле (Nature-стиль).

    Args:
        results_stream: список словарей с ключами:
            {"radius", "coupling", "nrmse_train", "nrmse_val", "nrmse_test"}.
        base_cfg: конфигурация эксперимента; variant и core_count используются в именах файлов.
        logx, logy: логарифмические шкалы по осям.
        ts: фиксированный таймстамп для серии файлов. Если None — создаётся новый.

    Returns:
        dict с путями до сохранённых файлов: {"p_c", "p_l", "p_r", "p_csv"}.
    """
    import csv
    from datetime import datetime
    from pathlib import Path

    save_figs_flag = bool(getattr(base_cfg.reservoir, "save_figs", False))
    out_dir = Path(__file__).parent
    variant_str = str(base_cfg.variant)
    core_count = int(base_cfg.core_count)
    fmt = str(plt.rcParams.get("savefig.format", "pdf")).lower()
    ts = ts or datetime.now().strftime('%Y%m%d-%H%M')

    p_c = out_dir / f"scan_nrmse_vs_coupling_{variant_str}_C{core_count}_{ts}.{fmt}"
    p_l = out_dir / f"scan_nrmse_vs_Lc_{variant_str}_C{core_count}_{ts}.{fmt}"
    p_r = out_dir / f"scan_nrmse_vs_radius_{variant_str}_C{core_count}_{ts}.{fmt}"
    p_csv = out_dir / f"scan_coupling_results_{variant_str}_C{core_count}_{ts}.csv"

    # валидные точки по coupling
    valid = [r for r in results_stream if np.isfinite(float(r.get("coupling", np.nan)))]
    if not valid:
        return {"p_c": None, "p_l": None, "p_r": None, "p_csv": None}

    coupl = np.array([float(d["coupling"]) for d in valid], float)
    Lc = np.where((coupl > 0) & np.isfinite(coupl), np.pi / (2.0 * coupl), np.nan)
    rad = np.array([float(d.get("radius", np.nan)) for d in valid], float)
    y_tr = np.array([float(d.get("nrmse_train", np.nan)) for d in valid], float)
    y_va = np.array([float(d.get("nrmse_val", np.nan)) for d in valid], float)

    def _sorted_xy(x_raw, y_raw):
        x = np.asarray(x_raw, float)
        y = np.asarray(y_raw, float)
        m = np.isfinite(x) & np.isfinite(y)
        if not np.any(m):
            return None, None
        x, y = x[m], y[m]
        order = np.argsort(x)
        return x[order], y[order]

    # — фигуры под Nature-стиль
    fig_c, ax_c = plt.subplots(figsize=(COL2, COL2 * 0.38), constrained_layout=True)
    fig_l, ax_l = plt.subplots(figsize=(COL2, COL2 * 0.38), constrained_layout=True)
    fig_r, ax_r = plt.subplots(figsize=(COL2, COL2 * 0.38), constrained_layout=True)

    # 1) NRMSE vs coupling
    ax_c.clear()
    xc_tr, yc_tr = _sorted_xy(coupl, y_tr)
    xc_va, yc_va = _sorted_xy(coupl, y_va)
    drew = False
    if xc_tr is not None:
        ax_c.plot(xc_tr, yc_tr,  # marker="o",
                  label="Train NRMSE")
        drew = True
    if xc_va is not None:
        ax_c.plot(xc_va, yc_va,  # marker="s",
                  linestyle="--", label="Val NRMSE")
        drew = True
    ax_c.set_xlabel("coupling coefficient, 1/m")
    ax_c.set_ylabel("NRMSE")
    # if logx:
    #     ax_c.set_xscale("log")
    if logy:
        ax_c.set_yscale("log")
    if drew:
        ax_c.legend()

    if save_figs_flag:
        try:
            fig_c.savefig(p_c)
        except Exception as e:
            print(f"[warn] savefig coupling failed: {e}")

    # 2) NRMSE vs L_coupling
    ax_l.clear()
    xl_tr, yl_tr = _sorted_xy(Lc, y_tr)
    xl_va, yl_va = _sorted_xy(Lc, y_va)
    drew = False
    if xl_tr is not None:
        ax_l.plot(xl_tr, yl_tr,  # marker="o",
                  label="Train NRMSE")
        drew = True
    if xl_va is not None:
        ax_l.plot(xl_va, yl_va,  # marker="s",
                  linestyle="--", label="Val NRMSE")
        drew = True
    ax_l.set_xlabel("coupling length, m")
    ax_l.set_ylabel("NRMSE")
    if logx:
        ax_l.set_xscale("log")
    if logy:
        ax_l.set_yscale("log")
    if drew:
        ax_l.legend()
    if save_figs_flag:
        try:
            fig_l.savefig(p_l)
        except Exception as e:
            print(f"[warn] savefig Lc failed: {e}")

    # 3) NRMSE vs radius
    ax_r.clear()
    xr_tr, yr_tr = _sorted_xy(rad, y_tr)
    xr_va, yr_va = _sorted_xy(rad, y_va)
    drew = False
    if xr_tr is not None:
        ax_r.plot(xr_tr, yr_tr,
                  # marker="o",
                  label="Train NRMSE")
        drew = True
    if xr_va is not None:
        ax_r.plot(xr_va, yr_va,
                  # marker="s",
                  linestyle="--", label="Val NRMSE")
        drew = True
    ax_r.set_xlabel("inter-core radius, µm")
    ax_r.set_ylabel("NRMSE")
    if logx:
        ax_r.set_xscale("log")
    if logy:
        ax_r.set_yscale("log")
    if drew:
        ax_r.legend()
    if save_figs_flag:
        try:
            fig_r.savefig(p_r)
        except Exception as e:
            print(f"[warn] savefig radius failed: {e}")

    # CSV: полный срез текущего results_stream
    if save_figs_flag:
        try:
            with open(p_csv, "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["radius", "coupling", "L_coupling",
                             "nrmse_train", "nrmse_val", "nrmse_test"])
                for d in results_stream:
                    c = float(d.get("coupling", float("nan")))
                    L = (np.pi / (2.0 * c)) if (np.isfinite(c) and c > 0) else float("nan")
                    wr.writerow([float(d.get("radius", float("nan"))),
                                 c, L,
                                 float(d.get("nrmse_train", float("nan"))),
                                 float(d.get("nrmse_val", float("nan"))),
                                 float(d.get("nrmse_test", float("nan")))])
        except Exception as e:
            print(f"[warn] write csv failed: {e}")

    return {"p_c": p_c, "p_l": p_l, "p_r": p_r, "p_csv": p_csv}


# --- helper: один радиус → один прогон (отдельный процесс) ---
def _scan_single_radius_task(args):
    """
    Вспомогательный воркер для multiprocessing.
    Принимает (radius, base_cfg, variant, force_rerun, save_cache) и возвращает
    словарь с radius, coupling, nrmse_train, nrmse_val, nrmse_test.
    """
    radius, base_cfg, variant, force_rerun, save_cache = args
    # try:
    # аккуратно клонируем конфиг и подставляем новый радиус второго слоя
    resv = base_cfg.reservoir
    lr = list(resv.layer_radii_array)
    if len(lr) < 2:
        # для 7-ядерного волокна нужен хотя бы 1 «кольцевой» слой
        return {"radius": float(radius), "coupling": float("nan"),
                "nrmse_train": float("nan"),
                "nrmse_val": float("nan"), "nrmse_test": float("nan")}
    lr[1] = float(radius)

    resv_silent = _dc.replace(
        resv,
        layer_radii_array=tuple(lr),
        num_threads=1,
        display_debug_plots=False,
        save_figs=False
    )
    cfg_i = _dc.replace(base_cfg, reservoir=resv_silent)

    # выбираем правильный сценарий
    res = run_experiments(cfg_i, n_trials_opt=0, free_run_horizon=0,
                          force_rerun=force_rerun, save_cache=save_cache)

    # фактический коэффициент связи через L_coupling (если есть)
    fp = res.get("fiber_params", {}) or {}
    Lc = float(fp.get("L_coupling", float("inf")))
    coupling = (np.pi / (2.0 * Lc)) if np.isfinite(Lc) and Lc > 0 else float("nan")

    m = res.get("metrics", {}) or {}

    return {
        "radius": float(radius),
        "coupling": float(coupling),
        "nrmse_train": float(m.get("nrmse_train", float("nan"))),
        "nrmse_val": float(m.get("nrmse_val", float("nan"))),
        "nrmse_test": float(m.get("nrmse_test", float("nan"))),
    }
    # except Exception as e:
    #     print(e)
    #     # в случае сбоя — безопасные NaN, чтобы не падала вся серия
    #     return {"radius": float(radius), "coupling": float("nan"),
    #             "nrmse_train": float("nan"),
    #             "nrmse_val": float("nan"), "nrmse_test": float("nan")}


def find_best_coupling_coefficient(radii: list,
                                   variant: str,
                                   base_cfg: ExperimentConfig,
                                   force_rerun=False,
                                   save_cache=False,
                                   merge_with_existing=False,
                                   *,
                                   n_jobs: int | None = None,
                                   logx: bool = False,
                                   logy: bool = False):
    """
    Перебирает значения layer_radii_array[1] (в микронах) для 7-ядерного волокна, запускает расчёт
    в нескольких процессах и ведёт журнал: train/val NRMSE vs коэффициент/длина связи и радиус.
    После каждого полученного результата обновляет CSV и три рисунка через _update_plots_and_save.

    Args:
        merge_with_existing: при True подмешивает строки из ранее сохранённых CSV (если найдены).
    """
    from fiberprop.parallel_runtime import mp_initializer
    from multiprocessing import get_context
    import psutil
    import csv
    from datetime import datetime
    from pathlib import Path

    # задания для пула
    tasks = [(float(r), base_cfg, variant, bool(force_rerun), bool(save_cache)) for r in radii]

    # число процессов
    if n_jobs is None or n_jobs <= 0:
        n_jobs = max(1, psutil.cpu_count(logical=False) or (psutil.cpu_count(logical=True) or 1))

    ctx = get_context("spawn")

    # флаги визуализации/сохранения
    save_figs_flag = bool(getattr(base_cfg.reservoir, "save_figs", False))

    # единый timestamp для всей серии (чтобы перезаписывать один и тот же набор файлов)
    ts = datetime.now().strftime('%Y%m%d-%H%M')

    # подготовка путей/фильтрации существующих CSV
    out_dir = Path(__file__).parent
    variant_str = str(base_cfg.variant)
    core_count = int(base_cfg.core_count)

    existing_results = []
    if merge_with_existing and save_figs_flag:
        pattern = f"scan_coupling_results_*.csv"
        existing_files = list(out_dir.glob(pattern))
        for csv_file in existing_files:
            try:
                with open(csv_file, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        conv = {}
                        for k, v in row.items():
                            try:
                                conv[k] = float(v) if str(v).strip() != '' else float('nan')
                            except ValueError:
                                conv[k] = v
                        existing_results.append(conv)
                print(f"Загружено {len(existing_results)} результатов из {csv_file.name}")
            except Exception as e:
                print(f"Ошибка при чтении {csv_file}: {e}")

    results_stream = existing_results.copy()

    # пул процессов
    with temporary_thread_limits(1):
        with ctx.Pool(processes=n_jobs, initializer=mp_initializer, initargs=(None, True)) as pool:
            for r in pool.imap_unordered(_scan_single_radius_task, tasks, chunksize=1):
                results_stream.append(r)
                print(r)
                if save_figs_flag:
                    _update_plots_and_save(results_stream, base_cfg, logx=logx, logy=logy, ts=ts)

    # итоговая фильтрация и финальная отрисовка
    results = [r for r in results_stream if np.isfinite(r.get("coupling", np.nan))
               and abs(r.get("nrmse_val") - r.get("nrmse_train")) < 0.1]
    if not results:
        print("find_best_coupling_coefficient: все прогоны вернули нечисловой coupling — пропуск.")
        return

    _update_plots_and_save(results, base_cfg, logx=logx, logy=logy, ts=ts)


def plot_combined_csv_results(base_cfg, logx: bool = False, logy: bool = False, save_figs: bool = True):
    """
    Находит все CSV-файлы с результатами сканирования в текущей папке,
    объединяет данные и строит графики NRMSE vs coupling / L_coupling / radius
    в макетном стиле (Nature, двухколоночная ширина).

    Args:
        base_cfg: конфигурация эксперимента (variant, core_count)
        logx: логарифмическая шкала по X
        logy: логарифмическая шкала по Y
        save_figs: сохранять ли графики в файлы (формат берётся из rcParams)
    """
    import pandas as pd

    variant_str = str(base_cfg.variant)
    core_count = int(base_cfg.core_count)
    out_dir = Path(__file__).parent

    # собираем все CSV нужного вида
    pattern = f"scan_coupling_results_{variant_str}_C{core_count}_*.csv"
    csv_files = list(out_dir.glob(pattern))
    if not csv_files:
        print(f"Не найдено CSV по шаблону: {pattern}")
        return
    print(f"Найдено CSV: {len(csv_files)}")

    # объединяем
    all_df = []
    for p in csv_files:
        try:
            df = pd.read_csv(p)
            df["source_file"] = p.name
            all_df.append(df)
            print(f"Загружено {len(df)} строк из {p.name}")
        except Exception as e:
            print(f"Ошибка чтения {p}: {e}")
    if not all_df:
        print("Нет данных для объединения")
        return

    combined_df = pd.concat(all_df, ignore_index=True)

    # удаляем дубликаты по всем столбцам, кроме 'source_file'
    cols = [c for c in combined_df.columns if c != "source_file"]
    combined_df = combined_df.drop_duplicates(subset=cols)
    print(f"Уникальных точек: {len(combined_df)}")

    # фильтруем валидные строки
    if "coupling" not in combined_df.columns or "radius" not in combined_df.columns:
        print("В CSV нет обязательных колонок: 'coupling' / 'radius'")
        return
    valid = combined_df[np.isfinite(combined_df["coupling"])]
    if len(valid) == 0:
        print("Нет валидных данных для построения графиков")
        return

    # гарантируем наличие L_coupling_m
    if "L_coupling_m" not in valid.columns:
        c = valid["coupling"].to_numpy(float)
        Lc = np.where((c > 0) & np.isfinite(c), np.pi / (2.0 * c), np.nan)
        valid = valid.assign(L_coupling_m=Lc)

    # извлекаем массивы
    coupl = valid["coupling"].to_numpy(float)
    Lc = valid["L_coupling_m"].to_numpy(float)
    rad = valid["radius"].to_numpy(float)
    y_tr = valid.get("nrmse_train", pd.Series(np.nan, index=valid.index)).to_numpy(float)
    y_va = valid.get("nrmse_val", pd.Series(np.nan, index=valid.index)).to_numpy(float)

    def _sorted_xy(x_raw, y_raw):
        x = np.asarray(x_raw, float)
        y = np.asarray(y_raw, float)
        m = np.isfinite(x) & np.isfinite(y)
        if not np.any(m):
            return None, None
        x, y = x[m], y[m]
        order = np.argsort(x)
        return x[order], y[order]

    # имена файлов вывода
    fmt = str(plt.rcParams.get("savefig.format", "pdf")).lower()
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    p_c = out_dir / f"combined_nrmse_vs_coupling_{variant_str}_C{core_count}_{ts}.{fmt}"
    p_l = out_dir / f"combined_nrmse_vs_Lc_{variant_str}_C{core_count}_{ts}.{fmt}"
    p_r = out_dir / f"combined_nrmse_vs_radius_{variant_str}_C{core_count}_{ts}.{fmt}"
    p_csv = out_dir / f"combined_results_{variant_str}_C{core_count}_{ts}.csv"

    # фигуры в стиле Nature: COL2 × 0.38, constrained_layout
    fig_c, ax_c = plt.subplots(figsize=(COL2, COL2 * 0.38), constrained_layout=True)
    fig_l, ax_l = plt.subplots(figsize=(COL2, COL2 * 0.38), constrained_layout=True)
    fig_r, ax_r = plt.subplots(figsize=(COL2, COL2 * 0.38), constrained_layout=True)

    # --- 1) NRMSE vs coupling
    xc_tr, yc_tr = _sorted_xy(coupl, y_tr)
    xc_va, yc_va = _sorted_xy(coupl, y_va)
    drew = False
    ax_c.cla()
    if xc_tr is not None:
        ax_c.plot(xc_tr, yc_tr, marker="o", label="Train NRMSE")
        drew = True
    if xc_va is not None:
        ax_c.plot(xc_va, yc_va, marker="s", linestyle="--", label="Val NRMSE")
        drew = True
    ax_c.set_xlabel("coupling coefficient, 1/m")
    ax_c.set_ylabel("NRMSE")
    # if logx:
    #     ax_c.set_xscale("log")
    if logy:
        ax_c.set_yscale("log")
    if drew:
        ax_c.legend()
    if save_figs:
        try:
            fig_c.savefig(p_c)
        except Exception as e:
            print(f"[warn] savefig coupling failed: {e}")

    # --- 2) NRMSE vs L_coupling
    xl_tr, yl_tr = _sorted_xy(Lc, y_tr)
    xl_va, yl_va = _sorted_xy(Lc, y_va)
    drew = False
    ax_l.cla()
    if xl_tr is not None:
        ax_l.plot(xl_tr, yl_tr, marker="o", label="Train NRMSE")
        drew = True
    if xl_va is not None:
        ax_l.plot(xl_va, yl_va, marker="s", linestyle="--", label="Val NRMSE")
        drew = True
    ax_l.set_xlabel("coupling length, m")
    ax_l.set_ylabel("NRMSE")
    if logx:
        ax_l.set_xscale("log")
    if logy:
        ax_l.set_yscale("log")
    if drew:
        ax_l.legend()
    if save_figs:
        try:
            fig_l.savefig(p_l)
        except Exception as e:
            print(f"[warn] savefig L_coupling failed: {e}")

    # --- 3) NRMSE vs radius
    xr_tr, yr_tr = _sorted_xy(rad, y_tr)
    xr_va, yr_va = _sorted_xy(rad, y_va)
    drew = False
    ax_r.cla()
    if xr_tr is not None:
        ax_r.plot(xr_tr, yr_tr, marker="o", label="Train NRMSE")
        drew = True
    if xr_va is not None:
        ax_r.plot(xr_va, yr_va, marker="s", linestyle="--", label="Val NRMSE")
        drew = True
    ax_r.set_xlabel("inter-core radius, µm")
    ax_r.set_ylabel("NRMSE")
    if logx:
        ax_r.set_xscale("log")
    if logy:
        ax_r.set_yscale("log")
    if drew:
        ax_r.legend()
    if save_figs:
        try:
            fig_r.savefig(p_r)
        except Exception as e:
            print(f"[warn] savefig radius failed: {e}")

    # экспорт объединённых данных (полезно для отслеживания)
    if save_figs:
        try:
            combined_df.to_csv(p_csv, index=False)
        except Exception as e:
            print(f"[warn] write combined csv failed: {e}")


def run_spatial_only_radii_optuna(base_cfg: ExperimentConfig,
                                  radii: list[float] | tuple[float, ...],
                                  *,
                                  n_trials_opt: int = 100,
                                  free_run_horizon: int = 20000,
                                  force_rerun: bool = False,
                                  save_cache: bool = False,
                                  logx: bool = False,
                                  logy: bool = False,
                                  param_space: dict | None = None):
    """
    Запускает серию оптимизаций `spatial_only` по списку радиусов (µm).
    На КАЖДЫЙ радиус выполняется РОВНО ОДНА оптимизация через optimize_hyperparams.
    После каждой оптимизации обновляет CSV и три рисунка через _update_plots_and_save.

    Параметры:
        base_cfg: базовая конфигурация эксперимента.
        radii: список радиусов (µm), которые по очереди подставляются в layer_radii_array[1].
        n_trials_opt: число Optuna-trials на один радиус.
        free_run_horizon, force_rerun, save_cache: параметры исполнения.
        logx, logy: логарифмические шкалы по осям для графиков.
        param_space: пространство гиперпараметров (см. формат в optimize_hyperparams).

    Возвращает:
        {
          "results_stream": [...],
          "paths": {"p_c", "p_l", "p_r", "p_csv"}
        }
    """
    # --- локальные импорты (как в других функциях) ---------------------------------------
    import csv
    import numpy as np
    import dataclasses as _dc
    from datetime import datetime
    from pathlib import Path
    from typing import TYPE_CHECKING, cast  # ← добавлено

    if TYPE_CHECKING:
        # Доступно только для тайпчекера; в рантайме _typeshed отсутствует
        from _typeshed import SupportsWrite  # noqa: F401  # ← добавлено

    # --- подготовка -----------------------------------------------------------------------
    cfg0 = _dc.replace(base_cfg, variant="spatial_only")

    results_stream: list[dict] = []
    ts_common = datetime.now().strftime('%Y%m%d-%H%M')

    # файл для подробных параметров (только best_metrics + fiber_params)
    out_dir = Path(__file__).parent
    variant_str = str(base_cfg.variant)
    core_count = int(base_cfg.core_count)
    p_opt_csv = out_dir / f"scan_optima_params_{variant_str}_C{core_count}_{ts_common}.csv"

    opt_headers_written = False
    opt_fieldnames: list[str] = []
    if p_opt_csv.exists():
        try:
            with open(p_opt_csv, "r", newline="") as f_in:
                rdr = csv.DictReader(f_in)
                if rdr.fieldnames:
                    opt_fieldnames = list(rdr.fieldnames)
                    opt_headers_written = True
        except Exception:
            opt_headers_written = False
            opt_fieldnames = []

    paths = {}
    # --- основной цикл по радиусам --------------------------------------------------------
    for i, r_um in enumerate(radii, 1):
        print(f"\n=== [{i}/{len(radii)}] radius={float(r_um):.6f} µm → Optuna trials={int(n_trials_opt)} ===")

        # подставляем радиус второго слоя
        lr = list(cfg0.reservoir.layer_radii_array)
        if len(lr) < 2:
            raise ValueError("Ожидается хотя бы один кольцевой слой: layer_radii_array длиной >= 2")
        lr[1] = float(r_um)

        cfg_i = _dc.replace(
            cfg0,
            reservoir=_dc.replace(
                cfg0.reservoir,
                layer_radii_array=tuple(lr),
                save_figs=getattr(cfg0.reservoir, "save_figs", False)
            )
        )

        # === РОВНО ОДНА оптимизация на радиус ===
        res = optimize_hyperparams(
            base_cfg=cfg_i,
            n_trials=n_trials_opt,
            # n_jobs=1,
            free_run_horizon=free_run_horizon,
            force_rerun=force_rerun,
            save_cache=save_cache,
            param_space=param_space
        )

        # достаём лучшее из результата оптимизации
        m = res.get("best_metrics") or {}
        fiber_params = res.get("fiber_params") or res.get("best_metrics_fiber_params") or {}

        # оценим связь: k = π / (2 L_coupling)
        L_coupling = float(fiber_params.get("L_coupling", np.nan))
        coupling = (np.pi / (2.0 * L_coupling)) if np.isfinite(L_coupling) and L_coupling > 0 else np.nan

        # поток результатов для графиков/агрегированного CSV
        results_stream.append({
            "radius": float(r_um),
            "coupling": float(coupling) if np.isfinite(coupling) else np.nan,
            "nrmse_train": float(m.get("nrmse_train", np.nan)),
            "nrmse_val": float(m.get("nrmse_val", np.nan)),
            "nrmse_test": float(m.get("nrmse_test", np.nan)),
        })

        paths = _update_plots_and_save(results_stream, base_cfg, logx=logx, logy=logy, ts=ts_common)

        # --- CSV: только поля из best_metrics и fiber_params (+ radius) ----------------
        # Заголовок фиксируем по первой записи в этой серии
        row = {"radius": float(r_um)}
        for k, v in m.items():
            row[str(k)] = v
        for k, v in fiber_params.items():
            row[str(k)] = v

        if not opt_headers_written:
            opt_fieldnames = list(row.keys())
            try:
                with open(p_opt_csv, "w", newline="") as f_out:
                    # cast со строковым типом: безопасно в рантайме и устраивает тайпчекеры
                    w = csv.DictWriter(cast("SupportsWrite[str]", f_out), fieldnames=opt_fieldnames)  # ← изменено
                    w.writeheader()
                    w.writerow(row)
                opt_headers_written = True
            except Exception as e:
                print(f"[warn] write optima csv header failed: {e}")
        else:
            try:
                with open(p_opt_csv, "a", newline="") as f_out:
                    w = csv.DictWriter(cast("SupportsWrite[str]", f_out), fieldnames=opt_fieldnames)  # ← изменено
                    aligned = {k: row.get(k, "") for k in opt_fieldnames}
                    w.writerow(aligned)
            except Exception as e:
                print(f"[warn] append optima csv failed: {e}")

    print("\nГотово.")
    return {"results_stream": results_stream, "paths": paths}


def _per_core(params, key: str, default: float, core_count: int):
    v = params.get(key, default)
    if isinstance(v, (list, tuple, np.ndarray)):  # строк тут быть не должно
        if len(v) == core_count:
            return tuple(v)
        if len(v) == 1:
            return (v[0],) * core_count
        raise ValueError(f"{key}: expected len 1 or {core_count}, got {len(v)}")
    return (v,) * core_count

# =========================
# Пример точки входа
# =========================

if __name__ == "__main__":

    # Базовые параметры
    force_rerun = False  # игнорировать кэш и считать заново
    save_cache = False

    ########### Parameters for the best regimes #############

    params = {}



    #####################################################

    layer_count = params.get("layer_count", 1)
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
            layer_radii_array[i] = 30  # 17.3 * 1  # 17.3 # [mkm]
        if i == 2:
            layer_radii_array[i] = 30 * 2  # [mkm]
        if i == 3:
            layer_radii_array[i] = 30 * 3  # [mkm]

        # layer_radii_array[i] = 17.3 * (i * 1.5) # [mkm]

    if "ppump" in params:
        pp = params["ppump"]
        if np.isscalar(pp):
            p_pump_tuple = (float(pp),) * int(core_count)
        else:
            p_pump_tuple = tuple(float(p) for p in pp)

        g0_t, psat_t = calc_g0_psat_from_p_pump_josab(
            p_pump_tuple,
            float(params.get("fiber_length_m", 0.1)),
            alpha=0,
        )

        params["g0"] = tuple(float(x) for x in np.asarray(g0_t).reshape(-1))
        params["psat"] = tuple(float(x) for x in np.asarray(psat_t).reshape(-1))

    temporal_modulation_frequency_ghz = params.get("temporal_modulation_frequency_ghz", 0.4)  # GHz

    variant = params.get("variant", "spatial_only")  # "spatial_only" "temporal_same_all_cores" "temporal_unique_per_core"

    if variant == "temporal_same_all_cores" or variant == "temporal_unique_per_core":
        temporal_mask_size = params.get("mask_size", 294)
    else:
        temporal_mask_size = 1

    mg_cfg = MGConfig(t_size=2 ** 10, tau=17, n=10, beta=0.2, gamma=0.1, initial_condition=1.2, dt=1.0)

    mask_cfg = MaskConfig(mask_size=temporal_mask_size, mask_kind="uniform", seed=42,
                          gain_in=params.get("gain_in", 30.96))

    reservoir_cfg = ReservoirConfig(
        fiber_length_m=params.get("fiber_length_m", 0.3),  # 0.1
        time_step_ps=params.get("time_step_ps", 1.0 / temporal_modulation_frequency_ghz * 1e+3),
        step_number_per_dimensionless_distance=20,
        upsampling=1,
        delay_factor_in_symbols=params.get("delay_factor_in_symbols", 1),
        delay_additional_in_mask_steps=0,
        layer_count=layer_count,
        layer_radii_array=layer_radii_array,
        g0_array=_per_core(params, "g0", 0.0158, core_count),
        psat_array=_per_core(params, "psat", 1.1054e-05, core_count),
        kappa=params.get("kappa", 0.793),  # 0.9
        delta_phase=params.get("delta_phase", 0),
        use_gpu=False,
        use_torch=False,
        num_threads=1,
        display_debug_plots=True,
        display_debug_info=False,
        save_figs=False,
        max_hours_total=24,
        precision='float64',
        use_dispersion=False,
        ase_noise_seed=params.get("ase_noise_seed", mask_cfg.seed), # включает шум в модели
        ase_noise_nf_db=params.get("ase_noise_nf_db", 50),
        ase_noise_optical_bw_hz=params.get("ase_noise_optical_bw_hz", 12.5e9),
        ase_noise_compute_effective_osnr=params.get("ase_noise_compute_effective_osnr", True), # nf_db=60 dB => OSNR=21 dB,
    )

    training_cfg = TrainingConfig(feature_mode="intensity", taps=1, ridge_alpha=0, washout=500,
                                  target_shift=1,
                                  train_frac=0.6, val_frac=0.2)  # для одиночного запуска

    base_cfg = ExperimentConfig(core_count=core_count, mg=mg_cfg, mask=mask_cfg,
                                reservoir=reservoir_cfg, training=training_cfg,
                                variant=variant)

    mg_cfg.t_size = 20000  # np.min([int(np.ceil(t_size / 500.0)) * 500, 5000])

    t = time()

    # ================== DEBUG: gain saturation curve at current operating point ==================
    # data_in_dbg, _mg = (None, None)
    # if variant == "temporal_unique_per_core":
    #     data_in_dbg, _mg = generate_input_temporal_unique_per_core(core_count, mg_cfg, mask_cfg)
    # elif variant == "temporal_same_all_cores":
    #     data_in_dbg, _mg = generate_input_temporal_same_all_cores(core_count, mg_cfg, mask_cfg)
    # elif variant == "spatial_only":
    #     data_in_dbg, _mg = generate_input_spatial_only(core_count, mg_cfg, mask_cfg)
    # else:
    #     raise ValueError(f"Unknown variant: {variant}")
    #
    # plot_laser_gain_curve_from_params(
    #         params,
    #         data_in_dbg,
    #         core_count=core_count,
    #         curve_kind="signal",
    #         core_index=None,          # по умолчанию центральное ядро
    #         plot_all_cores=True,     # True, если хотите наложить кривые всех ядер
    #         show_plot=True,
    #         save_plot=False,
    #     )
    #
    # # ============= Пример: одиночный прогон с сохранением артефактов и pseudo free-run ==================
    res = run_experiments(base_cfg, force_rerun=force_rerun, save_cache=save_cache)

    # исследование параметра регуляризации
    # sweep = ridge_alpha_sweep_diagnostics(
    #     res,
    #     title="Current main params",
    # )
    #
    # ridge_alpha_selection_compare_diagnostics(
    #     res,
    #     alphas=sweep["alpha_grid"],
    #     title="Current main params",
    #     selection_rule="1se_largest",
    #     normalize_selection_scores=False,
    # )

    if base_cfg.reservoir.ase_noise_seed is not None and base_cfg.reservoir.ase_noise_compute_effective_osnr:
        print("effective_output_osnr_test_db =", res["metrics"]["effective_output_osnr_test_db"])
    print("Val/Test NRMSE:", res["metrics"]["nrmse_val"], res["metrics"]["nrmse_test"])

    # =============== Optuna: поиск по κ, g0, Psat, L_fiber и gain_in (лог по мощности) ===================

    # Вариант «temporal_same_all_cores» для одной сердцевины;
    # окно задержки и размер маски тоже можно подстроить.
    # Для запуска optuna dashboard установи
    # pip install optuna-dashboard
    # и выполни в отдельной консоли после запуска расчета (путь укажи свой до папки проекта)
    # optuna-dashboard "C:/Users/Igor/YandexDisk/Code/Photonics/fiberprop/scripts/mcf_reservori_computing/mcf_optuna.journal" --port 8080
    # Открой в браузере http://127.0.0.1:8080/dashboard/
    # На линуксе порт 18080, поэтому
    # открой в браузере http://127.0.0.1:18080/dashboard

    # !!! в статье формула 10 аппроксимирует волокно с потерями 0.75 1/m. А у нас потерь нет!
    # test_calc_formula10_and_plot_fig7b(
    #     calc_g0_psat_func=calc_g0_psat_from_p_pump_josab,
    #     g0_to_gA_factor=1,  # или 1.0 — в зависимости от того, что именно возвращает твой g0
    #     p_pump_min_mw=25.0,
    #     p_pump_max_mw=210.0,
    #     n_grid=500,
    #     title_suffix="",
    #     show_psat_fig7a=True,
    # )

    # --- Bounds derived from pump->(g0, Psat) fits (Shtyrina et al., JOSAB 34(2), 227 (2017), doi:10.1364/JOSAB.34.000227) ---
    # Psat[mW] = 0.1292 * Ppump[mW] - 1.096  =>  Psat > 0 при Ppump > ~8.48 mW
    p_pump_min_w_for_positive_psat = _round((1.096 / 0.1292) * 1e-3 * 0.7, digit=3)  # W

    # Оценка верхних границ g0 и psat (на одно ядро) при заданной длине.
    # Подстрой p_pump_max_single_core_w под доступный (или планируемый) pump laser.
    p_pump_max_single_core_w = 3  # W
    fiber_length_for_bounds_m = float(params.get("fiber_length_m", 0.3))
    try:
        _g0_b, _psat_b = calc_g0_psat_from_p_pump_josab(p_pump_max_single_core_w, fiber_length_for_bounds_m, alpha=0)
        g0_max_bound = float(np.asarray(_g0_b).reshape(-1)[0])
        psat_max_bound = float(np.asarray(_psat_b).reshape(-1)[0])
    except Exception:
        g0_max_bound, psat_max_bound = None, None

    print(f"[pump->gain] Psat>0 requires Ppump > {p_pump_min_w_for_positive_psat * 1e3:.2f} mW per core.")
    if g0_max_bound is not None and psat_max_bound is not None:
        print(
            f"[pump->gain] Upper bound estimate (single core): g0~{g0_max_bound:.3g} 1/m, Psat~{psat_max_bound:.3g} W")

    # SLM для накачки: оптимизируем общую мощность pump laser и «веса» распределения по ядрам.
    # pump_slm_w_spec = [{"low": -3.0, "high": 3.0} for _ in range(core_count)]
    #
    # param_space = {
    #     "kappa": {"low": 0.2, "high": 0.99, "sig": 5},
    #     "delta_phase": 0,  # {"low": 0.0, "high": 2 * np.pi, "sig": 5},
    #
    #     # Pump-laser total power (W) + SLM weights -> per-core pump powers via softmax.
    #     "p_pump_total": {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 5},
    #     "pump_slm_w": pump_slm_w_spec,
    #
    #     # Если хочешь вернуться к независимой оптимизации ppump по ядрам — используй ключ "ppump" вместо двух выше.
    #     # "ppump": [{"low": 0.01, "high": 1.0, "log": True} for _ in range(core_count)],
    #
    #     "fiber_length_m": {"low": 0.001, "high": 10, "log": True, "sig": 5},
    #     "gain_in": {"low": 0.001, "high": 0.1, "log": True, "sig": 5},
    #     "mask_size": {"int": True, "low": 5, "high": 300, "step": 1},
    #     "delay_factor_in_symbols": {"int": True, "low": 1, "high": 50},
    #     "time_step_ps": reservoir_cfg.time_step_ps,
    #     # {"int": True, "low": 0.01 * 1e+3, "high": 100 * 1e+3, "log": True, "sig": 5},
    # }

    # param_space = {
    #     "kappa": {"low": 0.5, "high": 0.99, "sig": 4},
    #     "delta_phase": {"low": 0, "high": 2 * np.pi, "sig": 4},
    #     # "ppump": {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 4},
    #     "ppump": [
    #         {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 0
    #         {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 1
    #         {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 2
    #         {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 3
    #         {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 4
    #         {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 5
    #         {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 6
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 7
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 8
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 9
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 10
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 11
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 12
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 13
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 14
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 15
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 16
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 17
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": False, "sig": 3},  # ядро 18
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 4},  # ядро 19
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 4},  # ядро 20
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 4},  # ядро 21
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 4},  # ядро 22
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 4},  # ядро 23
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 4},  # ядро 24
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 4},  # ядро 25
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 4},  # ядро 26
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 4},  # ядро 27
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 4},  # ядро 28
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 4},  # ядро 29
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 4},  # ядро 30
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 4},  # ядро 31
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 4},  # ядро 32
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 4},  # ядро 33
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 4},  # ядро 34
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 4},  # ядро 35
    #         # {"low": p_pump_min_w_for_positive_psat, "high": p_pump_max_single_core_w, "log": True, "sig": 4},  # ядро 36
    #     ],
    #     "fiber_length_m": {"low": 0.01, "high": 0.5, "log": False, "sig": 3},
    #     "gain_in": {"low": 0.0001, "high": 1, "log": False, "sig": 4},
    #     "mask_size": 1, #{"int": True, "low": 5, "high": 350, "step": 1},
    #     "delay_factor_in_symbols": {"int": True, "low": 1, "high": 1},
    #     "time_step_ps": reservoir_cfg.time_step_ps,
    #     # {"int": True, "low": 0.01 * 1e+3, "high": 100 * 1e+3, "log": True},
    # }
    #
    # base_cfg.training.train_frac = 0.6
    # base_cfg.training.val_frac = 0.2
    # res = run_experiments(
    #     base_cfg,
    #     n_trials_opt=1000000,  # сколько попробовать конфигураций
    #     param_space=param_space,
    #     force_rerun=False,  # используй кэш, если ключ совпал
    #     save_cache=False,
    # )
    # print("Лучший результат Optuna:\n", res)

    # ==================== Зависимость NMRSE от коэффициента связи при наличии временной маски =========================

    # n_jobs = 5
    #
    # part1 = np.linspace(15, 30, n_jobs * 50, endpoint=False)
    # part2 = np.linspace(30, 60, n_jobs * 100, endpoint=True)
    # radii = np.concatenate((part1, part2))
    # np.random.shuffle(radii)
    #
    # find_best_coupling_coefficient([], # radii.tolist(),
    #                                merge_with_existing=True,
    #                                variant=variant, base_cfg=base_cfg, n_jobs=n_jobs, logx=True, logy=True)

    # ====================== Зависимость NMRSE от коэффициента связи без временной маски с оптимизацией ================

    # part1 = np.linspace(20, 30, 5, endpoint=False)
    # part2 = np.linspace(30, 60, 5, endpoint=True)
    # radii = np.concatenate((part1, part2))[::-1]
    # # np.random.shuffle(radii)
    #
    # param_space = {
    #     "kappa": {"low": 0.2, "high": 0.99},
    #     "delta_phase": {"low": 0.0, "high": 2 * np.pi},
    #     "g0": {"low": 0.001, "high": g0_max "log": True},
    #     "psat": {"low": 1e-5, "high": 1, "log": True},  # 1e-4  # фиксированное, не тюним
    #     "fiber_length_m": {"low": 0.01, "high": 1, "log": True},
    #     "gain_in": {"low": 1.0, "high": 100.0, "log": True},
    #     "mask_size": {"int": True, "low": 5, "high": 300, "step": 1},
    #     "delay_factor_in_symbols": {"int": True, "low": 1, "high": 50},
    #     # "time_step_ps": {"int": True, "low": 0.01 * 1e+3, "high": 100 * 1e+3, "log": True},
    # }
    #
    # run_spatial_only_radii_optuna(base_cfg, radii.tolist(), n_trials_opt=10, force_rerun=False, save_cache=False,
    #                               logx=True, logy=True,
    #                               param_space=param_space)

    print("Total elapsed time =", time() - t)
