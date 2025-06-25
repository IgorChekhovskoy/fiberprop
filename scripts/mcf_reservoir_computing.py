from fiberprop.solver import ComputationalParameters, EquationParameters, Solver, print_matrix
from fiberprop.fiber import Fiber, FiberMaterial, CoreConfig
from fiberprop.fiber_geometry import get_core_count
from fiberprop.light import Light
from fiberprop.base_functions import get_coupling_coefficients
from fiberprop.drawing import *
from time import time
from tqdm import trange
import numpy as np
from fiberprop.pulses import fundamental_soliton


def mackey_glass(t_size, tau=17, n=10, beta=2, gamma=1, initial_condition=1.2):
    t = np.zeros(t_size)
    t[0] = initial_condition
    for i in range(1, t_size):
        if i - tau < 0:
            t[i] = t[i-1] + (beta * t[i-1]**n) / (1 + t[i-1]**n) - gamma * t[i-1]
        else:
            t[i] = t[i-1] + (beta * t[i-1]**n) / (1 + t[i-1]**n) - gamma * t[i-1] + (beta * t[i-int(tau)]**n) / (1 + t[i-int(tau)]**n) - gamma * t[i-int(tau)]
    return t


def create_masks(eq_size, t_size, seed):
    np.random.seed(seed)
    masks = np.random.uniform(0, 1, (eq_size, t_size))
    return masks


def mackey_glass_masked(eq_size, t_size, seed, **mg_params):
    mg_series = mackey_glass(t_size, **mg_params)
    masks = create_masks(eq_size, t_size, seed)
    initial_conditions = masks * mg_series
    return initial_conditions


def generate_fundamental_soliton_train(eq_size,
                                       point_count,
                                       time_step_ps,
                                       beta2_ps2_m,
                                       gamma_1_w_m,
                                       pulse_width_ps=1.0,
                                       train_spacing_ps=None):
    """
    Creates an initial array shaped (eq_size, point_count) that contains a
    train of identical fundamental NLS solitons for the given β₂ and γ.

    The soliton is evaluated only on a local window ±3·train_spacing_ps
    to avoid overflow inside cosh(), then copied along the whole record.

    Parameters
    ----------
    eq_size : int
        Number of cores (rows) in the output array.
    point_count : int
        Number of temporal samples (columns).
    time_step_ps : float
        Time step between points, picoseconds.
    beta2_ps2_m : float
        Group-velocity dispersion β₂, ps²/m.
    gamma_1_w_m : float
        Non-linear coefficient γ, 1/(W·m).
    pulse_width_ps : float, optional
        Soliton width T₀ in picoseconds.  Default is 1 ps.
    train_spacing_ps : float or None, optional
        Time separation between neighbouring solitons.  Defaults to
        10 × pulse_width_ps.

    Returns
    -------
    np.ndarray
        Complex array with shape (eq_size, point_count) that can be used
        as `data_in`.
    """
    if train_spacing_ps is None:
        train_spacing_ps = 10.0 * pulse_width_ps

    # ---------- time axes -------------------------------------------------
    global_time_ps = (np.arange(point_count) - point_count / 2) * time_step_ps
    spacing_pts = int(round(train_spacing_ps / time_step_ps))

    # ---------- single-pulse window (±3 spacing) --------------------------
    window_half_ps = 5 * train_spacing_ps
    window_half_pts = int(round(window_half_ps / time_step_ps))
    local_time_ps = np.arange(-window_half_pts, window_half_pts + 1) * time_step_ps

    lamb_inv_ps = 1.0 / pulse_width_ps                      # λ = 1/T₀
    amplitude_w_sqrt = np.sqrt(abs(beta2_ps2_m) / gamma_1_w_m) * lamb_inv_ps

    print()
    print("soliton amplitude =", amplitude_w_sqrt)
    print("soliton pulse_width_ps =", pulse_width_ps)
    print("soliton train_spacing_ps =", train_spacing_ps)

    single_pulse = fundamental_soliton(local_time_ps,
                                       z=0.0,
                                       lamb=lamb_inv_ps,
                                       beta2=beta2_ps2_m,
                                       gamma=gamma_1_w_m)

    # ---------- train construction ---------------------------------------
    pulse_train = np.zeros(point_count, dtype=complex)
    centre_index = point_count // 2
    max_k = int((point_count // 2) / spacing_pts) + 1

    for k in range(-max_k, max_k + 1):
        centre = centre_index + k * spacing_pts
        start = centre - window_half_pts
        end = centre + window_half_pts + 1
        if start < 0 or end > point_count:
            continue  # skip incomplete pulses at the edges
        pulse_train[start:end] += single_pulse

    # ---------- duplicate for all cores ----------------------------------
    return np.tile(pulse_train, (eq_size, 1))


# ─────────────────────────────────────────────────────────────────────────────
def compute_characteristic_lengths(beta2_ps2_m: float,
                                 gamma_1_w_m: float,
                                 coupling_coefficient: float,
                                 data_in: np.ndarray,
                                 time_step_ps: float,
                                 central_core_ind: int = 0,
                                 g0_array=(),
                                 psat_array=(),
                                 display_debug_info: bool = False):
    """
    Оценивает ширину импульса через FWHM и печатает L_D, L_NL, L_G, L_sat.

    Параметры
    ----------
    beta2_ps2_m : float
        Групповая дисперсия β₂, [ps²/m].
    gamma_1_w_m : float
        Нелинейный коэффициент γ, [1/(W·m)].
    coupling_coefficient : float
        Коэффициент связи C, [1/m].
    data_in : np.ndarray
        Комплексный массив √W (C × M).
    time_step_ps : float
        Шаг временной сетки, [ps].
    central_core_ind : int, optional
        Сердцевина, по которой оцениваем импульс (по-умолчанию 0).
    g0_array, psat_array : array-like, optional
        g₀ [1/m] и Psat/Esat для оценки усилительных длин.
    display_debug_info : bool, optional
        True → печатать результаты.
    """
    if not display_debug_info:
        return

    power = np.abs(data_in[central_core_ind]) ** 2            # [W]
    P_peak = power.max() if power.size else 0.0
    idx_peak = power.argmax()

    # ---------- FWHM -----------------------
    half_level = 0.5 * P_peak
    # точки ≥ P_peak/2
    above_half = (power >= half_level)

    # ищем ближайшие к пику границы слева и справа
    left_idx = idx_peak
    while left_idx > 0 and above_half[left_idx]:
        left_idx -= 1
    right_idx = idx_peak
    while right_idx < power.size - 1 and above_half[right_idx]:
        right_idx += 1

    tau_fwhm_ps = (right_idx - left_idx) * time_step_ps       # [ps]

    # для sech-импульса:  τ_FWHM = 1.763 · T0
    T0_ps = tau_fwhm_ps / 1.763 if tau_fwhm_ps > 0 else np.inf

    L_D  = T0_ps ** 2 / abs(beta2_ps2_m) if beta2_ps2_m else np.inf
    L_NL = 1.0 / (gamma_1_w_m * P_peak)  if P_peak else np.inf
    L_coupling = np.pi / 2 / coupling_coefficient
    g0_c = float(g0_array[central_core_ind]) if len(g0_array) else 0.0
    L_G  = 1.0 / g0_c if g0_c > 0 else np.inf
    Esat = float(psat_array[central_core_ind]) if len(psat_array) else 0.0
    L_sat = Esat / P_peak if P_peak else np.inf

    print()
    print(f"P_peak = {P_peak:.3e} W")
    print(f"τ_FWHM = {tau_fwhm_ps:.3f} ps,   T0 = {T0_ps:.3f} ps")
    print(f"L_D    = {L_D:.3f} m  (дисперсионная длина)")
    print(f"L_NL   = {L_NL:.3f} m  (нелинейная длина)")
    print(f"L_coupling   = {L_coupling:.3f} m  (длина связи)")
    print(f"L_G    = {L_G:.3f} m  (длина усиления 1/g0)")
    print(f"L_sat  = {L_sat:.3f} m  (экв. путь до насыщения)")
    print()

    return L_D, L_NL, L_coupling, L_G, L_sat


def mcf_nn_reservoir_computing_for_debug(
        fiber_length_m=5.0,                 # длина MCF, m
        feedback_length_m=6.0,              # длина воздушного плеча, m
        abc_fraction=0.10,                  # доля сетки под ABC (с края)
        margin_fraction=0.02,               # отступ после ABC
        time_step_ps=0.1,                   # шаг по времени, ps
        step_number_per_dimensionless_distance=500,
        layer_count=1.0,
        layer_radii_array=(1,),             # радиусы колец, µm
        g0_array=(),
        psat_array=(),
        data_in=None,                       # ndarray (C, M_in)
        kappa=1.0,                          # коэффициент обратной связи
        use_gpu=False,
        display_debug_info=False,
        display_plots=False,
        save_gif=False
):
    # ─── входные данные ──────────────────────────────────────────
    if data_in is None:
        raise ValueError('data_in (C×M) должен быть задан')
    eq_size, _ = data_in.shape

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

    central_core_ind = int(eq_size / 2 + 1) if eq_size > 1 else 0

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

    # ─── буфер задержки ──────────────────────────────────────────
    beta1_air = 1 / light.c_light * 1e+12
    feedback_loop_propagation_time = feedback_length_m * beta1_air         # [ps]

    fiber_propagation_time = fiber_length_m * beta1                           # [ps]
    T = 0.5 * (fiber_propagation_time + feedback_loop_propagation_time) / (1 - abc_fraction - margin_fraction)
    M = 2**13
    time_step_ps = 2 * T / M

    # for debug only
    data_in = 100 * generate_fundamental_soliton_train(
        eq_size=eq_size,
        point_count=M * 8,
        time_step_ps=time_step_ps,
        beta2_ps2_m=beta2,
        gamma_1_w_m=gamma,
        pulse_width_ps=50
    )

    L_D, L_NL, L_coupling, L_G, L_sat = compute_characteristic_lengths(beta2_ps2_m=beta2,
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
    length_scale = np.min([L_D, L_NL, L_coupling])  # [m]

    abc_pts   = int(M * abc_fraction)
    margin_pts= int(M * margin_fraction)
    idx_in    = abc_pts + margin_pts
    idx_out   = M - idx_in - 1

    M_for_signal = idx_out - idx_in + 1

    fiber_length_dimensionless = fiber_length_m / length_scale
    n_z = step_number_per_dimensionless_distance * int(round(fiber_length_dimensionless))

    feedback_coeff = kappa * np.exp(1j * beta1_air * feedback_length_m)

    if display_debug_info:
        print("data_in.shape=", data_in.shape)
        print("data_in size =", data_in.shape[1] * time_step_ps, "ps")
        print("fiber_propagation_time =", fiber_propagation_time, "ps")
        print(f'feedback_loop_propagation_time={feedback_loop_propagation_time:.1f} ps')
        print(f'2T={2 * T:.1f} ps  M={M}  time_step_ps={time_step_ps:.2e} ps  ABC={abc_pts}  margin={margin_pts}')
        print(f'idx_in={idx_in}, idx_out={idx_out}')
        print("fiber_length_dimensionless =", fiber_length_dimensionless)
        print("length_scale =", length_scale)
        print("n_z =", n_z)

    # ─── Solver и параметры уравнения ────────────────────────────
    esat_array = np.asarray(psat_array) * 2 * T

    comp = ComputationalParameters(N=n_z, M=M,
                                   L1=0.0, L2=fiber_length_m,
                                   T1=-T, T2=T,
                                   damp_length=abc_fraction)

    eq = EquationParameters(core_configuration=core_configuration, size=eq_size,
                            ring_count=layer_count,
                            coupling_coefficient=coupling_coefficient, beta1=0,
                            beta2=0, gamma=gamma,
                            E_sat=esat_array, alpha=0.0, g_0=g0_array,
                            display_debug_info=display_debug_info)

    data_batch_count = int(np.floor(data_in.shape[1] / M_for_signal))

    temp_array = np.zeros([eq.size, M], dtype=np.complex128)

    for batch_index in range(data_batch_count):

        temp_array[:, idx_in:idx_in + M_for_signal] = (temp_array[:, idx_in:idx_in + M_for_signal] * feedback_coeff +
                + data_in[:, batch_index * M_for_signal: (batch_index + 1) * M_for_signal])

        solver = Solver(comp, eq,
                        initial_condition=temp_array,
                        stored_steps_count=201,
                        use_dimensional=True,
                        use_gpu=use_gpu,
                        use_torch=use_gpu,
                        display_debug_info=display_debug_info)

        solver.linear_coeffs_array = coupling_matrix

        print("batch_index =", batch_index, "of ", data_batch_count - 1)

        solver.run_numerical_simulation(draw_modulus=display_debug_info,
                                        draw_interval=100,
                                        save_gif=save_gif,
                                        yscale="linear")

        temp_array = solver.numerical_solution[solver.com.N]

        if display_plots:
            energies = [solver.energy[i, :] for i in range(solver.eq.size)]
            names = [f'$E_{{{i}}}$' for i in range(solver.eq.size)]
            plot2D_plotly(solver.z, energies, names=names, x_axis_label='z [m]', y_axis_label='energy [pJ]')

            peak_powers = [solver.peak_power[i, :] for i in range(solver.eq.size)]
            names = [f'$P_{{{i}}}$' for i in range(solver.eq.size)]
            plot2D_plotly(solver.z, peak_powers, names=names, x_axis_label='z [m]', y_axis_label='peak power [W]')

            plot2D_plotly(solver.t, [np.abs(solver.numerical_solution[0][central_core_ind]) ** 2,
                                     np.abs(solver.numerical_solution[-1][central_core_ind]) ** 2],
                          names=[f"$|U_3(z=0,t)|^2$", f"$|U_3(z=L,t)|^2$"], x_axis_label='t [ps]', y_axis_label='power [W]')

            # plot3D_plotly(solver.t, solver.z, np.abs(solver.numerical_solution[central_core_ind]) ** 2, f"$|U_3(z,t)|^2$")

    return solver.numerical_solution[-1]



def mcf_nn_reservoir_computing(
        data_in=None,                       # ndarray (C, M_in)
        fiber_length_m=5.0,                 # длина MCF, m
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

    central_core_ind = int(eq_size / 2 + 1) if eq_size > 1 else 0

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

    feedback_loop_propagation_time = 2 * T - fiber_propagation_time
    beta1_air = 1 / light.c_light * 1e+12
    feedback_length_m = feedback_loop_propagation_time / beta1_air  # длина воздушного плеча, m

    if display_debug_info:
        print()
        print("fiber_length_m =", fiber_length_m)
        print("feedback_length_m =", feedback_length_m)

    assert feedback_length_m > fiber_length_m

    L_D, L_NL, L_coupling, L_G, L_sat = compute_characteristic_lengths(beta2_ps2_m=beta2,
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
    
    comp = ComputationalParameters(N=n_z, M=M,
                                   L1=0.0, L2=fiber_length_m,
                                   T1=-T, T2=T)

    eq = EquationParameters(core_configuration=core_configuration, size=eq_size,
                            ring_count=layer_count,
                            coupling_coefficient=coupling_coefficient, beta1=0,
                            beta2=0, gamma=gamma,
                            E_sat=esat_array, alpha=0.0, g_0=g0_array,
                            display_debug_info=display_debug_info)

    solver = Solver(comp, eq,
                    initial_condition=data_in,
                    stored_steps_count=None if display_debug_info else 2,
                    use_dimensional=True,
                    use_gpu=use_gpu,
                    use_torch=use_gpu,
                    display_debug_info=display_debug_info)

    solver.linear_coeffs_array = coupling_matrix

    solver.run_numerical_simulation(draw_modulus=display_debug_info,
                                    draw_interval=10,
                                    save_gif=save_gif,
                                    yscale="linear")

    if display_plots:
        energies = [solver.energy[i, :] for i in range(solver.eq.size)]
        names = [f'$E_{{{i}}}$' for i in range(solver.eq.size)]
        plot2D_plotly(solver.z, energies, names=names, x_axis_label='z [m]', y_axis_label='energy [pJ]')

        peak_powers = [solver.peak_power[i, :] for i in range(solver.eq.size)]
        names = [f'$P_{{{i}}}$' for i in range(solver.eq.size)]
        plot2D_plotly(solver.z, peak_powers, names=names, x_axis_label='z [m]', y_axis_label='peak power [W]')

        plot2D_plotly(solver.t, [np.abs(solver.numerical_solution[0][central_core_ind]) ** 2,
                                 np.abs(solver.numerical_solution[-1][central_core_ind]) ** 2],
                      names=[f"$|U_3(z=0,t)|^2$", f"$|U_3(z=L,t)|^2$"], x_axis_label='t [ps]', y_axis_label='power [W]')

        # plot3D_plotly(solver.t, solver.z, np.abs(solver.numerical_solution[central_core_ind]) ** 2, f"$|U_3(z,t)|^2$")

    return solver.numerical_solution[-1] * np.exp(1j * beta1_air * feedback_length_m)


if __name__ == '__main__':

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

    M = 2**13
    mg_params = {
        'tau': 17,
        'n': 10,
        'beta': 2,
        'gamma': 1,
        'initial_condition': 1.2
    }
    seed = 42
    data_in = mackey_glass_masked(core_count, M, seed, **mg_params) * 10

    kappa = 0.9 # коэффициент обратной связи 0…1

    modulation_frequency = 40 # GHz

    data_out, feedback_length_m = mcf_nn_reservoir_computing(
        data_in=data_in,  # ndarray (C, M_in)
        fiber_length_m=1,  # физическая длина MCF, m
        time_step_ps=1/modulation_frequency*1e+3,  # шаг сетки t, ps
        step_number_per_dimensionless_distance=200,
        layer_count=layer_count,
        layer_radii_array=layer_radii_array,  # радиусы колец, µm
        g0_array=g0_array,
        psat_array=psat_array,
        use_gpu=False,
        display_debug_info=True,
        display_plots=True,
        save_gif=False
    )
