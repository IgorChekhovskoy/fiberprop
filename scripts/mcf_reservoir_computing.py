import numpy as np

from fiberprop.solver import ComputationalParameters, EquationParameters, Solver, CoreConfig, \
    get_core_count, print_matrix
from fiberprop.coupling_coefficient import Fiber, Light, FiberMaterial, get_coupling_coefficients
from fiberprop.drawing import *
from time import time


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


def mcf_nn_reservoir_computing(fiber_length_dimensionless=5,
                               time_window_halfwidth_dimensionless=30,
                               step_number_per_dimensionless_distance = 500,
                               layer_count=1,
                               layer_radii_array=[1], # [mkm]
                               core_radius=4, # [mkm]
                               g0_array=[], # [1/m]
                               psat_array=[], # [pJ]
                               data_in=[], # [sqrt(W)]
                               use_gpu=False,
                               display_debug_info=False,
                               display_plots=False
                               ):
    eq_size = data_in.shape[0]
    M = data_in.shape[1]

    lambda0 = 1.55  # mkm
    light = Light()
    light.lambda0 = lambda0

    fiber = Fiber()
    fiber.core_configuration = CoreConfig.hexagonal
    fiber.core_count = eq_size
    # fiber.ring_count = layer_count
    fiber.core_radius = core_radius
    fiber.cladding_diameter = 125.0 # [mkm]
    fiber.n2 = 3.2
    fiber.distance_to_fiber_center = 17.3 # [mkm]
    fiber.NA = 0.125
    fiber.core_material = FiberMaterial.SIO2
    fiber.material_concentration = 0.038
    fiber.set_refractive_indexes_by_lambda(lambda0)

    t1 = time()
    coup_mat, err_mat = get_coupling_coefficients(fiber, light, eps=1e-2)

    if display_debug_info:
        print("get_coupling_coefficients time =", time() - t1, " seconds")

        print_matrix(coup_mat)

    coupling_coefficient = coup_mat[0][1]
    coupling_coefficient_estimated_error = err_mat[0][1]

    if display_debug_info:
        print(f'Lambda = {fiber.distance_to_fiber_center * 2.0} mkm')
        print(f'k = {coupling_coefficient} +- {coupling_coefficient_estimated_error} 1/m')
        print(f'L = {0.5 * np.pi / coupling_coefficient} m \n')

    t1 = time()
    gamma, gamma_error = fiber.get_gamma(light, eps=1e-2)

    if display_debug_info:
        print("get_gamma time =", time() - t1, " seconds")
        print(f'Gamma = {gamma} +- {gamma_error} 1/(W*m)')

    t1 = time()
    beta2 = fiber.get_beta2(light)

    if display_debug_info:
        print("get_beta2 time =", time() - t1, " seconds")
        print(f'Beta2 = {beta2} (ps^2)/km')

    beta2 *= 1e-3 # [(ps^2)/m]

    # gamma = 1.3 * 1e-3  # [1/(W*m)] Для телекома
    # beta2 = -20 * 1e-3  # [ps^2/m] Для телекома
    # coupling_coefficient = 15.7 * 1e-3  # [1/m] Для телекома

    time_scale = np.sqrt(0.5 * abs(beta2) / coupling_coefficient)  # [ps]
    power_scale = coupling_coefficient / gamma  # [W]
    length_scale = 1 / coupling_coefficient  # [m]

    L1 = 0  # [m]
    L2 = fiber_length_dimensionless * length_scale  # [m]
    T = time_window_halfwidth_dimensionless * time_scale  # [ps]

    esat_array = psat_array * 2 * T

    computational_params = ComputationalParameters(N=step_number_per_dimensionless_distance * round(L2 / length_scale),
                                                   M=M, L1=L1, L2=L2, T1=-T, T2=+T)

    equation_params = EquationParameters(core_configuration=CoreConfig.hexagonal, size=eq_size, ring_count=layer_count,
                                         coupling_coefficient=coupling_coefficient, beta2=beta2, gamma=gamma,
                                         E_sat=esat_array, alpha=0.0, g_0=g0_array,
                                         display_debug_info=display_debug_info)

    solver = Solver(computational_params, equation_params, initial_condition=data_in,
                    use_dimensional=True, use_gpu=use_gpu, use_torch=True, display_debug_info=display_debug_info)

    solver.run_numerical_simulation()

    if display_plots:
        energies = [solver.energy[i, :] for i in range(solver.eq.size)]
        names = [f'$E_{{{i}}}$' for i in range(solver.eq.size)]
        plot2D_plotly(solver.z, energies, names=names, x_axis_label='z [m]', y_axis_label='energy [pJ]')

        peak_powers = [solver.peak_power[i, :] for i in range(solver.eq.size)]
        names = [f'$P_{{{i}}}$' for i in range(solver.eq.size)]
        plot2D_plotly(solver.z, peak_powers, names=names, x_axis_label='z [m]', y_axis_label='peak power [W]')

        plot2D_plotly(solver.t, [np.abs(solver.numerical_solution[0][3]) ** 2,
                                 np.abs(solver.numerical_solution[solver.com.N][3]) ** 2],
                      names=[f"$|U_3(z=0,t)|^2$", f"$|U_3(z=L,t)|^2$"], x_axis_label='t [ps]', y_axis_label='power [W]')

        # plot3D_plotly(solver.t, solver.z, np.abs(solver.numerical_solution[3]) ** 2, f"$|U_3(z,t)|^2$")

    return solver.numerical_solution[solver.com.N]


if __name__ == '__main__':

    layer_count = 1
    core_configuration = CoreConfig.hexagonal
    core_count = get_core_count(core_configuration=core_configuration, ring_count=layer_count)

    layer_radii_array = np.zeros(layer_count)
    for i in range(layer_count):
        layer_radii_array[i] = i + 1 # [mkm]

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
    data_in = mackey_glass_masked(core_count, M, seed, **mg_params)

    data_out = mcf_nn_reservoir_computing(fiber_length_dimensionless=5,
                                          time_window_halfwidth_dimensionless=30,
                                          step_number_per_dimensionless_distance = 500,
                                          layer_count=layer_count,
                                          layer_radii_array=layer_radii_array,  # [mkm]
                                          core_radius=4,  # [mkm]
                                          g0_array=g0_array,  # [1/m]
                                          psat_array=psat_array,  # [W]
                                          data_in=data_in,  # [sqrt(W)]
                                          use_gpu=False,
                                          display_debug_info=True,
                                          display_plots=False)


