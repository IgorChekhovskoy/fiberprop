import numpy as np

from fiberprop.solver import ComputationalParameters, EquationParameters, Solver, CoreConfig, make_eq_mask_from_ring_number
from fiberprop.coupling_coefficient import Fiber, Light, FiberMaterial, get_coupling_coefficients
from fiberprop.drawing import *


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
                               layer_number=1,
                               layer_radii_array=[1], # [mkm]
                               core_radius=4, # [mkm]
                               g0_array=[], # [1/m]
                               esat_array=[], # [pJ]
                               data_in=[], # [sqrt(W)]
                               display_plots=True
                               ):
    eq_size = data_in.shape[0]
    M = data_in.shape[1]

    lambda0 = 1.55  # mkm
    light = Light()
    light.lambda0 = lambda0

    fiber = Fiber()
    fiber.core_configuration = CoreConfig.hexagonal
    fiber.core_count = eq_size
    fiber.core_radius = core_radius
    fiber.cladding_diameter = 125.0 # [mkm]
    fiber.n2 = 3.2
    fiber.distance_to_fiber_center = 17.3 # [mkm]
    fiber.NA = 0.125
    fiber.core_material = FiberMaterial.SIO2
    fiber.material_concentration = 0.038
    fiber.set_refractive_indexes_by_lambda(lambda0)

    coup_mat, err_mat = get_coupling_coefficients(fiber, light, eps=1e-2, proc_num=1)

    coupling_coefficient = coup_mat[0][1]
    coupling_coefficient_estimated_error = err_mat[0][1]

    print(f'Lambda = {fiber.distance_to_fiber_center * 2.0} mkm')
    print(f'k = {coupling_coefficient} +- {coupling_coefficient_estimated_error} 1/m')
    print(f'L = {0.5 * np.pi / coupling_coefficient} m \n')

    gamma, gamma_error = fiber.get_gamma(light, eps=1e-2)
    print(f'Gamma = {gamma} +- {gamma_error} 1/(W*m)')
    beta2 = fiber.get_beta2(light)
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
    T = 30 * time_scale  # [ps]

    computational_params = ComputationalParameters(N=500 * round(L2 / length_scale), M=M, L1=L1, L2=L2, T1=-T, T2=+T)

    equation_params = EquationParameters(core_configuration=CoreConfig.hexagonal, size=eq_size, ring_number=layer_number,
                                         coupling_coefficient=coupling_coefficient, beta2=beta2, gamma=gamma,
                                         E_sat=esat_array, alpha=0.0, g_0=g0_array)

    solver = Solver(computational_params, equation_params,
                    use_dimensional=True, use_gpu=False, use_torch=True)

    solver.numerical_solution[0] = data_in

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

    layer_number = 1
    core_configuration = CoreConfig.hexagonal
    core_number, _ = make_eq_mask_from_ring_number(ring_number=layer_number, core_configuration=core_configuration)

    layer_radii_array = np.zeros(layer_number) # [mkm]
    for i in range(layer_number):
        layer_radii_array[i] = i + 1

    g0_array = np.zeros(core_number) # [1/m]
    for i in range(core_number):
        g0_array[i] = 1.0

    esat_array = np.zeros(core_number) # [pJ]
    for i in range(core_number):
        esat_array[i] = 1.0

    M = 2**13
    mg_params = {
        'tau': 17,
        'n': 10,
        'beta': 2,
        'gamma': 1,
        'initial_condition': 1.2
    }
    seed = 42
    data_in = mackey_glass_masked(core_number, M, seed, **mg_params)

    data_out = mcf_nn_reservoir_computing(fiber_length_dimensionless=5,
                               layer_number=layer_number,
                               layer_radii_array=layer_radii_array, # [mkm]
                               core_radius=4, # [mkm]
                               g0_array=g0_array, # [1/m]
                               esat_array=esat_array, # [pJ]
                               data_in=data_in, # [sqrt(W)]
                               display_plots=True)
