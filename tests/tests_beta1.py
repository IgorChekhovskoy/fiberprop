from fiberprop.solver import ComputationalParameters, EquationParameters, Solver
from fiberprop.fiber import Fiber, FiberMaterial, CoreConfig
from fiberprop.light import Light
from fiberprop.fiber_base_functions import get_coupling_coefficients

from fiberprop.drawing import *
from fiberprop.pulses import gaussian_pulse


def test_mcf_beta1_dimensionless():
    """
    PRA 2016, Fig.10, 91.6% combining, 6.37 compression
    """
    computational_params = ComputationalParameters(N=1000, M=2 ** 13, L1=0, L2=1.78, T1=-30, T2=30)
    equation_params = EquationParameters(core_configuration=CoreConfig.hexagonal, ring_count=1,
                                         beta1=10, beta2=-2.0, gamma=1.0,
                                         E_sat=0.0, alpha=0.0, g_0=0.0)

    solver = Solver(computational_params, equation_params,
                    pulses=gaussian_pulse, pulse_params_list={"p": 0.687, "tau": 1.775},
                    use_gpu=True, use_torch=True, display_debug_info=True)

    solver.run_numerical_simulation()

    energies = [solver.energy[i, :] for i in range(solver.eq.size)]
    names = [f'$E_{{{i}}}$' for i in range(solver.eq.size)]
    plot2D_plotly(solver.z, energies, names=names, x_axis_label='z')

    peak_powers = [solver.peak_power[i, :] for i in range(solver.eq.size)]
    names = [f'$P_{{{i}}}$' for i in range(solver.eq.size)]
    plot2D_plotly(solver.z, peak_powers, names=names, x_axis_label='z')

    # plot2D_plotly(solver.t, [np.abs(solver.numerical_solution[0][0])**2,
    #                          np.abs(solver.numerical_solution[solver.com.N][0])**2],
    #               names=[f"$|U_0(z=0,t)|^2$", f"$|U_0(z=L,t)|^2$"], x_axis_label='t')

    plot2D_plotly(solver.t, [np.abs(solver.numerical_solution[0][3]) ** 2,
                             np.abs(solver.numerical_solution[solver.com.N][3]) ** 2],
                  names=[f"$|U_3(z=0,t)|^2$", f"$|U_3(z=L,t)|^2$"], x_axis_label='t')

    # plot3D_plotly(solver.t, solver.z, np.abs(solver.numerical_solution[3]) ** 2, f"$|U_3(z,t)|^2$")
    # plot3D_matplotlib_interactive(solver.t, solver.z, np.abs(solver.numerical_solution[3]) ** 2, f"$|U_3(z,t)|^2$")
    # plot3D_plotly(solver.t, solver.z, solver.absolute_error, f"error$")
    # plot3D(solver.z, solver.t, np.abs(solver.numerical_solution[3]) ** 2, f"$|U_3(z,t)|^2$")

    # Переход к размерным величинам

    lambda0 = 1.55  # mkm
    light = Light(lambda0=lambda0)

    fiber = Fiber(
        core_configuration=solver.eq.core_configuration,
        core_count=solver.eq.size,
        core_radius=2.95,
        cladding_diameter=125.0,
        n2=3.2,
        distance_to_fiber_center=17.3,
        NA=0.125,
        core_material=FiberMaterial.SIO2,
        material_concentration=0.038
    )

    fiber.set_refractive_indexes_by_lambda(light.lambda0)

    # Расчёт коэффициентов связи
    coup_mat, err_mat = get_coupling_coefficients(fiber, light, eps=1e-2)

    coupling_coefficient = coup_mat[0][1]
    coupling_coefficient_estimated_error = err_mat[0][1]

    print(f'Lambda = {fiber.distance_to_fiber_center[0] * 2.0} мкм')
    print(f'k = {coupling_coefficient} +- {coupling_coefficient_estimated_error} 1/m')
    print(f'L = {0.5 * np.pi / coupling_coefficient} m \n')

    gamma, gamma_error = fiber.get_gamma(light, eps=1e-2)
    print(f'Gamma = {gamma} +- {gamma_error} 1/(W*m)')

    beta2 = fiber.get_beta2(light)
    print(f'Beta2 = {beta2 * 1e+3} (ps^2)/km')

    gamma = 1.3 * 1e-3  # [1/(W*m)] Для телекома
    beta2 = -20 * 1e-3  # [ps^2/m] Для телекома
    coupling_coefficient = 15.7 * 1e-3  # [1/m] Для телекома

    solver.convert_to_dimensional(coupling_coefficient, gamma, beta2)

    print(solver.z[solver.com.N], solver.t[0])

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

    print(solver.peak_power[3][solver.com.N] / solver.peak_power[3][0])
    print(solver.energy[3][solver.com.N] / solver.energy[3][0])

    assert abs(solver.peak_power[3][solver.com.N] / solver.peak_power[3][0] - 32.6) < 1e-1, "Expected 32.6"
    assert abs(solver.energy[3][solver.com.N] / solver.energy[3][0] - 6.41) < 1e-2, "Expected 6.41"


def test_mcf_beta1_dimensional():
    """
    PRA 2016, Fig.10, 91.6% combining, 6.37 compression
    """

    core_configuration = CoreConfig.hexagonal
    core_count=7

    lambda0 = 1.55  # mkm
    light = Light(lambda0=lambda0)

    # Волокно НЦВО
    fiber = Fiber(
        core_configuration=core_configuration,
        core_count=core_count,
        core_radius=2.95,
        cladding_diameter=125.0,
        n2=3.2,
        distance_to_fiber_center=17.3,
        NA=0.125,
        core_material=FiberMaterial.SIO2_AND_GEO2_ALLOY,
        material_concentration=0.038
    )

    fiber.set_refractive_indexes_by_lambda(light.lambda0)

    print("n_core =", fiber.n_core)
    print("n_cladding =", fiber.n_cladding)

    # Расчёт коэффициентов связи
    coup_mat, err_mat = get_coupling_coefficients(fiber, light, eps=1e-2)

    coupling_coefficient = coup_mat[0][1]
    coupling_coefficient_estimated_error = err_mat[0][1]

    print(f'Lambda = {fiber.distance_to_fiber_center[0] * 2.0} мкм')
    print(f'k = {coupling_coefficient} +- {coupling_coefficient_estimated_error} 1/m')
    print(f'L = {0.5 * np.pi / coupling_coefficient} m \n')

    gamma, gamma_error = fiber.get_gamma(light, eps=1e-2)
    print(f'Gamma = {gamma} +- {gamma_error} 1/(W*m)')

    b = fiber.get_b(light)
    print(f'B = {b}')

    beta = fiber.get_beta(light)
    print(f'Beta = {beta} 1/mkm')

    beta1 = fiber.get_beta1(light)
    print(f'Beta1 = {beta1} (ps)/m')

    beta2 = fiber.get_beta2(light)
    print(f'Beta2 = {beta2} (ps^2)/m')

    # gamma = 1.718601358688562 * 1e-3  # [1/(W*m)]
    # beta1 = 4.833695096477126 # [ps/m]
    # beta2 = -9.634911188459554 * 1e-3  # [ps^2/m]
    # coupling_coefficient = 1.3960166930953415 * 1e+2  # [1/m]
    #
    # gamma = 1.3 * 1e-3  # [1/(W*m)] Для телекома
    # beta1 = 0 # [ps/m]
    # beta2 = -20 * 1e-3  # [ps^2/m] Для телекома
    # coupling_coefficient = 15.7 * 1e-3  # [1/m] Для телекома

    time_scale = np.sqrt(0.5 * abs(beta2) / coupling_coefficient)  # [ps]
    power_scale = (coupling_coefficient / gamma)  # [W]
    length_scale = (1 / coupling_coefficient)  # [m]

    L1 = 0  # [m]
    L2 = 1.78 * length_scale  # [m]
    T = 30 * time_scale  # [ps]

    computational_params = ComputationalParameters(N=1000, M=2 ** 13, L1=L1, L2=L2, T1=-T, T2=+T)

    equation_params = EquationParameters(core_configuration=core_configuration, size=core_count, ring_count=1,
                                         coupling_coefficient=coupling_coefficient,
                                         beta1=beta1, beta2=beta2, gamma=gamma,
                                         E_sat=0.0, alpha=0.0, g_0=0.0)

    solver = Solver(computational_params, equation_params,
                    use_dimensional=True,
                    pulses=gaussian_pulse,
                    pulse_params_list={"p": 0.687 * power_scale,
                                       "tau": 1.775 * time_scale},
                    use_gpu=False, use_torch=False, display_debug_info=True)

    solver.run_numerical_simulation()

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

    print(solver.peak_power[3][solver.com.N] / solver.peak_power[3][0])
    print(solver.energy[3][solver.com.N] / solver.energy[3][0])

    assert abs(solver.peak_power[3][solver.com.N] / solver.peak_power[3][0] - 32.6) < 1e-1, "Expected 32.6"
    assert abs(solver.energy[3][solver.com.N] / solver.energy[3][0] - 6.41) < 1e-2, "Expected 6.41"


if __name__ == '__main__':
    # test_mcf_beta1_dimensionless()
    test_mcf_beta1_dimensional()
