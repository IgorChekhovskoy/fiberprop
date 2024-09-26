from importlib import reload

from fiberprop.solver import ComputationalParameters, EquationParameters, Solver, CoreConfig
from fiberprop.coupling_coefficient import Fiber, Light, CoreConfiguration, FiberMaterial, get_coupling_coefficients

from fiberprop import propagation
from fiberprop import ssfm_mcf

reload(propagation)
reload(ssfm_mcf)

from fiberprop.propagation import *
from fiberprop.ssfm_mcf import *


def test_mcf_compression():
    """
    PRA 2016, Fig.10, 91.6% combining, 6.37 compression
    """
    computational_params = ComputationalParameters(N=500, M=2 ** 11, L1=0, L2=1.78, T1=-30, T2=30)
    equation_params = EquationParameters(core_configuration=CoreConfig.hexagonal, size=7, ring_number=1, beta2=-2.0, gamma=1.0, E_sat=0.0, alpha=0.0, g_0=0.0)

    solver = Solver(computational_params, equation_params,
                    pulses=gaussian_pulse, pulse_params_list={"p": 0.687, "tau": 1.775},
                    use_gpu=True)

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

    plot2D_plotly(solver.t, [np.abs(solver.numerical_solution[0][3])**2,
                             np.abs(solver.numerical_solution[solver.com.N][3])**2],
                  names=[f"$|U_3(z=0,t)|^2$", f"$|U_3(z=L,t)|^2$"], x_axis_label='t')

    # plot3D_plotly(solver.t, solver.z, np.abs(solver.numerical_solution[3]) ** 2, f"$|U_3(z,t)|^2$")
    # plot3D_matplotlib_interactive(solver.t, solver.z, np.abs(solver.numerical_solution[3]) ** 2, f"$|U_3(z,t)|^2$")
    # plot3D_plotly(solver.t, solver.z, solver.absolute_error, f"error$")
    # plot3D(solver.z, solver.t, np.abs(solver.numerical_solution[3]) ** 2, f"$|U_3(z,t)|^2$")

    # Переход к размерным величинам

    lambda0 = 1.05018  # mkm
    light = Light()
    light.lambda0 = lambda0

    fiber = Fiber()
    fiber.core_configuration = CoreConfiguration.HEXAGONAL  # solver.eq.core_configuration
    fiber.core_count = solver.eq.size
    fiber.core_radius = 2.95
    fiber.cladding_diameter = 125.0
    fiber.n2 = 3.2
    fiber.distance_to_fiber_center = 17.3
    fiber.NA = 0.125
    fiber.core_material = FiberMaterial.SIO2_AND_GEO2_ALLOY
    fiber.material_concentration = 0.038
    fiber.set_refractive_indexes_by_lambda(lambda0)

    coup_mat, err_mat = get_coupling_coefficients(fiber, light, eps=1e-2, proc_num=1)

    couping_coefficient = coup_mat[0][1]
    coupling_coefficient_estimated_error = err_mat[0][1]

    print(f'Lambda = {fiber.distance_to_fiber_center * 2.0} mkm')
    print(f'k = {couping_coefficient} +- {coupling_coefficient_estimated_error} 1/cm')
    print(f'L = {0.5 * np.pi / couping_coefficient} cm \n')

    gamma, gamma_error = fiber.get_gamma(light, eps=1e-2)
    print(f'Gamma = {gamma} +- {gamma_error} 1/(W*m)')
    beta2 = fiber.get_beta2(light)
    print(f'Beta2 = {beta2} (ps^2)/km')

    # gamma = 1.3 * 1e-3  # Для телекома
    # beta2 = -20  # Для телекома
    # couping_coefficient = 15.7 * 1e-5  # Для телекома

    solver.convert_to_dimensional(couping_coefficient, gamma, beta2)

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


if __name__ == '__main__':
    test_mcf_compression()
