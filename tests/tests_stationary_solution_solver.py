import mkl
import numpy as np

from fiberprop.drawing import plot2D_plotly

from fiberprop.solver import ComputationalParameters, EquationParameters, Solver, CoreConfig, print_matrix
from fiberprop.pulses import fundamental_soliton


def test_stationary_solution_solver_conservative_mcf():

    beta2 = -2

    computational_params = ComputationalParameters(N=2000, M=2 ** 13, L1=0, L2=10, T1=-30, T2=30)
    equation_params = EquationParameters(core_configuration=CoreConfig.hexagonal, size=7, ring_count=1, beta2=beta2, gamma=1)

    solver = Solver(computational_params, equation_params,
                    pulses=fundamental_soliton, pulse_params_list=[
                                                                    {"beta2": beta2, "c": 1},  # для 0-го уравнения
                                                                    {"beta2": beta2, "c": 1},  # для 1-го уравнения
                                                                    {"beta2": beta2, "c": 1},  # для 2-го уравнения
                                                                    {"beta2": beta2, "c": 2},  # для 3-го уравнения
                                                                    {"beta2": beta2, "c": 1},  # для 4-го уравнения
                                                                    {"beta2": beta2, "c": 1},  # для 5-го уравнения
                                                                    {"beta2": beta2, "c": 1},  # для 6-го уравнения
                                                                ], )

    solver.find_stationary_solution(lambda_val=4, plot_graphs=False, yscale='log')

    fields = [np.abs(solver.numerical_solution[0][i])**2 for i in range(solver.eq.size)]
    names = [f'$|U_{{{i}}}(z=0, t)|^2$' for i in range(solver.eq.size)]
    plot2D_plotly(solver.t,fields, names=names, x_axis_label='t', yscale='log')

    solver.run_numerical_simulation()

    fields = [np.abs(solver.numerical_solution[solver.com.N][i]) ** 2 for i in range(solver.eq.size)]
    names = [f'$|U_{{{i}}}(z=L, t)|^2$' for i in range(solver.eq.size)]
    plot2D_plotly(solver.t, fields, names=names, x_axis_label='t', yscale='log')

    peak_powers = [solver.peak_power[i, :] for i in range(solver.eq.size)]
    names = [f'$P_{{{i}}}$' for i in range(solver.eq.size)]
    plot2D_plotly(solver.z, peak_powers, names=names, x_axis_label='z [m]', y_axis_label='peak power [W]')


def test_stationary_solution_solver_nonconservative_mcf_7core_hexagonal():

    beta2 = -2

    computational_params = ComputationalParameters(N=2000, M=2 ** 12, L1=0, L2=10, T1=-30, T2=30)
    equation_params = EquationParameters(core_configuration=CoreConfig.hexagonal, size=7, ring_number=1, beta2=beta2, gamma=1)

    solver = Solver(computational_params, equation_params,
                    pulses=fundamental_soliton, pulse_params_list=[
                                                                    {"beta2": beta2, "c": 1},  # для 0-го уравнения
                                                                    {"beta2": beta2, "c": 1},  # для 1-го уравнения
                                                                    {"beta2": beta2, "c": 1},  # для 2-го уравнения
                                                                    {"beta2": beta2, "c": 2},  # для 3-го уравнения
                                                                    {"beta2": beta2, "c": 1},  # для 4-го уравнения
                                                                    {"beta2": beta2, "c": 1},  # для 5-го уравнения
                                                                    {"beta2": beta2, "c": 1},  # для 6-го уравнения
                                                                ], )

    solver.find_stationary_solution(lambda_val=4, plot_graphs=False, yscale='log')

    fields_conservative = [np.abs(solver.numerical_solution[0][i])**2 for i in range(solver.eq.size)]
    names_conservative = [f'$|U_{{{i}}}(z=0, t)|^2$' for i in range(solver.eq.size)]
    # plot2D_plotly(solver.t,fields_conservative, names=names_conservative, x_axis_label='t', yscale='log')

    # --------------------------------

    equation_params = EquationParameters(core_configuration=CoreConfig.hexagonal, size=7, ring_number=1, beta2=beta2, gamma=1, E_sat=100,
                                         alpha=10, g_0=10)

    solver = Solver(computational_params, equation_params,
                    pulses=fundamental_soliton, pulse_params_list=[
            {"beta2": beta2, "c": 1},  # для 0-го уравнения
            {"beta2": beta2, "c": 1},  # для 1-го уравнения
            {"beta2": beta2, "c": 1},  # для 2-го уравнения
            {"beta2": beta2, "c": 2},  # для 3-го уравнения
            {"beta2": beta2, "c": 1},  # для 4-го уравнения
            {"beta2": beta2, "c": 1},  # для 5-го уравнения
            {"beta2": beta2, "c": 1},  # для 6-го уравнения
        ], )

    solver.find_stationary_solution(lambda_val=4, plot_graphs=False, yscale='log')

    fields_nonconservative = [np.abs(solver.numerical_solution[0][i]) ** 2 for i in range(solver.eq.size)]
    names_nonconservative = [f'$|Un_{{{i}}}(z=0, t)|^2$' for i in range(solver.eq.size)]
    # plot2D_plotly(solver.t, fields_nonconservative, names=names_nonconservative, x_axis_label='t', yscale='log')

    plot2D_plotly(solver.t, fields_conservative + fields_nonconservative,
                  names=names_conservative + names_nonconservative, x_axis_label='t', yscale='log')

    solver.run_numerical_simulation()

    fields = [np.abs(solver.numerical_solution[solver.com.N][i]) ** 2 for i in range(solver.eq.size)]
    names = [f'$|U_{{{i}}}(z=L, t)|^2$' for i in range(solver.eq.size)]
    plot2D_plotly(solver.t, fields, names=names, x_axis_label='t', yscale='log')

    peak_powers = [solver.peak_power[i, :] for i in range(solver.eq.size)]
    names = [f'$P_{{{i}}}$' for i in range(solver.eq.size)]
    plot2D_plotly(solver.z, peak_powers, names=names, x_axis_label='z [m]', y_axis_label='peak power [W]')


def set_linear_coeffs_array(solver, kappa):
    solver.linear_coeffs_array[0][0] = 0
    solver.linear_coeffs_array[0][1] = 1
    solver.linear_coeffs_array[1][0] = 1
    solver.linear_coeffs_array[1][1] = kappa


def test_stationary_solution_solver_nonconservative_mcf():

    lambda_val = 2

    nc = 4
    kappa = 2 / np.sqrt(nc)
    beta2 = -2
    gamma = [nc, 1]

    length = 1000

    computational_params = ComputationalParameters(N=100 * length, M=2 ** 12, L1=0, L2=length, T1=-120, T2=120)

    equation_params = EquationParameters(core_configuration=CoreConfig.empty_ring, size=2, beta2=beta2, gamma=gamma)

    solver = Solver(computational_params, equation_params,
                    pulses=fundamental_soliton, pulse_params_list=[
                                                                    {"beta2": beta2, "c": 1},  # для 0-го уравнения
                                                                    {"beta2": beta2, "c": 1},  # для 1-го уравнения
                                                                ], )

    set_linear_coeffs_array(solver, kappa)
    print_matrix(solver.linear_coeffs_array, "linear_coeffs_array")

    solver.find_stationary_solution(lambda_val=lambda_val, plot_graphs=False, yscale='log')

    fields_conservative = [np.abs(solver.numerical_solution[0][i])**2 for i in range(solver.eq.size)]
    names_conservative = [f'$|U_{{{i}}}(z=0, t)|^2$' for i in range(solver.eq.size)]
    # plot2D_plotly(solver.t,fields_conservative, names=names_conservative, x_axis_label='t', yscale='log')

    # --------------------------------

    equation_params = EquationParameters(core_configuration=CoreConfig.empty_ring, size=2, beta2=beta2, gamma=gamma,
                                         g_0=[1, 0],
                                         E_sat=[1, 1],
                                         alpha=[0.2, 0.1])

    solver = Solver(computational_params, equation_params, pulses=fundamental_soliton, pulse_params_list=[
            {"beta2": beta2, "c": 1},  # для 0-го уравнения
            {"beta2": beta2, "c": 1},  # для 1-го уравнения
        ], )

    set_linear_coeffs_array(solver, kappa)
    print_matrix(solver.linear_coeffs_array, "linear_coeffs_array")

    solver.find_stationary_solution(lambda_val=lambda_val, plot_graphs=False, yscale='log')

    fields_nonconservative = [np.abs(solver.numerical_solution[0][i]) ** 2 for i in range(solver.eq.size)]
    names_nonconservative = [f'$|Un_{{{i}}}(z=0, t)|^2$' for i in range(solver.eq.size)]
    # plot2D_plotly(solver.t, fields_nonconservative, names=names_nonconservative, x_axis_label='t', yscale='log')

    # plot2D_plotly(solver.t, fields_conservative + fields_nonconservative,
    #               names=names_conservative + names_nonconservative, x_axis_label='t', yscale='log')

    solver.run_numerical_simulation(print_modulus=True, print_interval=100, yscale='log')

    # fields = [np.abs(solver.numerical_solution[solver.com.N][i]) ** 2 for i in range(solver.eq.size)]
    # names = [f'$|U_{{{i}}}(z=L, t)|^2$' for i in range(solver.eq.size)]
    # plot2D_plotly(solver.t, fields, names=names, x_axis_label='t', yscale='log')
    #
    # peak_powers = [solver.peak_power[i, :] for i in range(solver.eq.size)]
    # names = [f'$P_{{{i}}}$' for i in range(solver.eq.size)]
    # plot2D_plotly(solver.z, peak_powers, names=names, x_axis_label='z [m]', y_axis_label='peak power [W]')


if __name__ == '__main__':

    mkl.set_num_threads(8)

    # test_stationary_solution_solver_conservative_mcf()
    test_stationary_solution_solver_nonconservative_mcf()
