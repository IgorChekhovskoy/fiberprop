import mkl
import numpy as np

from fiberprop.drawing import plot2D_plotly

from fiberprop.solver import ComputationalParameters, EquationParameters, Solver, print_matrix
from fiberprop.pulses import fundamental_soliton


def stationary_solution_solver_conservative_mcf():

    beta2 = -2

    computational_params = ComputationalParameters(N=2000, M=2 ** 13, L1=0, L2=10, T1=-30, T2=30)
    equation_params = EquationParameters(core_configuration=3, size=7, ring_number=1, beta2=beta2, gamma=1)

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


if __name__ == '__main__':

    mkl.set_num_threads(8)

    stationary_solution_solver_conservative_mcf()
