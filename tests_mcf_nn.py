from importlib import reload

import propagation
reload(propagation)
from propagation import *

import SSFM_MCF
reload(SSFM_MCF)
from SSFM_MCF import *


def test_mcf_nn_reservoir_computing():
    computational_params = ComputationalParameters(N=1000, M=2 ** 13, L1=0, L2=1.78, T1=-30, T2=30)
    equation_params = EquationParameters(core_configuration=3, size=7, ring_number=1, beta_2=-2.0, gamma=1.0, E_sat=1.0, alpha=1.1, g_0=1.4)

    solver = Solver(computational_params, equation_params, pulses=GaussianPulse, pulse_params_list={"p": 0.687, "tau": 1.775})
    solver.run()

    # plot3D_plotly(solver.t, solver.z, np.abs(solver.numerical_solution[3]) ** 2)
    # plot3D_matplotlib_interactive(solver.t, solver.z, np.abs(solver.numerical_solution[3]) ** 2)
    # plot3D_plotly(solver.t, solver.z, solver.absolute_error)
    # plot3D(solver.z, solver.t, np.abs(solver.numerical_solution[3]) ** 2, "U(z,t)")

    energies = [solver.energy[i, :] for i in range(solver.eq.size)]
    names = [f'$E_{{{i}}}$' for i in range(solver.eq.size)]
    plot2D_plotly(solver.z, energies, names=names)

    peak_powers = [solver.peak_power[i, :] for i in range(solver.eq.size)]
    names = [f'$P_{{{i}}}$' for i in range(solver.eq.size)]
    plot2D_plotly(solver.z, peak_powers, names=names)

    plot2D_plotly(solver.t, [np.abs(solver.numerical_solution[0][0])**2,
                             np.abs(solver.numerical_solution[solver.com.N][0])**2],
                  names=[f"$|U_0(z=0,t)|^2$", f"$|U_0(z=L,t)|^2$"])

    plot2D_plotly(solver.t, [np.abs(solver.numerical_solution[0][3])**2,
                             np.abs(solver.numerical_solution[solver.com.N][3])**2],
                  names=[f"$|U_3(z=0,t)|^2$", f"$|U_3(z=L,t)|^2$"])


if __name__ == '__main__':
    test_mcf_nn_reservoir_computing()
