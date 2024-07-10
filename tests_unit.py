from propagation import *


def test_equation_parameters_initialization_with_scalars():
    num_eq = 3
    beta_2_scalar = 1.0
    gamma_array = np.random.random(num_eq)
    E_sat_scalar = 0.5
    alpha_array = np.random.random(num_eq)
    # g_0_scalar = 0.1

    # Инициализация класса с параметрами
    params = EquationParameters(
        num_equations=num_eq,
        beta_2=beta_2_scalar,
        gamma=gamma_array,
        E_sat=E_sat_scalar,
        alpha=alpha_array,
        # g_0=g_0_scalar
    )

    print(params)


def test_computational_parameters_initialization():
    """ Пример инициализации объекта ComputationalParameters """
    params = ComputationalParameters(N=100, M=200, L1=0.0, L2=10.0, T1=0.0, T2=20.0)

    print(params)
