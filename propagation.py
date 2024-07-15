from cmath import exp as cexp
from tqdm import trange
from SSFM_MCF import *
from matrixes import *
from drawing import *
from scipy.fft import fftfreq
from dataclasses import dataclass, field
from typing import Union
from time import time


@dataclass
class ComputationalParameters:
    N: int = 0
    M: int = 0
    L1: float = 0.0
    L2: float = 0.0
    T1: float = 0.0
    T2: float = 0.0

    h: float = field(init=False, default=0.0)
    tau: float = field(init=False, default=0.0)

    def __post_init__(self):
        if self.N > 1:
            self.h = (self.L2 - self.L1) / (self.N - 1)
        else:
            self.h = 0.0

        if self.M > 1:
            self.tau = (self.T2 - self.T1) / (self.M - 1)
        else:
            self.tau = 0.0

@dataclass
class EquationParameters:
    num_equations: int = 1
    beta_2: Union[float, np.ndarray] = -1.0
    gamma: Union[float, np.ndarray] = 1.0
    E_sat: Union[float, np.ndarray] = 1.0
    alpha: Union[float, np.ndarray] = 0.0
    g_0: Union[float, np.ndarray] = 0.0

    def __post_init__(self):
        # Преобразование скалярных параметров в массивы одинаковых значений
        if isinstance(self.beta_2, (int, float)):
            self.beta_2 = np.full(self.num_equations, self.beta_2, dtype=float)
        if isinstance(self.gamma, (int, float)):
            self.gamma = np.full(self.num_equations, self.gamma, dtype=float)
        if isinstance(self.E_sat, (int, float)):
            self.E_sat = np.full(self.num_equations, self.E_sat, dtype=float)
        if isinstance(self.alpha, (int, float)):
            self.alpha = np.full(self.num_equations, self.alpha, dtype=float)
        if isinstance(self.g_0, (int, float)):
            self.g_0 = np.full(self.num_equations, self.g_0, dtype=float)


@dataclass
class SimulationRunner:
    def __init__(self, com: ComputationalParameters, eq: EquationParameters, pulse=gain_loss_soliton):
        self.com = com
        self.eq = eq
        self.pulse = pulse
        self.t = None
        self.omega = None
        self.D = None
        self.numerical_solution = None
        self.current_energy = None
        self.absolute_error = None
        self.C_norm = None
        self.L2_norm = None
        self.analytical_solution = None

    def initialize_arrays(self):
        self.t = np.linspace(self.com.T1, self.com.T2, self.com.M)
        self.omega = fftfreq(self.com.M - 1, self.com.tau) * 2 * pi

    def calculate_D_matrix(self):
        coupling_matrix = get_ring_coupling_matrix(self.eq.num_equations)
        self.D = get_pade_exponential2(create_freq_matrix(coupling_matrix, self.eq.beta_2,
                                                          self.eq.alpha, self.eq.g_0,
                                                          self.omega, self.com.h))

    def filter_params(self, func):
        # Получаем список параметров, которые принимает функция
        func_params = func.__code__.co_varnames[:func.__code__.co_argcount]
        # Фильтруем параметры, чтобы оставить только те, которые нужны функции
        return {k: v for k, v in vars(self.eq).items() if k in func_params}

    def run_numerical_simulation(self):
        self.numerical_solution = np.zeros((self.eq.num_equations, self.com.N, self.com.M-1), dtype=complex)
        self.current_energy = np.zeros(self.eq.num_equations, dtype=float)
        old = np.zeros((self.eq.num_equations, self.com.M - 1), dtype=complex)

        for k in range(self.eq.num_equations):
            pulse_params = self.filter_params(self.pulse)
            old[k] = self.pulse(t=self.t[:-1], x=0,
                                **{key: val[k] if isinstance(val, np.ndarray) else val for key, val in
                                   pulse_params.items()})
            self.numerical_solution[k][0] = old[k]

        for n in trange(self.com.N - 1):
            new = SSFMOrder2(old, self.current_energy, self.D, self.eq.gamma,
                             self.eq.E_sat, self.eq.g_0, self.com.h, self.com.tau)
            for i in range(self.eq.num_equations):
                self.numerical_solution[i][n+1] = new[i]
            old = new

    def get_analytical_solution(self):
        z = np.linspace(self.com.L1, self.com.L2, self.com.N)
        self.analytical_solution = np.zeros((self.eq.num_equations, self.com.N, self.com.M - 1),
                                            dtype=complex)  # сюда будет записываться решение

        for k in range(self.eq.num_equations):
            pulse_params = self.filter_params(self.pulse)
            pulse_params = {key: val[k] if isinstance(val, np.ndarray) else val for key, val in pulse_params.items()}
            for n, z_val in enumerate(z):
                self.analytical_solution[k, n] = self.pulse(t=self.t[:-1], x=z_val, **pulse_params)

    def calculate_error(self):
        self.absolute_error = abs(self.analytical_solution[self.eq.num_equations // 2] -
                                  self.numerical_solution[self.eq.num_equations // 2])
        self.C_norm = np.max(self.absolute_error[self.com.N - 1])
        print('C norm =\t', self.C_norm)
        self.L2_norm = get_energy_rectangles(self.absolute_error[self.com.N - 1] ** 2, self.com.tau)
        print('L2 norm =\t', self.L2_norm)

    def plot_error(self, plot=True):
        if plot:
            T_grid, Z_grid = np.meshgrid(self.t[:-1], np.linspace(self.com.L1, self.com.L2, self.com.N))
            name = 'абсолютная_ошибка-case1'
            plot3D(Z_grid, T_grid, self.absolute_error, name)

    def run(self):
        self.initialize_arrays()
        self.calculate_D_matrix()
        self.run_numerical_simulation()
        self.get_analytical_solution()
        self.calculate_error()
        self.plot_error()


def FullPropagation_Simulation(pulse, N, equation_number, h, tau, coupling_matrix, beta_2, gamma, E_sat, alpha, g_0,
                                    Fresnel_k, omega, delta, Delta, phi, ITER_NUM):

    """ Последовательное моделирование ITER_NUM итераций """

    M = pulse.shape[1]
    w = fftfreq(M, tau) * 2 * pi
    Dmat = get_pade_exponential2(create_freq_matrix(coupling_matrix, beta_2, alpha, g_0, w, h))

    current_pulse = np.copy(pulse)
    for _ in trange(ITER_NUM - 1):
        current_pulse = SimulatePropagation(current_pulse, N, equation_number, Dmat, gamma, E_sat, g_0, h, tau)
        current_pulse = RightBoundary(current_pulse, omega, delta, Delta, phi)
        current_pulse = SimulatePropagation(current_pulse, N, equation_number, Dmat, gamma, E_sat, g_0, h, tau)
        current_pulse = LeftBoundary(current_pulse, Fresnel_k)
    current_pulse = SimulatePropagation(current_pulse, N, equation_number, Dmat, gamma, E_sat, g_0, h, tau)
    current_pulse = RightBoundary(current_pulse, omega, delta, Delta, phi)
    current_pulse = SimulatePropagation(current_pulse, N, equation_number, Dmat, gamma, E_sat, g_0, h, tau)
    current_pulse = OutputCouplerCondition(current_pulse, Fresnel_k)
    return current_pulse


def OutputCouplerCondition(pulse, Fresnel_k):
    """ Коэффициент отсечения при выходе из волокна """
    return pulse * (1 - Fresnel_k)


def LeftBoundary(pulse, Fresnel_k):
    """ Условие на левой границе резонатора """
    return pulse * Fresnel_k


def RightBoundary(pulse, omega, delta, Delta, phi):
    """ Условие на правой границе резонатора """
    fourierPulse = fft(pulse, axis=1)
    fourierPulse = fourierPulse * np.exp(-(delta - omega)**2 / (2*Delta**2)) * cexp(1j*phi)
    new_pulse = ifft(fourierPulse, axis=1)
    return new_pulse


def SimulatePropagation(pulse, N, equation_number, h, tau, coupling_matrix, beta_2, gamma, E_sat, alpha, g_0):
    """ Строит решение методом SSFM, строит конечное значение поля в каждой сердцевине """

    M = pulse.shape[1]
    w = fftfreq(M, tau) * 2 * pi
    Dmat = get_pade_exponential2(create_freq_matrix(coupling_matrix, beta_2, alpha, g_0, w, h))

    current_energy = np.array([0] * equation_number, dtype=float)
    for n in range(N - 1):
        pulse = SSFMOrder2(pulse, current_energy, Dmat, gamma, E_sat, g_0, h, tau)
    return pulse


def SimulatePropagationNDN(pulse, N, equation_number, h, tau, coupling_matrix, beta_2, gamma, E_sat, alpha, g_0):
    """ Строит решение методом SSFM 1-го порядка с расщеплением вида nonlinear_step Dispersion nonlinear_step,
     используя объединение соседних половинных шагов """

    M = pulse.shape[1]
    w = fftfreq(M, tau) * 2*pi
    Dmat = get_pade_exponential2(create_freq_matrix(coupling_matrix, beta_2, alpha, g_0, w, h))

    current_energy = np.array([0] * equation_number, dtype=float)

    equation_number = len(pulse)
    if g_0 != 0:
        for i in range(equation_number):
            current_energy[i] = get_energy_rectangles(pulse[i], tau)
    nonlinear_step(pulse, gamma, E_sat, g_0, current_energy, h / 2)

    for n in range(N - 1):
        pulse = fft(pulse, axis=1)
        pulse = linear_step(pulse, Dmat)
        pulse = ifft(pulse, axis=1)

        if g_0 != 0:
            for i in range(equation_number):
                current_energy[i] = get_energy_rectangles(pulse[i], tau)
        nonlinear_step(pulse, gamma, E_sat, g_0, current_energy, h)

    if g_0 != 0:
        for i in range(equation_number):
            current_energy[i] = get_energy_rectangles(pulse[i], tau)
    nonlinear_step(pulse, gamma, E_sat, g_0, current_energy, -h / 2)

    return pulse


def SimulatePropagationDND(pulse, N, equation_number, h, tau, coupling_matrix, beta_2, gamma, E_sat, alpha, g_0):
    """ Строит решение методом SSFM 2-го порядка с расщеплением вида Dispersion nonlinear_step Dispersion,
     используя объединение соседних половинных шагов """

    M = pulse.shape[1]
    w = fftfreq(M, tau) * 2 * pi
    DmatH = get_pade_exponential2(create_freq_matrix(coupling_matrix, beta_2, alpha, g_0, w, h))
    DmatH2Plus = get_pade_exponential2(create_freq_matrix(coupling_matrix, beta_2, alpha, g_0, w, h / 2))
    DmatH2Minus = get_pade_exponential2(create_freq_matrix(coupling_matrix, beta_2, alpha, g_0, w, -h / 2))

    pulse = fft(pulse, axis=1)
    pulse = linear_step(pulse, DmatH2Plus)
    pulse = ifft(pulse, axis=1)

    current_energy = np.array([0] * equation_number, dtype=float)

    for n in range(N - 1):
        for i in range(equation_number):
            if g_0[i] != 0:
                current_energy[i] = get_energy_rectangles(pulse[i], tau)
        nonlinear_step(pulse, gamma, E_sat, g_0, current_energy, h)

        pulse = fft(pulse, axis=1)
        pulse = linear_step(pulse, DmatH)
        pulse = ifft(pulse, axis=1)

    pulse = fft(pulse, axis=1)
    pulse = linear_step(pulse, DmatH2Minus)
    pulse = ifft(pulse, axis=1)

    return pulse


def makeFull(tens):
    """ Добавляет последнюю точку по времени из периодичности условий (для полного поля во всех сердцевинах) """
    equation_number, N, M_ = tens.shape
    M = M_ + 1
    new = np.empty((equation_number, N, M), dtype=complex)
    for j in range(equation_number):
        for k in range(N):
            for i in range(M - 1):
                new[j][k][i] = tens[j][k][i]
            new[j][k][M - 1] = tens[j][k][0]
    return new


def makeFull1D(tens, equation_number, M):
    """ Добавляет последнюю точку по времени из периодичности условий (для последней точки по z во всех сердцевинах) """
    new = np.empty((equation_number, M), dtype=complex)
    for j in range(equation_number):
        for i in range(M - 1):
            new[j][i] = tens[j][i]
        new[j][M - 1] = tens[j][0]
    return new


