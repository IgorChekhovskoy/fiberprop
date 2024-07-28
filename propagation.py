from cmath import exp as cexp
from tqdm import trange

from scipy.fft import fft, ifft, fftfreq
from dataclasses import dataclass, field
from typing import Union
import numpy as np
from SSFM_MCF import SSFMOrder2, get_energy_rectangles, nonlinear_step, linear_step
from matrixes import get_central_coupling_matrix, create_freq_matrix, get_pade_exponential2
from pulses import gain_loss_soliton
from drawing import *
from math import sqrt, pi


@dataclass
class ComputationalParameters:
    N: int = 0  # количество шагов, а не точек
    M: int = 0  # количество учитываемых точек по времени
    L1: float = 0.0
    L2: float = 0.0
    T1: float = 0.0
    T2: float = 0.0

    h: float = field(init=False, default=0.0)
    tau: float = field(init=False, default=0.0)

    def __post_init__(self):
        if self.N > 0:
            self.h = (self.L2 - self.L1) / self.N
        else:
            self.h = 0.0

        if self.M > 0:
            self.tau = (self.T2 - self.T1) / self.M
        else:
            self.tau = 0.0


@dataclass
class EquationParameters:

    """ core_configuration - конфигурация сердцевин в MCF:
        0 - 1d круговая с центральной сердцевиной
        1 - 1d круговая без центральной сердцевиной
        2 - 2d квадратная решетка
        3 - 2d гексагональная решетка
        10 - уравнения Манакова
        
        ring_number - число колец для 2d конфигураций. Радиус, ограничивающий некоторое число колец в 2d конфигурациях
        """
    core_configuration: int = 1
    size: int = 1
    ring_number: float = 0
    
    beta_2: Union[float, np.ndarray] = -1.0
    gamma: Union[float, np.ndarray] = 1.0
    E_sat: Union[float, np.ndarray] = 1.0
    alpha: Union[float, np.ndarray] = 0.0
    g_0: Union[float, np.ndarray] = 0.0
    coupling_coefficient: Union[float, np.ndarray] = 1.0
    linear_coefficient: Union[float, np.ndarray] = 0.0
    linear_gain_coefficient: Union[float, np.ndarray] = 0.0
    nonlinear_cubic_coefficient: Union[float, np.ndarray] = 0.0

    def __post_init__(self):
        # Преобразование скалярных параметров в массивы одинаковых значений
        if isinstance(self.beta_2, (int, float)):
            self.beta_2 = np.full(self.size, self.beta_2, dtype=float)
        if isinstance(self.gamma, (int, float)):
            self.gamma = np.full(self.size, self.gamma, dtype=float)
        if isinstance(self.E_sat, (int, float)):
            self.E_sat = np.full(self.size, self.E_sat, dtype=float)
        if isinstance(self.alpha, (int, float)):
            self.alpha = np.full(self.size, self.alpha, dtype=float)
        if isinstance(self.g_0, (int, float)):
            self.g_0 = np.full(self.size, self.g_0, dtype=float)
        if isinstance(self.coupling_coefficient, (int, float)):
            self.coupling_coefficient = np.full(self.size, self.coupling_coefficient, dtype=float)
        # if isinstance(self.linear_coefficient, (int, float)):
        #     self.linear_coefficient = np.full(self.size, self.linear_coefficient, dtype=float)
        # if isinstance(self.linear_gain_coefficient, (int, float)):
        #     self.linear_gain_coefficient = np.full(self.size, self.linear_gain_coefficient, dtype=float)
        # if isinstance(self.nonlinear_cubic_coefficient, (int, float)):
        #     self.nonlinear_cubic_coefficient = np.full(self.size, self.nonlinear_cubic_coefficient, dtype=float)


@dataclass
class Mask:
    """
        Класс Mask представляет структуру, содержащую информацию о связях между сердцевинами.

        Атрибуты:
        ----------
        number_1d : int
            Номер ядра при одномерной нумерации, т.е. при записи системы в матричной форме.
        number_2d_x : int
            Первая координата ядра при двумерной нумерации (например, в гексагональной или квадратной решетке).
        number_2d_y : int
            Вторая координата ядра при двумерной нумерации.
        neighbors : np.ndarray
            Массив индексов соседних ядер.
        """
    number_1d: int
    number_2d_y: int
    number_2d_x: int
    neighbors: np.ndarray


def print_temp_array(temp_array_size, temp_array):
    """
    Печатает временный массив в консоль.

    Параметры:
    ----------
    temp_array_size : int
        Размер временного массива (предполагается квадратный массив).
    temp_array : np.ndarray
        Двумерный массив логических значений (bool), представляющий временный массив.
    """
    i_min, j_min = temp_array_size, temp_array_size
    i_max, j_max = 0, 0

    found = False
    for i in range(temp_array_size):
        for j in range(temp_array_size):
            if temp_array[i][j]:
                if not found:
                    i_min, j_min = i, j
                    i_max, j_max = i, j
                    found = True
                else:
                    if i < i_min: i_min = i
                    if i > i_max: i_max = i
                    if j < j_min: j_min = j
                    if j > j_max: j_max = j

    if not found:
        print("\n")
        return

    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            print("0 " if temp_array[i][j] else "  ", end="")
        print("\n")
    print("\n")


class Solver:

    def __init__(self, com: ComputationalParameters, eq: EquationParameters,
                 pulses=gain_loss_soliton, pulse_params_list=None):
        self.com = com
        self.eq = eq

        # Ensure pulses and pulse_params_list are lists or apply them to all equations
        if not isinstance(pulses, list):
            self.pulses = [pulses] * eq.size
        else:
            self.pulses = pulses

        if pulse_params_list is None:
            self.pulse_params_list = [{}] * eq.size
        elif not isinstance(pulse_params_list, list):
            self.pulse_params_list = [pulse_params_list] * eq.size
        else:
            self.pulse_params_list = pulse_params_list

        # Ensure the lists have the same length as the number of equations
        if len(self.pulses) != self.eq.size:
            raise ValueError("The number of pulse functions must match the number of equations.")
        if len(self.pulse_params_list) != self.eq.size:
            raise ValueError("The number of pulse parameter dictionaries must match the number of equations.")

        self.t = None
        self.z = None
        self.omega = None
        self.D = None
        self.numerical_solution = None
        self.energy = None
        self.peak_power = None
        self.absolute_error = None
        self.C_norm = None
        self.L2_norm = None
        self.analytical_solution = None

        self.linear_coeffs_array = None
        self.nonlinear_cubic_coeffs_array = None
        self.mask_array = None

        self.set_configuration()
        self.initialize_arrays()

    def make_eq_mask(self):
        length = max(int(self.eq.ring_number), self.eq.size)
        temp_array_size = int(2.0 * (1.0 + 3.0 * length * (length + 1.0)))
        temp_array = np.zeros((temp_array_size, temp_array_size), dtype=bool)
        center = temp_array_size // 2

        if self.eq.core_configuration == 0:
            for i in range(self.eq.size + 1):
                temp_array[0][i] = True

        elif self.eq.core_configuration in (1, 10):
            for i in range(self.eq.size):
                temp_array[0][i] = True

        elif self.eq.core_configuration == 2:
            for i in range(temp_array_size):
                for j in range(temp_array_size):
                    if (i - center) ** 2 + (j - center) ** 2 <= self.eq.ring_number ** 2 + 1e-13:
                        temp_array[i][j] = True

        elif self.eq.core_configuration == 3:
            h_i = 1.0
            h_j = 1.0 / sqrt(3.0)
            for i in range(temp_array_size):
                for j in range(temp_array_size):
                    if (h_i * (i - center)) ** 2 + (
                            h_j * (j - center)) ** 2 <= self.eq.ring_number ** 2 * 4.0 / 3.0 + 1e-10 and \
                            (i + j - 2 * center) % 2 == 0:
                        temp_array[i][j] = True

        print_temp_array(temp_array_size, temp_array)

        self.mask_array = []
        index_1d = 0
        for i in range(temp_array_size):
            for j in range(temp_array_size):
                if temp_array[i][j]:
                    self.mask_array.append(Mask(index_1d, i - temp_array_size // 2, j - temp_array_size // 2, []))
                    index_1d += 1

        return index_1d

    def set_configuration(self):
        if self.eq.core_configuration < 0 or (self.eq.core_configuration > 3 and self.eq.core_configuration != 10):
            raise ValueError("Non-existent core configuration!")

        if self.eq.ring_number < 0:
            raise ValueError("ring_number must be positive or zero!")

        self.eq.size = self.make_eq_mask()

        # Initialize arrays
        self.linear_coeffs_array = np.zeros((self.eq.size, self.eq.size),  dtype=float)  # dtype=complex)
        self.nonlinear_cubic_coeffs_array = np.zeros((self.eq.size, self.eq.size), dtype=float)

        if self.eq.core_configuration == 0:
            for j in range(1, self.eq.size):
                self.linear_coeffs_array[0][j] = 1.0 * self.eq.coupling_coefficient[j]
                self.linear_coeffs_array[j][0] = 1.0 * self.eq.coupling_coefficient[j]
                self.linear_coeffs_array[j][j] = -2.0 * self.eq.coupling_coefficient[j]
            for j in range(1, self.eq.size - 1):
                self.linear_coeffs_array[j][j + 1] = 1.0 * self.eq.coupling_coefficient[j]
                self.linear_coeffs_array[j + 1][j] = 1.0 * self.eq.coupling_coefficient[j]
            if self.eq.size > 1:
                self.linear_coeffs_array[1][self.eq.size - 1] = 1.0 * self.eq.coupling_coefficient[1]
                self.linear_coeffs_array[self.eq.size - 1][1] = 1.0 * self.eq.coupling_coefficient[self.eq.size - 1]

        elif self.eq.core_configuration == 1:
            for j in range(self.eq.size - 1):
                self.linear_coeffs_array[j][j + 1] = 1.0 * self.eq.coupling_coefficient[j]
                self.linear_coeffs_array[j + 1][j] = 1.0 * self.eq.coupling_coefficient[j]
            if self.eq.size > 1:
                for j in range(self.eq.size):
                    self.linear_coeffs_array[j][j] = -2.0 * self.eq.coupling_coefficient[j]
                self.linear_coeffs_array[0][self.eq.size - 1] = 1.0 * self.eq.coupling_coefficient[0]
                self.linear_coeffs_array[self.eq.size - 1][0] = 1.0 * self.eq.coupling_coefficient[self.eq.size - 1]

        elif self.eq.core_configuration == 2:
            if self.eq.size > 1:
                for j in range(self.eq.size):
                    self.linear_coeffs_array[j][j] = -4.0 * self.eq.coupling_coefficient[j]
                for j in range(self.eq.size):
                    for k in range(self.eq.size):
                        if j != k:
                            if abs(self.mask_array[j].number_2d_x - self.mask_array[k].number_2d_x) == 1 and \
                                    self.mask_array[j].number_2d_y == self.mask_array[k].number_2d_y:
                                self.linear_coeffs_array[j][k] = 1.0 * self.eq.coupling_coefficient[j]
                            if abs(self.mask_array[j].number_2d_y - self.mask_array[k].number_2d_y) == 1 and \
                                    self.mask_array[j].number_2d_x == self.mask_array[k].number_2d_x:
                                self.linear_coeffs_array[j][k] = 1.0 * self.eq.coupling_coefficient[j]

        elif self.eq.core_configuration == 3:
            if self.eq.size > 1:
                for j in range(self.eq.size):
                    self.linear_coeffs_array[j][j] = -6.0 * self.eq.coupling_coefficient[j]
                for j in range(self.eq.size):
                    for k in range(self.eq.size):
                        if j != k:
                            if abs(self.mask_array[j].number_2d_x - self.mask_array[k].number_2d_x) == 2 and \
                                    self.mask_array[j].number_2d_y == self.mask_array[k].number_2d_y:
                                self.linear_coeffs_array[j][k] = 1.0 * self.eq.coupling_coefficient[j]
                            if abs(self.mask_array[j].number_2d_x - self.mask_array[k].number_2d_x) == 1 and \
                                    abs(self.mask_array[j].number_2d_y - self.mask_array[k].number_2d_y) == 1:
                                self.linear_coeffs_array[j][k] = 1.0 * self.eq.coupling_coefficient[j]

        # for j in range(self.eq.size):
        #     self.linear_coeffs_array[j][j] += self.eq.linear_coefficient[j] - 1j * self.eq.linear_gain_coefficient[j]

        self.get_neighbors()

        # for j in range(self.eq.size):
        #     for k in range(self.eq.size):
        #         if self.eq.core_configuration == 10:
        #             if self.eq.size > 1:
        #                 self.nonlinear_cubic_coeffs_array[j][k] = self.eq.nonlinear_cubic_coefficient[j]
        #             else:
        #                 self.nonlinear_cubic_coeffs_array[j][k] = self.eq.nonlinear_cubic_coefficient[j]
        #         else:
        #             if j != k:
        #                 self.nonlinear_cubic_coeffs_array[j][k] = 0
        #             else:
        #                 self.nonlinear_cubic_coeffs_array[j][j] = self.eq.nonlinear_cubic_coefficient[j]

    def get_neighbors(self):
        for j in range(self.eq.size):
            self.mask_array[j].neighbors.clear()
            for k in range(self.eq.size):
                if (self.linear_coeffs_array[j][k].real != 0 or self.linear_coeffs_array[j][k].imag != 0) and j != k:
                    self.mask_array[j].neighbors.append(k)

    def initialize_arrays(self):
        self.t = np.linspace(self.com.T1, self.com.T2, self.com.M, endpoint=False)
        self.z = np.linspace(self.com.L1, self.com.L2, self.com.N + 1)
        self.omega = fftfreq(self.com.M, self.com.tau) * 2 * pi

        self.numerical_solution = np.zeros((self.com.N + 1, self.eq.size, self.com.M), dtype=complex)
        self.energy = np.zeros((self.eq.size, self.com.N + 1), dtype=float)
        self.peak_power = np.zeros((self.eq.size, self.com.N + 1), dtype=float)

        for k in range(self.eq.size):
            pulse_params = self.filter_params(self.pulses[k], self.pulse_params_list[k])
            if 'z' in self.pulses[k].__code__.co_varnames:
                self.numerical_solution[0][k] = self.pulses[k](t=self.t, z=0,
                                                               **{key: val[k] if isinstance(val, np.ndarray) else val
                                                                  for
                                                                  key, val in pulse_params.items()})
            else:
                self.numerical_solution[0][k] = self.pulses[k](t=self.t,
                                                               **{key: val[k] if isinstance(val, np.ndarray) else val
                                                                  for
                                                                  key, val in pulse_params.items()})

        for k in range(self.eq.size):
            self.energy[k][0] = get_energy_rectangles(self.numerical_solution[0][k], self.com.tau)

        for k in range(self.eq.size):
            self.peak_power[k][0] = np.max(np.abs(self.numerical_solution[0][k]) ** 2)

    def calculate_D_matrix(self):
        # coupling_matrix = get_central_coupling_matrix(self.eq.size)


        self.D = get_pade_exponential2(create_freq_matrix(self.linear_coeffs_array, self.eq.beta_2,
                                                          self.eq.alpha, self.eq.g_0,
                                                          self.omega, self.com.h))

    def filter_params(self, func, pulse_params):
        # Получаем список параметров, которые принимает функция
        func_params = func.__code__.co_varnames[:func.__code__.co_argcount]
        # Фильтруем параметры, чтобы оставить только те, которые нужны функции
        filtered_params = {k: v for k, v in vars(self.eq).items() if k in func_params}
        # Обновляем параметры с учетом pulse_params
        filtered_params.update({k: v for k, v in pulse_params.items() if k in func_params})
        return filtered_params

    def run_numerical_simulation(self):
        for n in trange(self.com.N):
            self.numerical_solution[n+1] = SSFMOrder2(self.numerical_solution[n], self.energy[:, n+1], self.D,
                                                      self.eq.gamma, self.eq.E_sat, self.eq.g_0, self.com.h, self.com.tau)

            for k in range(self.eq.size):
                self.energy[k][n+1] = get_energy_rectangles(self.numerical_solution[n+1][k], self.com.tau)

            for k in range(self.eq.size):
                self.peak_power[k][n+1] = np.max(np.abs(self.numerical_solution[n+1][k]) ** 2)

    def get_analytical_solution(self):
        self.analytical_solution = np.zeros((self.eq.size, self.com.N + 1, self.com.M),
                                            dtype=complex)  # сюда будет записываться решение

        for k in range(self.eq.size):
            pulse_params = self.filter_params(self.pulses[k], self.pulse_params_list[k])
            pulse_params = {key: val[k] if isinstance(val, np.ndarray) else val for key, val in pulse_params.items()}
            for n, z_val in enumerate(self.z):
                if 'z' in self.pulses[k].__code__.co_varnames:
                    self.analytical_solution[k, n] = self.pulses[k](t=self.t, z=z_val, **pulse_params)
                else:
                    self.analytical_solution[k, n] = self.pulses[k](t=self.t, **pulse_params)

    def calculate_error(self):
        self.absolute_error = abs(self.analytical_solution[self.eq.size // 2] -
                                  self.numerical_solution[:, self.eq.size // 2, :])
        self.C_norm = np.max(self.absolute_error[self.com.N])
        print('C norm =\t', self.C_norm)
        self.L2_norm = get_energy_rectangles(self.absolute_error[self.com.N]**2, self.com.tau)
        print('L2 norm =\t', self.L2_norm)

    def plot_error(self):
        name = 'абсолютная_ошибка-case1'
        plot3D(self.z, self.t, self.absolute_error, name)

    def run(self):
        self.calculate_D_matrix()
        self.run_numerical_simulation()
        # self.get_analytical_solution()
        # self.calculate_error()
        # self.plot_error()


def FullPropagation_Simulation(pulse, N, equation_number, h, tau, coupling_matrix, beta_2, gamma, E_sat, alpha, g_0,
                                    Fresnel_k, omega, delta, Delta, phi, ITER_NUM):

    """ Последовательное моделирование ITER_NUM итераций """

    M = pulse.shape[1]
    w = fftfreq(M, tau) * 2*pi
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
    w = fftfreq(M, tau) * 2*pi
    Dmat = get_pade_exponential2(create_freq_matrix(coupling_matrix, beta_2, alpha, g_0, w, h))

    energy = np.array([0] * equation_number, dtype=float)
    for n in range(N):
        pulse = SSFMOrder2(pulse, energy, Dmat, gamma, E_sat, g_0, h, tau)
    return pulse


def SimulatePropagationNDN(pulse, N, equation_number, h, tau, coupling_matrix, beta_2, gamma, E_sat, alpha, g_0):
    """ Строит решение методом SSFM 1-го порядка с расщеплением вида nonlinear_step Dispersion nonlinear_step,
     используя объединение соседних половинных шагов """

    M = pulse.shape[1]
    w = fftfreq(M, tau) * 2*pi
    Dmat = get_pade_exponential2(create_freq_matrix(coupling_matrix, beta_2, alpha, g_0, w, h))

    energy = np.array([0] * equation_number, dtype=float)

    equation_number = len(pulse)
    if g_0 != 0:
        for i in range(equation_number):
            energy[i] = get_energy_rectangles(pulse[i], tau)
    nonlinear_step(pulse, gamma, E_sat, g_0, energy, h / 2)

    for n in range(N):
        pulse = fft(pulse, axis=1)
        pulse = linear_step(pulse, Dmat)
        pulse = ifft(pulse, axis=1)

        if g_0 != 0:
            for i in range(equation_number):
                energy[i] = get_energy_rectangles(pulse[i], tau)
        nonlinear_step(pulse, gamma, E_sat, g_0, energy, h)

    if g_0 != 0:
        for i in range(equation_number):
            energy[i] = get_energy_rectangles(pulse[i], tau)
    nonlinear_step(pulse, gamma, E_sat, g_0, energy, -h / 2)

    return pulse


def SimulatePropagationDND(pulse, N, equation_number, h, tau, coupling_matrix, beta_2, gamma, E_sat, alpha, g_0):
    """ Строит решение методом SSFM 2-го порядка с расщеплением вида Dispersion nonlinear_step Dispersion,
     используя объединение соседних половинных шагов """

    M = pulse.shape[1]
    w = fftfreq(M, tau) * 2*pi
    DmatH = get_pade_exponential2(create_freq_matrix(coupling_matrix, beta_2, alpha, g_0, w, h))
    DmatH2Plus = get_pade_exponential2(create_freq_matrix(coupling_matrix, beta_2, alpha, g_0, w, h / 2))
    DmatH2Minus = get_pade_exponential2(create_freq_matrix(coupling_matrix, beta_2, alpha, g_0, w, -h / 2))

    pulse = fft(pulse, axis=1)
    pulse = linear_step(pulse, DmatH2Plus)
    pulse = ifft(pulse, axis=1)

    energy = np.array([0] * equation_number, dtype=float)

    for n in range(N):
        for i in range(equation_number):
            if g_0[i] != 0:
                energy[i] = get_energy_rectangles(pulse[i], tau)
        nonlinear_step(pulse, gamma, E_sat, g_0, energy, h)

        pulse = fft(pulse, axis=1)
        pulse = linear_step(pulse, DmatH)
        pulse = ifft(pulse, axis=1)

    pulse = fft(pulse, axis=1)
    pulse = linear_step(pulse, DmatH2Minus)
    pulse = ifft(pulse, axis=1)

    return pulse


def makeFull(tens):
    """ Добавляет последнюю точку по времени из периодичности условий (для полного поля во всех сердцевинах) """
    equation_number, N_add, M = tens.shape
    new = np.empty((equation_number, N_add, M+1), dtype=complex)
    for j in range(equation_number):
        for k in range(N_add):
            for i in range(M):
                new[j][k][i] = tens[j][k][i]
            new[j][k][M] = tens[j][k][0]
    return new


def makeFull1D(tens, equation_number, M):
    """ Добавляет последнюю точку по времени из периодичности условий (для последней точки по z во всех сердцевинах) """
    new = np.empty((equation_number, M+1), dtype=complex)
    for j in range(equation_number):
        for i in range(M):
            new[j][i] = tens[j][i]
        new[j][M] = tens[j][0]
    return new


