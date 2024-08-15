from tqdm import trange
from scipy.fft import fftfreq
from dataclasses import dataclass, field
from typing import Union
from math import sqrt, pi
from enum import Enum

from .matrices import create_freq_matrix, get_pade_exponential2, create_simple_dispersion_free_matrix
from .pulses import gain_loss_soliton
from .drawing import *
from .ssfm_mcf import ssfm_order2, get_energy_rectangles
from .stationary_solution_solver import find_stationary_solution

try:
    import torch

    USE_TORCH = True
    from ssfm_mcf_pytorch import get_energy_rectangles_pytorch, nonlinear_step_pytorch, linear_step_pytorch, \
        ssfm_order2_pytorch
except ImportError:
    USE_TORCH = False


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

    beta2: Union[float, np.ndarray, list] = -1.0
    gamma: Union[float, np.ndarray, list] = 1.0
    E_sat: Union[float, np.ndarray, list] = 1.0
    alpha: Union[float, np.ndarray, list] = 0.0
    g_0: Union[float, np.ndarray, list] = 0.0
    coupling_coefficient: Union[float, np.ndarray, list] = 1.0
    linear_coefficient: Union[float, np.ndarray, list] = 0.0
    linear_gain_coefficient: Union[float, np.ndarray, list] = 0.0

    def __post_init__(self):
        # Преобразование скалярных параметров и списков в массивы одинаковых значений
        if isinstance(self.beta2, (int, float, list)):
            self.beta2 = np.array(self.beta2, dtype=float)
            if self.beta2.ndim == 0:
                self.beta2 = np.full(self.size, self.beta2, dtype=float)
        if isinstance(self.gamma, (int, float, list)):
            self.gamma = np.array(self.gamma, dtype=float)
            if self.gamma.ndim == 0:
                self.gamma = np.full(self.size, self.gamma, dtype=float)
        if isinstance(self.E_sat, (int, float, list)):
            self.E_sat = np.array(self.E_sat, dtype=float)
            if self.E_sat.ndim == 0:
                self.E_sat = np.full(self.size, self.E_sat, dtype=float)
        if isinstance(self.alpha, (int, float, list)):
            self.alpha = np.array(self.alpha, dtype=float)
            if self.alpha.ndim == 0:
                self.alpha = np.full(self.size, self.alpha, dtype=float)
        if isinstance(self.g_0, (int, float, list)):
            self.g_0 = np.array(self.g_0, dtype=float)
            if self.g_0.ndim == 0:
                self.g_0 = np.full(self.size, self.g_0, dtype=float)
        if isinstance(self.coupling_coefficient, (int, float, list)):
            self.coupling_coefficient = np.array(self.coupling_coefficient, dtype=float)
            if self.coupling_coefficient.ndim == 0:
                self.coupling_coefficient = np.full(self.size, self.coupling_coefficient, dtype=float)
        # if isinstance(self.linear_coefficient, (int, float)):
        #     self.linear_coefficient = np.full(self.size, self.linear_coefficient, dtype=float)
        # if isinstance(self.linear_gain_coefficient, (int, float)):
        #     self.linear_gain_coefficient = np.full(self.size, self.linear_gain_coefficient, dtype=float)


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


def print_matrix(matrix, name='matrix'):
    """ Функция реализует вывод матрицы в консоль """
    print(f'\n{name}: ')
    for row in matrix:
        print('\t'.join(f'{value: .2f}' for value in row))
    print('\n')


class Solver:

    def __init__(self, com: ComputationalParameters, eq: EquationParameters,
                 pulses=gain_loss_soliton, pulse_params_list=None, use_gpu=False, precision='float64'):
        self.com = com
        self.eq = eq
        self.use_gpu = use_gpu and USE_TORCH  # Устанавливаем режим GPU только если PyTorch доступен
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.precision = precision
        self.dtype = torch.float32 if self.precision == 'float32' else torch.float64
        self.ctype = torch.complex64 if self.precision == 'float32' else torch.complex128

        print("use_gpu =", self.use_gpu)

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
        self.omega2 = None
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
            raise ValueError("Non-existent fiberprop configuration!")

        if self.eq.ring_number < 0:
            raise ValueError("ring_number must be positive or zero!")

        self.eq.size = self.make_eq_mask()

        # Initialize arrays
        self.linear_coeffs_array = np.zeros((self.eq.size, self.eq.size), dtype=float)  # dtype=complex)
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

        print_matrix(self.linear_coeffs_array, "linear_coeffs_array")

        self.get_neighbors()

        for j in range(self.eq.size):
            for k in range(self.eq.size):
                if self.eq.core_configuration == 10:
                    if self.eq.size > 1:
                        self.nonlinear_cubic_coeffs_array[j][k] = self.eq.gamma[j]
                    else:
                        self.nonlinear_cubic_coeffs_array[j][k] = self.eq.gamma[j]
                else:
                    if j != k:
                        self.nonlinear_cubic_coeffs_array[j][k] = 0
                    else:
                        self.nonlinear_cubic_coeffs_array[j][j] = self.eq.gamma[j]

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
        self.omega2 = self.omega ** 2

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
        # coupling_matrix = get_ring_coupling_matrix(self.eq.size)
        if all(self.eq.beta2) == 0.0:
            self.D = create_simple_dispersion_free_matrix(self.linear_coeffs_array, self.eq.alpha, self.eq.g_0,
                                                          self.com.h)
        else:
            self.D = get_pade_exponential2(create_freq_matrix(self.linear_coeffs_array, self.eq.beta2,
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

    # Основная функция моделирования
    def run_numerical_simulation(self, print_modulus=False, print_interval=10, yscale='linear'):

        if self.D is None:
            self.calculate_D_matrix()

        if self.use_gpu:
            # Инициализация тензоров на GPU с использованием to()
            psi_gpu = torch.tensor(self.numerical_solution[0], dtype=self.ctype).to(self.device, non_blocking=True)
            energy_gpu = torch.tensor(self.energy[:, 0], dtype=self.dtype).to(self.device, non_blocking=True)
            D_gpu = torch.tensor(self.D, dtype=self.ctype).to(self.device, non_blocking=True)
            gamma_gpu = torch.tensor(self.eq.gamma, dtype=self.dtype).to(self.device, non_blocking=True)
            E_sat_gpu = torch.tensor(self.eq.E_sat, dtype=self.dtype).to(self.device, non_blocking=True)
            g_0_gpu = torch.tensor(self.eq.g_0, dtype=self.dtype).to(self.device, non_blocking=True)

        # Инициализация графика, если нужно
        if print_modulus:
            fig, ax, line = init_modulus_plot(yscale=yscale)

        for n in trange(self.com.N):
            if self.use_gpu:
                # Выполнение на PyTorch
                psi_gpu = ssfm_order2_pytorch(psi_gpu, energy_gpu, D_gpu, gamma_gpu, E_sat_gpu, g_0_gpu, self.com.h,
                                              self.com.tau)

                # Копирование данных с GPU на CPU в конце итерации, с использованием pinned memory
                self.numerical_solution[n + 1] = psi_gpu.cpu().numpy()
                self.energy[:, n + 1] = energy_gpu.cpu().numpy()
            else:
                # Выполнение на NumPy
                self.numerical_solution[n + 1] = ssfm_order2(self.numerical_solution[n], self.energy[:, n], self.D,
                                                             self.eq.gamma, self.eq.E_sat, self.eq.g_0, self.com.h,
                                                             self.com.tau)

            if self.use_gpu:
                for k in range(self.eq.size):
                    energy_gpu[k] = get_energy_rectangles_pytorch(psi_gpu[k], self.com.tau)
                self.energy[:, n + 1] = energy_gpu.cpu().numpy()
            else:
                for k in range(self.eq.size):
                    self.energy[k][n + 1] = get_energy_rectangles(self.numerical_solution[n + 1][k], self.com.tau)

            for k in range(self.eq.size):
                self.peak_power[k][n + 1] = np.max(np.abs(self.numerical_solution[n + 1][k]) ** 2)

            # Обновление графика через каждые `print_interval` шагов, если включен флаг `print_modulus`
            if print_modulus and (n + 1) % print_interval == 0:
                update_modulus_plot(fig, ax, line, self.numerical_solution[n + 1], n)

        if self.use_gpu:
            self.numerical_solution[-1] = psi_gpu.cpu().numpy()
            self.energy[:, -1] = energy_gpu.cpu().numpy()

        # Закрытие интерактивного режима после завершения симуляции
        if print_modulus:
            finalize_plot()

    def get_analytical_solution(self):
        self.analytical_solution = np.zeros((self.com.N + 1, self.eq.size, self.com.M), dtype=complex)

        for n, z_val in enumerate(self.z):
            for k in range(self.eq.size):
                pulse_params = self.filter_params(self.pulses[k], self.pulse_params_list[k])
                pulse_params = {key: val[k] if isinstance(val, np.ndarray) else val for key, val in
                                pulse_params.items()}
                if 'z' in self.pulses[k].__code__.co_varnames:
                    self.analytical_solution[n, k] = self.pulses[k](t=self.t, z=z_val, **pulse_params)
                else:
                    self.analytical_solution[n, k] = self.pulses[k](t=self.t, **pulse_params)

    def calculate_error(self):
        self.absolute_error = abs(self.analytical_solution[:, self.eq.size // 2, :] -
                                  self.numerical_solution[:, self.eq.size // 2, :])
        self.C_norm = np.max(self.absolute_error[self.com.N])
        print('C norm =\t', self.C_norm)
        self.L2_norm = get_energy_rectangles(self.absolute_error[self.com.N] ** 2, self.com.tau)
        print('L2 norm =\t', self.L2_norm)

    def plot_error(self):
        plot3D(self.z, self.t, self.absolute_error, 'абсолютная_ошибка-case1')
        plot2D(self.z, self.absolute_error[:, self.com.M // 2] / abs(
            self.analytical_solution[:, self.eq.size // 2, self.com.M // 2]),
               'относительная_ошибка_в_пике-case1')

    def run_test(self):
        self.run_numerical_simulation()
        self.get_analytical_solution()
        self.calculate_error()
        self.plot_error()

    def convert_to_dimensional(self, coupling_coefficient, gamma, beta2):
        """ Функция приводит безразмерное решение к размерному виду
        Параметры:
            coupling_coefficient [1/cm]
            gamma [1/(W*m)]
            beta2 [ps^2/km]
        Изменяются величины:
            t, T1, T2, tau [ps]
            z, L1, L2, h [m]
            numerical_solution [W^(1/2)]
            peak_power [W]
            energy [J]
         """

        if self.eq.core_configuration == 0:
            self_coefficient = 2
        elif self.eq.core_configuration == 2:
            self_coefficient = 4
        elif self.eq.core_configuration == 3:
            self_coefficient = 6
        else:
            raise RuntimeError('Unsupportable MCF configuration')

        L = 1 / coupling_coefficient * 1e-2  # [m]
        self.com.L1 *= L
        self.com.L2 *= L
        self.com.h *= L
        self.z *= L

        T = np.sqrt(np.fabs(beta2) / (2 * coupling_coefficient * 1e+5))  # [ps]
        self.com.T1 *= T
        self.com.T2 *= T
        self.com.tau *= T
        self.t *= T

        P = coupling_coefficient * 1e+2 / gamma  # [W]
        K = np.exp(1j * self_coefficient * self.z[:, np.newaxis]) * np.sqrt(P)  # [W^(1/2)]
        K = K[:, np.newaxis, :]
        E = T * P  # [pJ]

        print("C =", coupling_coefficient, "1/cm")
        print("L =", L, "m")
        print("T =", T, "ps")
        print("P =", P, "W")
        print("E =", E, "pJ")

        self.numerical_solution *= K
        self.peak_power *= P
        self.energy *= E

    def find_stationary_solution(self, lambda_val, max_iter=200, tol=1e-11, plot_graphs=False, update_interval=0.01, yscale='linear'):
        self.numerical_solution[0] = find_stationary_solution(self.numerical_solution[0],
                                                              self.com.M, self.com.tau,
                                                              self.linear_coeffs_array,
                                                              self.nonlinear_cubic_coeffs_array,
                                                              -self.eq.beta2 * 0.5,
                                                              self.omega2,
                                                              self.mask_array,
                                                              self.eq.E_sat, self.eq.alpha, self.eq.g_0,
                                                              lambda_val,
                                                              max_iter=max_iter, tol=tol,
                                                              plot_graphs=plot_graphs, update_interval=update_interval, yscale=yscale)


# Функции для графиков
def init_modulus_plot(yscale='linear'):
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], label='|u|')
    ax.set_yscale(yscale)
    ax.set_xlabel('Index')
    ax.set_ylabel('Max Modulus')
    ax.set_ylim(bottom=1e-15)
    ax.legend()
    ax.grid(True)
    plt.ion()
    plt.show()
    return fig, ax, line


def update_modulus_plot(fig, ax, line, data, n):
    max_modulus = np.max(np.abs(data))
    line.set_xdata(range(len(data.flatten())))
    line.set_ydata(np.abs(data.flatten()))
    ax.relim()
    ax.autoscale_view()
    ax.set_title(f"Step {n + 1}: Max modulus |u| = {max_modulus: .4f}")
    fig.canvas.draw()
    fig.canvas.flush_events()


def finalize_plot():
    plt.ioff()
    plt.show()