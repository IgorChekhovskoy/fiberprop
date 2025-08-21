import time

from tqdm import trange
from scipy.fft import fftfreq
from dataclasses import dataclass, field
from typing import Union
from math import sqrt, pi
import os
os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
from numba import njit

from .fiber_geometry import make_eq_mask, CoreConfig, get_core_count
from .matrices import create_freq_matrix, get_pade_exponential2, create_simple_dispersion_free_matrix
from .pulses import zero_pulse
from .drawing import *
from .rk4 import rk4_step
from .ssfm_compact_scheme_mcf import ssfm_order2_ndn_compact_windowed, prepare_compact_solver_for_linear_step, \
    ssfm_order2_dnd_compact_windowed, ssfm_order2_dnd_compact_windowed_short
from .ssfm_mcf import ssfm_order2_ndn, get_energy_rectangles, ssfm_order1_resonator_nocos, \
    ssfm_order1_resonator_fullcos, \
    ssfm_order2_2_in_fourier_space, ssfm_order2_dnd, ssfm_order2_ndn_windowed, ssfm_order2_dnd_windowed_short
from .stationary_solution_solver import find_stationary_solution
from .utils import fft_derivative

from .threading_control import configure_threads, threading_report


try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from .ssfm_mcf_pytorch import ssfm_order2_pytorch, ssfm_order2_dnd_windowed_short_torch
from .rk4_pytorch import rk4_step_torch


@dataclass
class ComputationalParameters:
    """
        Класс для хранения вычислительных параметров моделирования.

        Определяет сеточные параметры и физические границы расчетной области.

        Атрибуты:
        ::
            N (int): Количество шагов по эволюционной переменной (не точек)
            M (int): Количество точек временной сетки
            L1 (float): Начало расчетной области по эволюционной переменной [m]
            L2 (float): Конец расчетной области по эволюционной переменной [m]
            T1 (float): Начало временного окна [ps]
            T2 (float): Конец временного окна [ps]
            damp_length (float): Доля узлов с поглощающими условиями на краях [безразмерная]
            method (str): Численный метод
            window_size (int): размер окна для расчета энергии при моделирвании обратной системы с обратной связью

        Вычисляемые атрибуты:
        ::
            h (float): Шаг по эволюционной переменной [m], вычисляется как (L2-L1)/N
            tau (float): Шаг по времени [ps], вычисляется как (T2-T1)/M

        Пример использования:
            >>> params = ComputationalParameters(N=1000, M=512, L1=0.0, L2=1.0, T1=-10.0, T2=10.0, method="ssfm_order2_ndn")
        """
    N: int = 0
    M: int = 0
    L1: float = 0.0
    L2: float = 0.0
    T1: float = 0.0
    T2: float = 0.0

    h: float = field(init=False, default=0.0)
    tau: float = field(init=False, default=0.0)
    damp_length: float = 0.0

    method: str = "ssfm_order2_ndn"
    window_size: int = -1
    offset_size: int = 0

    def __post_init__(self):
        if self.N > 0:
            self.h = (self.L2 - self.L1) / self.N
        else:
            self.h = 0.0

        if self.M > 0:
            self.tau = (self.T2 - self.T1) / self.M
        else:
            self.tau = 0.0

        if self.window_size == -1:
            self.window_size = self.M

    @staticmethod
    def get_info():
        """
        Функция выводит информацию о параметрах класса и их размерностях
        """
        print('\n\nComputationalParameters:')
        print('\"N\" -- количество шагов по эволюционной переменной, целое число;')
        print('\"M\" -- количество шагов по времени, целое число;')
        print('\"L1, L2\" -- границы расчётной области по эволюционной переменной [m];')
        print('\"T1, T2\" -- границы расчётной области по времени [ps].\n')
        print('\"damp_length\" -- доля узлов временной сетки, на которых по краям действует условие поглощения [1].\n')
        print('\"method\" -- выбранный численный метод.\n')
        print('\"window_size\" -- размер окна для расчета энергии при моделирвании обратной системы с обратной связью.\n')


@dataclass
class EquationParameters:
    """
    Класс для хранения физических параметров уравнений моделирования.

    Содержит параметры, определяющие свойства оптического волокна/резонатора и условия моделирования.

    Атрибуты:
    ::
        core_configuration (CoreConfig): Конфигурация сердцевин (из перечисления CoreConfig)
        size (int): Количество сердцевин в мультисердцевинном волокне
        ring_count (float): Количество коаксиальных колец для 2D конфигураций
        display_debug_info (bool): Выводить отладочную информацию

        beta1 (float | np.ndarray | list): Коэффициент групповой задержки [ps/m]
        beta2 (float | np.ndarray | list): Коэффициент дисперсии групповых скоростей [ps²/m]
        gamma (float | np.ndarray | list): Коэффициент нелинейности Керра [1/(W·m)]
        E_sat (float | np.ndarray | list): Энергия насыщения [pJ]
        alpha (float | np.ndarray | list): Коэффициент потерь [1/m]
        g_0 (float | np.ndarray | list): Ненасыщенное усиление [1/m]
        coupling_coefficient (float | np.ndarray | list): Коэффициент связи между сердцевинами [1/m]
        coupling_matrix (np.ndarray ): (size×size), если задана — перекрывает coupling_coefficient
        noise_amplitude (float): Амплитуда аддитивного белого шума  [sqrt(W/2)]

    Особенности:
    ::
        - Параметры beta1, beta2, gamma, E_sat, alpha, g_0 и coupling_coefficient могут быть заданы как:
          * Скаляр - одинаковое значение для всех сердцевин
          * Список/массив - индивидуальные значения для каждой сердцевины
        - При инициализации скалярные значения автоматически преобразуются в массивы

    Пример использования:
        >>> eq_params = EquationParameters(
        ...     core_configuration=CoreConfig.hexagonal,
        ...     size=7,
        ...     beta2=-20.5,  # Общее значение для всех сердцевин
        ...     gamma=[1.2, 1.3, 1.4, 1.3, 1.2, 1.1, 1.0],  # Индивидуальные значения
        ...     coupling_coefficient=0.5
        ... )
    """
    core_configuration: CoreConfig
    size: int = 1
    ring_count: float = 0
    display_debug_info: bool = False

    mask_array = None

    beta1: Union[float, np.ndarray, list] = 0.0
    beta2: Union[float, np.ndarray, list] = -1.0
    gamma: Union[float, np.ndarray, list] = 1.0
    E_sat: Union[float, np.ndarray, list] = 1.0
    alpha: Union[float, np.ndarray, list] = 0.0
    g_0: Union[float, np.ndarray, list] = 0.0
    coupling_coefficient: Union[float, np.ndarray, list] = 1.0
    coupling_matrix: np.ndarray | None = None  # (size×size), если задана — перекрывает coupling_coefficient
    noise_amplitude: float = 0.0  # амплитуда аддитивного белого шума (на каждом шаге)

    def __post_init__(self):

        if type(self.core_configuration) is not CoreConfig:
            raise ValueError("Non-existent fiberprop configuration!")

        if self.ring_count < 0:
            raise ValueError("ring_count must be positive or zero!")

        if self.core_configuration is CoreConfig.square or self.core_configuration is CoreConfig.hexagonal:
            self.size = get_core_count(self.core_configuration, self.ring_count)

        if self.display_debug_info:
            print("eq.size =", self.size)

        self.mask_array = make_eq_mask(
            core_configuration=self.core_configuration,
            size=self.size,
            ring_count=self.ring_count,
            display_debug_info=self.display_debug_info
        )

        # Преобразование скалярных параметров и списков в массивы одинаковых значений
        def _to_array(x, size):
            arr = np.asarray(x, dtype=float)
            if arr.ndim == 0:
                arr = np.full(size, float(arr), dtype=float)
            return arr

        self.beta1 = _to_array(self.beta1, self.size)
        self.beta2 = _to_array(self.beta2, self.size)
        self.gamma = _to_array(self.gamma, self.size)
        self.E_sat = _to_array(self.E_sat, self.size)
        self.alpha = _to_array(self.alpha, self.size)
        self.g_0 = _to_array(self.g_0, self.size)
        self.coupling_coefficient = _to_array(self.coupling_coefficient, self.size)


    @staticmethod
    def get_info():
        """
        Функция выводит информацию о параметрах класса и их размерностях
        """
        print('\n\nEquationParameters:')
        print('\"core_configuration\" -- конфигурация MCF, объект класса \"CoreConfig\";')
        print('\"size\" -- количество сердцевин в MCF, целое число;')
        print('\"ring_count\" -- количество коаксиальных колец в MCF ?, вещественное число.\n')

        print('\"beta1\" -- коэффициент групповой задержки [ps/m];')
        print('\"beta2\" -- коэффициент дисперсии групповых скоростей [ps^2/m];')
        print('\"gamma\" -- коэффициент нелинейности Керра [1/(W*m)];')
        print('\"E_sat\" -- энергия насыщения [pJ];')
        print('\"alpha\" -- коэффициент потерь [1/m];')
        print('\"g_0\" -- ненасыщенное усиление [1/m];')
        print('\"coupling_coefficient\" -- коэффициент линейных связей [1/m].\n')
        print('\"noise_amplitude\" -- амплитуда аддитивного белого равномерного шума, '
              'добавляемого на каждом шаге [sqrt(W/2)].\n')

        print('При решении время имеет размерность [ps],\n',
              'расстояние имеет размерность [m],\n',
              'мощность имеет размерность [W]\n.')


def print_matrix(matrix, name='matrix'):
    """ Функция реализует вывод матрицы в консоль """
    print(f'\n{name}: ')
    for row in matrix:
        print('\t'.join(f'{value: .2f}' for value in row))
    print('\n')


class Solver:
    """
        Основной класс для моделирования распространения сигнала в многосердцевинном волокне (multicore fiber -- MCF).

        Решает уравнения типа нелинейного уравнения Шрёдингера с учетом:
            - Дисперсии
            - Нелинейности Керра
            - Линейных связей между сердцевинами
            - Потерь/усиления
            - Шумовых эффектов

        Поддерживает как CPU (NumPy, PyTorch), так и GPU (PyTorch) вычисления.

        Параметры:
        ----------
            com : ComputationalParameters
                Вычислительные параметры (сетка, шаги и т.д.)
            eq : EquationParameters
                Физические параметры уравнений
            use_dimensional : bool, optional
                Использовать размерные величины (по умолчанию False)
            pulses : callable или list[callable], optional
                Функция(и) для генерации начальных импульсов
            pulse_params_list : dict или list[dict], optional
                Параметры для функций импульсов
            initial_condition : np.ndarray, optional
                Предварительно вычисленное начальное условие формы (equation_size, M). Нужно задавать на выбор либо
                pulses, либо initial_condition.
            use_gpu : bool, optional
                Использовать GPU для вычислений (требует PyTorch)
            use_torch : bool, optional
                Использовать PyTorch вместо NumPy
            precision : str, optional
                Точность вычислений: 'float32' или 'float64'
            num_threads : int | str | None, optional
                Число потоков для CPU ('default' | 'max' | int). По умолчанию 'default' - сколько задано в системе.
            display_debug_info : bool, optional
                Вывести отладочную информацию

        Атрибуты:
        ---------
            numerical_solution : np.ndarray
                3D массив решения формы (N+1, equation_size, M)
            energy : np.ndarray
                Энергия в каждой сердцевине по длине (equation_size, N+1)
            peak_power : np.ndarray
                Пиковая мощность в каждой сердцевине (equation_size, N+1)
            analytical_solution : np.ndarray
                Аналитическое решение (если доступно)

        Основные методы:
        ----------------
            run_numerical_simulation()     : Основной метод для запуска моделирования
            run_resonator_simulation_*()   : Методы для резонаторных конфигураций
            calculate_error()              : Расчет ошибки относительно аналитического решения
            convert_to_dimensionless()     : Нормировка уравнений к безразмерному виду
            find_stationary_solution()     : Поиск стационарных решений

        Примеры использования:
        ----------------------
        # Инициализация с аналитическими импульсами
            solver = Solver(
                com_params,
                eq_params,
                pulses=sech_pulse,
                pulse_params_list={'A': [1.0, 0.9, 1.1], 't0': 0}
            )

        # Инициализация с готовым начальным условием
            custom_initial = np.random.randn(7, 8192) + 1j*np.random.randn(7, 8192)
            solver = Solver(
                com_params,
                eq_params,
                initial_condition=custom_initial,
                use_torch=True
            )

        # Запуск симуляции и визуализация
            solver.run_numerical_simulation(draw_modulus=True)
            solver.plot_error()
        """

    def __init__(
            self,
            com: ComputationalParameters,
            eq: EquationParameters,
            stored_steps_count: int | None = None,
            use_dimensional=False,
            pulses=zero_pulse,
            pulse_params_list=None,
            initial_condition: np.ndarray = None,
            use_gpu=False,
            use_torch=False,
            precision='float64',
            num_threads: Union[int, str, None] = "default",
            display_debug_info=False
    ):
        configure_threads(num_threads)

        if display_debug_info:
            print(threading_report())

        self.gamma_h = None
        self.g0_h = None
        self.exp_g0h = None
        self.exp_2g0h = None

        self.gamma_h_half = None
        self.g0_h_half = None
        self.exp_g0h_half = None
        self.exp_2g0h_half = None

        self.E_sat_t = None
        self.g0_t = None

        self._taper_np = None
        self._taper_t = None

        self.com = com
        self.eq = eq
        self.use_dimensional = use_dimensional  # безразмерная или размерная задача
        self.precision = precision
        self.use_gpu = use_gpu and _TORCH_AVAILABLE  # Устанавливаем режим GPU только если PyTorch доступен
        self.use_torch = (use_torch and _TORCH_AVAILABLE) or (self.use_gpu and _TORCH_AVAILABLE)
        self.device = None
        if self.use_torch:
            self.device = torch.device('cuda' if self.use_gpu else 'cpu')

            self.dtype = torch.float32 if self.precision == "float32" else torch.float64
            self.ctype = torch.complex64 if self.precision == "float32" else torch.complex128
            # NumPy-эквиваленты — НУЖНЫ для осей/Numba/SciPy
            self._np_rdtype = np.float32 if self.precision == "float32" else np.float64
            self._np_cdtype = np.complex64 if self.precision == "float32" else np.complex128
        else:
            self.dtype = np.float32 if self.precision == "float32" else np.float64
            self.ctype = np.complex64 if self.precision == "float32" else np.complex128
            # при numpy-режиме это те же типы
            self._np_rdtype = self.dtype
            self._np_cdtype = self.ctype

        self.display_debug_info = display_debug_info

        if self.display_debug_info:
            if self.use_gpu:
                print("Using GPU", end=' ')
            else:
                print("Using CPU", end=' ')

            if self.use_torch:
                print("with PyTorch")
            else:
                print("with NumPy")

        self.linear_coeffs_array = None
        self.nonlinear_cubic_coeffs_array = None

        self.set_configuration()

        self.has_beta = not (np.all(self.eq.beta1 == 0.0) and np.all(self.eq.beta2 == 0.0))

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
        self.D_half = None
        self.invD_half = None
        self.numerical_solution = None
        self.numerical_solution_time = None
        self.energy = None
        self.peak_power = None
        self.phase_by_z = None
        self.ind_by_z_for_phase = int(self.com.M / 2)
        self.absolute_error = None
        self.C_norm = None
        self.L2_norm = None
        self.analytical_solution = None

        self.stored_steps_count = (
            com.N + 1 if stored_steps_count is None
            else max(2, min(stored_steps_count, com.N + 1))
        )
        self._save_every = max(1, round(com.N / (self.stored_steps_count - 1)))

        # Инициализация массивов
        self.initialize_arrays(initial_condition, pulses)

        # Обработка начальных условий
        if initial_condition is not None:
            self.validate_initial_condition(initial_condition)
            self.apply_initial_condition(initial_condition)
        elif pulses is not zero_pulse:
            self.initialize_with_pulses(pulses, pulse_params_list)

        self._prepare_taper()

    def set_configuration(self):

        # Initialize arrays
        self.linear_coeffs_array = np.zeros((self.eq.size, self.eq.size), dtype=float)  # dtype=complex)
        self.nonlinear_cubic_coeffs_array = np.zeros((self.eq.size, self.eq.size), dtype=float)

        cm = getattr(self.eq, "coupling_matrix", None)
        if cm is not None:
            A = np.asarray(cm)
            if A.shape != (self.eq.size, self.eq.size):
                raise ValueError(
                    f"EquationParameters.coupling_matrix must have shape {(self.eq.size, self.eq.size)}, "
                    f"got {A.shape}"
                )
            # симметризуем на всякий случай (реальные связи), приводим к float (у нас dtype=float)
            A = 0.5 * (A + A.T)
            self.linear_coeffs_array[:, :] = A.real
        else:
            central_coef = 0.0 if self.use_dimensional else 1.0  # у размерной задачи на диагонали должны быть нули
            if self.eq.core_configuration is CoreConfig.ring_with_center:
                for j in range(1, self.eq.size):
                    self.linear_coeffs_array[0][j] = 1.0 * self.eq.coupling_coefficient[j]
                    self.linear_coeffs_array[j][0] = 1.0 * self.eq.coupling_coefficient[j]
                    self.linear_coeffs_array[j][j] = -2.0 * self.eq.coupling_coefficient[j] * central_coef
                for j in range(1, self.eq.size - 1):
                    self.linear_coeffs_array[j][j + 1] = 1.0 * self.eq.coupling_coefficient[j]
                    self.linear_coeffs_array[j + 1][j] = 1.0 * self.eq.coupling_coefficient[j]
                if self.eq.size > 1:
                    self.linear_coeffs_array[1][self.eq.size - 1] = 1.0 * self.eq.coupling_coefficient[1]
                    self.linear_coeffs_array[self.eq.size - 1][1] = 1.0 * self.eq.coupling_coefficient[self.eq.size - 1]

            elif self.eq.core_configuration is CoreConfig.empty_ring:
                for j in range(self.eq.size - 1):
                    self.linear_coeffs_array[j][j + 1] = 1.0 * self.eq.coupling_coefficient[j]
                    self.linear_coeffs_array[j + 1][j] = 1.0 * self.eq.coupling_coefficient[j]
                if self.eq.size > 1:
                    for j in range(self.eq.size):
                        self.linear_coeffs_array[j][j] = -2.0 * self.eq.coupling_coefficient[j] * central_coef
                    self.linear_coeffs_array[0][self.eq.size - 1] = 1.0 * self.eq.coupling_coefficient[0]
                    self.linear_coeffs_array[self.eq.size - 1][0] = 1.0 * self.eq.coupling_coefficient[self.eq.size - 1]

            elif self.eq.core_configuration is CoreConfig.square:
                if self.eq.size > 1:
                    for j in range(self.eq.size):
                        self.linear_coeffs_array[j][j] = -4.0 * self.eq.coupling_coefficient[j] * central_coef
                    for j in range(self.eq.size):
                        for k in range(self.eq.size):
                            if j != k:
                                if abs(self.eq.mask_array[j].number_2d_x - self.eq.mask_array[k].number_2d_x) == 1 and \
                                        self.eq.mask_array[j].number_2d_y == self.eq.mask_array[k].number_2d_y:
                                    self.linear_coeffs_array[j][k] = 1.0 * self.eq.coupling_coefficient[j]
                                if abs(self.eq.mask_array[j].number_2d_y - self.eq.mask_array[k].number_2d_y) == 1 and \
                                        self.eq.mask_array[j].number_2d_x == self.eq.mask_array[k].number_2d_x:
                                    self.linear_coeffs_array[j][k] = 1.0 * self.eq.coupling_coefficient[j]

            elif self.eq.core_configuration is CoreConfig.hexagonal:
                if self.eq.size > 1:
                    for j in range(self.eq.size):
                        self.linear_coeffs_array[j][j] = -6.0 * self.eq.coupling_coefficient[j] * central_coef
                    for j in range(self.eq.size):
                        for k in range(self.eq.size):
                            if j != k:
                                if abs(self.eq.mask_array[j].number_2d_x - self.eq.mask_array[k].number_2d_x) == 2 and \
                                        self.eq.mask_array[j].number_2d_y == self.eq.mask_array[k].number_2d_y:
                                    self.linear_coeffs_array[j][k] = 1.0 * self.eq.coupling_coefficient[j]
                                if abs(self.eq.mask_array[j].number_2d_x - self.eq.mask_array[k].number_2d_x) == 1 and \
                                        abs(self.eq.mask_array[j].number_2d_y - self.eq.mask_array[k].number_2d_y) == 1:
                                    self.linear_coeffs_array[j][k] = 1.0 * self.eq.coupling_coefficient[j]

        if self.display_debug_info:
            print_matrix(self.linear_coeffs_array, "linear_coeffs_array")

        for j in range(self.eq.size):
            for k in range(self.eq.size):
                if self.eq.core_configuration is CoreConfig.manakov_eq:
                    if self.eq.size > 1:
                        self.nonlinear_cubic_coeffs_array[j][k] = self.eq.gamma[j]
                    else:
                        self.nonlinear_cubic_coeffs_array[j][k] = self.eq.gamma[j]
                else:
                    if j != k:
                        self.nonlinear_cubic_coeffs_array[j][k] = 0
                    else:
                        self.nonlinear_cubic_coeffs_array[j][j] = self.eq.gamma[j]

    def initialize_arrays(self, initial_condition=None, pulses=None):
        """Создает пустые массивы для хранения результатов"""
        self.t = np.linspace(self.com.T1, self.com.T2, self.com.M, endpoint=False, dtype=self._np_rdtype)
        self.z = np.linspace(self.com.L1, self.com.L2, self.com.N + 1, dtype=self._np_rdtype)
        self.omega = (fftfreq(self.com.M, self.com.tau) * 2 * pi).astype(self._np_rdtype)
        self.omega2 = (self.omega ** 2).astype(self._np_rdtype)

        if initial_condition is not None or pulses is not zero_pulse:
            self.numerical_solution = np.zeros((self.stored_steps_count, self.eq.size, self.com.M), dtype=self._np_cdtype)
        else:
            self.numerical_solution_time = np.zeros((self.eq.size, self.com.N + 1), dtype=self._np_cdtype)

        self.energy = np.zeros((self.eq.size, self.com.N + 1), dtype=self._np_rdtype)
        self.peak_power = np.zeros((self.eq.size, self.com.N + 1), dtype=self._np_rdtype)
        self.phase_by_z = np.zeros((self.eq.size, self.com.N + 1), dtype=self._np_rdtype)

    def validate_initial_condition(self, initial_condition: np.ndarray):
        """Проверяет корректность начального условия"""
        if not isinstance(initial_condition, np.ndarray):
            raise TypeError("Initial condition must be a numpy array")

        expected_shape = (self.eq.size, self.com.M)
        if initial_condition.shape != expected_shape:
            raise ValueError(
                f"Invalid initial condition shape: {initial_condition.shape}. "
                f"Expected: {expected_shape}"
            )

    def apply_initial_condition(self, initial_condition: np.ndarray):
        """Применяет заданное начальное условие"""
        # Конвертируем в комплексный тип если нужно
        self.numerical_solution[0] = initial_condition.astype(self._np_cdtype, copy=False)

        # Рассчитываем начальные характеристики
        for k in range(self.eq.size):
            self.energy[k][0] = get_energy_rectangles(self.numerical_solution[0][k], self.com.tau)
            self.peak_power[k][0] = np.max(np.abs(self.numerical_solution[0][k]) ** 2)
            if 0 <= self.ind_by_z_for_phase < self.com.N:
                self.phase_by_z[k][0] = np.angle(self.numerical_solution[0][k][self.ind_by_z_for_phase])

    def initialize_with_pulses(self, pulses, pulse_params_list):
        """Инициализация с помощью функций-генераторов импульсов"""
        if not isinstance(pulses, list):
            self.pulses = [pulses] * self.eq.size
        else:
            self.pulses = pulses

        if pulse_params_list is None:
            self.pulse_params_list = [{}] * self.eq.size
        elif not isinstance(pulse_params_list, list):
            self.pulse_params_list = [pulse_params_list] * self.eq.size
        else:
            self.pulse_params_list = pulse_params_list

        if len(self.pulses) != self.eq.size:
            raise ValueError("Number of pulse functions must match the number of equations")
        if len(self.pulse_params_list) != self.eq.size:
            raise ValueError("Number of pulse parameter dictionaries must match the number of equations")

        for k in range(self.eq.size):
            pulse_params = self.filter_params(self.pulses[k], self.pulse_params_list[k])

            pulse_params = {name: (val[k] if isinstance(val, np.ndarray) else val)
                            for name, val in pulse_params.items()}

            if 'z' in self.pulses[k].__code__.co_varnames:
                self.numerical_solution[0][k] = self.pulses[k](t=self.t, z=0, **pulse_params)
            else:
                self.numerical_solution[0][k] = self.pulses[k](t=self.t, **pulse_params)

        for k in range(self.eq.size):
            self.energy[k][0] = get_energy_rectangles(self.numerical_solution[0][k], self.com.tau)
            self.peak_power[k][0] = np.max(np.abs(self.numerical_solution[0][k]) ** 2)
            if 0 <= self.ind_by_z_for_phase < self.com.N:
                self.phase_by_z[k][0] = np.angle(self.numerical_solution[0][k][self.ind_by_z_for_phase])

    # ─── solver.py ─────────────────────────────────────────────
    def calculate_D_matrix(self, h) -> None:
        n, M = self.eq.size, self.com.M

        if np.all(self.eq.beta1 == 0.0) and np.all(self.eq.beta2 == 0.0):
            # build in double
            D64 = create_simple_dispersion_free_matrix(
                self.linear_coeffs_array.astype(np.float64, copy=False),
                self.eq.alpha.astype(np.float64, copy=False),
                self.eq.g_0.astype(np.float64, copy=False),
                float(h)
            )  # (n,n), complex128
            self.D = D64.astype(self._np_cdtype, copy=False)
        else:
            C64 = self.linear_coeffs_array.astype(np.float64, copy=False)
            b1 = self.eq.beta1.astype(np.float64, copy=False)
            b2 = self.eq.beta2.astype(np.float64, copy=False)
            a = self.eq.alpha.astype(np.float64, copy=False)
            g0 = self.eq.g_0.astype(np.float64, copy=False)
            om = self.omega.astype(np.float64, copy=False)

            D_flat64 = get_pade_exponential2(
                create_freq_matrix(C64, b1, b2, a, g0, om, float(h))
            )  # (n², M), complex128
            self.D = D_flat64.reshape(n, n, M).astype(self._np_cdtype, copy=False)

        if self.use_torch:
            self.D_pytorch = torch.as_tensor(self.D, dtype=self.ctype, device=self.device)

    def calculate_invD_half(self) -> None:
        """
        Строит обратную матрицу к D_half.
        """
        # Поддержка обоих случаев:
        # • D_half 3D: (n, n, M) – инвертируем покомпонентно по частоте
        # • D_half 2D: (n, n)    – одна общая матрица для всех ω (β1=β2=0)
        if self.D_half.ndim == 2:
            self.invD_half = np.linalg.inv(self.D_half)
            return

        n, M = self.eq.size, self.com.M
        self.invD_half = np.empty((n, n, M), dtype=self.D_half.dtype)
        for i in range(M):
            self.invD_half[:, :, i] = np.linalg.inv(self.D_half[:, :, i])

    def prepare_halfstep_constants(self):
        if self.gamma_h_half is not None:
            return

            # ── 1. CPU-массивы ───────────────────────────────────────────────
        self.gamma_h = self.eq.gamma *  self.com.h
        self.g0_h = self.eq.g_0 *  self.com.h
        self.exp_g0h = np.exp(self.g0_h)
        self.exp_2g0h = np.exp(2.0 * self.g0_h)

        self.gamma_h_half = 0.5 * self.gamma_h  # (n,)
        self.g0_h_half = 0.5 * self.g0_h
        self.exp_g0h_half = np.exp(self.g0_h_half)
        self.exp_2g0h_half = np.exp(2.0 * self.g0_h_half)

        # ── 2. PyTorch-тензоры (если используем torch) ───────────────────
        if self.use_torch:
            to_t = lambda arr: torch.as_tensor(arr,
                                               dtype=self.dtype,
                                               device=self.device)

            self.gamma_h_t = to_t(self.gamma_h)
            self.g0_h_t = to_t(self.g0_h)
            self.exp_g0h_t = to_t(self.exp_g0h)
            self.exp_2g0h_t = to_t(self.exp_2g0h)

            self.gamma_h_half_t = to_t(self.gamma_h_half)
            self.g0_h_half_t = to_t(self.g0_h_half)
            self.exp_g0h_half_t = to_t(self.exp_g0h_half)
            self.exp_2g0h_half_t = to_t(self.exp_2g0h_half)

            # Дополнительно кешируем E_sat и g0 на том же устройстве
            if self.E_sat_t is None:
                self.E_sat_t = to_t(self.eq.E_sat)
            if self.g0_t is None:
                self.g0_t = to_t(self.eq.g_0)

    def _prepare_taper(self):
        """
        Создаёт и кэширует одномерную маску taper (cos^6) для
        поглощающих граничных условий (ABC).

        ▸ Маска рассчитывается **один раз** при инициализации Solver
          или после изменения self.com.damp_length / self.com.M.

        ▸ Хранится
            self._taper_np : ndarray(M,)        – для CPU/NumPy
            self._taper_t  : torch.Tensor(M,)   – для PyTorch-ветки,
                                                 на том же device, dtype.

        ▸ Алгоритм:
            1. Доля узлов с поглощением  L = int(M * damp_length).
            2. Левая/правая кромка — окно cos⁶:
               taper[i] = cos(π/2 · (L−i)/L)^6,  i = 0…L-1.
            3. Центр массива заполняется единицами.

        ▸ Если damp_length == 0.0, маска не создаётся (None),
          а вызовы apply_absorbing_boundary* просто возвращают psi
          без изменений.

        ▸ После изменения размерности сетки или damp_length
          вызовите повторно self._prepare_taper().

        Возвращаемое значение: None (побочный эффект — заполнены
        self._taper_np / self._taper_t).
        """
        M = self.com.M
        Lp = self.com.damp_length  # доля PML
        if Lp == 0.0:
            self._taper_np = None
            self._taper_t = None
            return

        p = 1  # степень косинуса
        q = 0  # степень полинома
        σ = 1.0  # амплитуда затухания
        κ = 1.0  # фазовый поворот (≈σ)
        L = int(M * Lp)

        # левая/правая кромка
        i = np.arange(L, dtype=float)
        x = (L - i) / L  # 1…0
        # edge = np.exp(-(σ + 1j * κ) * x ** q)
        # edge = np.cos(np.pi / 2 * (L - i) / L) ** p
        cos_part = np.cos(np.pi / 2 * x / 4) ** p
        exp_part = np.exp(-(σ + 1j * κ) * x ** q)
        edge = cos_part# * exp_part

        taper = np.ones(self.com.M, dtype=self._np_cdtype)
        taper[:L] = edge.astype(self._np_rdtype)
        taper[-L:] = edge[::-1].astype(self._np_rdtype)
        self._taper_np = taper
        if self.use_torch:
            self._taper_t = torch.as_tensor(taper, dtype=self.ctype, device=self.device)

    def filter_params(self, func, pulse_params):
        # Получаем список параметров, которые принимает функция
        func_params = func.__code__.co_varnames[:func.__code__.co_argcount]
        # Фильтруем параметры, чтобы оставить только те, которые нужны функции
        filtered_params = {k: v for k, v in vars(self.eq).items() if k in func_params}
        # Обновляем параметры с учетом pulse_params
        filtered_params.update({k: v for k, v in pulse_params.items() if k in func_params})
        return filtered_params


    def calculate_metrics(self, psi, n, save_every, save_idx):
        abs2 = np.abs(psi) ** 2
        self.energy[:, n + 1] = abs2.sum(1) * self.com.tau
        self.peak_power[:, n + 1] = abs2.max(1)

        if 0 <= self.ind_by_z_for_phase < self.com.M:
            self.phase_by_z[:, n + 1] = np.angle(psi[:, self.ind_by_z_for_phase])

        # ---------- сохранить шаг? -------------------------------
        is_save_step = ((n + 1) % save_every == 0) or (n == self.com.N - 1)
        if is_save_step:
            save_idx += 1
            if save_idx < self.stored_steps_count:
                self.numerical_solution[save_idx] = psi


    # Основная функция моделирования
    def run_numerical_simulation(
            self,
            draw_modulus: bool = False,
            draw_interval: int = 10,
            save_gif: bool = False,
            yscale: str = "linear"
    ) -> float:
        """
        Основной цикл расчёта.

        • Все вычисления – на GPU (если self.use_torch = True).
        • Буфер gpu_buffer хранит ≤ draw_interval шагов.
        • Копирование GPU→CPU выполняется асинхронно (non_blocking=True) одним батчем,
          когда (n+1) % draw_interval == 0 или в самом конце.
        • На CPU попадают:
            – срезы поля psi,
            – энергия,
            – пиковая мощность.
          Формат и значения идентичны прежней версии.

          Возвращает время работы без учета создания и вычисления вспомогательных массивов
        """

        # ───── инициализация ────────────────────────────────────────────────────

        if self.com.method == "ssfm_order2_ndn" or self.com.method == "ssfm_order2_ndn_windowed":
            if self.D is None:
                self.calculate_D_matrix(self.com.h)

        if self.com.method == "ssfm_order2_dnd" or self.com.method == "ssfm_order2_dnd_windowed_short":
            if self.D_half is None:
                self.calculate_D_matrix(self.com.h * 0.5)
                self.D_half = self.D.copy()

                if self.D_half.ndim == 3:
                    d_perm = self.D_half.transpose(2, 0, 1)  # -> (M, n, n)
                    result_perm = d_perm @ d_perm
                    self.D = result_perm.transpose(1, 2, 0)  # -> (n, n, M)
                else:
                    # β1=β2=0: оператор не зависит от ω, D_half — (n,n)
                    self.D = self.D_half @ self.D_half

            if self.invD_half is None:
                self.calculate_invD_half()

        if self.com.method == "ssfm_order2_ndn_compact_windowed":
            prepare_compact_solver_for_linear_step(self, self.com.h)

        if self.com.method == "ssfm_order2_dnd_compact_windowed" or self.com.method == "ssfm_order2_dnd_compact_windowed_short":
            prepare_compact_solver_for_linear_step(self, self.com.h * 0.5)

        if self.gamma_h_half is None:
            self.prepare_halfstep_constants()

        save_every = self._save_every  # шаг между сохранёнными
        save_idx = 0  # 0-й шаг уже записан
        tau = self.com.tau

        # ───── график │ опционально ──────────────────────────────────────
        if draw_modulus:
            fig, ax, line = init_modulus_plot(
                save_gif=save_gif, yscale=yscale, scaling_mode='history',
                t=self.t
            )

        # ─────────────────────────────────────── CPU-ветка ───────────────
        psi_next = self.numerical_solution[0]

        t_start = time.time()

        if not self.use_torch:
            if self.com.method == "ssfm_order2_dnd_windowed_short":
                self.numerical_solution[-1] = ssfm_order2_dnd_windowed_short(self, window_size=self.com.window_size,
                                                                             damp_length=self.com.damp_length,
                                                                             disable_progress_bar=not self.display_debug_info)
                self.calculate_metrics(self.numerical_solution[-1], self.com.N - 1, save_every, save_idx)
            elif self.com.method == "ssfm_order2_dnd_compact_windowed_short":
                self.numerical_solution[-1] = ssfm_order2_dnd_compact_windowed_short(self, window_size=self.com.window_size, damp_length=self.com.damp_length)
                self.calculate_metrics(self.numerical_solution[-1], self.com.N - 1, save_every, save_idx)
            else:
                for n in trange(self.com.N, disable=not self.display_debug_info):
                    if self.com.method == "ssfm_order2_ndn":
                        psi_next = ssfm_order2_ndn(
                            psi_next,
                            self.energy[:, n],
                            self,
                            self.com.h, tau,
                            self.com.damp_length,
                            self.eq.noise_amplitude,
                        )
                    elif self.com.method == "ssfm_order2_dnd":
                        psi_next = ssfm_order2_dnd(
                            psi_next,
                            self.energy[:, n],
                            self,
                            self.com.h, tau,
                            self.com.damp_length,
                            self.eq.noise_amplitude,
                        )
                    elif self.com.method == "ssfm_order2_ndn_compact_windowed":
                        psi_next = ssfm_order2_ndn_compact_windowed(
                            psi_next,
                            self.energy[:, n],
                            self,
                            self.com.h, tau, self.com.M,
                            self.com.damp_length,
                            self.eq.noise_amplitude,
                        )
                    elif self.com.method == "ssfm_order2_dnd_compact_windowed":
                        psi_next = ssfm_order2_dnd_compact_windowed(
                            psi_next,
                            self.energy[:, n],
                            self,
                            self.com.h, tau, self.com.M,
                            self.com.damp_length,
                            self.eq.noise_amplitude,
                        )

                    # ---------- метрики --------------------------------------
                    self.calculate_metrics(psi_next, n, save_every, save_idx)

                    # ---------- обновить график ------------------------------
                    if draw_modulus and ((n + 1) % draw_interval == 0):
                        update_modulus_plot(fig, ax, line, psi_next, self.t, n)

        # ─────────────────────────────────────── GPU-ветка ───────────────
        else:
            psi_gpu = torch.as_tensor(
                self.numerical_solution[0], dtype=self.ctype, device=self.device
            )

            if self.com.method == "ssfm_order2_dnd_windowed_short":

                self.numerical_solution[-1] = ssfm_order2_dnd_windowed_short_torch(
                    self,
                    self.com.window_size,
                    self.com.damp_length,
                    disable_progress_bar=not self.display_debug_info
                ).cpu().numpy()

                self.calculate_metrics(self.numerical_solution[-1], self.com.N - 1, save_every, save_idx)
            else:
                for n in trange(self.com.N, disable=not self.display_debug_info):
                    # ---------- очередной шаг (GPU) --------------------------
                    psi_gpu = ssfm_order2_pytorch(
                        psi_gpu, self.energy[:, n],  # пред. энергия (CPU)
                        self,
                        self.com.h, tau,
                        self.com.damp_length,
                        self.eq.noise_amplitude,
                    )

                    # ---------- метрики (GPU → CPU) --------------------------
                    abs2 = psi_gpu.abs() ** 2
                    energy_step = (abs2.sum(1) * tau).cpu().numpy()
                    peak_step = abs2.max(1).values.cpu().numpy()
                    self.energy[:, n + 1] = energy_step
                    self.peak_power[:, n + 1] = peak_step

                    if 0 <= self.ind_by_z_for_phase < self.com.M:
                        self.phase_by_z[:, n + 1] = np.angle(psi_gpu[:, self.ind_by_z_for_phase])

                    # ---------- сохранить шаг? -------------------------------
                    is_save_step = ((n + 1) % save_every == 0) or (n == self.com.N - 1)
                    if is_save_step:
                        save_idx += 1
                        if save_idx < self.stored_steps_count:
                            self.numerical_solution[save_idx] = (
                                psi_gpu.detach().cpu().numpy()
                            )

                    # ---------- обновить график ------------------------------
                    if draw_modulus and ((n + 1) % draw_interval == 0):
                        psi_cpu_for_plot = psi_gpu.detach().cpu().numpy()
                        update_modulus_plot(
                            fig, ax, line, psi_cpu_for_plot, self.t, n
                        )

        # ───── финализация графика ───────────────────────────────────────
        if draw_modulus:
            finalize_plot()

        return time.time() - t_start

    # Основная функция моделирования
    def run_numerical_simulation_in_frequency_domain(
            self,
            draw_modulus: bool = False,
            draw_interval: int = 10,
            save_gif: bool = False,
            yscale: str = "linear"
    ):
        """
        Основной цикл расчёта.

        • Все вычисления – на GPU (если self.use_torch = True).
        • Буфер gpu_buffer хранит ≤ draw_interval шагов.
        • Копирование GPU→CPU выполняется асинхронно (non_blocking=True) одним батчем,
          когда (n+1) % draw_interval == 0 или в самом конце.
        • На CPU попадают:
            – срезы поля psi,
            – энергия,
            – пиковая мощность.
          Формат и значения идентичны прежней версии.
        """

        # ───── инициализация ────────────────────────────────────────────────────
        if self.D is None:
            self.calculate_D_matrix(self.com.h / 2)
        if self.gamma_h_half is None:
            self.prepare_halfstep_constants()

        save_every = self._save_every  # шаг между сохранёнными
        save_idx = 0  # 0-й шаг уже записан
        tau = self.com.tau

        # ───── график │ опционально ──────────────────────────────────────
        if draw_modulus:
            fig, ax, line = init_modulus_plot(
                save_gif=save_gif, yscale=yscale, scaling_mode='history',
                t=np.fft.fftshift(self.omega)
            )

        # ─────────────────────────────────────── CPU-ветка ───────────────
        psi_next = self.numerical_solution[0]

        if not self.use_torch:
            for n in trange(self.com.N, disable=not self.display_debug_info):
                # ---------- очередной шаг --------------------------------
                psi_next = ssfm_order2_2_in_fourier_space(
                    psi_next,  # последний сохранённый
                    self.energy[:, n],  # энергия предыдущего
                    self,
                    self.com.h, tau,
                )

                # ---------- метрики --------------------------------------
                abs2 = np.abs(psi_next) ** 2
                self.energy[:, n + 1] = abs2.sum(1) * tau
                self.peak_power[:, n + 1] = abs2.max(1)

                # ---------- сохранить шаг? -------------------------------
                is_save_step = ((n + 1) % save_every == 0) or (n == self.com.N - 1)
                if is_save_step:
                    save_idx += 1
                    if save_idx < self.stored_steps_count:
                        self.numerical_solution[save_idx] = psi_next

                # ---------- обновить график ------------------------------
                if draw_modulus and ((n + 1) % draw_interval == 0):
                    update_modulus_plot(fig, ax, line, np.fft.ifft(psi_next), self.z, n)

        # ─────────────────────────────────────── GPU-ветка ───────────────
        else:
            psi_gpu = torch.as_tensor(
                self.numerical_solution[0], dtype=self.ctype, device=self.device
            )

            for n in trange(self.com.N, disable=not self.display_debug_info):
                # ---------- очередной шаг (GPU) --------------------------
                psi_gpu = ssfm_order2_pytorch(
                    psi_gpu, self.energy[:, n],  # пред. энергия (CPU)
                    self,
                    self.com.h, tau,
                    self.com.damp_length,
                    self.eq.noise_amplitude,
                )

                # ---------- метрики (GPU → CPU) --------------------------
                abs2 = psi_gpu.abs() ** 2
                energy_step = (abs2.sum(1) * tau).cpu().numpy()
                peak_step = abs2.max(1).values.cpu().numpy()
                self.energy[:, n + 1] = energy_step
                self.peak_power[:, n + 1] = peak_step

                # ---------- сохранить шаг? -------------------------------
                is_save_step = ((n + 1) % save_every == 0) or (n == self.com.N - 1)
                if is_save_step:
                    save_idx += 1
                    if save_idx < self.stored_steps_count:
                        self.numerical_solution[save_idx] = (
                            psi_gpu.detach().cpu().numpy()
                        )

                # ---------- обновить график ------------------------------
                if draw_modulus and ((n + 1) % draw_interval == 0):
                    psi_cpu_for_plot = psi_gpu.detach().cpu().numpy()
                    update_modulus_plot(
                        fig, ax, line, psi_cpu_for_plot, self.z, n
                    )

        # ───── финализация графика ───────────────────────────────────────
        if draw_modulus:
            finalize_plot()

    def _rhs_u(self, u, v, dz):
        # du/dt = v
        return v

    def _rhs_v(self, u, v, dz):
        # ∂u/∂z: 2-й порядок внутри и на краях
        Cn, Nz = u.shape
        du_dz = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dz)
        if Nz >= 3:
            du_dz[:, 0] = (-3.0 * u[:, 0] + 4.0 * u[:, 1] - u[:, 2]) / (2.0 * dz)
            du_dz[:, -1] = (3.0 * u[:, -1] - 4.0 * u[:, -2] + u[:, -3]) / (2.0 * dz)
        elif Nz == 2:
            du_dz[:, 0] = (u[:, 1] - u[:, 0]) / dz
            du_dz[:, -1] = (u[:, -1] - u[:, 0]) / dz
        else:  # Nz == 1
            du_dz[:, 0] = 0.0

        # i * C * u
        coupling = self.linear_coeffs_array @ u  # (eq_size, Nz)

        # Энергия по z (трапеция) -> форма (eq_size, 1), без keepdims
        abs2 = np.abs(u) ** 2
        if Nz == 1:
            Ez = dz * abs2  # (eq_size, 1)
        elif Nz == 2:
            Ez = dz * 0.5 * (abs2[:, :1] + abs2[:, 1:])  # (eq_size, 1)
        else:
            edge = 0.5 * (abs2[:, :1] + abs2[:, -1:])
            mid = np.sum(abs2[:, 1:-1], axis=1, keepdims=True)
            Ez = dz * (edge + mid)  # (eq_size, 1)

        # g(t) = g0 / (1 + E/E_sat) по каждому каналу
        g = self.eq.g_0[:, None] / (1.0 + Ez / self.eq.E_sat[:, None])  # (eq_size, 1)

        term = (1j * du_dz
                + 1j * self.eq.beta1[:, None] * v
                + self.eq.gamma[:, None] * abs2 * u
                - 1j * 0.5 * g * u
                + 1j * 0.5 * self.eq.alpha[:, None] * u
                + 1j * coupling)

        return (2.0 / self.eq.beta2[:, None]) * term

    if _TORCH_AVAILABLE:
        def _rhs_u_torch(self, u, v, dz_t):
            # du/dt = v
            return v

        def _rhs_v_torch(self, u: torch.Tensor, v: torch.Tensor, dz_t: torch.Tensor):
            Cn, Nz = u.shape

            # ∂u/∂z: 2-й порядок внутри и на краях
            du_dz = (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) / (2.0 * dz_t)
            if Nz >= 3:
                du_dz[:, 0] = (-3.0 * u[:, 0] + 4.0 * u[:, 1] - u[:, 2]) / (2.0 * dz_t)
                du_dz[:, -1] = (3.0 * u[:, -1] - 4.0 * u[:, -2] + u[:, -3]) / (2.0 * dz_t)
            elif Nz == 2:
                du_dz[:, 0] = (u[:, 1] - u[:, 0]) / dz_t
                du_dz[:, -1] = (u[:, -1] - u[:, 0]) / dz_t
            else:
                du_dz[:, 0] = 0.0

            # i * C * u
            C_t = torch.as_tensor(self.linear_coeffs_array, dtype=self.ctype, device=self.device)
            coupling = torch.matmul(C_t, u)

            # Энергия по z (трапеция) -> (eq_size, 1)
            abs2 = torch.abs(u) ** 2
            if Nz == 1:
                Ez = dz_t * abs2
            elif Nz == 2:
                Ez = dz_t * 0.5 * (abs2[:, :1] + abs2[:, 1:])
            else:
                edge = 0.5 * (abs2[:, :1] + abs2[:, -1:])
                mid = abs2[:, 1:-1].sum(dim=1, keepdim=True)
                Ez = dz_t * (edge + mid)

            g = self.g0_t[:, None] / (1.0 + Ez / self.E_sat_t[:, None])

            term = (1j * du_dz
                    + 1j * self.eq.beta1_t[:, None] * v
                    + self.eq.gamma_t[:, None] * abs2 * u
                    - 1j * 0.5 * g * u
                    + 1j * 0.5 * self.eq.alpha_t[:, None] * u
                    + 1j * coupling)

            return (2.0 / self.eq.beta2_t[:, None]) * term

    def run_numerical_simulation_time(self,
                                      draw_modulus: bool = False,
                                      draw_interval: int = 10,
                                      save_gif: bool = False,
                                      yscale: str = 'linear'):
        """
        Временная эволюция (t-шаги) с поддержкой как NumPy, так и PyTorch.
        Один внешний цикл идёт по времени, внутри — RK-4 шаг.
        """

        C, Nz     = self.eq.size, self.z.size
        dt, dz    = self.com.tau, self.com.h

        # ---------- подготовка PyTorch констант --------------------------
        if self.use_torch:
            dtype, ctype = self.dtype, self.ctype
            device       = self.device

            beta1_t = torch.as_tensor(self.eq.beta1, dtype=dtype, device=device)
            beta2_t = torch.as_tensor(self.eq.beta2, dtype=dtype, device=device)
            gamma_t = torch.as_tensor(self.eq.gamma, dtype=dtype, device=device)
            alpha_t = torch.as_tensor(self.eq.alpha, dtype=dtype, device=device)

            # кэшируем в self, чтобы RHS мог пользоваться
            self.eq.beta1_t = beta1_t
            self.eq.beta2_t = beta2_t
            self.eq.gamma_t = gamma_t
            self.eq.alpha_t = alpha_t

            dz_t = torch.tensor(dz, dtype=dtype, device=device)

        # ---------- поля u, v = du/dt ------------------------------------
        if self.use_torch:
            u = torch.zeros((C, Nz), dtype=ctype, device=self.device)
            v = torch.zeros_like(u)
        else:
            u = np.zeros((C, Nz), dtype=self.ctype)
            v = np.zeros_like(u)

        # ---------- pre-compute d(bc)/dt ---------------------------------
        bc        = self.boundary_condition
        bc_prime  = fft_derivative(self.boundary_condition, dt, axis=1) # np.gradient(bc, self.com.tau, axis=1)

        if self.use_torch:
            bc_t       = torch.as_tensor(bc,       dtype=ctype, device=self.device)
            bc_prime_t = torch.as_tensor(bc_prime, dtype=ctype, device=self.device)
            fb_coef_t  = torch.as_tensor(self.feedback_coefficient,
                                         dtype=ctype, device=self.device)

        # ------------------------------------------------------------------

        # буферы истории на правом краю (берём 1D срез по z=-1)
        hist_u_L = []
        hist_v_L = []
        # буферы входа
        hist_bc = [bc[:, k].copy() for k in range(self.com.M)]
        hist_bc_prime = [bc_prime[:, k].copy() for k in range(self.com.M)]

        # параметры задержки
        tau_fb_ps = self.feedback_delay_ps  # задай в вызывающем коде
        delay_steps_float = tau_fb_ps / dt
        kappa_phase = self.feedback_coefficient  # комплексный, с фазой воздуха

        # ---------- график ------------------------------------------------
        if draw_modulus:
            fig, ax, line = init_modulus_plot(yscale=yscale, t=self.z,
                                              scaling_mode='history',
                                              save_gif=save_gif)

        # ---------- буферы метрик (CPU — мало, GPU — много) ---------------
        if self.use_torch:
            buf_gpu: list[torch.Tensor] = []
            pend   : list[torch.cuda.Event] = []
            first_unsaved = 0

            energy_gpu = torch.zeros((C, self.com.M + 1),
                                     dtype=self.dtype, device=self.device)
            peak_gpu   = torch.zeros_like(energy_gpu)

            energy_gpu[:, 0] = 0.0
            peak_gpu[:, 0]   = 0.0
        else:
            self.energy[:, 0] = 0.0
            self.peak_power[:, 0] = 0.0

        # -----------------------------------------------------------------

        # установить начальный край z=0
        u[:, 0] = hist_bc[0]
        v[:, 0] = hist_bc_prime[0]
        # начальная история правого края — нули
        hist_u_L.append(np.zeros(u.shape[0], dtype=u.dtype))
        hist_v_L.append(np.zeros(v.shape[0], dtype=v.dtype))

        # ---------- главный цикл по времени ------------------------------
        for m in trange(self.com.M, disable=not draw_modulus and not self.display_debug_info):

            if self.use_torch:
                u, v = rk4_step_torch(
                    u, v, dt, dz_t,
                    self._rhs_u_torch, self._rhs_v_torch,
                    fb_coef_t,
                    bc_t[:, m], bc_prime_t[:, m]
                )
                # --- метрики на GPU
                abs2 = u.abs() ** 2
                energy_gpu[:, m + 1] = abs2.sum(1) * dz
                peak_gpu[:, m + 1]   = abs2.max(1).values

                # буфер кадров
                buf_gpu.append(u.detach().clone())

                if draw_modulus and ((m + 1) % draw_interval == 0):
                    batch = torch.stack(buf_gpu, dim=0)  # (k, C, Nz)
                    k     = batch.shape[0]
                    batch_cpu  = batch.to("cpu", non_blocking=True)
                    e_cpu      = energy_gpu[:, first_unsaved + 1:first_unsaved + 1 + k].to("cpu", non_blocking=True)
                    p_cpu      = peak_gpu[:,   first_unsaved + 1:first_unsaved + 1 + k].to("cpu", non_blocking=True)

                    evt = torch.cuda.Event()
                    torch.cuda.current_stream().record_event(evt)
                    pend.append((evt, batch_cpu, e_cpu, p_cpu, first_unsaved, k))

                    buf_gpu.clear()
                    first_unsaved += k

                # --- обработка завершённых копий --------------------------
                while pend and pend[0][0].query():
                    _, b_cpu, e_cpu, p_cpu, st, k = pend.pop(0)
                    self.numerical_solution_time = b_cpu[-1].numpy()  # последняя
                    # self.energy[:, st + 1:st + 1 + k]      = e_cpu.numpy()
                    # self.peak_power[:, st + 1:st + 1 + k]  = p_cpu.numpy()

                    if draw_modulus:
                        update_modulus_plot(fig, ax, line,
                                            b_cpu[-1].numpy(),
                                            self.t, m)

            else:
                # ----------- CPU-ветка ------------------------------------
                u, v = rk4_step(
                    u, v, dt, dz,
                    self._rhs_u, self._rhs_v,
                    self.feedback_coefficient,
                    bc[:, m], bc_prime[:, m]
                )

                # u, v = rk4_step_delayed_bc(
                #     u, v, dt, dz,
                #     self._rhs_u, self._rhs_v,  # :contentReference[oaicite:5]{index=5}
                #     hist_u_L, hist_v_L,
                #     hist_bc, hist_bc_prime,
                #     m, delay_steps_float, kappa_phase
                # )

                # после получения (u,v) на t_{m+1}: обновим историю правого края
                hist_u_L.append(u[:, -1].copy())
                hist_v_L.append(v[:, -1].copy())

                self.numerical_solution_time = u
                # self.energy[:, m + 1]      = np.trapz(np.abs(u) ** 2, dx=dz, axis=1)
                # self.peak_power[:, m + 1]  = np.max(np.abs(u) ** 2, axis=1)

                if draw_modulus and (m + 1) % draw_interval == 0:
                    update_modulus_plot(fig, ax, line, u, self.t, m, xlabel='z [m]')

        # ---------- «хвост» буфера GPU ------------------------------------
        if self.use_torch:
            if buf_gpu:
                batch = torch.stack(buf_gpu, dim=0)
                k     = batch.shape[0]
                batch_cpu = batch.to("cpu")
                self.numerical_solution_time = batch_cpu[-1].numpy()

                e_cpu = energy_gpu[:, first_unsaved + 1:first_unsaved + 1 + k].cpu()
                p_cpu = peak_gpu[:,   first_unsaved + 1:first_unsaved + 1 + k].cpu()
                # self.energy[:, first_unsaved + 1:first_unsaved + 1 + k]     = e_cpu.numpy()
                # self.peak_power[:, first_unsaved + 1:first_unsaved + 1 + k] = p_cpu.numpy()

        if draw_modulus:
            finalize_plot()


    def run_resonator_simulation_nocos(self, backward_energy):
        """
        Без учёта взаимодействия частот прямой и обратной волн.
        в перспективе для более высокого порядка можно добавить флаг
        """
        if self.D is None:
            self.calculate_D_matrix(self.com.h)

        fast_nocos_resonator_run(self.com.N, self.eq.size, self.numerical_solution, self.energy, backward_energy,
                                 self.D, self.eq.gamma, self.eq.E_sat, self.eq.g_0, self.com.h, self.com.tau,
                                 self.eq.noise_amplitude)

    def run_resonator_simulation_fullcos(self, backward_solution, draw_modulus=False, draw_interval=10):
        """
        С учётом взаимодействия частот прямой и обратной волн.
        в перспективе для более высокого порядка можно добавить флаг
        """
        if self.D is None:
            self.calculate_D_matrix(self.com.h)

        # Инициализация графика, если нужно
        if draw_modulus:
            fig, ax, line = init_modulus_plot(t=self.t)

        for n in trange(self.com.N, disable=not self.display_debug_info):
            # Выполнение на NumPy
            self.numerical_solution[n + 1] = ssfm_order1_resonator_fullcos(self.numerical_solution[n], backward_solution[:, n],
                                                                           self.D, self.eq.gamma, self.eq.E_sat, self.eq.g_0,
                                                                           self.com.h, self.com.tau, self.eq.noise_amplitude)

            for k in range(self.eq.size):
                self.energy[k][n + 1] = get_energy_rectangles(self.numerical_solution[n + 1][k], self.com.tau)

            for k in range(self.eq.size):
                self.peak_power[k][n + 1] = np.max(np.abs(self.numerical_solution[n + 1][k]) ** 2)

            # Обновление графика через каждые `draw_interval` шагов, если включен флаг `draw_modulus`
            if draw_modulus and (n + 1) % draw_interval == 0:
                update_modulus_plot(fig, ax, line, self.numerical_solution[n + 1], self.z, n)

        # Закрытие интерактивного режима после завершения симуляции
        if draw_modulus:
            finalize_plot()

    def get_analytical_solution(self):
        # TODO: Надо бы как-то корректно обработать случаи, когда есть аналитическое решение, а когда нет
        #  (для разных импульсов в зависимости от параметров волокна)
        if ((any([pulse != zero_pulse for pulse in self.pulses]) or
             self.eq.core_configuration is not CoreConfig.empty_ring) and
                self.eq.coupling_coefficient != 0.0):
            raise RuntimeError("Does not exist correctly analytical solution for this case yet")
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
        T, Z = np.meshgrid(self.t, self.z)
        plot3D(self.z, self.t, self.absolute_error, 'абсолютная_ошибка-case1')
        plot2D(self.z, self.absolute_error[:, self.com.M // 2] / abs(
            self.analytical_solution[:, self.eq.size // 2, self.com.M // 2]),
               'относительная_ошибка_в_пике-case1')

    def run_test(self):
        self.run_numerical_simulation()
        self.get_analytical_solution()
        self.calculate_error()
        self.plot_error()

    def set_reflective_index_perturbations(self, perturbation_arr):
        """
        Функция изменяет матрицу связей на диагонали
        """
        if not self.use_dimensional:
            raise RuntimeError("You can set perturbations of the reflective indexes only in dimensional case")
        self.linear_coeffs_array += np.diag(perturbation_arr)

    def convert_to_dimensionless(self, coupling_coefficient, gamma, beta2,
                                 reserve_power_scale=1, reserve_time_scale=1, reserve_length_scale=1,
                                 print_linear_coeffs_array=True):
        """
        Функция приводит размерное уравнение

         i dU/dz + i beta1 dU/dt - beta2/2 d^2U/dt^2 + gamma |U|^2 U - i g0 U / (2 (1 + E / Esat)) + i alpha U / 2 + Summ C_{ij} U_j= 0

         к безразмерному виду

        i dU/dz + i beta1/T/C dU/dt + d^2U/dt^2 + |U|^2 U - i g0 U / (2 C (1 + E / Esat)) + i alpha U / 2 / C + Summ c_{ij} U_j = 0.

        Примечание:
        ::
            reserve_time_scale -- масштаб по времени, если дисперсия равна нулю;
            reserve_length_scale -- масштаб по длине, если коэффициент связи равен нулю;
            reserve_power_scale -- масштаб мощности на периферии, если нелинейность равна нулю.
        Параметры:
        ::
            coupling_coefficient [1/m]
            gamma [1/(W*m)]
            beta2 [ps^2/m]
        """

        if self.eq.core_configuration is CoreConfig.empty_ring:
            self_coefficient = 2
        elif self.eq.core_configuration is CoreConfig.hexagonal:
            self_coefficient = 6
        else:
            raise RuntimeError('Unsupportable MCF configuration')

        time_scale = sqrt(0.5*abs(beta2) / coupling_coefficient) if beta2 != 0.0 else reserve_time_scale  # [ps]
        power_scale = (coupling_coefficient / gamma) if gamma != 0.0 else reserve_power_scale  # [W]
        length_scale = (1 / coupling_coefficient) if coupling_coefficient != 0.0 else reserve_length_scale  # [m]
        energy_scale = power_scale * time_scale

        self.com.T1 /= time_scale  # [1]
        self.com.T2 /= time_scale  # [1]
        self.com.tau /= time_scale  # [1]
        if self.t is not None: self.t /= time_scale  # [1]
        if self.omega is not None: self.omega *= time_scale  # [1]
        if self.omega2 is not None: self.omega2 *= time_scale ** 2  # [1]

        self.com.L1 /= length_scale  # [1]
        self.com.L2 /= length_scale  # [1]
        self.com.h /= length_scale  # [1]
        if self.z is not None: self.z /= length_scale  # [1]

        self.eq.beta1 = self.eq.beta1 / (time_scale * coupling_coefficient) if self.eq.beta1 != 0.0 else 0.0  # [1]
        self.eq.beta2 = 2 * np.sign(beta2) if beta2 != 0.0 else 0.0  # [1]
        self.eq.gamma = 1.0 if gamma != 0.0 else 0.0  # [1]
        self.eq.E_sat /= energy_scale  # [1]
        self.eq.alpha /= coupling_coefficient  # [1]
        self.eq.g_0 /= coupling_coefficient  # [1]
        dimensional_coupling_coefficient = self.eq.coupling_coefficient
        self.eq.coupling_coefficient = 1.0  # [1]
        self.use_dimensional = False
        self.eq.__post_init__()
        self.calculate_D_matrix(self.com.h)

        self.linear_coeffs_array /= dimensional_coupling_coefficient
        self.linear_coeffs_array += np.diag(np.full(self.eq.size, -self_coefficient))
        if print_linear_coeffs_array:
            print_matrix(self.linear_coeffs_array, "linear_coeffs_array")

        if gamma != 0.0:  # Пока нет реализации для уравнений Манакова
            self.nonlinear_cubic_coeffs_array = np.where(self.nonlinear_cubic_coeffs_array != 0, 1.0, self.nonlinear_cubic_coeffs_array)
        else:
            self.nonlinear_cubic_coeffs_array = np.zeros_like(self.nonlinear_cubic_coeffs_array)

        cores = np.arange(self.eq.size, dtype=float)
        _, Zn, Tn = np.meshgrid(cores, self.z, self.t)  # [1], нормированные расчётные сетки по t по z
        if self.numerical_solution is not None: self.numerical_solution /= sqrt(power_scale) * np.exp(1j*Zn*self_coefficient)  # [1]
        if self.analytical_solution is not None: self.analytical_solution /= sqrt(power_scale) * np.exp(1j*Zn*self_coefficient)  # [1]
        if self.absolute_error is not None: self.absolute_error /= sqrt(power_scale)  # [1]
        if self.C_norm is not None: self.C_norm /= sqrt(power_scale)  # [1]
        if self.peak_power is not None: self.peak_power /= power_scale  # [1]
        if self.energy is not None: self.energy /= energy_scale  # [1]
        if self.L2_norm is not None: self.L2_norm /= energy_scale  # [1]


    def convert_to_dimensional(self, coupling_coefficient, gamma, beta2,
                               reserve_power_scale=1, reserve_time_scale=1, reserve_length_scale=1,
                               print_linear_coeffs_array=True):
        """ Функция приводит безразмерное уравнение

         i dU/dz + i beta1 dU/dt + d^2U/dt^2 + |U|^2 U - i g0 U / (2 (1 + E / Esat)) + i alpha U / 2 + Summ c_{ij} U_j = 0

         к размерному виду

         i dU/dz + i beta1 T C dU/dt - beta2/2 d^2U/dt^2 + gamma |U|^2 U - i g0 C U / (2 (1 + E / Esat)) + i alpha C U / 2 + Summ C_{ij} U_j= 0.

        Примечание:
        ::
            reserve_time_scale -- масштаб по времени, если дисперсия равна нулю;
            reserve_length_scale -- масштаб по длине, если коэффициент связи равен нулю;
            reserve_power_scale -- масштаб мощности на периферии, если нелинейность равна нулю.
        Параметры:
        ::
            coupling_coefficient [1/m]
            gamma [1/(W*m)]
            beta2 [ps^2/m]
         """

        if self.eq.core_configuration is CoreConfig.empty_ring:
            self_coefficient = 2
        elif self.eq.core_configuration is CoreConfig.hexagonal:
            self_coefficient = 6
        else:
            raise RuntimeError('Unsupportable MCF configuration')

        time_scale = sqrt(0.5*abs(beta2) / coupling_coefficient) if beta2 != 0.0 else reserve_time_scale  # [ps]
        power_scale = (coupling_coefficient / gamma) if gamma != 0.0 else reserve_power_scale  # [W]
        length_scale = (1 / coupling_coefficient) if coupling_coefficient != 0.0 else reserve_length_scale  # [m]
        energy_scale = power_scale * time_scale

        self.com.T1 *= time_scale  # [ps]
        self.com.T2 *= time_scale  # [ps]
        self.com.tau *= time_scale  # [ps]
        if self.t is not None: self.t *= time_scale  # [ps]
        if self.omega is not None: self.omega /= time_scale  # [THz]
        if self.omega2 is not None: self.omega2 /= time_scale**2  # [THz^2]

        self.com.L1 *= length_scale  # [m]
        self.com.L2 *= length_scale  # [m]
        self.com.h *= length_scale  # [m]
        if self.z is not None: self.z *= length_scale  # [m]

        self.eq.beta1 *= time_scale * coupling_coefficient  # [ps/m]
        self.eq.beta2 = beta2  # [ps^2/m]
        self.eq.gamma = gamma  # [1/(W*m)]
        self.eq.E_sat *= energy_scale  # [pJ]
        self.eq.alpha *= coupling_coefficient  # [1/m]
        self.eq.g_0 *= coupling_coefficient  # [1/m]
        self.eq.coupling_coefficient = coupling_coefficient  # [1/m]
        self.use_dimensional = True
        self.eq.__post_init__()
        self.calculate_D_matrix(self.com.h)

        self.linear_coeffs_array -= np.diag(np.full(self.eq.size, -self_coefficient))
        self.linear_coeffs_array *= coupling_coefficient  # Пока нет реализации для уравнений Манакова
        if print_linear_coeffs_array:
            print_matrix(self.linear_coeffs_array, "linear_coeffs_array")

        self.nonlinear_cubic_coeffs_array *= gamma

        cores = np.arange(self.eq.size, dtype=float)
        _, Zn, Tn = np.meshgrid(cores, self.z/length_scale, self.t/time_scale)  # [1], нормированные расчётные сетки по t по z
        if self.numerical_solution is not None: self.numerical_solution *= sqrt(power_scale) * np.exp(1j*Zn*self_coefficient)  # [sqrt(W)]
        if self.analytical_solution is not None: self.analytical_solution *= sqrt(power_scale) * np.exp(1j*Zn*self_coefficient)  # [sqrt(W)]
        if self.absolute_error is not None: self.absolute_error *= sqrt(power_scale)  # [sqrt(W)]
        if self.C_norm is not None: self.C_norm *= sqrt(power_scale)  # [sqrt(W)]
        if self.peak_power is not None: self.peak_power *= power_scale  # [W]
        if self.energy is not None: self.energy *= energy_scale  # [pJ]
        if self.L2_norm is not None: self.L2_norm *= energy_scale  # [pJ]


    def find_stationary_solution(self, lambda_val, max_iter=200, tol=1e-11, plot_graphs=False, update_interval=0.01, yscale='linear'):
        self.numerical_solution[0] = find_stationary_solution(self.numerical_solution[0],
                                                              self.com.M, self.com.tau,
                                                              self.linear_coeffs_array,
                                                              self.nonlinear_cubic_coeffs_array,
                                                              -self.eq.beta2 * 0.5,
                                                              self.omega2,
                                                              self.eq.mask_array,
                                                              self.eq.E_sat, self.eq.alpha, self.eq.g_0,
                                                              lambda_val,
                                                              max_iter=max_iter, tol=tol,
                                                              plot_graphs=plot_graphs, update_interval=update_interval, yscale=yscale)


# Функции для графиков
def init_modulus_plot(yscale: str = 'linear', *,
                      scaling_mode: str = 'history',      # 'step' | 'history'
                      margin: float = 1.1,
                      t: np.ndarray | None = None,     # массив времени (M,)  – для подписи
                      nticks: int = 5,                 # сколько t-меток в сегменте (последняя убирается)
                      save_gif: bool = False,
                      gif_path: str = 'evolution.gif',
                      fps: int = 10):
    """
    Возвращает fig, ax, line.
    scaling_mode:
        'step'    – Y-лимит по максимуму ТЕКУЩЕГО кадра
        'history' – Y-лимит по глобальному максимуму всех кадров
    """

    matplotlib.use('qt5agg', force=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    line, = ax.plot([], [], lw=1.3)

    ax.set_yscale(yscale)
    ax.set_ylabel(r'|u|')
    ax.set_ylim(1e-5, 1)
    ax.grid(True, which='both', axis='y')

    # служебные поля
    ax._mcf_scaling_mode = scaling_mode
    ax._mcf_margin = margin
    ax._mcf_global_max = 1e-5
    ax._mcf_M = None
    ax._mcf_vlines = []
    ax._mcf_coretexts = []
    ax._mcf_t = t
    ax._mcf_nticks = nticks
    ax._mcf_tick_cache = None

    # ленивый GIF-writer
    ax._mcf_save_gif = False
    ax._mcf_gif_writer = None
    if save_gif:
        try:
            import imageio
            ax._mcf_gif_writer = imageio.get_writer(gif_path, mode='I', fps=fps)
            ax._mcf_save_gif = True
        except ImportError:
            import warnings
            warnings.warn(
                "⚠ Модуль imageio не найден — GIF-запись отключена. "
                "Установите `pip install imageio`.", RuntimeWarning)

    plt.ion()
    plt.show()
    return fig, ax, line


def update_modulus_plot(fig, ax, line, data_2d: np.ndarray, z: np.ndarray, step: int, xlabel='t [ps]'):
    """Обновляет картинку и (опционально) дописывает кадр в GIF."""
    C, M = data_2d.shape
    if ax._mcf_M is None:
        ax._mcf_M = M

    # линия |u|
    y = np.abs(data_2d).ravel()
    x = np.add.outer(np.arange(C) * M, np.arange(M)).ravel()
    line.set_data(x, y)
    max_mod = y.max()

    # масштаб Y
    if ax._mcf_scaling_mode == 'step':
        ylim_top = max_mod * ax._mcf_margin
    else:
        ax._mcf_global_max = max(ax._mcf_global_max, max_mod)
        ylim_top = ax._mcf_global_max * ax._mcf_margin
    ax.set_ylim(1e-5, ylim_top)

    # пределы X без пустых полей
    ax.set_xlim(0, C * M - 1)

    # пунктиры-разделители
    for ln in ax._mcf_vlines: ln.remove()
    ax._mcf_vlines = [ax.axvline(k * M, color='gray', ls='--', lw=.6, alpha=.35)
                      for k in range(1, C)]

    # minor-ticks времени t (последний тик не выводим)
    t = ax._mcf_t
    if t is not None:
        key = (C, M, ax._mcf_nticks)
        if ax._mcf_tick_cache != key:
            ax._mcf_tick_cache = key
            idx = np.linspace(0, M - 1, ax._mcf_nticks, dtype=int)[:-1]
            pos, lbl = [], []
            for c in range(C):
                offs = c * M
                pos.extend(offs + idx)
                lbl.extend([f'{t[i]:.1f}' for i in idx])
            ax.set_xticks(pos, minor=True)
            ax.set_xticklabels(lbl, minor=True, rotation=90, fontsize=7)
    ax.set_xticks([])  # major-ticks прячем
    ax.set_xlabel(xlabel)

    # номера ядер сверху
    for txt in ax._mcf_coretexts: txt.remove()
    ax._mcf_coretexts.clear()
    y_text = ylim_top * .92
    for c in range(C):
        xpos = (c + .5) * M
        ax._mcf_coretexts.append(
            ax.text(xpos, y_text, str(c),
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold',
                    color='navy', alpha=.9))

    # рендер на экран
    z_step_index = step + 1
    evolutionary_variable_name = "z"

    if xlabel.find("z") > -1:
        z_step_index = step
        evolutionary_variable_name = "t"

    ax.set_title(f'step = {step + 1} of {z.size - 1}, {evolutionary_variable_name} = {z[z_step_index]:.3g},    max |u| = {max_mod:.3g}')
    fig.canvas.draw_idle()
    plt.pause(0.001)

    # запись в GIF
    if ax._mcf_save_gif:
        fig.canvas.draw()  # ensure rendered
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

        ren = fig.canvas.get_renderer()
        h, w = int(ren.height), int(ren.width)  # ← cast к int

        frame = buf.reshape(h, w, 3)
        ax._mcf_gif_writer.append_data(frame)


# ───────────────────────────────────────────────────────────────────────────
def finalize_plot():
    ax = plt.gca()
    if getattr(ax, '_mcf_save_gif', False) and ax._mcf_gif_writer:
        ax._mcf_gif_writer.close()
    plt.ioff(); plt.show()


@njit(inline='always')
def fast_nocos_resonator_run(N, eq_size, numsol_array, self_energy, backward_energy,
                             D_mat, gamma, E_sat, g_0, h, tau, noise_amplitude):
    for k in range(eq_size):
        self_energy[k][0] = get_energy_rectangles(numsol_array[0][k], tau)
    for n in range(N):
        numsol_array[n + 1] = ssfm_order1_resonator_nocos(numsol_array[n], self_energy[:, n], backward_energy[:, n],
                                                          D_mat, gamma, E_sat, g_0,
                                                          h, tau, noise_amplitude)
        for j in range(eq_size):
            self_energy[j][n + 1] = get_energy_rectangles(numsol_array[n + 1][j], tau)
