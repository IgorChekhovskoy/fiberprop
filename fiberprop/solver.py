from tqdm import trange
from scipy.fft import fftfreq
from dataclasses import dataclass, field
from typing import Union
from math import sqrt, pi
from enum import Enum
from numba import njit

from .matrices import create_freq_matrix, get_pade_exponential2, create_simple_dispersion_free_matrix
from .pulses import zero_pulse
from .drawing import *
from .ssfm_mcf import ssfm_order2, get_energy_rectangles, ssfm_order1_resonator_nocos, ssfm_order1_resonator_fullcos
from .stationary_solution_solver import find_stationary_solution

try:
    import torch
    is_torch_available = True
except ImportError:
    is_torch_available = False

from .ssfm_mcf_pytorch import get_energy_rectangles_pytorch, ssfm_order2_pytorch

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

        Вычисляемые атрибуты:
        ::
            h (float): Шаг по эволюционной переменной [m], вычисляется как (L2-L1)/N
            tau (float): Шаг по времени [ps], вычисляется как (T2-T1)/M

        Пример использования:
            >>> params = ComputationalParameters(N=1000, M=512, L1=0.0, L2=1.0, T1=-10.0, T2=10.0)
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

    def __post_init__(self):
        if self.N > 0:
            self.h = (self.L2 - self.L1) / self.N
        else:
            self.h = 0.0

        if self.M > 0:
            self.tau = (self.T2 - self.T1) / self.M
        else:
            self.tau = 0.0

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


@dataclass
class CoreConfig(Enum):
    """
    Перечисление возможных конфигураций сердцевин.

    Варианты:
    ::
        not_set (0): Конфигурация не определена (значение по умолчанию)
        empty_ring (1): 1D круговая конфигурация без центральной сердцевины
        square (2): 2D квадратная решётка сердцевин
        hexagonal (3): 2D гексагональная (шестиугольная) решётка сердцевин
        manakov_eq (4): Модель с уравнениями Манакова (учёт поляризационных эффектов)
        dual_core (5): Двухсердцевинная конфигурация
        ring_with_center (6): 1D комбинированная конфигурация (кольцо + центральная сердцевина)
    """
    not_set: int = 0
    empty_ring: int = 1
    square: int = 2
    hexagonal: int = 3
    manakov_eq: int = 4
    dual_core: int = 5
    ring_with_center: int = 6

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

        beta2 (float | np.ndarray | list): Коэффициент дисперсии групповых скоростей [ps²/m]
        gamma (float | np.ndarray | list): Коэффициент нелинейности Керра [1/(W·m)]
        E_sat (float | np.ndarray | list): Энергия насыщения [pJ]
        alpha (float | np.ndarray | list): Коэффициент потерь [1/m]
        g_0 (float | np.ndarray | list): Ненасыщенное усиление [1/m]
        coupling_coefficient (float | np.ndarray | list): Коэффициент связи между сердцевинами [1/m]
        noise_amplitude (float): Амплитуда аддитивного белого шума  [sqrt(W/2)]

    Особенности:
    ::
        - Параметры beta2, gamma, E_sat, alpha, g_0 и coupling_coefficient могут быть заданы как:
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
    display_debug_info: bool = True

    mask_array = None

    beta2: Union[float, np.ndarray, list] = -1.0
    gamma: Union[float, np.ndarray, list] = 1.0
    E_sat: Union[float, np.ndarray, list] = 1.0
    alpha: Union[float, np.ndarray, list] = 0.0
    g_0: Union[float, np.ndarray, list] = 0.0
    coupling_coefficient: Union[float, np.ndarray, list] = 1.0
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

        self.make_eq_mask()

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

    def make_eq_mask(self):
        temp_array_size = int((1.0 + self.size * (self.size + 1.0)))
        temp_array = np.zeros((temp_array_size, temp_array_size), dtype=bool)
        center = temp_array_size // 2

        if self.core_configuration is CoreConfig.ring_with_center:
            for i in range(self.size + 1):
                temp_array[0][i] = True

        elif ((self.core_configuration is CoreConfig.empty_ring) or
              (self.core_configuration is CoreConfig.manakov_eq)):
            for i in range(self.size):
                temp_array[0][i] = True

        elif self.core_configuration is CoreConfig.square:
            for i in range(temp_array_size):
                for j in range(temp_array_size):
                    if (i - center) ** 2 + (j - center) ** 2 <= self.ring_count ** 2 + 1e-13:
                        temp_array[i][j] = True

        elif self.core_configuration is CoreConfig.hexagonal:
            h_i = 1.0
            h_j = 1.0 / sqrt(3.0)
            for i in range(temp_array_size):
                for j in range(temp_array_size):
                    if (h_i * (i - center)) ** 2 + (
                            h_j * (j - center)) ** 2 <= self.ring_count ** 2 * 4.0 / 3.0 + 1e-10 and \
                            (i + j - 2 * center) % 2 == 0:
                        temp_array[i][j] = True

        if self.display_debug_info:
            print_temp_array(temp_array_size, temp_array)

        self.mask_array = []
        index_1d = 0
        for i in range(temp_array_size):
            for j in range(temp_array_size):
                if temp_array[i][j]:
                    self.mask_array.append(Mask(index_1d, i - temp_array_size // 2, j - temp_array_size // 2, []))
                    index_1d += 1
        # print("core count = ", index_1d)

    @staticmethod
    def get_info():
        """
        Функция выводит информацию о параметрах класса и их размерностях
        """
        print('\n\nEquationParameters:')
        print('\"core_configuration\" -- конфигурация MCF, объект класса \"CoreConfig\";')
        print('\"size\" -- количество сердцевин в MCF, целое число;')
        print('\"ring_count\" -- количество коаксиальных колец в MCF ?, вещественное число.\n')

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


def get_core_count(core_configuration, ring_count: float) -> int:
    """
    Определяет количество сердцевин (core_count) в зависимости от номера кольца (ring_count).

    Параметры:
        core_configuration (CoreConfig): конфигурация расположения сердцевин
        ring_count (float): Номер кольца (должен быть >= 0)

    Возвращает:
        int: Количество сердцевин

    Исключения:
        ValueError: Если ring_count отрицательный или слишком большой
    """
    if ring_count < 0:
        raise ValueError("Номер кольца не может быть отрицательным")

    if core_configuration is CoreConfig.square:
        return 0 # TODO
    elif core_configuration is CoreConfig.hexagonal:
        sqrt3 = np.sqrt(3)  # ≈ 1.732
        sqrt7 = np.sqrt(7)  # ≈ 2.645
        sqrt12 = np.sqrt(12)  # ≈ 3.464
        sqrt13 = np.sqrt(13)

        if 0 <= ring_count < 1:
            return 1
        elif 1 <= ring_count < sqrt3:
            return 7
        elif sqrt3 <= ring_count < 2:
            return 13
        elif 2 <= ring_count < sqrt7:
            return 19
        elif sqrt7 <= ring_count < 3:
            return 31
        elif 3 <= ring_count < sqrt12:
            return 37
        elif sqrt12 <= ring_count < sqrt13:
            return 43
        else:
            raise ValueError("Тебе куда столько ядер? Солить будешь?")
    return 0


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
            solver.run_numerical_simulation(print_modulus=True)
            solver.plot_error()
        """

    def __init__(
            self,
            com: ComputationalParameters,
            eq: EquationParameters,
            use_dimensional=False,
            pulses=zero_pulse,
            pulse_params_list=None,
            initial_condition: np.ndarray = None,
            use_gpu=False,
            use_torch=False,
            precision='float64',
            display_debug_info=True
    ):
        self.com = com
        self.eq = eq
        self.use_dimensional = use_dimensional  # безразмерная или размерная задача
        self.use_gpu = use_gpu and is_torch_available  # Устанавливаем режим GPU только если PyTorch доступен
        self.use_torch = (use_torch and is_torch_available) or (self.use_gpu and is_torch_available)
        self.device = None
        if self.use_torch:
            self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.precision = precision
        self.dtype = torch.float32 if self.precision == 'float32' else torch.float64
        self.ctype = torch.complex64 if self.precision == 'float32' else torch.complex128

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

        # Инициализация массивов
        self.initialize_arrays()

        # Обработка начальных условий
        if initial_condition is not None:
            self.validate_initial_condition(initial_condition)
            self.apply_initial_condition(initial_condition)
        else:
            self.initialize_with_pulses(pulses, pulse_params_list)

    def set_configuration(self):

        # Initialize arrays
        self.linear_coeffs_array = np.zeros((self.eq.size, self.eq.size), dtype=float)  # dtype=complex)
        self.nonlinear_cubic_coeffs_array = np.zeros((self.eq.size, self.eq.size), dtype=float)

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

        self.get_neighbors()

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

    def get_neighbors(self):
        for j in range(self.eq.size):
            self.eq.mask_array[j].neighbors.clear()
            for k in range(self.eq.size):
                if (self.linear_coeffs_array[j][k].real != 0 or self.linear_coeffs_array[j][k].imag != 0) and j != k:
                    self.eq.mask_array[j].neighbors.append(k)

    def initialize_arrays(self):
        """Создает пустые массивы для хранения результатов"""
        self.t = np.linspace(self.com.T1, self.com.T2, self.com.M, endpoint=False)
        self.z = np.linspace(self.com.L1, self.com.L2, self.com.N + 1)
        self.omega = fftfreq(self.com.M, self.com.tau) * 2 * pi
        self.omega2 = self.omega ** 2

        self.numerical_solution = np.zeros((self.com.N + 1, self.eq.size, self.com.M), dtype=complex)
        self.energy = np.zeros((self.eq.size, self.com.N + 1), dtype=float)
        self.peak_power = np.zeros((self.eq.size, self.com.N + 1), dtype=float)

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
        self.numerical_solution[0] = initial_condition.astype(complex)

        # Рассчитываем начальную энергию и мощность
        for k in range(self.eq.size):
            self.energy[k][0] = get_energy_rectangles(self.numerical_solution[0][k], self.com.tau)
            self.peak_power[k][0] = np.max(np.abs(self.numerical_solution[0][k]) ** 2)

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
            if 'z' in self.pulses[k].__code__.co_varnames:
                self.numerical_solution[0][k] = self.pulses[k](
                    t=self.t, z=0, **pulse_params
                )
            else:
                self.numerical_solution[0][k] = self.pulses[k](t=self.t, **pulse_params)

        for k in range(self.eq.size):
            self.energy[k][0] = get_energy_rectangles(self.numerical_solution[0][k], self.com.tau)
            self.peak_power[k][0] = np.max(np.abs(self.numerical_solution[0][k]) ** 2)

    def calculate_D_matrix(self):
        if all(self.eq.beta2 == 0.0):
            single_matrix = create_simple_dispersion_free_matrix(self.linear_coeffs_array, self.eq.alpha,
                                                                 self.eq.g_0, self.com.h)
            flat_mat = single_matrix.flatten()
            self.D = np.full((self.com.M, self.eq.size**2), flat_mat).T
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
        # TODO: Может выделить случай use_gpu в отдельную функцию?
        if self.use_torch:
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
            if self.use_torch:
                # Выполнение на PyTorch
                psi_gpu = ssfm_order2_pytorch(psi_gpu, energy_gpu, D_gpu, gamma_gpu, E_sat_gpu, g_0_gpu, self.com.h,
                                              self.com.tau, self.eq.noise_amplitude)

                # Копирование данных с GPU на CPU в конце итерации, с использованием pinned memory
                self.numerical_solution[n + 1] = psi_gpu.cpu().numpy()
                self.energy[:, n + 1] = energy_gpu.cpu().numpy()
            else:
                # Выполнение на NumPy
                self.numerical_solution[n + 1] = ssfm_order2(self.numerical_solution[n], self.energy[:, n], self.D,
                                                             self.eq.gamma, self.eq.E_sat, self.eq.g_0, self.com.h,
                                                             self.com.tau, self.com.damp_length, self.eq.noise_amplitude)

            if self.use_torch:
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

        if self.use_torch:
            self.numerical_solution[-1] = psi_gpu.cpu().numpy()
            self.energy[:, -1] = energy_gpu.cpu().numpy()

        # Закрытие интерактивного режима после завершения симуляции
        if print_modulus:
            finalize_plot()

    def run_resonator_simulation_nocos(self, backward_energy):
        """
        Без учёта взаимодействия частот прямой и обратной волн.
        в перспективе для более высокого порядка можно добавить флаг
        """
        if self.D is None:
            self.calculate_D_matrix()

        fast_nocos_resonator_run(self.com.N, self.eq.size, self.numerical_solution, self.energy, backward_energy,
                                 self.D, self.eq.gamma, self.eq.E_sat, self.eq.g_0, self.com.h, self.com.tau,
                                 self.eq.noise_amplitude)

    def run_resonator_simulation_fullcos(self, backward_solution, print_modulus=False, print_interval=10):
        """
        С учётом взаимодействия частот прямой и обратной волн.
        в перспективе для более высокого порядка можно добавить флаг
        """
        if self.D is None:
            self.calculate_D_matrix()

        # Инициализация графика, если нужно
        if print_modulus:
            fig, ax, line = init_modulus_plot()

        for n in trange(self.com.N):
            # Выполнение на NumPy
            self.numerical_solution[n + 1] = ssfm_order1_resonator_fullcos(self.numerical_solution[n], backward_solution[:, n],
                                                                           self.D, self.eq.gamma, self.eq.E_sat, self.eq.g_0,
                                                                           self.com.h, self.com.tau, self.eq.noise_amplitude)

            for k in range(self.eq.size):
                self.energy[k][n + 1] = get_energy_rectangles(self.numerical_solution[n + 1][k], self.com.tau)

            for k in range(self.eq.size):
                self.peak_power[k][n + 1] = np.max(np.abs(self.numerical_solution[n + 1][k]) ** 2)

            # Обновление графика через каждые `print_interval` шагов, если включен флаг `print_modulus`
            if print_modulus and (n + 1) % print_interval == 0:
                update_modulus_plot(fig, ax, line, self.numerical_solution[n + 1], n)

        # Закрытие интерактивного режима после завершения симуляции
        if print_modulus:
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

         i dU/dz - beta2/2 d^2U/dt^2 + gamma |U|^2 U = 0

         к безразмерному виду

        i dU/dz + d^2U/dt^2 + |U|^2 U = 0.

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

        self.eq.beta2 = 2 * np.sign(beta2) if beta2 != 0.0 else 0.0  # [1]
        self.eq.gamma = 1.0 if gamma != 0.0 else 0.0  # [1]
        self.eq.E_sat /= energy_scale  # [1]
        self.eq.alpha /= coupling_coefficient  # [1]
        self.eq.g_0 /= coupling_coefficient  # [1]
        dimensional_coupling_coefficient = self.eq.coupling_coefficient
        self.eq.coupling_coefficient = 1.0  # [1]
        self.use_dimensional = False
        self.eq.__post_init__()
        self.calculate_D_matrix()

        self.linear_coeffs_array /= dimensional_coupling_coefficient
        self.linear_coeffs_array += np.diag(np.full(self.eq.size, -self_coefficient))
        if print_linear_coeffs_array:
            print_matrix(self.linear_coeffs_array, "linear_coeffs_array")

        self.get_neighbors()
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

         i dU/dz + d^2U/dt^2 + |U|^2 U = 0

         к размерному виду

         i dU/dz - beta2/2 d^2U/dt^2 + gamma |U|^2 U = 0.

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

        self.eq.beta2 = beta2  # [ps^2/m]
        self.eq.gamma = gamma  # [1/(W*m)]
        self.eq.E_sat *= energy_scale  # [pJ]
        self.eq.alpha *= coupling_coefficient  # [1/m]
        self.eq.g_0 *= coupling_coefficient  # [1/m]
        self.eq.coupling_coefficient = coupling_coefficient  # [1/m]
        self.use_dimensional = True
        self.eq.__post_init__()
        self.calculate_D_matrix()

        self.linear_coeffs_array -= np.diag(np.full(self.eq.size, -self_coefficient))
        self.linear_coeffs_array *= coupling_coefficient  # Пока нет реализации для уравнений Манакова
        if print_linear_coeffs_array:
            print_matrix(self.linear_coeffs_array, "linear_coeffs_array")

        self.get_neighbors()
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
