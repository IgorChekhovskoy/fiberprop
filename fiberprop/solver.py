from tqdm import trange
from scipy.fft import fftfreq
from dataclasses import dataclass, field
from typing import Union
from math import sqrt, pi
from numba import njit

from .fiber_geometry import make_eq_mask, CoreConfig, get_core_count
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

from .ssfm_mcf_pytorch import ssfm_order2_pytorch

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
        if isinstance(self.beta1, (int, float, list)):
            self.beta1 = np.array(self.beta1, dtype=float)
            if self.beta1.ndim == 0:
                self.beta1 = np.full(self.size, self.beta1, dtype=float)
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
            display_debug_info=False
    ):
        self.exp_2g0h_full = None
        self.exp_g0h_full = None
        self.g0_h_full = None
        self.gamma_h_full = None

        self.exp_2g0h_half = None
        self.exp_g0h_half = None
        self.g0_h_half = None
        self.gamma_h_half = None

        self.E_sat_t = None
        self.g0_t = None

        self._taper_np = None
        self._taper_t = None

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
        self.numerical_solution = None
        self.energy = None
        self.peak_power = None
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
        self.initialize_arrays()

        # Обработка начальных условий
        if initial_condition is not None:
            self.validate_initial_condition(initial_condition)
            self.apply_initial_condition(initial_condition)
        else:
            self.initialize_with_pulses(pulses, pulse_params_list)

        self._prepare_taper()

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

    def initialize_arrays(self):
        """Создает пустые массивы для хранения результатов"""
        self.t = np.linspace(self.com.T1, self.com.T2, self.com.M, endpoint=False)
        self.z = np.linspace(self.com.L1, self.com.L2, self.com.N + 1)
        self.omega = fftfreq(self.com.M, self.com.tau) * 2 * pi
        self.omega2 = self.omega ** 2

        self.numerical_solution = np.zeros((self.stored_steps_count, self.eq.size, self.com.M), dtype=complex)
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

            pulse_params = {name: (val[k] if isinstance(val, np.ndarray) else val)
                            for name, val in pulse_params.items()}

            if 'z' in self.pulses[k].__code__.co_varnames:
                self.numerical_solution[0][k] = self.pulses[k](t=self.t, z=0, **pulse_params)
            else:
                self.numerical_solution[0][k] = self.pulses[k](t=self.t, **pulse_params)

        for k in range(self.eq.size):
            self.energy[k][0] = get_energy_rectangles(self.numerical_solution[0][k], self.com.tau)
            self.peak_power[k][0] = np.max(np.abs(self.numerical_solution[0][k]) ** 2)

    # ─── solver.py ─────────────────────────────────────────────
    def calculate_D_matrix(self) -> None:
        """
        Строит дисперсионно-связную матрицу и подготавливает
        представления, удобные для CPU- и GPU-веток.
        """
        n, M = self.eq.size, self.com.M

        # 1) плоская матрица (n², M)
        if np.all(self.eq.beta1 == 0.0) and np.all(self.eq.beta2 == 0.0):
            self.D = create_simple_dispersion_free_matrix(
                self.linear_coeffs_array, self.eq.alpha,
                self.eq.g_0, self.com.h
            )  # (n, n)
        else:
            self.D = get_pade_exponential2(
                create_freq_matrix(
                    self.linear_coeffs_array,
                    self.eq.beta1, self.eq.beta2,
                    self.eq.alpha, self.eq.g_0,
                    self.omega, self.com.h
                )
            )  # (n², M)

            # 2) представление (n, n, M)  — NumPy-view без копии
            self.D = self.D.reshape(n, n, M)

        # 3) сразу готовим тензор на GPU, если будем считать в torch
        if self.use_torch:
            self.D_pytorch = torch.as_tensor(
                self.D,
                dtype=self.ctype,
                device=self.device
            )

    def prepare_halfstep_constants(self):
        if self.gamma_h_half is not None:
            return

            # ── 1. CPU-массивы ───────────────────────────────────────────────
        half_step = 0.5 * self.com.h
        self.gamma_h_half = self.eq.gamma * half_step  # (n,)
        self.g0_h_half = self.eq.g_0 * half_step
        self.exp_g0h_half = np.exp(self.g0_h_half)
        self.exp_2g0h_half = np.exp(2.0 * self.g0_h_half)

        # ── 2. PyTorch-тензоры (если используем torch) ───────────────────
        if self.use_torch:
            to_t = lambda arr: torch.as_tensor(arr,
                                               dtype=self.dtype,
                                               device=self.device)

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

        taper = np.ones(M, dtype=np.complex128)
        taper[:L] = edge
        taper[M - L:] = edge[::-1]

        self._taper_np = taper
        if self.use_torch:
            self._taper_t = torch.as_tensor(
                taper, dtype=self.ctype, device=self.device)

    def filter_params(self, func, pulse_params):
        # Получаем список параметров, которые принимает функция
        func_params = func.__code__.co_varnames[:func.__code__.co_argcount]
        # Фильтруем параметры, чтобы оставить только те, которые нужны функции
        filtered_params = {k: v for k, v in vars(self.eq).items() if k in func_params}
        # Обновляем параметры с учетом pulse_params
        filtered_params.update({k: v for k, v in pulse_params.items() if k in func_params})
        return filtered_params

    # Основная функция моделирования
    def run_numerical_simulation(
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
            self.calculate_D_matrix()
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

        if not self.use_torch:
            for n in trange(self.com.N):
                # ---------- очередной шаг --------------------------------
                psi_next = ssfm_order2(
                    psi_next,  # последний сохранённый
                    self.energy[:, n],  # энергия предыдущего
                    self,
                    self.com.h, tau,
                    self.com.damp_length,
                    self.eq.noise_amplitude,
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
                    update_modulus_plot(fig, ax, line, psi_next, self.z, n)

        # ─────────────────────────────────────── GPU-ветка ───────────────
        else:
            psi_gpu = torch.as_tensor(
                self.numerical_solution[0], dtype=self.ctype, device=self.device
            )

            for n in trange(self.com.N):
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

    def run_resonator_simulation_fullcos(self, backward_solution, draw_modulus=False, draw_interval=10):
        """
        С учётом взаимодействия частот прямой и обратной волн.
        в перспективе для более высокого порядка можно добавить флаг
        """
        if self.D is None:
            self.calculate_D_matrix()

        # Инициализация графика, если нужно
        if draw_modulus:
            fig, ax, line = init_modulus_plot(t=self.t)

        for n in trange(self.com.N):
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

         i dU/dz + i beta1 dU/dt - beta2/2 d^2U/dt^2 + gamma |U|^2 U = 0

         к безразмерному виду

        i dU/dz + i beta1/T/C dU/dt + d^2U/dt^2 + |U|^2 U = 0.

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
        self.calculate_D_matrix()

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

         i dU/dz + i beta1 dU/dt + d^2U/dt^2 + |U|^2 U = 0

         к размерному виду

         i dU/dz + i beta1 T C dU/dt - beta2/2 d^2U/dt^2 + gamma |U|^2 U = 0.

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
        self.calculate_D_matrix()

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
    ax.set_ylim(1e-15, 1)
    ax.grid(True, which='both', axis='y')

    # служебные поля
    ax._mcf_scaling_mode = scaling_mode
    ax._mcf_margin = margin
    ax._mcf_global_max = 1e-15
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

    plt.ion();
    plt.show()
    return fig, ax, line


def update_modulus_plot(fig, ax, line, data_2d: np.ndarray, z: np.ndarray, step: int):
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
    ax.set_ylim(1e-15, ylim_top)

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
    ax.set_xlabel('t [ps]')

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
    ax.set_title(f'step = {step + 1} of {z.size - 1}, z = {z[step + 1]:.3g},    max |u| = {max_mod:.3g}')
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
