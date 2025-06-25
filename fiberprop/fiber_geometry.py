# fiber_geometry.py

from dataclasses import dataclass
from enum import Enum
import numpy as np
from math import sqrt


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
        Массив индексов соседних ядер. Обновление neighbors происходит только в Solver.
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


def get_ring_count(core_configuration, core_count: int) -> float:
    """
    Определяет число колец (ring_count) по числу сердцевин для данной конфигурации.

    Параметры:
        core_configuration (CoreConfig): конфигурация расположения сердцевин
        core_count (int): число сердцевин

    Возвращает:
        float: приблизительное значение ring_count

    Исключения:
        ValueError: если конфигурация неизвестна или число сердцевин не поддерживается
    """
    if core_count <= 0:
        raise ValueError("Число сердцевин должно быть положительным")

    if core_configuration is CoreConfig.square:
        raise NotImplementedError("Обратная функция для квадратной решетки не реализована")

    elif core_configuration is CoreConfig.hexagonal:
        if core_count == 1:
            return 0.0
        elif core_count == 7:
            return 1.0
        elif core_count == 13:
            return np.sqrt(3)
        elif core_count == 19:
            return 2.0
        elif core_count == 31:
            return np.sqrt(7)
        elif core_count == 37:
            return 3.0
        elif core_count == 43:
            return np.sqrt(12)
        else:
            raise ValueError(f"Неизвестное количество сердцевин: {core_count}")

    return 0


def make_eq_mask(core_configuration: CoreConfig,
                 size: int,
                 ring_count: float,
                 display_debug_info: bool = False) -> list[Mask]:
    """
    Строит список Mask (ядра + соседи) на основе конфигурации.

    Параметры:
    ----------
    core_configuration : CoreConfig
        Тип конфигурации сердцевин (hexagonal, square и т.д.)
    size : int
        Размерность (для 1D конфигураций — число сердцевин, для 2D — количество колец)
    ring_count : float
        Количество колец (только для 2D конфигураций)
    display_debug_info : bool
        Флаг печати отладочной информации

    Возвращает:
    -----------
    mask_array : list[Mask]
        Список объектов Mask, содержащих координаты и соседей каждого ядра
    """

    from math import sqrt
    temp_array_size = int((1.0 + size * (size + 1.0)))
    temp_array = np.zeros((temp_array_size, temp_array_size), dtype=bool)
    center = temp_array_size // 2

    if core_configuration is CoreConfig.ring_with_center:
        for i in range(size + 1):
            temp_array[0][i] = True

    elif ((core_configuration is CoreConfig.empty_ring) or
          (core_configuration is CoreConfig.manakov_eq)):
        for i in range(size):
            temp_array[0][i] = True

    elif core_configuration is CoreConfig.square:
        for i in range(temp_array_size):
            for j in range(temp_array_size):
                if (i - center) ** 2 + (j - center) ** 2 <= ring_count ** 2 + 1e-13:
                    temp_array[i][j] = True

    elif core_configuration is CoreConfig.hexagonal:
        h_i = 1.0
        h_j = 1.0 / sqrt(3.0)
        for i in range(temp_array_size):
            for j in range(temp_array_size):
                if (h_i * (i - center)) ** 2 + (
                        h_j * (j - center)) ** 2 <= ring_count ** 2 * 4.0 / 3.0 + 1e-10 and \
                        (i + j - 2 * center) % 2 == 0:
                    temp_array[i][j] = True

    if display_debug_info:
        print_temp_array(temp_array_size, temp_array)

    mask_array = []
    index_1d = 0
    for i in range(temp_array_size):
        for j in range(temp_array_size):
            if temp_array[i][j]:
                mask_array.append(Mask(index_1d, i - center, j - center, []))
                index_1d += 1

    # Соседей определяем здесь же
    for j in range(len(mask_array)):
        mask_j = mask_array[j]
        mask_j.neighbors = []
        for k in range(len(mask_array)):
            if j == k:
                continue

            dx = abs(mask_j.number_2d_x - mask_array[k].number_2d_x)
            dy = abs(mask_j.number_2d_y - mask_array[k].number_2d_y)

            if core_configuration is CoreConfig.square:
                if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
                    mask_j.neighbors.append(k)

            elif core_configuration is CoreConfig.hexagonal:
                if (dx == 2 and dy == 0) or (dx == 1 and dy == 1):
                    mask_j.neighbors.append(k)

            elif core_configuration in [CoreConfig.empty_ring, CoreConfig.ring_with_center]:
                if abs(j - k) == 1 or abs(j - k) == len(mask_array) - 1:
                    mask_j.neighbors.append(k)
                if core_configuration is CoreConfig.ring_with_center and (j == 0 or k == 0):
                    mask_j.neighbors.append(k)

    return mask_array
