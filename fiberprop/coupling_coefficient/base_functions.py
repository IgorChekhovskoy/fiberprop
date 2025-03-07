import math
import numpy as np
import scipy.integrate as si
import scipy.special as sp
import matplotlib.pyplot as plt

from fiberprop.solver import CoreConfig

# Корни функции Бесселя. Первый индекс - порядок функции Бесселя (J0, J1, J2 или J3).
# Второй - номер корня минус один.
BESSEL_ROOTS = np.array([
    [2.404825557695773, 5.520078110286311, 8.653727912911013, 11.791534439014281],
    [3.831705970207512, 7.015586669815618, 10.173468135062723, 13.323691936314223],
    [5.135622301840682, 8.417244140399866, 11.619841172149059, 14.795951782351260],
    [6.380161895923983, 9.761023129981670, 13.015200721698433, 16.223466160318768]
], dtype=float)


def scipy_double_integral_by_circle(R, eps, fiber, light, core_center_coords, core_indexes, func):
    """Вычисляет двойной интеграл по круговой области в полярных координатах."""

    def polar_integrand(r, theta):
        # Преобразование полярных координат в декартовы
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        # Вычисление значения функции с учетом якобиана (r)
        return func(fiber, light, core_center_coords, core_indexes, x, y) * r

    # Вычисление интеграла с использованием scipy.dblquad
    integral, error = si.dblquad(
        polar_integrand,
        0, 2 * math.pi,  # Пределы угла theta (внешний интеграл)
        lambda theta: 0,  # Нижний предел радиуса r для каждого theta
        lambda theta: R,  # Верхний предел радиуса r для каждого theta
        epsabs=eps,
        epsrel=eps
    )

    return (integral, error)


def int_f2(fiber, light, core_center_coords, core_indexes, x, y):
    """ Интеграл от квадрата моды """
    x0, y0 = core_center_coords[core_indexes[0]]
    R = fiber.cladding_diameter * 0.5
    temp = 0.0
    if x**2 + y**2 <= R**2:
        temp = get_lp_mode(0, 1, fiber, light, x - x0, y - y0)**2
    return temp


def int_f4(fiber, light, core_center_coords, core_indexes, x, y):
    """ Интеграл от моды в четвёртой степени """
    x0, y0 = core_center_coords[core_indexes[0]]
    R = fiber.cladding_diameter * 0.5
    temp = 0.0
    if x**2 + y**2 <= R**2:
        temp = get_lp_mode(0, 1, fiber, light, x - x0, y - y0)**4
    return temp


def n_mode(fiber, core_center_coords, core_indexes, x, y):
    """ Коэффициент в интеграле, описывающем связь между двумя сердцевинами """
    c_diam = fiber.cladding_diameter
    if x**2 + y**2 < c_diam**2:
        R = fiber.core_radius
        for i, (x0, y0) in enumerate(core_center_coords):
            if (x - x0)**2 + (y - y0)**2 < R**2 and i != core_indexes[0]:
                return fiber.n_core**2 - fiber.n_cladding**2
        return 0.0
    return 0.0


def int_integral(fiber, light, core_center_coords, core_indexes, x, y):
    """ Интеграл, описывающий связь между двумя сердцевинами """
    x0, y0 = core_center_coords[core_indexes[0]]
    x1, y1 = core_center_coords[core_indexes[1]]
    R = fiber.cladding_diameter * 0.5
    temp = 0.0
    if x**2 + y**2 <= R**2:
        cc = n_mode(fiber, core_center_coords, core_indexes, x, y)
        temp += cc * get_lp_mode(0, 1, fiber, light, x - x0, y - y0) * get_lp_mode(0, 1, fiber, light, x - x1, y - y1)
    return temp


def plot_core_centers(core_center_coords, core_radius, cladding_diameter, title='Fiber scheme', color='red'):
    """
    Функция для отрисовки центров ядер на плоскости (в мкм) с учетом их радиуса.

    Входные параметры:
      - core_center_coords: список кортежей (x, y) с координатами центров ядер
      - core_radius: радиус ядра
      - cladding_diameter: диаметр волокна
      - title: заголовок графика (по умолчанию 'Core centers positions')
      - color: цвет кругов (по умолчанию 'red')
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Рисуем круги для каждого ядра
    for (x, y) in core_center_coords:
        circle = plt.Circle((x, y), core_radius, color=color, alpha=0.5)
        ax.add_patch(circle)

    # Рисуем окружность, обозначающую границу волокна
    fiber_circle = plt.Circle((0, 0), cladding_diameter / 2, color='black', fill=False, linestyle='--')
    ax.add_patch(fiber_circle)

    # Устанавливаем границы графика
    limit = cladding_diameter / 2 + core_radius
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    plt.xlabel('X [μm]')
    plt.ylabel('Y [μm]')
    plt.title(title)
    plt.grid(True)
    plt.show()


def get_coupling_coefficients(fiber, light, eps=1e-3, display_debug_info=False):
    """
    Вычисляет матрицу коэффициентов связи между сердцевинами многожильного оптического волокна.

    Параметры:
        fiber (Fiber): Объект волокна с заданными параметрами.
        light (Light): Параметры излучения (длина волны и др.).
        eps (float, optional): Точность вычисления интегралов. По умолчанию 1e-3.
        display_debug_info (bool, optional): Если True, отображает график расположения сердцевин.

    Возвращает:
        tuple:
            - numpy.ndarray: Матрица коэффициентов связи [1/м].
            - numpy.ndarray: Матрица абсолютных ошибок [1/м].

    Поддерживаемые конфигурации волокна:
        - Пустое кольцо (empty_ring)
        - Шестиугольная решетка (hexagonal)
        - Двухсердцевинное волокно (dual_core)

    Примечания:
        - Коэффициенты связи вычисляются на основе численного интегрирования.
        - Используется кэширование значений интегралов для ускорения расчетов.
        - Коэффициенты нормируются на 1e+2 для удобства представления.
    """

    R = 0.5 * fiber.cladding_diameter
    core_count = fiber.core_count
    distance_to_fiber_center = fiber.distance_to_fiber_center
    core_center_coords = []

    if fiber.core_configuration is CoreConfig.empty_ring:
        for i in range(core_count):
            phi = 2.0 * math.pi * i / core_count
            coords = (distance_to_fiber_center * math.cos(phi), distance_to_fiber_center * math.sin(phi))
            core_center_coords.append(coords)
    elif fiber.core_configuration is CoreConfig.hexagonal:
        for i in range(core_count):
            # Вычисляем двумерный радиус для определения кольца
            dimensional_radius = np.sqrt(
                (fiber.eq.mask_array[i].number_2d_x * 0.5) ** 2 +
                (fiber.eq.mask_array[i].number_2d_y * 0.5 * np.sqrt(3)) ** 2
            )
            ring_index = int(np.ceil(dimensional_radius))
            x_coord = distance_to_fiber_center[ring_index] * fiber.eq.mask_array[i].number_2d_x * 0.5 / max(ring_index, 1)
            y_coord = distance_to_fiber_center[ring_index] * fiber.eq.mask_array[i].number_2d_y * 0.5 * np.sqrt(3) / max(ring_index, 1)
            core_center_coords.append((x_coord, y_coord))
    elif fiber.core_configuration is CoreConfig.dual_core:
        return get_coupling_coeff_2_core_fiber(fiber, light)
    else:
        raise ValueError('This fiber configuration is not yet supported')

    if display_debug_info:
        plot_core_centers(core_center_coords, fiber.core_radius, fiber.cladding_diameter)

    # Предполагаем, что все ядра идентичны, поэтому интеграл по диагонали (self-coupling) одинаков для всех.
    diag_result = scipy_double_integral_by_circle(R, eps, fiber, light, core_center_coords, (0, 0), int_f2)
    diag_val = diag_result[0]
    diag_up = diag_result[0] + diag_result[1]
    diag_low = diag_result[0]- diag_result[1]

    k = 0.5 * (light.k0 ** 2) / fiber.get_beta(light)

    # Инициализация матриц для коэффициентов связи и ошибок
    coup_mat = np.zeros((core_count, core_count), dtype=float)
    error_mat = np.zeros((core_count, core_count), dtype=float)

    # Словарь для кэширования результатов off-diagonal интегралов по расстоянию
    cache = {}

    # Вычисление коэффициентов связи для пар ядер (только для пар с разными ядрами)
    for m in range(core_count):
        for p in range(m):
            # Расчет расстояния между центрами ядер m и p
            dx = core_center_coords[m][0] - core_center_coords[p][0]
            dy = core_center_coords[m][1] - core_center_coords[p][1]
            d = math.sqrt(dx * dx + dy * dy)
            # Используем округление для устранения неточностей при сравнении
            d_key = round(d, 6)
            if d_key not in cache:
                # Вычисление off-diagonal интеграла с использованием функции int_integral
                result = scipy_double_integral_by_circle(R, eps, fiber, light, core_center_coords, (m, p), int_integral)
                cache[d_key] = result
            else:
                result = cache[d_key]

            int2 = result[0]
            int2_up = result[0] + result[1]
            int2_low = result[0] - result[1]

            # Вычисление коэффициента связи qmp для пары ядер (m, p)
            # Поскольку ядра идентичны, используем один и тот же интеграл по диагонали для обоих ядер.
            qmp = k * int2 / (diag_val ** 0.5 * diag_val ** 0.5)
            coupling = qmp * 1e+4
            coup_mat[m][p] = coupling
            coup_mat[p][m] = coupling

            # Вычисление ошибок с учетом верхней и нижней оценок интегралов
            full_err_up = k * int2_up / (diag_low ** 0.5 * diag_low ** 0.5)
            full_err_low = k * int2_low / (diag_up ** 0.5 * diag_up ** 0.5)
            error = (full_err_up - full_err_low) * 1e+4 / 2
            error_mat[m][p] = error
            error_mat[p][m] = error

    return coup_mat * 1e+2, error_mat * 1e+2


def get_coupling_coeff_2_core_fiber(fiber, light):
    """ Коэффициент связи для двухсердцевинного волокна """
    V = fiber.core_radius * light.k0 * (
        ((1.0 + fiber.delta_n_core) * fiber.n_cladding)**2 - fiber.n_cladding**2)**0.5

    c0 = 5.2789 - 3.663 * V + 0.3841 * V**2
    c1 = -0.7769 + 1.2252 * V - 0.0152 * V**2
    c2 = -0.0175 - 0.0064 * V - 0.0009 * V**2

    d = 2.0 * fiber.distance_to_fiber_center[0] / fiber.core_radius

    coup_mat = np.zeros((2, 2), dtype=float)
    coup_mat[0][1] = math.pi * V * math.exp(-(c0 + c1 * d + c2 * d**2)) / (
        2.0 * light.k0 * fiber.n_cladding * fiber.core_radius**2)
    coup_mat[1][0] = coup_mat[0][1]

    error_mat = np.zeros((2, 2), dtype=float)
    return coup_mat, error_mat


def get_lp_mode(l, m, fiber, light, x, y):
    r = (x**2 + y**2)**0.5
    phi = np.arctan2(y, x)

    v = fiber.core_radius * light.k0 * fiber.NA

    if l == 0 and m == 1:
        u = (1.0 + 2.0**0.5) * v / (1.0 + (4.0 + v**4)**0.25)
        w = (v**2 - u**2)**0.5
    else:
        if l >= 0 and m >= 1:
            uc = BESSEL_ROOTS[int(abs(l - 1))][m - 1]
            if uc > v:
                print(f"Mode LP{l}{m} for Core Radius = {fiber.core_radius} mkm: ERROR: This mode is not allowed for this fiber geometry!")
                return 1

            s = (uc**2 - l**2 - 1)**0.5
            u = uc * math.exp((math.asin(s / uc) - math.asin(s / v)) / s)
            w = (v**2 - u**2)**0.5
        else:
            print("ERROR: Such LP mode does not exist!")
            return 2

    core_radius = fiber.core_radius
    if r < core_radius:
        return sp.jv(l, u * r / core_radius) / sp.jv(l, u) * math.cos(l * phi)
    else:
        return sp.kn(l, w * r / core_radius) / sp.kn(l, w) * math.cos(l * phi)
