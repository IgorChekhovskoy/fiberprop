from fiberprop.solver import CoreConfig
import numpy as np
import scipy.integrate as si
import scipy.special as sp
import multiprocessing
import math

# Корни функции Бесселя. Первый индекс - порядок функции Бесселя (J0, J1, J2 или J3).
# Второй - номер корня минус один.
BESSEL_ROOTS = np.array([
    [2.404825557695773, 5.520078110286311, 8.653727912911013, 11.791534439014281],
    [3.831705970207512, 7.015586669815618, 10.173468135062723, 13.323691936314223],
    [5.135622301840682, 8.417244140399866, 11.619841172149059, 14.795951782351260],
    [6.380161895923983, 9.761023129981670, 13.015200721698433, 16.223466160318768]
], dtype=float)


def scipy_double_integral_by_circle(R, eps, fiber, light, core_center_coords, core_indexes, func):
    """ Интеграл библиотечным методом и Fortran по круглой области """
    return si.dblquad(
        lambda x, y: func(fiber, light, core_center_coords, core_indexes, x, y),
        -R, R,
        lambda y: -math.sqrt(R**2 - y**2),
        lambda y: math.sqrt(R**2 - y**2),
        epsabs=eps,
        epsrel=eps
    )


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


def get_coupling_coefficients(fiber, light, eps=1e-3, proc_num=1):
    """ Первый выход - матрица связей [1/m], второй выход - матрица ожидаемых абсолютных ошибок [1/m] """
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
        delta_x = 0.0
        delta_y = 0.0
        core_center_coords.append((delta_x, delta_y))
        for i in range(core_count - 1):
            phi = 2.0 * math.pi * i / (core_count - 1)
            x_coord = distance_to_fiber_center * math.cos(phi) + delta_x
            y_coord = distance_to_fiber_center * math.sin(phi) + delta_y
            core_center_coords.append((x_coord, y_coord))
    elif fiber.core_configuration is CoreConfig.dual_core:
        coup_mat = get_coupling_coeff_2_core_fiber(fiber, light)
        return coup_mat
    else:
        raise ValueError('This fiber configuration is not yet supported')

    # pool_result = []
    # work_args = []
    # for m in range(1, core_count):
    #     like_pair = (m, m)
    #     work_args.append((R, eps, fiber, light, core_center_coords, like_pair, int_f2))
    #     for p in range(m):
    #         like_pair = (m, p)
    #         work_args.append((R, eps, fiber, light, core_center_coords, like_pair, int_integral))

    # with multiprocessing.Pool(proc_num) as pool:
    #     pool_result = pool.starmap(scipy_double_integral_by_circle, work_args)

    pool_result = []
    for m in range(1, core_count):
        like_pair = (m, m)
        pool_result.append(scipy_double_integral_by_circle(R, eps, fiber, light, core_center_coords, like_pair, int_f2))
        for p in range(m):
            like_pair = (m, p)
            pool_result.append(scipy_double_integral_by_circle(R, eps, fiber, light, core_center_coords, like_pair, int_integral))

    coup_mat = np.zeros((core_count, core_count), dtype=float)
    error_mat = np.zeros((core_count, core_count), dtype=float)
    idx = 0
    for m in range(1, core_count):
        Im = pool_result[idx][0]
        Im_up_est = pool_result[idx][0] + pool_result[idx][1]
        Im_low_est = pool_result[idx][0] - pool_result[idx][1]
        idx += 1
        for p in range(m):
            Ip = Im
            Ip_up_est = Im_up_est
            Ip_low_est = Im_low_est
            k = 0.5 * (light.k0**2) / fiber.get_beta(light)
            int2 = pool_result[idx][0]
            int2_up_est = pool_result[idx][0] + pool_result[idx][1]
            int2_low_est = pool_result[idx][0] - pool_result[idx][1]
            idx += 1
            qmp = k * int2 / (Im * Ip)**0.5
            coup_mat[m][p] = qmp * 1e+4
            coup_mat[p][m] = qmp * 1e+4
            full_err_up_est = k * int2_up_est / (Ip_low_est * Im_low_est)**0.5
            full_err_low_est = k * int2_low_est / (Ip_up_est * Im_up_est)**0.5
            error_mat[m][p] = (full_err_up_est - full_err_low_est) * 1e+4 / 2
            error_mat[p][m] = (full_err_up_est - full_err_low_est) * 1e+4 / 2

    return coup_mat * 1e+2, error_mat * 1e+2


def get_coupling_coeff_2_core_fiber(fiber, light):
    """ Коэффициент связи для двухсердцевинного волокна """
    V = fiber.core_radius * light.k0 * (
        ((1.0 + fiber.delta_n_core) * fiber.n_cladding)**2 - fiber.n_cladding**2)**0.5

    c0 = 5.2789 - 3.663 * V + 0.3841 * V**2
    c1 = -0.7769 + 1.2252 * V - 0.0152 * V**2
    c2 = -0.0175 - 0.0064 * V - 0.0009 * V**2

    d = 2.0 * fiber.distance_to_fiber_center / fiber.core_radius

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
