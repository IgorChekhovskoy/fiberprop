import math
import multiprocessing
import numpy as np
import copy
import time

from fiberprop.solver import CoreConfig
from fiberprop.coupling_coefficient.base_functions import get_coupling_coefficients, get_lp_mode
from fiberprop.coupling_coefficient.fiber import Fiber, FiberMaterial
from fiberprop.coupling_coefficient.light import Light


def print_matrix(matrix, name='matrix'):
    """ Функция реализует вывод матрицы в консоль """
    print(f'\n{name}: ')
    for row in matrix:
        print('\t'.join(f'{value: .3f}' for value in row))
    print('\n')


def example_of_coefficients_calculation():
    """ Пример расчётов матрицы и коэффициента связи, Керровской нелинейности и ДГС """
    start = time.time()

    lambda0 = 1.05018
    light = Light()
    light.lambda0 = lambda0

    fiber = Fiber()
    fiber.core_configuration = CoreConfig.hexagonal
    fiber.core_count = 7
    fiber.core_radius = 2.95
    fiber.cladding_diameter = 125.0
    fiber.n2 = 3.2
    fiber.distance_to_fiber_center = 17.3
    fiber.NA = 0.125
    fiber.core_material = FiberMaterial.SIO2_AND_GEO2_ALLOY
    fiber.material_concentration = 0.038
    fiber.set_refractive_indexes_by_lambda(lambda0)

    eps = 1e-3  # Желаемая точность интеграла
    # TODO : Георгий, у меня стабильно медленнее работает код при proc_num > 1
    proc_num = 1  # multiprocessing.cpu_count()  # Количество параллельных процессов

    coup_mat, err_mat = get_coupling_coefficients(fiber, light, eps, proc_num)
    print_matrix(coup_mat, 'Coupling matrix')
    print_matrix(err_mat, 'Matrix of estimated absolute errors')

    end = time.time()
    print(f'get_coupling_coefficients() work time: {end - start} s')

    couping_coefficient = coup_mat[0][1]
    coupling_coefficient_estimated_error = err_mat[0][1]
    print(f'Lambda = {fiber.distance_to_fiber_center * 2.0} mkm')
    print(f'k = {couping_coefficient} +- {coupling_coefficient_estimated_error} 1/cm')
    print(f'L = {0.5 * np.pi / couping_coefficient} cm \n')

    gamma, gamma_error = fiber.get_gamma(light, eps)
    print(f'Gamma = {gamma} +- {gamma_error} 1/(W*m)')
    beta2 = fiber.get_beta2(light)
    print(f'Beta2 = {beta2} (ps^2)/km')

    end = time.time()
    print(f'Full work time: {end - start} s')
    return couping_coefficient, gamma, beta2


def save_mode_distribution(fiber, light, grid_size, first_mode_param_count, second_mode_param_count):
    """ Тест. Сохранение распределений мод """
    cladding_diameter = fiber.cladding_diameter
    h = cladding_diameter / grid_size
    with open('Modes.txt', 'wt') as output_file:
        output_file.write('Variables=x,y')
        for l in range(first_mode_param_count):
            for m in range(second_mode_param_count):
                output_file.write(f',LP{l}{m + 1}')
        output_file.write(f'\nZone i={grid_size + 1} j={grid_size + 1}\n')

        for i in range(grid_size):
            for j in range(grid_size):
                x = -0.5 * cladding_diameter + i * h
                y = -0.5 * cladding_diameter + j * h
                output_file.write(f'{x:.17f}\t{y:.17f}\t')
                if x ** 2 + y ** 2 < (0.25 * cladding_diameter ** 2):
                    for l in range(first_mode_param_count):
                        for m in range(second_mode_param_count):
                            output_file.write(f'{get_lp_mode(l, m + 1, fiber, light, x, y):.17f}\t')
                else:
                    for l in range(first_mode_param_count):
                        for m in range(second_mode_param_count):
                            output_file.write(f'{-1.0:.17f}\t')
                output_file.write('\n')


def test1():
    """ Heterogeneous multi-fiberprop fibers: proposal and design principle, Koshiba, at all. 2009 """
    light = Light()
    light.lambda0 = 1.55
    fiber = Fiber()
    fiber.delta_n_core = 0.0035
    fiber.core_radius = 4.5
    fiber.distance_to_fiber_center = 0.5 * (5.0 * fiber.core_radius)

    beta = fiber.get_beta(light)
    print(f'b = {beta}')

    coeffs = get_coupling_coefficients(fiber, light, 1e-6, 4)[0]
    print(f'Lambda = {fiber.distance_to_fiber_center * 2.0} mkm \n')

    print(f'k1 = {coeffs[0][1] * 1e+4} 1/cm')
    print(f'L1 = {0.5 * np.pi / (coeffs[0][1] * 1e+4)} cm')
    print(f'k1 = {coeffs[0][1] * 1e+6} 1/m')
    print(f'L1 = {0.5 * np.pi / (coeffs[0][1] * 1e+6)} m')


def get_coupling_of_distance(fiber, light, d, couplings):
    """ Для двухъядерного волокна строит зависимость коэффициента связи
    от расстояния между ядрами, выраженного в числе радиусов ядер """
    fiber = copy.deepcopy(fiber)
    fiber.core_count = 2
    fiber.set_refractive_indexes_by_lambda(light.lambda0)
    N = len(d)
    couplings.resize(N, refcheck=False)

    for i in range(N):
        fiber.distance_to_fiber_center = 0.5 * fiber.core_radius * d[i]
        coeffs = get_coupling_coefficients(fiber, light, 1e-6, 4)[0]
        couplings[i] = coeffs[0][1]
        print(f'i = {i}')


def get_C_of_R_and_d(fiber, light, r1, r2, d1, d2, N):
    """ Зависимость коэффициента связи от радиуса ядер и расстояния между ними в радиусах """
    d = np.linspace(d1, d2, N + 1, dtype=float)
    r = np.linspace(r1, r2, N + 1, dtype=float)

    array_size = (N + 1) * (N + 1)
    couplings = np.empty(array_size, dtype=float)
    L = np.empty(array_size, dtype=float)
    T = np.empty(array_size, dtype=float)
    P = np.empty(array_size, dtype=float)

    fiber.set_refractive_indexes_by_lambda(light.lambda0)
    fiber.core_count = 2

    for i in range(N + 1):
        fiber.core_radius = r[i]
        gamma = fiber.get_gamma(light, 1e-6)  # [1/(W*m)]
        b2 = fiber.get_beta2(light)  # [(ps^2)/km]
        for j in range(N + 1):
            fiber.distance_to_fiber_center = 0.5 * (d[j] * r[i])
            coeffs = get_coupling_coefficients(fiber, light, 1e-6, 4)
            couplings[i * (N + 1) + j] = coeffs[0][1]  # [1/m]

            L[i * (N + 1) + j] = 1.0 / (couplings[i * (N + 1) + j] * 1e+6)  # [m]
            T[i * (N + 1) + j] = (-0.5 * b2 / (couplings[i * (N + 1) + j] * 1e+9)) ** 0.5  # [ps]
            P[i * (N + 1) + j] = 1e+6 * couplings[i * (N + 1) + j] / gamma  # [W]

            print(f'i = {i} j = {j} \n')

    with open('C_of_R_and_d.txt', 'wt') as output_file:
        output_file.write('Variables=Radius[mkm],distance[Radius],C[1/m],log10(C),L[m],T[ps],P[W]\n')
        output_file.write(f'Zone i={N + 1} j={N + 1}\n')
        for i in range(N + 1):
            for j in range(N + 1):
                output_file.write(f'{r[i]:.2f}\t {d[j]:.2f}\t {couplings[i * (N + 1) + j] * 1e+6:.7e}\t '
                                  f'{math.log10(couplings[i * (N + 1) + j]):.7f}\t {L[i * (N + 1) + j]:.7f}\t '
                                  f'{T[i * (N + 1) + j]:.7f}\t {P[i * (N + 1) + j]:.7f}\n')


def get_LTP_of_C(fiber, light, d1, d2, N):
    d = np.linspace(d1, d2, N + 1, dtype=float)

    gamma = fiber.get_gamma(light, 1e-6)  # [1/(W*m)]
    couplings = np.array([], dtype=float)
    get_coupling_of_distance(fiber, light, d, couplings)  # [1/mkm]

    L = np.empty(N + 1, dtype=float)
    T = np.empty(N + 1, dtype=float)
    P = np.empty(N + 1, dtype=float)

    for i in range(N + 1):
        print(f'C = {couplings[i] * 1e+6:.5e}\n')
        L[i] = 1.0 / (couplings[i] * 1e+6)  # [m]

        b2 = fiber.get_beta2(light)  # [ps^2/km]
        T[i] = math.sqrt(-0.5 * b2 / (couplings[i] * 1e+9))  # [ps]
        P[i] = 1e+6 * couplings[i] / gamma  # [W]

    with open('L_T_P_Of_C.txt', 'wt') as output_file:
        output_file.write('Variables=d,L[m],T[ps],P[W]\n')
        for i in range(N):
            output_file.write(f'{d[i]:.2f}\t {L[i]:.7f}\t {T[i]:.7f}\t {P[i]:.7f}\n')


def draw_beta_of_lambda(fiber, light, l1, l2, N):
    """ Функция анализирует константу распространения и выводит в файл её производные для различных длин волн """
    fiber = copy.deepcopy(fiber)
    light = copy.deepcopy(light)
    l = np.linspace(l1, l2, N + 1)

    beta = np.empty(N + 1, dtype=float)
    beta1 = np.empty(N + 1, dtype=float)
    beta2 = np.empty(N + 1, dtype=float)

    for i in range(N + 1):
        light.lambda0 = l[i]
        fiber.set_refractive_indexes_by_lambda(l[i])
        beta[i] = fiber.get_beta(light)
        beta1[i] = fiber.get_beta1(light)
        beta2[i] = fiber.get_beta2(light)

    with open('beta_Of_lambda.txt', 'wt') as output_file:
        output_file.write('Variables=lambda[mkm],beta[1/mkm],beta1[ns/m],beta2[ps^2/km]\n')
        for i in range(N + 1):
            output_file.write(f'{l[i]:.17f}\t {beta[i]:.17f}\t {beta1[i]:.17f}\t {beta2[i]:.17f}\n')