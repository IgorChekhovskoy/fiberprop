from Base_functions import *
from Fiber import *
import numpy as np
import copy
import time


def printMatrix(mat, name='matrix'):
    """  Функция реализует вывод матрицы в консоль
    """
    print('\n')
    print(f'{name}:')
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            print('{:.3f}'.format(mat[i][j]), end='\t')
        print('\n')
    return

def example_of_coefficients_calculation():
    """ Функция демонстрирует пример расчётов матрицы и коэффициента связи, Керровской нелинейности и ДГС """
    start = time.time()

    lambda0 = 1.05018
    light = Light()
    light.SetLambda0(lambda0)

    fiber = Fiber()
    fiber.SetCoreConfiguration(FiberConfig.Hexagonal)
    fiber.SetCoreCount(7)
    fiber.SetCoreRadius(2.95)
    fiber.SetCladdingDiameter(125.0)
    fiber.SetN2(3.2)
    fiber.SetDistanceToFiberCenter(17.3)
    fiber.SetNA(0.125)
    fiber.SetCoreMaterial(FiberMaterial.SiO2andGeO2_alloy)
    fiber.SetMaterialConcentration(0.038)
    fiber.SetRefractiveIndexesByLambda(lambda0)

    eps = 1e-6  # желаемая точность интеграла
    procNum = multiprocessing.cpu_count()  # количество параллельных процессов

    coupMat, errMat = getCouplingCoefficients(eps, fiber, light, procNum)
    printMatrix(coupMat, 'Coupling matrix')
    printMatrix(errMat, 'Matrix of estimate absolute errors')

    koef = coupMat[0][1]  # коэффициент связи
    koefEstErr = errMat[0][1]  # верхняя оценка абсолютной погрешности определения коэффициента связи
    print('Lambda = {} mkm'.format(fiber.GetDistanceToFiberCenter() * 2.0))
    print('k = {} +- {} 1/cm'.format(koef, koefEstErr))
    print('L = {} cm \n'.format(0.5 * np.pi / koef))

    gamma, gammaError = fiber.GetGamma(eps, light)  # коэффициент нелинейности и верхняя оценка его абсолютной погрешности
    print('Gamma = {} +- {} 1/(W*m)'.format(gamma, gammaError))
    beta2 = fiber.GetBeta2(light)
    print('Beta2 = {} (ps^2)/km'.format(beta2))

    end = time.time()
    print(f'Full work time: {end - start} s')
    return (koef, gamma, beta2)


def saveModeDistribution(fiber, light, gridSize, firstModeParameterCount, secondModeParameterCount):
    """ Тест. Сохранение распределений мод """
    claddingDiameter = fiber.GetCladdingDiameter()
    h = claddingDiameter / gridSize
    output_file = open('Modes.txt', 'wt')
    output_file.write('Variables=x,y')

    for l in range(firstModeParameterCount):
        for m in range(secondModeParameterCount):
            output_file.write(',LP{}{}'.format(l, m + 1))

    output_file.write('\nZone i={} j={}\n'.format(gridSize + 1, gridSize + 1))

    for i in range(gridSize):
        for j in range(gridSize):
            x = -0.5 * claddingDiameter + i * h
            y = -0.5 * claddingDiameter + j * h
            output_file.write('{:.17f}\t{:.17f}\t'.format(x, y))
            if x ** 2 + y ** 2 < (0.25 * claddingDiameter ** 2):
                for l in range(firstModeParameterCount):
                    for m in range(secondModeParameterCount):
                        output_file.write('{:.17f}\t'.format(getLPmode(l, m + 1, fiber, light, x, y)))
            else:
                for l in range(firstModeParameterCount):
                    for m in range(secondModeParameterCount):
                        output_file.write('{:.17f}\t'.format(-1.0))
            output_file.write('\n')

    output_file.close()
    return

def test1():
    """ Heterogeneous multi-core fibers: proposal and design principle, Koshiba, at all. 2009 """
    light = Light()
    light.SetLambda0(1.55)
    fiber = Fiber()
    fiber.SetDeltaNCore(0.0035)
    fiber.SetCoreRadius(4.5)
    fiber.SetDistanceToFiberCenter(0.5 * (5.0 * fiber.GetCoreRadius()))

    b = fiber.GetBeta(light)
    print('b = {}'.format(b))

    coeffs = getCouplingCoefficients(1e-6, fiber, light, 4)[0]

    # Сравнение с аналитической формулой для 2-ядерного волокна из Agraval "Applications of Nonlinear Fiber Optics". Eq.2.1.24
    coeffs2 = getCouplingCoefficients(1e-6, fiber, light, 4)[0]

    print('Lambda = {} mkm \n'.format(fiber.GetDistanceToFiberCenter() * 2.0))

    print('k1 = {} 1/cm'.format(coeffs[0][1] * 1e+4))
    print('L1 = {} cm'.format(0.5 * np.pi / (coeffs[0][1] * 1e+4)))
    print('k1 = {} 1/m'.format(coeffs[0][1] * 1e+6))
    print('L1 = {} m'.format(0.5 * np.pi / (coeffs[0][1] * 1e+6)))

    print('k2 = {} 1/cm'.format(coeffs2[0][1] * 1e+4))
    print('L2 = {} cm'.format(0.5 * np.pi / (coeffs2[0][1] * 1e+4)))
    print('k2 = {} 1/m'.format(coeffs2[0][1] * 1e+6))
    print('L2 = {} m'.format(0.5 * np.pi / (coeffs2[0][1] * 1e+6)))
    return

def getCouplingOfDistance(fiber, light, d, couplings):
    """ Для двухъядерного волокна строит зависимость коэффициента связи
    от расстояния между ядрами, выраженного в числе радиусов ядер """
    fiber = copy.deepcopy(fiber)
    fiber.SetCoreCount(2)
    fiber.SetRefractiveIndexesByLambda(light.GetLambda0())
    N = len(d)
    couplings.resize(N, refcheck=False)

    for i in range(N):
        fiber.SetDistanceToFiberCenter(0.5 * fiber.GetCoreRadius() * d[i])
        coeffs = getCouplingCoefficients(1e-6, fiber, light, 4)[0]
        couplings[i] = coeffs[0][1]
        print('i = {}'.format(i))
    return

def get_C_of_R_and_d(fiber, light, r1, r2, d1, d2, N):  # fiber передаётся по ссылке, но изменяется
    """ Зависимость коэффициента связи от радиуса ядер и расстояния между ними в радиусах """
    d = np.linspace(d1, d2, N + 1, dtype=float)
    r = np.linspace(r1, r2, N + 1, dtype=float)

    arraySize = (N + 1) * (N + 1)
    couplings = np.empty(arraySize, dtype=float)
    L = np.empty(arraySize, dtype=float)
    T = np.empty(arraySize, dtype=float)
    P = np.empty(arraySize, dtype=float)

    fiber.SetRefractiveIndexesByLambda(light.getLambda0())
    fiber.SetCoreCount(2)

    for i in range(N + 1):
        fiber.SetCoreRadius(r[i])
        gamma = fiber.GetGamma(1e-6, light)  # [1/(W*m)]
        b2 = fiber.GetBeta2(light)  # [(ps^2)/km]
        for j in range(N + 1):
            fiber.SetDistanceToFiberCenter(0.5 * (d[j] * r[i]))
            coeffs = getCouplingCoefficients(1e-6, fiber, light, 4)
            couplings[i * (N + 1) + j] = coeffs[0][1]  # [1/m]

            L[i * (N + 1) + j] = 1.0 / (couplings[i * (N + 1) + j] * 1e+6)  # [m]
            T[i * (N + 1) + j] = (-0.5 * b2 / (couplings[i * (N + 1) + j] * 1e+9)) ** 0.5  # [ps]
            P[i * (N + 1) + j] = 1e+6 * couplings[i * (N + 1) + j] / gamma  # [W]

            print('i = {} j = {} \n'.format(i, j))

    output_file = open('C_of_R_and_d.txt', 'wt')

    output_file.write('Variables=Radius[mkm],distance[Radius],C[1/m],log10(C),L[m],T[ps],P[W]\n')
    output_file.write('Zone i={} j={}\n'.format(N + 1, N + 1))

    for i in range(N + 1):
        for j in range(N + 1):
            output_file.write('{:.2f}\t {:.2f}\t {:.7e}\t {:.7f}\t {:.7f}\t {:.7f}\t {:.7f}\n'.
                              format(r[i], d[j],
                                     couplings[i * (N + 1) + j] * 1e+6,
                                     math.log10(couplings[i * (N + 1) + j]),
                                     L[i * (N + 1) + j],
                                     T[i * (N + 1) + j],
                                     P[i * (N + 1) + j]))

    output_file.close()
    return

def getLTPOfC(fiber, light, d1, d2, N):
    d = np.linspace(d1, d2, N + 1, dtype=float)

    gamma = fiber.GetGamma(1e-6, light)  # [1/(W*m)]
    couplings = np.array([], dtype=float)
    getCouplingOfDistance(fiber, light, d, couplings)  # [1/mkm]

    L = np.empty(N + 1, dtype=float)
    T = np.empty(N + 1, dtype=float)
    P = np.empty(N + 1, dtype=float)

    for i in range(N + 1):
        print('C = {:.5e}\n'.format(couplings[i] * 1e+6))
        L[i] = 1.0 / (couplings[i] * 1e+6)  # [m]

        b2 = fiber.GetBeta2(light)  # [ps^2/km]
        T[i] = math.sqrt(-0.5 * b2 / (couplings[i] * 1e+9))  # [ps]
        P[i] = 1e+6 * couplings[i] / gamma  # [W]

    output_file = open('L_T_P_Of_C.txt', 'wt')
    output_file.write('Variables=d,L[m],T[ps],P[W]\n')
    for i in range(N):
        output_file.write('{:.2f}\t {:.7f}\t {:.7f}\t {:.7f}\n'.format(d[i], L[i], T[i], P[i]))

    output_file.close()
    return

def drowBetaOfLambda(fiber, light, l1, l2, N):
    """ Функция анализирует константу распространения и выводит в файл её производные для различных длин волн
    """
    fiber = copy.deepcopy(fiber)
    light = copy.deepcopy(light)
    l = np.linspace(l1, l2, N + 1)

    beta = np.empty(N + 1, dtype=float)
    beta1 = np.empty(N + 1, dtype=float)
    beta2 = np.empty(N + 1, dtype=float)

    for i in range(N + 1):
        light.SetLambda0(l[i])
        fiber.SetRefractiveIndexesByLambda(l[i])
        beta[i] = fiber.GetBeta(light)
        beta1[i] = fiber.GetBeta1(light)
        beta2[i] = fiber.GetBeta2(light)

    output_file = open('beta_Of_lambda.txt', 'wt')
    output_file.write('Variables=lambda[mkm],beta[1/mkm],beta1[ns/m],beta2[ps^2/km]\n')
    for i in range(N + 1):
        output_file.write('%.17f\t %.17f\t %.17f\t %.17f\n' % (l[i], beta[i], beta1[i], beta2[i]))
    output_file.close()
    return

