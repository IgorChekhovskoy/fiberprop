from Fiber import FiberConfig
import scipy.integrate as si
import scipy.special as sp
import multiprocessing
import numpy as np
import math

# Корни функции Бесселя. Первый индекс - порядок функции Бесселя (J0, J1, J2 или J3).
# Второй - номер корня минус один.
besselRoots = np.array([[2.404825557695773, 5.520078110286311, 8.653727912911013, 11.791534439014281],
                        [3.831705970207512, 7.015586669815618, 10.173468135062723, 13.323691936314223],
                        [5.135622301840682, 8.417244140399866, 11.619841172149059, 14.795951782351260],
                        [6.380161895923983, 9.761023129981670, 13.015200721698433, 16.223466160318768]], dtype=float)

def SciPy_doubleIntegralByCircle(R, eps, fiber, light, coreCenterCoords, coreIndexes, Func):
    """ Интеграл библиатечным методом и Fortran по круглой области """
    return si.dblquad(lambda x, y: Func(fiber, light, coreCenterCoords, coreIndexes, x, y),
                      -R, R, lambda y: -math.sqrt(R**2 - y**2), lambda y: math.sqrt(R**2 - y**2), epsabs=eps)

def IntF2(fiber, light, coreCenterCoords, coreIndexes, x, y):
    """ Интеграл от квадрата моды """
    x0 = coreCenterCoords[coreIndexes[0]][0]
    y0 = coreCenterCoords[coreIndexes[0]][1]
    R = fiber.GetCladdingDiameter() * 0.5
    temp = 0.0
    if x**2 + y**2 <= R**2:
        temp = (getLPmode(0, 1, fiber, light, x - x0, y - y0))**2
    return temp


def IntF4(fiber, light, coreCenterCoords, coreIndexes, x, y):
    """ Интеграл от моды в четвёртой степени """
    x0 = coreCenterCoords[coreIndexes[0]][0]
    y0 = coreCenterCoords[coreIndexes[0]][1]
    R = fiber.GetCladdingDiameter() * 0.5
    temp = 0.0
    if x**2 + y**2 <= R**2:
        temp = (getLPmode(0, 1, fiber, light, x - x0, y - y0))**4
    return temp


def nMode(fiber, coreCenterCoords, coreIndexes, x, y):
    """ Коэффициент в интеграле, описывающем связь между двумя сердцевинами """
    cDiam = fiber.GetCladdingDiameter()
    if x**2 + y**2 < (cDiam**2):  # почему тут диаметр ? в данном случае не важно, всё равно 0 возвращает вне ядер
        R = fiber.GetCoreRadius()
        for i in range(len(coreCenterCoords)):
            x0 = coreCenterCoords[i][0]
            y0 = coreCenterCoords[i][1]
            if (x - x0)**2 + (y - y0)**2 < R**2 and i != coreIndexes[0]:
                return ((fiber.GetNCore())**2 - (fiber.GetNCladding())**2)
        return 0.0
    return 0.0


def Int(fiber, light, coreCenterCoords, coreIndexes, x, y):
    """ Интеграл, описывающий связь между двумя сердцевинами """
    x0 = coreCenterCoords[coreIndexes[0]][0]
    y0 = coreCenterCoords[coreIndexes[0]][1]
    x1 = coreCenterCoords[coreIndexes[1]][0]
    y1 = coreCenterCoords[coreIndexes[1]][1]

    R = fiber.GetCladdingDiameter() * 0.5
    temp = 0.0
    if x**2 + y**2 <= R**2:
        cc = nMode(fiber, coreCenterCoords, coreIndexes, x, y)
        temp += cc * getLPmode(0, 1, fiber, light, x - x0, y - y0) * getLPmode(0, 1, fiber, light, x - x1, y - y1)
    return temp


### Nonlinear Propagation in Multimode and Multicore Fibers: Generalization of the Manakov Equations. Eq.32
def getCouplingCoefficients(eps, fiber, light, procNum):
    """ Первый выход - матрица связей [1/cm], второй выход - матрица ожидаемых абсолютных ошибок [1/cm] """
    R = 0.5 * fiber.GetCladdingDiameter()
    coreCount = fiber.GetCoreCount()
    distanceToFiberCenter = fiber.GetDistanceToFiberCenter()
    coreCenterCoords = []

    ### подготовка данных о конфигурации волокна для передачи в подынтегральную функцию
    if fiber.GetCoreConfiguration == FiberConfig.Ring:
        for i in range(coreCount):
            phi = 2.0 * math.pi * i / coreCount
            coords = (distanceToFiberCenter * math.cos(phi), distanceToFiberCenter * math.sin(phi))
            coreCenterCoords.append(coords)
    elif fiber.GetCoreConfiguration() == FiberConfig.Hexagonal:
        deltaX = 0.0
        deltaY = 0.0
        coords = (deltaX, deltaY)
        coreCenterCoords.append(coords)
        for i in range(coreCount - 1):
            phi = 2.0 * math.pi * i / (coreCount - 1)
            Xcoord = distanceToFiberCenter * math.cos(phi) + deltaX
            Ycoord = distanceToFiberCenter * math.sin(phi) + deltaY
            coords = (Xcoord, Ycoord)
            coreCenterCoords.append(coords)
    elif fiber.GetCoreConfiguretion() == FiberConfig.Dual_core:
        coupMat = getCouplingCoeff2CoreFiber(fiber, light)
        return coupMat
    else:
        raise ValueError('this fiber configuration is not yet supported')

    ### создание массива аргументов для паралельных процессов
    work_args = []
    for m in range(1, coreCount):
        like_pair = (m, m)
        work_args.append((R, eps, fiber, light, coreCenterCoords, like_pair, IntF2))
        for p in range(m):
            like_pair = (m, p)
            work_args.append((R, eps, fiber, light, coreCenterCoords, like_pair, Int))

    ### параллельное выполнение интегрирования
    p = multiprocessing.Pool(procNum)
    poolResult = p.starmap(SciPy_doubleIntegralByCircle, work_args)
    p.close()
    p.join()

    ### обработка результатов интегрирования
    coupMat = np.zeros((coreCount, coreCount), dtype=float)
    errorMat = np.zeros((coreCount, coreCount), dtype=float)
    idx = 0
    for m in range(1, coreCount):
        Im = poolResult[idx][0]
        Im_upEst = poolResult[idx][0] + poolResult[idx][1]
        Im_lowEst = poolResult[idx][0] - poolResult[idx][1]
        idx += 1
        for p in range(m):
            Ip = Im
            Ip_upEst = Im_upEst
            Ip_lowEst = Im_lowEst
            k = 0.5 * (light.GetK0()**2) / fiber.GetBeta(light)  # [1/mkm]
            Int2 = poolResult[idx][0]  # [mkm^2]
            Int2_upEst = poolResult[idx][0] + poolResult[idx][1]
            Int2_lowEst = poolResult[idx][0] - poolResult[idx][1]
            idx += 1
            qmp = k * Int2  # [mkm]
            qmp /= (Im * Ip)**0.5  # [1/mkm]
            coupMat[m][p] = qmp * 1e+4  # [1/cm]
            coupMat[p][m] = qmp * 1e+4  # [1/cm]
            FullErr_upEst = k * Int2_upEst / (Ip_lowEst * Im_lowEst)**0.5
            FullErr_lowEst = k * Int2_lowEst / (Ip_upEst * Im_upEst)**0.5
            errorMat[m][p] = (FullErr_upEst - FullErr_lowEst)*1e+4 / 2  # [1/cm]
            errorMat[p][m] = (FullErr_upEst - FullErr_lowEst)*1e+4 / 2  # [1/cm]
    return coupMat, errorMat


### Agraval "Applications of Nonlinear Fiber Optics". Eq.2.1.24
def getCouplingCoeff2CoreFiber(fiber, light):
    """ Коэффициент связи для двухсердцевинного волокна """
    V = fiber.GetCoreRadius() * light.GetK0() * (
            ((1.0 + fiber.GetDeltaNCore()) * fiber.GetNCladding())**2 - (fiber.GetNCladding())**2)**0.5
    print('V = {}'.format(V))

    c0 = 5.2789 - 3.663 * V + 0.3841 * V**2
    c1 = -0.7769 + 1.2252 * V - 0.0152 * V**2
    c2 = -0.0175 - 0.0064 * V - 0.0009 * V**2

    d = 2.0 * fiber.GetDistanceToFiberCenter() / fiber.GetCoreRadius()
    print('d = {}'.format(d))

    coupMat = np.zeros((2,2), dtype=float)
    errorMat = np.zeros((2,2), dtype=float)
    coupMat[0][1] = math.pi * V * math.exp(-(c0 + c1*d + c2*d**2)) / \
                    (2.0 * light.GetK0() * fiber.GetNCladding() * (fiber.GetCoreRadius())**2)
    coupMat[1][0] = coupMat[0][1]
    return coupMat, errorMat


# "Weakly Guiding Fibers" (D. Gloge)
def getLPmode(l, m, fiber, light, x, y):
    r = (x**2 + y**2)**0.5
    phi = np.arctan2(y, x)

    v = fiber.GetCoreRadius() * light.GetK0() * fiber.GetNA()  # Eq.3.

    u = float()
    w = float()

    if l == 0 and m == 1:
        u = (1.0 + (2.0)**0.5) * v / (1.0 + (4.0 + v**4)**0.25)
        # Eq.18. Такая формула только для моды LP01 (HE11)
        w = (v**2 - u**2)**0.5
    else:
        if l >= 0 and m >= 1:
            uc = besselRoots[math.fabs(l - 1)][m - 1]  # Корень функции Бесселя
            if uc > v:
                print('Mode LP{}{} for Core Radius = {} mkm: '.format(l, m, fiber.GetCoreRadius()))
                print('ERROR: this mode is not allowed for this fiber geometry!')
                # содержательнее сообщения должны быть (perror)
                return 1

            s = (uc**2 - l**2 - 1)**0.5  # Eq.16
            u = uc * math.exp((math.asin(s / uc) - math.asin(s / v)) / s)  # Eq.17
            w = (v**2 - u**2)**0.5
        else:
            print('ERROR: Such LP mode does not exist!')  # содержательнее сообщения должны быть (perror)
            return 2

    coreRadius = fiber.GetCoreRadius()
    if r < coreRadius:
        return (sp.jv(l, u * r / coreRadius) / sp.jv(l, u) * math.cos(l * phi))
    else:
        return (sp.kn(l, w * r / coreRadius) / sp.kn(l, w) * math.cos(l * phi))

