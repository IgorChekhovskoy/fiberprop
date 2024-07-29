from Light import Light
import multiprocessing
from enum import Enum
import math
import copy


class FiberConfig(Enum):
    Ring_with_central = 0
    Ring = 1
    Square = 2
    Hexagonal = 3
    Dual_core = 4
    Not_set = 5

class FiberMaterial(Enum):
    BK7 = 0  # не знаю, что это
    SiO2 = 1  # стекло из чистого оксида кремния
    GeO2 = 2  # стекло из чистого оксида германия
    SiO2andGeO2_alloy = 3  # стекло - сплав оксида кремния и оксида германия
    Grek_for_core = 4  # не знаю, что это
    Not_set = 5

class Fiber:
    """ Параметры волокна """
    ### Параметры конфигурации
    __coreConfiguration = FiberConfig.Not_set
    __coreCount = 0  ################# кол-во ядер
    __claddingDiameter = 0.0  ######## диаметр оболочки [mkm]
    __coreRadius = 0.0  ############## радиус ядра [mkm]
    __distanceToFiberCenter = 0.0  ### расстояние периферии до центра волокна [mkm]
    ### Параметры материала волокна
    __coreMaterial = FiberMaterial.Not_set
    __materialConcentration = 0.0  ### Молярная доля (не процент, а доля) примеси к кремнию. Если материал - сплав.
    __nCladding = 0.0  ############### Коэффициент преломления оболочки
    __deltaNCore = 0.0  ############## Разница в коэффициенте преломления между ядром и оболочкой (в долях)
    __NA = 0.0  ###################### Численная апертура
    __nCore = 0.0  ################### коэф-т преломления волокна в сердцевинах
    __n2 = 0.0  ###################### нелинейный коэффициент преломления материала [10^(-20) m^2/W]

    def GetN2(self):
        if self.__n2 == 0.0:
            raise RuntimeError('nonlinear index n2 not set')
        return self.__n2

    def GetCladdingDiameter(self):
        if self.__claddingDiameter == 0.0:
            raise RuntimeError('cladding diameter not set')
        return self.__claddingDiameter

    def GetCoreRadius(self):
        if self.__coreRadius == 0.0:
            raise RuntimeError('core radius not set')
        return self.__coreRadius

    def GetNCladding(self):
        if self.__nCladding == 0.0:
            raise RuntimeError('cladding index nCladding not set')
        return self.__nCladding

    def GetNCore(self):
        if self.__nCore == 0.0:
            raise RuntimeError('core index nCore not set')
        return self.__nCore

    def GetCoreConfiguration(self):
        if self.__coreConfiguration == FiberConfig.Not_set:
            raise RuntimeError('core configuration not set')
        return self.__coreConfiguration

    def GetCoreCount(self):
        if self.__coreCount == 0:
            raise RuntimeError('number of cores not set')
        return self.__coreCount

    def GetDistanceToFiberCenter(self):
        if self.__distanceToFiberCenter == 0.0:
            raise RuntimeError('distance to fiber center not set')
        return self.__distanceToFiberCenter

    def GetDeltaNCore(self):
        if self.__deltaNCore == 0.0:
            raise RuntimeError('refractive index difference not set')
        return self.__deltaNCore

    def GetNA(self):
        if self.__NA == 0.0:
            raise RuntimeError('numerical aperture NA not set')
        return self.__NA

    def GetCoreMaterial(self):
        if self.__coreMaterial == FiberMaterial.Not_set:
            raise RuntimeError('core material not set')
        return self.__coreMaterial

    def GetMaterialConcentration(self):
        if self.__materialConcentration == 0.0:
            raise RuntimeError('core material concentration not set')
        return self.__materialConcentration

    def GetSellmeierCoefficients(self):
        """ Функция возвращает коэффициенты Зельмейера для различных материалов волокна """
        res = []
        if self.GetCoreMaterial() == FiberMaterial.BK7:
            B = [1.03961212, 0.231792344, 1.01046945]  # [1]
            C = [math.sqrt(6.00069867e-3), math.sqrt(2.00179144e-2), math.sqrt(1.03560653e+2)]  # [mkm]
            res.append(B)
            res.append(C)
        elif self.GetCoreMaterial() == FiberMaterial.SiO2:
            B = [0.69616630, 0.40794260, 0.89747940]  # [1]
            C = [0.68404300e-1, 0.11624140, 0.98961610e+1]  # [mkm]
            res.append(B)
            res.append(C)
        elif self.GetCoreMaterial() == FiberMaterial.GeO2:
            B = [0.80686642, 0.71815848, 0.85416831]  # [1]
            C = [0.68972606e-1, 0.15396605, 0.11841931e+2]  # [mkm]
            res.append(B)
            res.append(C)
        elif self.GetCoreMaterial() == FiberMaterial.SiO2andGeO2_alloy:
            X = self.GetMaterialConcentration()  # молярная доля германия в сплаве
            ### можно найти в работе J.W. Fleming Applied Optics, V. 23, n. 24 p. 4486 (1984)
            B = [0.69616630 + X*(0.80686642-0.69616630),
                 0.40794260 + X*(0.71815848-0.40794260),
                 0.89747940 + X*(0.85416831-0.89747940)]  # [1]
            C = [0.68404300e-1 + X*(0.68972606e-1-0.68404300e-1),
                 0.11624140 + X*(0.15396605-0.11624140),
                 0.11841931e+2 + X*(0.11841931e+2-0.11841931e+2)]  # [mkm]
            res.append(B)
            res.append(C)
        elif self.GetCoreMaterial() == FiberMaterial.Grek_for_core:
            B = [0.711040, 0.451885, 0.704048]  # [1]
            C = [0.064270, 0.129408, 9.425478]  # [mkm]
            res.append(B)
            res.append(C)
        else:
            raise ValueError('this fiber material is not yet supported')
        return res

    def SetCoreMaterial(self, arg):
        """ Параметр материала сердцевин, FiberMaterial """
        if type(arg) != FiberMaterial:
            raise TypeError('core material should be enum \'FiberMaterial\'')
        self.__coreMaterial = arg
        return

    def SetMaterialConcentration(self, arg):
        """ Молярная доля (не процент, а доля) примеси к кремнию [1], float.
         Параметр следует указывать только, если материал - сплав. """
        if type(arg) != float:
            raise TypeError('core material concentration should be float')
        self.__materialConcentration = arg
        return

    def SetN2(self, arg):
        """ Нелинейный коэффициент преломления материала [10^(-20) m^2/W], float """
        if type(arg) != float:
            raise TypeError('n2 should be float')
        self.__n2 = arg
        return

    def SetCladdingDiameter(self, arg):
        """ Диаметр оболочки [mkm], float """
        if type(arg) != float:
            raise TypeError('claddingDiameter should be float')
        self.__claddingDiameter = arg
        return

    def SetNA(self, arg):
        """ Числовая апертура, float """
        if type(arg) != float:
            raise TypeError('NA should be float')
        self.__NA = arg
        return

    def SetNCladding(self, arg):
        """ Коэффициент преломления оболочки [1], float """
        if type(arg) != float:
            raise TypeError('nCladding should be float')
        self.__nCladding = arg
        return

    def SetNCore(self, arg):
        """ Коэффициент преломления волокна в сердцевинах [1], float """
        if type(arg) != float:
            raise TypeError('nCore should be float')
        self.__nCore = arg
        return

    def SetCoreConfiguration(self, arg):
        """ Параметр конфигурации волокна, FiberConfig """
        if type(arg) != FiberConfig:
            raise TypeError('coreConfiguration should be enum \'FiberConfig\'')
        self.__coreConfiguration = arg
        return

    def SetCoreCount(self, arg):
        """ Количество ядер [1], int """
        if type(arg) != int:
            raise TypeError('coreCount should be integer')
        self.__coreCount = arg
        return

    def SetDistanceToFiberCenter(self, arg):
        """ Расстояние периферии до центра волокна [mkm], float """
        if type(arg) != float:
            raise TypeError('distanceToFiberCenter should be float')
        self.__distanceToFiberCenter = arg
        return

    def SetDeltaNCore(self, arg):
        """ Разница в коэффициенте преломления между ядром и оболочкой (в долях), float """
        if type(arg) != float:
            raise TypeError('deltaNCore should be float')
        self.__deltaNCore = arg
        return

    def SetCoreRadius(self, arg):
        """ Радиус ядра [mkm], float """
        if type(arg) != float:
            raise TypeError('coreRadius should be float')
        self.__coreRadius = arg
        return

    def SetRefractiveIndexesByLambda(self, lambda0):
        """ Функция вычисляет коэффициент преломления сердцевины в зависимости от длины волны излучения,
        используя формулу Зельмейера по 3-м коэффициентам. Коффициент преломления оболочки рассчитывается
        из числовой апертуры. """
        if type(lambda0) != float:
            raise TypeError('lambda0 should be float')

        SellmeierCoefs = self.GetSellmeierCoefficients()
        B = SellmeierCoefs[0]
        l = SellmeierCoefs[1]
        self.__nCore = math.sqrt(1.0 +
                                 B[0] * lambda0**2 / (lambda0**2 - l[0]**2) +
                                 B[1] * lambda0**2 / (lambda0**2 - l[1]**2) +
                                 B[2] * lambda0**2 / (lambda0**2 - l[2]**2))
        NA = self.GetNA()
        self.__nCladding = math.sqrt(self.__nCore**2 - NA**2)
        return

    def SetRefractiveIndexesByOmega(self, omega0):
        if type(omega0) != float:
            raise TypeError('omega0 should be float')
        light = Light()
        light.SetOmega0(omega0)
        lambda0 = light.GetLambda0()
        self.SetRefractiveIndexesByLambda(lambda0)
        return

    def GetB(self, light):  # "Weakly Guiding Fibers" (D. Gloge)
        """ Коэффициент нормировки постоянной распростренения [1] """
        v = self.GetCoreRadius() * light.GetK0() * self.GetNA()  # Eq.3.  [1]
        u = (1.0 + (2.0**0.5)) * v / (1.0 + (4.0 + v**4)**0.25)  # Eq.18. Такая формула только для моды LP01 (HE11) [1]
        b = 1.0 - (u / v)**2  # Eq.20
        return b

    def GetBeta(self, light):  # "Weakly Guiding Fibers" (D. Gloge)
        """ Постоянная распространения \beta [1/mkm] """
        b = self.GetB(light)
        nCore = self.GetNCore()
        delta = (nCore**2 - (self.GetNCladding())**2) / (2.0 * nCore**2)  # [1]
        beta = self.GetNCladding() * light.GetK0() * (1.0 + 2.0 * b * delta)**0.5  # Грек  [1/mkm]
        return beta

    def GetBeta1(self, light):
        """ Обратная групповая скорость \\beta_1 = первая производная
        постоянной распространения \\beta по частоте в точке центральной частоты \\omega_0 """
        light = copy.deepcopy(light)
        fiber = copy.deepcopy(self)
        lambda0 = light.GetLambda0()  # [mkm]

        deltaLambda = 0.001

        VCenter = fiber.GetCoreRadius() * fiber.GetNA() * light.GetK0()
        nCoreCenter = fiber.GetNCore()
        nCladCenter = fiber.GetNCladding()
        bCenter = fiber.GetB(light)

        Delta = (nCoreCenter**2 - nCladCenter**2) / (2.0 * nCoreCenter**2)

        fiber.SetRefractiveIndexesByLambda(lambda0 - deltaLambda)
        light.SetLambda0(lambda0 - deltaLambda)
        VLeft = fiber.GetCoreRadius() * fiber.GetNA() * light.GetK0()
        nCoreLeft = fiber.GetNCore()
        nCladLeft = fiber.GetNCladding()
        bLeft = fiber.GetB(light)

        fiber.SetRefractiveIndexesByLambda(lambda0 + deltaLambda)
        light.SetLambda0(lambda0 + deltaLambda)
        VRight = fiber.GetCoreRadius() * fiber.GetNA() * light.GetK0()
        nCoreRight = fiber.GetNCore()
        nCladRight = fiber.GetNCladding()
        bRight = fiber.GetB(light)

        nCoreFirstDerivative = (nCoreRight - nCoreLeft) / (2.0 * deltaLambda)  # [1/mkm]
        nCladFirstDerivative = (nCladRight - nCladLeft) / (2.0 * deltaLambda)  # [1/mkm]

        # Групповой индекс сердцевины и оболочки
        NCore = nCoreCenter - lambda0 * nCoreFirstDerivative
        NClad = nCladCenter - lambda0 * nCladFirstDerivative

        bFirstDerivative = (bRight - bLeft) / (VRight - VLeft)  # [1]

        AV = 0.5 * (bFirstDerivative * VCenter + 2.0 * bCenter)  # [1]
        dbdwA = (NCore * AV + NClad * (1.0 - AV) + NClad * Delta * (AV - bCenter)) / light.GetC_light()  # [s/m]
        return dbdwA * 1e+9  # [mks/km]

    def GetBeta2(self, light):  # Nonlinear Fiber Optics, Fifth Edition 2013, Govind P. Agrawal
        """ Дисперсия групповой скорости \\beta_2 = вторая производная
            постоянной распространения \\beta по частоте в точке центральной частоты \\omega_0 """
        light = copy.deepcopy(light)
        fiber = copy.deepcopy(self)
        lambda0 = light.GetLambda0()  # [mkm]
        deltaLambda = 0.001

        fiber.SetRefractiveIndexesByLambda(lambda0 - deltaLambda)
        light.SetLambda0(lambda0 - deltaLambda)
        beta1Left = fiber.GetBeta1(light) * 1e-9  # [s/m]

        fiber.SetRefractiveIndexesByLambda(lambda0 + deltaLambda)
        light.SetLambda0(lambda0 + deltaLambda)
        beta1Right = fiber.GetBeta1(light) * 1e-9  # [s/m]

        D = ((beta1Right - beta1Left) / (2.0 * deltaLambda))  # [10^6 s/m^2]
        beta2 = -1 * lambda0**2 * D / (2.0 * math.pi * light.GetC_light()) * 1e+21  # [ps^2/km]
        return beta2

    def GetGamma(self, eps, light):  # Nonlinear Fiber Optics, Fifth Edition 2013, Govind P. Agrawal(2.3.29 - 2.3.30)
        """ Первый выход - коэффициент нелинейности Керра,
         второй выход - оценка абсолютной погрешности его вычисления """
        from Base_functions import IntF2, IntF4, SciPy_doubleIntegralByCircle
        n2 = self.GetN2()  # 10^{-20} [m^2/W]
        lamb = light.GetLambda0()  # [mkm]

        R = self.GetCladdingDiameter()
        coreCenterCoords = [(0.0, 0.0)]
        like_pair = (0, 0)

        # parallel
        work_args = [(R, eps, self, light, coreCenterCoords, like_pair, IntF2),
                     (R, eps, self, light, coreCenterCoords, like_pair, IntF4)]
        p = multiprocessing.Pool(2)  # количество процессов в pool, здесь надо только 2
        poolResult = p.starmap(SciPy_doubleIntegralByCircle, work_args)
        p.close()
        p.join()

        IF2 = poolResult[0][0]
        IF4 = poolResult[1][0]
        IF2_upEst = poolResult[0][0] + poolResult[0][1]
        IF2_lowEst = poolResult[0][0] - poolResult[0][1]
        IF4_upEst = poolResult[1][0] + poolResult[1][1]
        IF4_lowEst = poolResult[1][0] - poolResult[1][1]

        Aeff = IF2**2 / IF4  # [mkm^2]
        Aeff_upEst = IF2_upEst**2 / IF4_lowEst  # [mkm^2]
        Aeff_lowEst = IF2_lowEst**2 / IF4_upEst  # [mkm^2]

        gamma = 2.0 * math.pi * n2 / (lamb * Aeff) * 1e-2  # [1 / (W * m)]
        gamma_upEst = 2.0 * math.pi * n2 / (lamb * Aeff_lowEst) * 1e-2  # [1 / (W * m)]
        gamma_lowEst = 2.0 * math.pi * n2 / (lamb * Aeff_upEst) * 1e-2  # [1 / (W * m)]
        gammaError = (gamma_upEst - gamma_lowEst) / 2  # [1 / (W * m)]
        return gamma, gammaError

