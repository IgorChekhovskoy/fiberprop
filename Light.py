import numpy as np


class Light:
    """ Параметры излучения """
    __C_light = 299792458.0  ### скорость света в вакууме [m/s]
    __lambda0 = 0.0  ########### центральная длина волны [mkm]
    __k0 = 0.0  ################ волновое число [1/mkm]
    __omega0 = 0.0  ############ центральная частота [GHz]

    def SetLambda0(self, lambda0):
        """ Функция устанавливает параметры излучения по центральной длине волны [mkm], float """
        if type(lambda0) != float:
            raise TypeError('lambda0 should be float')
        self.__lambda0 = lambda0
        self.__k0 = 2 * np.pi / lambda0
        self.__omega0 = 2 * np.pi * self.__C_light * 1e-3 / lambda0
        return

    def SetOmega0(self, omega0):
        """ Функция устанавливает параметры излучения по центральной частоте [GHz], float """
        if type(omega0) != float:
            raise TypeError('omega0 should be float')
        self.__lambda0 = 2 * np.pi * self.__C_light * 1e-3 / omega0
        self.__k0 = 2 * np.pi / self.__lambda0
        self.__omega0 = omega0
        return

    def GetC_light(self):
        return self.__C_light

    def GetK0(self):
        if self.__k0 == 0.0:
            raise RuntimeError('wave number k0 not set')
        return self.__k0

    def GetOmega0(self):
        if self.__omega0 == 0.0:
            raise RuntimeError('wave frequency omega0 not set')
        return self.__omega0

    def GetLambda0(self):
        if self.__lambda0 == 0.0:
            raise RuntimeError('wave length lambda0 not set')
        return self.__lambda0
