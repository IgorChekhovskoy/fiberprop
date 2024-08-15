import numpy as np


class Light:
    """ Параметры излучения """
    _C_LIGHT = 299792458.0  # скорость света в вакууме [m/s]

    def __init__(self):
        self._lambda0 = 0.0  # центральная длина волны [мкм]
        self._k0 = 0.0       # волновое число [1/мкм]
        self._omega0 = 0.0   # центральная частота [ГГц]

    @property
    def c_light(self):
        return self._C_LIGHT

    @property
    def k0(self):
        if self._k0 == 0.0:
            raise RuntimeError('Wave number k0 not set')
        return self._k0

    @property
    def omega0(self):
        if self._omega0 == 0.0:
            raise RuntimeError('Wave frequency omega0 not set')
        return self._omega0

    @property
    def lambda0(self):
        if self._lambda0 == 0.0:
            raise RuntimeError('Wave length lambda0 not set')
        return self._lambda0

    @lambda0.setter
    def lambda0(self, value):
        if not isinstance(value, float):
            raise TypeError('lambda0 should be float')
        self._lambda0 = value
        self._k0 = 2 * np.pi / value
        self._omega0 = 2 * np.pi * self._C_LIGHT * 1e-3 / value

    @omega0.setter
    def omega0(self, value):
        if not isinstance(value, float):
            raise TypeError('omega0 should be float')
        self._omega0 = value
        self._lambda0 = 2 * np.pi * self._C_LIGHT * 1e-3 / value
        self._k0 = 2 * np.pi / self._lambda0