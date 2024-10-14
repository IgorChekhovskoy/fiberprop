from fiberprop.solver import CoreConfig
import multiprocessing
from enum import Enum
import math
import copy
import numpy as np

from .light import Light


class FiberMaterial(Enum):
    not_set = 0
    SIO2 = 1
    GEO2 = 2
    SIO2_AND_GEO2_ALLOY = 3
    GREK_FOR_CORE = 4
    BK7 = 5


class Fiber:
    """ Параметры волокна """
    # Параметры конфигурации
    _core_configuration = CoreConfig.not_set
    _core_count = 0
    _cladding_diameter = 0.0
    _core_radius = 0.0
    _distance_to_fiber_center = 0.0

    # Параметры материала волокна
    _core_material = FiberMaterial.not_set
    _material_concentration = 0.0
    _n_cladding = 0.0
    _delta_n_core = 0.0
    _NA = 0.0
    _n_core = 0.0
    _n2 = 0.0

    @property
    def n2(self):
        if self._n2 == 0.0:
            raise RuntimeError('Nonlinear index n2 not set')
        return self._n2

    @property
    def cladding_diameter(self):
        if self._cladding_diameter == 0.0:
            raise RuntimeError('Cladding diameter not set')
        return self._cladding_diameter

    @property
    def core_radius(self):
        if self._core_radius == 0.0:
            raise RuntimeError('Core radius not set')
        return self._core_radius

    @property
    def n_cladding(self):
        if self._n_cladding == 0.0:
            raise RuntimeError('Cladding index nCladding not set')
        return self._n_cladding

    @property
    def n_core(self):
        if self._n_core == 0.0:
            raise RuntimeError('Core index nCore not set')
        return self._n_core

    @property
    def core_configuration(self):
        if self._core_configuration is CoreConfig.not_set:
            raise RuntimeError('Core configuration not set')
        return self._core_configuration

    @property
    def core_count(self):
        if self._core_count == 0:
            raise RuntimeError('Number of cores not set')
        return self._core_count

    @property
    def distance_to_fiber_center(self):
        if self._distance_to_fiber_center == 0.0:
            raise RuntimeError('Distance to fiber center not set')
        return self._distance_to_fiber_center

    @property
    def delta_n_core(self):
        if self._delta_n_core == 0.0:
            raise RuntimeError('Refractive index difference not set')
        return self._delta_n_core

    @property
    def NA(self):
        if self._NA == 0.0:
            raise RuntimeError('Numerical aperture NA not set')
        return self._NA

    @property
    def core_material(self):
        if self._core_material is FiberMaterial.not_set:
            raise RuntimeError('Core material not set')
        return self._core_material

    @property
    def material_concentration(self):
        if self._material_concentration == 0.0:
            raise RuntimeError('Core material concentration not set')
        return self._material_concentration

    def get_sellmeier_coefficients(self):
        """ Возвращает коэффициенты Зельмейера для различных материалов волокна """
        res = []
        if self.core_material is FiberMaterial.BK7:
            B = [1.03961212, 0.231792344, 1.01046945]
            C = [np.sqrt(6.00069867e-3), np.sqrt(2.00179144e-2), np.sqrt(1.03560653e+2)]
        elif self.core_material is FiberMaterial.SIO2:
            B = [0.69616630, 0.40794260, 0.89747940]
            C = [0.68404300e-1, 0.11624140, 0.98961610e+1]
        elif self.core_material is FiberMaterial.GEO2:
            B = [0.80686642, 0.71815848, 0.85416831]
            C = [0.68972606e-1, 0.15396605, 0.11841931e+2]
        elif self.core_material is FiberMaterial.SIO2_AND_GEO2_ALLOY:
            X = self.material_concentration
            B = [0.69616630 + X * (0.80686642 - 0.69616630),
                 0.40794260 + X * (0.71815848 - 0.40794260),
                 0.89747940 + X * (0.85416831 - 0.89747940)]
            C = [0.68404300e-1 + X * (0.68972606e-1 - 0.68404300e-1),
                 0.11624140 + X * (0.15396605 - 0.11624140),
                 0.11841931e+2 + X * (0.11841931e+2 - 0.11841931e+2)]
        elif self.core_material is FiberMaterial.GREK_FOR_CORE:
            B = [0.711040, 0.451885, 0.704048]
            C = [0.064270, 0.129408, 9.425478]
        else:
            raise ValueError('This fiber material is not yet supported')
        res.append(B)
        res.append(C)
        return res

    @core_material.setter
    def core_material(self, arg):
        if not isinstance(arg, FiberMaterial):
            raise TypeError('Core material should be of type FiberMaterial')
        self._core_material = arg

    @material_concentration.setter
    def material_concentration(self, arg):
        if not isinstance(arg, float):
            raise TypeError('Core material concentration should be float')
        self._material_concentration = arg

    @n2.setter
    def n2(self, arg):
        if not isinstance(arg, float):
            raise TypeError('n2 should be float')
        self._n2 = arg

    @cladding_diameter.setter
    def cladding_diameter(self, arg):
        if not isinstance(arg, float):
            raise TypeError('Cladding diameter should be float')
        self._cladding_diameter = arg

    @NA.setter
    def NA(self, arg):
        if not isinstance(arg, float):
            raise TypeError('NA should be float')
        self._NA = arg

    @n_cladding.setter
    def n_cladding(self, arg):
        if not isinstance(arg, float):
            raise TypeError('nCladding should be float')
        self._n_cladding = arg

    @n_core.setter
    def n_core(self, arg):
        if not isinstance(arg, float):
            raise TypeError('nCore should be float')
        self._n_core = arg

    @core_configuration.setter
    def core_configuration(self, arg):
        if not isinstance(arg, CoreConfig):
            raise TypeError('Core configuration should be of type FiberConfig')
        self._core_configuration = arg

    @core_count.setter
    def core_count(self, arg):
        if not isinstance(arg, int):
            raise TypeError('Core count should be integer')
        self._core_count = arg

    @distance_to_fiber_center.setter
    def distance_to_fiber_center(self, arg):
        if not isinstance(arg, float):
            raise TypeError('Distance to fiber center should be float')
        self._distance_to_fiber_center = arg

    @delta_n_core.setter
    def delta_n_core(self, arg):
        if not isinstance(arg, float):
            raise TypeError('Delta nCore should be float')
        self._delta_n_core = arg

    @core_radius.setter
    def core_radius(self, arg):
        if not isinstance(arg, float):
            raise TypeError('Core radius should be float')
        self._core_radius = arg

    def set_refractive_indexes_by_lambda(self, lambda0):
        if not isinstance(lambda0, float):
            raise TypeError('lambda0 should be float')

        sellmeier_coeffs = self.get_sellmeier_coefficients()
        B = sellmeier_coeffs[0]
        l = sellmeier_coeffs[1]
        self._n_core = math.sqrt(1.0 +
                                 B[0] * lambda0 ** 2 / (lambda0 ** 2 - l[0] ** 2) +
                                 B[1] * lambda0 ** 2 / (lambda0 ** 2 - l[1] ** 2) +
                                 B[2] * lambda0 ** 2 / (lambda0 ** 2 - l[2] ** 2))
        self._n_cladding = math.sqrt(self._n_core ** 2 - self.NA ** 2)

    def set_refractive_indexes_by_omega(self, omega0):
        if not isinstance(omega0, float):
            raise TypeError('omega0 should be float')
        light = Light()
        light.omega0 = omega0
        self.set_refractive_indexes_by_lambda(light.lambda0)

    def get_b(self, light):
        v = self.core_radius * light.k0 * self.NA
        u = (1.0 + (2.0 ** 0.5)) * v / (1.0 + (4.0 + v ** 4) ** 0.25)
        return 1.0 - (u / v) ** 2

    def get_beta(self, light):
        b = self.get_b(light)
        delta = (self.n_core ** 2 - self.n_cladding ** 2) / (2.0 * self.n_core ** 2)
        return self.n_cladding * light.k0 * (1.0 + 2.0 * b * delta) ** 0.5

    def get_beta1(self, light):
        light_copy = copy.deepcopy(light)
        fiber_copy = copy.deepcopy(self)
        lambda0 = light_copy.lambda0

        delta_lambda = 0.001

        v_center = fiber_copy.core_radius * fiber_copy.NA * light_copy.k0
        n_core_center = fiber_copy.n_core
        n_clad_center = fiber_copy.n_cladding
        b_center = fiber_copy.get_b(light_copy)

        delta = (n_core_center ** 2 - n_clad_center ** 2) / (2.0 * n_core_center ** 2)

        fiber_copy.set_refractive_indexes_by_lambda(lambda0 - delta_lambda)
        light_copy.lambda0 = lambda0 - delta_lambda
        v_left = fiber_copy.core_radius * fiber_copy.NA * light_copy.k0
        n_core_left = fiber_copy.n_core
        n_clad_left = fiber_copy.n_cladding
        b_left = fiber_copy.get_b(light_copy)

        fiber_copy.set_refractive_indexes_by_lambda(lambda0 + delta_lambda)
        light_copy.lambda0 = lambda0 + delta_lambda
        v_right = fiber_copy.core_radius * fiber_copy.NA * light_copy.k0
        n_core_right = fiber_copy.n_core
        n_clad_right = fiber_copy.n_cladding
        b_right = fiber_copy.get_b(light_copy)

        n_core_first_derivative = (n_core_right - n_core_left) / (2.0 * delta_lambda)
        n_clad_first_derivative = (n_clad_right - n_clad_left) / (2.0 * delta_lambda)

        NCore = n_core_center - lambda0 * n_core_first_derivative
        NClad = n_clad_center - lambda0 * n_clad_first_derivative

        b_first_derivative = (b_right - b_left) / (v_right - v_left)

        AV = 0.5 * (b_first_derivative * v_center + 2.0 * b_center)
        dbdwA = (NCore * AV + NClad * (1.0 - AV) + NClad * delta * (AV - b_center)) / light_copy.c_light
        return dbdwA * 1e+9

    def get_beta2(self, light):
        light_copy = copy.deepcopy(light)
        fiber_copy = copy.deepcopy(self)
        lambda0 = light_copy.lambda0
        delta_lambda = 0.001

        fiber_copy.set_refractive_indexes_by_lambda(lambda0 - delta_lambda)
        light_copy.lambda0 = lambda0 - delta_lambda
        beta1_left = fiber_copy.get_beta1(light_copy) * 1e-9

        fiber_copy.set_refractive_indexes_by_lambda(lambda0 + delta_lambda)
        light_copy.lambda0 = lambda0 + delta_lambda
        beta1_right = fiber_copy.get_beta1(light_copy) * 1e-9

        D = (beta1_right - beta1_left) / (2.0 * delta_lambda)
        beta2 = -lambda0 ** 2 * D / (2.0 * math.pi * light_copy.c_light) * 1e+21
        return beta2

    def get_gamma(self, light, eps=1e-3):
        from .base_functions import int_f2, int_f4, scipy_double_integral_by_circle
        n2 = self.n2
        lamb = light.lambda0

        R = self.cladding_diameter
        core_center_coords = [(0.0, 0.0)]
        like_pair = (0, 0)

        work_args = [(R, eps, self, light, core_center_coords, like_pair, int_f2),
                     (R, eps, self, light, core_center_coords, like_pair, int_f4)]
        with multiprocessing.Pool(2) as pool:
            pool_result = pool.starmap(scipy_double_integral_by_circle, work_args)

        IF2, IF4 = pool_result[0][0], pool_result[1][0]
        IF2_up_est, IF2_low_est = pool_result[0][0] + pool_result[0][1], pool_result[0][0] - pool_result[0][1]
        IF4_up_est, IF4_low_est = pool_result[1][0] + pool_result[1][1], pool_result[1][0] - pool_result[1][1]

        Aeff = IF2 ** 2 / IF4
        Aeff_up_est = IF2_up_est ** 2 / IF4_low_est
        Aeff_low_est = IF2_low_est ** 2 / IF4_up_est

        gamma = 2.0 * math.pi * n2 / (lamb * Aeff) * 1e-2
        gamma_up_est = 2.0 * math.pi * n2 / (lamb * Aeff_low_est) * 1e-2
        gamma_low_est = 2.0 * math.pi * n2 / (lamb * Aeff_up_est) * 1e-2
        gamma_error = (gamma_up_est - gamma_low_est) / 2
        return gamma, gamma_error
