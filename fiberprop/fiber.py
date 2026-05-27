from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import math
import copy
import multiprocessing
from typing import List, Union

from .light import Light
from .fiber_geometry import CoreConfig, Mask, make_eq_mask, get_ring_count, get_core_count
from .fiber_base_functions import int_f2, int_f4, scipy_double_integral_by_circle, get_lp_mode_radial_integral


class FiberMaterial(Enum):
    not_set = 0
    SIO2 = 1
    GEO2 = 2
    SIO2_AND_GEO2_ALLOY = 3
    BK7 = 5 # Borosilicate crown glass


@dataclass
class Fiber:
    """
    Класс Fiber.

    Хранит параметры многосердцевинного волокна (MCF) и позволяет вычислять:

    - коэффициенты преломления сердцевин и оболочки;
    - параметры дисперсии (beta, beta1, beta2);
    - коэффициент нелинейности (gamma);
    - эффективную площадь моды (Aeff);
    - маску расположения сердцевин (mask_array).

    Параметры конструктора:

    - core_configuration (CoreConfig): Конфигурация сердцевин (ring, hexagonal и т.п.)
    - core_count (int): Число сердцевин.
    - ring_count (float): Число колец (определяется автоматически при core_count).
    - cladding_diameter (float): Диаметр оболочки [mkm].
    - core_radius (float): Радиус сердцевины [mkm].
    - distance_to_fiber_center (float | list | np.ndarray): Расстояние(я) до центра волокна.
    - core_material (FiberMaterial): Материал сердцевин.
    - material_concentration (float): Концентрация материала (для сплава SiO2-GeO2) (Dispersion in GeO2 -SiO 2 glasses, James W.Fleming, Eq.(2))
    - n_cladding (float): Показатель преломления оболочки.
    - delta_n_core (float): Разница показателей преломления сердцевина - оболочка.
    - NA (float): Числовая апертура.
    - n_core (float): Показатель преломления сердцевины.
    - n2 (float): Коэффициент нелинейности [1e-20 * m²/W].
    - mask_array (List[Mask]): Массив масок положения сердцевин (создаётся автоматически).
    """
    core_configuration: CoreConfig
    core_count: int = 0
    ring_count: float = 0.0
    cladding_diameter: float = 0.0
    core_radius: float = 0.0
    distance_to_fiber_center: Union[float, List[float], np.ndarray] = 0.0

    core_material: FiberMaterial = FiberMaterial.not_set
    material_concentration: float = 0.0
    n_cladding: float = 0.0
    delta_n_core: float = 0.0
    NA: float = 0.0
    n_core: float = 0.0
    n2: float = 0.0

    mask_array: List[Mask] = field(init=False)

    def __post_init__(self):
        if self.ring_count < 0:
            raise ValueError("ring_count must be >= 0")

        rc = int(round(self.ring_count))
        cc = int(self.core_count)

        # 2) нормализация случая ring_count == 0
        if rc == 0:
            # «ноль колец» означает минимум одно центральное ядро
            # если пользователь явно не просил больше, фиксируем 1
            if cc <= 1:
                self.ring_count = 0.0
                self.core_count = 1
            else:
                # пользователь хочет >1 ядра — вычисляем, сколько колец нужно
                self.ring_count = get_ring_count(self.core_configuration, cc)
                self.core_count = cc
        else:
            # 3) задано количество колец → восстанавливаем число ядер
            self.ring_count = float(rc)
            self.core_count = get_core_count(self.core_configuration, self.ring_count)

        self.mask_array = make_eq_mask(
            core_configuration=self.core_configuration,
            size=self.core_count,
            ring_count=self.ring_count,
            display_debug_info=False
        )

        if isinstance(self.distance_to_fiber_center, (int, float)):
            self.distance_to_fiber_center = [self.distance_to_fiber_center]
        if isinstance(self.distance_to_fiber_center, (list, np.ndarray)) and len(self.distance_to_fiber_center) == 1 and self.ring_count > 0:
            self.distance_to_fiber_center *= (int(self.ring_count) + 1)


    def get_sellmeier_coefficients(self):
        """
        Возвращает коэффициенты Зельмейера (B, C) для выбранного материала сердцевин.

        Выход:

        - List[List[float]] — список двух списков:
          - B — коэффициенты числителя;
          - C — корни знаменателя (в mkm).

        Используется в функции `set_refractive_indexes_by_lambda`.
        """
        if self.core_material == FiberMaterial.BK7:
            B = np.array([1.03961212, 0.231792344, 1.01046945])
            C = np.array([6.00069867e-3, 2.00179144e-2, 1.03560653e+2])
        elif self.core_material == FiberMaterial.SIO2:
            B = np.array([0.69616630, 0.40794260, 0.89747940])
            C = np.array([0.68404300e-1, 0.11624140, 0.98961610e+1])**2
        elif self.core_material == FiberMaterial.GEO2:
            B = np.array([0.80686642, 0.71815848, 0.85416831])
            C = np.array([0.68972606e-1, 0.15396605, 0.11841931e+2])**2
        elif self.core_material == FiberMaterial.SIO2_AND_GEO2_ALLOY:
            X = self.material_concentration

            B_SIO2 = np.array([0.69616630, 0.40794260, 0.89747940]) # [1]
            B_GEO2 = np.array([0.80686642, 0.71815848, 0.85416831]) # [1]

            C_SIO2 = np.array([0.0684043, 0.1162414, 9.896161])**2     # [mkm^2]
            C_GEO2 = np.array([0.068972606, 0.15396605, 11.841931])**2 # [mkm^2]

            B = B_SIO2 + X * (B_GEO2 - B_SIO2)
            C = C_SIO2 + X * (C_GEO2 - C_SIO2)
        else:
            raise ValueError('Unsupported fiber material')
        return B, C

    def set_refractive_indexes_by_lambda(self, lambda0: float):
        """
        Устанавливает n_core и n_cladding для данной длины волны.

        Параметры:
        - lambda0 (float): Центральная длина волны [mkm].
        """

        B, C = self.get_sellmeier_coefficients()
        self.n_core = math.sqrt(1.0 + sum(B[i] * lambda0**2 / (lambda0**2 - C[i]) for i in range(3)))
        self.n_cladding = math.sqrt(self.n_core**2 - self.NA**2)

    def set_refractive_indexes_by_omega(self, omega0: float):
        """
        Устанавливает n_core и n_cladding по частоте omega0.

        Параметры:
        - omega0 (float): Центральная частота [GHz].
        """
        light = Light()
        light.omega0 = omega0
        self.set_refractive_indexes_by_lambda(light.lambda0)

    def get_b(self, light: Light):
        """
        Возвращает нормированный параметр моды b.

        Параметры:

        - light (Light): Параметры излучения.

        Выход:

        - b (float) — безразмерный параметр b.

        Определяет распределение моды в волокне, зависит от V-числа.
        """
        v = self.core_radius * light.k0 * self.NA
        u = (1.0 + math.sqrt(2.0)) * v / (1.0 + (4.0 + v**4)**0.25)
        return 1.0 - (u / v) ** 2

    def get_beta(self, light: Light):
        """
        Возвращает фазовую постоянную beta.

        Параметры:

        - light (Light): Параметры излучения.

        Выход:

        - beta (float) — [1/mkm].

        Можно умножить на 1e6 для перевода в [1/m].
        """
        b = self.get_b(light)
        delta = (self.n_core**2 - self.n_cladding**2) / (2.0 * self.n_core**2)
        return self.n_cladding * light.k0 * math.sqrt(1.0 + 2.0 * b * delta)

    def get_beta1(self, light: Light) -> float:
        """
        Возвращает коэффициент групповой задержки beta1.

        Параметры:
        - light (Light): Параметры излучения.

        Выход:
        - beta1 (float) — [ps/m].
        """

        # Копируем объекты, чтобы не портить исходные
        lc = copy.deepcopy(light)
        fc = copy.deepcopy(self)

        lambda0 = lc.lambda0
        delta_lambda = 0.001

        # Центр
        V_center = fc.core_radius * fc.NA * lc.k0
        n_core_center = fc.n_core
        n_clad_center = fc.n_cladding
        b_center = fc.get_b(lc)

        Delta = (n_core_center ** 2 - n_clad_center ** 2) / (2.0 * n_core_center ** 2)

        # Left
        fc.set_refractive_indexes_by_lambda(lambda0 - delta_lambda)
        lc.lambda0 = lambda0 - delta_lambda
        V_left = fc.core_radius * fc.NA * lc.k0
        n_core_left = fc.n_core
        n_clad_left = fc.n_cladding
        b_left = fc.get_b(lc)

        # Right
        fc.set_refractive_indexes_by_lambda(lambda0 + delta_lambda)
        lc.lambda0 = lambda0 + delta_lambda
        V_right = fc.core_radius * fc.NA * lc.k0
        n_core_right = fc.n_core
        n_clad_right = fc.n_cladding
        b_right = fc.get_b(lc)

        # Производные
        n_core_first_derivative = (n_core_right - n_core_left) / (2.0 * delta_lambda)  # [1/mkm]
        n_clad_first_derivative = (n_clad_right - n_clad_left) / (2.0 * delta_lambda)  # [1/mkm]

        # Group indices
        N_core = n_core_center - lambda0 * n_core_first_derivative
        N_clad = n_clad_center - lambda0 * n_clad_first_derivative

        # b' по V
        b_first_derivative = (b_right - b_left) / (V_right - V_left)

        # AV
        AV = 0.5 * (b_first_derivative * V_center + 2.0 * b_center)

        # db/dωA
        dbdwA = (N_core * AV + N_clad * (1.0 - AV) + N_clad * Delta * (AV - b_center)) / light.c_light  # [s/m]

        return dbdwA * 1e12  # [ps/m]

    def get_beta2(self, light: Light):
        """
       Возвращает коэффициент дисперсии групповой скорости beta2.

       Параметры:

       - light (Light): Параметры излучения.

       Выход:

       - beta2 (float) — [ps²/m].

       Определяется вторым численным дифференцированием beta1 по omega.
       """
        lc, fc = copy.deepcopy(light), copy.deepcopy(self)
        l0 = lc.lambda0
        dl = 0.001

        fc.set_refractive_indexes_by_lambda(l0 - dl)
        lc.lambda0 = l0 - dl
        b1_l = fc.get_beta1(lc) * 1e-12

        fc.set_refractive_indexes_by_lambda(l0 + dl)
        lc.lambda0 = l0 + dl
        b1_r = fc.get_beta1(lc) * 1e-12

        D = (b1_r - b1_l) / (2.0 * dl)
        beta2 = -l0 ** 2 * D / (2.0 * math.pi * lc.c_light) * 1e18
        return beta2

    def get_gamma(self, light: Light, eps=1e-3):
        """
        Возвращает коэффициент нелинейности Kerr gamma для LP01 моды.

        Параметры:
        - light (Light): Параметры излучения.
        - eps (float)  : Относительная точность trapz-интеграла (1/samples).

        Выход:
        - gamma       (float) — [1/(W·m)].
        - gamma_error (float) — грубая оценка ошибки [1/(W·m)].

        Теперь вместо двух медленных dblquad-интегралов
        используется радиальная формула  2π∫|LP01(r)|ⁿ r dr,
        т.е. всего ДВА одномерных trapz, ⇒ ускорение ×100-×1000.
        """
        samples = int(1 / eps) + 1  # точность ≈ eps
        IF2 = get_lp_mode_radial_integral(2, self, light, samples)  # ∬|u|²
        IF4 = get_lp_mode_radial_integral(4, self, light, samples)  # ∬|u|⁴

        Aeff = IF2 ** 2 / IF4  # (⟨|u|²⟩)² / ⟨|u|⁴⟩
        gamma = 2.0 * math.pi * self.n2 / (light.lambda0 * Aeff) * 1e-2

        return gamma
