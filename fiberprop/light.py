from dataclasses import dataclass, field
import numpy as np
from typing import Optional


from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class Light:
    _lambda0: Optional[float] = field(default=None, init=False)  # внутренняя переменная
    _omega0: Optional[float] = field(default=None, init=False)   # внутренняя переменная
    _k0: Optional[float] = field(init=False, default=None)       # [1/mkm]
    c_light: float = field(default=299792458.0, init=False)      # [m/s], константа

    # Конструктор с явными параметрами
    def __init__(self, lambda0: Optional[float] = None, omega0: Optional[float] = None):
        if lambda0 is not None:
            self.lambda0 = lambda0  # setter вызовется → всё пересчитается
        elif omega0 is not None:
            self.omega0 = omega0    # setter вызовется → всё пересчитается
        else:
            raise ValueError("You must specify either lambda0 or omega0")

    # Property for lambda0
    @property
    def lambda0(self):
        return self._lambda0

    @lambda0.setter
    def lambda0(self, value):
        self._lambda0 = value
        self._update_from_lambda()

    # Property for omega0
    @property
    def omega0(self):
        return self._omega0

    @omega0.setter
    def omega0(self, value):
        self._omega0 = value
        self._update_from_omega()

    # Property for k0 (read-only)
    @property
    def k0(self):
        return self._k0

    def _update_from_lambda(self):
        self._k0 = 2 * np.pi / self._lambda0
        self._omega0 = 2 * np.pi * self.c_light * 1e-3 / self._lambda0  # ГГц

    def _update_from_omega(self):
        self._lambda0 = 2 * np.pi * self.c_light * 1e-3 / self._omega0
        self._k0 = 2 * np.pi / self._lambda0


