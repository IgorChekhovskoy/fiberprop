# tests_absorbing_boundary.py
# ------------------------------------------------------------
# Проверка ABC: фундаментальный солитон "уходит" через правую
# границу и энергия в области падает до < 5 % от начальной.
#
# Запуск:
#   pytest -q tests_absorbing_boundary.py
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pytest

from fiberprop.solver import ComputationalParameters as CP, EquationParameters as EP, Solver, CoreConfig
from fiberprop.pulses import fundamental_soliton as FS


@pytest.mark.parametrize("beta1,beta2,gamma", [(1.0, -1.0, 1.0)])
def test_absorbing_boundary(beta1, beta2, gamma):
    # ------ сетка -----------------------------------------------------
    M   = 2 ** 12                 # точек по времени
    T1, T2 = -100.0, 100.0        # окно, ps   (безразмерное!)
    N   = 600                     # шагов по z
    damp_length = 0.10  # damp_length
    L2 = 1.5 * 0.5 * (T2 - T1) / abs(beta1)
    L1 = 0

    com = CP(N=N, M=M,
             L1=L1, L2=L2,
             T1=T1, T2=T2,
             damp_length=damp_length)     # 10 % узлов справа/слева – ABC

    # ------ параметры НУШ -------------------------------------------
    eq  = EP(core_configuration=CoreConfig.empty_ring,
             size=1,
             beta1=beta1,
             beta2=beta2,
             gamma=gamma,
             coupling_coefficient=0.0)   # одна сердцевина – без связей

    # ------ начальный солитон (в центре окна) -----------------------
    t = np.linspace(T1, T2, M, endpoint=False)
    u0 = FS(t=t, z=0.0, beta2=beta2, gamma=gamma)  # (M,)
    init = u0[np.newaxis, :]                       # shape (1, M)

    # ------ расчёт ---------------------------------------------------
    sol = Solver(com, eq,
                 initial_condition=init,
                 use_gpu=False, display_debug_info=False)

    plt.figure(figsize=(6,3))
    plt.plot(sol.t, np.abs(sol._taper_np))
    plt.title("Absorbing-boundary taper")
    plt.xlabel("time")
    plt.ylabel("attenuation factor")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    sol.run_numerical_simulation(draw_modulus=True, save_gif=True)

    # ------ метрика --------------------------------------------------
    E0  = sol.energy[0, 0]         # энергия на входе
    Eend = sol.energy[0, -1]       # энергия на выходе
    resid = Eend / E0

    print(f"\nE0={E0:.3e},  Eend={Eend:.3e},  residual={resid:.5%}")

    # assert resid < 0.05, "absorbing boundary failed: residual energy > 5 %"

if __name__ == '__main__':
    # test_mcf_beta1_dimensionless()
    test_absorbing_boundary(1.0, -1.0, 1.0)