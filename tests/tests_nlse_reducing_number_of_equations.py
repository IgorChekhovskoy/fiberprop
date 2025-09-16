# test_solver_collapse_hex7_equivalence.py
import numpy as np
import pytest

from fiberprop.solver import ComputationalParameters, EquationParameters, Solver
from fiberprop.fiber import CoreConfig
from fiberprop.pulses import gaussian_pulse


def _l2_rel(a: np.ndarray, b: np.ndarray, eps: float = 1e-15) -> float:
    """Relative L2 error."""
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(b) + eps
    return float(num / den)


def _energy_per_core(u_t: np.ndarray, dt: float) -> np.ndarray:
    """
    u_t: (cores, M) complex field at fixed z
    Returns: (cores,) energies ~ ∫ |u|^2 dt (общий множитель не важен — мы сравниваем одинаково посчитанные величины)
    """
    return np.sum(np.abs(u_t) ** 2, axis=1) * dt


@pytest.mark.parametrize("use_torch", [False, True])
def test_solver_collapse_hex7_equivalence(use_torch):
    """
    Проверка корректности «collapse» на 7-яд. гексагональной MCF (центр + 6 одинаковых внешних ядер).
    Алгоритм:
      1) считаем полную систему (7 уравнений),
      2) считаем схлопнутую систему (2 уравнения) и разворачиваем поле назад,
      3) сравниваем поля U(t) и энергии E на z=0 и z=L.
    См. физическую мотивацию случая в PRA-2016 (Chekhovskoy et al., 2016).
    """

    # Компактные параметры, чтобы тест проходил быстрее (можно увеличить при желании)
    com = ComputationalParameters(N=400, M=2 ** 12, L1=0.0, L2=1.78, T1=-30.0, T2=30.0)
    eq = EquationParameters(
        core_configuration=CoreConfig.hexagonal,
        ring_count=1,
        beta2=-2.0,
        gamma=1.0,
        E_sat=0.0,
        alpha=0.0,
        g_0=0.0,
    )

    pulse_params = {"p": 0.687, "tau": 1.775}

    # --- Полная система ---
    solver_full = Solver(
        com,
        eq,
        pulses=gaussian_pulse,
        pulse_params_list=pulse_params,
        use_gpu=False,          # в CI обычно без GPU
        use_torch=use_torch,
        display_debug_info=False,
    )
    solver_full.run_numerical_simulation()

    assert solver_full.eq.size == 7, "Ожидаем 7 ядер в полной системе (центр + 6)."

    # Сохранить поля на z=0 и z=L
    u0_full = solver_full.numerical_solution[0]                 # (7, M)
    uL_full = solver_full.numerical_solution[solver_full.com.N] # (7, M)

    # --- Схлопнутая система ---
    solver_col = Solver(
        com,
        eq,
        pulses=gaussian_pulse,
        pulse_params_list=pulse_params,
        use_gpu=False,
        use_torch=use_torch,
        display_debug_info=False,
    )

    # Для надёжности не требуем явной проверки нач. симметрии (она есть, но генерация psi скрыта внутри Solver)
    ok = solver_col.collapse_if_possible(require_initial_symmetry=False)
    assert ok, "Схлопывание должно быть возможно для 7-ядерной гекс. конфигурации при симметричных параметрах."
    assert solver_col.eq.size == 2, "Ожидаем 2 уравнения (центр и кольцо) после схлопывания."

    solver_col.run_numerical_simulation()

    # Разворачиваем схлопнутое поле назад к 7 ядрам на z=0 и z=L
    u0_col_expanded = solver_col.expand_field(solver_col.numerical_solution[0])                  # (7, M)
    uL_col_expanded = solver_col.expand_field(solver_col.numerical_solution[solver_col.com.N])   # (7, M)

    # --- Сравнение полей ---
    # Центр (индекс 0) и все внешние (1..6) должны совпасть с развёрнутым решением
    rtol_field = 1e-8
    atol_field = 1e-10
    assert np.allclose(u0_col_expanded, u0_full, rtol=rtol_field, atol=atol_field), \
        f"Несовпадение U(z=0): rel={_l2_rel(u0_col_expanded, u0_full):.3e}"
    assert np.allclose(uL_col_expanded, uL_full, rtol=rtol_field, atol=atol_field), \
        f"Несовпадение U(z=L): rel={_l2_rel(uL_col_expanded, uL_full):.3e}"

    # --- Сравнение энергий (подсчёт напрямую из |U|^2) ---
    dt = float(solver_full.t[1] - solver_full.t[0])
    E0_full = _energy_per_core(u0_full, dt)
    EL_full = _energy_per_core(uL_full, dt)

    E0_col = _energy_per_core(u0_col_expanded, dt)
    EL_col = _energy_per_core(uL_col_expanded, dt)

    rtol_E = 1e-8
    atol_E = 1e-10
    assert np.allclose(E0_col, E0_full, rtol=rtol_E, atol=atol_E), "Энергии на входе должны совпадать."
    assert np.allclose(EL_col, EL_full, rtol=rtol_E, atol=atol_E), "Энергии на выходе должны совпадать."

    # --- Восстановление исходной системы ---
    solver_col.restore_full_system()
    assert solver_col.eq.size == 7, "restore_full_system() должен вернуть исходную размерность 7."


if __name__ == "__main__":
    # локальный запуск:  python -m pytest -q test_solver_collapse_hex7_equivalence.py
    pytest.main([__file__, "-q"])
