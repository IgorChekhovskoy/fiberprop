# FiberProp

Проект FiberProp предназначен для моделирования и анализа многосердцевинных световодов (multicore fiber — MCF) и устройств на их основе.
В частности, проект посвящён моделированию волоконных лазеров с резонатором в виде отрезка MCF.

Моделирование осуществляется с помощью системы связанных нелинейных уравнений Шрёдингера (НУШ).
В проекте реализованы численные методы на основе метода расщепления по физическим процессам (split-step Fourier method — SSFM).

## Структура проекта

```
fiberprop/
  coupling_coefficient/   — расчёт коэффициентов связи
  ssfm_mcf.py             — SSFM на NumPy/Numba
  ssfm_mcf_pytorch.py     — SSFM на PyTorch (GPU)
  ssfm_julia.py           — SSFM на Julia (опционально)
  solver.py               — классы Solver, ComputationalParameters, EquationParameters
  pulses.py               — начальные условия
  drawing.py              — визуализация
  ...
julia/
  FiberpropSSFM.jl        — Julia-модуль с FFTW-планами и NL/L-шагами
scripts/
  benchmark_julia_ssfm.py — сравнение скорости Python vs Julia
tests/                    — pytest-тесты
data/                     — результаты расчётов и бенчмарков
```

## Установка

Проект использует [uv](https://github.com/astral-sh/uv) для управления зависимостями.

```sh
git clone https://github.com/IgorChekhovskoy/fiberprop.git
cd fiberprop
uv sync
```

Все зависимости (включая Julia-бэкенд) устанавливаются автоматически.
Julia 1.9+ и нужные Julia-пакеты (`FFTW.jl`, `PythonCall.jl`) загружаются при первом запуске через `juliapkg`.

## Использование

```python
from fiberprop.solver import Solver, ComputationalParameters as CP, EquationParameters as EP, CoreConfig
import numpy as np

com = CP(N=1000, M=1024, L1=0.0, L2=1.0, T1=-30.0, T2=30.0)
eq = EP(core_configuration=CoreConfig.hexagonal, size=7, beta2=-1.0, gamma=1.0)

t = np.linspace(-30.0, 30.0, com.M)
psi0 = np.tile((1.0 / np.cosh(t)).astype(np.complex128), (eq.size, 1))

solver = Solver(com=com, eq=eq, initial_condition=psi0)
solver.run_numerical_simulation()
```

## Julia Backend

FiberProp включает опциональный Julia-бэкенд (`ssfm_order2_ndn_julia`) на основе FFTW.jl.

### Использование

Достаточно изменить параметр `method`:

```python
com = CP(N=1000, M=1024, L1=0.0, L2=1.0, T1=-30.0, T2=30.0,
         method="ssfm_order2_ndn_julia")
```

### Параллелизм

```bash
export JULIA_NUM_THREADS=4
uv run python your_script.py
```

Подробнее: [`specs/001-julia-ssfm-solver/quickstart.md`](specs/001-julia-ssfm-solver/quickstart.md)

## Тестирование

```sh
uv run pytest
```

Для запуска только тестов Julia-бэкенда:

```sh
uv run pytest tests/tests_julia_ssfm.py -v
```

## Вклад в проект

Создайте форк репозитория, внесите изменения и отправьте pull request. Все предложения приветствуются!

## Лицензия

Проект распространяется по лицензии MIT. См. файл [LICENSE](LICENSE).

## Контакты

- Игорь Чеховской — i.s.chekhovskoy@gmail.com · [GitHub](https://github.com/IgorChekhovskoy)
- Георгий Патрин — g.patrin@g.nsu.ru · [GitHub](https://github.com/GeorgePatrin)
