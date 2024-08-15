# FiberProp

Проект FiberProp предназначен для моделирования и анализа многосердцевинных световодов (multicore fiber - MCF) и устройств на основе них.
В частности, проект посвящен моделированию волоконных лазеров с резонатором в виде отрезка MCF.

Моделирование осуществляется с помощью системы связанных нелинейных уравнений Шредингера (НУШ).

В проекте реализованы численные методы на основе метода расщепления по физическим процессам (split-step Fourier method - SSFM).

## Структура проекта

- **data/**: данные для проекта (экспериментальные, расчетные).
- **docs/**: документация проекта.
- **examples/**: примеры использования проекта.
- **notebooks/**: Jupyter Notebooks для демонстрации возможностей и тестирования кода.
- **fiberprop/**
  - **coupling_coefficient/**
    - `__init__.py`
    - `base_functions.py`: базовые функции, используемые в проекте.
    - `fiber.py`: описание класса Fiber.
    - `light.py`: описание класса Light.
  - `dimensionless.py`: функции для работы с безразмерными величинами.
  - `drawing.py`: функции для визуализации данных.
  - `io.py`: функции для ввода и вывода данных.
  - `matrices.py`: функции для работы с матрицами.
  - `propagation.py`: функции для моделирования распространения света в волокне.
  - `pulses.py`: функции для задания начальных данных.
  - `solver.py`: основные классы для моделирования.
  - `signal_characteristics.py`: функции для вычисления характеристик сигнала.
  - `spectrum_characteristics.py`: функции для анализа спектральных характеристик.
  - `ssfm_mcf.py`: реализация SSFM с помощью NumPy.
  - `ssfm_mcf_pytorch.py`: реализация SSFM с помощью PyTorch.
  - `stationary_solution_solver.py`: функции для нахождения стационарных решений уравнений.
  - `utils.py`: различные вспомогательные функции.
- **scripts/** скрипты для демонстрации возможностей и тестирования кода.
- **tests/**
  - **test_coupling_coefficient/**
    - `__init__.py`
    - `tests_coupling_coefficient.py`: тесты для модуля `coupling_coefficient`.
  - `__init__.py`
  - `tests.py`: основные тесты для всего проекта.
  - `tests_mcf_nn.py`: тесты для нейронных сетей на основе MCF.
  - `tests_mcf_compression.py`: тесты для моделирования сжатия и сложения оптических импульсов с помощью MCF.
  - `tests_stationary_solution_solver.py`: тесты для функций нахождения стационарных решений уравнений.
  - `tests_unit.py`: модульные тесты.

## Установка

1. Склонируйте репозиторий:
    ```sh
    git clone https://github.com/IgorChekhovskoy/fiberprop.git
    ```

2. Перейдите в каталог проекта:
    ```sh
    cd fiberprop
    ```

3. Создайте виртуальное окружение и активируйте его:
    ```sh
    python -m venv venv
    source venv/bin/activate  # Для Windows используйте `venv\Scripts\activate`
    ```

4. Установите зависимости:
    ```sh
    pip install -r requirements.txt
    ```

## Использование

Для запуска основного скрипта используйте:
```sh
python fiberprop/coupling_coefficient/main.py
```

Пример использования кода для выполнения численного моделирования:

```python
from fiberprop.solver import Solver, ComputationalParameters, EquationParameters
from fiberprop.pulses import gaussian_pulse

# Установите параметры вычислений и уравнений
computational_params = ComputationalParameters(N=1000, M=2 ** 13, L1=0, L2=1.78, T1=-30, T2=30)
equation_params = EquationParameters(core_configuration=3, size=7, ring_number=1, beta_2=-2.0, gamma=1.0, E_sat=0.1,
                                     alpha=0.1, g_0=1)

# Создайте объект Solver
solver = Solver(computational_params, equation_params, pulses=gaussian_pulse,
                pulse_params_list={"p": 0.687, "tau": 1.775}, use_gpu=True)

# Запустите численное моделирование
solver.run_test()
```

## Тестирование

Для запуска тестов используйте `pytest`.

1. Установите `pytest`, если он ещё не установлен:

```sh
pip install pytest
```
2. Перейдите в директорию проекта и выполните команду:
```sh
pytest
```

## Вклад в проект
Если вы хотите внести вклад в проект, пожалуйста, создайте форк репозитория, внесите необходимые изменения и отправьте
pull request. Все предложения и замечания приветствуются!

## Лицензия

Этот проект распространяется по лицензии MIT. См. файл [LICENSE](LICENSE) для подробной информации.

## Контакты

Если у вас есть вопросы или предложения по проекту, вы можете связаться с нами:

- Игорь Чеховской
  - Электронная почта: i.s.chekhovskoy@gmail.com
  - GitHub: [https://github.com/IgorChekhovskoy](https://github.com/IgorChekhovskoy)

- Георгий Патрин
  - Электронная почта: g.patrin@g.nsu.ru
  - GitHub: [https://github.com/GeorgePatrin](https://github.com/GeorgePatrin)
