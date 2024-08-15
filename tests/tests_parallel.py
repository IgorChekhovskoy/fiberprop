import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from joblib import Parallel, delayed
from numba import njit, prange
import dask.array as da
import torch
import torch.multiprocessing as mp
import os

# Функция для вычисления суммы квадратов чисел в диапазоне
def sum_of_squares(n):
    return sum(i * i for i in range(n))

# Тестовая функция для модуля multiprocessing
def test_multiprocessing(n, num_workers):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(sum_of_squares, n) for _ in range(num_workers)]
        results = [f.result() for f in futures]

# Тестовая функция для модуля concurrent.futures.ThreadPoolExecutor
def test_threading(n, num_workers):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(sum_of_squares, n) for _ in range(num_workers)]
        results = [f.result() for f in futures]

# Тестовая функция для модуля joblib
def test_joblib(n, num_workers):
    results = Parallel(n_jobs=num_workers)(delayed(sum_of_squares)(n) for _ in range(num_workers))

# Альтернативная функция для numba
@njit(parallel=True)
def sum_of_squares_parallel(n, num_workers):
    results = np.zeros(num_workers)
    for w in prange(num_workers):
        for i in range(n):
            results[w] += i * i
    return results

def test_numba_parallel(n, num_workers):
    sum_of_squares_parallel(n, num_workers)

# Тестовая функция для dask
def test_dask(n, num_workers):
    x = da.arange(n, chunks=n//num_workers)
    y = x ** 2
    result = y.sum().compute()

# Функция для PyTorch worker
def pytorch_worker_fn(worker_id, n, results, device):
    result = torch.sum(torch.arange(n, device=device) ** 2)
    results[worker_id] = result.item()

# Тестовая функция для PyTorch
def test_pytorch_multiprocessing(n, num_workers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp.set_start_method('spawn', force=True)  # Установить метод запуска процессов
    results = mp.Manager().list([0] * num_workers)  # Общий список для результатов
    processes = []

    for worker_id in range(num_workers):
        p = mp.Process(target=pytorch_worker_fn, args=(worker_id, n, results, device))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return results

# Функция для измерения времени выполнения
def measure_time(func, n, num_workers, label):
    times = []
    for _ in range(num_tests):
        start_time = time.time()
        func(n, num_workers)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        print(f"{label} with {num_workers} workers took {elapsed_time:.6f} seconds")
    return np.median(times)

# Основная функция тестирования
def run_parallel_tests():
    # Параметры теста
    n = 10**6
    global num_tests  # Добавляем глобальную переменную
    num_tests = 5
    num_workers_list = [1, 2, 4, 8]  # Ограничиваем до 8 потоков
    multiprocessing_times = []
    threading_times = []
    joblib_times = []
    numba_parallel_times = []
    dask_times = []
    pytorch_multiprocessing_times = []

    # Проведение тестов
    for num_workers in num_workers_list:
        print(f'Testing with {num_workers} workers...')
        multiprocessing_time = measure_time(test_multiprocessing, n, num_workers, "Multiprocessing")
        threading_time = measure_time(test_threading, n, num_workers, "Threading")
        joblib_time = measure_time(test_joblib, n, num_workers, "Joblib")
        numba_parallel_time = measure_time(test_numba_parallel, n, num_workers, "Numba Parallel")
        dask_time = measure_time(test_dask, n, num_workers, "Dask")
        pytorch_multiprocessing_time = measure_time(test_pytorch_multiprocessing, n, num_workers, "PyTorch Multiprocessing")

        multiprocessing_times.append(multiprocessing_time)
        threading_times.append(threading_time)
        joblib_times.append(joblib_time)
        numba_parallel_times.append(numba_parallel_time)
        dask_times.append(dask_time)
        pytorch_multiprocessing_times.append(pytorch_multiprocessing_time)

    # Вычисление ускорения для каждого метода
    multiprocessing_base_time = multiprocessing_times[0]
    threading_base_time = threading_times[0]
    joblib_base_time = joblib_times[0]
    numba_parallel_base_time = numba_parallel_times[0]
    dask_base_time = dask_times[0]
    pytorch_multiprocessing_base_time = pytorch_multiprocessing_times[0]

    multiprocessing_speedup = [multiprocessing_base_time / t for t in multiprocessing_times]
    threading_speedup = [threading_base_time / t for t in threading_times]
    joblib_speedup = [joblib_base_time / t for t in joblib_times]
    numba_parallel_speedup = [numba_parallel_base_time / t for t in numba_parallel_times]
    dask_speedup = [dask_base_time / t for t in dask_times]
    pytorch_multiprocessing_speedup = [pytorch_multiprocessing_base_time / t for t in pytorch_multiprocessing_times]

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(num_workers_list, multiprocessing_speedup, label='Multiprocessing', marker='o')
    plt.plot(num_workers_list, threading_speedup, label='Threading', marker='o')
    plt.plot(num_workers_list, joblib_speedup, label='Joblib', marker='o')
    plt.plot(num_workers_list, numba_parallel_speedup, label='Numba Parallel', marker='o')
    plt.plot(num_workers_list, dask_speedup, label='Dask', marker='o')
    plt.plot(num_workers_list, pytorch_multiprocessing_speedup, label='PyTorch Multiprocessing', marker='o')
    plt.xlabel('Number of Workers')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Number of Workers')
    plt.legend()
    plt.grid(True)
    plt.show()

# Для MKL можно использовать библиотеку mkl, если она установлена
try:
    import mkl


    def set_num_threads(num_threads):
        mkl.set_num_threads(num_threads)
        print(f"MKL: Number of threads set to {num_threads}")
except ImportError:
    def set_num_threads(num_threads):
        # Для OpenBLAS используем переменные окружения
        os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        print(f"Environment variable set: OPENBLAS_NUM_THREADS={num_threads}, MKL_NUM_THREADS={num_threads}")


# Тестовая функция для умножения больших матриц
def matrix_multiplication_test(size, num_threads, num_runs):
    set_num_threads(num_threads)

    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    total_time = 0.0
    for _ in range(num_runs):
        start_time = time.time()
        C = np.dot(A, B)
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

    avg_time = total_time / num_runs
    print(f"Average time for {num_runs} runs with {num_threads} threads: {avg_time:.4f} seconds.")
    return avg_time


# Основной тест
if __name__ == "__main__":
    size = 2000  # Размер матриц
    num_runs = 5  # Количество запусков для усреднения времени

    # Тестируем с различным числом потоков
    times = {}
    for num_threads in [1, 2, 4, 8]:
        avg_time = matrix_multiplication_test(size, num_threads, num_runs)
        times[num_threads] = avg_time

    # Расчет ускорения (speedup)
    speedup = {threads: times[1] / time for threads, time in times.items()}

    # Вывод результатов
    print("\nSummary of times and speedup:")
    for num_threads, avg_time in times.items():
        print(f"{num_threads} threads: {avg_time:.4f} seconds, Speedup: {speedup[num_threads]:.2f}x")

if __name__ == "__main__":
    run_parallel_tests()