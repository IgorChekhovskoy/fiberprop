import numpy as np
from scipy.fftpack import fft, ifft
from scipy.linalg.blas import zaxpy
import matplotlib.pyplot as plt
import time


def get_energy(u_modsqr_d, tau):
    return np.sum(u_modsqr_d) * tau


def get_mod_sqr(u):
    return np.abs(u) ** 2


def get_cubic_nonlinearity(u, u_modsqr_d, nonlinear_cubic_coeffs_array, tau, E_sat, alpha, g_0):
    energy_u = tau * np.sum(u_modsqr_d, axis=1)[:, np.newaxis]
    new_term = -0.5j * (g_0[:, np.newaxis] / (1 + energy_u / E_sat[:, np.newaxis]) - alpha[:, np.newaxis]) * u

    cubic_nonlinearity = nonlinear_cubic_coeffs_array.diagonal()[:, np.newaxis] * u_modsqr_d * u

    return cubic_nonlinearity + new_term


def get_v(u, u_modsqr_d, linear_coeffs_array, nonlinear_cubic_coeffs_array, M, mask_array, tau, E_sat, alpha, g_0):
    v = np.zeros_like(u, dtype=complex)

    # Рассчитываем нелинейную часть
    v += get_cubic_nonlinearity(u, u_modsqr_d, nonlinear_cubic_coeffs_array, tau, E_sat, alpha, g_0)

    # t = time.time()
    # Добавляем линейную часть
    for l in range(len(u)):
        v[l] = zaxpy(u[l], v[l], a=linear_coeffs_array[l][l])  # Выполнение v[l] += linear_coeffs_array[l][l] * u[l]
        for neighbor in mask_array[l].neighbors:
            zaxpy(u[neighbor], v[l], a=linear_coeffs_array[l][neighbor])  # Выполнение v[l] += linear_coeffs_array[l][neighbor] * u[neighbor]
    # print(time.time() - t)

    # v += np.einsum('ij,jm->im', linear_coeffs_array, u)

    # Выполнение прямого преобразования Фурье
    v = fft(v, axis=1) / np.sqrt(M)
    return v


# def plot_graph_init(u, v, yscale='linear'):
#     num_plots = u.shape[0]
#     fig, axs = plt.subplots(num_plots, 1, figsize=(12, 6), sharex=True)
#     lines_u = []
#     lines_v = []
#
#     for i in range(num_plots):
#         line_u, = axs[i].plot(np.abs(u[i]), label=f'|u_{i}|')
#         line_v, = axs[i].plot(np.abs(v[i]), label=f'|v_{i}|', linestyle='--')
#         axs[i].axhline(0, color='black', linewidth=0.5)
#         axs[i].set_yscale(yscale)  # Устанавливаем масштабирование по оси y
#         axs[i].legend()  # Переносим информацию в легенду
#         axs[i].grid(True)
#         lines_u.append(line_u)
#         lines_v.append(line_v)
#
#     plt.xlabel('Index')
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)  # Увеличиваем верхний отступ для заголовка
#     plt.ion()  # Включаем интерактивный режим
#     plt.show()
#
#     return fig, axs, lines_u, lines_v
#
#
# def plot_graph_update(fig, axs, lines_u, lines_v, u, v, energy_m, counter_ext, counter_int, update_interval):
#     for i, (line_u, line_v) in enumerate(zip(lines_u, lines_v)):
#         line_u.set_ydata(np.abs(u[i]))
#         line_v.set_ydata(np.abs(v[i]))
#         axs[i].relim()
#         axs[i].autoscale_view()
#
#     fig.suptitle(f"External Iteration: {counter_ext}, Internal Iteration: {counter_int}, energy_m: {energy_m: .6e}")
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#
#     # Управляем скоростью обновления графика
#     time.sleep(update_interval)


def plot_graph_init(u, v, yscale='linear'):
    fig, ax = plt.subplots(figsize=(12, 6))

    line_u, = ax.plot(np.abs(u).flatten(), label='|u|')
    line_v, = ax.plot(np.abs(v).flatten(), label='|v|', linestyle='--')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_yscale(yscale)
    ax.legend()
    ax.grid(True)

    plt.xlabel('Index')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.ion()
    plt.show()

    return fig, ax, line_u, line_v


def plot_graph_update(fig, ax, line_u, line_v, u, v, energy_m, counter_ext, counter_int, update_interval):
    line_u.set_ydata(np.abs(u).flatten())
    line_v.set_ydata(np.abs(v).flatten())
    ax.relim()
    ax.autoscale_view()

    fig.suptitle(f"External Iteration: {counter_ext}, Internal Iteration: {counter_int}, energy_m: {energy_m: .6e}")
    fig.canvas.draw()
    fig.canvas.flush_events()

    time.sleep(update_interval)


def shift_to_center_based_on_global_max(u):
    size, M = u.shape
    mid_index = M // 2  # Индекс середины

    # Находим индексы максимальных значений для каждого ряда
    max_indices = np.argmax(np.abs(u), axis=1)

    # Находим глобальный максимум среди всех рядов
    global_max_index = np.argmax(np.abs(u.flatten()))  # Индекс глобального максимума в массиве
    global_max_row, global_max_col = np.unravel_index(global_max_index, u.shape)  # Преобразуем в индекс строки и столбца

    # Рассчитываем сдвиг для глобального максимума
    shift = mid_index - max_indices[global_max_row]

    # Выполняем сдвиг для всех рядов на одно и то же количество позиций
    shifted_u = np.roll(u, shift, axis=1)

    return shifted_u


def find_stationary_solution(u_initial, M, tau, linear_coeffs_array, nonlinear_cubic_coeffs_array,
                             second_derivative_coeffs_array, omega2, mask_array, E_sat, alpha, g_0,
                             lambda_val,
                             max_iter=200, tol=1e-11, plot_graphs=False, update_interval=0.0, yscale='linear'):
    """
    Finds the stationary solution for the given initial condition.

    Parameters:
    ----------
    u_initial : np.ndarray
        Initial condition (size, M).
    M : int
        Number of points in time.
    tau : float
        Time step size.
    linear_coeffs_array : np.ndarray
        Linear coefficients array (size, size).
    nonlinear_cubic_coeffs_array : np.ndarray
        Nonlinear cubic coefficients array (size, size).
    omega2 : np.ndarray
        Array of omega squared values (M,).
    lambda_val : float
        Lambda parameter.
    mask_array : list
        List of Mask objects that contain information about neighbors.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.
    plot_graphs : bool, optional
        If True, plots the graphs of u and V during iterations. Default is False.
    update_interval : float, optional
        Time in seconds to wait between graph updates. Default is 0.1 seconds.
    yscale : str, optional
        Scale for y-axis: 'linear' or 'log'. Default is 'linear'.

    Returns:
    -------
    np.ndarray
        Stationary solution (size, M).
    """

    u = np.copy(u_initial)
    u_modsqr_d = get_mod_sqr(u)
    energy = get_energy(u_modsqr_d, tau)

    initial_energy = energy
    e_min = 1e-2 * initial_energy
    e_max = 1e+2 * initial_energy

    if plot_graphs:
        fig, axs, lines_u, lines_v = plot_graph_init(u, u, yscale=yscale)

    counter_ext = 0
    while counter_ext < max_iter:
        energy_m = 0.5 * (e_min + e_max)
        energy_prev = 0
        counter_int = 0

        while counter_int < max_iter:
            u_modsqr_d = get_mod_sqr(u)

            v = get_v(u, u_modsqr_d, linear_coeffs_array, nonlinear_cubic_coeffs_array, M, mask_array, tau, E_sat, alpha, g_0)

            v /= (lambda_val ** 2 + second_derivative_coeffs_array[:, np.newaxis] * omega2)

            u_modsqr_d = get_mod_sqr(v)
            energy = get_energy(u_modsqr_d, tau)

            normalize = np.sqrt(max(0.0, energy_m / energy))
            v *= normalize

            v = ifft(v, axis=1) * np.sqrt(M)

            u, v = v, u

            if plot_graphs:
                plot_graph_update(fig, axs, lines_u, lines_v, u, v, energy_m, counter_ext, counter_int, update_interval)

            error_energy_int = abs(energy_prev - energy) / energy
            if error_energy_int < tol:
                break
            energy_prev = energy
            counter_int += 1

        delta_energy = energy - energy_m
        error_energy_ext = abs(delta_energy) / energy_m
        if error_energy_ext < tol:
            break

        if delta_energy > 0.0:
            e_max = energy_m
        else:
            e_min = energy_m

        counter_ext += 1

    u[abs(u) < 1e-15] = 0.0
    u = shift_to_center_based_on_global_max(u)
    return u
