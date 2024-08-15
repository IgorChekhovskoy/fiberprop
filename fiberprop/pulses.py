import numpy as np


def KuznetsovMaSoliton(t, z, par=1):
    """ Солитон Кузнецова-Ма """
    b = np.sqrt(8 * par * (1 - 2 * par))
    w = 2 * np.sqrt(1 - 2 * par)
    numer = 2 * (1 - 2 * par) * np.cosh(b * z) + 1j * b * np.sinh(b * z)
    den = np.sqrt(2 * par) * np.cos(w * t) - np.cosh(b * z)
    return (1 + numer / den) * np.exp(1j * z)


def fundamental_soliton(t, z, beta2, lamb=1, c=1):
    """ Классический солитон """
    return c * lamb / np.cosh(lamb*t) * np.exp(-1j * beta2 * lamb ** 2 * z)


def gain_loss_soliton(t, z, beta2, gamma, E_sat, alpha, g_0):
    """ Солитон для случая равномерного усиления и без учёта быстрого насыщения поглощения """
    E_s = E_sat * (g_0 - alpha) / alpha
    multiplier = E_s * np.sqrt(gamma / (-4*beta2))
    numer = np.exp(-1j * E_s**2 * gamma**2 * z / (8*beta2))
    denum = np.cosh(-t * E_s * gamma / (2*beta2))
    return multiplier * (numer / denum)


def any_soliton(t, z, a=1, theta=1, tau=1, phi_z=1):
    mul1 = a * np.exp(1j * phi_z * z)
    mul2 = 1 / (np.cosh(t*tau)**(1 - 1j*theta/tau))
    return mul1 * mul2


def gaussian_pulse(t, p=0.687, tau=1.775, phase=0, chirp=0):
    """ Импульс Гаусса """
    power = -t**2 / (2 * tau**2) * (1 + 1j*chirp)
    return np.sqrt(p) * np.exp(power) * np.exp(1j*phase)


def laser_pulse(t, p, tau, phase=0, chirp=0):
    """ Лазерный импульс, для дипломной работы """
    simple = 1 / np.cosh(1.7627 * t/tau)
    return np.sqrt(p) * np.exp(1j*phase) * simple**(1 + 1j*chirp)


def get_analytical_field(params_dict, t_arr, z_arr, function):
    """ По функции и некоторым параметрам строит поле её значений (аналитическое решение для одной сердцевины) """
    N = len(z_arr)
    M = len(t_arr) + 1
    analytical_solution = np.zeros((N, M - 1), dtype=complex)  # сюда будет записываться решение
    for n, z in enumerate(z_arr):
        analytical_solution[n] = function(t_arr, z, **params_dict)
    return analytical_solution





