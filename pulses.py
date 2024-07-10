import numpy as np


def KuznetsovMaSoliton(t, x, par=1):
    """ Солитон Кузнецова-Ма """
    b = np.sqrt(8 * par * (1 - 2 * par))
    w = 2 * np.sqrt(1 - 2 * par)
    numer = 2 * (1 - 2 * par) * np.cosh(b * x) + 1j * b * np.sinh(b * x)
    den = np.sqrt(2 * par) * np.cos(w * t) - np.cosh(b * x)
    return (1 + numer / den) * np.exp(1j * x)


def fundamental_soliton(t, x, beta_2, lamb=1, c=1):
    """ Классический солитон """
    return c * lamb / np.cosh(lamb*t) * np.exp(-1j * beta_2 * lamb**2 * x)


def gain_loss_soliton(t, x, beta_2, gamma, E_sat, alpha, g_0):
    """ Солитон для случая равномерного усиления и без учёта быстрого насыщения поглощения """
    E_s = E_sat * (g_0 - alpha) / alpha
    multiplier = E_s * np.sqrt(gamma / (-4*beta_2))
    numer = np.exp(-1j * E_s**2 * gamma**2 * x / (8*beta_2))
    denum = np.cosh(-t * E_s * gamma / (2*beta_2))
    return multiplier * (numer / denum)


def AnySoliton(t, x, a, theta, tau, Phi_z):
    mul1 = a * np.exp(1j*Phi_z*x)
    mul2 = 1 / (np.cosh(t*tau)**(1 - 1j*theta/tau))
    return mul1 * mul2


def GaussianPulse(t, p=0.687, tau=1.775, phase=0, chirp=0):
    """ Импульс Гаусса """
    power = -t**2 / (2 * tau**2) * (1 + 1j*chirp)
    return np.sqrt(p) * np.exp(power) * np.exp(1j*phase)


def LaserPulse(t, p, tau, phase=0, chirp=0):
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





