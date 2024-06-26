import numpy as np
import matplotlib.pyplot as plt


def GetScales(beta_2, gamma_2, C):  # эта реализация функции только для гексагонального волокна
    """ Функция возвращает значения параметров, на которые нужно умножить
    безразмерное решение, чтобы перейти к размерному. """
    K = C / gamma_2
    L = 1 / C
    T = np.sqrt(np.fabs(beta_2) / (2 * C))
    return K, L, T


def norm2dim(a, b, c, d, h, tau, beta_2, gamma_2, C, U, N, M, MCF_config):
    """ Функция приводит безразмерное решение к размерному виду """
    self_coef = 0
    if MCF_config == 'ring':
        self_coef = 2
    elif MCF_config == 'hexagonal':
        self_coef = 6
    elif MCF_config == 'nothing':
        self_coef = 0
    else:
        raise RuntimeError('Unsupportable MCF configuration')

    zn = np.linspace(a, b, N)
    tn = np.linspace(c, d, M)
    Tn, Zn = np.meshgrid(tn[:-1], zn)
    K = np.exp(1j * self_coef * Zn) * np.sqrt(C / gamma_2)
    A = K * U
    L = 1 / C
    a *= L
    b *= L
    h *= L
    T = np.sqrt(np.fabs(beta_2) / (2 * C))
    c *= T
    d *= T
    tau *= T
    z = np.linspace(a, b, N)
    t = np.linspace(c, d, M)
    T, Z = np.meshgrid(t[:-1], z)
    return T, Z, A, h, tau, a, b, c, d

