import numpy as np


def get_scales(beta_2, gamma, C):  # эта реализация функции только для гексагонального волокна
    """ Функция возвращает значения параметров, на которые нужно умножить
    безразмерное решение, чтобы перейти к размерному. """
    K = C / gamma
    L = 1 / C
    T = np.sqrt(np.fabs(beta_2) / (2 * C))
    return K, L, T


def norm2dim(L1, L2, T1, T2, h, tau, beta_2, gamma, C, U, N, M, core_configuration):
    """ Функция приводит безразмерное решение к размерному виду """

    if core_configuration == 'ring':
        self_coefficient = 2
    elif core_configuration == 'square':
        self_coefficient = 4
    elif core_configuration == 'hexagonal':
        self_coefficient = 6
    elif core_configuration == 'nothing':
        self_coefficient = 0
    else:
        raise RuntimeError('Unsupportable MCF configuration')

    zn = np.linspace(L1, L2, N)
    tn = np.linspace(T1, T2, M, endpoint=False)
    Tn, Zn = np.meshgrid(tn, zn)

    K = np.exp(1j * self_coefficient * Zn) * np.sqrt(C / gamma)
    A = K * U

    L = 1 / C
    L1 *= L
    L2 *= L
    h *= L

    T = np.sqrt(np.fabs(beta_2) / (2 * C))
    T1 *= T
    T2 *= T
    tau *= T

    z = np.linspace(L1, L2, N)
    t = np.linspace(T1, T2, M, endpoint=False)
    T, Z = np.meshgrid(t, z)

    return T, Z, A, h, tau, L1, L2, T1, T2
