from scipy.fft import fft, ifft
from pulses import *


def GetEnergy_Simpson(arr_func, time_step):
    """ Возвращает величину энергии (интеграл считается по формуле Симпсона) """
    n = len(arr_func)
    summ = arr_func[n - 2] + 4*arr_func[n-1] + arr_func[0]
    for i in range(1, n - 1, 2):
        summ += arr_func[i - 1] + 4*arr_func[i] + arr_func[i + 1]
    return summ * time_step / 3


def GetEnergy_Rectangles(arr_func, time_step):
    """ Возвращает величину энергии (интеграл считается по формуле левых прямоугольников) """
    return sum(abs(arr_func)**2 * time_step)


def FFTforVector(psi):
    """ Векторизация функции преобразования Фурье """
    n = len(psi)
    res = []
    for i in range(n):
        res.append(fft(psi[i]))
    return np.array(res)


def iFFTforVector(psi):
    """ Векторизация функции обратного преобразования Фурье """
    n = len(psi)
    res = []
    for i in range(n):
        res.append(ifft(psi[i]))
    return np.array(res)


def NonLinear(psi, gamma, E_sat, g_0, CurrEnergy, step):
    """ Нелинейный оператор (Керр и насыщение) """
    n = len(psi)
    for i in range(n):
        P_k = abs(psi[i])**2
        E_k = CurrEnergy[i]
        if g_0 == 0:  # нет усиления
            psi[i] = psi[i] * np.exp(1j * gamma * P_k * step)
            continue
        if E_k == 0:
            continue
        E = np.sqrt((E_k ** 2 + 2 * E_k * E_sat) * np.exp(2 * g_0 * step) + E_sat ** 2) - E_sat
        C = -gamma * P_k * (E_k + E_sat - E_sat * np.log(E_k + 2 * E_sat)) / (g_0 * E_k) + np.angle(psi[i])
        P = P_k * np.exp(g_0 * step) * np.sqrt((E_k + 2 * E_sat) / E_k) * np.sqrt(E / (E + 2 * E_sat))
        phi = gamma * P_k * (E + E_sat - E_sat * np.log(E + 2 * E_sat)) / (g_0 * E_k) + C
        psi[i] = np.sqrt(P) * np.exp(1j * phi)


def DispAndCoup(psi, Dmat):
    """ Линейный оператор (связи, дисперсия и потери) """
    n = len(psi)
    resV = np.zeros_like(psi)
    for i in range(n):
        for j in range(n):
            resV[i] += psi[j] * Dmat[i*n + j]
    return resV


def SSFMOrder2(psi, CurrentEnergy, D, gamma, E_sat, g_0, h, tau):
    """ Реализация схемы расщепления """
    num = len(psi)
    if g_0 != 0:
        for i in range(num):
            CurrentEnergy[i] = GetEnergy_Rectangles(psi[i], tau)
    NonLinear(psi, gamma, E_sat, g_0, CurrentEnergy, h/2)
    psi = FFTforVector(psi)
    psi = DispAndCoup(psi, D)
    psi = iFFTforVector(psi)
    if g_0 != 0:
        for i in range(num):
            CurrentEnergy[i] = GetEnergy_Rectangles(psi[i], tau)
    NonLinear(psi, gamma, E_sat, g_0, CurrentEnergy, h/2)
    return psi



