import numpy as np
from math import factorial, cos, pi, sqrt


def GetRingCouplingMatrix(n):
    elem = [1, -2, 1]
    lstlen = len(elem)
    a = elem[lstlen % 2:] + [0] * (n - lstlen) + elem[:lstlen % 2]
    asw = [[a[k - i] for k in range(n)] for i in range(n)]
    return np.array(asw)


def GetCentralCouplingMatrix(n):
    asw = []
    k = sqrt(2 * (1 - cos(2 * pi / (n - 1))))
    for i in range(n):
        row = []
        if i == 0:
            row = [1 for j in range(n)]
            row[0] = -(n - 1)
        else:
            rowPart = [k, -(1 + 2 * k), k] + [0] * (n - 4)
            shift = i - 2
            rowPart = rowPart[-shift:] + rowPart[:-shift]
            row = [1] + rowPart
        row = np.array(row)
        asw.append(row)
    return np.array(asw)


def CreateMyFreqMatrix(mat, beta_2, alpha, g_0, w, step):
    asw = []
    n = len(mat)
    for i in range(n * n):
        x = i // n
        y = i % n
        row = []
        if x == y:
            row = [step * (1j * mat[x][y] + (1j * beta_2 * w_k**2 - alpha - g_0) / 2) for w_k in w]
        else:
            row = [step * 1j * mat[x][y]] * len(w)
        asw.append(row)
    return np.array(asw)


def PadeExpForMatrix(matrix, k=6, m=6):
    leftPart = np.zeros((len(matrix), len(matrix[0])), dtype=complex)
    rightPart = np.zeros((len(matrix), len(matrix[0])), dtype=complex)
    for i in range(m + 1):
        leftPart += factorial(k + m - i) * factorial(m) / (
                    factorial(k + m) * factorial(m - i) * factorial(i)) * np.linalg.matrix_power(-matrix, i)
    for j in range(k + 1):
        rightPart += factorial(k + m - j) * factorial(k) / (
                    factorial(k + m) * factorial(k - j) * factorial(j)) * np.linalg.matrix_power(matrix, j)
    return np.dot(np.linalg.inv(leftPart), rightPart)


def PadeExpForMyFreqMatrix(FreqMat):
    ret = np.empty_like(FreqMat)
    n = len(FreqMat)
    side = int(sqrt(n))
    m = len(FreqMat[0])
    for i in range(m):
        vec = np.zeros(n, dtype=complex)
        for j in range(n):
            vec[j] = FreqMat[j][i]
        mat = PadeExpForMatrix(np.reshape(vec, (side, side)))
        vec = np.reshape(mat, n)
        for j in range(n):
            ret[j][i] = vec[j]
    return ret

