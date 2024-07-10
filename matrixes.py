import numpy as np
from scipy.linalg import expm
from math import factorial, cos, pi, sqrt
from numba import jit


@jit(nopython=True)
def get_ring_coupling_matrix(n):
    elem = [1, -2, 1]
    lstlen = len(elem)
    a = elem[lstlen % 2:] + [0] * (n - lstlen) + elem[:lstlen % 2]
    asw = [[a[k - i] for k in range(n)] for i in range(n)]
    return np.array(asw)


@jit(nopython=True)
def get_cental_coupling_matrix(n):
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


@jit(nopython=True)
def create_freq_matrix(mat, beta_2, alpha, g_0, omega, step):
    n = len(mat)
    m = len(omega)
    asw = np.empty((n * n, m), dtype=np.complex128)

    diag_elements = step * (1j * np.diag(mat)[:, None] + (1j * beta_2[:, None] * omega ** 2 - alpha[:, None] - g_0[:, None]) / 2)
    off_diag_elements = step * 1j * mat

    for i in range(n * n):
        x = i // n
        y = i % n
        if x == y:
            asw[i] = diag_elements[x]
        else:
            asw[i] = off_diag_elements[x, y]

    return asw


def precompute_coeffs(k, m):
    left_coeffs = np.zeros(m + 1)
    right_coeffs = np.zeros(k + 1)
    for i in range(m + 1):
        left_coeffs[i] = factorial(k + m - i) * factorial(m) / (factorial(k + m) * factorial(m - i) * factorial(i))
    for j in range(k + 1):
        right_coeffs[j] = factorial(k + m - j) * factorial(k) / (factorial(k + m) * factorial(k - j) * factorial(j))
    return left_coeffs, right_coeffs

def PadeExpForMatrix(matrix, left_coeffs, right_coeffs, leftPart, rightPart):
    np.copyto(leftPart, np.zeros_like(matrix))
    np.copyto(rightPart, np.zeros_like(matrix))

    for i, coeff in enumerate(left_coeffs):
        leftPart += coeff * np.linalg.matrix_power(-matrix, i)

    for j, coeff in enumerate(right_coeffs):
        rightPart += coeff * np.linalg.matrix_power(matrix, j)

    return np.dot(np.linalg.inv(leftPart), rightPart)

def get_pade_exponential(FreqMat, k=6, m=6):
    n, p = FreqMat.shape
    side = int(sqrt(n))
    ret = np.empty((n, p), dtype=complex)
    left_coeffs, right_coeffs = precompute_coeffs(k, m)
    leftPart = np.zeros((side, side), dtype=complex)
    rightPart = np.zeros((side, side), dtype=complex)
    temp_matrix = np.empty((side, side), dtype=complex)

    for i in range(p):
        vec = FreqMat[:, i]
        np.copyto(temp_matrix, np.reshape(vec, (side, side)))
        mat = PadeExpForMatrix(temp_matrix, left_coeffs, right_coeffs, leftPart, rightPart)
        ret[:, i] = np.reshape(mat, n)

    return ret


#@jit(nopython=True)
def get_pade_exponential2(FreqMat):
    n, p = FreqMat.shape
    side = int(sqrt(n))
    ret = np.empty((n, p), dtype=complex)
    temp_matrix = np.empty((side, side), dtype=complex)

    for i in range(p):
        temp_matrix[:, :] = FreqMat[:, i].reshape(side, side)
        mat = expm(temp_matrix)
        ret[:, i] = mat.ravel()

    return ret
