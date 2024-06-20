from tests import *


if __name__ == '__main__':
    N = 100
    M = 1024
    num = 7
    beta_2 = -1.0
    gamma = 1.0
    E_sat = 1.0
    alpha = 0.1
    g_0 = 0.4
    a, b = 0, 1
    c, d = -25, 25
    test_case2(N, M, num, beta_2, gamma, E_sat, alpha, g_0, a, b, c, d)
