from tests import *


if __name__ == '__main__':
    N = 100
    M = 2**10 + 1
    num = 7
    beta_2 = -1.0
    gamma = 1.0
    E_sat = 1.0
    alpha = 0.1
    g_0 = 0.4
    L1, L2 = 0, 1
    T1, T2 = -25, 25
    # test_case3(N, M, num, beta_2, gamma, E_sat, alpha, g_0, a, b, T1, T2, plot=True)

    test_case_MCF_2core(plot=True)
