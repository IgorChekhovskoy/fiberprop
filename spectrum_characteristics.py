from scipy.fft import fftfreq, fftshift, ifft
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from cmath import pi, sqrt
import numpy as np


def PrintSpectrumInfo(A, B, n, N, M, z1, z2, t1, t2, step, dirName):
    """ Поле спектра на тепловой карте, спектр в конечной точке (нормальный и логарифмический масштаб) """
    S = np.empty((n, N, M - 1), dtype=float)
    for i in range(n):
        for k in range(N):
            S[i][k] = (abs(fftshift(ifft(A[i][k])) * (t2 - t1) / sqrt(2 * pi)) ** 2)

    S_single = np.empty((N, M - 1), dtype=float)
    for k in range(N):
        S_single[k] = (abs(fftshift(ifft(B[k])) * (t2 - t1) / sqrt(2 * pi)) ** 2)

    central_wavelength = 1050.18 * 1e-9  # m
    ligthVelocity = 299792458 * 1e-15  # m/fs
    central_freq = (ligthVelocity) / central_wavelength  # Hz*1e15

    z_2 = np.linspace(z1, z2, N)
    omega_2 = fftshift(fftfreq(M - 1, step)) + central_freq  # Hz*1e15 TODO : Георгий, тут не перепутана ли круговая и обычная?
    wavelengths = ligthVelocity / omega_2 * 1e9  # nm
    W, Z = np.meshgrid(wavelengths, z_2)

    fig = plt.figure(figsize=(22, 20))
    axs = fig.subplots(4, 2)
    for j in range(n + 1):
        k = 0
        m = j
        if j > 2:
            k = 1
            m -= 3
        else:
            m += 1
        current_title = 'spectrum, single core'
        current_field = S_single
        if j < n:
            current_title = 'spectrum, {} core'.format(j)
            current_field = S[j]
        if j == n:
            m = 0
            k = 0
        ax = axs[m, k]
        ax.set_title(current_title)
        ax.set_xlabel('wavelength, nm')
        ax.set_ylabel('z, cm')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax.grid(which='minor', color='gray', linestyle=':')
        ax.set_ylim(bottom=z1, top=z2)
        ax.set_xlim(left=900, right=1200)
        pcm = ax.pcolormesh(W, Z, current_field, cmap='nipy_spectral')
        fig.colorbar(pcm, ax=ax)
    name = 'поле_спектра'
    path = dirName + '/' + name
    fig.savefig('{}.png'.format(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('SpectrumField_HeatMap: \tDone')
    #######################################################
    fig = plt.figure(figsize=(22, 20))
    axs = fig.subplots(4, 2)
    for j in range(n):
        k = 0
        m = j
        if j > 2:
            k = 1
            m -= 3
        else:
            m += 1
        ax = axs[m, k]
        ax.set_title('signal spectrum in the last point, {} core'.format(j))
        ax.set_xlabel('wavelength, nm')
        ax.plot(wavelengths, S_single[0], color='black', linewidth=2, label='input')
        ax.plot(wavelengths, S[j][N - 1], color='blue', linewidth=2, label='output spectrum, {} core'.format(j))
        ax.plot(wavelengths, S_single[N - 1], color='green', linewidth=2, label='output spectrum, single core')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        # ax.grid(which='major', color = 'k')
        ax.grid(which='minor', color='gray', linestyle=':')
        ax.set_xlim(left=900, right=1200)
        ax.set_ylim(bottom=0, top=max(max(S[j][N - 1]), max(S_single[N - 1]), max(S_single[0])) * 1.05)
        ax.legend(fontsize='12', facecolor='white', loc='upper left')
    axs[0, 0].set_visible(False)
    dir = dirName
    name = 'спектр_в_конечной_точке'
    path = dir + '/' + name
    fig.savefig('{}.png'.format(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('EndPointSpectrum: \tDone')
    #######################################################
    fig = plt.figure(figsize=(22, 20))
    axs = fig.subplots(4, 2)
    for j in range(n):
        k = 0
        m = j
        if j > 2:
            k = 1
            m -= 3
        else:
            m += 1
        ax = axs[m, k]
        ax.set_title('spectrum (log scale) in the last point, {} core'.format(j))
        ax.set_xlabel('wavelength, nm')
        ax.semilogy(wavelengths, S_single[0], color='black', linewidth=2, label='input')
        ax.semilogy(wavelengths, S[j][N - 1], color='blue', linewidth=2, label='output spectrum, {} core'.format(j))
        ax.semilogy(wavelengths, S_single[N - 1], color='green', linewidth=2, label='output spectrum, single core')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        # ax.grid(which='major', color = 'k')
        ax.grid(which='minor', color='gray', linestyle=':')
        ax.set_xlim(left=900, right=1200)
        ax.set_ylim(bottom=10, top=max(max(S[j][N - 1]), max(S_single[N - 1]), max(S_single[0])) * 5)
        ax.legend(fontsize='12', facecolor='white', loc='upper left')
    axs[0, 0].set_visible(False)
    dir = dirName
    name = 'спектр_(логарифмический_масштаб)_в_конечной_точке'
    path = dir + '/' + name
    fig.savefig('{}.png'.format(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('EndPointSpectrum_logScale: \tDone')
    return S, S_single, wavelengths

