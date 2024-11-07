from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import numpy as np

from fiberprop.ssfm_mcf import get_simpson_integral


def VolumetricPowerField(dirName, Z, T, A, B):
    """ Функция строит и сохраняет поле 3D мощностей """
    figMCF = plt.figure(figsize=(24, 8))  # графики для MCF
    n1 = 0
    ax0 = figMCF.add_subplot(1, 2, 1, projection='3d')
    ax0.set_xlabel('propagation distance, cm')
    ax0.set_ylabel('time, fs')
    ax0.set_title('power [W] of {} fiberprop'.format(n1))
    ax0.plot_surface(Z, T, abs(A[n1]) ** 2, cmap='inferno')
    n2 = 1
    ax1 = figMCF.add_subplot(1, 2, 2, projection='3d')
    ax1.set_xlabel('propagation distance, cm')
    ax1.set_ylabel('time, fs')
    ax1.set_title('power [W] of {} fiberprop'.format(n2))
    ax1.plot_surface(Z, T, abs(A[n2]) ** 2, cmap='inferno')
    name = 'поле_мощности_3D_MCF'
    path = dirName + '/' + name
    figMCF.savefig('{}.png'.format(path), bbox_inches='tight', facecolor='white')
    plt.close(figMCF)

    figSC = plt.figure(figsize=(11, 8))  # график для односердцевинного волокна
    ax = figSC.add_subplot(111, projection='3d')
    ax.set_title('single fiberprop')
    ax.plot_surface(Z, T, abs(B) ** 2, cmap='inferno')
    name = 'поле_мощности_3D_single_core'
    path = dirName + '/' + name
    figSC.savefig('{}.png'.format(path), bbox_inches='tight', facecolor='white')
    plt.close(figSC)
    print('VolumetricPowerField: \tDone')
    return


def VolumetricErrorField(dirName, Z, T, A, B):
    """ Функция строит и сохраняет поле 3D мощностей """
    figSC = plt.figure(figsize=(11, 8))  # график для односердцевинного волокна
    field = abs(A - B)
    ax = figSC.add_subplot(111, projection='3d')
    ax.set_title('absolute error')
    ax.plot_surface(Z, T, field, cmap='inferno')
    name = 'поле_ошибки_3D'
    path = dirName + '/' + name
    figSC.savefig('{}.png'.format(path), bbox_inches='tight', facecolor='white')
    plt.close(figSC)
    print('VolumetricErrorField: \tDone')
    return


def EnergyEvo(dirName, A, B, n, M, N, step, z1, z2):
    """ Графики изменения энергии при распространении по волокну """
    P = abs(A) ** 2
    P_single = abs(B) ** 2
    asw = []
    z = np.arange(N)
    for i in range(n + 1):
        if i == n:
            F = lambda x: get_simpson_integral(P_single[x], step)
            E_z = np.vectorize(F)(z)
            asw.append(E_z / 1e6)
            continue
        F = lambda x: get_simpson_integral(P[i][x], step)
        E_z = np.vectorize(F)(z)
        asw.append(E_z / 1e6)
    fig, ax = plt.subplots(figsize=(12, 6))
    title = 'energy'
    ax.set_title(title)
    ax.set_ylabel('energy, nJ')
    ax.set_xlabel('z, cm')
    ax.set_ylim(bottom=0, top=np.max(asw) * 1.05)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xlim(left=z1, right=z2)
    ax.grid(which='major', color='k')
    ax.grid(which='minor', color='gray', linestyle=':')
    arg = np.linspace(z1, z2, N)
    for j in range(n):
        ax.plot(arg, asw[j], linewidth=3, label='{} fiberprop'.format(j))
    ax.plot(arg, asw[n], linewidth=3, color='black', label='single fiberprop')
    ax.legend(fontsize='12', facecolor='white')
    name = 'энергия'
    path = dirName + '/' + name
    fig.savefig('{}.png'.format(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('EnergyEvo: \tDone')
    return asw


def PowerEvo(dirName, A, B, n, M, N, z1, z2):
    """ Графики изменения максимальной мощности и центральной мощности при распространении по волокну """
    P = abs(A) ** 2
    P_single = abs(B) ** 2
    asw_central = []
    asw_highest = []
    z = np.arange(N)
    for i in range(n + 1):
        if i == 0:
            F_central = lambda x: P_single[x][M // 2]
            P_central = np.vectorize(F_central)(z)
            asw_central.append(P_central / 1e3)
            F_highest = lambda x: np.max(P_single[x])
            P_highest = np.vectorize(F_highest)(z)
            asw_highest.append(P_highest / 1e3)
            continue
        F_central = lambda x: P[i - 1][x][M // 2]
        P_central = np.vectorize(F_central)(z)
        asw_central.append(P_central / 1e3)
        F_highest = lambda x: np.max(P[i - 1][x])
        P_highest = np.vectorize(F_highest)(z)
        asw_highest.append(P_highest / 1e3)
    fig = plt.figure(figsize=(24, 6))

    ax0 = fig.add_subplot(1, 2, 1)
    title0 = 'highest power'
    ax0.set_title(title0)
    ax0.set_ylabel('power, kW')
    ax0.set_xlabel('z, cm')
    ax0.set_ylim(bottom=0, top=np.max(asw_highest) * 1.05)
    # ax0.xaxis.set_major_locator(ticker.MultipleLocator(5))
    # ax0.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax0.set_xlim(left=z1, right=z2)
    ax0.grid(which='major', color='k')
    ax0.grid(which='minor', color='gray', linestyle=':')
    arg = np.linspace(z1, z2, N)
    for j in range(1, n + 1):
        ax0.plot(arg, asw_highest[j], linewidth=3, label='{} fiberprop'.format(j - 1))
    ax0.plot(arg, asw_highest[0], linewidth=3, color='black', label='single fiberprop')
    ax0.legend(fontsize='12', facecolor='white')

    ax1 = fig.add_subplot(1, 2, 2)
    title1 = 'power at t=0'
    ax1.set_title(title1)
    ax1.set_ylabel('power, kW')
    ax1.set_xlabel('z, cm')
    ax1.set_ylim(bottom=0, top=np.max(asw_central) * 1.05)
    # ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
    # ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.set_xlim(left=z1, right=z2)
    ax1.grid(which='major', color='k')
    ax1.grid(which='minor', color='gray', linestyle=':')
    arg = np.linspace(z1, z2, N)
    for j in range(1, n + 1):
        ax1.plot(arg, asw_central[j], linewidth=3, label='{} fiberprop'.format(j - 1))
    ax1.plot(arg, asw_central[0], linewidth=3, color='black', label='single fiberprop')
    ax1.legend(fontsize='12', facecolor='white', loc='upper right')

    name = 'характеристики_мощности'
    path = dirName + '/' + name
    fig.savefig('{}.png'.format(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('PowerEvo: \tDone')
    return asw_highest, asw_central


def OutputSignal(dirName, A, B, n, t1, t2, z1, z2, N, M, outZ):
    """ Сопоставление входного сигнала и на длине outZ """
    all_z = np.linspace(z1, z2, N)
    outId = np.argmin(abs(all_z - outZ))
    actArg = all_z[outId]
    arg = np.linspace(t1, t2, M - 1)
    fig = plt.figure(figsize=(18, 20))
    axs = fig.subplots(4, 2)
    inputSignal = (abs(B[0]) ** 2) / 1e3
    outputSingleCore = (abs(B[outId]) ** 2) / 1e3
    for j in range(n):
        outputMultiCore_j = (abs(A[j][outId]) ** 2) / 1e3
        k = 0
        m = j
        if j > 2:
            k = 1
            m -= 3
        else:
            m += 1
        ax = axs[m, k]
        ax.set_title('input-output comparison at z={}cm, {} fiberprop'.format(actArg, j))
        ax.set_ylabel('power, kW')
        ax.set_xlabel('time, fs')
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
        ax.grid(which='major', color='k')
        ax.grid(which='minor', color='gray', linestyle=':')
        ax.set_ylim(bottom=0, top=max(max(outputMultiCore_j), max(inputSignal), max(outputSingleCore)) * 1.05)
        ax.set_xlim(left=t1 * 2 / 3, right=t2 * 2 / 3)
        ax.plot(arg, inputSignal, color='black', linewidth=3, label='input')
        ax.plot(arg, outputSingleCore, color='green', linewidth=3, label='output, single fiberprop')
        ax.plot(arg, outputMultiCore_j, color='blue', linewidth=3, label='output, {} fiberprop'.format(j))
        ax.legend(fontsize='12', facecolor='white', loc='upper right')
    axs[0, 0].set_visible(False)
    name = 'сравнение_вход-выход_при_z={}cm'.format(actArg)
    path = dirName + '/' + name
    fig.savefig('{}.png'.format(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'OutputSignal, L={actArg}: \tDone')

    fig1 = plt.figure(figsize=(18, 20))
    axs1 = fig1.subplots(4, 2)
    inputSignal = (abs(B[0]) ** 2) / 1e3
    outputSingleCore = (abs(B[outId]) ** 2) / 1e3
    for j in range(n):
        outputMultiCore_j = (abs(A[j][outId]) ** 2) / 1e3
        k = 0
        m = j
        if j > 2:
            k = 1
            m -= 3
        else:
            m += 1
        ax = axs1[m, k]
        ax.set_title('input-output comparison at z={}cm, {} fiberprop'.format(actArg, j))
        ax.set_ylabel('power, kW')
        ax.set_xlabel('time, fs')
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
        ax.grid(which='major', color='k')
        ax.grid(which='minor', color='gray', linestyle=':')
        ax.set_ylim(bottom=0.01, top=max(max(outputMultiCore_j), max(inputSignal), max(outputSingleCore)) * 1.05)
        ax.set_xlim(left=t1 * 2 / 3, right=t2 * 2 / 3)
        ax.semilogy(arg, inputSignal, color='black', linewidth=3, label='input')
        ax.semilogy(arg, outputSingleCore, color='green', linewidth=3, label='output, single fiberprop')
        ax.semilogy(arg, outputMultiCore_j, color='blue', linewidth=3, label='output, {} fiberprop'.format(j))
        ax.legend(fontsize='12', facecolor='white', loc='upper right')
    axs1[0, 0].set_visible(False)
    name = 'сравнение_вход-выход_при_z(log_scale)={}cm'.format(actArg)
    path = dirName + '/' + name
    fig1.savefig('{}.png'.format(path), bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print(f'OutputSignal_logScale, L={actArg}: \tDone')
    return


def PeakPowerCoord(dirName, highestPowEvo, A, B, n, step, M, N, z1, z2, t1, t2):
    """ Координаты пиков мощности """
    P = abs(A) ** 2
    P_single = abs(B) ** 2
    asw_all = []
    asw_central = []
    z = np.arange(N)
    t = np.linspace(t1, t2, M)
    for i in range(n + 1):
        if i == n:
            ID = np.argmax(highestPowEvo[0])
            idx = np.argmax(P_single[ID])
            epsilon = (P_single[ID][idx] - P_single[ID][idx - 1]) * 0.1  # здесь 0.1 - эмпирический коэффициент
            F_all = lambda x: find_peaks(P_single[x], prominence=max(P_single[x]) * 0.01, width=M // 200)[0]
            vals = np.vectorize(F_all, otypes=[np.ndarray])(z)
            asw_all.append(vals)
            F_central = lambda x: \
                np.where((P_single[x] <= np.max(P_single[x]) + epsilon) &
                         (P_single[x] >= np.max(P_single[x]) - epsilon))[0]
            coordP_z = np.vectorize(F_central, otypes=[np.ndarray])(z) * step + t1
            asw_central.append(coordP_z)
            continue
        ID = np.argmax(highestPowEvo[i + 1])
        idx = np.argmax(P[i][ID])
        epsilon = (P[i][ID][idx] - P[i][ID][idx - 1]) * 0.1  # здесь 0.1 - эмпирический коэффициент
        F_all = lambda x: find_peaks(P[i][x], prominence=max(P[i][x]) * 0.01, width=M // 200)[0]
        vals = np.vectorize(F_all, otypes=[np.ndarray])(z)
        asw_all.append(vals)
        F_central = lambda x: np.where((P[i][x] <= np.max(P[i][x]) + epsilon) &
                                       (P[i][x] >= np.max(P[i][x]) - epsilon))[0]
        coordP_z = np.vectorize(F_central, otypes=[np.ndarray])(z) * step + t1
        asw_central.append(coordP_z)
    arg = np.linspace(z1, z2, N)
    fig = plt.figure(figsize=(18, 20))
    axs = fig.subplots(4, 2)
    for j in range(n + 1):
        k = 0
        m = j
        if j > 2:
            k = 1
            m -= 3
        else:
            m += 1
        current_title = 'peak coordinates EVO, {} fiberprop'.format(j)
        max_color = 'blue'
        all_color = 'green'
        if j == n:
            current_title = 'peak coordinates EVO, single fiberprop'
            max_color = 'yellow'
            all_color = 'red'
            m = 0
            k = 0
        ax = axs[m, k]
        ax.set_title(current_title)
        ax.set_ylabel('z, cm')
        ax.set_xlabel('time, fs')
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
        ax.grid(which='major', color='k')
        ax.grid(which='minor', color = 'gray', linestyle = ':')
        ax.set_xlim(left=t1 * 2 / 3, right=t2 * 2 / 3)
        ax.set_ylim(bottom=z1, top=z2)
        indexes = np.arange(0, N, 4)
        for k in indexes:
            if k == indexes[-1]:
                ax.scatter(asw_central[j][k], [arg[k]] * len(asw_central[j][k]), s=6, c=max_color, label='max peak')
                res = t[asw_all[j][k]]
                ax.scatter(res, [arg[k]] * len(res), s=3, c=all_color, label='all peaks')
                ax.scatter((-1) * res, [arg[k]] * len(res), s=3, c=all_color)
                continue
            ax.scatter(asw_central[j][k], [arg[k]] * len(asw_central[j][k]), s=6, c=max_color)
            res = t[asw_all[j][k]]
            ax.scatter(res, [arg[k]] * len(res), s=3, c=all_color)
            ax.scatter((-1) * res, [arg[k]] * len(res), s=3, c=all_color)
        ax.legend(fontsize='12', facecolor='white', loc='lower left')
    name = 'координаты_пиков_во_времени'
    path = dirName + '/' + name
    fig.savefig('{}.png'.format(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('PeakPowerCoord: \tDone')
    return asw_central, asw_all


def PowerField(A, B, n, N, M, z1, z2, t1, t2, dirName):
    """ Поля мощности на тепловой карте """
    z_2 = np.linspace(z1, z2, N)
    t_2 = np.linspace(t1, t2, M)
    T, Z = np.meshgrid(t_2[:-1], z_2)
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
        field = abs(B) ** 2
        current_title = 'signal power, single fiberprop'
        if j < n:
            field = abs(A[j]) ** 2
            current_title = 'signal power, {} fiberprop'.format(j)
        if j == n:
            m = 0
            k = 0
        ax = axs[m, k]
        ax.set_title(current_title)
        ax.set_xlabel('time, fs')
        ax.set_ylabel('z, cm')
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
        ax.grid(which='minor', color='gray', linestyle=':')
        ax.set_ylim(bottom=z1, top=z2)
        ax.set_xlim(left=t1 * 2 / 3, right=t2 * 2 / 3)
        pcm = ax.pcolormesh(T, Z, field, cmap='nipy_spectral')
        fig.colorbar(pcm, ax=ax)
    name = 'поле_мощности_сигнала'
    path = dirName + '/' + name
    fig.savefig('{}.png'.format(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('PowerField: \tDone')
    return


def convert(elem):
    """ Функция переводит значение фазы комплексного числа из [-pi;pi] в [0;2*pi] """
    return elem if elem >= 0 else (elem + 2 * np.pi)


def CentralPhase(dirName, A, B, n, N, M, z1, z2):
    """ Фаза в нулевой точке по времени """
    asw = []
    choose = []
    z = np.arange(N)
    for i in range(n):
        F = lambda x: A[i][x][M // 2]
        vals = np.vectorize(F)(z)
        phaseA = np.angle(vals)
        asw.append(np.vectorize(convert)(phaseA))
        choose.append(asw[i][0])
    F_single = lambda x: B[x][M // 2]
    vals_single = np.vectorize(F_single)(z)
    phaseA_single = np.angle(vals_single)
    asw_single = np.vectorize(convert)(phaseA_single)
    criteria = asw_single[0]
    arg = np.linspace(z1, z2, N)
    fig = plt.figure(figsize=(18, 20))
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
        ax.set_title('phase at t=0, {} fiberprop'.format(j))
        ax.set_ylabel('phase, rad')
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.grid(which='major', color='k')
        ax.grid(which='minor', color='gray', linestyle=':')
        ax.set_xlabel('z, cm')
        ax.set_ylim(bottom=0, top=2 * np.pi)
        ax.set_xlim(left=z1, right=z2)
        ax.scatter(arg, asw[j], s=5, c='b', label='phase at {} fiberprop'.format(j))
        if choose[j] == criteria and sum(abs(A[j][0])) > 1:
            ax.scatter(arg, asw_single, s=5, c='g', label='phase at single fiberprop')
        ax.legend(fontsize='12', facecolor='white', loc='upper right')
    axs[0, 0].set_visible(False)
    name = 'фаза_при_t=0'
    path = dirName + '/' + name
    fig.savefig('{}.png'.format(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('CentralPhase: \tDone')
    return asw


def InstantFrequencyField(A, B, n, N, M, z1, z2, t1, t2, step, dirName, smoothBorder=50):
    """ Поле мгновенных частот """
    Phase = np.angle(A)
    IFreq = np.empty((n, N, M - 3), dtype=float)
    for i in range(n):
        for k in range(N):
            for j in range(1, M - 2):
                buff = (Phase[i][k][j + 1] - Phase[i][k][j - 1]) / (2 * step) * 1e3 / (2 * np.pi)
                if j == 1:
                    buff = 0
                IFreq[i][k][j - 1] = buff if abs(buff) < smoothBorder else IFreq[i][k][j - 2]

    Phase_single = np.angle(B)
    IFreq_single = np.empty((N, M - 3), dtype=float)
    for k in range(N):
        for j in range(1, M - 2):
            buff = (Phase_single[k][j + 1] - Phase_single[k][j - 1]) / (2 * step) * 1e3 / (2 * np.pi)
            if j == 1:
                buff = 0
            IFreq_single[k][j - 1] = buff if abs(buff) < smoothBorder else IFreq_single[k][j - 2]

    z_2 = np.linspace(z1, z2, N)
    t_2 = np.linspace(t1, t2, M)
    T, Z = np.meshgrid(t_2[1:-2], z_2)
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
        field = IFreq_single
        current_title = 'instant frequency [THz], single fiberprop'
        if j < n:
            field = IFreq[j]
            current_title = 'instant frequency [THz], {} fiberprop'.format(j)
        if j == n:
            m = 0
            k = 0
        ax = axs[m, k]
        ax.set_title(current_title)
        ax.set_xlabel('time, fs')
        ax.set_ylabel('z, cm')
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
        ax.grid(which='minor', color='black', linestyle=':')
        ax.set_ylim(bottom=z1, top=z2)
        ax.set_xlim(left=t1 * 2 / 3, right=t2 * 2 / 3)
        pcm = ax.pcolormesh(T, Z, field, cmap='nipy_spectral')
        fig.colorbar(pcm, ax=ax)
    name = 'поле_мгновенной_частоты_сигнала'
    path = dirName + '/' + name
    fig.savefig('{}.png'.format(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('InstantFrequencyField: \tDone')
    return IFreq, IFreq_single


def GetXChirp(IFreqField, Field, time, M, x, way):
    """ Функция рассчитывает чирп трёх видов в зависимости от аргумента way
    (по полной ширине, по ширине на полувысоте, в малой окрестности t=0) """
    maxVal = max(abs(Field[x]))
    epsilonPointsNum = 0
    if way == 'averaging_at_FW':
        epsilonPointsNum = M // 2 - np.argmin(abs(abs(Field[x][:M // 2]) - maxVal * 0.07))
    if way == 'averaging_at_FWHM':
        epsilonPointsNum = M // 2 - np.argmin(abs(abs(Field[x][:M // 2]) - maxVal / 2))
    if way == 'around_t=0':
        epsilonPointsNum = 5
    if maxVal == 0:
        epsilonPointsNum = 2
    lbound = (M - 2) // 2 - (epsilonPointsNum - 1)
    rbound = (M - 2) // 2 + epsilonPointsNum
    epsilonArray = time[1:-1][lbound:rbound]
    freqArray = IFreqField[x][lbound:rbound]
    numerator = epsilonArray * freqArray
    k = sum(numerator[abs(freqArray) < 50]) / sum((epsilonArray[abs(freqArray) < 50]) ** 2)
    return k


def CentralPhaseChirp(dirName, IFreq, IFreq_single, A, B, n, z1, z2, t1, t2, N, M, way):
    """ График чирпа трёх видов в зависимости от аргумента way """
    t_2 = np.linspace(t1, t2, M)
    asw = []
    choose = []
    z = np.arange(N)
    for i in range(n):
        F = lambda x: GetXChirp(IFreq[i], A[i], t_2, M, x, way)
        vals = np.vectorize(F)(z)
        asw.append(vals)
        choose.append(vals[0])
    F_single = lambda x: GetXChirp(IFreq_single, B, t_2, M, x, way)
    vals_single = np.vectorize(F_single)(z)
    asw_single = vals_single
    criteria = asw_single[0]
    arg = np.linspace(z1, z2, N)
    fig = plt.figure(figsize=(18, 20))
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
        ax.set_title('chirp, {} fiberprop'.format(j))
        ax.set_ylabel('chirp, THz/fs')
        ax.set_xlabel('z, cm')
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.grid(which='major', color='k')
        ax.grid(which='minor', color='gray', linestyle=':')
        stuff = asw[j]
        ax.set_ylim(bottom=min(min(stuff), min(asw_single)) * 1.05, top=max(max(stuff), max(asw_single)) * 1.05)
        ax.set_xlim(left=z1, right=z2)
        ax.plot(arg, stuff, color='blue', linewidth=3, label='chirp at {} fiberprop'.format(j))
        if choose[j] == criteria and sum(abs(A[j][0])) > 1:
            ax.plot(arg, asw_single, color='green', linewidth=3, label='chirp at single fiberprop')
        ax.legend(fontsize='12', facecolor='white', loc='upper left')
    axs[0, 0].set_visible(False)
    name = 'чирп_сигнала_' + way
    path = dirName + '/' + name
    fig.savefig('{}.png'.format(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'CentralPhaseChirp, {way}: \tDone')
    return asw


def HalfPeakWidth(dirName, highestPowEvo, A, B, n, step, N, M, z1, z2, t1):
    """ Сечение поля мощности на полувысоте """
    P = abs(A) ** 2
    P_single = abs(B) ** 2
    asw_hard = []
    asw_simple = []
    z = np.arange(N)
    for i in range(n):
        ID = np.argmax(highestPowEvo[i + 1])
        epsilon = np.min(np.fabs(P[i][ID] - np.max(P[i][ID]) / 2)) * 0.25  # здесь 0.3 - эмпирический коэффициент
        F_simple = lambda x: np.fabs(P[i][x] - max(P[i][x]) / 2)[:M // 2].argmin()
        coords_simple = np.vectorize(F_simple)(z)
        width_z_simple = coords_simple * step + t1
        asw_simple.append(width_z_simple)
        F_hard = lambda x: \
            np.where((P[i][x] <= np.max(P[i][x]) / 2 + epsilon) & (P[i][x] >= np.max(P[i][x]) / 2 - epsilon))[0]
        width_z_hard = np.vectorize(F_hard, otypes=[np.ndarray])(z) * step + t1
        asw_hard.append(width_z_hard)
    ID = np.argmax(highestPowEvo[0])
    epsilon = np.min(np.fabs(P_single[ID] - np.max(P_single[ID]) / 2)) * 0.25  # здесь 0.2 - эмпирический коэффициент
    asw_hard_single = []
    asw_simple_single = []
    F_simple_single = lambda x: np.fabs(P_single[x] - max(P_single[x]) / 2)[:M // 2].argmin()
    coords_simple = np.vectorize(F_simple_single)(z)
    asw_simple_single = coords_simple * step + t1
    F_hard_single = lambda x: \
        np.where(
            (P_single[x] <= np.max(P_single[x]) / 2 + epsilon) & (P_single[x] >= np.max(P_single[x]) / 2 - epsilon))[0]
    asw_hard_single = np.vectorize(F_hard_single, otypes=[np.ndarray])(z) * step + t1
    arg = np.linspace(z1, z2, N)
    fig = plt.figure(figsize=(18, 20))
    axs = fig.subplots(4, 2)
    for j in range(n):
        r = 0
        m = j
        if j > 2:
            r = 1
            m -= 3
        else:
            m += 1
        ax = axs[m, r]
        ax.set_title('slice at half height, {} fiberprop'.format(j))
        ax.set_xlabel('time, fs')
        ax.set_ylabel('z, cm')
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
        ax.grid(which='major', color='k')
        ax.grid(which='minor', color = 'gray', linestyle = ':')
        ax.set_xlim(left=t1 * 2 / 3, right=-t1 * 2 / 3)
        ax.set_ylim(bottom=z1, top=z2)
        indexes = np.arange(0, N, 3)
        for k in indexes:
            if k == indexes[-1]:
                ax.scatter(asw_hard[j][k], [arg[k]] * len(asw_hard[j][k]), s=5, c='r')
                ax.scatter(asw_simple[j][k], arg[k], s=5, color='r')
                ax.scatter((-1) * asw_simple[j][k], arg[k], s=5, color='r',
                           label='slice, {} fiberprop'.format(j))  #############
                ax.scatter(asw_hard_single[k], [arg[k]] * len(asw_hard_single[k]), s=5, color='gray')
                ax.scatter(asw_simple_single[k], arg[k], s=5, color='gray')
                ax.scatter((-1) * asw_simple_single[k], arg[k], s=5, color='gray', label='slice, single fiberprop')
                continue
            ax.scatter(asw_hard[j][k], [arg[k]] * len(asw_hard[j][k]), s=5, c='r')
            ax.scatter(asw_simple[j][k], arg[k], s=5, color='r')
            ax.scatter((-1) * asw_simple[j][k], arg[k], s=5, color='r')  ###################
            ax.scatter(asw_hard_single[k], [arg[k]] * len(asw_hard_single[k]), s=5, color='gray')
            ax.scatter(asw_simple_single[k], arg[k], s=5, color='gray')
            ax.scatter((-1) * asw_simple_single[k], arg[k], s=5, color='gray')
        ax.legend(fontsize='12', facecolor='white', loc='lower left')
    axs[0, 0].set_visible(False)
    name = 'ширина_на_полувысоте'
    path = dirName + '/' + name
    fig.savefig('{}.png'.format(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('HalfPeakWidth: \tDone')
    return asw_hard, asw_simple


def PowerDistribution(dirName, A, N, M, num, t_step, t1, t2, z1, z2, outZ, config):
    """ Функция демонстрирует распределение средних мощностей по сердцевинам """
    all_z = np.linspace(z1, z2, N)
    outId = np.argmin(abs(all_z - outZ))
    actArg = all_z[outId]
    P = (abs(A) ** 2) * 1e-3  # множитель только для того, чтобы числа не были слишком большими

    fig, ax = plt.subplots(figsize=(6, 6))
    title = f'power_distribution_at_{actArg}_cm'
    ax.set_title(title)
    ax.set_ylim(bottom=-2, top=2)
    ax.set_xlim(left=-2, right=2)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    R = 1
    rad = 0.25
    points = []
    texts = []
    sumPow = 0.0
    for i in range(num):
        energy = get_simpson_integral(P[i][outId], t_step)
        meanPower = energy/(t2-t1)
        sumPow += meanPower
    if config == 'ring':
        for i in range(num):
            phi = 2 * np.pi * i / (num)
            point = (-R * np.cos(phi), R * np.sin(phi))
            points.append(point)
            energy = get_simpson_integral(P[i][outId], t_step)
            meanPower = energy / (t2 - t1)
            value = 100 * meanPower / sumPow  # в процентах
            text = f'{i} fiberprop:\n{round(value, 1)}%'
            texts.append(text)
    elif config == 'ring_with_central':
        point = (0, 0)
        points.append(point)
        energy = get_simpson_integral(P[0][outId], t_step)
        meanPower = energy / (t2 - t1)
        value = 100 * meanPower / sumPow  # в процентах
        text = f'0 fiberprop:\n{round(value, 1)}%'
        texts.append(text)
        for i in range(1, num):
            phi = 2 * np.pi * (i - 1) / (num - 1)
            point = (-R * np.cos(phi), R * np.sin(phi))
            points.append(point)
            energy = get_simpson_integral(P[i][outId], t_step)
            meanPower = energy / (t2 - t1)
            value = 100 * meanPower / sumPow  # в процентах
            text = f'{i} fiberprop:\n{round(value, 1)}%'
            texts.append(text)
    else:
        raise ValueError('Incorrect fiber type')

    for i in range(num):
        point = points[i]
        text = texts[i]
        ax.add_patch(plt.Circle(point, rad, color='yellow'))
        ax.text(point[0], point[1], text, fontsize=15)
    path = dirName + '/' + title
    fig.savefig('{}.png'.format(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('PowerDistribution: \tDone')
    return


def GetACF(dirName, A, core_id, N, M, step, t1, t2):
    """ Строит автокорреляционную функцию """
    funcArr = abs(A[core_id][N - 1]) ** 2
    scale = 1 / np.sqrt(step * sum(funcArr ** 2))
    workFunc = funcArr * scale
    acf = np.zeros(M - 1, dtype=float)
    for i in range(M // 2):
        acf[i] = step * sum(workFunc[(M // 2 - i):] * workFunc[:-(M // 2 - i)])
    acf[M // 2] = step * sum(workFunc ** 2)
    for j in range(M // 2 + 1, M - 1):
        acf[j] = step * sum(workFunc[(j - M // 2):] * workFunc[:-(j - M // 2)])

    HP_id = np.argmin(np.fabs(acf[:M // 2] - 0.5))
    t = np.linspace(t1, t2, M)[:-1]
    HP_coord = -1*t[HP_id]
    width = 2 * HP_coord

    fig, ax = plt.subplots(figsize=(12, 6))
    title = f'outputACF,_{core_id}_core'
    ax.set_title(title)
    ax.set_ylabel('Intensity, a.u.')
    ax.set_xlabel('Time, fs')
    ax.set_ylim(bottom=0, top=1.05)
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.grid(which='major', color='k')
    ax.grid(which='minor', color='gray', linestyle=':')

    lb = t1
    rb = t2
    lb_id = np.argmin(abs(t - lb))
    rb_id = np.argmin(abs(t - rb))
    ax.set_xlim(left=lb, right=rb)
    arg = t[lb_id:rb_id]
    ax.plot(arg, acf[lb_id:rb_id], linewidth=3, color='blue')
    ax.vlines(-HP_coord, 0, 0.5, linewidth=1, color='red')
    ax.vlines(HP_coord, 0, 0.5, linewidth=1, color='red')
    ax.hlines(acf[HP_id], -HP_coord, HP_coord, linewidth=1, color='red')
    text1 = f'{round(width, 2)} fs'
    ax.text(-200, acf[HP_id], text1, fontsize=15)
    text2 = f'approximate duration: {round(0.65 * width, 2)} fs'
    ax.text(lb+100, 0.9, text2, fontsize=15)

    path = dirName + '/' + title
    fig.savefig('{}.png'.format(path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('GetACF: \tDone')
    return acf[lb_id:rb_id]


