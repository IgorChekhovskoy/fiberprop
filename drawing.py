import matplotlib.pyplot as plt


def plot3D(Z, T, Field, name, save=False):
    """ Функция выводит и сохраняет поле 3D значений """
    fig = plt.figure(figsize=(12, 8))  # график
    ax = fig.add_subplot(1, 1, 1, projection='3d')  # система координат
    ax.set_xlabel('z, m')
    ax.set_ylabel('t, fs')
    ax.set_title(name)
    ax.plot_surface(Z, T, Field, cmap='inferno')
    plt.show()
    if save:
        fig.savefig('{}.png'.format(name), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('plot3D: \tDone')
    return


def plot2D(t_arrr, func_arr, name='название не указано'):
    """ Функция строит и выводит двумерный график """
    fig = plt.figure(figsize=(9, 6), frameon=True)
    plt.style.use('ggplot')
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 15
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    ax = fig.add_subplot(111)

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')

    ax.set(facecolor='w')
    ax.grid('axis = "both"', color='gray')

    ax.set_xlabel('t, fs', labelpad=-10)

    ax.plot(t_arrr, func_arr, color='blue', linestyle='-', linewidth=2, label=name)
    ax.legend(loc=2)

    plt.show()
    plt.close(fig)
    print('plot2D: \tDone')
    return

