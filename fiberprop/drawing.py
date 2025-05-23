import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from sympy import false


def plot3D(z, t, field, name, filename=None, file_format='png', yscale='linear'):
    """ Функция выводит и сохраняет поле 3D значений """
    fig = plt.figure(figsize=(12, 8))  # график
    ax = fig.add_subplot(1, 1, 1, projection='3d')  # система координат
    ax.set_xlabel('z')
    ax.set_ylabel('t')
    ax.set_title(name)

    t_2d, z_2d = np.meshgrid(t, z)
    ax.plot_surface(z_2d, t_2d, field, cmap='inferno')

    ax.set_yscale(yscale)  # Устанавливаем масштабирование по оси y

    plt.show()
    if filename:
        fig.savefig(f'{filename}.{file_format}', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('plot3D: \tDone')
    return


def plot3D_plotly(t, z, field, name, filename=None, file_format='html', yscale='linear'):
    """ Функция строит и выводит трехмерный график с использованием Plotly """

    t_2d, z_2d = np.meshgrid(t, z)

    fig = go.Figure(data=[go.Surface(z=field, x=t_2d, y=z)])

    fig.update_layout(
        title=name,
        scene=dict(
            xaxis=dict(title='t', exponentformat='power', showgrid=True, gridcolor='gray'),
            yaxis=dict(title='z', exponentformat='power', showgrid=True, gridcolor='gray', type=yscale),
            zaxis=dict(title='U(z,t)', exponentformat='power', showgrid=True, gridcolor='gray')
        ),
        font=dict(family='Times New Roman', size=15, color='black'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    if filename:
        if file_format == 'pdf':
            fig.write_image(f"{filename}.pdf")
        elif file_format == 'png':
            fig.write_image(f"{filename}.png")
        else:
            fig.write_html(f"{filename}.html")
        print(f'plot3D_plotly: \tDone. Saved to {filename}.{file_format}')
    else:
        fig.show()
        print('plot3D_plotly: \tDone')


def plot2D(t_arrr, func_arr, name='', filename=None, file_format='png', yscale='linear'):
    """ Функция строит и выводит одномерный график """
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

    ax.set_yscale(yscale)  # Устанавливаем масштабирование по оси y

    ax.plot(t_arrr, func_arr, color='blue', linestyle='-', linewidth=2, label=name)
    ax.legend(loc=2)

    plt.show()
    plt.close(fig)
    print('plot2D: \tDone')
    return


def plot2D_plotly(x_arrr, funcs_arr, x_axis_label='t', y_axis_label='', title_text='График функций',
                  names=None, filename=None, file_format='png', yscale='linear'):
    """ Функция строит и выводит одномерный график с использованием Plotly """

    fig = go.Figure()

    # Если funcs_arr не является списком, преобразуем его в список
    if not isinstance(funcs_arr, list):
        funcs_arr = [funcs_arr]

    # Если имена не предоставлены, создаем имена по умолчанию
    if names is None:
        names = [f'Function {i+1}' for i in range(len(funcs_arr))]

    for func_arr, name in zip(funcs_arr, names):
        fig.add_trace(go.Scatter(x=x_arrr, y=func_arr,
                                 mode='lines',
                                 name=name,
                                 line=dict(
                                     width=3
                                 )))

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=20, family='Times New Roman')),
        xaxis_title=dict(text=x_axis_label, font=dict(size=18, family='Times New Roman')),
        yaxis_title=dict(text=y_axis_label, font=dict(size=18, family='Times New Roman')),
        font=dict(family='Times New Roman', size=15, color='black'),
        legend=dict(x=0, y=1, traceorder='normal', bgcolor='rgba(255,255,255,0.5)', bordercolor='Black', borderwidth=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=20, t=60, b=60),
        xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True, ticks='outside', showgrid=True,
                   gridwidth=0.5, gridcolor='gray'),
        yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True, ticks='outside', showgrid=True,
                   gridwidth=0.5, gridcolor='gray', exponentformat='power', type=yscale)
    )

    if filename:
        if file_format == 'pdf':
            fig.write_image(f"{filename}.pdf")
        elif file_format == 'png':
            fig.write_image(f"{filename}.png")
        else:
            fig.write_html(f"{filename}.html")
        print(f'plot2D_plotly: \tDone. Saved to {filename}.{file_format}')
    else:
        fig.show()
        print('plot2D_plotly: \tDone')


def plot2D_dict_ForVideo(fig, ax, x_arr, func_dict, log_scale=False, marker_flag=True):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Красный, зеленый, синий и т.д.
    markers = ['o', 's', 'D', '^', 'v', '<', '>']  # Круг, квадрат, ромб и т.д.

    if log_scale:
        for i, line_label in enumerate(func_dict):
            color, marker = colors[i], markers[i]
            marker = marker if marker_flag else ''
            ax.semilogy(x_arr, func_dict[line_label], color=color, linestyle='-', marker=marker,
                        linewidth=2, label=line_label)
    else:
        for i, line_label in enumerate(func_dict):
            color, marker = colors[i], markers[i]
            marker = marker if marker_flag else ''
            ax.plot(x_arr, func_dict[line_label], color=color, linestyle='-', marker=marker,
                    linewidth=2, label=line_label)
    ax.legend(loc=2)


def plot2D_dict(x_arr, func_dict, y_min=0, y_max=10, xlabel='x', ylabel='y', title='', log_scale=False, marker_flag=True):
    """ Функция строит и выводит набор двумерных графиков на одном рисунке по словарю """
    fig, ax = create_single_ax(xlabel, ylabel, y_min, y_max, title)
    plot2D_dict_ForVideo(fig, ax, x_arr, func_dict, log_scale, marker_flag)
    plt.show()
    plt.close(fig)
    print('plot2D_dict: \tDone')


def plot2D_multicore_ForVideo(fig, axs, x_arr, func_dict):
    for i, label in enumerate(func_dict):
        axs[i].plot(x_arr, func_dict[label], color='red', linestyle='-', label=label)
        axs[i].legend()


def plot2D_multicore(x_arr, func_dict, xlabel='', ylabel='', y_min=0, y_max=4.2):
    fig, axs = create_multicore_axs(len(func_dict), xlabel, ylabel, y_min, y_max)
    plot2D_multicore_ForVideo(fig, axs, x_arr, func_dict)
    plt.show()
    plt.close(fig)
    print('plot2D_multicore: \tDone')


def create_multicore_axs(axs_num, xlabel, ylabel, y_min, y_max):
    ncols = 2
    nrows = axs_num // ncols if (axs_num % ncols == 0) else (axs_num // ncols + 1)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, nrows * 6), frameon=True)
    plt.style.use('ggplot')
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'

    axs = axs.flatten()
    for ax in axs:
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.set(facecolor='w')
        ax.grid('axis = "both"', color='gray')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(y_min, y_max)

    if not (axs_num % ncols == 0):
        empty_num = nrows * ncols - axs_num
        for i in range(empty_num):
            fig.delaxes(axs[-1 - i])
    return fig, axs


def create_single_ax(xlabel, ylabel, y_min, y_max, title=''):
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

    ax.set_title(title)
    ax.set_xlabel(xlabel, labelpad=5)
    ax.set_ylabel(ylabel, labelpad=5)
    ax.set_ylim(y_min, y_max)
    return fig, ax
