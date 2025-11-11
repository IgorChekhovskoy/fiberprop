import math
import numpy as np
import scipy.integrate as si
import scipy.special as sp
import matplotlib.pyplot as plt

from fiberprop.solver import CoreConfig

# Корни функции Бесселя. Первый индекс - порядок функции Бесселя (J0, J1, J2 или J3).
# Второй - номер корня минус один.
BESSEL_ROOTS = np.array([
    [2.404825557695773, 5.520078110286311, 8.653727912911013, 11.791534439014281],
    [3.831705970207512, 7.015586669815618, 10.173468135062723, 13.323691936314223],
    [5.135622301840682, 8.417244140399866, 11.619841172149059, 14.795951782351260],
    [6.380161895923983, 9.761023129981670, 13.015200721698433, 16.223466160318768]
], dtype=float)


def scipy_double_integral_by_circle(R, eps, fiber, light, core_center_coords, core_indexes, func):
    """Вычисляет двойной интеграл по круговой области в полярных координатах."""
    import math
    import numpy as np
    from numpy.polynomial.legendre import leggauss
    from scipy.integrate import nquad  # fallback для гладких случаев

    a = fiber.core_radius  # радиус сердцевины

    # ── Ветвь со связностью (кусочно-определённый интегранд): сумма по дискам j≠m в локальных полярных координатах
    if func is int_integral:
        m = core_indexes[0]

        # БЫЛО: Nphi ~ 1/eps → миллионы узлов и падение по памяти.
        # СДЕЛАЛОСЬ: логарифм. рост + разумные верхние ограничения.
        t = max(1.0, -math.log10(max(eps, 1e-15)))   #  ~ число верных знаков
        Nphi = int(np.clip(48 * t, 96, 2048))        #  96 … 2048, растёт ~ O(log(1/eps))
        Nr   = int(np.clip(24 * math.sqrt(t), 48, 512))  # 48 … 512, помедленнее по радиусу

        # Узлы/веса Гаусса–Лежандра на [-1,1] и аффинные преобразования
        t_phi, w_phi = leggauss(Nphi)
        t_r,   w_r   = leggauss(Nr)

        phi_nodes   = math.pi * (t_phi + 1.0)   # φ ∈ [0, 2π]
        phi_weights = math.pi * w_phi
        rho_nodes   = 0.5 * a * (t_r + 1.0)    # ρ ∈ [0, a]
        rho_weights = 0.5 * a * w_r

        def integrate_over_one_disk(xc, yc):
            Phi, Rho = np.meshgrid(phi_nodes, rho_nodes, indexing='xy')
            X = xc + Rho * np.cos(Phi)
            Y = yc + Rho * np.sin(Phi)
            # исходный интегранд в декартовых + якобиан локальных полярных (× ρ)
            vals = np.vectorize(
                lambda xx, yy: func(fiber, light, core_center_coords, core_indexes, xx, yy)
            )(X, Y) * Rho
            return (vals * rho_weights[:, None] * phi_weights[None, :]).sum()

        res = 0.0
        for j, (xc, yc) in enumerate(core_center_coords):
            if j == m:
                continue  # в собственном диске Δn^2 = 0
            res += integrate_over_one_disk(xc, yc)

        # Оценка погрешности: та же формула, но на более редкой сетке (быстро и без аллокаций чудовищного размера)
        if Nr > 24 and Nphi > 96:
            Nr2   = max(24, Nr // 2)
            Nphi2 = max(96, Nphi // 2)
            t_phi2, w_phi2 = leggauss(Nphi2)
            t_r2,   w_r2   = leggauss(Nr2)
            phi_nodes2   = math.pi * (t_phi2 + 1.0)
            phi_weights2 = math.pi * w_phi2
            rho_nodes2   = 0.5 * a * (t_r2 + 1.0)
            rho_weights2 = 0.5 * a * w_r2

            def integrate_over_one_disk_coarse(xc, yc):
                Phi, Rho = np.meshgrid(phi_nodes2, rho_nodes2, indexing='xy')
                X = xc + Rho * np.cos(Phi)
                Y = yc + Rho * np.sin(Phi)
                vals = np.vectorize(
                    lambda xx, yy: func(fiber, light, core_center_coords, core_indexes, xx, yy)
                )(X, Y) * Rho
                return (vals * rho_weights2[:, None] * phi_weights2[None, :]).sum()

            res2 = 0.0
            for j, (xc, yc) in enumerate(core_center_coords):
                if j == m:
                    continue
                res2 += integrate_over_one_disk_coarse(xc, yc)
            err = abs(res - res2)
        else:
            err = 0.0

        return (float(res), float(err))

    # ── Гладкие случаи (например, IntF2/IntF4): оставляем nquad + breakpoints
    def polar_integrand(r, theta):
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        return func(fiber, light, core_center_coords, core_indexes, x, y) * r

    def r_limits(theta):
        return (0.0, R)

    def r_breakpoints(theta):
        pts = []
        ct, st = math.cos(theta), math.sin(theta)
        for (xc, yc) in core_center_coords:
            c = xc * ct + yc * st
            disc = c * c - (xc * xc + yc * yc - a * a)
            if disc >= 0.0:
                s = math.sqrt(disc)
                r1 = c - s
                r2 = c + s
                if 0.0 < r1 < R:
                    pts.append(r1)
                if 0.0 < r2 < R:
                    pts.append(r2)
        return {'epsabs': eps, 'epsrel': eps, 'limit': 1500, 'points': pts}

    theta_opts = {'epsabs': eps, 'epsrel': eps, 'limit': 1500}

    integral, error = nquad(
        lambda r, theta: polar_integrand(r, theta),
        ranges=[lambda theta: r_limits(theta), (0.0, 2.0 * math.pi)],
        opts=[r_breakpoints, theta_opts]
    )

    return (integral, error)



def int_f2(fiber, light, core_center_coords, core_indexes, x, y):
    """ Интеграл от квадрата моды """
    x0, y0 = core_center_coords[core_indexes[0]]
    R = fiber.cladding_diameter * 0.5
    temp = 0.0
    if x**2 + y**2 <= R**2:
        temp = get_lp_mode(0, 1, fiber, light, x - x0, y - y0)**2
    return temp


def int_f4(fiber, light, core_center_coords, core_indexes, x, y):
    """ Интеграл от моды в четвёртой степени """
    x0, y0 = core_center_coords[core_indexes[0]]
    R = fiber.cladding_diameter * 0.5
    temp = 0.0
    if x**2 + y**2 <= R**2:
        temp = get_lp_mode(0, 1, fiber, light, x - x0, y - y0)**4
    return temp


def n_mode(fiber, core_center_coords, core_indexes, x, y):
    """ Коэффициент в интеграле, описывающем связь между двумя сердцевинами """
    c_diam = fiber.cladding_diameter
    if x**2 + y**2 < c_diam**2:
        R = fiber.core_radius
        for i, (x0, y0) in enumerate(core_center_coords):
            if (x - x0)**2 + (y - y0)**2 < R**2 and i != core_indexes[0]:
                return fiber.n_core**2 - fiber.n_cladding**2
        return 0.0
    return 0.0


def int_integral(fiber, light, core_center_coords, core_indexes, x, y):
    """ Интеграл, описывающий связь между двумя сердцевинами """
    x0, y0 = core_center_coords[core_indexes[0]]
    x1, y1 = core_center_coords[core_indexes[1]]
    R = fiber.cladding_diameter * 0.5
    temp = 0.0
    if x**2 + y**2 <= R**2:
        cc = n_mode(fiber, core_center_coords, core_indexes, x, y)
        temp += cc * get_lp_mode(0, 1, fiber, light, x - x0, y - y0) * get_lp_mode(0, 1, fiber, light, x - x1, y - y1)
    return temp


def plot_core_centers(core_center_coords, core_radius, cladding_diameter,
                      title='Fiber scheme', color='red',
                      annotate_indices=False, scale_bar_um=None,
                      save_path=None, show=True):
    """
    Функция для отрисовки центров ядер на плоскости (в мкм) с учетом их радиуса.

    Входные параметры:
      - core_center_coords: список кортежей (x, y) [мкм]
      - core_radius: радиус сердцевины [мкм]
      - cladding_diameter: диаметр оболочки [мкм]
      - title: заголовок графика
      - color: цвет заливки сердцевин
      - annotate_indices: нумеровать ли ядра (по порядку в списке)
      - scale_bar_um: длина масштабной линейки в мкм (None → не рисовать)
      - save_path: путь для сохранения (None → не сохранять)
      - show: показывать ли окно с графиком (False полезно на кластере)
    Возвращает:
      (fig, ax)
    """

    if show or save_path:
        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')

        # Рисуем окружность, обозначающую границу волокна (оболочка)
        R_fiber = cladding_diameter * 0.5
        lw = plt.rcParams.get("lines.linewidth", 1.0)
        fiber_circle = plt.Circle((0.0, 0.0), R_fiber,
                                  facecolor='none', edgecolor='black',
                                  linestyle='-', linewidth=lw)
        ax.add_patch(fiber_circle)

        # Рисуем круги сердцевин
        edgecolor = 'black'
        for idx, (x, y) in enumerate(core_center_coords):
            circle = plt.Circle((x, y), core_radius,
                                facecolor=color, edgecolor=edgecolor,
                                linewidth=lw, alpha=0.65)
            ax.add_patch(circle)
            if annotate_indices:
                ax.text(x, y, str(idx+1),
                        ha='center', va='center',
                        fontsize=plt.rcParams.get("font.size", 10) * 0.8)

        # Границы с небольшим полем
        pad = 0.08 * cladding_diameter
        limit = R_fiber + core_radius + pad
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)

        # Подписи и оформление (шрифты/размеры — из rcParams/.mplstyle)
        ax.set_xlabel('X [µm]')
        ax.set_ylabel('Y [µm]')
        ax.set_title(title)
        ax.grid(False)

        # Простая масштабная линейка (если нужна)
        if scale_bar_um and scale_bar_um > 0:
            x0 = -limit + 0.12 * (2 * limit)
            y0 = -limit + 0.10 * (2 * limit)
            ax.plot([x0, x0 + scale_bar_um], [y0, y0],
                    solid_capstyle='butt', linewidth=lw)
            ax.text(x0 + scale_bar_um * 0.5, y0 - 0.03 * (2 * limit),
                    f'{int(scale_bar_um)} µm', ha='center', va='top')

        # Сохранение — строго по твоим rcParams (dpi/bbox/transparent и т.п.)
        if save_path:
            fig.savefig(
                save_path,
                dpi=plt.rcParams.get("savefig.dpi", "figure"),
                bbox_inches=plt.rcParams.get("savefig.bbox", None),
                transparent=plt.rcParams.get("savefig.transparent", False),
                pad_inches=plt.rcParams.get("savefig.pad_inches", 0.1),
            )

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax


def get_coupling_coefficients(fiber, light, eps=1e-3,
                              display_debug_plots=False,
                              save_debug_plot_path=None,
                              auto_expand_R=False,
                              auto_tol=1e-6,
                              plotly_debug=False):
    """
    Вычисляет матрицу коэффициентов связи между сердцевинами многожильного оптического волокна.

    Параметры:
        fiber (Fiber): Объект волокна с заданными параметрами.
        light (Light): Параметры излучения (длина волны и др.).
        eps (float, optional): Точность вычисления интегралов. По умолчанию 1e-3.
        display_debug_plots (bool, optional): Если True, отображает график расположения сердцевин.
        save_debug_plot_path (str | None): Базовый путь для сохранения отладочных графиков (если указан).
        auto_expand_R (bool): Автоматически увеличить радиус интегрирования R до сходимости по диагонали.
        auto_tol (float): Допуск по сходимости диагонального интеграла при авто-расширении R.
        plotly_debug (bool): Рисовать Plotly-графики интегрирования для каждой пары ядер.

    Возвращает:
        numpy.ndarray: Матрица коэффициентов связи [1/m].

    Поддерживаемые конфигурации волокна:
        - Пустое кольцо (empty_ring)
        - Шестиугольная решетка (hexagonal)
        - Двухсердцевинное волокно (dual_core)

    Примечания:
        - Коэффициенты связи вычисляются на основе численного интегрирования.
        - Используется кэширование значений интегралов для ускорения расчетов.
        - Коэффициенты нормируются на 1e+2 для удобства представления.
    """

    # -----------------------------------
    # Геометрия сердцевин (как было)
    # -----------------------------------
    R = 0.5 * fiber.cladding_diameter
    core_count = fiber.core_count
    distance_to_fiber_center = fiber.distance_to_fiber_center
    core_center_coords = []

    if fiber.core_configuration is CoreConfig.empty_ring:
        for i in range(core_count):
            phi = 2.0 * math.pi * i / core_count
            coords = (distance_to_fiber_center * math.cos(phi), distance_to_fiber_center * math.sin(phi))
            core_center_coords.append(coords)
    elif fiber.core_configuration is CoreConfig.hexagonal:
        for i in range(core_count):
            # Вычисляем двумерный радиус для определения кольца
            dimensional_radius = np.sqrt(
                (fiber.mask_array[i].number_2d_x * 0.5) ** 2 +
                (fiber.mask_array[i].number_2d_y * 0.5 * np.sqrt(3)) ** 2
            )
            ring_index = int(np.ceil(dimensional_radius))
            x_coord = distance_to_fiber_center[ring_index] * fiber.mask_array[i].number_2d_x * 0.5 / max(ring_index, 1)
            y_coord = distance_to_fiber_center[ring_index] * fiber.mask_array[i].number_2d_y * 0.5 * np.sqrt(3) / max(ring_index, 1)
            core_center_coords.append((x_coord, y_coord))
    elif fiber.core_configuration is CoreConfig.dual_core:
        coup_mat, _ = get_coupling_coeff_2_core_fiber(fiber, light)
        return coup_mat
    else:
        raise ValueError('This fiber configuration is not yet supported')

    # --- Авто-подстройка оболочки: если край какого-либо ядра вылезает за текущий R, увеличиваем диаметр с запасом.
    # Берём новый радиус оболочки R_new = d_max + 3 * core_radius (запас = три радиуса ядра).
    if core_center_coords:
        d_max = max(math.hypot(x, y) for (x, y) in core_center_coords)
        R_core = fiber.core_radius
        R_curr = 0.5 * fiber.cladding_diameter
        if d_max + R_core > R_curr:
            R_new = d_max + 3.0 * R_core
            fiber.cladding_diameter = 2.0 * R_new

    # После возможной подстройки оболочки обновляем R
    R = 0.5 * fiber.cladding_diameter

    # --- Дополнительный авто-подбор R по хвосту K0 и проверке сходимости (минимальные вставки)
    if auto_expand_R:
        a = fiber.core_radius
        # Параметры LP01: v^2 = u^2 + w^2, в оболочке хвост ~ K0(w r / a)
        v = a * light.k0 * fiber.NA
        u = (1.0 + 2.0**0.5) * v / (1.0 + (4.0 + v**4)**0.25)
        w = max(1e-9, (v**2 - u**2)**0.5)
        # при r>>a: K0(x) ~ sqrt(pi/(2x)) e^{-x}  → выбираем R с запасом по экспоненте
        if core_center_coords:
            d_max = max(math.hypot(x, y) for (x, y) in core_center_coords)
        else:
            d_max = 0.0
        R_need = d_max + a + (a / (2.0 * w)) * math.log(max(1.0/auto_tol, 1.0))
        if R < R_need:
            fiber.cladding_diameter = 2.0 * R_need
            R = 0.5 * fiber.cladding_diameter

        # Быстрая проверка сходимости диагонального интеграла (1D)
        samples_chk = max(256, int(1/eps)+1)
        def _diag_at(Ru):
            old = fiber.cladding_diameter
            fiber.cladding_diameter = 2.0 * Ru
            val = get_lp_mode_radial_integral(2, fiber, light, samples_chk)
            fiber.cladding_diameter = old
            return val
        diag_R = _diag_at(R)
        diag_R2 = _diag_at(1.2 * R)
        if abs(diag_R2 - diag_R) / max(abs(diag_R2), 1e-300) > auto_tol:
            fiber.cladding_diameter = 2.0 * (1.5 * R)
            R = 0.5 * fiber.cladding_diameter

    plot_core_centers(core_center_coords, fiber.core_radius, fiber.cladding_diameter,
                      title='Fiber scheme', color='red',
                      annotate_indices=False, scale_bar_um=None,
                      save_path=save_debug_plot_path, show=display_debug_plots)

    # -----------------------------------
    # Нормировка (как было)
    # -----------------------------------
    samples = int(1 / eps) + 1  # ≈ та же точность, но без dblquad
    diag_val = get_lp_mode_radial_integral(2, fiber, light, samples)

    k_prefactor = 0.5 * (light.k0 ** 2) / fiber.get_beta(light)

    # ──────────────────────────────────────────────────────────────────────
    # БЛОК УСКОРЕНИЯ: предварительный Ханкель-просчёт для LP01 (ũ и ĝ)
    # ──────────────────────────────────────────────────────────────────────
    r = np.linspace(0.0, R, samples)
    u_r = np.array([get_lp_mode(0, 1, fiber, light, ri, 0.0) for ri in r])

    k_max = 20.0 / fiber.core_radius  # верхняя граница достаточно высокая
    k_arr = np.linspace(0.0, k_max, samples)  # равномерная сетка в k-пространстве
    J_mat = sp.j0(np.outer(k_arr, r))  # матрица Бесселя J0(k·r)

    u_tilde = np.trapz(u_r * r * J_mat, r, axis=1)  # û(k)

    mask = r <= fiber.core_radius  # только область ядра
    g_r = u_r * mask
    g_tilde = np.trapz(g_r * r * J_mat, r, axis=1)  # ĝ(k)

    delta_n2 = fiber.n_core ** 2 - fiber.n_cladding ** 2

    # Инициализация матриц для коэффициентов связи и ошибок
    coup_mat = np.zeros((core_count, core_count), dtype=float)

    # Словарь для кэширования результатов off-diagonal интегралов по расстоянию
    cache = {}

    # -----------------------------------
    # Вычисление q_mp и (опционально) Plotly-визуализация
    # -----------------------------------
    for m in range(core_count):
        for p in range(m):
            # Расчет расстояния между центрами ядер m и p
            dx = core_center_coords[m][0] - core_center_coords[p][0]
            dy = core_center_coords[m][1] - core_center_coords[p][1]
            d = math.sqrt(dx * dx + dy * dy)
            d_key = round(d, 3)
            if d_key not in cache:
                integrand = k_arr * u_tilde * g_tilde * sp.j0(k_arr * d)
                int2 = 2.0 * math.pi * delta_n2 * np.trapz(integrand, k_arr)
                cache[d_key] = int2
            else:
                int2 = cache[d_key]

            qmp = k_prefactor * int2 / diag_val
            coupling = qmp * 1e+4
            coup_mat[m][p] = coupling
            coup_mat[p][m] = coupling

            # --- Plotly debug (1D сечение y=0): u_m(x), u_p(x), ядра, оболочка, область интегрирования
            if plotly_debug:
                try:
                    import plotly.graph_objects as go
                    x = np.linspace(-R, R, 1200)
                    um = np.array([get_lp_mode(0, 1, fiber, light, xk - core_center_coords[m][0], 0.0) for xk in x])
                    up = np.array([get_lp_mode(0, 1, fiber, light, xk - core_center_coords[p][0], 0.0) for xk in x])

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=um, name=f"u_m (m={m})", mode="lines"))
                    fig.add_trace(go.Scatter(x=x, y=up, name=f"u_p (p={p})", mode="lines"))

                    # Вертикальные линии оболочки
                    fig.add_vline(x=-R, line=dict(dash="dash"), annotation_text="-R")
                    fig.add_vline(x= R, line=dict(dash="dash"), annotation_text="R")

                    # Центры ядер и границы их радиусов
                    for idx, (xc, yc) in enumerate(core_center_coords):
                        fig.add_vline(x=xc, line=dict(width=1), annotation_text=f"core {idx}")
                        fig.add_vrect(x0=xc - fiber.core_radius, x1=xc + fiber.core_radius,
                                      fillcolor="rgba(255,0,0,0.08)" if idx==p else "rgba(0,0,255,0.06)",
                                      line_width=0)

                    # Заголовок
                    fig.update_layout(
                        title=f"1D integration view (pair m={m}, p={p}, d={d:.2f} µm) — R={R:.1f} µm",
                        xaxis_title="x [µm]  (y=0)",
                        yaxis_title="Field amplitude",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0)
                    )

                    # Сохранение, если задан путь
                    if save_debug_plot_path:
                        base = f"{save_debug_plot_path}_1d_m{m}_p{p}"
                        try:
                            fig.write_image(base + ".pdf")
                            # svg, если есть kaleido
                            # fig.write_image(base + ".svg")
                        except Exception:
                            fig.write_html(base + ".html")
                except Exception:
                    # тихо пропускаем, если plotly не установлен
                    pass

    return np.abs(coup_mat) * 1e+2


def get_coupling_coefficients_2d(fiber, light, eps=1e-3,
                                 display_debug_plots=False,
                                 save_debug_plot_path=None,
                                 auto_expand_R=False,
                                 auto_tol=1e-6,
                                 plotly_debug=False):
    """
    ДВУМЕРНЫЙ расчёт матрицы коэффициентов связи между сердцевинами многожильного оптического волокна
    через прямой интеграл по области (полярный интеграл).

    Параметры:
        fiber (Fiber): Объект волокна с заданными параметрами.
        light (Light): Параметры излучения (длина волны и др.).
        eps (float, optional): Точность численных интегралов (epsabs=epsrel=eps). По умолчанию 1e-3.
        display_debug_plots (bool, optional): Если True, отображает схему расположения сердцевин и границ.
        save_debug_plot_path (str | None): Куда сохранить схему, если нужно (по умолчанию None).
        auto_expand_R (bool): Автоматически увеличить радиус интегрирования R до сходимости по диагонали.
        auto_tol (float): Допуск по сходимости диагонального интеграла при авто-расширении R.
        plotly_debug (bool): Рисовать Plotly-графики интегрирования для каждой пары ядер.

    Возвращает:
        tuple:
            - numpy.ndarray: Матрица коэффициентов связи [1/m].
            - numpy.ndarray: Матрица абсолютных ошибок [1/m].
    """
    import math
    import numpy as np

    R = 0.5 * fiber.cladding_diameter
    core_count = fiber.core_count
    distance_to_fiber_center = fiber.distance_to_fiber_center
    core_center_coords = []

    from fiberprop.solver import CoreConfig
    if fiber.core_configuration is CoreConfig.empty_ring:
        for i in range(core_count):
            phi = 2.0 * math.pi * i / core_count
            coords = (distance_to_fiber_center * math.cos(phi), distance_to_fiber_center * math.sin(phi))
            core_center_coords.append(coords)
    elif fiber.core_configuration is CoreConfig.hexagonal:
        for i in range(core_count):
            dimensional_radius = np.sqrt(
                (fiber.mask_array[i].number_2d_x * 0.5) ** 2 +
                (fiber.mask_array[i].number_2d_y * 0.5 * np.sqrt(3)) ** 2
            )
            ring_index = int(np.ceil(dimensional_radius))
            x_coord = distance_to_fiber_center[ring_index] * fiber.mask_array[i].number_2d_x * 0.5 / max(ring_index, 1)
            y_coord = distance_to_fiber_center[ring_index] * fiber.mask_array[i].number_2d_y * 0.5 * np.sqrt(3) / max(ring_index, 1)
            core_center_coords.append((x_coord, y_coord))
    elif fiber.core_configuration is CoreConfig.dual_core:
        return get_coupling_coeff_2_core_fiber(fiber, light)
    else:
        raise ValueError('This fiber configuration is not yet supported')

    # Автоподстройка R (как было)
    if core_center_coords:
        d_max = max(math.hypot(x, y) for (x, y) in core_center_coords)
        R_core = fiber.core_radius
        R_curr = 0.5 * fiber.cladding_diameter
        if d_max + R_core > R_curr:
            R_new = d_max + 3.0 * R_core
            fiber.cladding_diameter = 2.0 * R_new
    R = 0.5 * fiber.cladding_diameter

    # (опционально) авторасширение R по хвосту K0 и проверке сходимости
    if auto_expand_R:
        a = fiber.core_radius
        v = a * light.k0 * fiber.NA
        u = (1.0 + 2.0**0.5) * v / (1.0 + (4.0 + v**4)**0.25)
        w = max(1e-9, (v**2 - u**2)**0.5)
        d_max = max(math.hypot(x, y) for (x, y) in core_center_coords) if core_center_coords else 0.0
        R_need = d_max + a + (a / (2.0 * w)) * math.log(max(1.0/auto_tol, 1.0))
        if R < R_need:
            fiber.cladding_diameter = 2.0 * R_need
            R = 0.5 * fiber.cladding_diameter
        # простая проверка по диагонали
        diag_R, _  = scipy_double_integral_by_circle(R,     eps, fiber, light, core_center_coords, (0, 0), int_f2)
        diag_R2, _ = scipy_double_integral_by_circle(1.2*R, eps, fiber, light, core_center_coords, (0, 0), int_f2)
        if abs(diag_R2 - diag_R) / max(abs(diag_R2), 1e-300) > auto_tol:
            fiber.cladding_diameter = 2.0 * (1.5 * R)
            R = 0.5 * fiber.cladding_diameter

    plot_core_centers(core_center_coords, fiber.core_radius, fiber.cladding_diameter,
                      title='Fiber scheme', color='red',
                      annotate_indices=False, scale_bar_um=None,
                      save_path=save_debug_plot_path, show=display_debug_plots)

    # Диагональный интеграл (⟨u,u⟩)
    diag_integral, diag_err = scipy_double_integral_by_circle(
        R, eps, fiber, light, core_center_coords, (0, 0), int_f2
    )
    diag_val = diag_integral
    diag_up  = diag_integral + diag_err
    diag_low = max(1e-300, diag_integral - diag_err)

    # Префактор (как было)
    k_prefactor = 0.5 * (light.k0 ** 2) / fiber.get_beta(light)

    coup_mat  = np.zeros((core_count, core_count), dtype=float)
    error_mat = np.zeros((core_count, core_count), dtype=float)
    cache = {}

    for m in range(core_count):
        for p in range(m):
            dx = core_center_coords[m][0] - core_center_coords[p][0]
            dy = core_center_coords[m][1] - core_center_coords[p][1]
            d = math.hypot(dx, dy)
            d_key = round(d, 6)

            if d_key not in cache:
                int_val, int_err = scipy_double_integral_by_circle(
                    R, eps, fiber, light, core_center_coords, (m, p), int_integral
                )
                cache[d_key] = (int_val, int_err)
            else:
                int_val, int_err = cache[d_key]

            int2     = int_val
            int2_up  = int_val + int_err
            int2_low = int_val - int_err

            qmp = k_prefactor * int2 / (diag_val ** 0.5 * diag_val ** 0.5)
            coupling = qmp * 1e+4
            coup_mat[m][p] = coup_mat[p][m] = coupling

            full_err_up  = k_prefactor * int2_up  / (diag_low ** 0.5 * diag_low ** 0.5)
            full_err_low = k_prefactor * int2_low / (diag_up  ** 0.5 * diag_up  ** 0.5)
            error = abs(full_err_up - full_err_low) * 0.5 * 1e+4
            error_mat[m][p] = error_mat[p][m] = error

    return np.abs(coup_mat) * 1e+2, error_mat * 1e+2



def get_coupling_coeff_2_core_fiber(fiber, light):
    """ Коэффициент связи для двухсердцевинного волокна """

    fiber.delta_n_core = fiber.n_core - fiber.n_cladding

    V = fiber.core_radius * light.k0 * (
        ((1.0 + fiber.delta_n_core) * fiber.n_cladding)**2 - fiber.n_cladding**2)**0.5

    c0 = 5.2789 - 3.663 * V + 0.3841 * V**2
    c1 = -0.7769 + 1.2252 * V - 0.0152 * V**2
    c2 = -0.0175 - 0.0064 * V - 0.0009 * V**2

    d = 2.0 * fiber.distance_to_fiber_center[0] / fiber.core_radius

    coup_mat = np.zeros((2, 2), dtype=float)
    coup_mat[0][1] = math.pi * V * math.exp(-(c0 + c1 * d + c2 * d**2)) / (
        2.0 * light.k0 * fiber.n_cladding * fiber.core_radius**2)
    coup_mat[1][0] = coup_mat[0][1]

    error_mat = np.zeros((2, 2), dtype=float)
    return coup_mat, error_mat


def get_lp_mode(l, m, fiber, light, x, y):
    r = (x**2 + y**2)**0.5
    phi = np.arctan2(y, x)

    v = fiber.core_radius * light.k0 * fiber.NA

    if l == 0 and m == 1:
        u = (1.0 + 2.0**0.5) * v / (1.0 + (4.0 + v**4)**0.25)
        w = (v**2 - u**2)**0.5
    else:
        if l >= 0 and m >= 1:
            uc = BESSEL_ROOTS[int(abs(l - 1))][m - 1]
            if uc > v:
                print(f"Mode LP{l}{m} for Core Radius = {fiber.core_radius} mkm: ERROR: This mode is not allowed for this fiber geometry!")
                return 1

            s = (uc**2 - l**2 - 1)**0.5
            u = uc * math.exp((math.asin(s / uc) - math.asin(s / v)) / s)
            w = (v**2 - u**2)**0.5
        else:
            print("ERROR: Such LP mode does not exist!")
            return 2

    core_radius = fiber.core_radius
    if r < core_radius:
        return sp.jv(l, u * r / core_radius) / sp.jv(l, u) * math.cos(l * phi)
    else:
        return sp.kn(l, w * r / core_radius) / sp.kn(l, w) * math.cos(l * phi)

def get_lp_mode_radial_integral(power, fiber, light, samples):
    """
    2π ∫ |LP01(r)|**power · r dr            ← 1-D trapz
    • power = 2 → IF2   (нормировка)
      power = 4 → IF4   (A_eff, gamma   – если понадобится)
    """
    R = 0.5 * fiber.cladding_diameter      # [µm] ограничимся оболочкой
    r = np.linspace(0.0, R, samples)
    u = np.array([get_lp_mode(0, 1, fiber, light, ri, 0.0) for ri in r])   # scalar call
    return 2 * np.pi * np.trapz((u**power) * r, r)
