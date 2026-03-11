import numpy as np
from pathlib import Path
from typing import Any, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import json, hashlib

from fiberprop.fiber import CoreConfig
from fiberprop.fiber_geometry import get_core_count


MM = 1 / 25.4
COL1, COL15, COL2 = 89 * MM, 136 * MM, 183 * MM  # Nature: 1, 1.5, 2 колонки


CACHE_DIR = Path("mcf_rc_cache")


def json_dumps_compact(obj):
    """
    Компактная JSON-сериализация с безопасным приведением numpy-типов:
    - numpy scalars (np.int64, np.float64, np.bool_) -> обычные int/float/bool через .item()
    - numpy.ndarray -> list через .tolist()
    - коллекции и словари обходим рекурсивно
    """

    def _to_jsonable(x):
        # numpy скаляры → базовые типы
        if isinstance(x, np.generic):
            return x.item()
        # numpy массивы → списки
        if isinstance(x, np.ndarray):
            return x.tolist()
        # словари → рекурсивно
        if isinstance(x, dict):
            # ключи на всякий случай приводим к строке (если вдруг попались не-строки)
            return {str(k): _to_jsonable(v) for k, v in x.items()}
        # последовательности/множества → рекурсивно в список
        if isinstance(x, (list, tuple, set)):
            return [_to_jsonable(v) for v in x]
        # остальное — как есть
        return x

    return json.dumps(_to_jsonable(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_of_json(obj: Any) -> str:
    s = json_dumps_compact(obj).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    NRMSE = RMSE / std(y_true). Возвращает NaN для пустых входов без генерации ворнингов NumPy.
    Если std≈0: RMSE≈0 → 0.0, иначе → +inf.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    n = y_true.size
    if n == 0:
        return float('nan')  # валидно, и без предупреждений

    # На всякий случай подровняем длины (если расходятся).
    if y_pred.size != n:
        m = min(n, y_pred.size)
        if m == 0:
            return float('nan')
        y_true = y_true[:m]
        y_pred = y_pred[:m]

    diff = y_true - y_pred
    rmse = float(np.sqrt((diff * diff).mean()))  # безопасно: массив гарантированно непустой

    s = float(np.std(y_true))  # безопасно: массив гарантированно непустой
    if not np.isfinite(s) or s < eps:
        return 0.0 if rmse < eps else float('inf')

    return rmse / s


def learning_curve_for_result_plotly(
        result: dict,
        *,
        add_bias: bool = True,
) -> dict:
    """
    Строит лёрнинг-кривую (validation NRMSE vs S_train) на УЖЕ посчитанных состояниях,
    без пересчёта физики. Ожидает в result поля:
      - "X_train", "y_train", "X_val", "y_val".
    Внутри:
      • берёт сетку S_train (6 точек от ~10% до 100% train),
      • в каждой точке подбирает alpha риджа по лог-сетке через ОДНУ SVD,
      • возвращает список метрик + сохраняет интерактивный HTML-график (Plotly).

    Возвращает:
      {
        "curve": [{"S_train", "alpha_best", "nrmse_train", "nrmse_val"}, ...],
        "plot_saved_to": "<путь к .html>" | None
      }
    """

    # ---- входные массивы ----
    Xtr_full = result["X_train"]
    ytr_full = result["y_train"]
    Xva = result["X_val"]
    yva = result["y_val"]

    n_train = Xtr_full.shape[0]
    # Сетка S_train: 6 точек от ~10% до 100% train
    lo = max(5, n_train // 10)
    train_sizes_syms = sorted(set(np.linspace(lo, n_train, num=100, dtype=int).tolist()))

    # Сетка alpha по умолчанию
    alphas = np.logspace(-6, 2, 41)

    curve = []
    for S in train_sizes_syms:
        if S < 2:
            continue
        S_use = min(S, n_train)
        Xtr = Xtr_full[:S_use, :]
        ytr = ytr_full[:S_use, :]

        if add_bias:
            Xb_tr = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))])
            Xb_va = np.hstack([Xva, np.ones((Xva.shape[0], 1))])
        else:
            Xb_tr, Xb_va = Xtr, Xva

        # --- одна SVD на весь путь по alpha ---
        U, Svd, Vt = np.linalg.svd(Xb_tr, full_matrices=False)
        Ut_y = U.T @ ytr

        best_alpha, best_W, best_val = None, None, np.inf
        for a in alphas:
            shrink = Svd / (Svd * Svd + a)
            W = (Vt.T * shrink) @ Ut_y
            val = nrmse(yva, Xb_va @ W)
            if val < best_val:
                best_val, best_alpha, best_W = val, float(a), W
        print("best_alpha =", best_alpha)
        nrmse_tr = nrmse(ytr, Xb_tr @ best_W)

        curve.append({
            "S_train": int(S_use),
            "alpha_best": float(best_alpha),
            "nrmse_train": float(nrmse_tr),
            "nrmse_val": float(best_val),
        })

    # ---- график (Plotly) ----
    plot_saved_to = None
    try:
        import plotly.graph_objects as go

        Ss = [d["S_train"] for d in curve]
        vals = [d["nrmse_val"] for d in curve]
        trs = [d["nrmse_train"] for d in curve]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Ss, y=vals, mode="lines+markers", name="val NRMSE"))
        fig.add_trace(go.Scatter(x=Ss, y=trs, mode="lines+markers", name="train NRMSE", line=dict(dash="dash")))
        fig.update_layout(
            title="Learning curve (validation NRMSE vs S_train)",
            xaxis_title="Число обучающих символов S_train",
            yaxis_title="NRMSE",
            template="plotly_white",
            legend=dict(x=0.02, y=0.98),
        )

        # Автогенерация имени HTML рядом с экспериментом.
        # Пытаемся использовать несколько полей из result["params"], если есть
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "lc"
        try:
            p = result.get("params", {})
            ms = p.get("mask_size", "M")
            df = p.get("delay_factor_in_symbols", "D")
            up = p.get("upsampling", "U")
            tag = f"lc_m{ms}_d{df}_u{up}"
        except Exception:
            pass
        plot_saved_to = f"{tag}_{stamp}.html"
        # fig.write_html(plot_saved_to, include_plotlyjs="cdn")
        # Покажем интерактивно
        fig.show()
    except Exception as _e:
        # не роняем эксперимент из-за графика
        result["_learning_curve_plot_error"] = str(_e)

    return {"curve": curve, "plot_saved_to": plot_saved_to}


def calc_p_pump_per_core_from_slm(p_pump_total_w: float,
                                  slm_weights: Union[np.ndarray, list, tuple],
                                  *,
                                  temperature: float = 1.0,
                                  min_fraction: float = 0.0) -> tuple[float, ...]:
    """
    Делёж общей мощности накачки между ядрами через SLM как softmax(weights).

    p_i = P_total * f_i
    f = softmax(weights/temperature)

    min_fraction:
      если > 0, добавляем небольшой равномерный «пол» долей:
        f = min_fraction/C + (1-min_fraction)*softmax
      чтобы избежать p_i == 0.
    """
    if p_pump_total_w < 0:
        raise ValueError("p_pump_total_w must be >= 0")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    w = np.asarray(slm_weights, dtype=np.float64).reshape(-1)
    if w.size == 0:
        raise ValueError("slm_weights must be non-empty")

    w = (w / float(temperature)) - float(np.mean(w))
    w = w - float(np.max(w))
    exp_w = np.exp(w)
    soft = exp_w / float(np.sum(exp_w))

    if min_fraction:
        mf = float(min_fraction)
        if not (0.0 <= mf < 1.0):
            raise ValueError("min_fraction must be in [0, 1)")
        c = float(w.size)
        soft = (mf / c) + (1.0 - mf) * soft

    p = float(p_pump_total_w) * soft
    return tuple(float(x) for x in p)


def calc_g0_psat_from_p_pump_josab(p_pump_w: Union[float, np.ndarray, list, tuple],
                                   fiber_length_m: float,
                                   *,
                                   alpha: float = 0.75,  # потери, были в статье при выводе формулы (10)
                                   psat_fit_a: float = 0.1292,
                                   psat_fit_b_mw: float = -1.096,
                                   gA_A_per_m: float = -2.25e5,
                                   gA_B_per_mw_per_m: float = 2.3e4,
                                   gA_C_per_mw: float = -4.54e3,
                                   gA_D_per_m: float = 0.05) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """
    Пересчёт (g0, psat) из мощности накачки P_pump по аппроксимациям из:
      O. V. Shtyrina et al., JOSA B 34(2), 227 (2017), doi:10.1364/JOSAB.34.000227.

    Используем согласованную интерпретацию:
      - величины g0 и коэффициенты A,B,D — в 1/м (Np/m), не в dB/m;
      - в (10) в экспоненте нужен знак "+": exp(+D * L), чтобы совпадать с табл. 2.

    В статье:
      (9)  Psat[mW] = a * Ppump[mW] + b
      (10) g0[1/m]  = (A + B*Ppump[mW]) / (1 - C*Ppump[mW] * exp(+D*L))

    Возвращает:
      (g0_per_m_tuple, psat_w_tuple) длиной = числу элементов во входе (или 1 для скаляра).
    """
    p_w = np.asarray(p_pump_w, dtype=np.float64)
    if np.any(p_w < 0):
        raise ValueError("p_pump_w must be >= 0")

    L = float(fiber_length_m)
    p_mw = p_w * 1e3

    # (9): Psat in mW -> W
    psat_mw = psat_fit_a * p_mw + psat_fit_b_mw
    psat_mw[psat_mw <= 0] = 0
    # if np.any(psat_mw <= 0):
    #    raise ValueError("psat_fit gives non-positive Psat; adjust pump power bounds")
    psat_w = psat_mw * 1e-3

    # (10) with corrected sign in exponent; all gain-related coefficients are in 1/m or compatible units
    exp_term = np.exp(float(gA_D_per_m) * L)
    denom = 1.0 - float(gA_C_per_mw) * p_mw * exp_term
    if np.any(denom <= 0):
        raise ValueError("invalid g0 denominator; check pump power / length bounds")

    g0_per_m = (float(gA_A_per_m) + float(gA_B_per_mw_per_m) * p_mw) / denom  # + (alpha - 0.75)
    if np.any(g0_per_m <= 0):
        g0_per_m[g0_per_m <= 0] = 0
        # raise ValueError("g0_fit gives non-positive gain; check pump power / length bounds")

    if np.any(psat_mw <= 0):
        psat_mw[psat_mw <= 0] = 1e-10
        g0_per_m[psat_mw <= 0] = 0

    g0_tuple = tuple(float(x) for x in np.ravel(g0_per_m))
    psat_tuple = tuple(float(x) for x in np.ravel(psat_w))
    return g0_tuple, psat_tuple


def test_calc_formula10_and_plot_fig7b(
        calc_g0_psat_func,
        *,
        g0_to_gA_factor: float = 4.343,
        n_grid: int = 400,
        p_pump_min_mw: float = 10,
        p_pump_max_mw: float = 210.0,
        title_suffix: str = "",
        show_psat_fig7a: bool = True,
) -> None:
    """
    Тестирование функции, реализующей аппроксимацию (10) из JOSA B 34(2), 227 (2017):
      - сравнение с Table 2 (gA vs Ppump, LA)
      - построение Fig. 7(b)-подобного графика: точки (Table 2) + пунктир (модель)

    ВАЖНО: на Fig. 7(b) изображено gA * L_A (интегральный коэффициент усиления на участке активного волокна),
    поэтому на графике 7(b) и в соответствующих метриках сравниваем именно (gA_pred * L) и (gA_table * L).

    Параметры:
      calc_g0_psat_func:
        функция вида:
          g0_tuple, psat_tuple = calc_g0_psat_func(p_pump_w, fiber_length_m, **kwargs)
        где p_pump_w может быть скаляром или массивом, а выход — tuple(float, ...).

      g0_to_gA_factor:
        множитель для приведения выходного g0 к величине на графике/в Table 2:
          gA_pred = g0 * g0_to_gA_factor

        Типичные варианты:
          - 4.343, если g0 в 1/m (неперы/м), а gA в dB/m;
          - 1.0, если g0 уже в dB/m;
          - 1.0, если и таблица, и g0 в 1/m.

      show_psat_fig7a:
        дополнительно рисовать Fig.7(a)-подобный график Psat(Ppump) (линия (9) + точки).

    Ничего не сохраняет в файлы: только печать и plt.show().
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # ---- Data from Table 2 (JOSA B 34(2), 227 (2017)) ----
    # Pumps in mW, lengths in m, gA values as printed in the table (with missing entries = NaN).
    pumps_mw = np.array([31.2, 42.3, 61.4, 151.0, 198.0], dtype=np.float64)
    lengths_m = np.array([0.52, 1.08, 2.0, 2.5], dtype=np.float64)

    # Shape: (len(pumps), len(lengths))
    gA_table = np.array(
        [
            [3.60, 3.52, 3.01, 2.71],
            [3.78, 3.83, 3.43, 3.07],
            [4.02, 4.24, 3.65, 3.39],
            [4.54, 4.53, 4.14, np.nan],
            [4.50, 4.73, 4.25, np.nan],
        ],
        dtype=np.float64,
    )

    def _as_1d_tuple(x) -> tuple:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        return tuple(float(v) for v in arr)

    def _call_model_gA(p_pump_mw_1d: np.ndarray, L_m: float) -> np.ndarray:
        p_w = (p_pump_mw_1d.astype(np.float64) * 1e-3)
        g0_t, _psat_t = calc_g0_psat_func(_as_1d_tuple(p_w), float(L_m))
        g0 = np.asarray(g0_t, dtype=np.float64).reshape(-1)
        if g0.size != p_pump_mw_1d.size:
            raise ValueError(
                f"calc_g0_psat_func returned g0 of size {g0.size}, expected {p_pump_mw_1d.size} (same as input)."
            )
        return g0 * float(g0_to_gA_factor)

    # ---- Evaluate model on Table 2 points ----
    gA_pred = np.full_like(gA_table, np.nan, dtype=np.float64)

    for j, L in enumerate(lengths_m):
        gA_pred[:, j] = _call_model_gA(pumps_mw, float(L))

    # ---- Convert to Fig. 7(b) quantity: gA * L ----
    # Broadcasting: (P, L) * (L,) -> (P, L)
    gAL_table = gA_table * lengths_m[None, :]
    gAL_pred = gA_pred * lengths_m[None, :]

    # ---- Print error metrics ----
    mask = np.isfinite(gA_table)
    err = gAL_pred[mask] - gAL_table[mask]
    rmse_all = float(np.sqrt(np.mean(err * err))) if err.size else float("nan")
    mae_all = float(np.mean(np.abs(err))) if err.size else float("nan")

    print("=" * 100)
    print("Table 2 check (gA_table*L vs gA_pred*L)  [Fig. 7(b) quantity]")
    print(f"g0_to_gA_factor = {g0_to_gA_factor}")
    print(f"Points compared  = {int(np.sum(mask))}")
    print(f"RMSE             = {rmse_all:.6g}")
    print(f"MAE              = {mae_all:.6g}")

    # Per-length RMSE/MAE
    for j, L in enumerate(lengths_m):
        m = np.isfinite(gA_table[:, j])
        if not np.any(m):
            continue
        e = gAL_pred[m, j] - gAL_table[m, j]
        rmse = float(np.sqrt(np.mean(e * e)))
        mae = float(np.mean(np.abs(e)))
        print(f"  L={L:g} m: RMSE={rmse:.6g}, MAE={mae:.6g}, N={int(np.sum(m))}")
    print("=" * 100)

    # Detailed table
    print("Detailed points (Ppump_mW, L_m, gA_table*L, gA_pred*L, err):")
    for i, P in enumerate(pumps_mw):
        for j, L in enumerate(lengths_m):
            if not np.isfinite(gA_table[i, j]):
                continue
            gt = float(gAL_table[i, j])
            gp = float(gAL_pred[i, j])
            print(f"  {P:7.1f}  {L:5.2f}   {gt:10.5f}   {gp:10.5f}   {gp - gt:+10.5f}")

    # ---- Plot Fig. 7(b)-like: gA*L vs Ppump with dashed model curves ----
    p_grid = np.linspace(float(p_pump_min_mw), float(p_pump_max_mw), int(n_grid), dtype=np.float64)

    fig, ax = plt.subplots()
    for j, L in enumerate(lengths_m):
        gA_grid = _call_model_gA(p_grid, float(L))
        gAL_grid = gA_grid * float(L)
        (line,) = ax.plot(p_grid, gAL_grid, "--", linewidth=2.0, label=f"fit (L={L:g} m)")

        m = np.isfinite(gA_table[:, j])
        if np.any(m):
            ax.scatter(
                pumps_mw[m],
                gAL_table[m, j],
                s=55,
                marker="o",
                edgecolors="none",
                color=line.get_color(),
                label=f"Table 2 (L={L:g} m)",
            )

    ax.set_xlabel("Ppump (mW)")
    ax.set_ylabel("gA * L_A (Table 2 units · m)")
    ax.set_title(f"Fig. 7(b) reproduction: points (Table 2) + dashed (model){title_suffix}")
    ax.grid(True, which="both", linewidth=0.6)
    ax.legend(loc="best", fontsize=9)

    # ---- Optional Fig. 7(a)-like: Psat vs pump (line (9) + points from calc) ----
    if show_psat_fig7a:
        # Psat linear fit from Eq. (9) in the paper (in mW): Psat = 0.1292*Ppump - 1.096
        psat_fit_mw = 0.1292 * p_grid - 1.096

        fig2, ax2 = plt.subplots()
        ax2.plot(p_grid, psat_fit_mw, "--", linewidth=2.0, label="Eq. (9) fit (paper)")

        # show psat points inferred from calc at several lengths (paper says Psat ~ independent of L)
        for j, L in enumerate(lengths_m):
            p_w = (pumps_mw * 1e-3).astype(np.float64)
            _g0_t, psat_t = calc_g0_psat_func(_as_1d_tuple(p_w), float(L))
            psat_w = np.asarray(psat_t, dtype=np.float64).reshape(-1)
            psat_mw = psat_w * 1e3

            ax2.scatter(
                pumps_mw,
                psat_mw,
                s=55,
                marker="o",
                edgecolors="none",
                label=f"Psat from calc (L={L:g} m)",
            )

        ax2.set_xlabel("Ppump (mW)")
        ax2.set_ylabel("Psat (mW)")
        ax2.set_title(f"Fig. 7(a) reproduction: dashed Eq.(9) + points (calc){title_suffix}")
        ax2.grid(True, which="both", linewidth=0.6)
        ax2.legend(loc="best", fontsize=9)

    plt.show()


def plot_laser_gain_curve_from_params(
        params: dict,
        signal: np.ndarray,
        core_count: int | None = None,
        *,
        curve_kind: str = "signal",
        n_grid: int = 400,
        # --- pump-curve options (curve_kind="pump") ---
        p_pump_min_w: float | None = None,
        p_pump_max_w: float | None = None,
        # --- signal-curve options (curve_kind="signal") ---
        p_sig_min_w: float | None = None,
        p_sig_max_w: float | None = None,
        x_max_mult_psat: float = 3.0,
        weak_frac_psat: float = 0.1,
        annotate_regions: bool = True,
        core_index: int | None = None,
        plot_all_cores: bool = False,
        # --- output ---
        show_plot: bool = True,
        save_plot: bool = False,
        save_path: str | Path | None = None,
        title: str | None = None,
) -> dict:
    """
    Два режима:

    1) curve_kind="signal" (по умолчанию): "кривая усиления" как НЕЛИНЕЙНАЯ зависимость усилителя
       от входной мощности сигнала при фиксированных (g0, Psat), которые берутся из накачки (ppump).
       Рисуется:
         - Gain(dB) vs Pin
         - Pout vs Pin
       И отмечается операционная точка, определяемая ИЗ СИГНАЛА:
         Pin_op_per_core = mean(|signal|^2, axis=time)

       Модель насыщения по мощности:
         dP/dz = [g0 / (1 + P/Psat)] * P
       => ln P + P/Psat = const + g0 L
       => Pout = Psat * W( (Pin/Psat) * exp(Pin/Psat + g0 L) )

    2) curve_kind="pump": зависимости (g0, Psat, G_ss) от Ppump.

    Важно:
      - signal должен быть формы (C, M) (ядра × отсчёты времени),
      - мощность в ваших единицах берётся как |signal|^2,
      - средняя мощность на ядро: mean(|signal|^2) по времени.
    """

    def _to_per_core(v, c: int, *, name: str) -> tuple[float, ...]:
        if v is None:
            raise ValueError(f"{name} is None")
        if np.isscalar(v):
            return (float(v),) * c
        vv = tuple(float(x) for x in v)
        if len(vv) == 1:
            return (vv[0],) * c
        if len(vv) == c:
            return vv
        raise ValueError(f"{name}: expected len 1 or {c}, got {len(vv)}")

    def _default_fig_path(basename: str) -> Path:
        fmt = str(plt.rcParams.get("savefig.format", "png")).lower()
        out_dir = Path(__file__).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{basename}.{fmt}"
        if p.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            p = out_dir / f"{basename}_{ts}.{fmt}"
        return p

    def _lambertw_real(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = np.maximum(x, 0.0)

        try:
            from scipy.special import lambertw as _lw  # type: ignore
            y = _lw(x)
            return np.asarray(np.real(y), dtype=np.float64)
        except Exception:
            # Ньютонизация для y*exp(y)=x; старт log1p(x) хорошо работает для x>=0
            y = np.log1p(x)
            y[x == 0.0] = 0.0
            for _ in range(80):
                ey = np.exp(y)
                f = y * ey - x
                fp = ey * (y + 1.0)
                step = np.where(fp != 0.0, f / fp, 0.0)
                y_new = y - step
                if float(np.max(np.abs(step))) < 1e-12:
                    y = y_new
                    break
                y = y_new
            return y

    def _pout_from_pin(pin_w: np.ndarray, g0_1_per_m: float, psat_w: float, l_m: float) -> np.ndarray:
        pin_w = np.asarray(pin_w, dtype=np.float64)
        pin_w = np.maximum(pin_w, 0.0)

        if psat_w <= 0.0 or g0_1_per_m <= 0.0:
            return np.zeros_like(pin_w)

        pin_over = pin_w / psat_w
        expo = pin_over + float(g0_1_per_m) * float(l_m)

        expo = np.clip(expo, -700.0, 700.0)
        arg = pin_over * np.exp(expo)

        w = _lambertw_real(arg)
        pout = psat_w * w
        pout[pin_w == 0.0] = 0.0
        return pout

    # --- validate signal ---
    sig = np.asarray(signal)
    if sig.ndim != 2:
        raise ValueError(f"signal must be 2D array (C, M). Got shape={sig.shape}")
    c_sig = int(sig.shape[0])
    if c_sig <= 0:
        raise ValueError("signal must have C>=1 (first dimension).")

    # --- core_count ---
    if core_count is None:
        c_cnt = c_sig
    else:
        c_cnt = int(core_count)
        if c_cnt != c_sig:
            raise ValueError(f"core_count={c_cnt} mismatch with signal.shape[0]={c_sig}")
    if c_cnt <= 0:
        raise ValueError("core_count must be > 0")

    # --- mean power per core from signal (операционная точка) ---
    pin_op_arr = np.mean(np.abs(sig) ** 2, axis=1).astype(float, copy=False)  # (C,)

    # --- длина активного участка ---
    l_m = float(params.get("fiber_length_m", 0.3))
    if l_m <= 0.0:
        raise ValueError("fiber_length_m must be > 0")

    # --- per-core Ppump ---
    if "ppump" in params:
        p_pump_per_core_w = _to_per_core(params["ppump"], c_cnt, name="params['ppump']")
    else:
        has_total = ("p_pump_total_w" in params) or ("p_pump_total" in params)
        has_slm = ("pump_slm_w" in params)
        if not (has_total and has_slm):
            raise ValueError("Need 'ppump' OR ('p_pump_total(_w)' + 'pump_slm_w') in params.")

        p_total_key = "p_pump_total_w" if ("p_pump_total_w" in params) else "p_pump_total"
        p_pump_total_w = float(params[p_total_key])
        pump_slm_w = _to_per_core(params["pump_slm_w"], c_cnt, name="params['pump_slm_w']")

        slm_temperature = float(params.get("pump_slm_temperature", 1.0))
        slm_min_fraction = float(params.get("pump_slm_min_fraction", 0.0))

        p_pump_per_core_w = calc_p_pump_per_core_from_slm(
            p_pump_total_w,
            pump_slm_w,
            temperature=slm_temperature,
            min_fraction=slm_min_fraction,
        )

    p_pump_arr = np.asarray(p_pump_per_core_w, dtype=np.float64).reshape(-1)
    if np.any(p_pump_arr < 0.0):
        raise ValueError("Ppump must be >= 0")

    # --- (g0, Psat) per core from pump ---
    g0_t, psat_t = calc_g0_psat_from_p_pump_josab(tuple(p_pump_arr.tolist()), l_m, alpha=0)
    g0_per_core = np.asarray(g0_t, dtype=np.float64).reshape(-1)
    psat_per_core_w = np.asarray(psat_t, dtype=np.float64).reshape(-1)
    if g0_per_core.size != c_cnt or psat_per_core_w.size != c_cnt:
        raise RuntimeError("calc_g0_psat_from_p_pump_josab returned unexpected sizes")

    # --- core_index (для signal-curve) ---
    if core_index is None:
        core_index = int(np.floor(c_cnt / 2)) if c_cnt > 1 else 0
    core_index = int(core_index)
    if not (0 <= core_index < c_cnt):
        raise ValueError(f"core_index must be in [0, {c_cnt - 1}]")

    # --- figure width fallback (чтобы не зависеть от COL2/круговых импортов) ---
    col2 = float(globals().get("COL2", 183.0 / 25.4))  # inches; COL2 в вашем main ≈ 7.2"

    # ======================================================================================
    # (A) PUMP CURVE
    # ======================================================================================
    if str(curve_kind).lower() == "pump":
        pmin = float(p_pump_min_w) if p_pump_min_w is not None else 0.0
        pmax_auto = float(np.max(p_pump_arr)) if p_pump_arr.size else 0.0
        pmax = float(p_pump_max_w) if p_pump_max_w is not None else max(1.3 * pmax_auto, 0.05)
        if pmax <= pmin:
            pmax = pmin + max(0.05, 0.1 * (abs(pmin) + 1.0))

        n_grid_i = int(max(50, n_grid))
        p_grid_w = np.linspace(pmin, pmax, n_grid_i, dtype=np.float64)

        g0_g, psat_g = calc_g0_psat_from_p_pump_josab(p_grid_w, l_m, alpha=0)
        g0_grid = np.asarray(g0_g, dtype=np.float64).reshape(-1)
        psat_grid_w = np.asarray(psat_g, dtype=np.float64).reshape(-1)

        db_factor = 10.0 / float(np.log(10.0))
        gain_db_grid = db_factor * g0_grid * l_m
        gain_db_per_core = db_factor * g0_per_core * l_m

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(col2, col2 * 0.62), constrained_layout=True, sharex=True
        )

        p_grid_mw = p_grid_w * 1e3
        pp_mw = p_pump_arr * 1e3

        ax1.plot(p_grid_mw, gain_db_grid, label="small-signal gain (dB)")
        ax1.scatter(pp_mw, gain_db_per_core, s=40, marker="o", label="selected points")
        for i, (x, y) in enumerate(zip(pp_mw.tolist(), gain_db_per_core.tolist())):
            ax1.annotate(str(i), (x, y), textcoords="offset points", xytext=(4, 4),
                         ha="left", va="bottom", fontsize=8)

        ax1.set_ylabel(r"$G_{\mathrm{ss}}$ (dB)")
        ax1.grid(True, which="both", linewidth=0.6)
        ax1.legend(loc="best")

        ax2.plot(p_grid_mw, psat_grid_w * 1e3, linestyle="--", label=r"$P_{\mathrm{sat}}$ (mW)")
        ax2.scatter(pp_mw, psat_per_core_w * 1e3, s=40, marker="o", label="selected points")
        for i, (x, y) in enumerate(zip(pp_mw.tolist(), (psat_per_core_w * 1e3).tolist())):
            ax2.annotate(str(i), (x, y), textcoords="offset points", xytext=(4, 4),
                         ha="left", va="bottom", fontsize=8)

        ax2.set_xlabel(r"$P_{\mathrm{pump}}$ per core (mW)")
        ax2.set_ylabel(r"$P_{\mathrm{sat}}$ (mW)")
        ax2.grid(True, which="both", linewidth=0.6)
        ax2.legend(loc="best")

        if title is None:
            title = f"Pump → (g0, Psat) curve (L={l_m:g} m)"
        fig.suptitle(title, x=0.01, ha="left")

        saved_to = None
        if save_plot:
            saved_to = Path(save_path) if save_path is not None else _default_fig_path("laser_gain_curve_pump")
            try:
                fig.savefig(saved_to)
            except Exception as e:
                print(f"[warn] savefig failed: {e}")
                saved_to = None

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return {
            "curve_kind": "pump",
            "fiber_length_m": float(l_m),
            "core_count": int(c_cnt),
            "p_grid_w": p_grid_w,
            "g0_grid_1_per_m": g0_grid,
            "psat_grid_w": psat_grid_w,
            "gain_db_grid": gain_db_grid,
            "p_pump_per_core_w": tuple(float(x) for x in p_pump_arr.tolist()),
            "g0_per_core_1_per_m": tuple(float(x) for x in g0_per_core.tolist()),
            "psat_per_core_w": tuple(float(x) for x in psat_per_core_w.tolist()),
            "pin_op_per_core_w": tuple(float(x) for x in pin_op_arr.tolist()),
            "saved_to": str(saved_to) if saved_to is not None else None,
        }

    # ======================================================================================
    # (B) SIGNAL SATURATION CURVE
    # ======================================================================================
    g0_sel = float(g0_per_core[core_index])
    psat_sel = float(psat_per_core_w[core_index])

    pin_ref = float(pin_op_arr[core_index]) if np.isfinite(pin_op_arr[core_index]) else 0.0
    ps_ref = float(psat_sel) if np.isfinite(psat_sel) else 0.0

    xmin = float(p_sig_min_w) if p_sig_min_w is not None else 0.0

    if p_sig_max_w is not None:
        xmax = float(p_sig_max_w)
    else:
        xmax = float(x_max_mult_psat) * ps_ref if ps_ref > 0.0 else max(2.0 * pin_ref, 1e-3)
        if pin_ref > 0.0 and pin_ref > 0.9 * xmax:
            xmax = 1.3 * pin_ref
        if xmax <= xmin:
            xmax = xmin + max(1e-3, 0.1 * (abs(xmin) + 1.0))

    n_grid_i = int(max(80, n_grid))
    pin_grid_w = np.linspace(xmin, xmax, n_grid_i, dtype=np.float64)

    pout_grid_w = _pout_from_pin(pin_grid_w, g0_sel, psat_sel, l_m)

    eps = 1e-30
    gain_lin = pout_grid_w / np.maximum(pin_grid_w, eps)
    gain_db = 10.0 * np.log10(np.maximum(gain_lin, eps))
    if pin_grid_w.size:
        gain_db[pin_grid_w == 0.0] = (10.0 / float(np.log(10.0))) * g0_sel * l_m

    all_pout = None
    all_gdb = None
    if plot_all_cores and c_cnt > 1:
        all_pout = []
        all_gdb = []
        for c in range(c_cnt):
            g0_c = float(g0_per_core[c])
            ps_c = float(psat_per_core_w[c])
            pout_c = _pout_from_pin(pin_grid_w, g0_c, ps_c, l_m)
            glin_c = pout_c / np.maximum(pin_grid_w, eps)
            gdb_c = 10.0 * np.log10(np.maximum(glin_c, eps))
            if pin_grid_w.size:
                gdb_c[pin_grid_w == 0.0] = (10.0 / float(np.log(10.0))) * g0_c * l_m
            all_pout.append(pout_c)
            all_gdb.append(gdb_c)
        all_pout = np.asarray(all_pout, dtype=np.float64)
        all_gdb = np.asarray(all_gdb, dtype=np.float64)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(col2, col2 * 0.62), constrained_layout=True, sharex=True
    )

    pin_mw = pin_grid_w * 1e3
    pout_mw = pout_grid_w * 1e3

    if all_gdb is not None:
        for c in range(c_cnt):
            ax1.plot(pin_mw, all_gdb[c], linewidth=1.2, alpha=0.55, label=None)
    ax1.plot(pin_mw, gain_db, linewidth=2.0, label=f"core {core_index}: Gain (dB)")

    g_ss_db = (10.0 / float(np.log(10.0))) * max(g0_sel, 0.0) * l_m
    ax1.axhline(g_ss_db, linestyle="--", linewidth=1.2, label="small-signal limit")

    if pin_ref > 0.0:
        pout_ref = float(_pout_from_pin(np.array([pin_ref]), g0_sel, psat_sel, l_m)[0])
        g_ref_db = 10.0 * np.log10(max(pout_ref / max(pin_ref, eps), eps))
        ax1.scatter([pin_ref * 1e3], [g_ref_db], s=55, marker="o", label="operating point")
        ax1.annotate(
            f"Pin={pin_ref * 1e3:.0f} mW\nG={g_ref_db:.2f} dB",
            (pin_ref * 1e3, g_ref_db),
            textcoords="offset points",
            xytext=(8, 10),
            ha="left",
            va="bottom",
            fontsize=9,
        )
    else:
        pout_ref = float("nan")

    ax1.set_ylabel("Gain (dB)")
    ax1.grid(True, which="both", linewidth=0.6)
    ax1.legend(loc="best")

    if all_pout is not None:
        for c in range(c_cnt):
            ax2.plot(pin_mw, all_pout[c] * 1e3, linewidth=1.2, alpha=0.55, label=None)

    ax2.plot(pin_mw, pout_mw, linewidth=2.0, label=f"core {core_index}: $P_{{out}}$")
    ax2.plot(pin_mw, pin_mw, linestyle=":", linewidth=1.2, label=r"$P_{out}=P_{in}$")
    if pin_ref > 0.0 and np.isfinite(pout_ref):
        ax2.scatter([pin_ref * 1e3], [pout_ref * 1e3], s=55, marker="o", label="operating point")
        ax2.annotate(
            f"Pout={pout_ref * 1e3:.0f} mW",
            (pin_ref * 1e3, pout_ref * 1e3),
            textcoords="offset points",
            xytext=(8, -14),
            ha="left",
            va="top",
            fontsize=9,
        )

    ax2.set_xlabel(r"$P_{in}$ (mW)")
    ax2.set_ylabel(r"$P_{out}$ (mW)")
    ax2.grid(True, which="both", linewidth=0.6)
    ax2.legend(loc="best")

    if annotate_regions and psat_sel > 0.0:
        p_weak = float(weak_frac_psat) * psat_sel
        p_mid = psat_sel
        p_max_vis = float(np.max(pin_grid_w)) if pin_grid_w.size else 0.0

        for ax in (ax1, ax2):
            for v in (p_weak, p_mid):
                if xmin < v < p_max_vis:
                    ax.axvline(v * 1e3, linestyle=":", linewidth=1.0, alpha=0.8)

        y0, y1 = ax1.get_ylim()
        y_txt = y1 - 0.06 * (y1 - y0)

        def _clamp(x, a, b):
            return max(a, min(b, x))

        x1w = _clamp(0.5 * p_weak, xmin + 1e-12, p_max_vis)
        x2w = _clamp(0.5 * (p_weak + p_mid), xmin + 1e-12, p_max_vis)
        x3w = _clamp(0.5 * (p_mid + p_max_vis), xmin + 1e-12, p_max_vis)

        ax1.text(x1w * 1e3, y_txt, r"$P_{in}\ll P_{sat}$", fontsize=9, ha="center", va="top")
        ax1.text(x2w * 1e3, y_txt, r"$P_{in}\sim P_{sat}$", fontsize=9, ha="center", va="top")
        ax1.text(x3w * 1e3, y_txt, r"$P_{in}\gtrsim P_{sat}$", fontsize=9, ha="center", va="top")

        if p_max_vis <= 3.2 * psat_sel:
            ax1.text(
                (p_max_vis * 1e3) * 0.985,
                y_txt,
                r"$\rightarrow$ deeper saturation",
                fontsize=8,
                ha="right",
                va="top",
            )

    if title is None:
        title = (f"Gain saturation curve | core {core_index} | L={l_m:g} m | "
                 f"g0={g0_sel:.3g} 1/m | Psat={psat_sel * 1e3:.3g} mW")
    fig.suptitle(title, x=0.01, ha="left")

    saved_to = None
    if save_plot:
        saved_to = Path(save_path) if save_path is not None else _default_fig_path(
            f"laser_gain_saturation_core{core_index}"
        )
        try:
            fig.savefig(saved_to)
        except Exception as e:
            print(f"[warn] savefig failed: {e}")
            saved_to = None

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "curve_kind": "signal",
        "fiber_length_m": float(l_m),
        "core_count": int(c_cnt),
        "core_index": int(core_index),
        "x_max_mult_psat": float(x_max_mult_psat),
        "weak_frac_psat": float(weak_frac_psat),
        "p_pump_per_core_w": tuple(float(x) for x in p_pump_arr.tolist()),
        "g0_per_core_1_per_m": tuple(float(x) for x in g0_per_core.tolist()),
        "psat_per_core_w": tuple(float(x) for x in psat_per_core_w.tolist()),
        "pin_op_per_core_w": tuple(float(x) for x in pin_op_arr.tolist()),
        "pin_grid_w": pin_grid_w,
        "pout_grid_w": pout_grid_w,
        "gain_db_grid": gain_db,
        "saved_to": str(saved_to) if saved_to is not None else None,
    }


def apply_readout(X: np.ndarray, W: np.ndarray, add_bias: bool = True) -> np.ndarray:
    if add_bias:
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    else:
        Xb = X
    return Xb @ W


def ridge_alpha_sweep_diagnostics(
        result: dict,
        *,
        alphas: np.ndarray | list[float] | tuple[float, ...] | None = None,
        alpha_min: float | None = None,
        alpha_max: float | None = None,
        n_alpha: int = 61,
        add_bias: bool = True,
        show_plots: bool = True,
        title: str = "Ridge alpha sweep diagnostics",
        print_summary: bool = True,
        # авто-настройка сетки
        auto_lo_factor_vs_smax2: float = 1e-12,
        auto_hi_factor_vs_smax2: float = 1e3,
        auto_lo_factor_vs_sminpos2: float = 1e-6,
        auto_min_decades: float = 12.0,
        # выбор "устойчивых" alpha
        select_plateau_rtol: tuple[float, ...] = (5e-3, 1e-2),
        select_cond_targets: tuple[float, ...] = (1e8, 1e6),
        val_blocks_for_se: int = 8,
        alpha_label_precision: int = 3,
        # --- НОВОЕ: block-wise устойчивость ---
        val_block_quantile: float = 0.9,
        block_p90_rtol: float = 0.02,
        block_max_rtol: float = 0.2,
) -> dict:
    """
    Диагностика влияния ridge-regularization (alpha) на фиксированном наборе состояний резервуара.

    Ожидается, что result содержит:
      X_train, y_train, X_val, y_val, X_test, y_test.

    ВАЖНО:
      - Здесь НЕТ "поджима alpha": перебираем ровно те alpha, что заданы/сгенерированы.
      - Если alphas не задан, и alpha_min/alpha_max тоже не заданы, то сетка строится автоматически из SVD спектра.

    Для каждого alpha решаем ridge через одну SVD:
      Xb = U diag(s) V^T,
      W(alpha) = V diag(s / (s^2 + alpha)) U^T y.

    Дополнительно считаем:
      - cond_ridge(alpha) = (s_max^2 + alpha) / (s_min_pos^2 + alpha),
      - df(alpha) = sum_i s_i^2 / (s_i^2 + alpha),
      - ||W||_F(alpha).

    НОВОЕ (block-wise для val, если валид. достаточно длинная):
      - считаем NRMSE по нескольким последовательным временным блокам,
      - строим mean ± SE, а также p90 и max по блокам,
      - рекомендуемый alpha старается избегать провалов по блокам (p90/max) и быть более регуляризованным (1SE).
    """

    required = ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")
    missing = [k for k in required if k not in result]
    if missing:
        raise KeyError(
            f"ridge_alpha_sweep_diagnostics: в result нет ключей {missing}. "
            f"Нужно вызывать после run_single_experiment/run_experiments (в одиночном режиме)."
        )

    Xtr = np.asarray(result["X_train"])
    ytr = np.asarray(result["y_train"])
    Xva = np.asarray(result["X_val"])
    yva = np.asarray(result["y_val"])
    Xte = np.asarray(result["X_test"])
    yte = np.asarray(result["y_test"])

    if Xtr.ndim != 2 or ytr.ndim != 2:
        raise ValueError("Ожидаются X_train формы (N,F) и y_train формы (N,T).")
    if Xtr.shape[0] != ytr.shape[0]:
        raise ValueError("X_train и y_train должны иметь одинаковое число строк (samples).")

    n_tr, f_tr = int(Xtr.shape[0]), int(Xtr.shape[1])
    n_va = int(Xva.shape[0]) if Xva.ndim == 2 else 0
    n_te = int(Xte.shape[0]) if Xte.ndim == 2 else 0

    # --- SVD на train ---
    Xtr64 = Xtr.astype(np.float64, copy=False)
    ytr64 = ytr.astype(np.float64, copy=False)

    if add_bias:
        Xtr_b = np.hstack([Xtr64, np.ones((n_tr, 1), dtype=np.float64)])
    else:
        Xtr_b = Xtr64

    U, s, Vt = np.linalg.svd(Xtr_b, full_matrices=False)
    s = np.asarray(s, dtype=np.float64)
    s2 = s * s

    if s2.size == 0:
        raise ValueError("SVD вернул пустой спектр; проверьте X_train.")

    s_max2 = float(s2[0])
    s2_pos = s2[s2 > 0.0]
    s_minpos2 = float(s2_pos.min()) if s2_pos.size else 0.0

    cond_x = float("inf")
    if s2_pos.size:
        cond_x = float(np.sqrt(s_max2 / max(s_minpos2, np.finfo(np.float64).tiny)))

    Ut_y = U.T @ ytr64  # (M, T)

    # --- сетка alpha ---
    def _auto_alpha_grid() -> np.ndarray:
        if s_max2 <= 0.0:
            lo = 1e-12
            hi = 1e3
        else:
            lo1 = float(s_max2) * float(auto_lo_factor_vs_smax2)
            lo2 = float(s_minpos2) * float(auto_lo_factor_vs_sminpos2) if s_minpos2 > 0.0 else 0.0
            lo = max(lo1, lo2, np.finfo(np.float64).tiny)

            hi = float(s_max2) * float(auto_hi_factor_vs_smax2)
            if hi <= lo:
                hi = lo * 1e6

            decades = float(np.log10(hi) - np.log10(lo))
            if decades < float(auto_min_decades):
                hi = lo * (10.0 ** float(auto_min_decades))

        grid = np.logspace(np.log10(lo), np.log10(hi), int(max(2, n_alpha)))
        return np.unique(grid.astype(np.float64, copy=False))

    if alphas is not None:
        alpha_grid = np.asarray(alphas, dtype=np.float64).reshape(-1)
        alpha_grid = alpha_grid[np.isfinite(alpha_grid)]
        alpha_grid = alpha_grid[alpha_grid > 0.0]
        if alpha_grid.size == 0:
            raise ValueError("alphas заданы, но после фильтрации не осталось положительных конечных значений.")
        alpha_grid = np.unique(alpha_grid)
    else:
        if alpha_min is not None and alpha_max is not None:
            a0 = float(alpha_min)
            a1 = float(alpha_max)
            if not (np.isfinite(a0) and np.isfinite(a1) and a0 > 0.0 and a1 > 0.0):
                raise ValueError("alpha_min/alpha_max должны быть конечными и > 0.")
            if a0 > a1:
                a0, a1 = a1, a0
            alpha_grid = np.logspace(np.log10(a0), np.log10(a1), int(max(2, n_alpha))).astype(np.float64)
            alpha_grid = np.unique(alpha_grid)
        else:
            alpha_grid = _auto_alpha_grid()

    alpha_lo = float(alpha_grid[0])
    alpha_hi = float(alpha_grid[-1])

    # --- вспомогательные ---
    def _predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
        return apply_readout(X, W, add_bias=add_bias)

    def _safe_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0 or y_pred.size == 0:
            return float("nan")
        n = min(int(y_true.shape[0]), int(y_pred.shape[0]))
        if n < 2:
            return float("nan")
        return float(nrmse(y_true[:n], y_pred[:n]))

    def _fmt_alpha(x: float) -> str:
        if not np.isfinite(x) or x <= 0.0:
            return "—"
        return f"{x:.{int(alpha_label_precision)}g}"

    # --- block-wise оценка для val (mean±SE, p90, max) ---
    def _block_slices(n: int, k: int) -> list[slice]:
        n = int(n)
        k = int(k)
        if n <= 0 or k <= 1:
            return [slice(0, n)]
        k = max(2, min(k, n))
        edges = np.linspace(0, n, num=k + 1, dtype=int)
        out = []
        for i in range(k):
            a, b = int(edges[i]), int(edges[i + 1])
            if b - a >= 1:
                out.append(slice(a, b))
        return out if out else [slice(0, n)]

    def _sigma_global(y: np.ndarray, eps: float = 1e-12) -> float:
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size < 2:
            return float("nan")
        s_ = float(np.std(y))
        return s_ if (np.isfinite(s_) and s_ >= eps) else float("nan")

    def _block_nrmse_with_global_sigma(y_true: np.ndarray, y_pred: np.ndarray, blocks: list[slice]) -> np.ndarray:
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        n = min(y_true.size, y_pred.size)
        if n < 2:
            return np.full(len(blocks), np.nan, dtype=float)

        y_true = y_true[:n]
        y_pred = y_pred[:n]
        s_ = _sigma_global(y_true)
        if not np.isfinite(s_):
            return np.full(len(blocks), np.nan, dtype=float)

        out = []
        for sl in blocks:
            a, b = int(sl.start or 0), int(sl.stop or 0)
            a = max(0, min(a, n))
            b = max(0, min(b, n))
            if b - a < 2:
                out.append(np.nan)
                continue
            d = y_true[a:b] - y_pred[a:b]
            rmse_ = float(np.sqrt(np.mean(d * d)))
            out.append(rmse_ / s_)
        return np.asarray(out, dtype=float)

    blocks_va = _block_slices(n_va, int(val_blocks_for_se)) if n_va >= 2 else [slice(0, n_va)]
    use_blocks = (n_va >= 8) and (len(blocks_va) >= 2)

    q = float(val_block_quantile)
    q = 0.9 if not np.isfinite(q) else min(max(q, 0.5), 0.99)

    # --- sweep ---
    rows: list[dict] = []
    val_mean = []
    val_se = []
    val_p90 = []
    val_max = []

    for a in alpha_grid:
        a = float(a)

        shrink = s / (s2 + a)  # (M,)
        W = (Vt.T * shrink) @ Ut_y  # (F(+1), T)

        ytr_hat = _predict(Xtr, W)
        yva_hat = _predict(Xva, W) if n_va >= 1 else np.zeros_like(yva)
        yte_hat = _predict(Xte, W) if n_te >= 1 else np.zeros_like(yte)

        df = float(np.sum(s2 / (s2 + a)))

        if s_minpos2 > 0.0:
            cond_r = float((s_max2 + a) / (s_minpos2 + a))
        else:
            cond_r = float((s_max2 + a) / max(a, np.finfo(np.float64).tiny))

        w_norm_f = float(np.linalg.norm(W))

        nrmse_tr = _safe_nrmse(ytr, ytr_hat)
        nrmse_va = _safe_nrmse(yva, yva_hat)
        nrmse_te = _safe_nrmse(yte, yte_hat)

        if use_blocks:
            b = _block_nrmse_with_global_sigma(yva, yva_hat, blocks_va)
            m = np.isfinite(b)
            if np.any(m):
                b_use = b[m]
                mu = float(np.mean(b_use))
                se = float(np.std(b_use, ddof=1) / np.sqrt(b_use.size)) if b_use.size >= 2 else float("nan")
                p90 = float(np.quantile(b_use, q))
                mx = float(np.max(b_use))
            else:
                mu, se, p90, mx = float("nan"), float("nan"), float("nan"), float("nan")
        else:
            mu, se, p90, mx = float("nan"), float("nan"), float("nan"), float("nan")

        val_mean.append(mu)
        val_se.append(se)
        val_p90.append(p90)
        val_max.append(mx)

        rows.append(
            dict(
                alpha=a,
                nrmse_train=nrmse_tr,
                nrmse_val=nrmse_va,
                nrmse_test=nrmse_te,
                df=df,
                cond_x=float(cond_x),
                cond_ridge=cond_r,
                w_norm_f=w_norm_f,
                val_mean_block=mu,
                val_se_block=se,
                val_p90_block=p90,
                val_max_block=mx,
            )
        )

    # --- выбор критерия (val если есть, иначе test) ---
    has_val = (n_va >= 2) and np.any(np.isfinite([r["nrmse_val"] for r in rows]))
    key = "nrmse_val" if has_val else "nrmse_test"

    a_arr = np.array([r["alpha"] for r in rows], dtype=np.float64)
    e_tr = np.array([r["nrmse_train"] for r in rows], dtype=np.float64)
    e_va = np.array([r["nrmse_val"] for r in rows], dtype=np.float64)
    e_te = np.array([r["nrmse_test"] for r in rows], dtype=np.float64)
    df_a = np.array([r["df"] for r in rows], dtype=np.float64)
    cr_a = np.array([r["cond_ridge"] for r in rows], dtype=np.float64)
    wn_a = np.array([r["w_norm_f"] for r in rows], dtype=np.float64)

    mu_va = np.array(val_mean, dtype=np.float64)
    se_va = np.array(val_se, dtype=np.float64)
    p90_va = np.array(val_p90, dtype=np.float64)
    mx_va = np.array(val_max, dtype=np.float64)

    # --- селекторы alpha ---
    def _alpha_minimize(metric: np.ndarray) -> float | None:
        m = np.isfinite(metric)
        if not np.any(m):
            return None
        idx = int(np.argmin(metric[m]))
        idx = int(np.arange(metric.size)[m][idx])
        return float(a_arr[idx])

    def _alpha_largest_within(metric: np.ndarray, rtol: float) -> float | None:
        m = np.isfinite(metric)
        if not np.any(m):
            return None
        minv = float(np.min(metric[m]))
        thr = (1.0 + float(rtol)) * minv
        ok = m & (metric <= thr)
        if not np.any(ok):
            return None
        return float(np.max(a_arr[ok]))

    def _alpha_1se(metric_mean: np.ndarray, metric_se: np.ndarray) -> float | None:
        m = np.isfinite(metric_mean) & np.isfinite(metric_se)
        if not np.any(m):
            return None
        idx_min = int(np.argmin(metric_mean[m]))
        idx_min = int(np.arange(metric_mean.size)[m][idx_min])
        thr = float(metric_mean[idx_min] + metric_se[idx_min])
        ok = m & (metric_mean <= thr)
        if not np.any(ok):
            return None
        return float(np.max(a_arr[ok]))

    def _alpha_with_cond_cap(metric: np.ndarray, rtol: float, cond_cap: float) -> float | None:
        m = np.isfinite(metric) & np.isfinite(cr_a)
        if not np.any(m):
            return None
        minv = float(np.min(metric[m]))
        thr = (1.0 + float(rtol)) * minv
        ok = m & (metric <= thr) & (cr_a <= float(cond_cap))
        if not np.any(ok):
            return None
        return float(np.max(a_arr[ok]))

    def _alpha_largest_feasible(mask_ok: np.ndarray) -> float | None:
        mask_ok = np.asarray(mask_ok, dtype=bool)
        if not np.any(mask_ok):
            return None
        return float(np.max(a_arr[mask_ok]))

    metric = e_va if key == "nrmse_val" else e_te

    alpha_candidates: dict[str, float] = {}

    a_min = _alpha_minimize(metric)
    if a_min is not None:
        alpha_candidates[f"{key}_min"] = float(a_min)

    for rtol in tuple(select_plateau_rtol):
        a_pl = _alpha_largest_within(metric, float(rtol))
        if a_pl is not None:
            alpha_candidates[f"plateau_{rtol:.3g}"] = float(a_pl)

    # 1SE и block-устойчивость имеет смысл только когда есть блоки
    a_1se = None
    if use_blocks and key == "nrmse_val":
        a_1se = _alpha_1se(mu_va, se_va)
        if a_1se is not None:
            alpha_candidates["val_1se_mean"] = float(a_1se)

        # --- НОВОЕ: кандидаты по p90/max ---
        mp = np.isfinite(p90_va)
        if np.any(mp):
            p90_min = float(np.min(p90_va[mp]))
            thr_p90 = (1.0 + float(block_p90_rtol)) * p90_min
            ok_p90 = mp & (p90_va <= thr_p90)
            a_p90 = _alpha_largest_feasible(ok_p90)
            if a_p90 is not None:
                alpha_candidates[f"val_p90_plateau_{block_p90_rtol:.3g}"] = float(a_p90)

        mmx = np.isfinite(mx_va)
        if np.any(mmx):
            mx_min = float(np.min(mx_va[mmx]))
            thr_mx = (1.0 + float(block_max_rtol)) * mx_min
            ok_mx = mmx & (mx_va <= thr_mx)
            a_mx = _alpha_largest_feasible(ok_mx)
            if a_mx is not None:
                alpha_candidates[f"val_max_plateau_{block_max_rtol:.3g}"] = float(a_mx)

    # cond-capped (в пределах плато по основной метрике)
    rtol_for_cond = float(select_plateau_rtol[-1]) if select_plateau_rtol else 1e-2
    for cap in tuple(select_cond_targets):
        a_cc = _alpha_with_cond_cap(metric, rtol_for_cond, float(cap))
        if a_cc is not None:
            alpha_candidates[f"plateau_{rtol_for_cond:.3g}_cond<= {cap:.0e}"] = float(a_cc)

    # --- НОВОЕ: recommended = 1SE(mean) + p90/max защита (+ опционально cond) ---
    alpha_recommended = None
    if use_blocks and key == "nrmse_val" and np.any(np.isfinite(mu_va)) and np.any(np.isfinite(se_va)):
        m = np.isfinite(mu_va) & np.isfinite(se_va)
        if np.any(m):
            idx0 = int(np.argmin(mu_va[m]))
            idx0 = int(np.arange(mu_va.size)[m][idx0])
            thr_mu = float(mu_va[idx0] + se_va[idx0])

            mp = np.isfinite(p90_va)
            mmx = np.isfinite(mx_va)
            thr_p90 = float("inf")
            thr_mx = float("inf")

            if np.any(mp):
                thr_p90 = (1.0 + float(block_p90_rtol)) * float(np.min(p90_va[mp]))
            if np.any(mmx):
                thr_mx = (1.0 + float(block_max_rtol)) * float(np.min(mx_va[mmx]))

            ok = np.isfinite(a_arr)
            ok = ok & np.isfinite(mu_va) & (mu_va <= thr_mu)

            if np.any(mp):
                ok = ok & np.isfinite(p90_va) & (p90_va <= thr_p90)
            if np.any(mmx):
                ok = ok & np.isfinite(mx_va) & (mx_va <= thr_mx)

            # если получилось пусто — ослабляем max, потом p90
            a_rec = _alpha_largest_feasible(ok)
            if a_rec is None and np.any(mp):
                ok2 = np.isfinite(a_arr) & np.isfinite(mu_va) & (mu_va <= thr_mu) & np.isfinite(p90_va) & (p90_va <= thr_p90)
                a_rec = _alpha_largest_feasible(ok2)
            if a_rec is None:
                ok3 = np.isfinite(a_arr) & np.isfinite(mu_va) & (mu_va <= thr_mu)
                a_rec = _alpha_largest_feasible(ok3)

            alpha_recommended = float(a_rec) if a_rec is not None else None

    # fallback, если блоков нет
    if alpha_recommended is None:
        # приоритет: самый маленький (минимум) -> самый правый plateau -> самый строгий cond-cap
        if a_min is not None:
            alpha_recommended = float(a_min)
        pl_keys = [k for k in alpha_candidates.keys() if k.startswith("plateau_")]
        if pl_keys:
            def _rtol_from_key(k: str) -> float:
                try:
                    return float(k.split("_")[1])
                except Exception:
                    return -1.0
            pl_keys = sorted(pl_keys, key=_rtol_from_key)
            alpha_recommended = float(alpha_candidates[pl_keys[-1]])
        cond_keys = [k for k in alpha_candidates.keys() if "cond<=" in k]
        if cond_keys:
            def _cap_from_key(k: str) -> float:
                try:
                    s_ = k.split("cond<=")[1].strip()
                    return float(s_.replace(" ", ""))
                except Exception:
                    return float("inf")
            cond_keys = sorted(cond_keys, key=_cap_from_key)
            alpha_recommended = float(alpha_candidates[cond_keys[0]])

    # --- сводка ---
    if print_summary:
        print("=" * 90)
        print(f"[ridge-sweep] X_train: N={n_tr}, F={f_tr}, add_bias={bool(add_bias)}")
        print(f"[ridge-sweep] X_val:   N={n_va}   | X_test: N={n_te}")
        print(f"[ridge-sweep] cond(X_train(+bias)) ≈ {cond_x:.3g}")
        print(f"[ridge-sweep] s_max^2={s_max2:.3g}, s_min_pos^2={s_minpos2:.3g}")
        print(f"[ridge-sweep] alpha grid: [{alpha_lo:.3g}, {alpha_hi:.3g}]  points={int(a_arr.size)}")
        print(f"[ridge-sweep] criterion: {key}")
        if alpha_recommended is not None:
            print(f"[ridge-sweep] alpha_recommended={alpha_recommended:.6g}")
        if alpha_candidates:
            print("[ridge-sweep] alpha candidates:")
            for k in sorted(alpha_candidates.keys()):
                print(f"  {k:>32s} : {alpha_candidates[k]:.6g}")
        if use_blocks and key == "nrmse_val":
            print(f"[ridge-sweep] val blocks={int(len(blocks_va))}, block quantile={q:.3g}, "
                  f"p90_rtol={float(block_p90_rtol):.3g}, max_rtol={float(block_max_rtol):.3g}")
        print("=" * 90)

    # --- графики ---
    eps = np.finfo(np.float64).tiny
    e_tr_p = np.maximum(eps, e_tr)
    e_va_p = np.maximum(eps, e_va)
    e_te_p = np.maximum(eps, e_te)
    df_p = np.maximum(eps, df_a)
    cr_p = np.maximum(eps, cr_a)
    wn_p = np.maximum(eps, wn_a)

    col2 = float(globals().get("COL2", 183.0 / 25.4))

    def _alpha_lines() -> list[tuple[str, float, dict]]:
        out = []
        if alpha_recommended is not None:
            out.append(("recommended", float(alpha_recommended), dict(linestyle="--", linewidth=2.6)))

        styles = [
            dict(linestyle="--", linewidth=1.8),
            dict(linestyle="-.", linewidth=1.8),
            dict(linestyle=":", linewidth=2.2),
            dict(linestyle=(0, (3, 1, 1, 1)), linewidth=1.8),
        ]
        keys = list(alpha_candidates.keys())
        j = 0
        for k in keys:
            aval = float(alpha_candidates[k])
            if alpha_recommended is not None and np.isclose(aval, float(alpha_recommended)):
                continue
            st = styles[j % len(styles)].copy()
            out.append((k, aval, st))
            j += 1
        return out

    alpha_lines = _alpha_lines()

    # (1) NRMSE vs alpha
    fig1, ax1 = plt.subplots(figsize=(col2, col2 * 0.55), constrained_layout=True)
    ax1.loglog(a_arr, e_tr_p, marker="o", linewidth=1.6, label=f"train (N={n_tr})")
    if n_va >= 2:
        ax1.loglog(a_arr, e_va_p, marker="o", linewidth=1.6, label=f"val (N={n_va})")
    if n_te >= 2:
        ax1.loglog(a_arr, e_te_p, marker="o", linewidth=1.6, label=f"test (N={n_te})")
    for name, aval, st in alpha_lines:
        ax1.axvline(aval, **st, label=f"alpha: {name} = {_fmt_alpha(aval)}")
    ax1.set_xlabel(r"ridge regularization $\alpha$")
    ax1.set_ylabel("NRMSE (log scale)")
    ax1.set_title(f"{title}: NRMSE vs alpha  |  F={f_tr}{'+1' if add_bias else ''}, cond(X)≈{cond_x:.2g}",
                  loc="left")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="best")

    # (2) Spectrum
    fig2, ax2 = plt.subplots(figsize=(col2, col2 * 0.55), constrained_layout=True)
    idx = np.arange(1, int(s2.size) + 1, dtype=np.int64)
    ax2.semilogy(idx, np.maximum(eps, s2), linewidth=1.6, label=r"spectrum $s_i^2$ (train)")
    ax2.axhline(alpha_lo, linestyle=":", linewidth=1.2, label=f"alpha grid min/max ({_fmt_alpha(alpha_lo)})")
    ax2.axhline(alpha_hi, linestyle=":", linewidth=1.2)
    for name, aval, st in alpha_lines:
        ax2.axhline(aval, **st, label=f"alpha: {name} = {_fmt_alpha(aval)}")
    ax2.set_xlabel("singular mode index i (sorted)")
    ax2.set_ylabel(r"$s_i^2$ (log scale)")
    ax2.set_title(f"{title}: singular spectrum and chosen alphas", loc="left")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="best")

    # (3) df, cond_ridge, ||W||_F
    fig3, ax3 = plt.subplots(figsize=(col2, col2 * 0.55), constrained_layout=True)
    ax3.loglog(a_arr, df_p, marker="o", linewidth=1.6, label=r"$df(\alpha)=\sum s_i^2/(s_i^2+\alpha)$")
    ax3.loglog(a_arr, cr_p, marker="o", linewidth=1.6, label=r"$cond_{ridge}(\alpha)$")
    ax3.loglog(a_arr, wn_p, marker="o", linewidth=1.6, label=r"$\|W(\alpha)\|_F$")
    for name, aval, st in alpha_lines:
        ax3.axvline(aval, **st, label=f"alpha: {name} = {_fmt_alpha(aval)}")
    ax3.set_xlabel(r"ridge regularization $\alpha$")
    ax3.set_ylabel("value (log scale)")
    ax3.set_title(f"{title}: conditioning/complexity vs alpha", loc="left")
    ax3.grid(True, which="both", alpha=0.25)
    ax3.legend(loc="best")

    # (4) Block-wise val: mean±SE + p90 + max
    if use_blocks and key == "nrmse_val" and np.any(np.isfinite(mu_va)):
        fig4, ax4 = plt.subplots(figsize=(col2, col2 * 0.55), constrained_layout=True)
        mu_p = np.maximum(eps, mu_va)
        ax4.loglog(a_arr, mu_p, marker="o", linewidth=1.8, label="val mean (block-wise)")

        se_ok = np.isfinite(se_va) & (se_va > 0)
        if np.any(se_ok):
            upper = np.maximum(eps, mu_va + se_va)
            lower = np.maximum(eps, mu_va - se_va)
            ax4.fill_between(a_arr, lower, upper, alpha=0.18, label="±SE (blocks)")

        p_ok = np.isfinite(p90_va)
        if np.any(p_ok):
            ax4.loglog(a_arr, np.maximum(eps, p90_va), linewidth=1.8, linestyle="--",
                       label=f"val p{int(round(q * 100))} (blocks)")

        mx_ok = np.isfinite(mx_va)
        if np.any(mx_ok):
            ax4.loglog(a_arr, np.maximum(eps, mx_va), linewidth=1.8, linestyle=":",
                       label="val max (blocks)")

        for name, aval, st in alpha_lines:
            ax4.axvline(aval, **st, label=f"alpha: {name} = {_fmt_alpha(aval)}")

        ax4.set_xlabel(r"ridge regularization $\alpha$")
        ax4.set_ylabel("NRMSE (log scale)")
        ax4.set_title(f"{title}: val block-wise diagnostics (mean±SE, p{int(round(q * 100))}, max)", loc="left")
        ax4.grid(True, which="both", alpha=0.25)
        ax4.legend(loc="best")

    if show_plots:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        if use_blocks and key == "nrmse_val":
            try:
                plt.close(fig4)
            except Exception:
                pass

    return dict(
        rows=rows,
        alpha_grid=alpha_grid.copy(),
        alpha_candidates=alpha_candidates,
        alpha_recommended=alpha_recommended,
        best_by=key,
        singular_values=s.copy(),
        singular_values_sq=s2.copy(),
        cond_x=float(cond_x),
        alpha_grid_is_auto=bool(alphas is None and (alpha_min is None or alpha_max is None)),
        alpha_grid_lo=float(alpha_lo),
        alpha_grid_hi=float(alpha_hi),
        val_blocks=int(len(blocks_va)),
        val_mean_block=mu_va.copy(),
        val_se_block=se_va.copy(),
        val_p90_block=p90_va.copy(),
        val_max_block=mx_va.copy(),
        val_block_quantile=float(q),
        block_p90_rtol=float(block_p90_rtol),
        block_max_rtol=float(block_max_rtol),
    )
