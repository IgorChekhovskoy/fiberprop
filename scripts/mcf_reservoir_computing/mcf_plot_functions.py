from pathlib import Path

import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, TYPE_CHECKING
from datetime import datetime

from scripts.mcf_reservoir_computing.mcf_auxiliary_functions import (
    apply_readout,
    nrmse,
)

if TYPE_CHECKING:
    from scripts.mcf_reservoir_computing.mcf_reservoir_computing import (
        ExperimentConfig,
        NARMA10Config,
        MGConfig,
        TaskName,
    )
else:
    ExperimentConfig = object
    NARMA10Config = object
    MGConfig = object
    TaskName = str

MM = 1 / 25.4
COL1, COL15, COL2 = 89 * MM, 136 * MM, 183 * MM  # Nature: 1, 1.5, 2 колонки


def _main_mcf_module():
    """
    Ленивый доступ к основному модулю без циклического top-level import.

    Важно:
      • если mcf_reservoir_computing.py запущен как скрипт, нужные функции лежат в __main__;
      • если модуль импортирован как пакет, берём его из sys.modules по полному имени;
      • прямой import делаем только как fallback.
    """
    main_module = sys.modules.get("__main__")
    if main_module is not None and hasattr(main_module, "create_mask"):
        return main_module

    module_name = "scripts.mcf_reservoir_computing.mcf_reservoir_computing"
    module = sys.modules.get(module_name)
    if module is not None and hasattr(module, "create_mask"):
        return module

    from scripts.mcf_reservoir_computing import mcf_reservoir_computing as module
    return module


def create_mask(mask_size: int, rng: np.random.Generator, kind: str = "rademacher") -> np.ndarray:
    return _main_mcf_module().create_mask(mask_size, rng, kind=kind)


def get_washout_samples(cfg) -> int:
    return _main_mcf_module().get_washout_samples(cfg)


def mackey_glass(t_size, tau=17.0, n=10, beta=0.2, gamma=0.1, initial_condition=1.2, dt=1.0):
    return _main_mcf_module().mackey_glass(
        t_size,
        tau=tau,
        n=n,
        beta=beta,
        gamma=gamma,
        initial_condition=initial_condition,
        dt=dt,
    )


def split_train_val_test(xw_len: int,
                         train_frac: float,
                         val_frac: float) -> tuple[slice, slice, slice]:
    return _main_mcf_module().split_train_val_test(xw_len, train_frac, val_frac)


def task_debug_meta(task_name: TaskName) -> dict[str, str]:
    return _main_mcf_module().task_debug_meta(task_name)


def _task_sequences_full_for_debug(task_name: TaskName,
                                   task_cfg: Union[MGConfig, NARMA10Config]
                                   ) -> tuple[np.ndarray, np.ndarray, int]:
    mcf = _main_mcf_module()

    if task_name == "mackey_glass":
        if not isinstance(task_cfg, mcf.MGConfig):
            raise TypeError("For task_name='mackey_glass', task_cfg must be MGConfig")

        warmup = int(task_cfg.warmup)
        x_full = mcf.mackey_glass(task_cfg.t_size + warmup,
                                  tau=task_cfg.tau,
                                  n=task_cfg.n,
                                  beta=task_cfg.beta,
                                  gamma=task_cfg.gamma,
                                  initial_condition=task_cfg.initial_condition,
                                  dt=task_cfg.dt)

        x_used = x_full[warmup:].astype(float, copy=False)
        mu = float(np.mean(x_used))
        sigma = float(np.std(x_used))
        if sigma < 1e-12:
            x_full_norm = np.zeros_like(x_full, dtype=float)
        else:
            x_full_norm = (x_full - mu) / sigma

        return x_full_norm, x_full_norm, warmup

    if task_name == "narma10":
        if not isinstance(task_cfg, mcf.NARMA10Config):
            raise TypeError("For task_name='narma10', task_cfg must be NARMA10Config")

        warmup = int(task_cfg.warmup)
        total_size = int(task_cfg.t_size) + warmup
        rng = np.random.default_rng(task_cfg.seed)

        u_full = rng.uniform(task_cfg.u_low, task_cfg.u_high, size=total_size).astype(float, copy=False)
        y_full = np.zeros(total_size, dtype=float)

        for t in range(9, total_size - 1):
            y_full[t + 1] = (
                0.3 * y_full[t]
                + 0.05 * y_full[t] * np.sum(y_full[t - 9:t + 1])
                + 1.5 * u_full[t - 9] * u_full[t]
                + 0.1
            )

        u_used = u_full[warmup:].astype(float, copy=False)
        u_mu = float(np.mean(u_used))
        u_sigma = float(np.std(u_used))
        if u_sigma < 1e-12:
            u_full_norm = np.zeros_like(u_full, dtype=float)
        else:
            u_full_norm = (u_full - u_mu) / u_sigma

        y_used = y_full[warmup:].astype(float, copy=False)
        y_mu = float(np.mean(y_used))
        y_sigma = float(np.std(y_used))
        if y_sigma < 1e-12:
            y_full_norm = np.zeros_like(y_full, dtype=float)
        else:
            y_full_norm = (y_full - y_mu) / y_sigma

        return u_full_norm, y_full_norm, warmup

    raise ValueError(f"Unknown task_name: {task_name}")

def _plot_temporal_masks(ax, masks: np.ndarray, mask_kind: str):
    """Heatmap временных масок: ось X — индекс внутри символа, ось Y — ядро."""
    C, M = masks.shape
    im = ax.imshow(masks, aspect="auto", interpolation="nearest",
                   extent=[0, M, C, 0], cmap="coolwarm")
    ax.set_title(f"Temporal masks (size={M})", loc="left")
    ax.set_xlabel("mask element index")
    ax.set_ylabel("core index")
    return im


def _plot_spatial_weights(ax, weights: np.ndarray, title: str = "Spatial weights by cores"):
    """Bar-чарт по ядрам."""
    C = weights.shape[0]
    ax.bar(np.arange(C), weights, width=0.7)
    ax.set_title(title, loc="left")
    ax.set_xlabel("core index")
    ax.set_ylabel("weight")


def _reconstruct_masks_or_weights(core_count: int,
                                  variant: str,
                                  mask_size: int,
                                  mask_kind: str,
                                  seed: int | None) -> dict:
    """
    Возвращает один из словарей:
      • {'type':'temporal', 'masks': (C,M)}        — для temporal_* (без каких-либо 'weights')
      • {'type':'spatial',  'weights': (C,)}       — для spatial_only

    Для spatial_only в идеальном (lossless) случае нормируем weights так, чтобы sum(weights**2) = 1.
    Тогда суммарная мощность входа не зависит от core_count, а gain_in задаёт общий масштаб поля.
    """
    rng = np.random.default_rng(seed)

    if variant == "temporal_unique_per_core":
        masks = np.empty((core_count, mask_size), dtype=float)
        for c in range(core_count):
            masks[c] = create_mask(mask_size, rng, kind=mask_kind)
        # НИКАКИХ spatial weights для temporal-режимов
        return {"type": "temporal", "masks": masks}

    if variant == "temporal_same_all_cores":
        mask = create_mask(mask_size, rng, kind=mask_kind)
        masks = np.tile(mask, (core_count, 1))
        # НИКАКИХ spatial weights для temporal-режимов
        return {"type": "temporal", "masks": masks}

    if variant == "spatial_only":
        # В spatial_only подаём постоянные веса на ядра (коэффициенты по полю).
        # Идеальный SLM: сохраняем суммарную мощность -> нормировка по L2.
        weights = rng.uniform(0.0, 1.0, size=core_count)
        w_norm = float(np.linalg.norm(weights))
        if w_norm > 0.0:
            weights = weights / w_norm
        else:
            weights = np.zeros(core_count, dtype=float)
            weights[0] = 1.0
        return {"type": "spatial", "weights": weights}

    raise ValueError(f"Unknown variant: {variant}")


def debug_plot_input_overview(cfg,
                              input_series: np.ndarray,
                              target_series: np.ndarray,
                              task_name: TaskName):
    """
    Рисует:
      1) Полный ряд задачи с пометками: warmup, shift, washout, train/val/test.
      2) Маски (по времени) для каждого ядра ИЛИ пространственные веса (bar),
         в зависимости от варианта.
    """
    meta = task_debug_meta(task_name)

    input_series = np.asarray(input_series, dtype=float).reshape(-1)
    target_series = np.asarray(target_series, dtype=float).reshape(-1)
    if input_series.shape[0] != target_series.shape[0]:
        raise ValueError("input_series и target_series должны иметь одинаковую длину")

    input_full, target_full, warmup = _task_sequences_full_for_debug(task_name, cfg.task_cfg)

    S = int(target_series.shape[0])
    if input_full.shape[0] < warmup + S or target_full.shape[0] < warmup + S:
        raise ValueError("Полные ряды для debug-графика короче, чем ожидается по конфигу")

    input_used_ref = input_full[warmup:warmup + S]
    target_used_ref = target_full[warmup:warmup + S]

    diff_in = float(np.max(np.abs(input_used_ref - input_series))) if S > 0 else 0.0
    diff_tg = float(np.max(np.abs(target_used_ref - target_series))) if S > 0 else 0.0
    if diff_in > 1e-6:
        print(f"[WARN] debug_plot_input_overview: input_series mismatch (max|diff|={diff_in:.3e})")
    if diff_tg > 1e-6:
        print(f"[WARN] debug_plot_input_overview: target_series mismatch (max|diff|={diff_tg:.3e})")

    taps = int(getattr(cfg.training, "taps", 1) or 1)
    taps_drop = max(0, taps - 1)
    shift_syms = int(getattr(cfg.training, "target_shift", 0) or 0)

    M_eff = cfg.mask.mask_size if str(cfg.variant).startswith("temporal_") else 1
    w_samples = get_washout_samples(cfg)
    w_syms = int(np.ceil(int(w_samples) / max(1, int(M_eff))))

    align_drop = taps_drop + shift_syms
    N_eff = max(0, S - align_drop - w_syms)

    train_frac = float(getattr(cfg.training, "train_frac", 0.6))
    val_frac = float(getattr(cfg.training, "val_frac", 0.2))
    sl_train, sl_val, sl_test = split_train_val_test(N_eff, train_frac, val_frac)
    n_train = int(sl_train.stop - sl_train.start)
    n_val = int(sl_val.stop - sl_val.start)
    n_test = int(sl_test.stop - sl_test.start)

    i_warmup_L = 0
    i_warmup_R = int(warmup)
    i_taps_L = i_warmup_R
    i_taps_R = i_taps_L + taps_drop
    i_shift_L = i_taps_R
    i_shift_R = i_shift_L + shift_syms
    i_wash_L = i_shift_R
    i_wash_R = i_wash_L + w_syms
    i_tr_L = i_wash_R
    i_tr_R = i_tr_L + n_train
    i_va_L = i_tr_R
    i_va_R = i_va_L + n_val
    i_te_L = i_va_R
    i_te_R = i_te_L + n_test

    fig = plt.figure(figsize=(COL2, COL2 * 0.62))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.6], hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    x_full = np.arange(target_full.shape[0])

    if np.array_equal(input_full, target_full):
        ax1.plot(x_full, target_full, label=meta["input_label"])
    else:
        ax1.plot(x_full, input_full, label=meta["input_label"])
        ax1.plot(x_full, target_full, label=meta["target_label"], alpha=0.85)

    ax1.set_xlim(x_full[0], x_full[-1])
    ax1.margins(x=0.0)

    if int(warmup) > 0:
        ax1.axvline(i_warmup_R, color="k", lw=1, alpha=0.25)

    def span_if(a, b, color, label, alpha=0.18):
        a, b = int(a), int(b)
        if b - a < 1:
            return False
        a_plot = max(a, 0)
        b_plot = min(b, int(x_full[-1]) + 1)
        if b_plot - a_plot < 1:
            return False
        ax1.axvspan(a_plot, b_plot, color=color, alpha=alpha, label=label)
        return True

    shown = []
    if warmup > 0 and span_if(i_warmup_L, i_warmup_R, "#888888", "warmup"):
        shown.append("warmup")
    if taps_drop > 0 and span_if(i_taps_L, i_taps_R, "#17becf", f"taps-1={taps_drop}"):
        shown.append("taps")
    if shift_syms > 0 and span_if(i_shift_L, i_shift_R, "#1f77b4", "target shift"):
        shown.append("target shift")
    if w_syms > 0 and span_if(i_wash_L, i_wash_R, "#ff7f0e", "washout"):
        shown.append("washout")
    if span_if(i_tr_L, i_tr_R, "#2ca02c", "train"):
        shown.append("train")
    if span_if(i_va_L, i_va_R, "#9467bd", "val"):
        shown.append("val")
    if span_if(i_te_L, i_te_R, "#d62728", "test"):
        shown.append("test")

    title_suffix = "/".join(shown) if shown else ""
    ax1.set_title(f"{meta['task_title']}{': ' + title_suffix if title_suffix else ''}", loc="left")
    ax1.set_xlabel("symbol index")
    ax1.set_ylabel(meta["y_label"])

    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        leg = ax1.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True)
        leg.get_frame().set_facecolor((1, 1, 1, 0.6))
        leg.get_frame().set_edgecolor((0, 0, 0, 0.3))

    masks_info = _reconstruct_masks_or_weights(core_count=cfg.core_count,
                                               variant=cfg.variant,
                                               mask_size=cfg.mask.mask_size,
                                               mask_kind=cfg.mask.mask_kind,
                                               seed=cfg.mask.seed)

    ax2 = fig.add_subplot(gs[1, 0])
    if masks_info["type"] == "temporal":
        im = _plot_temporal_masks(ax2, masks_info["masks"], cfg.mask.mask_kind)
        cbar = fig.colorbar(im, ax=ax2, fraction=0.46 / 10, pad=0.04)
        cbar.set_label("mask value")
    else:
        _plot_spatial_weights(ax2, masks_info["weights"], title="Spatial weights by cores")

    _maybe_savefig(fig, f"input_overview_{task_name}_{cfg.variant}_C{cfg.core_count}",
                   enabled=getattr(cfg.reservoir, "save_figs", False))
    plt.show()


def debug_plot_mg_attractor(cfg,
                            mg_series_used: np.ndarray,
                            title: str = ""):
    """
    3D-визуализация аттрактора Mackey–Glass по delay-embedding: (x(t), x(t-τ), x(t-2τ)).
    Сегменты shift/washout/train/val/test рисуются только если их длина ≥ 2.
    График оформлен под публикации: без сетки/панелей, ortho-проекция, equal-aspect.
    """
    from matplotlib.ticker import MaxNLocator
    import matplotlib.patheffects as pe

    x1d = np.asarray(mg_series_used, dtype=float).ravel()
    S = x1d.size
    tau_samples = max(1, int(round(float(cfg.task_cfg.tau) / float(cfg.task_cfg.dt))))
    off = 2 * tau_samples
    if S <= off + 1:
        print(f"debug_plot_mg_attractor: серия короче 2τ (S={S}, 2τ={off}) — пропуск.")
        return

    # delay-вложение
    X = x1d[off:]  # x(t)
    Y = x1d[tau_samples:-tau_samples]  # x(t-τ)
    Z = x1d[:-off]  # x(t-2τ)
    L = X.shape[0]

    # границы сегментов в ИНДЕКСАХ mg_series_used (0..S)
    shift_syms = int(getattr(cfg.training, "target_shift", 0))

    # ЕДИНО: washout в отсчётах → в символы
    w_samples = get_washout_samples(cfg)
    M_eff = cfg.mask.mask_size if str(cfg.variant).startswith("temporal_") else 1
    w_syms = int(np.ceil(w_samples / max(1, int(M_eff))))

    N_eff = S - shift_syms - w_syms
    n_train = int(N_eff * cfg.training.train_frac) if N_eff > 0 else 0
    n_val = int(N_eff * cfg.training.val_frac) if N_eff > 0 else 0
    n_test = max(0, N_eff - n_train - n_val)

    i_shift = (0, max(0, shift_syms))
    i_wash = (i_shift[1], i_shift[1] + max(0, w_syms))
    i_tr = (i_wash[1], i_wash[1] + max(0, n_train))
    i_va = (i_tr[1], i_tr[1] + max(0, n_val))
    i_te = (i_va[1], min(S, i_va[1] + max(0, n_test)))

    fig = plt.figure(figsize=(COL2 * 0.62, COL2 * 0.62), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    ax.set_proj_type('ortho')
    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
    ax.view_init(elev=20, azim=-15)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0, 0, 0, 0))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))

    ax.set_title(title, loc="left")
    ax.set_xlabel("x(t)")
    ax.set_ylabel("x(t − τ)")
    ax.set_zlabel("x(t − 2τ)")

    base_lw = float(plt.rcParams.get("lines.linewidth", 1.0))

    def plot_segment(name, color, bounds, min_len: int = 2):
        a, b = int(bounds[0]), int(bounds[1])
        aa, bb = max(a, off), min(b, S)
        if bb - aa >= min_len:
            lo = max(0, min(aa - off, L))
            hi = max(0, min(bb - off, L))
            if hi - lo >= min_len:
                (line,) = ax.plot(X[lo:hi], Y[lo:hi], Z[lo:hi], color=color, label=name)
                line.set_path_effects([
                    pe.Stroke(linewidth=base_lw * 1.8, foreground='white'),
                    pe.Normal()
                ])
                return True
        return False

    segments = [
        ("target shift", "#1f77b4", i_shift),
        ("washout", "#ff7f0e", i_wash),
        ("train", "#2ca02c", i_tr),
        ("val", "#9467bd", i_va),
        ("test", "#d62728", i_te),
    ]
    for name, color, bounds in segments:
        plot_segment(name, color, bounds)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        leg = ax.legend(uniq.values(), uniq.keys(),
                        bbox_to_anchor=(0.02, 0.98), loc="upper left",
                        frameon=True, title="Segments:")
        leg.get_frame().set_facecolor((1, 1, 1, 0.6))
        leg.get_frame().set_edgecolor((0, 0, 0, 0.3))

    if getattr(cfg.reservoir, "save_figs", False):
        p = _default_fig_path(f"mg_attractor_{cfg.variant}_C{cfg.core_count}")
        try:
            fig.savefig(p, bbox_inches="tight", pad_inches=0.04)
        except Exception as e:
            print(f"[warn] savefig failed: {e}")

    plt.show()


def debug_plot_post_training_comparison(cfg,
                                        y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        title: str = "Comparison: truth vs prediction",
                                        n_show: int = 2000,
                                        start: int = 0) -> float:
    """
    Рисует сравнение на тесте и возвращает NRMSE по ВСЕМУ переданному сегменту.

    Важно: отображаемое окно (start/n_show) влияет только на картинку, а не на метрику.
    Это нужно, чтобы NRMSE здесь совпадал с тем, что рисуется/считается по train/val/test в других местах.

    Устойчиво работает при коротких окнах: если точек < 2, вместо линии рисуется маркер,
    легенда и NRMSE подавляются, а set_xlim не вызывается (чтобы не ловить warning).
    """
    meta = task_debug_meta(getattr(cfg, "task_name", "mackey_glass"))

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    N = min(y_true.shape[0], y_pred.shape[0])
    if N == 0:
        fig, ax = plt.subplots(figsize=(COL2, COL2 * 0.33), constrained_layout=True)
        ax.set_title(f"{title}   •   NRMSE=—", loc="left")
        ax.set_xlabel("symbol index")
        ax.set_ylabel(meta["y_label"])
        _maybe_savefig(fig, f"post_training_comparison_{cfg.variant}_C{cfg.core_count}",
                       enabled=getattr(cfg.reservoir, "save_figs", False))
        plt.show()
        return float('nan')

    y_true = y_true[:N]
    y_pred = y_pred[:N]

    err = float('nan') if N < 2 else float(nrmse(y_true, y_pred))

    start = int(max(0, start))
    end = int(min(N, start + int(max(1, n_show))))
    x = np.arange(start, end)
    Lvis = int(end - start)

    if Lvis < 2 and N >= 2:
        start, end = max(0, N - 2), N
        x = np.arange(start, end)
        Lvis = end - start

    fig, ax = plt.subplots(figsize=(COL2, COL2 * 0.33), constrained_layout=True)

    if Lvis >= 2:
        ax.plot(x, y_true[start:end], label="ground truth")
        ax.plot(x, y_pred[start:end], label="prediction")
        ax.set_xlim(x[0], x[-1])
    else:
        ax.plot(x, y_true[start:end], ls="none", marker="o", ms=3)
        ax.plot(x, y_pred[start:end], ls="none", marker="o", ms=3)

    ax.set_title(
        f"{title}" + (f"   •   NRMSE={err:.4f}" if np.isfinite(err) else "   •   NRMSE=—"),
        loc="left"
    )
    ax.set_xlabel("symbol index")
    ax.set_ylabel(meta["y_label"])
    ax.margins(x=0.0)

    if Lvis >= 2:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True)
            leg.get_frame().set_facecolor((1, 1, 1, 0.6))
            leg.get_frame().set_edgecolor((0, 0, 0, 0.3))

    _maybe_savefig(fig, f"post_training_comparison_{cfg.variant}_C{cfg.core_count}",
                   enabled=getattr(cfg.reservoir, "save_figs", False))
    plt.show()

    return err


def debug_plot_readout_train_val_test(res: dict,
                                      title: str = "Mackey-Glass series: prediction") -> dict:
    """
    Склейка train→val→test: истина и прогноз. Каждая зона и подпись добавляется
    только если длина сегмента ≥ 2. Метрики для сегментов короче 2 точек → NaN.
    Легенда с полупрозрачным фоном у кромки.
    """
    cfg = res.get("cfg")
    meta = task_debug_meta(getattr(cfg, "task_name", "mackey_glass"))

    W = res["W_out"]
    Xtr, ytr = res["X_train"], res["y_train"].reshape(-1, 1)
    Xva, yva = res["X_val"], res["y_val"].reshape(-1, 1)
    Xte, yte = res["X_test"], res["y_test"].reshape(-1, 1)

    ytr_hat = apply_readout(Xtr, W)
    yva_hat = apply_readout(Xva, W) if Xva.size else np.zeros_like(yva)
    yte_hat = apply_readout(Xte, W) if Xte.size else np.zeros_like(yte)

    def _nrmse_safe(y, yhat):
        y = np.asarray(y).ravel()
        yhat = np.asarray(yhat).ravel()
        if y.size < 2 or yhat.size < 2:
            return float('nan')
        return float(nrmse(y, yhat))

    m = {
        "nrmse_train": _nrmse_safe(ytr, ytr_hat),
        "nrmse_val": _nrmse_safe(yva, yva_hat),
        "nrmse_test": _nrmse_safe(yte, yte_hat),
    }

    y_all = np.concatenate([ytr, yva, yte], axis=0).reshape(-1)
    yhat_all = np.concatenate([ytr_hat, yva_hat, yte_hat], axis=0).reshape(-1)
    N = y_all.size

    n_tr = int(ytr.shape[0])
    n_va = int(yva.shape[0])
    n_te = int(yte.shape[0])

    b_tr = (0, n_tr)
    b_va = (n_tr, n_tr + n_va)
    b_te = (n_tr + n_va, n_tr + n_va + n_te)

    fig, ax = plt.subplots(figsize=(COL2, COL2 * 0.33), constrained_layout=True)
    x = np.arange(N)

    ax.plot(x, y_all, label="ground truth")
    ax.plot(x, yhat_all, label="prediction")

    def span_if(bounds, color, label):
        lo, hi = int(bounds[0]), int(bounds[1])
        if hi - lo >= 2:
            ax.axvspan(lo, hi, color=color, alpha=0.18, label=label)
            return True
        return False

    def _fmt_nrmse(v: float) -> str:
        return f"{v:.4f}" if np.isfinite(v) else "—"

    span_if(b_tr, "#2ca02c", f"train  (NRMSE={_fmt_nrmse(m['nrmse_train'])})")
    span_if(b_va, "#9467bd", f"val    (NRMSE={_fmt_nrmse(m['nrmse_val'])})")
    span_if(b_te, "#d62728", f"test   (NRMSE={_fmt_nrmse(m['nrmse_test'])})")

    if n_tr >= 1:
        ax.axvline(b_tr[1], color="k", lw=1, alpha=0.6)
    if n_va >= 1:
        ax.axvline(b_va[1], color="k", lw=1, alpha=0.6)

    ax.set_title(title, loc="left")
    ax.set_xlabel("symbol index")
    ax.set_ylabel(meta["y_label"])
    ax.margins(x=0.0)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True)
        leg.get_frame().set_facecolor((1, 1, 1, 0.6))
        leg.get_frame().set_edgecolor((0, 0, 0, 0.3))

    _maybe_savefig(fig, f"readout_concat_{cfg.variant}_C{cfg.core_count}",
                   enabled=getattr(cfg.reservoir, "save_figs", False))
    plt.show()

    return m


def _default_fig_path(basename: str) -> Path:
    fmt = str(plt.rcParams.get("savefig.format", "pdf")).lower()
    out_dir = Path(__file__).parent  # сохраняем рядом со скриптом
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{basename}.{fmt}"
    if p.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = out_dir / f"{basename}_{ts}.{fmt}"
    return p


def _maybe_savefig(fig,
                   basename: str,
                   explicit_path: Optional[Union[str, Path]] = None,
                   enabled: Optional[bool] = None) -> Optional[Path]:
    """
    explicit_path задан → сохраняем туда (игнорируем enabled).
    explicit_path не задан → сохраняем в <папка скрипта>/<basename>.<fmt>, только если enabled=True.
    Параметры сохранения полностью из rcParams (формат, dpi, bbox, прозрачность и т.д.).
    """
    try:
        if explicit_path is not None:
            fig.savefig(explicit_path)
            return Path(explicit_path)
        if enabled:
            p = _default_fig_path(basename)
            fig.savefig(p)
            return p
    except Exception as e:
        print(f"[warn] savefig failed: {e}")
    return None

def update_plots_and_save(results_stream: list[dict],
                           base_cfg: ExperimentConfig,
                           *,
                           logx: bool = False,
                           logy: bool = False,
                           ts: str | None = None) -> dict:
    """
    Рисует три графика (NRMSE vs coupling / L_coupling / radius) и пишет CSV.
    Поведение и стиль согласованы с функциями отрисовки в модуле (Nature-стиль).

    Args:
        results_stream: список словарей с ключами:
            {"radius", "coupling", "nrmse_train", "nrmse_val", "nrmse_test"}.
        base_cfg: конфигурация эксперимента; variant и core_count используются в именах файлов.
        logx, logy: логарифмические шкалы по осям.
        ts: фиксированный таймстамп для серии файлов. Если None — создаётся новый.

    Returns:
        dict с путями до сохранённых файлов: {"p_c", "p_l", "p_r", "p_csv"}.
    """
    import csv
    from datetime import datetime
    from pathlib import Path

    save_figs_flag = bool(getattr(base_cfg.reservoir, "save_figs", False))
    out_dir = Path(__file__).parent
    variant_str = str(base_cfg.variant)
    core_count = int(base_cfg.core_count)
    fmt = str(plt.rcParams.get("savefig.format", "pdf")).lower()
    ts = ts or datetime.now().strftime('%Y%m%d-%H%M')

    p_c = out_dir / f"scan_nrmse_vs_coupling_{variant_str}_C{core_count}_{ts}.{fmt}"
    p_l = out_dir / f"scan_nrmse_vs_Lc_{variant_str}_C{core_count}_{ts}.{fmt}"
    p_r = out_dir / f"scan_nrmse_vs_radius_{variant_str}_C{core_count}_{ts}.{fmt}"
    p_csv = out_dir / f"scan_coupling_results_{variant_str}_C{core_count}_{ts}.csv"

    # валидные точки по coupling
    valid = [r for r in results_stream if np.isfinite(float(r.get("coupling", np.nan)))]
    if not valid:
        return {"p_c": None, "p_l": None, "p_r": None, "p_csv": None}

    coupl = np.array([float(d["coupling"]) for d in valid], float)
    Lc = np.where((coupl > 0) & np.isfinite(coupl), np.pi / (2.0 * coupl), np.nan)
    rad = np.array([float(d.get("radius", np.nan)) for d in valid], float)
    y_tr = np.array([float(d.get("nrmse_train", np.nan)) for d in valid], float)
    y_va = np.array([float(d.get("nrmse_val", np.nan)) for d in valid], float)

    def _sorted_xy(x_raw, y_raw):
        x = np.asarray(x_raw, float)
        y = np.asarray(y_raw, float)
        m = np.isfinite(x) & np.isfinite(y)
        if not np.any(m):
            return None, None
        x, y = x[m], y[m]
        order = np.argsort(x)
        return x[order], y[order]

    # — фигуры под Nature-стиль
    fig_c, ax_c = plt.subplots(figsize=(COL2, COL2 * 0.38), constrained_layout=True)
    fig_l, ax_l = plt.subplots(figsize=(COL2, COL2 * 0.38), constrained_layout=True)
    fig_r, ax_r = plt.subplots(figsize=(COL2, COL2 * 0.38), constrained_layout=True)

    # 1) NRMSE vs coupling
    ax_c.clear()
    xc_tr, yc_tr = _sorted_xy(coupl, y_tr)
    xc_va, yc_va = _sorted_xy(coupl, y_va)
    drew = False
    if xc_tr is not None:
        ax_c.plot(xc_tr, yc_tr,  # marker="o",
                  label="Train NRMSE")
        drew = True
    if xc_va is not None:
        ax_c.plot(xc_va, yc_va,  # marker="s",
                  linestyle="--", label="Val NRMSE")
        drew = True
    ax_c.set_xlabel("coupling coefficient, 1/m")
    ax_c.set_ylabel("NRMSE")
    # if logx:
    #     ax_c.set_xscale("log")
    if logy:
        ax_c.set_yscale("log")
    if drew:
        ax_c.legend()

    if save_figs_flag:
        try:
            fig_c.savefig(p_c)
        except Exception as e:
            print(f"[warn] savefig coupling failed: {e}")

    # 2) NRMSE vs L_coupling
    ax_l.clear()
    xl_tr, yl_tr = _sorted_xy(Lc, y_tr)
    xl_va, yl_va = _sorted_xy(Lc, y_va)
    drew = False
    if xl_tr is not None:
        ax_l.plot(xl_tr, yl_tr,  # marker="o",
                  label="Train NRMSE")
        drew = True
    if xl_va is not None:
        ax_l.plot(xl_va, yl_va,  # marker="s",
                  linestyle="--", label="Val NRMSE")
        drew = True
    ax_l.set_xlabel("coupling length, m")
    ax_l.set_ylabel("NRMSE")
    if logx:
        ax_l.set_xscale("log")
    if logy:
        ax_l.set_yscale("log")
    if drew:
        ax_l.legend()
    if save_figs_flag:
        try:
            fig_l.savefig(p_l)
        except Exception as e:
            print(f"[warn] savefig Lc failed: {e}")

    # 3) NRMSE vs radius
    ax_r.clear()
    xr_tr, yr_tr = _sorted_xy(rad, y_tr)
    xr_va, yr_va = _sorted_xy(rad, y_va)
    drew = False
    if xr_tr is not None:
        ax_r.plot(xr_tr, yr_tr,
                  # marker="o",
                  label="Train NRMSE")
        drew = True
    if xr_va is not None:
        ax_r.plot(xr_va, yr_va,
                  # marker="s",
                  linestyle="--", label="Val NRMSE")
        drew = True
    ax_r.set_xlabel("inter-core radius, µm")
    ax_r.set_ylabel("NRMSE")
    if logx:
        ax_r.set_xscale("log")
    if logy:
        ax_r.set_yscale("log")
    if drew:
        ax_r.legend()
    if save_figs_flag:
        try:
            fig_r.savefig(p_r)
        except Exception as e:
            print(f"[warn] savefig radius failed: {e}")

    # CSV: полный срез текущего results_stream
    if save_figs_flag:
        try:
            with open(p_csv, "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["radius", "coupling", "L_coupling",
                             "nrmse_train", "nrmse_val", "nrmse_test"])
                for d in results_stream:
                    c = float(d.get("coupling", float("nan")))
                    L = (np.pi / (2.0 * c)) if (np.isfinite(c) and c > 0) else float("nan")
                    wr.writerow([float(d.get("radius", float("nan"))),
                                 c, L,
                                 float(d.get("nrmse_train", float("nan"))),
                                 float(d.get("nrmse_val", float("nan"))),
                                 float(d.get("nrmse_test", float("nan")))])
        except Exception as e:
            print(f"[warn] write csv failed: {e}")

    return {"p_c": p_c, "p_l": p_l, "p_r": p_r, "p_csv": p_csv}


def plot_combined_csv_results(base_cfg, logx: bool = False, logy: bool = False, save_figs: bool = True):
    """
    Находит все CSV-файлы с результатами сканирования в текущей папке,
    объединяет данные и строит графики NRMSE vs coupling / L_coupling / radius
    в макетном стиле (Nature, двухколоночная ширина).

    Args:
        base_cfg: конфигурация эксперимента (variant, core_count)
        logx: логарифмическая шкала по X
        logy: логарифмическая шкала по Y
        save_figs: сохранять ли графики в файлы (формат берётся из rcParams)
    """
    import pandas as pd

    variant_str = str(base_cfg.variant)
    core_count = int(base_cfg.core_count)
    out_dir = Path(__file__).parent

    # собираем все CSV нужного вида
    pattern = f"scan_coupling_results_{variant_str}_C{core_count}_*.csv"
    csv_files = list(out_dir.glob(pattern))
    if not csv_files:
        print(f"Не найдено CSV по шаблону: {pattern}")
        return
    print(f"Найдено CSV: {len(csv_files)}")

    # объединяем
    all_df = []
    for p in csv_files:
        try:
            df = pd.read_csv(p)
            df["source_file"] = p.name
            all_df.append(df)
            print(f"Загружено {len(df)} строк из {p.name}")
        except Exception as e:
            print(f"Ошибка чтения {p}: {e}")
    if not all_df:
        print("Нет данных для объединения")
        return

    combined_df = pd.concat(all_df, ignore_index=True)

    # удаляем дубликаты по всем столбцам, кроме 'source_file'
    cols = [c for c in combined_df.columns if c != "source_file"]
    combined_df = combined_df.drop_duplicates(subset=cols)
    print(f"Уникальных точек: {len(combined_df)}")

    # фильтруем валидные строки
    if "coupling" not in combined_df.columns or "radius" not in combined_df.columns:
        print("В CSV нет обязательных колонок: 'coupling' / 'radius'")
        return
    valid = combined_df[np.isfinite(combined_df["coupling"])]
    if len(valid) == 0:
        print("Нет валидных данных для построения графиков")
        return

    # гарантируем наличие L_coupling_m
    if "L_coupling_m" not in valid.columns:
        c = valid["coupling"].to_numpy(float)
        Lc = np.where((c > 0) & np.isfinite(c), np.pi / (2.0 * c), np.nan)
        valid = valid.assign(L_coupling_m=Lc)

    # извлекаем массивы
    coupl = valid["coupling"].to_numpy(float)
    Lc = valid["L_coupling_m"].to_numpy(float)
    rad = valid["radius"].to_numpy(float)
    y_tr = valid.get("nrmse_train", pd.Series(np.nan, index=valid.index)).to_numpy(float)
    y_va = valid.get("nrmse_val", pd.Series(np.nan, index=valid.index)).to_numpy(float)

    def _sorted_xy(x_raw, y_raw):
        x = np.asarray(x_raw, float)
        y = np.asarray(y_raw, float)
        m = np.isfinite(x) & np.isfinite(y)
        if not np.any(m):
            return None, None
        x, y = x[m], y[m]
        order = np.argsort(x)
        return x[order], y[order]

    # имена файлов вывода
    fmt = str(plt.rcParams.get("savefig.format", "pdf")).lower()
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    p_c = out_dir / f"combined_nrmse_vs_coupling_{variant_str}_C{core_count}_{ts}.{fmt}"
    p_l = out_dir / f"combined_nrmse_vs_Lc_{variant_str}_C{core_count}_{ts}.{fmt}"
    p_r = out_dir / f"combined_nrmse_vs_radius_{variant_str}_C{core_count}_{ts}.{fmt}"
    p_csv = out_dir / f"combined_results_{variant_str}_C{core_count}_{ts}.csv"

    # фигуры в стиле Nature: COL2 × 0.38, constrained_layout
    fig_c, ax_c = plt.subplots(figsize=(COL2, COL2 * 0.38), constrained_layout=True)
    fig_l, ax_l = plt.subplots(figsize=(COL2, COL2 * 0.38), constrained_layout=True)
    fig_r, ax_r = plt.subplots(figsize=(COL2, COL2 * 0.38), constrained_layout=True)

    # --- 1) NRMSE vs coupling
    xc_tr, yc_tr = _sorted_xy(coupl, y_tr)
    xc_va, yc_va = _sorted_xy(coupl, y_va)
    drew = False
    ax_c.cla()
    if xc_tr is not None:
        ax_c.plot(xc_tr, yc_tr, marker="o", label="Train NRMSE")
        drew = True
    if xc_va is not None:
        ax_c.plot(xc_va, yc_va, marker="s", linestyle="--", label="Val NRMSE")
        drew = True
    ax_c.set_xlabel("coupling coefficient, 1/m")
    ax_c.set_ylabel("NRMSE")
    # if logx:
    #     ax_c.set_xscale("log")
    if logy:
        ax_c.set_yscale("log")
    if drew:
        ax_c.legend()
    if save_figs:
        try:
            fig_c.savefig(p_c)
        except Exception as e:
            print(f"[warn] savefig coupling failed: {e}")

    # --- 2) NRMSE vs L_coupling
    xl_tr, yl_tr = _sorted_xy(Lc, y_tr)
    xl_va, yl_va = _sorted_xy(Lc, y_va)
    drew = False
    ax_l.cla()
    if xl_tr is not None:
        ax_l.plot(xl_tr, yl_tr, marker="o", label="Train NRMSE")
        drew = True
    if xl_va is not None:
        ax_l.plot(xl_va, yl_va, marker="s", linestyle="--", label="Val NRMSE")
        drew = True
    ax_l.set_xlabel("coupling length, m")
    ax_l.set_ylabel("NRMSE")
    if logx:
        ax_l.set_xscale("log")
    if logy:
        ax_l.set_yscale("log")
    if drew:
        ax_l.legend()
    if save_figs:
        try:
            fig_l.savefig(p_l)
        except Exception as e:
            print(f"[warn] savefig L_coupling failed: {e}")

    # --- 3) NRMSE vs radius
    xr_tr, yr_tr = _sorted_xy(rad, y_tr)
    xr_va, yr_va = _sorted_xy(rad, y_va)
    drew = False
    ax_r.cla()
    if xr_tr is not None:
        ax_r.plot(xr_tr, yr_tr, marker="o", label="Train NRMSE")
        drew = True
    if xr_va is not None:
        ax_r.plot(xr_va, yr_va, marker="s", linestyle="--", label="Val NRMSE")
        drew = True
    ax_r.set_xlabel("inter-core radius, µm")
    ax_r.set_ylabel("NRMSE")
    if logx:
        ax_r.set_xscale("log")
    if logy:
        ax_r.set_yscale("log")
    if drew:
        ax_r.legend()
    if save_figs:
        try:
            fig_r.savefig(p_r)
        except Exception as e:
            print(f"[warn] savefig radius failed: {e}")

    # экспорт объединённых данных (полезно для отслеживания)
    if save_figs:
        try:
            combined_df.to_csv(p_csv, index=False)
        except Exception as e:
            print(f"[warn] write combined csv failed: {e}")


def plot_mg_dataset_split_for_article(cfg,
                                      target_series: Optional[np.ndarray] = None,
                                      *,
                                      title: str = "Mackey--Glass series with chronological data split",
                                      save_fig: Optional[bool] = None,
                                      explicit_path: Optional[Union[str, Path]] = None,
                                      show_plot: bool = True):
    """
    Публикационный график для статьи:
      1) показывает только ряд Mackey--Glass;
      2) показывает только train, validation, test;
      3) не показывает target shift, внутренний initial drop и temporal masks отдельными областями.

    Важно:
      • warmup генератора задачи уже отброшен в target_series;
      • cfg.training.washout — внутреннее техническое имя для дополнительного initial drop;
      • cfg.training.target_shift нужен для one-step-ahead задачи, но отдельной областью не рисуется.
    """
    if str(getattr(cfg, "task_name", "")) != "mackey_glass":
        raise ValueError("plot_mg_dataset_split_for_article is intended only for task_name='mackey_glass'")

    if target_series is None:
        _, target_full, task_warmup = _task_sequences_full_for_debug("mackey_glass", cfg.task_cfg)
        target_series = target_full[int(task_warmup):]

    target_series = np.asarray(target_series, dtype=float).reshape(-1)
    if target_series.size < 10:
        raise ValueError("Mackey--Glass series is too short for plotting")

    M_eff = cfg.mask.mask_size if str(cfg.variant).startswith("temporal_") else 1
    initial_drop_samples = get_washout_samples(cfg)
    initial_drop_syms = int(np.ceil(int(initial_drop_samples) / max(1, int(M_eff))))

    shift_syms = int(getattr(cfg.training, "target_shift", 0) or 0)
    shift_syms = max(0, shift_syms)

    usable_len = int(target_series.size) - initial_drop_syms - shift_syms
    if usable_len <= 0:
        raise ValueError(
            "Not enough Mackey--Glass symbols after initial_drop and target_shift: "
            f"S={target_series.size}, initial_drop={initial_drop_syms}, target_shift={shift_syms}"
        )

    train_frac = float(getattr(cfg.training, "train_frac", 0.8))
    val_frac = float(getattr(cfg.training, "val_frac", 0.1))
    sl_train, sl_val, sl_test = split_train_val_test(usable_len, train_frac, val_frac)

    n_train = int(sl_train.stop - sl_train.start)
    n_val = int(sl_val.stop - sl_val.start)
    n_test = int(sl_test.stop - sl_test.start)

    i_drop_L = 0
    i_drop_R = initial_drop_syms
    i_tr_L = i_drop_R
    i_tr_R = i_tr_L + n_train
    i_va_L = i_tr_R
    i_va_R = i_va_L + n_val
    i_te_L = i_va_R
    i_te_R = i_te_L + n_test

    visible_len = i_te_R
    if visible_len > target_series.size:
        visible_len = int(target_series.size)

    x = np.arange(visible_len, dtype=int)
    y = target_series[:visible_len]

    fig, ax = plt.subplots(figsize=(COL2, COL2 * 0.38), constrained_layout=True)

    ax.plot(x, y, linewidth=1.1, label="Mackey--Glass series")
    ax.set_xlim(int(x[0]), int(x[-1]))
    ax.margins(x=0.0)

    def span_if(a, b, color, label, alpha=0.16):
        a, b = int(a), int(b)
        if b - a < 1:
            return
        a_plot = max(a, int(x[0]))
        b_plot = min(b, int(x[-1]) + 1)
        if b_plot - a_plot < 1:
            return
        ax.axvspan(a_plot, b_plot, color=color, alpha=alpha, label=label)

    span_if(i_drop_L, i_drop_R, "#888888", "initial drop")
    span_if(i_tr_L, i_tr_R, "#2ca02c", "train")
    span_if(i_va_L, i_va_R, "#9467bd", "validation")
    span_if(i_te_L, i_te_R, "#d62728", "test")

    for xpos in (i_drop_R, i_tr_R, i_va_R):
        if int(x[0]) < xpos < int(x[-1]):
            ax.axvline(xpos, color="k", linewidth=0.8, alpha=0.25)

    ax.set_title(title, loc="left")
    ax.set_xlabel("symbol index")
    ax.set_ylabel("normalized amplitude")

    split_text = (
        f"initial drop: {initial_drop_syms}, "
        f"train/validation/test: {n_train}/{n_val}/{n_test}"
    )
    ax.text(
        0.01,
        0.02,
        split_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70),
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        leg = ax.legend(uniq.values(), uniq.keys(), loc="upper right", frameon=True)
        leg.get_frame().set_facecolor((1, 1, 1, 0.70))
        leg.get_frame().set_edgecolor((0, 0, 0, 0.25))

    if save_fig is None:
        save_fig = bool(getattr(cfg.reservoir, "save_figs", False))

    _maybe_savefig(
        fig,
        "mg_dataset_split_for_article",
        explicit_path=explicit_path,
        enabled=save_fig,
    )

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# Article figures: final overrides/additions
# =============================================================================


def _hex_core_centers_for_scheme(ring_count: int,
                                 core_pitch_um: float) -> np.ndarray:
    """
    Координаты центров сердцевин для hexagonal MCF.

    Важно: используется та же логика, что и в fiber_base_functions.get_coupling_coefficients(...),
    чтобы объединённый рисунок повторял одиночные схемы волокна.
    """
    from fiberprop.fiber_geometry import CoreConfig, make_eq_mask, get_core_count

    ring_count = int(ring_count)
    if ring_count < 0:
        raise ValueError("ring_count must be >= 0")

    core_count = int(get_core_count(CoreConfig.hexagonal, ring_count))
    mask_array = make_eq_mask(
        core_configuration=CoreConfig.hexagonal,
        size=core_count,
        ring_count=ring_count,
        display_debug_info=False,
    )
    distance_to_fiber_center = [float(k) * float(core_pitch_um) for k in range(ring_count + 1)]

    core_center_coords = []
    for i in range(core_count):
        dimensional_radius = np.sqrt(
            (mask_array[i].number_2d_x * 0.5) ** 2 +
            (mask_array[i].number_2d_y * 0.5 * np.sqrt(3.0)) ** 2
        )
        ring_index = int(np.ceil(dimensional_radius))
        x_coord = distance_to_fiber_center[ring_index] * mask_array[i].number_2d_x * 0.5 / max(ring_index, 1)
        y_coord = distance_to_fiber_center[ring_index] * mask_array[i].number_2d_y * 0.5 * np.sqrt(3.0) / max(ring_index, 1)
        core_center_coords.append((x_coord, y_coord))

    return np.asarray(core_center_coords, dtype=float)


def plot_mcf_fiber_schemes_for_article(*,
                                        core_pitch_um: float = 30.0,
                                        core_radius_um: float = 2.95,
                                        cladding_diameter_um: float = 250.0,
                                        title: Optional[str] = None,
                                        save_fig: bool = True,
                                        explicit_path: Optional[Union[str, Path]] = None,
                                        show_plot: bool = True):
    """
    Рисует две схемы MCF в одном файле в стиле Fig. 2 статьи:
      (a) 7-core hexagonal MCF;
      (b) 19-core hexagonal MCF.

    Панели рисуются через уже существующую функцию fiber_base_functions.plot_core_centers(...),
    чтобы одиночные и объединённые схемы имели одинаковую геометрию и стиль.
    """
    from fiberprop.fiber_base_functions import plot_core_centers

    fig, axes = plt.subplots(1, 2, figsize=(COL2, COL2 * 0.47), constrained_layout=False)
    fig.subplots_adjust(left=0.090, right=0.985, bottom=0.205, top=0.960, wspace=0.30)

    coords_7 = _hex_core_centers_for_scheme(1, core_pitch_um)
    coords_19 = _hex_core_centers_for_scheme(2, core_pitch_um)

    plot_core_centers(
        coords_7,
        core_radius_um,
        cladding_diameter_um,
        title="",
        color="red",
        annotate_indices=False,
        scale_bar_um=None,
        save_path=None,
        show=False,
        ax=axes[0],
    )
    plot_core_centers(
        coords_19,
        core_radius_um,
        cladding_diameter_um,
        title="",
        color="red",
        annotate_indices=False,
        scale_bar_um=None,
        save_path=None,
        show=False,
        ax=axes[1],
    )

    for ax, panel_label in zip(axes, ("(a)", "(b)")):
        ax.text(
            0.045,
            0.955,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
        )

    if title:
        fig.suptitle(title, x=0.090, ha="left")

    _maybe_savefig(
        fig,
        "mcf_fiber_schemes_for_article",
        explicit_path=explicit_path,
        enabled=save_fig,
    )

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_fit_predict_scatter(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             tau=None,
                             n_power=None,
                             plot_params: Optional[dict] = None,
                             *,
                             cfg: Optional[ExperimentConfig] = None,
                             title: Optional[str] = None,
                             split_name: str = "validation",
                             max_points: Optional[int] = None,
                             save_fig: Optional[bool] = None,
                             explicit_path: Optional[Union[str, Path]] = None) -> float:
    """
    Рисует scatter-график качества readout: истинные значения против прогноза.

    Функция строит квадратную диаграмму ``ground truth``--``prediction`` для выбранного
    сегмента данных и добавляет диагональ идеального прогноза. Метрика NRMSE считается
    по всем переданным точкам после выравнивания длин ``y_true`` и ``y_pred``; параметр
    ``max_points`` ограничивает только число отображаемых точек и не влияет на метрику.

    Параметры
    ---------
    y_true, y_pred:
        Одномерные массивы истинных и предсказанных значений. Массивы приводятся к
        ``float`` и обрезаются до общей минимальной длины.
    tau, n_power:
        Необязательные параметры для совместимости со старыми вызовами. В новой
        отрисовке не используются, кроме случая, когда пользователь явно передаёт
        собственный ``title`` через внешний код.
    plot_params:
        Необязательный словарь с настройками отрисовки. Поддерживаются ключи
        ``figsize``, ``s`` и ``scatter_alpha``.
    cfg:
        Конфигурация эксперимента. Если задана, используется для подписи задачи и
        для определения флага сохранения рисунков.
    title:
        Необязательное имя задачи для заголовка. Если не задано, имя берётся из
        ``cfg``. Итоговый заголовок всегда имеет вид
        ``<имя задачи>: <segment> NRMSE=<value>``.
    split_name:
        Имя сегмента данных: например ``"train"``, ``"validation"`` или ``"test"``.
    max_points:
        Максимальное число точек, которое нужно отрисовать. Метрика считается по
        всем валидным точкам.
    save_fig:
        Явно включает или отключает сохранение. Если ``None`` и задан ``cfg``, берётся
        ``cfg.reservoir.save_figs``.
    explicit_path:
        Полный путь для сохранения. Если задан, имеет приоритет над автоматическим
        именем файла.

    Возвращает
    ----------
    float
        NRMSE между ``y_true`` и ``y_pred`` по всем переданным валидным точкам.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    n = min(y_true.shape[0], y_pred.shape[0])
    def _fit_predict_title_base(raw_title: Optional[str]) -> str:
        if raw_title is None:
            if cfg is not None:
                meta = task_debug_meta(getattr(cfg, "task_name", "mackey_glass"))
                raw_title = meta["task_title"]
            else:
                raw_title = "Fit-predict"

        base = str(raw_title).replace("\n", " ")
        redundant_title_part = " ".join(("training", "and", "prediction"))
        for phrase in (redundant_title_part, "fit-predict", "scatter"):
            base = base.replace(phrase, "")
        base = " ".join(base.split()).strip(" :")
        if ":" in base:
            base = base.split(":", 1)[0].strip()
        if base.endswith(" series"):
            base = base[:-len(" series")].strip()
        return base or "Fit-predict"

    title_base = _fit_predict_title_base(title)
    split_label = str(split_name).strip() or "split"

    if n == 0:
        fig, ax = plt.subplots(figsize=(COL2 * 0.62, COL2 * 0.62), constrained_layout=True)
        ax.set_title(f"{title_base}: {split_label} NRMSE=—", loc="left")
        ax.set_xlabel("ground truth")
        ax.set_ylabel("prediction")
        _maybe_savefig(fig, "fit_predict_scatter", explicit_path=explicit_path, enabled=bool(save_fig))
        plt.show()
        return float("nan")

    y_true = y_true[:n]
    y_pred = y_pred[:n]

    err = float("nan") if n < 2 else float(nrmse(y_true, y_pred))

    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(finite_mask):
        fig, ax = plt.subplots(figsize=(COL2 * 0.62, COL2 * 0.62), constrained_layout=True)
        ax.set_title(f"{title_base}: {split_label} NRMSE=—", loc="left")
        ax.set_xlabel("ground truth")
        ax.set_ylabel("prediction")
        _maybe_savefig(fig, "fit_predict_scatter", explicit_path=explicit_path, enabled=bool(save_fig))
        plt.show()
        return float("nan")

    yt = y_true[finite_mask]
    yp = y_pred[finite_mask]

    if max_points is not None and int(max_points) > 0 and yt.shape[0] > int(max_points):
        idx = np.linspace(0, yt.shape[0] - 1, int(max_points), dtype=int)
        yt_plot = yt[idx]
        yp_plot = yp[idx]
    else:
        yt_plot = yt
        yp_plot = yp

    min_v = float(min(np.min(yt), np.min(yp)))
    max_v = float(max(np.max(yt), np.max(yp)))

    if np.isclose(min_v, max_v):
        pad = 1.0 if np.isclose(min_v, 0.0) else 0.05 * abs(min_v)
    else:
        pad = 0.04 * (max_v - min_v)

    lo = min_v - pad
    hi = max_v + pad

    if plot_params is not None and "figsize" in plot_params:
        figsize = plot_params["figsize"]
    else:
        figsize = (COL2 * 0.62, COL2 * 0.62)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    marker_size = 7.0
    marker_alpha = 0.48
    if plot_params is not None:
        marker_size = float(plot_params.get("s", marker_size))
        marker_alpha = float(plot_params.get("scatter_alpha", marker_alpha))

    ax.scatter(
        yt_plot,
        yp_plot,
        s=marker_size,
        alpha=marker_alpha,
        linewidths=0.0,
        rasterized=yt_plot.shape[0] > 3000,
        label="samples",
    )

    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color="0.20", label="ideal prediction")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    for spine in ax.spines.values():
        spine.set_linewidth(float(plt.rcParams.get("axes.linewidth", 1.0)))
        spine.set_visible(True)

    ax.set_title(f"{title_base}: {split_label} NRMSE={err:.4g}", loc="left")
    ax.set_xlabel("ground truth")
    ax.set_ylabel("prediction")
    ax.legend(loc="upper left", frameon=True)

    if save_fig is None:
        save_fig = bool(getattr(getattr(cfg, "reservoir", None), "save_figs", False)) if cfg is not None else False

    safe_split = str(split_name).strip().replace(" ", "_") or "split"
    _maybe_savefig(
        fig,
        f"fit_predict_scatter_{safe_split}",
        explicit_path=explicit_path,
        enabled=save_fig,
    )
    plt.show()

    return err


def plot_mg_series_and_attractor_for_article(cfg,
                                            target_series: Optional[np.ndarray] = None,
                                            *,
                                            title: Optional[str] = None,
                                            save_fig: Optional[bool] = None,
                                            explicit_path: Optional[Union[str, Path]] = None,
                                            show_plot: bool = True,
                                            train_frac: float = 0.8,
                                            val_frac: float = 0.1,
                                            attractor_line_alpha: float = 0.82,
                                            attractor_linewidth: float = 0.9):
    """
    Финальный рисунок для статьи в стиле текущего Fig. 4:
      (a) слева — нормированный ряд Mackey-Glass с warm-up / train / validation / test;
      (b) справа — аттрактор Mackey-Glass в координатах (x(t), x(t-tau), x(t-2tau)).

    По умолчанию используется split 8:1:1, а warm-up берётся из cfg.task_cfg.warmup.
    Возвращает fig. Для сохранения используется общий _maybe_savefig и rcParams/savefig.format.
    """
    from matplotlib.ticker import FixedLocator
    from matplotlib.lines import Line2D

    if str(getattr(cfg, "task_name", "")) != "mackey_glass":
        raise ValueError("plot_mg_series_and_attractor_for_article is intended only for task_name='mackey_glass'")

    _input_full, target_full, task_warmup = _task_sequences_full_for_debug("mackey_glass", cfg.task_cfg)
    full_series = np.asarray(target_full, dtype=float).reshape(-1)
    task_warmup = int(task_warmup)

    if target_series is None:
        n_seq = int(getattr(cfg.task_cfg, "t_size", full_series.size - task_warmup))
    else:
        n_seq = int(np.asarray(target_series).reshape(-1).size)

    n_seq = max(1, min(n_seq, int(full_series.size) - task_warmup))
    if task_warmup + n_seq > full_series.size:
        raise ValueError("Mackey-Glass full series is too short for the requested warm-up and retained sequence")

    train_frac = float(train_frac)
    val_frac = float(val_frac)
    if train_frac < 0.0 or val_frac < 0.0 or train_frac + val_frac >= 1.0:
        raise ValueError("train_frac and val_frac must be non-negative and train_frac + val_frac < 1")

    n_train = int(n_seq * train_frac)
    n_val = int(n_seq * val_frac)
    n_test = int(n_seq - n_train - n_val)

    i_warm_L = 0
    i_warm_R = task_warmup
    i_tr_L = i_warm_R
    i_tr_R = i_tr_L + n_train
    i_va_L = i_tr_R
    i_va_R = i_va_L + n_val
    i_te_L = i_va_R
    i_te_R = i_te_L + n_test

    visible_len = int(i_te_R)
    x = np.arange(visible_len, dtype=int)
    y = full_series[:visible_len]

    retained = full_series[task_warmup:task_warmup + n_seq]
    tau_samples = max(1, int(round(float(cfg.task_cfg.tau) / float(cfg.task_cfg.dt))))
    off = 2 * tau_samples
    if retained.size <= off + 1:
        raise ValueError(f"Mackey-Glass retained series is shorter than 2tau: S={retained.size}, 2tau={off}")

    X = retained[off:]
    Y = retained[tau_samples:-tau_samples]
    Z = retained[:-off]
    L = X.shape[0]

    i_tr_attr = (0, n_train)
    i_va_attr = (n_train, n_train + n_val)
    i_te_attr = (n_train + n_val, n_train + n_val + n_test)

    fig = plt.figure(figsize=(COL2 * 1.58, COL2 * 0.52), constrained_layout=False)
    ax1 = fig.add_axes([0.055, 0.205, 0.600, 0.705])
    ax2 = fig.add_axes([0.680, 0.155, 0.305, 0.790], projection="3d")

    ax1.plot(x, y, linewidth=1.0, label="Mackey-Glass series", zorder=3)
    ax1.set_xlim(int(x[0]), int(x[-1]))
    ax1.margins(x=0.0)

    def span_if(a, b, color, label, alpha=0.16):
        a, b = int(a), int(b)
        if b - a < 1:
            return
        a_plot = max(a, int(x[0]))
        b_plot = min(b, int(x[-1]) + 1)
        if b_plot - a_plot < 1:
            return
        ax1.axvspan(a_plot, b_plot, color=color, alpha=alpha, label=label, zorder=1)

    span_if(i_warm_L, i_warm_R, "#888888", "warm-up")
    span_if(i_tr_L, i_tr_R, "#2ca02c", "train")
    span_if(i_va_L, i_va_R, "#9467bd", "validation")
    span_if(i_te_L, i_te_R, "#d62728", "test")

    for xpos in (i_warm_R, i_tr_R, i_va_R):
        if int(x[0]) < xpos < int(x[-1]):
            ax1.axvline(xpos, color="k", linewidth=0.8, alpha=0.25, zorder=2)

    ax1.set_title("Mackey-Glass series with chronological data split", loc="left", pad=7.0)
    ax1.set_xlabel("symbol index")
    ax1.set_ylabel("normalized amplitude")

    split_text = f"warm-up: {task_warmup}, train/validation/test: {n_train}/{n_val}/{n_test}"
    ax1.text(
        0.012,
        0.035,
        split_text,
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.5),
        zorder=5,
    )

    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        leg = ax1.legend(
            uniq.values(),
            uniq.keys(),
            loc="upper right",
            bbox_to_anchor=(0.985, 0.985),
            frameon=True,
            handlelength=1.8,
            labelspacing=0.25,
            borderpad=0.35,
        )
        leg.get_frame().set_facecolor((1, 1, 1, 0.74))
        leg.get_frame().set_edgecolor((0, 0, 0, 0.28))

    ax2.set_proj_type("ortho")
    ax2.view_init(elev=20, azim=-15)
    ax2.grid(False)

    xyz_all = np.concatenate([X, Y, Z])
    finite_xyz = xyz_all[np.isfinite(xyz_all)]
    lim = float(np.nanmax(np.abs(finite_xyz))) if finite_xyz.size else 1.0
    lim = max(1.6, 1.02 * lim)
    ticks = [-1.5, 0.0, 1.5]

    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_zlim(-lim, lim)
    ax2.set_box_aspect((1.0, 1.0, 0.86))

    for axis in (ax2.xaxis, ax2.yaxis, ax2.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0, 0, 0, 0))
        axis.set_major_locator(FixedLocator(ticks))

    ax2.set_xlabel(r"$x(t)$", labelpad=3.0)
    ax2.set_ylabel(r"$x(t-\tau)$", labelpad=3.0)
    ax2.set_zlabel(r"$x(t-2\tau)$", labelpad=3.0)

    line_alpha = float(np.clip(attractor_line_alpha, 0.0, 1.0))
    lw = float(attractor_linewidth)
    legend_items = []

    def plot_segment(name, color, bounds, min_len: int = 2):
        a, b = int(bounds[0]), int(bounds[1])
        aa, bb = max(a, off), min(b, retained.size)
        if bb - aa >= min_len:
            lo = max(0, min(aa - off, L))
            hi = max(0, min(bb - off, L))
            if hi - lo >= min_len:
                ax2.plot(
                    X[lo:hi],
                    Y[lo:hi],
                    Z[lo:hi],
                    color=color,
                    alpha=line_alpha,
                    linewidth=lw,
                    solid_capstyle="round",
                    label=name,
                )
                legend_items.append(Line2D([0], [0], color=color, linewidth=1.25, label=name))
                return True
        return False

    for name, color, bounds in (
            ("train", "#2ca02c", i_tr_attr),
            ("validation", "#9467bd", i_va_attr),
            ("test", "#d62728", i_te_attr)):
        plot_segment(name, color, bounds)

    if legend_items:
        uniq = {}
        for item in legend_items:
            uniq[item.get_label()] = item
        leg = ax2.legend(
            uniq.values(),
            uniq.keys(),
            bbox_to_anchor=(0.03, 0.98),
            loc="upper left",
            frameon=True,
            title="Segments:",
            handlelength=1.7,
            labelspacing=0.25,
            borderpad=0.35,
        )
        leg.get_frame().set_facecolor((1, 1, 1, 0.74))
        leg.get_frame().set_edgecolor((0, 0, 0, 0.30))

    fig.text(0.355, 0.050, "(a)", ha="center", va="center")
    fig.text(0.832, 0.050, "(b)", ha="center", va="center")

    if title:
        fig.suptitle(title, x=0.055, ha="left")

    if save_fig is None:
        save_fig = bool(getattr(cfg.reservoir, "save_figs", False))

    _maybe_savefig(
        fig,
        "mg_series_and_attractor_for_article",
        explicit_path=explicit_path,
        enabled=save_fig,
    )

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig

