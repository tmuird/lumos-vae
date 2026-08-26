"""
Spectral quality metrics and comparison utilities.
All peak detection operates in absolute physical space (cts/sec) - no L-inf normalisation.
"""

from __future__ import annotations

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, wasserstein_distance
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import linear_sum_assignment
from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse
from sklearn.metrics import mean_squared_error, auc

# -- SI-PSNR helpers -----------------------------------------------------------


def _zero_mean(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)


def _fix_range(gt: np.ndarray, x: np.ndarray) -> np.ndarray:
    a = np.sum(gt * x) / (np.sum(x * x) + 1e-10)
    return x * a


def _fix(gt: np.ndarray, x: np.ndarray) -> np.ndarray:
    gt_ = _zero_mean(gt)
    return _fix_range(gt_, _zero_mean(x))


# -- Core scalar metrics -------------------------------------------------------


def spectral_angle_mapper(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-10)
    b_n = b / (np.linalg.norm(b) + 1e-10)
    return float(np.degrees(np.arccos(np.clip(np.dot(a_n, b_n), -1.0, 1.0))))


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    return float(pearsonr(a.ravel(), b.ravel())[0])


def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(gt, pred, data_range=np.max(gt)))


def scale_invariant_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_std = np.std(gt) + 1e-10
    range_parameter = (np.max(gt) - np.min(gt)) / gt_std
    gt_ = _zero_mean(gt) / gt_std
    pred_fixed = _fix(gt_, pred)
    mse = mean_squared_error(gt_, pred_fixed)
    return (
        float("inf") if mse == 0 else float(10 * np.log10((range_parameter**2) / mse))
    )


def nrmse(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(normalized_root_mse(gt, pred, normalization="min-max"))


def rmse(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(gt, pred)))


def wasserstein_dist(gt: np.ndarray, pred: np.ndarray, wn_axis: np.ndarray) -> float:
    gt_pos = np.clip(gt, 0, None)
    pred_pos = np.clip(pred, 0, None)
    gt_sum = np.sum(gt_pos) + 1e-10
    pred_sum = np.sum(pred_pos) + 1e-10
    return float(
        wasserstein_distance(
            wn_axis,
            wn_axis,
            u_weights=gt_pos / gt_sum,
            v_weights=pred_pos / pred_sum,
        )
    )


# -- SavGol window optimisation ------------------------------------------------

_OPT_METRIC_CHOICES = ("wasserstein", "psnr", "si_psnr", "pearson", "nrmse")


def _opt_score(
    gt: np.ndarray, pred: np.ndarray, wn_axis: np.ndarray, metric: str
) -> float:
    """Return a cost (lower = better) for the given metric."""
    if metric == "wasserstein":
        return wasserstein_dist(gt, pred, wn_axis)
    if metric == "psnr":
        v = psnr(gt, pred)
        return -v if np.isfinite(v) else np.inf
    if metric == "si_psnr":
        v = scale_invariant_psnr(gt, pred)
        return -v if np.isfinite(v) else np.inf
    if metric == "pearson":
        return -pearson_r(gt, pred)
    if metric == "nrmse":
        return nrmse(gt, pred)
    raise ValueError(
        f"Unknown opt_metric {metric!r}. Choose from {_OPT_METRIC_CHOICES}."
    )


def optimise_savgol_window(
    rate_inputs: np.ndarray,
    gts: np.ndarray,
    wn_axis: np.ndarray,
    baseline_fn,
    poly_order: int = 3,
    window_min: int = 5,
    window_max: int = 101,
    n_samples: int = 500,
    rng=None,
    opt_metric: str = "wasserstein",
    poly_order_candidates: tuple | list | None = None,
) -> tuple[int, int, float]:
    """
    Joint grid-search over (window, poly_order) minimising mean cost on a
    random subset of samples.

    Parameters
    ----------
    rate_inputs          : [N, W]  input spectra (counts/sec)
    gts                  : [N, W]  noiseless GT spectra
    baseline_fn          : callable  accepts smoothed spectrum, returns (baseline, info)
    poly_order           : fallback order used when poly_order_candidates is None
    poly_order_candidates: sequence of ints to search jointly with the window.
                           None uses [poly_order] only (single-order search).
                           Typical: [2, 3, 4, 5]
    opt_metric           : "wasserstein" | "psnr" | "si_psnr" | "pearson" | "nrmse"
                           For PSNR / SI-PSNR / Pearson the negative value is minimised.
    rng                  : np.random.Generator or None

    Returns
    -------
    (best_window, best_poly_order, best_cost)
    """
    if opt_metric not in _OPT_METRIC_CHOICES:
        raise ValueError(
            f"opt_metric={opt_metric!r}. Choose from {_OPT_METRIC_CHOICES}."
        )

    orders = (
        list(poly_order_candidates)
        if poly_order_candidates is not None
        else [poly_order]
    )

    rng = rng or np.random.default_rng(0)
    n = min(n_samples, len(rate_inputs))
    idx = rng.choice(len(rate_inputs), n, replace=False)
    sub_in = rate_inputs[idx]
    sub_gt = gts[idx]

    best_w = window_min if window_min % 2 == 1 else window_min + 1
    best_po = orders[0]
    best_cost = np.inf

    for po in orders:
        # window must be odd and strictly greater than poly_order
        w_start = window_min if window_min % 2 == 1 else window_min + 1
        for w in range(w_start, window_max + 1, 2):
            if w <= po:
                continue
            scores = []
            for rate, gt in zip(sub_in, sub_gt):
                try:
                    sm = savgol_filter(rate.astype(np.float64), w, po)
                    base, _ = baseline_fn(sm)
                    scores.append(_opt_score(gt, sm - base, wn_axis, opt_metric))
                except Exception:
                    pass
            if scores:
                mean_cost = float(np.mean(scores))
                if mean_cost < best_cost:
                    best_cost = mean_cost
                    best_w = w
                    best_po = po

    return best_w, best_po, best_cost


# -- Top-K determination -------------------------------------------------------


def determine_top_k(
    all_gt: np.ndarray,
    wn_axis: np.ndarray,
    prominence: float,
    distance: int,
    strategy: str = "gt_mean",
    fixed_k: int = 50,
) -> tuple[int, float, int]:
    """
    Compute the per-sample predicted peak budget from GT peak statistics.

    Returns
    -------
    (top_k, mean_count, median_count)
    """
    counts = np.array(
        [
            len(find_peaks(all_gt[i], prominence=prominence, distance=distance)[0])
            for i in range(len(all_gt))
        ]
    )
    mean_c = float(counts.mean())
    median_c = int(np.median(counts))

    if strategy == "gt_mean":
        top_k = max(1, int(round(mean_c)))
    elif strategy == "gt_median":
        top_k = max(1, median_c)
    else:
        top_k = fixed_k

    return top_k, mean_c, median_c


# -- Peak detection metrics (absolute space) -----------------------------------


def peak_match_stats(
    ref: np.ndarray,
    pred: np.ndarray,
    wn_axis: np.ndarray,
    ref_prominence: float = 0.02,
    pred_prominence: float = 0.0,
    distance: int = 3,
    match_threshold_cm: float = 5.0,
    top_k_pred: int | None = None,
) -> dict:
    """
    Hungarian-matched peak recall / precision in absolute cts/sec space.

    top_k_pred : retain only the top-k most prominent predicted peaks.
                 Prevents noisy methods winning by over-detection.
    """
    ref_pk, _ = find_peaks(ref, prominence=ref_prominence, distance=distance)
    pred_pk, props = find_peaks(pred, prominence=pred_prominence, distance=distance)

    if top_k_pred is not None and len(pred_pk) > top_k_pred:
        order = np.argsort(props["prominences"])[-top_k_pred:]
        pred_pk = pred_pk[order]

    if len(ref_pk) == 0:
        return dict(
            n_ref=0,
            n_pred=len(pred_pk),
            mean_pos_err_cm=None,
            recall=0.0,
            precision=0.0,
            f1=0.0,
        )
    if len(pred_pk) == 0:
        return dict(
            n_ref=len(ref_pk),
            n_pred=0,
            mean_pos_err_cm=None,
            recall=0.0,
            precision=0.0,
            f1=0.0,
        )

    cost = np.abs(wn_axis[ref_pk][:, None] - wn_axis[pred_pk][None, :])
    row, col = linear_sum_assignment(cost)
    matched = cost[row, col]
    tp_mask = matched < match_threshold_cm
    n_tp = int(tp_mask.sum())

    recall = n_tp / len(ref_pk)
    precision = n_tp / len(pred_pk)
    f1 = 2 * recall * precision / (recall + precision + 1e-8)
    mean_err = float(matched[tp_mask].mean()) if n_tp > 0 else None

    return dict(
        n_ref=len(ref_pk),
        n_pred=len(pred_pk),
        mean_pos_err_cm=mean_err,
        recall=recall,
        precision=precision,
        f1=f1,
    )


def evaluate_all(
    reference: np.ndarray,
    candidates: dict[str, np.ndarray],
    wn_axis: np.ndarray,
    distance: int = 3,
    match_threshold_cm: float = 5.0,
    gt_prominence: float = 0.02,
    pred_prominence: float = 0.0,
    top_k_pred: int | None = None,
) -> dict[str, dict]:
    """All scalar metrics without L-inf normalisation. top_k_pred caps pred peaks."""
    ref = reference.squeeze()
    out = {}
    for name, spec in candidates.items():
        s = spec.squeeze()
        stats = peak_match_stats(
            ref,
            s,
            wn_axis,
            ref_prominence=gt_prominence,
            pred_prominence=pred_prominence,
            distance=distance,
            match_threshold_cm=match_threshold_cm,
            top_k_pred=top_k_pred,
        )
        out[name] = dict(
            sam=spectral_angle_mapper(ref, s),
            r=pearson_r(ref, s),
            psnr=psnr(ref, s),
            si_psnr=scale_invariant_psnr(ref, s),
            nrmse=nrmse(ref, s),
            rmse=rmse(ref, s),
            wd=wasserstein_dist(ref, s, wn_axis),
            peaks=stats,
        )
    return out


# -- Global PR curve -----------------------------------------------------------


def compute_global_pr_curve(
    global_gt_peaks_dict: dict,
    global_pred_peaks_list: list,
    total_gt: int,
    wn_axis: np.ndarray,
    match_threshold_cm: float = 5.0,
) -> tuple:
    """
    Global PR curve over the entire dataset in wavenumber space.

    Returns
    -------
    precisions, recalls, f1s, confidences, ap, max_f1, tp_errors_cm
    """
    if total_gt == 0 or not global_pred_peaks_list:
        empty = np.array([])
        return (
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            0.0,
            0.0,
            empty,
        )

    unmatched = copy.deepcopy(global_gt_peaks_dict)
    global_pred_peaks_list = sorted(
        global_pred_peaks_list, key=lambda x: x[2], reverse=True
    )

    n = len(global_pred_peaks_list)
    tps = np.zeros(n)
    fps = np.zeros(n)
    tp_errs = []

    for i, (samp_id, pk_idx, conf) in enumerate(global_pred_peaks_list):
        close = [
            (g, abs(wn_axis[g] - wn_axis[pk_idx]))
            for g in unmatched.get(samp_id, [])
            if abs(wn_axis[g] - wn_axis[pk_idx]) <= match_threshold_cm
        ]
        if close:
            best_g, best_d = min(close, key=lambda x: x[1])
            unmatched[samp_id].remove(best_g)
            tps[i] = 1
            tp_errs.append(best_d)
        else:
            fps[i] = 1

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    recalls = cum_tp / total_gt
    denom = cum_tp + cum_fp
    precisions = np.divide(cum_tp, denom, out=np.ones_like(cum_tp), where=denom > 0)

    precisions = np.concatenate(([1.0], precisions))
    recalls = np.concatenate(([0.0], recalls))

    ap = float(auc(recalls, precisions))
    f1s = np.zeros_like(precisions)
    mask = (precisions + recalls) > 0
    f1s[mask] = (
        2 * (precisions[mask] * recalls[mask]) / (precisions[mask] + recalls[mask])
    )

    return (
        precisions,
        recalls,
        f1s,
        np.zeros(len(precisions)),
        ap,
        float(np.max(f1s)),
        np.array(tp_errs),
    )


# -- Height error --------------------------------------------------------------


def height_error_at_threshold(
    gt_peaks_dict: dict,
    gt_heights_dict: dict,
    pred_peaks_list: list,
    pred_heights_dict: dict,
    wn_axis: np.ndarray,
    match_threshold_cm: float,
) -> float:
    """Mean |gt_height - pred_height| (cts/s) for TP peaks at one threshold."""
    if not pred_peaks_list:
        return np.nan

    unmatched = copy.deepcopy(gt_peaks_dict)
    pred_sorted = sorted(pred_peaks_list, key=lambda x: x[2], reverse=True)
    errors = []

    for samp_id, pk_idx, _ in pred_sorted:
        close = [
            (g, abs(wn_axis[g] - wn_axis[pk_idx]))
            for g in unmatched.get(samp_id, [])
            if abs(wn_axis[g] - wn_axis[pk_idx]) <= match_threshold_cm
        ]
        if close:
            best_g, _ = min(close, key=lambda x: x[1])
            unmatched[samp_id].remove(best_g)
            gt_h = gt_heights_dict.get(samp_id, {}).get(int(best_g), np.nan)
            pred_h = pred_heights_dict.get((samp_id, int(pk_idx)), np.nan)
            if not (np.isnan(gt_h) or np.isnan(pred_h)):
                errors.append(abs(gt_h - pred_h))

    return float(np.mean(errors)) if errors else np.nan


# -- Paper metrics at a fixed threshold ---------------------------------------


def compute_paper_metrics(
    gt_peaks_dict: dict,
    gt_heights_dict: dict,
    pred_peaks_list: list,
    pred_heights_dict: dict,
    wn_axis: np.ndarray,
    thresh: float,
    total_gt: int,
) -> dict:
    """
    AP, max-F1, mean horizontal shift (cm^-1), mean vertical height error (cts/s)
    at one fixed threshold.  Ready for the NeurIPS paper table.
    """
    _, _, _, _, ap, max_f1, tp_errs = compute_global_pr_curve(
        gt_peaks_dict,
        pred_peaks_list,
        total_gt,
        wn_axis=wn_axis,
        match_threshold_cm=thresh,
    )
    mean_shift = float(np.mean(tp_errs)) if len(tp_errs) > 0 else np.nan
    ht_err = height_error_at_threshold(
        gt_peaks_dict,
        gt_heights_dict,
        pred_peaks_list,
        pred_heights_dict,
        wn_axis,
        thresh,
    )
    return dict(ap=ap, max_f1=max_f1, mean_shift=mean_shift, height_err=ht_err)


# -- Plotting ------------------------------------------------------------------


def plot_global_pr_f1_curves(
    global_results: dict,
    palette: dict | None = None,
    suptitle: str | None = None,
) -> None:
    """Precision-Recall and F1-vs-Recall curves for all methods."""
    fig, (ax_pr, ax_f1) = plt.subplots(1, 2, figsize=(10, 4.5))
    palette = palette or {}

    for method, res in global_results.items():
        colour = palette.get(method, "#333333")
        prec, rec, f1s = res["precisions"], res["recalls"], res["f1s"]
        ap, max_f1 = res["auprc"], res["max_f1"]
        is_ours = "LUMOS" in method
        lw, z = (2.5, 5) if is_ours else (1.5, 3)

        ax_pr.plot(
            rec, prec, lw=lw, color=colour, zorder=z, label=f"{method} (AP={ap:.3f})"
        )
        ax_f1.plot(
            rec,
            f1s,
            lw=lw,
            color=colour,
            zorder=z,
            label=f"{method} (Max F1={max_f1:.3f})",
        )

    for ax, ylabel, title in [
        (ax_pr, "Precision", "Global Precision-Recall Curve"),
        (ax_f1, "F1 Score", "Global F1 Score vs Recall"),
    ]:
        ax.set_xlabel("Recall")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, linestyle=":", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)

    if suptitle:
        plt.suptitle(suptitle, fontweight="bold")
    plt.tight_layout()
