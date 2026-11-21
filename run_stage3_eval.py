"""Stage 3 Run 3c — Frozen Model Evaluation, Paired Statistical Tests, Stress Tests & Figures.

HARD CONSTRAINTS:
  - Evaluates strictly frozen checkpoints from Run 3b (no training, no model edits).
  - All tuning (Savitzky-Golay, Gaussian smoother, selective-hybrid thresholds) is performed
    strictly on ``val_sel`` (1,000 disjoint indices of ``val_id.npz``).
  - NEVER opens, reads, or references ``test_id.npz`` or ``test_ood.npz``.
  - Verifies ``verify_frozen_hashes()["ok"] == True`` and ``get_test_access_counts()["data_read"] == 0``
    before and after evaluation.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from src.baseline import fit_physical_odmr
from src.baseline_variants import fit_physical_with_prior
from src.config import CANONICAL_FREQS
from src.dataset_gen import get_test_access_counts
from src.hybrid import OpenNVPredictor, SelectiveHybridConfig, tune_selective_hybrid_thresholds
from src.integrity import verify_frozen_hashes
from src.noise_models import (
    sample_baseline_drift,
    sample_colored_noise,
    sample_mains_hum,
    sample_white_noise,
)
from src.physics import canonicalize_b_vec, generate_4axis_spectrum, solve_axis_transitions
from src.real_data import DEFAULT_RAW_EXP_DIR, load_and_validate_real_trace
from src.stats import bootstrap_summary_cis, mcnemar_paired_test, paired_bootstrap_diff_ci
from src.train import get_val_sel_indices, load_val_sel_arrays

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "synthetic"
CKPT_DIR = ROOT_DIR / "models" / "checkpoints"
ASSETS_DIR = ROOT_DIR / "assets"
RESULTS_JSON = ROOT_DIR / "models" / "stage3c_eval_results.json"

REGIME_NAMES = {0: "Hard", 1: "Medium", 2: "Easy"}


# ---------------------------------------------------------------------------
# Multiprocessing worker functions (top-level for Windows spawn pickling)
# ---------------------------------------------------------------------------


def _worker_fit_physical_plain(
    args: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int, int, Tuple[float, float]]
) -> Dict[str, Any]:
    spec, vm, warm_b, n_starts, poly_deg, hwhm_bounds = args
    t0 = time.perf_counter()
    res = fit_physical_odmr(
        CANONICAL_FREQS,
        np.asarray(spec, dtype=np.float64),
        valid_mask=np.asarray(vm, dtype=bool),
        warm_start_b_vec=None if warm_b is None else np.asarray(warm_b, dtype=np.float64),
        n_starts=int(n_starts),
        variant="plain",
        poly_degree=int(poly_deg),
        hwhm_bounds=hwhm_bounds,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "b_mag": float(res["b_mag"]),
        "b_vec": np.asarray(res["b_vec"], dtype=np.float64),
        "status": str(res["status"]),
        "runtime_ms": float(dt_ms),
    }


def _worker_fit_physical_prior(
    args: Tuple[np.ndarray, np.ndarray, int, int]
) -> Dict[str, Any]:
    spec, vm, n_starts, poly_deg = args
    t0 = time.perf_counter()
    res = fit_physical_with_prior(
        CANONICAL_FREQS,
        np.asarray(spec, dtype=np.float64),
        valid_mask=np.asarray(vm, dtype=bool),
        n_starts=int(n_starts),
        poly_degree=int(poly_deg),
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "b_mag": float(res["b_mag"]),
        "b_vec": np.asarray(res["b_vec"], dtype=np.float64),
        "status": str(res["status"]),
        "runtime_ms": float(dt_ms),
    }


def _worker_sg_plus_fitter(
    args: Tuple[np.ndarray, np.ndarray, int, int]
) -> Dict[str, Any]:
    spec, vm, win_len, polyorder = args
    t0 = time.perf_counter()
    y_sm = apply_savgol_single(spec, vm, win_len, polyorder)
    res = fit_physical_odmr(
        CANONICAL_FREQS,
        y_sm,
        valid_mask=np.asarray(vm, dtype=bool),
        n_starts=3,
        variant="plain",
        poly_degree=3,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "b_mag": float(res["b_mag"]),
        "b_vec": np.asarray(res["b_vec"], dtype=np.float64),
        "status": str(res["status"]),
        "runtime_ms": float(dt_ms),
    }


# ---------------------------------------------------------------------------
# Metric & signal processing helpers
# ---------------------------------------------------------------------------


def apply_savgol_single(
    spec: np.ndarray, valid_mask: np.ndarray, window_length: int, polyorder: int
) -> np.ndarray:
    y = np.asarray(spec, dtype=np.float64).copy()
    vm = np.asarray(valid_mask, dtype=bool)
    if np.all(vm):
        return savgol_filter(y, window_length=window_length, polyorder=polyorder)
    idx = np.where(vm)[0]
    if idx.size < window_length:
        return y
    out = y.copy()
    out[idx] = savgol_filter(y[idx], window_length=window_length, polyorder=polyorder)
    out[~vm] = 1.0
    return out


def apply_savgol_batch(
    x: np.ndarray, vm: np.ndarray, window_length: int, polyorder: int
) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    for i in range(x.shape[0]):
        out[i] = apply_savgol_single(x[i], vm[i], window_length, polyorder)
    return out


def apply_gaussian_batch(x: np.ndarray, vm: np.ndarray, sigma: float) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    for i in range(x.shape[0]):
        y = np.asarray(x[i], dtype=np.float64)
        m = np.asarray(vm[i], dtype=bool)
        if np.all(m):
            out[i] = gaussian_filter1d(y, sigma=sigma, mode="nearest")
        else:
            idx = np.where(m)[0]
            tmp = np.ones_like(y)
            if idx.size > 4:
                tmp[idx] = gaussian_filter1d(y[idx], sigma=sigma, mode="nearest")
            out[i] = tmp
    return out


def compute_b_metrics(
    b_pred_mt: np.ndarray,
    b_true_mt: np.ndarray,
    b_vec_pred: Optional[np.ndarray] = None,
    b_vec_true: Optional[np.ndarray] = None,
    runtime_ms: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    bp = np.asarray(b_pred_mt, dtype=np.float64)
    bt = np.asarray(b_true_mt, dtype=np.float64)
    err_ut = np.abs(bp - bt) * 1000.0
    signed_ut = (bp - bt) * 1000.0
    fail_100 = err_ut > 100.0

    if b_vec_pred is not None and b_vec_true is not None:
        vp = np.asarray(b_vec_pred, dtype=np.float64)
        vt = np.asarray(b_vec_true, dtype=np.float64)
        canon_vp = np.array([canonicalize_b_vec(v) for v in vp], dtype=np.float64)
        canon_vt = np.array([canonicalize_b_vec(v) for v in vt], dtype=np.float64)
        wedge_mt = np.linalg.norm(canon_vp - canon_vt, axis=1)
        wb = wedge_mt > 0.15
    else:
        wedge_mt = np.abs(bp - bt)
        wb = wedge_mt > 0.15

    out: Dict[str, Any] = {
        "n": int(err_ut.size),
        "median_ut": round(float(np.median(err_ut)), 2),
        "p90_ut": round(float(np.percentile(err_ut, 90)), 2),
        "p95_ut": round(float(np.percentile(err_ut, 95)), 2),
        "rms_ut": round(float(np.sqrt(np.mean(err_ut * err_ut))), 2),
        "mae_ut": round(float(np.mean(err_ut)), 2),
        "signed_bias_ut": round(float(np.mean(signed_ut)), 2),
        "fail_100_pct": round(float(np.mean(fail_100) * 100.0), 2),
        "wrong_basin_pct": round(float(np.mean(wb) * 100.0), 2),
    }
    if runtime_ms is not None:
        rt = np.asarray(runtime_ms, dtype=np.float64)
        out["mean_runtime_ms"] = round(float(np.mean(rt)), 2)
        out["median_runtime_ms"] = round(float(np.median(rt)), 2)
    return out


def compute_denoising_metrics(
    x_noisy: np.ndarray,
    y_hat: np.ndarray,
    y_clean: np.ndarray,
    valid_mask: np.ndarray,
    dip_thresh: float = 0.0015,
) -> Dict[str, Any]:
    """Compute exact per-spectrum denoising metrics over valid points."""
    x = np.asarray(x_noisy, dtype=np.float64)
    yh = np.asarray(y_hat, dtype=np.float64)
    yc = np.asarray(y_clean, dtype=np.float64)
    vm = np.asarray(valid_mask, dtype=bool)
    n = x.shape[0]

    rmse_in = np.zeros(n, dtype=np.float64)
    rmse_out = np.zeros(n, dtype=np.float64)
    dsnr_db = np.zeros(n, dtype=np.float64)
    dip_rmse = np.zeros(n, dtype=np.float64)
    pos_err_mhz = np.zeros(n, dtype=np.float64)

    for i in range(n):
        m = vm[i]
        d_in = x[i, m] - yc[i, m]
        d_out = yh[i, m] - yc[i, m]
        ri = float(np.sqrt(np.mean(d_in * d_in)))
        ro = float(np.sqrt(np.mean(d_out * d_out)))
        rmse_in[i] = ri
        rmse_out[i] = ro
        dsnr_db[i] = 20.0 * np.log10(max(ri, 1e-12) / max(ro, 1e-12))

        dip_m = m & ((1.0 - yc[i]) > dip_thresh)
        if np.any(dip_m):
            dd = yh[i, dip_m] - yc[i, dip_m]
            dip_rmse[i] = float(np.sqrt(np.mean(dd * dd)))
        else:
            dip_rmse[i] = ro

        # Primary detected dip minimum position error (MHz)
        f_v = CANONICAL_FREQS[m]
        idx_true = int(np.argmax(1.0 - yc[i, m]))
        idx_hat = int(np.argmax(1.0 - yh[i, m]))
        pos_err_mhz[i] = abs(float(f_v[idx_hat]) - float(f_v[idx_true]))

    return {
        "n": int(n),
        "mean_rmse_in": round(float(np.mean(rmse_in)), 6),
        "mean_rmse_out": round(float(np.mean(rmse_out)), 6),
        "mean_dsnr_db": round(float(np.mean(dsnr_db)), 3),
        "median_dsnr_db": round(float(np.median(dsnr_db)), 3),
        "mean_dip_rmse": round(float(np.mean(dip_rmse)), 6),
        "median_dip_rmse": round(float(np.median(dip_rmse)), 6),
        "mean_pos_err_mhz": round(float(np.mean(pos_err_mhz)), 4),
        "median_pos_err_mhz": round(float(np.median(pos_err_mhz)), 4),
        "p90_pos_err_mhz": round(float(np.percentile(pos_err_mhz, 90)), 4),
        "_per_sample_dip_rmse": dip_rmse,
        "_per_sample_pos_err_mhz": pos_err_mhz,
    }


def paired_bootstrap_one_sided_p(
    a: np.ndarray,
    b: np.ndarray,
    stat_fn: Any,
    margin: float = 0.0,
    n_boot: int = 4000,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Paired bootstrap test for H1: stat(a) - stat(b) < margin vs H0: >= margin."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    n = arr_a.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = stat_fn(arr_a[idx]) - stat_fn(arr_b[idx])
    pt = float(stat_fn(arr_a[None, :])[0] - stat_fn(arr_b[None, :])[0])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Conservative 1-sided bootstrap p-value: fraction of resamples with diff >= margin
    p_one_sided = float((np.sum(diffs >= margin) + 1.0) / (n_boot + 1.0))
    return {
        "point_diff": round(pt, 5),
        "ci_95": [round(float(lo), 5), round(float(hi), 5)],
        "margin": float(margin),
        "p_one_sided": p_one_sided,
    }


def holm_bonferroni_adjust(p_values: Sequence[float]) -> List[float]:
    """Return Holm-Bonferroni step-down adjusted p-values preserving original order."""
    m = len(p_values)
    order = np.argsort(p_values)
    adj = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = float(p_values[idx]) * float(m - rank)
        running_max = max(running_max, val)
        adj[idx] = min(1.0, running_max)
    return [float(x) for x in adj]


def load_split_subset(npz_name: str, indices: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Load non-test synthetic split arrays safely (explicitly forbids test_*.npz)."""
    if npz_name.startswith("test_"):
        raise PermissionError(f"LOCKED TEST VAULT: Stage 3 forbids opening {npz_name}!")
    with np.load(DATA_DIR / npz_name, allow_pickle=False) as d:
        if indices is None:
            return {k: d[k] for k in d.files}
        return {k: d[k][indices] for k in d.files if d[k].ndim >= 1 and d[k].shape[0] > int(np.max(indices))}


def run_parallel_fitter(
    worker_fn: Any, task_list: List[Any], max_workers: int = 4
) -> List[Dict[str, Any]]:
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(worker_fn, task_list, chunksize=8))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_bias_and_pareto(
    bin_stats: Dict[str, Any],
    pareto_methods: List[Dict[str, Any]],
    coverage_curve: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))

    # Panel 1: Signed bias vs |B| bin
    ax0 = axes[0]
    centers = bin_stats["bin_centers_mt"]
    ax0.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.6)
    ax0.plot(centers, bin_stats["fitter_bias_ut"], "s--", color="#d62728", lw=1.8, label="Frozen Physical Fitter")
    ax0.plot(centers, bin_stats["nn_bias_ut"], "o-", color="#1f77b4", lw=2.0, label="Model A (NN Direct)")
    ax0.plot(centers, bin_stats["model_b_bias_ut"], "^-.", color="#9467bd", lw=1.6, label="Model B (Sim2Real, NN Direct)")
    ax0.set_title("Signed Bias mean(B_pred - B_true) vs |B| on val_rep (N=3000)", fontsize=10, fontweight="bold")
    ax0.set_xlabel("True |B| Bin Center (mT)")
    ax0.set_ylabel("Signed Bias (uT)")
    ax0.grid(True, ls=":", alpha=0.45)
    ax0.legend(fontsize=8.2)

    # Panel 2: Pareto frontier (P90 Error vs Mean CPU Runtime)
    ax1 = axes[1]
    cc_rt = [c["mean_runtime_ms"] for c in coverage_curve]
    cc_p90 = [c["p90_ut"] for c in coverage_curve]
    ax1.plot(cc_rt, cc_p90, "-", color="#2ca02c", lw=2.0, alpha=0.85, label="Selective Hybrid Pareto Sweep")
    ax1.scatter(cc_rt, cc_p90, color="#2ca02c", s=24, alpha=0.85)

    markers = {"Frozen Fitter": ("s", "#d62728"), "Physical + Prior": ("D", "#8c564b"),
               "SG + Fitter": ("v", "#e377c2"), "Model A (NN Direct)": ("o", "#1f77b4"),
               "Model A (Warm-Start)": ("P", "#ff7f0e"), "Model A (Selective Hybrid)": ("*", "#2ca02c")}
    for m in pareto_methods:
        mk, col = markers.get(m["label"], ("o", "#333333"))
        sz = 150 if mk == "*" else 70
        ax1.scatter([m["mean_runtime_ms"]], [m["p90_ut"]], marker=mk, color=col, s=sz, zorder=5, label=m["label"])
        ax1.annotate(
            f"{m['label']}\n({m['mean_runtime_ms']:.0f}ms, {m['p90_ut']:.0f}uT)",
            (m["mean_runtime_ms"], m["p90_ut"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7.2,
        )
    ax1.set_xscale("log")
    ax1.set_title("Accuracy-Runtime Pareto Frontier (val_rep Paired N=600)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Mean CPU Runtime per Spectrum (ms, log scale)")
    ax1.set_ylabel("P90 |ΔB| (uT)")
    ax1.grid(True, which="both", ls=":", alpha=0.45)
    ax1.legend(fontsize=7.5, loc="upper right")

    fig.suptitle("OpenNV-Denoise Stage 3 — Calibration Bias & Speed-Accuracy Pareto Frontier", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_and_hallucination(
    unc_data: Dict[str, Any],
    halluc_data: Dict[str, Any],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    # Panel 1: Empirical 80% coverage by regime on val_rep vs val_ood
    ax0 = axes[0, 0]
    cats = ["Overall", "Hard", "Medium", "Easy"]
    x = np.arange(len(cats))
    w = 0.36
    cov_id = [unc_data["val_rep_cov80"][c] for c in cats]
    cov_ood = [unc_data["val_ood_cov80"][c] for c in cats]
    ax0.bar(x - w / 2, cov_id, width=w, color="#1f77b4", label="val_rep (ID, N=3000)")
    ax0.bar(x + w / 2, cov_ood, width=w, color="#ff7f0e", label="val_ood (OOD, N=1500)")
    ax0.axhline(80.0, color="#d62728", ls="--", lw=1.4, label="Nominal 80% Target")
    ax0.set_xticks(x)
    ax0.set_xticklabels(cats)
    ax0.set_ylim(0, 105)
    ax0.set_ylabel("Empirical Coverage (%)")
    ax0.set_title("80% Prediction Interval Coverage [q10, q90]", fontsize=10, fontweight="bold")
    ax0.grid(True, axis="y", ls=":", alpha=0.45)
    ax0.legend(fontsize=8.0)

    # Panel 2: Risk-Coverage curve on val_rep and val_ood
    ax1 = axes[0, 1]
    rc_id = unc_data["risk_coverage_val_rep"]
    rc_ood = unc_data["risk_coverage_val_ood"]
    ax1.plot(rc_id["coverage_pct"], rc_id["median_ut"], "o-", color="#1f77b4", lw=1.8, label="val_rep Median |ΔB|")
    ax1.plot(rc_id["coverage_pct"], rc_id["p90_ut"], "s--", color="#1f77b4", lw=1.5, alpha=0.75, label="val_rep P90 |ΔB|")
    ax1.plot(rc_ood["coverage_pct"], rc_ood["median_ut"], "o-", color="#ff7f0e", lw=1.8, label="val_ood Median |ΔB|")
    ax1.set_title("Risk-Coverage Curve (Sorted by Model A Uncertainty)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Retained Coverage (% Lowest Predicted Uncertainty)")
    ax1.set_ylabel("|ΔB| Error (uT)")
    ax1.grid(True, ls=":", alpha=0.45)
    ax1.legend(fontsize=8.0)

    # Panel 3: No-dip hallucination depth histogram
    ax2 = axes[1, 0]
    false_depths = np.asarray(halluc_data["no_dip"]["_raw_max_depths"], dtype=np.float64) * 100.0
    ax2.hist(false_depths, bins=25, color="#9467bd", alpha=0.8, edgecolor="black", lw=0.4)
    ax2.axvline(halluc_data["no_dip"]["p50_depth"] * 100.0, color="black", ls="--", lw=1.3,
                label=f"P50={halluc_data['no_dip']['p50_depth']*100:.2f}%")
    ax2.axvline(halluc_data["no_dip"]["p90_depth"] * 100.0, color="#d62728", ls="-.", lw=1.3,
                label=f"P90={halluc_data['no_dip']['p90_depth']*100:.2f}%")
    ax2.set_title("Hallucination Test (a): Max False Dip on No-Dip Spectra (N=200)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Max Predicted Absorption Depth (% of Baseline)")
    ax2.set_ylabel("Count")
    ax2.grid(True, ls=":", alpha=0.45)
    ax2.legend(fontsize=8.0)

    # Panel 4: Dip-shift tracking & Spurious dip attenuation summary
    ax3 = axes[1, 1]
    shifts = halluc_data["dip_shift"]["shifts_mhz"]
    shift_errs = halluc_data["dip_shift"]["median_abs_err_mhz"]
    ax3.bar([f"{s:+.0f} MHz" for s in shifts], shift_errs, color="#2ca02c", alpha=0.85, width=0.55)
    ax3.set_title(
        f"Dip-Shift Tracking Median Error (MHz)\nSpurious Wing-Dip Attenuation: "
        f"{halluc_data['spurious_dip']['median_attenuation_pct']:.1f}%",
        fontsize=9.5,
        fontweight="bold",
    )
    ax3.set_ylabel("Median |f_pred - f_true| (MHz)")
    ax3.grid(True, axis="y", ls=":", alpha=0.45)

    fig.suptitle("OpenNV-Denoise Stage 3 — Uncertainty Calibration & Hallucination Stress Tests", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_denoising_and_real_examples(
    syn_examples: List[Dict[str, Any]],
    real_examples: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.0))

    # Row 1: 3 synthetic val_rep examples (Hard, Medium, Easy)
    for col, ex in enumerate(syn_examples[:3]):
        ax = axes[0, col]
        ax.plot(CANONICAL_FREQS, ex["x_noisy"], color="#bbbbbb", lw=0.8, alpha=0.85, label="Noisy Input")
        ax.plot(CANONICAL_FREQS, ex["y_sg"], color="#ff7f0e", lw=1.2, ls="--", alpha=0.9, label="Tuned Savitzky-Golay")
        ax.plot(CANONICAL_FREQS, ex["y_clean"], color="black", lw=1.5, label="Ground-Truth Clean")
        ax.plot(CANONICAL_FREQS, ex["y_nn"], color="#1f77b4", lw=1.7, label="Model A Denoised")
        ax.set_title(
            f"val_rep {ex['regime']} | |B|={ex['b_true']:.2f} mT | SNR_w={ex['snr_w']:.2f}\n"
            f"NN |ΔB|={ex['nn_err_ut']:.1f} uT vs Fitter {ex['fit_err_ut']:.1f} uT",
            fontsize=9.0,
            fontweight="bold",
        )
        ax.set_xlabel("Frequency (MHz)")
        if col == 0:
            ax.set_ylabel("Normalized Fluorescence")
        ax.grid(True, ls=":", alpha=0.4)
        if col == 0:
            ax.legend(fontsize=7.3, loc="lower left")

    # Row 2: Real check traces (small_field_3 clean, small_field_3 + noise SNR~1, small_field_4 + noise SNR~1)
    for col, ex in enumerate(real_examples[:3]):
        ax = axes[1, col]
        ax.plot(CANONICAL_FREQS, ex["x_input"], color="#bbbbbb", lw=0.9, alpha=0.85, label=ex["input_label"])
        ax.plot(CANONICAL_FREQS, ex["y_ref"], color="black", lw=1.3, label="Original Real Trace")
        ax.plot(CANONICAL_FREQS, ex["y_model_b"], color="#2ca02c", lw=1.7, label="Model B Denoised")
        ax.set_title(ex["title"], fontsize=8.8, fontweight="bold")
        ax.set_xlabel("Frequency (MHz)")
        if col == 0:
            ax.set_ylabel("Normalized Fluorescence")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(fontsize=7.3, loc="lower left")

    fig.suptitle(
        "OpenNV-Denoise Stage 3 — Synthetic Denoising Examples (val_rep) & Real trianam Check Traces (Model B)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Run 3c pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    torch.set_num_threads(4)
    t_all = time.perf_counter()

    # 0. Strict integrity & test-vault check BEFORE evaluation
    h_pre = verify_frozen_hashes()
    assert h_pre["ok"], f"Frozen hash verification FAILED before Run 3c: {h_pre}"
    v_pre = get_test_access_counts()
    assert v_pre["data_read"] == 0, f"Test vault breach before Run 3c: {v_pre}"
    print(f"[PRE-CHECK] Frozen hashes OK ({h_pre['checked_files']} files), test data_read = {v_pre['data_read']}", flush=True)

    # Load frozen checkpoints from Run 3b
    pred_a_s0 = OpenNVPredictor.from_checkpoint(str(CKPT_DIR / "model_a_seed0.pt"))
    pred_a_s1 = OpenNVPredictor.from_checkpoint(str(CKPT_DIR / "model_a_seed1.pt"))
    pred_a_s2 = OpenNVPredictor.from_checkpoint(str(CKPT_DIR / "model_a_seed2.pt"))
    pred_b = OpenNVPredictor.from_checkpoint(str(CKPT_DIR / "model_b_sim2real.pt"))

    # Calibrate single-spectrum single-thread NN inference latency (N=50)
    val_sel = load_val_sel_arrays()
    t_nn_lat = []
    for i in range(50):
        out_1 = pred_a_s0.predict_single(val_sel["X_noisy"][i], val_sel["valid_mask"][i], mode="nn_direct")
        t_nn_lat.append(out_1["nn_runtime_ms"])
    nn_single_ms_mean = float(np.mean(t_nn_lat))
    nn_single_ms_med = float(np.median(t_nn_lat))
    print(f"[LATENCY] Model A single-spectrum CPU NN inference: mean={nn_single_ms_mean:.2f} ms, med={nn_single_ms_med:.2f} ms", flush=True)

    # -----------------------------------------------------------------------
    # 1. STRICTLY ON val_sel: Tune SG smoother, Gaussian smoother, MAD-floor
    #    diagnostic, and selective_hybrid thresholds
    # -----------------------------------------------------------------------
    print("\n=== STEP 1: Tuning Strictly on val_sel (N=1,000) ===", flush=True)
    best_sg = {"dip_rmse": float("inf"), "window_length": 15, "polyorder": 3}
    sg_grid_log = []
    for wl in (7, 11, 15, 19, 25, 31, 39):
        for po in (2, 3, 4):
            if po >= wl:
                continue
            y_sg = apply_savgol_batch(val_sel["X_noisy"], val_sel["valid_mask"], wl, po)
            m_sg = compute_denoising_metrics(val_sel["X_noisy"], y_sg, val_sel["Y_clean"], val_sel["valid_mask"])
            sg_grid_log.append({"wl": wl, "po": po, "mean_dip_rmse": m_sg["mean_dip_rmse"], "median_pos_err_mhz": m_sg["median_pos_err_mhz"]})
            if m_sg["mean_dip_rmse"] < best_sg["dip_rmse"]:
                best_sg = {"dip_rmse": m_sg["mean_dip_rmse"], "window_length": wl, "polyorder": po, "pos_err_mhz": m_sg["median_pos_err_mhz"]}

    best_gauss = {"dip_rmse": float("inf"), "sigma": 2.0}
    for sig in (0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0):
        y_g = apply_gaussian_batch(val_sel["X_noisy"], val_sel["valid_mask"], sig)
        m_g = compute_denoising_metrics(val_sel["X_noisy"], y_g, val_sel["Y_clean"], val_sel["valid_mask"])
        if m_g["mean_dip_rmse"] < best_gauss["dip_rmse"]:
            best_gauss = {"dip_rmse": m_g["mean_dip_rmse"], "sigma": sig, "pos_err_mhz": m_g["median_pos_err_mhz"]}

    print(f"  Tuned SG on val_sel: wl={best_sg['window_length']}, poly={best_sg['polyorder']} (dip_rmse={best_sg['dip_rmse']:.6f})", flush=True)
    print(f"  Tuned Gaussian on val_sel: sigma={best_gauss['sigma']} (dip_rmse={best_gauss['dip_rmse']:.6f})", flush=True)

    # Predict NN on all val_sel
    nn_val_sel = pred_a_s0.predict_nn_batch(val_sel["X_noisy"], val_sel["valid_mask"])

    # Stratified N=300 subset of val_sel (100 Hard, 100 Medium, 100 Easy) for selective_hybrid tuning
    sel_sub_idx = np.concatenate([
        np.where(val_sel["aux_regime_id"] == 0)[0][:100],
        np.where(val_sel["aux_regime_id"] == 1)[0][:100],
        np.where(val_sel["aux_regime_id"] == 2)[0][:100],
    ])
    tasks_sel_ws = [
        (val_sel["X_noisy"][i], val_sel["valid_mask"][i], nn_val_sel["b_vec_canon"][i], 3, 3, (3.0, 9.0))
        for i in sel_sub_idx
    ]
    res_sel_ws = run_parallel_fitter(_worker_fit_physical_plain, tasks_sel_ws, max_workers=4)
    sel_ws_b = np.array([r["b_mag"] for r in res_sel_ws], dtype=np.float64)
    sel_ws_ms = float(np.mean([r["runtime_ms"] for r in res_sel_ws]))

    # MAD-floor diagnostic on first N=100 stratified subset of val_sel (34 Hard, 33 Med, 33 Easy)
    mad_idx = np.concatenate([
        np.where(val_sel["aux_regime_id"] == 0)[0][:34],
        np.where(val_sel["aux_regime_id"] == 1)[0][:33],
        np.where(val_sel["aux_regime_id"] == 2)[0][:33],
    ])
    tasks_mad = [
        (nn_val_sel["y_clean_hat"][i], val_sel["valid_mask"][i], nn_val_sel["b_vec_canon"][i], 1, 1, (3.0, 9.0))
        for i in mad_idx
    ]
    res_mad = run_parallel_fitter(_worker_fit_physical_plain, tasks_mad, max_workers=4)
    mad_diag = {
        "n": int(mad_idx.size),
        "hybrid_denoised_fit": compute_b_metrics(
            np.array([r["b_mag"] for r in res_mad]),
            val_sel["Y_b_mag"][mad_idx],
            np.array([r["b_vec"] for r in res_mad]),
            val_sel["aux_b_vec_canon"][mad_idx],
            np.array([r["runtime_ms"] + nn_single_ms_mean for r in res_mad]),
        ),
        "nn_direct_same_100": compute_b_metrics(
            nn_val_sel["b_mag_phys"][mad_idx],
            val_sel["Y_b_mag"][mad_idx],
            nn_val_sel["b_vec_canon"][mad_idx],
            val_sel["aux_b_vec_canon"][mad_idx],
        ),
        "converged_pct": round(float(np.mean([r["status"] == "converged" for r in res_mad]) * 100.0), 2),
        "out_of_range_pct": round(float(np.mean([r["status"] == "out_of_range" for r in res_mad]) * 100.0), 2),
    }
    print(f"  MAD-floor diagnostic (val_sel N=100): denoised_fit med={mad_diag['hybrid_denoised_fit']['median_ut']} uT, "
          f"p90={mad_diag['hybrid_denoised_fit']['p90_ut']} uT, conv={mad_diag['converged_pct']}% vs NN direct med={mad_diag['nn_direct_same_100']['median_ut']} uT", flush=True)

    # Tune selective_hybrid thresholds strictly on val_sel N=300
    sel_tune = tune_selective_hybrid_thresholds(
        nn_b_mag=nn_val_sel["b_mag_phys"][sel_sub_idx],
        nn_unc_mt=nn_val_sel["uncertainty_mt"][sel_sub_idx],
        nn_tetra_mt=nn_val_sel["tetra_residual"][sel_sub_idx],
        fit_b_mag=sel_ws_b,
        b_true_mt=val_sel["Y_b_mag"][sel_sub_idx],
        nn_ms_per_spec=nn_single_ms_mean,
        fit_ms_per_spec=sel_ws_ms,
        max_fallback_frac=0.45,
    )
    t_unc_locked = float(sel_tune["best"]["unc_threshold_mt"])
    t_tet_locked = float(sel_tune["best"]["tetra_threshold_mt"])
    print(f"  Locked selective_hybrid thresholds (val_sel): unc>{t_unc_locked:.4f} mT, tetra>{t_tet_locked:.4f} mT "
          f"(val_sel fallback={sel_tune['best']['fallback_pct']}%, P90={sel_tune['best']['val_sel_p90_ut']} uT)", flush=True)

    # -----------------------------------------------------------------------
    # 2. Load val_baseline_ref.npz & Evaluate val_rep (3,000), val_id_natural
    #    (1,500), val_ood (1,500), and all 9 ood_* splits
    # -----------------------------------------------------------------------
    print("\n=== STEP 2: Reporting Evaluations (val_rep, val_id_natural, val_ood, 9 ood_*) ===", flush=True)
    ref = dict(np.load(DATA_DIR / "val_baseline_ref.npz", allow_pickle=False))
    ref_asym = dict(np.load(DATA_DIR / "ood_asym_baseline_ref.npz", allow_pickle=False))

    val_rep_idx = ref["val_id_indices"]
    val_rep = load_split_subset("val_id.npz", val_rep_idx)
    assert np.intersect1d(val_rep_idx, get_val_sel_indices()).size == 0, "val_rep / val_sel overlap!"

    # Model A (3 seeds) + Model B on all 3,000 val_rep
    rep_a_s0 = pred_a_s0.predict_nn_batch(val_rep["X_noisy"], val_rep["valid_mask"])
    rep_a_s1 = pred_a_s1.predict_nn_batch(val_rep["X_noisy"], val_rep["valid_mask"])
    rep_a_s2 = pred_a_s2.predict_nn_batch(val_rep["X_noisy"], val_rep["valid_mask"])
    rep_b = pred_b.predict_nn_batch(val_rep["X_noisy"], val_rep["valid_mask"])

    def _summarize_by_regime(pred_dict: Dict[str, np.ndarray], split_d: Dict[str, np.ndarray], rt_ms: float) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        rt_arr = np.full(split_d["Y_b_mag"].shape[0], rt_ms, dtype=np.float64)
        out["Overall"] = compute_b_metrics(pred_dict["b_mag_phys"], split_d["Y_b_mag"], pred_dict["b_vec_canon"], split_d["aux_b_vec_canon"], rt_arr)
        for rid, rname in REGIME_NAMES.items():
            m = split_d["aux_regime_id"] == rid
            out[rname] = compute_b_metrics(pred_dict["b_mag_phys"][m], split_d["Y_b_mag"][m], pred_dict["b_vec_canon"][m], split_d["aux_b_vec_canon"][m], rt_arr[m])
        return out

    def _summarize_ref_by_regime(prefix: str, ref_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        err_ut = np.asarray(ref_dict[f"{prefix}_abs_err_ut"], dtype=np.float64)
        signed_ut = (np.asarray(ref_dict[f"{prefix}_b_est"], dtype=np.float64) - np.asarray(ref_dict[f"{prefix}_b_true"], dtype=np.float64)) * 1000.0
        fail = np.asarray(ref_dict[f"{prefix}_fail_100"], dtype=bool)
        wb = np.asarray(ref_dict[f"{prefix}_wrong_basin"], dtype=bool)
        rt = np.asarray(ref_dict[f"{prefix}_runtime_ms"], dtype=np.float64)
        reg = np.asarray(ref_dict[f"{prefix}_regime_id"], dtype=np.int64)

        def _pack(mask: np.ndarray) -> Dict[str, Any]:
            e = err_ut[mask]
            return {
                "n": int(np.sum(mask)),
                "median_ut": round(float(np.median(e)), 2),
                "p90_ut": round(float(np.percentile(e, 90)), 2),
                "p95_ut": round(float(np.percentile(e, 95)), 2),
                "rms_ut": round(float(np.sqrt(np.mean(e * e))), 2),
                "mae_ut": round(float(np.mean(e)), 2),
                "signed_bias_ut": round(float(np.mean(signed_ut[mask])), 2),
                "fail_100_pct": round(float(np.mean(fail[mask]) * 100.0), 2),
                "wrong_basin_pct": round(float(np.mean(wb[mask]) * 100.0), 2),
                "mean_runtime_ms": round(float(np.mean(rt[mask])), 2),
                "median_runtime_ms": round(float(np.median(rt[mask])), 2),
            }

        out["Overall"] = _pack(np.ones(err_ut.size, dtype=bool))
        for rid, rname in REGIME_NAMES.items():
            if np.any(reg == rid):
                out[rname] = _pack(reg == rid)
        return out

    val_rep_summary = {
        "frozen_physical_fitter": _summarize_ref_by_regime("val_id", ref),
        "model_a_seed0": _summarize_by_regime(rep_a_s0, val_rep, nn_single_ms_mean),
        "model_a_seed1": _summarize_by_regime(rep_a_s1, val_rep, nn_single_ms_mean),
        "model_a_seed2": _summarize_by_regime(rep_a_s2, val_rep, nn_single_ms_mean),
        "model_b_sim2real": _summarize_by_regime(rep_b, val_rep, nn_single_ms_mean),
    }

    # 3-seed mean +/- std on val_rep
    seed_spread_val_rep: Dict[str, Any] = {}
    for reg_key in ("Overall", "Hard", "Medium", "Easy"):
        s_list = [val_rep_summary[f"model_a_seed{k}"][reg_key] for k in (0, 1, 2)]
        seed_spread_val_rep[reg_key] = {
            m: {
                "mean": round(float(np.mean([d[m] for d in s_list])), 2),
                "std": round(float(np.std([d[m] for d in s_list], ddof=1)), 2),
                "seeds": [d[m] for d in s_list],
            }
            for m in ("median_ut", "p90_ut", "rms_ut", "fail_100_pct", "wrong_basin_pct")
        }

    print(f"  val_rep (N=3000) Model A Seed 0: Overall Med={val_rep_summary['model_a_seed0']['Overall']['median_ut']} uT, "
          f"P90={val_rep_summary['model_a_seed0']['Overall']['p90_ut']} uT, WB={val_rep_summary['model_a_seed0']['Overall']['wrong_basin_pct']}%", flush=True)
    print(f"  val_rep (N=3000) Hard: Model A WB={val_rep_summary['model_a_seed0']['Hard']['wrong_basin_pct']}% vs Fitter {val_rep_summary['frozen_physical_fitter']['Hard']['wrong_basin_pct']}%", flush=True)
    print(f"  val_rep (N=3000) Easy: Model A Med={val_rep_summary['model_a_seed0']['Easy']['median_ut']} uT vs Fitter {val_rep_summary['frozen_physical_fitter']['Easy']['median_ut']} uT", flush=True)

    # val_id_natural (1,500) & val_ood (1,500 paired with ref)
    val_nat = load_split_subset("val_id_natural.npz", ref["val_id_natural_indices"])
    nat_a_s0 = pred_a_s0.predict_nn_batch(val_nat["X_noisy"], val_nat["valid_mask"])
    nat_b = pred_b.predict_nn_batch(val_nat["X_noisy"], val_nat["valid_mask"])
    val_nat_summary = {
        "frozen_physical_fitter": _summarize_ref_by_regime("val_id_natural", ref),
        "model_a_seed0": _summarize_by_regime(nat_a_s0, val_nat, nn_single_ms_mean),
        "model_b_sim2real": _summarize_by_regime(nat_b, val_nat, nn_single_ms_mean),
    }

    val_ood_idx = ref["val_ood_indices"]
    val_ood = load_split_subset("val_ood.npz", val_ood_idx)
    ood_a_s0 = pred_a_s0.predict_nn_batch(val_ood["X_noisy"], val_ood["valid_mask"])
    ood_a_s1 = pred_a_s1.predict_nn_batch(val_ood["X_noisy"], val_ood["valid_mask"])
    ood_a_s2 = pred_a_s2.predict_nn_batch(val_ood["X_noisy"], val_ood["valid_mask"])
    ood_b = pred_b.predict_nn_batch(val_ood["X_noisy"], val_ood["valid_mask"])
    val_ood_summary = {
        "frozen_physical_fitter": _summarize_ref_by_regime("val_ood", ref),
        "model_a_seed0": _summarize_by_regime(ood_a_s0, val_ood, nn_single_ms_mean),
        "model_a_seed1": _summarize_by_regime(ood_a_s1, val_ood, nn_single_ms_mean),
        "model_a_seed2": _summarize_by_regime(ood_a_s2, val_ood, nn_single_ms_mean),
        "model_b_sim2real": _summarize_by_regime(ood_b, val_ood, nn_single_ms_mean),
    }
    print(f"  val_ood (N=1500) Model A Seed 0: Overall Med={val_ood_summary['model_a_seed0']['Overall']['median_ut']} uT, "
          f"P90={val_ood_summary['model_a_seed0']['Overall']['p90_ut']} uT vs Fitter Med={val_ood_summary['frozen_physical_fitter']['Overall']['median_ut']} uT", flush=True)

    # All 9 single-factor OOD splits (full 1,000 for NN + exact paired ref subset vs Fitter)
    ood_factor_names = [
        "ood_hwhm", "ood_contrast", "ood_d0", "ood_noise", "ood_drift",
        "ood_hum", "ood_grid", "ood_high_field", "ood_asym",
    ]
    ood_factors_summary: Dict[str, Any] = {}
    for fname in ood_factor_names:
        full_d = load_split_subset(f"{fname}.npz")
        pa_full = pred_a_s0.predict_nn_batch(full_d["X_noisy"], full_d["valid_mask"])
        pb_full = pred_b.predict_nn_batch(full_d["X_noisy"], full_d["valid_mask"])
        if fname == "ood_asym":
            r_idx = ref_asym["indices"]
            f_metrics = {
                "n": int(r_idx.size),
                "median_ut": round(float(np.median(ref_asym["abs_err_ut"])), 2),
                "p90_ut": round(float(np.percentile(ref_asym["abs_err_ut"], 90)), 2),
                "rms_ut": round(float(np.sqrt(np.mean(ref_asym["abs_err_ut"] ** 2))), 2),
                "fail_100_pct": round(float(np.mean(ref_asym["fail_100"]) * 100.0), 2),
                "wrong_basin_pct": round(float(np.mean(ref_asym["wrong_basin"]) * 100.0), 2),
            }
        else:
            r_idx = ref[f"{fname}_indices"]
            f_metrics = _summarize_ref_by_regime(fname, ref)["Overall"]

        ood_factors_summary[fname] = {
            "paired_n": int(r_idx.size),
            "fitter_paired": f_metrics,
            "model_a_paired": compute_b_metrics(
                pa_full["b_mag_phys"][r_idx], full_d["Y_b_mag"][r_idx],
                pa_full["b_vec_canon"][r_idx], full_d["aux_b_vec_canon"][r_idx],
            ),
            "model_b_paired": compute_b_metrics(
                pb_full["b_mag_phys"][r_idx], full_d["Y_b_mag"][r_idx],
                pb_full["b_vec_canon"][r_idx], full_d["aux_b_vec_canon"][r_idx],
            ),
            "model_a_full1000": compute_b_metrics(
                pa_full["b_mag_phys"], full_d["Y_b_mag"],
                pa_full["b_vec_canon"], full_d["aux_b_vec_canon"],
            ),
            "model_b_full1000": compute_b_metrics(
                pb_full["b_mag_phys"], full_d["Y_b_mag"],
                pb_full["b_vec_canon"], full_d["aux_b_vec_canon"],
            ),
        }

    # -----------------------------------------------------------------------
    # 3. Paired Stratified Subsets (val_rep N=600 & val_ood N=300) for Slow
    #    Baselines & Hybrid Modes
    # -----------------------------------------------------------------------
    print("\n=== STEP 3: Paired Slow Baselines & Hybrid Modes (val_rep N=600, val_ood N=300) ===", flush=True)
    # Deterministic first 200 of each regime inside val_rep (positions 0..199, 1000..1199, 2000..2199 of val_rep)
    sub_rep_pos = np.concatenate([
        np.where(val_rep["aux_regime_id"] == 0)[0][:200],
        np.where(val_rep["aux_regime_id"] == 1)[0][:200],
        np.where(val_rep["aux_regime_id"] == 2)[0][:200],
    ])
    # Verify exact correspondence with preregistered original indices 0..199, 1333..1532, 2666..2865
    orig_sub_ids = val_rep_idx[sub_rep_pos]
    assert int(orig_sub_ids[0]) == 0 and int(orig_sub_ids[200]) == 1333 and int(orig_sub_ids[400]) == 2666

    # Run slow methods in parallel on val_rep N=600
    print("  Running physical_with_prior on val_rep N=600...", flush=True)
    res_rep_prior = run_parallel_fitter(
        _worker_fit_physical_prior,
        [(val_rep["X_noisy"][i], val_rep["valid_mask"][i], 5, 2) for i in sub_rep_pos],
    )
    print("  Running SG + frozen fitter on val_rep N=600...", flush=True)
    res_rep_sgfit = run_parallel_fitter(
        _worker_sg_plus_fitter,
        [(val_rep["X_noisy"][i], val_rep["valid_mask"][i], best_sg["window_length"], best_sg["polyorder"]) for i in sub_rep_pos],
    )
    print("  Running Model A hybrid_warm_start (n_starts=3, poly=3) on val_rep N=600...", flush=True)
    res_rep_hws = run_parallel_fitter(
        _worker_fit_physical_plain,
        [(val_rep["X_noisy"][i], val_rep["valid_mask"][i], rep_a_s0["b_vec_canon"][i], 3, 3, (3.0, 9.0)) for i in sub_rep_pos],
    )
    print("  Running Model A hybrid_denoised_fit (n_starts=1, poly=1) on val_rep N=600...", flush=True)
    res_rep_hdf = run_parallel_fitter(
        _worker_fit_physical_plain,
        [(rep_a_s0["y_clean_hat"][i], val_rep["valid_mask"][i], rep_a_s0["b_vec_canon"][i], 1, 1, (3.0, 9.0)) for i in sub_rep_pos],
    )

    # Assemble selective_hybrid on val_rep N=600 using strictly locked val_sel thresholds
    sub_unc = rep_a_s0["uncertainty_mt"][sub_rep_pos]
    sub_tet = rep_a_s0["tetra_residual"][sub_rep_pos]
    use_fb_rep = (sub_unc > t_unc_locked) | (sub_tet > t_tet_locked)
    hws_b_rep = np.array([r["b_mag"] for r in res_rep_hws], dtype=np.float64)
    hws_v_rep = np.array([r["b_vec"] for r in res_rep_hws], dtype=np.float64)
    hws_ms_rep = np.array([r["runtime_ms"] for r in res_rep_hws], dtype=np.float64)

    sel_b_rep = np.where(use_fb_rep, hws_b_rep, rep_a_s0["b_mag_phys"][sub_rep_pos])
    sel_v_rep = np.where(use_fb_rep[:, None], hws_v_rep, rep_a_s0["b_vec_canon"][sub_rep_pos])
    sel_rt_rep = nn_single_ms_mean + np.where(use_fb_rep, hws_ms_rep, 0.0)
    sel_undet_rep = use_fb_rep & (np.abs(hws_b_rep - rep_a_s0["b_mag_phys"][sub_rep_pos]) > 0.30)

    # Also evaluate selective_hybrid Pareto coverage sweep on val_rep N=600 (reporting curve)
    rep_pareto_curve = []
    for q_u in (50, 60, 70, 75, 80, 85, 90, 95, 100):
        thr_u = float(np.percentile(val_sel_unc := nn_val_sel["uncertainty_mt"], q_u)) if q_u < 100 else float("inf")
        uf = sub_unc > thr_u
        b_c = np.where(uf, hws_b_rep, rep_a_s0["b_mag_phys"][sub_rep_pos])
        e_c = np.abs(b_c - val_rep["Y_b_mag"][sub_rep_pos]) * 1000.0
        rt_c = nn_single_ms_mean + np.where(uf, hws_ms_rep, 0.0)
        rep_pareto_curve.append({
            "q_val_sel": q_u,
            "fallback_pct": round(float(np.mean(uf) * 100.0), 2),
            "median_ut": round(float(np.median(e_c)), 2),
            "p90_ut": round(float(np.percentile(e_c, 90)), 2),
            "fail_100_pct": round(float(np.mean(e_c > 100.0) * 100.0), 2),
            "mean_runtime_ms": round(float(np.mean(rt_c)), 2),
        })

    # Same paired subset on val_ood (N=300: 100 per regime)
    sub_ood_pos = np.concatenate([
        np.where(val_ood["aux_regime_id"] == 0)[0][:100],
        np.where(val_ood["aux_regime_id"] == 1)[0][:100],
        np.where(val_ood["aux_regime_id"] == 2)[0][:100],
    ])
    print("  Running Model A hybrid_warm_start & physical_with_prior on val_ood N=300...", flush=True)
    res_ood_prior = run_parallel_fitter(
        _worker_fit_physical_prior,
        [(val_ood["X_noisy"][i], val_ood["valid_mask"][i], 5, 2) for i in sub_ood_pos],
    )
    res_ood_hws = run_parallel_fitter(
        _worker_fit_physical_plain,
        [(val_ood["X_noisy"][i], val_ood["valid_mask"][i], ood_a_s0["b_vec_canon"][i], 3, 3, (3.0, 9.0)) for i in sub_ood_pos],
    )
    use_fb_ood = (ood_a_s0["uncertainty_mt"][sub_ood_pos] > t_unc_locked) | (ood_a_s0["tetra_residual"][sub_ood_pos] > t_tet_locked)
    hws_b_ood = np.array([r["b_mag"] for r in res_ood_hws], dtype=np.float64)
    hws_v_ood = np.array([r["b_vec"] for r in res_ood_hws], dtype=np.float64)
    hws_ms_ood = np.array([r["runtime_ms"] for r in res_ood_hws], dtype=np.float64)
    sel_b_ood = np.where(use_fb_ood, hws_b_ood, ood_a_s0["b_mag_phys"][sub_ood_pos])
    sel_v_ood = np.where(use_fb_ood[:, None], hws_v_ood, ood_a_s0["b_vec_canon"][sub_ood_pos])
    sel_rt_ood = nn_single_ms_mean + np.where(use_fb_ood, hws_ms_ood, 0.0)

    def _pack_paired_subset(
        pos: np.ndarray,
        split_d: Dict[str, np.ndarray],
        ref_prefix: str,
        nn_p: Dict[str, np.ndarray],
        res_prior: List[Dict[str, Any]],
        res_hws: List[Dict[str, Any]],
        sel_b: np.ndarray,
        sel_v: np.ndarray,
        sel_rt: np.ndarray,
        use_fb: np.ndarray,
        res_sgfit: Optional[List[Dict[str, Any]]] = None,
        res_hdf: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        bt = split_d["Y_b_mag"][pos]
        vt = split_d["aux_b_vec_canon"][pos]
        reg = split_d["aux_regime_id"][pos]

        def _by_reg(bp: np.ndarray, vp: np.ndarray, rt: np.ndarray) -> Dict[str, Any]:
            d = {"Overall": compute_b_metrics(bp, bt, vp, vt, rt)}
            for rid, rname in REGIME_NAMES.items():
                m = reg == rid
                d[rname] = compute_b_metrics(bp[m], bt[m], vp[m], vt[m], rt[m])
            return d

        f_b = np.asarray(ref[f"{ref_prefix}_b_est"][pos], dtype=np.float64)
        f_v = np.asarray(ref[f"{ref_prefix}_b_vec_est"][pos], dtype=np.float64)
        f_rt = np.asarray(ref[f"{ref_prefix}_runtime_ms"][pos], dtype=np.float64)

        out = {
            "n": int(pos.size),
            "frozen_physical_fitter": _by_reg(f_b, f_v, f_rt),
            "physical_with_prior": _by_reg(
                np.array([r["b_mag"] for r in res_prior]),
                np.array([r["b_vec"] for r in res_prior]),
                np.array([r["runtime_ms"] for r in res_prior]),
            ),
            "model_a_nn_direct": _by_reg(
                nn_p["b_mag_phys"][pos],
                nn_p["b_vec_canon"][pos],
                np.full(pos.size, nn_single_ms_mean),
            ),
            "model_a_hybrid_warm_start": _by_reg(
                np.array([r["b_mag"] for r in res_hws]),
                np.array([r["b_vec"] for r in res_hws]),
                np.array([r["runtime_ms"] + nn_single_ms_mean for r in res_hws]),
            ),
            "model_a_selective_hybrid": _by_reg(sel_b, sel_v, sel_rt),
            "selective_hybrid_fallback_pct": round(float(np.mean(use_fb) * 100.0), 2),
        }
        if res_sgfit is not None:
            out["sg_plus_fitter"] = _by_reg(
                np.array([r["b_mag"] for r in res_sgfit]),
                np.array([r["b_vec"] for r in res_sgfit]),
                np.array([r["runtime_ms"] for r in res_sgfit]),
            )
        if res_hdf is not None:
            out["model_a_hybrid_denoised_fit"] = _by_reg(
                np.array([r["b_mag"] for r in res_hdf]),
                np.array([r["b_vec"] for r in res_hdf]),
                np.array([r["runtime_ms"] + nn_single_ms_mean for r in res_hdf]),
            )
        return out

    paired_rep_600 = _pack_paired_subset(
        sub_rep_pos, val_rep, "val_id", rep_a_s0, res_rep_prior, res_rep_hws,
        sel_b_rep, sel_v_rep, sel_rt_rep, use_fb_rep, res_rep_sgfit, res_rep_hdf,
    )
    paired_rep_600["selective_hybrid_undetermined_pct"] = round(float(np.mean(sel_undet_rep) * 100.0), 2)
    paired_ood_300 = _pack_paired_subset(
        sub_ood_pos, val_ood, "val_ood", ood_a_s0, res_ood_prior, res_ood_hws,
        sel_b_ood, sel_v_ood, sel_rt_ood, use_fb_ood,
    )

    print(f"  val_rep N=600 Selective Hybrid (fallback={paired_rep_600['selective_hybrid_fallback_pct']}%): "
          f"Med={paired_rep_600['model_a_selective_hybrid']['Overall']['median_ut']} uT, "
          f"P90={paired_rep_600['model_a_selective_hybrid']['Overall']['p90_ut']} uT, "
          f"Fail={paired_rep_600['model_a_selective_hybrid']['Overall']['fail_100_pct']}%, "
          f"RT={paired_rep_600['model_a_selective_hybrid']['Overall']['mean_runtime_ms']} ms "
          f"(vs Fitter P90={paired_rep_600['frozen_physical_fitter']['Overall']['p90_ut']} uT, "
          f"Fail={paired_rep_600['frozen_physical_fitter']['Overall']['fail_100_pct']}%, "
          f"RT={paired_rep_600['frozen_physical_fitter']['Overall']['mean_runtime_ms']} ms)", flush=True)

    # -----------------------------------------------------------------------
    # 4. Slices, Signed Bias & Pareto Plot
    # -----------------------------------------------------------------------
    print("\n=== STEP 4: Slices & Signed Bias Analysis ===", flush=True)
    edges = np.linspace(0.05, 3.20, 7)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_rows = []
    fitter_bias_list, nn_bias_list, mb_bias_list = [], [], []
    for bi in range(6):
        lo, hi = float(edges[bi]), float(edges[bi + 1])
        m = (val_rep["Y_b_mag"] >= lo) & (val_rep["Y_b_mag"] <= hi if bi == 5 else val_rep["Y_b_mag"] < hi)
        f_b_m = compute_b_metrics(ref["val_id_b_est"][m], val_rep["Y_b_mag"][m], ref["val_id_b_vec_est"][m], val_rep["aux_b_vec_canon"][m])
        a_b_m = compute_b_metrics(rep_a_s0["b_mag_phys"][m], val_rep["Y_b_mag"][m], rep_a_s0["b_vec_canon"][m], val_rep["aux_b_vec_canon"][m])
        b_b_m = compute_b_metrics(rep_b["b_mag_phys"][m], val_rep["Y_b_mag"][m], rep_b["b_vec_canon"][m], val_rep["aux_b_vec_canon"][m])
        fitter_bias_list.append(f_b_m["signed_bias_ut"])
        nn_bias_list.append(a_b_m["signed_bias_ut"])
        mb_bias_list.append(b_b_m["signed_bias_ut"])
        bin_rows.append({"bin_mt": [round(lo, 3), round(hi, 3)], "n": int(np.sum(m)), "fitter": f_b_m, "model_a": a_b_m, "model_b": b_b_m})

    def _slice_flags(split_d: Dict[str, np.ndarray], nn_p: Dict[str, np.ndarray], ref_prefix: str) -> Dict[str, Any]:
        out = {}
        for flag_key in ("aux_weak_splitting", "aux_axis_degenerate"):
            out[flag_key] = {}
            for val in (False, True):
                m = split_d[flag_key].astype(bool) == val
                out[flag_key][str(val)] = {
                    "n": int(np.sum(m)),
                    "fitter": compute_b_metrics(ref[f"{ref_prefix}_b_est"][m], split_d["Y_b_mag"][m], ref[f"{ref_prefix}_b_vec_est"][m], split_d["aux_b_vec_canon"][m]),
                    "model_a": compute_b_metrics(nn_p["b_mag_phys"][m], split_d["Y_b_mag"][m], nn_p["b_vec_canon"][m], split_d["aux_b_vec_canon"][m]),
                }
        return out

    slices_summary = {
        "val_rep_b_bins": bin_rows,
        "val_rep_flags": _slice_flags(val_rep, rep_a_s0, "val_id"),
        "val_ood_flags": _slice_flags(val_ood, ood_a_s0, "val_ood"),
    }

    pareto_points = [
        {"label": "Frozen Fitter", **paired_rep_600["frozen_physical_fitter"]["Overall"]},
        {"label": "Physical + Prior", **paired_rep_600["physical_with_prior"]["Overall"]},
        {"label": "SG + Fitter", **paired_rep_600["sg_plus_fitter"]["Overall"]},
        {"label": "Model A (NN Direct)", **paired_rep_600["model_a_nn_direct"]["Overall"]},
        {"label": "Model A (Warm-Start)", **paired_rep_600["model_a_hybrid_warm_start"]["Overall"]},
        {"label": "Model A (Selective Hybrid)", **paired_rep_600["model_a_selective_hybrid"]["Overall"]},
    ]
    plot_bias_and_pareto(
        {
            "bin_centers_mt": centers.tolist(),
            "fitter_bias_ut": fitter_bias_list,
            "nn_bias_ut": nn_bias_list,
            "model_b_bias_ut": mb_bias_list,
        },
        pareto_points,
        rep_pareto_curve,
        ASSETS_DIR / "stage3_bias_and_pareto.png",
    )

    # -----------------------------------------------------------------------
    # 5. Denoising Quality Comparison (val_rep N=3000 & val_ood N=1500)
    # -----------------------------------------------------------------------
    print("\n=== STEP 5: Denoising Quality Comparison (H4 & OOD) ===", flush=True)
    y_sg_rep = apply_savgol_batch(val_rep["X_noisy"], val_rep["valid_mask"], best_sg["window_length"], best_sg["polyorder"])
    y_ga_rep = apply_gaussian_batch(val_rep["X_noisy"], val_rep["valid_mask"], best_gauss["sigma"])
    y_sg_ood = apply_savgol_batch(val_ood["X_noisy"], val_ood["valid_mask"], best_sg["window_length"], best_sg["polyorder"])
    y_ga_ood = apply_gaussian_batch(val_ood["X_noisy"], val_ood["valid_mask"], best_gauss["sigma"])

    den_rep_raw = compute_denoising_metrics(val_rep["X_noisy"], val_rep["X_noisy"], val_rep["Y_clean"], val_rep["valid_mask"])
    den_rep_sg = compute_denoising_metrics(val_rep["X_noisy"], y_sg_rep, val_rep["Y_clean"], val_rep["valid_mask"])
    den_rep_ga = compute_denoising_metrics(val_rep["X_noisy"], y_ga_rep, val_rep["Y_clean"], val_rep["valid_mask"])
    den_rep_nn = compute_denoising_metrics(val_rep["X_noisy"], rep_a_s0["y_clean_hat"], val_rep["Y_clean"], val_rep["valid_mask"])

    den_ood_raw = compute_denoising_metrics(val_ood["X_noisy"], val_ood["X_noisy"], val_ood["Y_clean"], val_ood["valid_mask"])
    den_ood_sg = compute_denoising_metrics(val_ood["X_noisy"], y_sg_ood, val_ood["Y_clean"], val_ood["valid_mask"])
    den_ood_ga = compute_denoising_metrics(val_ood["X_noisy"], y_ga_ood, val_ood["Y_clean"], val_ood["valid_mask"])
    den_ood_nn = compute_denoising_metrics(val_ood["X_noisy"], ood_a_s0["y_clean_hat"], val_ood["Y_clean"], val_ood["valid_mask"])

    def _strip_priv(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    denoising_summary = {
        "tuned_sg_params": best_sg,
        "tuned_gaussian_params": best_gauss,
        "val_rep": {
            "raw_input": _strip_priv(den_rep_raw),
            "tuned_savgol": _strip_priv(den_rep_sg),
            "tuned_gaussian": _strip_priv(den_rep_ga),
            "model_a_nn": _strip_priv(den_rep_nn),
        },
        "val_ood": {
            "raw_input": _strip_priv(den_ood_raw),
            "tuned_savgol": _strip_priv(den_ood_sg),
            "tuned_gaussian": _strip_priv(den_ood_ga),
            "model_a_nn": _strip_priv(den_ood_nn),
        },
    }
    print(f"  val_rep Dip-Region RMSE: Raw={den_rep_raw['mean_dip_rmse']:.6f}, SG={den_rep_sg['mean_dip_rmse']:.6f}, "
          f"Gauss={den_rep_ga['mean_dip_rmse']:.6f}, Model A={den_rep_nn['mean_dip_rmse']:.6f}", flush=True)
    print(f"  val_rep Median Dip-Pos Err (MHz): SG={den_rep_sg['median_pos_err_mhz']:.4f}, Model A={den_rep_nn['median_pos_err_mhz']:.4f}", flush=True)

    # -----------------------------------------------------------------------
    # 6. Uncertainty Calibration & Hallucination / Robustness Stress Tests
    # -----------------------------------------------------------------------
    print("\n=== STEP 6: Uncertainty Calibration & Hallucination Stress Tests ===", flush=True)

    def _cov80_by_regime(nn_p: Dict[str, np.ndarray], split_d: Dict[str, np.ndarray]) -> Dict[str, float]:
        bt = split_d["Y_b_mag"]
        inside = (nn_p["q10"] <= bt) & (bt <= nn_p["q90"])
        out = {"Overall": round(float(np.mean(inside) * 100.0), 2)}
        for rid, rname in REGIME_NAMES.items():
            m = split_d["aux_regime_id"] == rid
            out[rname] = round(float(np.mean(inside[m]) * 100.0), 2)
        return out

    def _risk_coverage(nn_p: Dict[str, np.ndarray], split_d: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
        err_ut = np.abs(nn_p["b_mag_phys"] - split_d["Y_b_mag"]) * 1000.0
        order = np.argsort(nn_p["uncertainty_mt"])
        e_sorted = err_ut[order]
        covs = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        meds, p90s = [], []
        for c in covs:
            k = max(1, int(round(c / 100.0 * e_sorted.size)))
            meds.append(round(float(np.median(e_sorted[:k])), 2))
            p90s.append(round(float(np.percentile(e_sorted[:k], 90)), 2))
        return {"coverage_pct": covs, "median_ut": meds, "p90_ut": p90s}

    unc_summary = {
        "val_rep_cov80": _cov80_by_regime(rep_a_s0, val_rep),
        "val_ood_cov80": _cov80_by_regime(ood_a_s0, val_ood),
        "risk_coverage_val_rep": _risk_coverage(rep_a_s0, val_rep),
        "risk_coverage_val_ood": _risk_coverage(ood_a_s0, val_ood),
    }

    # Hallucination stress tests (deterministic RNG seed = 777001)
    rng_h = np.random.default_rng(777001)

    # (a) No-dip spectra (N = 200): baseline drift + hum + noise around 1.0
    n_nodip = 200
    x_nodip = np.zeros((n_nodip, 512), dtype=np.float32)
    for i in range(n_nodip):
        drift, _ = sample_baseline_drift(CANONICAL_FREQS, rng_h, amplitude_range=(0.002, 0.012))
        hum, _ = sample_mains_hum(512, rng_h, amplitude_range=(0.0005, 0.003))
        wn = sample_white_noise(512, sigma=float(rng_h.uniform(0.002, 0.008)), rng=rng_h)
        x_nodip[i] = (1.0 + drift + hum + wn).astype(np.float32)
    p_nodip = pred_a_s0.predict_nn_batch(x_nodip, np.ones_like(x_nodip, dtype=bool))
    max_false_depths = np.max(1.0 - p_nodip["y_clean_hat"], axis=1)
    no_dip_res = {
        "n": n_nodip,
        "p50_depth": round(float(np.percentile(max_false_depths, 50)), 5),
        "p90_depth": round(float(np.percentile(max_false_depths, 90)), 5),
        "p99_depth": round(float(np.percentile(max_false_depths, 99)), 5),
        "max_depth": round(float(np.max(max_false_depths)), 5),
        "median_pred_b_mt": round(float(np.median(p_nodip["b_mag_phys"])), 4),
        "median_unc_mt": round(float(np.median(p_nodip["uncertainty_mt"])), 4),
        "_raw_max_depths": max_false_depths.tolist(),
    }

    # (b) Dip-shift tracking: shift D0 by [-10, -5, +5, +10] MHz on 100 clean spectra + moderate noise
    shifts_mhz = [-10.0, -5.0, 5.0, 10.0]
    shift_med_errs = []
    shift_p90_errs = []
    for sh in shifts_mhz:
        x_sh = np.zeros((100, 512), dtype=np.float32)
        y_sh = np.zeros((100, 512), dtype=np.float64)
        for i in range(100):
            b_v = val_rep["aux_b_vec_canon"][2000 + i].astype(np.float64)  # Easy-like clear spectra
            hwhm = float(val_rep["Y_hwhm"][2000 + i])
            con = float(val_rep["aux_contrast"][2000 + i])
            yc, _ = generate_4axis_spectrum(CANONICAL_FREQS, b_v, d0=2870.0 + sh, hwhm=hwhm, contrast=con, eta=0.35, e_mhz=0.0)
            y_sh[i] = yc
            x_sh[i] = (yc + sample_white_noise(512, sigma=0.0025, rng=rng_h)).astype(np.float32)
        p_sh = pred_a_s0.predict_nn_batch(x_sh, np.ones_like(x_sh, dtype=bool))
        # Track primary dip shift error
        errs_sh = []
        for i in range(100):
            f_t = float(CANONICAL_FREQS[int(np.argmax(1.0 - y_sh[i]))])
            f_p = float(CANONICAL_FREQS[int(np.argmax(1.0 - p_sh["y_clean_hat"][i]))])
            errs_sh.append(abs(f_p - f_t))
        shift_med_errs.append(round(float(np.median(errs_sh)), 3))
        shift_p90_errs.append(round(float(np.percentile(errs_sh, 90)), 3))

    # (c) Extra spurious narrow wing dip (N=100)
    n_spur = 100
    x_spur = np.zeros((n_spur, 512), dtype=np.float32)
    spur_centers = np.zeros(n_spur, dtype=np.float64)
    in_depth = 0.018
    for i in range(n_spur):
        yc = val_rep["Y_clean"][2100 + i].astype(np.float64)
        fc = float(rng_h.choice([rng_h.uniform(2758.0, 2776.0), rng_h.uniform(2964.0, 2982.0)]))
        spur_centers[i] = fc
        spur = in_depth * np.exp(-0.5 * ((CANONICAL_FREQS - fc) / 1.5) ** 2)
        x_spur[i] = (yc - spur + sample_white_noise(512, sigma=0.0015, rng=rng_h)).astype(np.float32)
    p_spur = pred_a_s0.predict_nn_batch(x_spur, np.ones_like(x_spur, dtype=bool))
    attens = []
    for i in range(n_spur):
        win = np.abs(CANONICAL_FREQS - spur_centers[i]) <= 3.5
        base_dip = float(np.max(1.0 - val_rep["Y_clean"][2100 + i, win]))
        out_dip = max(0.0, float(np.max(1.0 - p_spur["y_clean_hat"][i, win])) - base_dip)
        attens.append(max(0.0, (1.0 - out_dip / in_depth) * 100.0))

    # (d) Spike-corrupted (N=100) & cropped valid_mask (N=100)
    idx_rob = np.arange(2200, 2300)
    x_spike = val_rep["X_noisy"][idx_rob].copy()
    for i in range(100):
        n_sp = int(rng_h.integers(1, 4))
        sp_pos = rng_h.integers(10, 502, size=n_sp)
        x_spike[i, sp_pos] += rng_h.choice([-1.0, 1.0], size=n_sp) * rng_h.uniform(0.03, 0.08, size=n_sp)
    p_orig_rob = pred_a_s0.predict_nn_batch(val_rep["X_noisy"][idx_rob], val_rep["valid_mask"][idx_rob])
    p_spike = pred_a_s0.predict_nn_batch(x_spike, val_rep["valid_mask"][idx_rob])

    vm_crop = val_rep["valid_mask"][idx_rob].copy()
    vm_crop[:, :50] = False
    vm_crop[:, -50:] = False
    x_crop = val_rep["X_noisy"][idx_rob].copy()
    x_crop[~vm_crop] = 1.0
    p_crop = pred_a_s0.predict_nn_batch(x_crop, vm_crop)
    masked_wing_max = float(np.max(np.abs(1.0 - p_crop["y_clean_hat"][~vm_crop])))

    halluc_summary = {
        "no_dip": no_dip_res,
        "dip_shift": {
            "shifts_mhz": shifts_mhz,
            "median_abs_err_mhz": shift_med_errs,
            "p90_abs_err_mhz": shift_p90_errs,
        },
        "spurious_dip": {
            "n": n_spur,
            "injected_depth": in_depth,
            "median_attenuation_pct": round(float(np.median(attens)), 2),
            "p10_attenuation_pct": round(float(np.percentile(attens, 10)), 2),
            "p90_attenuation_pct": round(float(np.percentile(attens, 90)), 2),
        },
        "spike_and_crop": {
            "clean_easy_subset": compute_b_metrics(p_orig_rob["b_mag_phys"], val_rep["Y_b_mag"][idx_rob]),
            "spike_corrupted": compute_b_metrics(p_spike["b_mag_phys"], val_rep["Y_b_mag"][idx_rob]),
            "cropped_wings_20pct": compute_b_metrics(p_crop["b_mag_phys"], val_rep["Y_b_mag"][idx_rob]),
            "cropped_masked_region_max_dev_from_1": masked_wing_max,
        },
    }
    print(f"  Hallucination (a) No-dip P50={no_dip_res['p50_depth']:.5f}, P90={no_dip_res['p90_depth']:.5f}; "
          f"(c) Spurious-dip median attenuation={halluc_summary['spurious_dip']['median_attenuation_pct']}%", flush=True)

    plot_uncertainty_and_hallucination(unc_summary, halluc_summary, ASSETS_DIR / "stage3_uncertainty_and_hallucination.png")
    # Remove raw array before JSON serialization
    halluc_summary["no_dip"].pop("_raw_max_depths", None)

    # -----------------------------------------------------------------------
    # 7. Real Experimental Check Traces & Noise-Injection Stress Test
    # -----------------------------------------------------------------------
    print("\n=== STEP 7: Real trianam Traces & Noise-Injection Stress Test ===", flush=True)
    trianam_all = [
        ("trianam_small_field_1", "trianam_sf1_f.mat", "trianam_sf1_y.mat", "positive_contrast", 3.8170, "tuning"),
        ("trianam_small_field_2", "trianam_sf2_f.mat", "trianam_sf2_y.mat", "positive_contrast", 4.1862, "tuning"),
        ("trianam_small_field_3", "trianam_sf3_f.mat", "trianam_sf3_y.mat", "positive_contrast", 4.2887, "check"),
        ("trianam_small_field_4", "trianam_sf4_f.mat", "trianam_sf4_y.mat", "positive_contrast", 4.4104, "check"),
        ("trianam_folder_2_101G", "trianam_f2_f.mat", "trianam_f2_y.mat", "negative_contrast", 10.1870, "check_out_of_window"),
        ("trianam_folder_3_48G_hf", "trianam_f3_f.mat", "trianam_f3_y.mat", "negative_contrast", 4.8320, "check_out_of_window"),
        ("trianam_folder_4_69G", "trianam_f4_f.mat", "trianam_f4_y.mat", "negative_contrast", 6.9500, "check_out_of_window"),
    ]

    real_trace_results = []
    loaded_real_traces: Dict[str, Dict[str, Any]] = {}
    for tid, f_file, y_file, conv, b_true, split_role in trianam_all:
        fp, yp = DEFAULT_RAW_EXP_DIR / f_file, DEFAULT_RAW_EXP_DIR / y_file
        if not (fp.exists() and yp.exists()):
            continue
        mf, my = scipy.io.loadmat(fp), scipy.io.loadmat(yp)
        kf = [k for k in mf.keys() if not k.startswith("__")][0]
        ky = [k for k in my.keys() if not k.startswith("__")][0]
        rec = load_and_validate_real_trace(
            np.asarray(mf[kf], dtype=np.float64).squeeze(),
            np.asarray(my[ky], dtype=np.float64).squeeze(),
            trace_id=tid,
            source_name="trianam",
            license_id="GPL-3.0",
            citation="Haim et al. (2024)",
            nv_type="4-axis NV ensemble",
            declared_unit="MHz",
            spectrum_convention=conv,
            ground_truth_b_mt=b_true,
        )
        spec_c = rec["resampled_spectrum"]
        vm_c = rec["valid_mask"]
        loaded_real_traces[tid] = {"spec": spec_c, "vm": vm_c, "b_true": b_true, "rec": rec}

        out_a = pred_a_s0.predict_single(spec_c, vm_c, mode="nn_direct")
        out_b = pred_b.predict_single(spec_c, vm_c, mode="nn_direct")
        fit_def = fit_physical_odmr(CANONICAL_FREQS, spec_c, valid_mask=vm_c, n_starts=3, poly_degree=3)
        fit_wide = fit_physical_odmr(CANONICAL_FREQS, spec_c, valid_mask=vm_c, n_starts=3, poly_degree=3, hwhm_bounds=(2.0, 9.0))

        row = {
            "trace_id": tid,
            "role": split_role,
            "b_true_mt": b_true,
            "exceeds_model_a_3p20_scope": bool(b_true > 3.20),
            "exceeds_model_b_4p60_scope": bool(b_true > 4.60),
            "dips_outside_canonical_2750_2990": bool(rec["dips_outside_canonical_2750_2990"]),
            "model_a_b_pred_mt": round(out_a["b_mag"], 4),
            "model_a_abs_err_ut": round(abs(out_a["b_mag"] - b_true) * 1000.0, 1),
            "model_a_unc_mt": round(out_a["uncertainty_mt"], 4),
            "model_b_b_pred_mt": round(out_b["b_mag"], 4),
            "model_b_abs_err_ut": round(abs(out_b["b_mag"] - b_true) * 1000.0, 1),
            "model_b_unc_mt": round(out_b["uncertainty_mt"], 4),
            "fitter_default_b_pred_mt": round(float(fit_def["b_mag"]), 4),
            "fitter_default_abs_err_ut": round(abs(float(fit_def["b_mag"]) - b_true) * 1000.0, 1),
            "fitter_hwhm2_9_b_pred_mt": round(float(fit_wide["b_mag"]), 4),
            "fitter_hwhm2_9_abs_err_ut": round(abs(float(fit_wide["b_mag"]) - b_true) * 1000.0, 1),
        }
        real_trace_results.append(row)
        print(f"  {tid:24s} ({split_role:17s}) | True={b_true:.3f} mT | Model A={row['model_a_b_pred_mt']:.3f} ({row['model_a_abs_err_ut']:.0f}uT) | "
              f"Model B={row['model_b_b_pred_mt']:.3f} ({row['model_b_abs_err_ut']:.0f}uT) | Fitter={row['fitter_default_b_pred_mt']:.3f}", flush=True)

    # Noise-Injection Stress Test on real check traces small_field_3 & small_field_4 (20 trials x 2 SNR levels)
    rng_ni = np.random.default_rng(888001)
    noise_inj_results: Dict[str, Any] = {}
    real_plot_examples: List[Dict[str, Any]] = []

    if "trianam_small_field_3" in loaded_real_traces:
        s3 = loaded_real_traces["trianam_small_field_3"]
        o3_b = pred_b.predict_single(s3["spec"], s3["vm"], mode="nn_direct")
        real_plot_examples.append({
            "x_input": s3["spec"],
            "y_ref": s3["spec"],
            "y_model_b": o3_b["y_clean_hat"],
            "input_label": "Real Check Trace (High SNR)",
            "title": f"Real Check: sf3 (|B|={s3['b_true']:.3f} mT)\nModel B={o3_b['b_mag']:.3f} mT (|ΔB|={abs(o3_b['b_mag']-s3['b_true'])*1000:.0f} uT)",
        })

    for tid in ("trianam_small_field_3", "trianam_small_field_4"):
        if tid not in loaded_real_traces:
            continue
        tr = loaded_real_traces[tid]
        y_ref = np.asarray(tr["spec"], dtype=np.float64)
        vm = np.asarray(tr["vm"], dtype=bool)
        b_true = float(tr["b_true"])
        dip_depth_ref = float(np.max(1.0 - y_ref[vm]))

        noise_inj_results[tid] = {}
        for target_snr in (2.0, 1.0):
            n_trials = 20
            x_noisy_list = []
            for _ in range(n_trials):
                # Target total std = dip_depth_ref / target_snr split across white + pink + drift + hum
                target_std = dip_depth_ref / target_snr
                drift, _ = sample_baseline_drift(CANONICAL_FREQS, rng_ni, amplitude_range=(0.002, 0.008))
                hum, _ = sample_mains_hum(512, rng_ni, amplitude_range=(0.0005, 0.002))
                w_noise = sample_white_noise(512, sigma=0.75 * target_std, rng=rng_ni)
                p_noise, _ = sample_colored_noise(512, sigma_white=0.60 * target_std, rng=rng_ni, pink_ratio=0.6)
                raw_corrupt = drift + hum + w_noise + 0.5 * p_noise
                # Scale corruption so empirical total RMS matches target_std cleanly
                cur_std = float(np.std(raw_corrupt[vm]))
                scaled_corrupt = raw_corrupt * (target_std / max(cur_std, 1e-9))
                x_noisy_list.append((y_ref + scaled_corrupt).astype(np.float32))
            x_arr = np.stack(x_noisy_list, axis=0)
            vm_arr = np.repeat(vm[None, :], n_trials, axis=0)

            pb_ni = pred_b.predict_nn_batch(x_arr, vm_arr)
            pa_ni = pred_a_s0.predict_nn_batch(x_arr, vm_arr)

            # Run frozen fitter on raw noisy AND on Model B denoised (in parallel)
            fit_raw_ni = run_parallel_fitter(
                _worker_fit_physical_plain,
                [(x_arr[i], vm, None, 3, 3, (2.0, 9.0)) for i in range(n_trials)],
            )
            fit_den_ni = run_parallel_fitter(
                _worker_fit_physical_plain,
                [(pb_ni["y_clean_hat"][i], vm, pb_ni["b_vec_canon"][i], 1, 1, (2.0, 9.0)) for i in range(n_trials)],
            )

            err_fit_raw = np.abs(np.array([r["b_mag"] for r in fit_raw_ni]) - b_true) * 1000.0
            err_fit_den = np.abs(np.array([r["b_mag"] for r in fit_den_ni]) - b_true) * 1000.0
            err_mb = np.abs(pb_ni["b_mag_phys"] - b_true) * 1000.0
            err_ma = np.abs(pa_ni["b_mag_phys"] - b_true) * 1000.0
            rmse_in_ni = float(np.mean(np.sqrt(np.mean((x_arr[:, vm] - y_ref[None, vm]) ** 2, axis=1))))
            rmse_out_ni = float(np.mean(np.sqrt(np.mean((pb_ni["y_clean_hat"][:, vm] - y_ref[None, vm]) ** 2, axis=1))))

            noise_inj_results[tid][f"snr_{target_snr:.1f}"] = {
                "n_trials": n_trials,
                "rmse_in": round(rmse_in_ni, 5),
                "rmse_out_model_b": round(rmse_out_ni, 5),
                "fitter_raw_median_ut": round(float(np.median(err_fit_raw)), 1),
                "fitter_raw_p90_ut": round(float(np.percentile(err_fit_raw, 90)), 1),
                "model_b_nn_median_ut": round(float(np.median(err_mb)), 1),
                "model_b_nn_p90_ut": round(float(np.percentile(err_mb, 90)), 1),
                "model_b_denoised_fit_median_ut": round(float(np.median(err_fit_den)), 1),
                "model_b_denoised_fit_p90_ut": round(float(np.percentile(err_fit_den, 90)), 1),
                "model_a_nn_out_of_scope_median_ut": round(float(np.median(err_ma)), 1),
            }

            if target_snr == 1.0:
                short_name = tid.replace("trianam_small_field_", "sf")
                real_plot_examples.append({
                    "x_input": x_arr[0],
                    "y_ref": y_ref,
                    "y_model_b": pb_ni["y_clean_hat"][0],
                    "input_label": f"Noise-Injected (SNR~1.0)",
                    "title": (
                        f"Real {short_name} + Hardware Noise (SNR~1.0)\n"
                        f"Model B={pb_ni['b_mag_phys'][0]:.3f} mT (Med {float(np.median(err_mb)):.0f}uT vs Fitter {float(np.median(err_fit_raw)):.0f}uT)"
                    ),
                })

    # Synthetic denoising examples for top row of stage3_denoising_and_real_examples.png
    syn_examples = []
    for rid, rname in ((0, "Hard"), (1, "Medium"), (2, "Easy")):
        pos_list = np.where((val_rep["aux_regime_id"] == rid) & (~val_rep["aux_weak_splitting"].astype(bool)))[0]
        p_idx = int(pos_list[5])
        syn_examples.append({
            "regime": rname,
            "b_true": float(val_rep["Y_b_mag"][p_idx]),
            "snr_w": float(val_rep["aux_eff_snr_white"][p_idx]),
            "x_noisy": val_rep["X_noisy"][p_idx],
            "y_clean": val_rep["Y_clean"][p_idx],
            "y_sg": y_sg_rep[p_idx],
            "y_nn": rep_a_s0["y_clean_hat"][p_idx],
            "nn_err_ut": float(abs(rep_a_s0["b_mag_phys"][p_idx] - val_rep["Y_b_mag"][p_idx]) * 1000.0),
            "fit_err_ut": float(ref["val_id_abs_err_ut"][p_idx]),
        })
    plot_denoising_and_real_examples(syn_examples, real_plot_examples, ASSETS_DIR / "stage3_denoising_and_real_examples.png")

    # -----------------------------------------------------------------------
    # 8. Preregistered Hypotheses H1–H4 (M = 4, Holm-Bonferroni at alpha=0.05)
    # -----------------------------------------------------------------------
    print("\n=== STEP 8: Preregistered Hypotheses Verdicts (H1–H4, Holm-Bonferroni M=4) ===", flush=True)
    # H1: Hard-regime wrong-basin reduction (val_rep Hard, N=1,000)
    m_hard = val_rep["aux_regime_id"] == 0
    wb_fitter_h = np.asarray(ref["val_id_wrong_basin"][m_hard], dtype=bool)
    vp_h = np.array([canonicalize_b_vec(v) for v in rep_a_s0["b_vec_canon"][m_hard]])
    vt_h = np.array([canonicalize_b_vec(v) for v in val_rep["aux_b_vec_canon"][m_hard]])
    wb_nn_h = np.linalg.norm(vp_h - vt_h, axis=1) > 0.15
    h1_mcn = mcnemar_paired_test(wb_nn_h, wb_fitter_h)
    # Directional check: only small p if NN < fitter
    p_h1_raw = float(h1_mcn["p_value"]) if h1_mcn["fail_rate_a"] < h1_mcn["fail_rate_b"] else 1.0

    # H2: Easy-regime non-inferiority of NN direct (X = +15.0 uT, val_rep Easy, N=1,000)
    m_easy = val_rep["aux_regime_id"] == 2
    err_nn_e = np.abs(rep_a_s0["b_mag_phys"][m_easy] - val_rep["Y_b_mag"][m_easy]) * 1000.0
    err_fit_e = np.asarray(ref["val_id_abs_err_ut"][m_easy], dtype=np.float64)
    h2_boot = paired_bootstrap_one_sided_p(
        err_nn_e, err_fit_e, lambda arr: np.median(arr, axis=1), margin=15.0, n_boot=4000, seed=3002
    )
    p_h2_raw = float(h2_boot["p_one_sided"])

    # H3: Selective hybrid improves tail accuracy (P90 or Fail>100 uT) at lower mean runtime (val_rep N=600)
    err_sel_600 = np.abs(sel_b_rep - val_rep["Y_b_mag"][sub_rep_pos]) * 1000.0
    err_fit_600 = np.asarray(ref["val_id_abs_err_ut"][sub_rep_pos], dtype=np.float64)
    fail_sel_600 = err_sel_600 > 100.0
    fail_fit_600 = np.asarray(ref["val_id_fail_100"][sub_rep_pos], dtype=bool)
    h3_mcn = mcnemar_paired_test(fail_sel_600, fail_fit_600)
    h3_p90_boot = paired_bootstrap_one_sided_p(
        err_sel_600, err_fit_600, lambda arr: np.percentile(arr, 90.0, axis=1), margin=0.0, n_boot=4000, seed=3003
    )
    rt_sel_mean = float(np.mean(sel_rt_rep))
    rt_fit_mean = float(np.mean(ref["val_id_runtime_ms"][sub_rep_pos]))
    p_h3_fail = float(h3_mcn["p_value"]) if h3_mcn["fail_rate_a"] < h3_mcn["fail_rate_b"] else 1.0
    p_h3_p90 = float(h3_p90_boot["p_one_sided"])
    # Bonferroni-conservative combination of the two sub-criteria (2 * min) when runtime is lower
    p_h3_raw = min(1.0, 2.0 * min(p_h3_fail, p_h3_p90)) if (rt_sel_mean < rt_fit_mean) else 1.0

    # H4: NN denoiser beats tuned SG in Dip-Region RMSE without Dip-Position degradation (val_rep N=3,000)
    h4_boot = paired_bootstrap_one_sided_p(
        den_rep_nn["_per_sample_dip_rmse"],
        den_rep_sg["_per_sample_dip_rmse"],
        lambda arr: np.mean(arr, axis=1),
        margin=0.0,
        n_boot=4000,
        seed=3004,
    )
    pos_ok = den_rep_nn["median_pos_err_mhz"] <= (den_rep_sg["median_pos_err_mhz"] + 0.10)
    p_h4_raw = float(h4_boot["p_one_sided"]) if pos_ok else 1.0

    p_raws = [p_h1_raw, p_h2_raw, p_h3_raw, p_h4_raw]
    p_adjs = holm_bonferroni_adjust(p_raws)

    def _verdict(cond_dir: bool, cond_contra: bool, p_adj: float) -> str:
        if cond_dir and p_adj < 0.05:
            return "supported"
        if cond_contra:
            return "not supported"
        return "inconclusive"

    hypotheses = {
        "H1_hard_wrong_basin": {
            "nn_wb_pct": round(h1_mcn["fail_rate_a"] * 100.0, 2),
            "fitter_wb_pct": round(h1_mcn["fail_rate_b"] * 100.0, 2),
            "diff_pct_pts": round(h1_mcn["fail_rate_diff"] * 100.0, 2),
            "mcnemar": h1_mcn,
            "p_raw": p_h1_raw,
            "p_holm_adj": p_adjs[0],
            "verdict": _verdict(
                h1_mcn["fail_rate_a"] < h1_mcn["fail_rate_b"],
                h1_mcn["fail_rate_a"] >= h1_mcn["fail_rate_b"],
                p_adjs[0],
            ),
        },
        "H2_easy_non_inferiority": {
            "nn_med_ut": round(float(np.median(err_nn_e)), 2),
            "fitter_med_ut": round(float(np.median(err_fit_e)), 2),
            "margin_ut": 15.0,
            "paired_bootstrap": h2_boot,
            "p_raw": p_h2_raw,
            "p_holm_adj": p_adjs[1],
            "verdict": _verdict(
                h2_boot["point_diff"] < 15.0,
                h2_boot["ci_95"][0] >= 15.0,
                p_adjs[1],
            ),
        },
        "H3_selective_hybrid_pareto": {
            "selective_mean_rt_ms": round(rt_sel_mean, 2),
            "fitter_mean_rt_ms": round(rt_fit_mean, 2),
            "speedup_x": round(rt_fit_mean / max(rt_sel_mean, 1e-6), 2),
            "fallback_pct": paired_rep_600["selective_hybrid_fallback_pct"],
            "fail_mcnemar": h3_mcn,
            "p90_bootstrap": h3_p90_boot,
            "p_fail_raw": p_h3_fail,
            "p_p90_raw": p_h3_p90,
            "p_raw_combined": p_h3_raw,
            "p_holm_adj": p_adjs[2],
            "verdict": _verdict(
                (rt_sel_mean < rt_fit_mean) and ((h3_mcn["fail_rate_a"] < h3_mcn["fail_rate_b"]) or (h3_p90_boot["point_diff"] < 0.0)),
                (rt_sel_mean >= rt_fit_mean) or ((h3_mcn["fail_rate_a"] >= h3_mcn["fail_rate_b"]) and (h3_p90_boot["point_diff"] >= 0.0)),
                p_adjs[2],
            ),
        },
        "H4_denoiser_vs_savgol": {
            "nn_mean_dip_rmse": den_rep_nn["mean_dip_rmse"],
            "sg_mean_dip_rmse": den_rep_sg["mean_dip_rmse"],
            "nn_median_pos_err_mhz": den_rep_nn["median_pos_err_mhz"],
            "sg_median_pos_err_mhz": den_rep_sg["median_pos_err_mhz"],
            "pos_non_degradation_ok": pos_ok,
            "paired_bootstrap": h4_boot,
            "p_raw": p_h4_raw,
            "p_holm_adj": p_adjs[3],
            "verdict": _verdict(
                (den_rep_nn["mean_dip_rmse"] < den_rep_sg["mean_dip_rmse"]) and pos_ok,
                (den_rep_nn["mean_dip_rmse"] >= den_rep_sg["mean_dip_rmse"]) or (not pos_ok),
                p_adjs[3],
            ),
        },
    }

    for hk, hv in hypotheses.items():
        print(f"  {hk:28s}: p_raw={hv.get('p_raw', hv.get('p_raw_combined')):.5g}, p_holm={hv['p_holm_adj']:.5g} -> VERDICT: {hv['verdict'].upper()}", flush=True)

    # 9. Final integrity & vault verification
    h_post = verify_frozen_hashes()
    assert h_post["ok"], f"Frozen hash verification FAILED after Run 3c: {h_post}"
    v_post = get_test_access_counts()
    assert v_post["data_read"] == 0, f"Test vault breach after Run 3c: {v_post}"

    payload = {
        "total_eval_wall_sec": round(time.perf_counter() - t_all, 2),
        "integrity_post_check": {"frozen_hashes_ok": h_post["ok"], "checked_files": h_post["checked_files"], "test_data_read": v_post["data_read"]},
        "nn_single_spectrum_cpu_ms": {"mean": round(nn_single_ms_mean, 2), "median": round(nn_single_ms_med, 2)},
        "val_sel_tuning": {
            "best_savgol": best_sg,
            "best_gaussian": best_gauss,
            "mad_floor_diagnostic": mad_diag,
            "selective_hybrid_tuning": sel_tune,
        },
        "val_rep_3000": val_rep_summary,
        "val_rep_model_a_3seed_spread": seed_spread_val_rep,
        "val_id_natural_1500": val_nat_summary,
        "val_ood_1500": val_ood_summary,
        "ood_9_factors": ood_factors_summary,
        "paired_val_rep_600": paired_rep_600,
        "paired_val_rep_600_pareto_curve": rep_pareto_curve,
        "paired_val_ood_300": paired_ood_300,
        "slices": slices_summary,
        "denoising": denoising_summary,
        "uncertainty": unc_summary,
        "hallucination_and_robustness": halluc_summary,
        "real_trianam_traces": real_trace_results,
        "real_noise_injection_stress": noise_inj_results,
        "preregistered_hypotheses": hypotheses,
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nRUN 3c COMPLETE in {payload['total_eval_wall_sec']} s! Saved: {RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
