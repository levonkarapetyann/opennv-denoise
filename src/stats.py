"""Paired bootstrap confidence intervals and McNemar tests for ODMR estimators."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.stats import binomtest


def bootstrap_metric_ci(
    values: np.ndarray,
    metric_fn: Callable[[np.ndarray], np.ndarray],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute point estimate and bootstrap percentile CI for a 1D array metric.

    Args:
        values: 1D sample array of shape ``(N,)``.
        metric_fn: Vectorized callable mapping an array of shape ``(B, N)``
            along ``axis=-1`` to a 1D array of shape ``(B,)``.
        n_boot: Number of bootstrap resamples.
        ci: Confidence level in ``(0, 1)`` (default ``0.95``).
        seed: Deterministic random seed.

    Returns:
        Tuple ``(point_estimate, ci_low, ci_high)``.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Expected non-empty 1D array, got shape {arr.shape}.")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci}.")
    if n_boot < 10:
        raise ValueError(f"n_boot must be >= 10, got {n_boot}.")

    n = arr.shape[0]
    point = float(metric_fn(arr[None, :])[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = arr[idx]
    boot_vals = np.asarray(metric_fn(samples), dtype=np.float64)
    alpha = 0.5 * (1.0 - ci) * 100.0
    lo, hi = np.percentile(boot_vals, [alpha, 100.0 - alpha])
    return point, float(lo), float(hi)


def bootstrap_summary_cis(
    errs: np.ndarray,
    fail_flags: Optional[np.ndarray] = None,
    wrong_basin_flags: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute bootstrap CIs for median, P90, RMS, MAE, failure rate, and wrong-basin rate.

    Args:
        errs: 1D array of absolute errors ``|B_est - B_true|`` of shape ``(N,)``.
        fail_flags: Optional 1D boolean array of failure indicators of shape ``(N,)``.
        wrong_basin_flags: Optional 1D boolean array of wrong-basin indicators of shape ``(N,)``.
        n_boot: Number of bootstrap resamples.
        ci: Confidence level in ``(0, 1)``.
        seed: Deterministic random seed.

    Returns:
        Dictionary mapping metric names (``"median"``, ``"p90"``, ``"rms"``,
        ``"mae"``, and optionally ``"fail_rate"``, ``"wrong_basin_rate"``)
        to ``(point_estimate, ci_low, ci_high)``.
    """
    err_arr = np.asarray(errs, dtype=np.float64)
    if err_arr.ndim != 1 or err_arr.size == 0:
        raise ValueError(f"Expected non-empty 1D errs array, got shape {err_arr.shape}.")

    n = err_arr.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_e = err_arr[idx]
    alpha = 0.5 * (1.0 - ci) * 100.0
    q_bounds = [alpha, 100.0 - alpha]

    med_pt = float(np.median(err_arr))
    med_boot = np.median(boot_e, axis=1)
    med_lo, med_hi = np.percentile(med_boot, q_bounds)

    p90_pt = float(np.percentile(err_arr, 90.0))
    p90_boot = np.percentile(boot_e, 90.0, axis=1)
    p90_lo, p90_hi = np.percentile(p90_boot, q_bounds)

    rms_pt = float(np.sqrt(np.mean(err_arr * err_arr)))
    rms_boot = np.sqrt(np.mean(boot_e * boot_e, axis=1))
    rms_lo, rms_hi = np.percentile(rms_boot, q_bounds)

    mae_pt = float(np.mean(err_arr))
    mae_boot = np.mean(boot_e, axis=1)
    mae_lo, mae_hi = np.percentile(mae_boot, q_bounds)

    out: Dict[str, Tuple[float, float, float]] = {
        "median": (med_pt, float(med_lo), float(med_hi)),
        "p90": (p90_pt, float(p90_lo), float(p90_hi)),
        "rms": (rms_pt, float(rms_lo), float(rms_hi)),
        "mae": (mae_pt, float(mae_lo), float(mae_hi)),
    }

    if fail_flags is not None:
        f_arr = np.asarray(fail_flags, dtype=np.float64)
        if f_arr.shape != err_arr.shape:
            raise ValueError("fail_flags must match errs shape.")
        f_pt = float(np.mean(f_arr))
        f_boot = np.mean(f_arr[idx], axis=1)
        f_lo, f_hi = np.percentile(f_boot, q_bounds)
        out["fail_rate"] = (f_pt, float(f_lo), float(f_hi))

    if wrong_basin_flags is not None:
        wb_arr = np.asarray(wrong_basin_flags, dtype=np.float64)
        if wb_arr.shape != err_arr.shape:
            raise ValueError("wrong_basin_flags must match errs shape.")
        wb_pt = float(np.mean(wb_arr))
        wb_boot = np.mean(wb_arr[idx], axis=1)
        wb_lo, wb_hi = np.percentile(wb_boot, q_bounds)
        out["wrong_basin_rate"] = (wb_pt, float(wb_lo), float(wb_hi))

    return out


def paired_bootstrap_diff_ci(
    err_a: np.ndarray,
    err_b: np.ndarray,
    fail_a: Optional[np.ndarray] = None,
    fail_b: Optional[np.ndarray] = None,
    fail_threshold: Optional[float] = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute paired bootstrap CIs for differences (Method A - Method B) on the same spectra.

    Resamples spectrum indices ``idx`` jointly so both methods are evaluated on
    the exact same bootstrap sample in every draw, preserving pairwise correlation.

    Args:
        err_a: 1D array of absolute errors for Method A of shape ``(N,)``.
        err_b: 1D array of absolute errors for Method B of shape ``(N,)``.
        fail_a: Optional 1D boolean failure indicators for Method A of shape ``(N,)``.
        fail_b: Optional 1D boolean failure indicators for Method B of shape ``(N,)``.
        fail_threshold: If ``fail_a`` / ``fail_b`` are omitted and ``fail_threshold``
            is provided, defines failure as ``err > fail_threshold``.
        n_boot: Number of paired bootstrap resamples.
        ci: Confidence level in ``(0, 1)`` (default ``0.95``).
        seed: Deterministic random seed.

    Returns:
        Dictionary with keys ``"median_diff"``, ``"rms_diff"``, ``"mae_diff"``,
        ``"p90_diff"``, and (when failure flags or threshold are provided)
        ``"fail_rate_diff"``, each mapping to ``(point_diff, ci_low, ci_high)``
        for ``Method A - Method B``.
    """
    a = np.asarray(err_a, dtype=np.float64)
    b = np.asarray(err_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError(
            f"err_a and err_b must be non-empty 1D arrays of identical shape, got {a.shape} vs {b.shape}."
        )
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci}.")
    if n_boot < 10:
        raise ValueError(f"n_boot must be >= 10, got {n_boot}.")

    n = a.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_a = a[idx]
    boot_b = b[idx]

    alpha = 0.5 * (1.0 - ci) * 100.0
    q_bounds = [alpha, 100.0 - alpha]

    med_diff_pt = float(np.median(a) - np.median(b))
    med_diff_boot = np.median(boot_a, axis=1) - np.median(boot_b, axis=1)
    med_lo, med_hi = np.percentile(med_diff_boot, q_bounds)

    rms_a_pt = float(np.sqrt(np.mean(a * a)))
    rms_b_pt = float(np.sqrt(np.mean(b * b)))
    rms_diff_pt = rms_a_pt - rms_b_pt
    rms_diff_boot = np.sqrt(np.mean(boot_a * boot_a, axis=1)) - np.sqrt(
        np.mean(boot_b * boot_b, axis=1)
    )
    rms_lo, rms_hi = np.percentile(rms_diff_boot, q_bounds)

    mae_diff_pt = float(np.mean(a) - np.mean(b))
    mae_diff_boot = np.mean(boot_a, axis=1) - np.mean(boot_b, axis=1)
    mae_lo, mae_hi = np.percentile(mae_diff_boot, q_bounds)

    p90_diff_pt = float(np.percentile(a, 90.0) - np.percentile(b, 90.0))
    p90_diff_boot = np.percentile(boot_a, 90.0, axis=1) - np.percentile(boot_b, 90.0, axis=1)
    p90_lo, p90_hi = np.percentile(p90_diff_boot, q_bounds)

    result: Dict[str, Tuple[float, float, float]] = {
        "median_diff": (med_diff_pt, float(med_lo), float(med_hi)),
        "rms_diff": (rms_diff_pt, float(rms_lo), float(rms_hi)),
        "mae_diff": (mae_diff_pt, float(mae_lo), float(mae_hi)),
        "p90_diff": (p90_diff_pt, float(p90_lo), float(p90_hi)),
    }

    if fail_a is None and fail_b is None and fail_threshold is not None:
        fa = (a > float(fail_threshold)).astype(np.float64)
        fb = (b > float(fail_threshold)).astype(np.float64)
    elif fail_a is not None and fail_b is not None:
        fa = np.asarray(fail_a, dtype=bool).astype(np.float64)
        fb = np.asarray(fail_b, dtype=bool).astype(np.float64)
        if fa.shape != a.shape or fb.shape != b.shape:
            raise ValueError("fail_a and fail_b must have the same shape as err_a and err_b.")
    else:
        fa = None
        fb = None

    if fa is not None and fb is not None:
        fail_diff_pt = float(np.mean(fa) - np.mean(fb))
        fail_diff_boot = np.mean(fa[idx], axis=1) - np.mean(fb[idx], axis=1)
        f_lo, f_hi = np.percentile(fail_diff_boot, q_bounds)
        result["fail_rate_diff"] = (fail_diff_pt, float(f_lo), float(f_hi))

    return result


def mcnemar_paired_test(
    fail_a: np.ndarray,
    fail_b: np.ndarray,
) -> Dict[str, Any]:
    """Perform McNemar's test comparing paired binary failure flags on the same spectra.

    Evaluates the null hypothesis that Method A and Method B have equal marginal
    failure probabilities by conditioning on the discordant pairs ``(n_a_only, n_b_only)``.
    Computes both the exact two-sided binomial p-value and the continuity-corrected
    McNemar chi-square statistic.

    Args:
        fail_a: 1D boolean (or 0/1) array of failure flags for Method A of shape ``(N,)``.
        fail_b: 1D boolean (or 0/1) array of failure flags for Method B of shape ``(N,)``.

    Returns:
        Dictionary containing:
          - ``"n_samples"``: Total paired observations ``N``.
          - ``"n_a_only"``: Count where A failed and B succeeded (``n_10``).
          - ``"n_b_only"``: Count where A succeeded and B failed (``n_01``).
          - ``"n_both_fail"``: Count where both failed (``n_11``).
          - ``"n_both_pass"``: Count where both succeeded (``n_00``).
          - ``"n_discordant"``: ``n_a_only + n_b_only``.
          - ``"fail_rate_a"``: Fraction of A failures.
          - ``"fail_rate_b"``: Fraction of B failures.
          - ``"fail_rate_diff"``: ``fail_rate_a - fail_rate_b``.
          - ``"chi2_cc"``: Continuity-corrected McNemar chi-square statistic.
          - ``"p_value"``: Exact two-sided binomial test p-value.
    """
    fa = np.asarray(fail_a, dtype=bool)
    fb = np.asarray(fail_b, dtype=bool)
    if fa.ndim != 1 or fb.ndim != 1 or fa.shape != fb.shape or fa.size == 0:
        raise ValueError(
            f"fail_a and fail_b must be non-empty 1D arrays of identical shape, got {fa.shape} vs {fb.shape}."
        )

    n = int(fa.size)
    n_a_only = int(np.sum(fa & (~fb)))
    n_b_only = int(np.sum((~fa) & fb))
    n_both_fail = int(np.sum(fa & fb))
    n_both_pass = int(np.sum((~fa) & (~fb)))
    n_discordant = n_a_only + n_b_only

    if n_discordant == 0:
        p_val = 1.0
        chi2_cc = 0.0
    else:
        p_val = float(
            binomtest(min(n_a_only, n_b_only), n=n_discordant, p=0.5, alternative="two-sided").pvalue
        )
        abs_diff = abs(n_a_only - n_b_only)
        num = max(0.0, float(abs_diff) - 1.0)
        chi2_cc = (num * num) / float(n_discordant)

    rate_a = float(np.mean(fa))
    rate_b = float(np.mean(fb))
    return {
        "n_samples": n,
        "n_a_only": n_a_only,
        "n_b_only": n_b_only,
        "n_both_fail": n_both_fail,
        "n_both_pass": n_both_pass,
        "n_discordant": n_discordant,
        "fail_rate_a": rate_a,
        "fail_rate_b": rate_b,
        "fail_rate_diff": rate_a - rate_b,
        "chi2_cc": float(chi2_cc),
        "p_value": p_val,
    }
