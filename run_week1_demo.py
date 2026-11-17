"""Stage 1.5 Reporting-Honesty Benchmark (v2): Full Descriptive Statistics & Variant Evaluation.

Strict Seed & Freeze Policy:
  - src/baseline.py is FROZEN at tag ``baseline-frozen`` (unmodified).
  - DEV_SEED   = 10001  : Used exclusively for development diagnostics and tuning prior lambda = 2.5e-4.
  - TEST_SEED  = 90001  : Single degenerate-axis demo vector B = (1.15, 0.75, 0.40) mT (not representative).
  - FINAL_SEED = 500001 : Held-out 200-spectrum evaluation set on S^2 (incl. 20% near-degenerate).

Statistical Reporting Policy (Task 2):
  - Reports full-sample RMS (over ALL runs, no exclusions), Median (P50), P90, P95, P99, and MAE of |err|.
  - Reports relative failure rate (|err_i| > 5 * CRB_i), wrong-basin rate (canonical-wedge distance > 0.15 mT),
    and legacy absolute failure rate (|err_i| > 100 uT).
  - ``sig_core`` (RMS of non-failed runs where |err_i| <= 5 * CRB_i) is included ONLY as an auxiliary column
    explicitly labelled "conditional on success; biased low by construction" and is NOT divided by CRB.
  - The CRB ratio column ``RMS/CRB_m`` divides the full-sample RMS (all runs) by the dataset median per-spectrum
    white-noise CRB (``CRB_med``).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.baseline import fit_lorentzian_naive, fit_physical_odmr
from src.baseline_variants import (
    DEFAULT_PRIOR_LAMBDA,
    fit_informed_harmonics,
    fit_informed_unknown_freq,
    fit_lorentzian_naive_fixed,
    fit_physical_with_prior,
)
from src.config import CANONICAL_FREQS, D0_MHZ, NV_AXES
from src.noise import HardwareNoiseConfig, apply_realistic_noise
from src.physics import (
    ODMRParams,
    canonicalize_b_vec,
    pseudo_voigt_dip,
    simulate_clean_odmr,
    solve_axis_transitions,
)

DEV_SEED: int = 10001
TEST_SEED: int = 90001
FINAL_SEED: int = 500001
WRONG_BASIN_THRESH_MT: float = 0.15


@dataclass(frozen=True)
class SNRRegime:
    """Configuration for a single COTS SNR benchmark regime."""

    label: str
    demo_eff_snr_outer: float
    noise_cfg: HardwareNoiseConfig


def compute_crb_b_mag_ut(
    freqs: np.ndarray,
    b_vec: np.ndarray,
    noise_cfg: HardwareNoiseConfig,
    d0_mhz: float = D0_MHZ,
    gamma_mhz: float = 4.5,
    contrast: float = 0.03,
    eta: float = 0.35,
) -> float:
    """Compute white-noise Cramér-Rao lower bound (in uT) for |B|."""
    b_mag = float(np.linalg.norm(b_vec))
    u_hat = b_vec / max(b_mag, 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(u_hat[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(u_hat, ref)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(u_hat, t1)
    u_grid = (freqs - D0_MHZ) / 120.0

    def model_from_params(p: np.ndarray) -> np.ndarray:
        bv = (p[0] * u_hat) + (p[1] * t1) + (p[2] * t2)
        sim = simulate_clean_odmr(
            freqs,
            ODMRParams(
                b_field_mt=(float(bv[0]), float(bv[1]), float(bv[2])),
                base_contrast=float(p[5]),
                linewidth_mhz=float(p[4]),
                voigt_eta=eta,
                d0_mhz=float(p[3]),
            ),
        )["spectrum"]
        return sim + (p[6] - 1.0) + p[7] * u_grid

    p0 = np.array([b_mag, 0.0, 0.0, d0_mhz, gamma_mhz, contrast, 1.0, 0.0], dtype=np.float64)
    mu0 = model_from_params(p0)
    signal_depth = max(float(np.max(mu0) - np.min(mu0)), 1e-6)
    noise_std = signal_depth / float(noise_cfg.target_snr)
    var_i = (noise_std * noise_std) * np.clip(mu0, 0.5, 1.5)

    eps = np.array([1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6], dtype=np.float64)
    jac = np.zeros((len(freqs), 8), dtype=np.float64)
    for j in range(8):
        dp = np.zeros(8, dtype=np.float64)
        dp[j] = eps[j]
        jac[:, j] = (model_from_params(p0 + dp) - model_from_params(p0 - dp)) / (2.0 * eps[j])

    fim = (jac.T / var_i[None, :]) @ jac
    cov = np.linalg.pinv(fim, rcond=1e-12)
    return float(np.sqrt(max(cov[0, 0], 0.0))) * 1000.0


def compute_dip_depths(
    freqs: np.ndarray,
    b_vec: np.ndarray,
    d0_mhz: float,
    gamma_mhz: float,
    contrast: float,
    eta: float = 0.35,
    clean_spec: np.ndarray | None = None,
) -> Tuple[float, float]:
    """Compute (deepest_dip_depth, outer_informative_dip_depth) for a spectrum (Task 7)."""
    if clean_spec is None:
        clean_spec = simulate_clean_odmr(
            freqs,
            ODMRParams(
                b_field_mt=(float(b_vec[0]), float(b_vec[1]), float(b_vec[2])),
                base_contrast=contrast,
                linewidth_mhz=gamma_mhz,
                voigt_eta=eta,
                d0_mhz=d0_mhz,
            ),
        )["spectrum"]
    deepest_depth = float(1.0 - np.min(clean_spec))

    # Outermost informative axis (axis with largest |B_par| projection, which has highest weight in |B|)
    b_par = np.abs(NV_AXES @ b_vec)
    b_sq = float(np.dot(b_vec, b_vec))
    b_perp = np.sqrt(np.maximum(0.0, b_sq - b_par * b_par))
    outer_idx = int(np.argmax(b_par))
    fm, fp, cf = solve_axis_transitions(float(b_par[outer_idx]), float(b_perp[outer_idx]), d0=d0_mhz)
    eff_amp = contrast * 0.28 * float(cf) / 3.0
    # Evaluate the 3 hyperfine components of the outermost axis transition around f_minus
    hf_prof = np.zeros_like(freqs)
    for dhf in (-2.16, 0.0, 2.16):
        hf_prof += pseudo_voigt_dip(freqs, float(fm) + dhf, eff_amp, gamma_mhz, eta)
    outer_dip_depth = float(np.max(hf_prof))
    return deepest_depth, outer_dip_depth


def _run_all_methods_on_spectrum(
    clean_spectrum: np.ndarray,
    noise_cfg: HardwareNoiseConfig,
    seed: int,
) -> Dict[str, Any]:
    """Generate noisy spectrum via apply_realistic_noise(return_components=True) and run all 11 fitters."""
    rng = np.random.default_rng(seed)
    noisy_out = apply_realistic_noise(clean_spectrum, noise_cfg, rng=rng, return_components=True)
    assert isinstance(noisy_out, tuple)
    noisy, comps = noisy_out

    true_bl_drift_only = 1.0 + comps["drift"]
    true_bl_full = 1.0 + comps["drift"] + comps["hum"]
    total_noise_rms = float(np.sqrt(np.mean((noisy - clean_spectrum) ** 2)))
    dwell_sec = noise_cfg.dwell_time_ms * 1e-3

    r_naive_frozen = fit_lorentzian_naive(CANONICAL_FREQS, noisy)
    r_naive_fixed = fit_lorentzian_naive_fixed(CANONICAL_FREQS, noisy)
    r_plain_deg2 = fit_physical_odmr(CANONICAL_FREQS, noisy, variant="plain", poly_degree=2)
    r_plain_deg3 = fit_physical_odmr(CANONICAL_FREQS, noisy, variant="plain", poly_degree=3)
    r_prior_deg2 = fit_physical_with_prior(
        CANONICAL_FREQS, noisy, poly_degree=2, prior_lambda=DEFAULT_PRIOR_LAMBDA
    )
    r_prior_deg3 = fit_physical_with_prior(
        CANONICAL_FREQS, noisy, poly_degree=3, prior_lambda=DEFAULT_PRIOR_LAMBDA
    )
    r_inf_unk = fit_informed_unknown_freq(
        CANONICAL_FREQS,
        noisy,
        dwell_time=dwell_sec,
        poly_degree=2,
        precomputed_basin=r_plain_deg2,
    )
    r_inf_harm = fit_informed_harmonics(
        CANONICAL_FREQS,
        noisy,
        dwell_time=dwell_sec,
        poly_degree=2,
        precomputed_basin=r_plain_deg2,
    )
    r_inf_deg3 = fit_physical_odmr(
        CANONICAL_FREQS, noisy, dwell_time=dwell_sec, variant="informed", poly_degree=3
    )
    r_oracle_drift = fit_physical_odmr(
        CANONICAL_FREQS, noisy, oracle_baseline=true_bl_drift_only, variant="oracle_drift"
    )
    r_oracle_full = fit_physical_odmr(
        CANONICAL_FREQS, noisy, oracle_baseline=true_bl_full, variant="oracle_baseline"
    )
    return {
        "noisy": noisy,
        "total_noise_rms": total_noise_rms,
        "true_bl_drift_only": true_bl_drift_only,
        "true_bl_full": true_bl_full,
        "naive_frozen": r_naive_frozen,
        "naive_fixed": r_naive_fixed,
        "plain_deg2": r_plain_deg2,
        "plain_deg3": r_plain_deg3,
        "prior_deg2": r_prior_deg2,
        "prior_deg3": r_prior_deg3,
        "inf_unk": r_inf_unk,
        "inf_harm": r_inf_harm,
        "inf_deg3": r_inf_deg3,
        "oracle_drift": r_oracle_drift,
        "oracle_full": r_oracle_full,
    }


def _demo_trial_worker(args: Tuple[int, np.ndarray, HardwareNoiseConfig]) -> Dict[str, Any]:
    trial_idx, clean_spectrum, noise_cfg = args
    seed = TEST_SEED + int(noise_cfg.target_snr * 1000) + trial_idx * 17
    out = _run_all_methods_on_spectrum(clean_spectrum, noise_cfg, seed)
    out["trial_idx"] = trial_idx
    return out


def _final_set_worker(args: Tuple[Dict[str, Any], HardwareNoiseConfig]) -> Dict[str, Any]:
    item, noise_cfg = args
    idx = int(item["idx"])
    seed = FINAL_SEED + int(noise_cfg.target_snr * 1000) + idx * 29
    out = _run_all_methods_on_spectrum(item["clean"], noise_cfg, seed)
    crb_ref_ut = compute_crb_b_mag_ut(
        CANONICAL_FREQS, item["b_vec"], noise_cfg, d0_mhz=D0_MHZ, gamma_mhz=4.5, contrast=0.03
    )
    crb_true_ut = compute_crb_b_mag_ut(
        CANONICAL_FREQS,
        item["b_vec"],
        noise_cfg,
        d0_mhz=float(item["d0"]),
        gamma_mhz=float(item["gamma"]),
        contrast=float(item["contrast"]),
    )
    out["idx"] = idx
    out["b_mag"] = item["b_mag"]
    out["b_canon"] = item["b_canon"]
    out["crb_ref_ut"] = crb_ref_ut
    out["crb_true_ut"] = crb_true_ut
    out["eff_snr_deepest"] = float(item["deepest_depth"]) / max(out["total_noise_rms"], 1e-12)
    out["eff_snr_outer"] = float(item["outer_depth"]) / max(out["total_noise_rms"], 1e-12)
    return out


def generate_final_evaluation_set(n_spectra: int = 200, seed: int = FINAL_SEED) -> List[Dict[str, Any]]:
    """Generate the 200-spectrum held-out FINAL set on S^2 (including 20% near-degenerate directions)."""
    rng = np.random.default_rng(seed)
    items: List[Dict[str, Any]] = []
    for i in range(n_spectra):
        if i % 10 == 0:
            t = float(rng.uniform(0.10, 0.48))
            u = np.array([t, 1.0 - t + float(rng.normal(0.0, 0.02)), 1.0], dtype=np.float64)
            u = rng.choice([-1.0, 1.0], size=3) * rng.permutation(u)
        elif i % 10 == 1:
            t = float(rng.uniform(0.15, 0.85))
            u = np.array([t, t + float(rng.normal(0.0, 0.02)), 1.0], dtype=np.float64)
            u = rng.choice([-1.0, 1.0], size=3) * rng.permutation(u)
        else:
            u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        b_mag = float(rng.uniform(0.1, 3.2))
        b_vec = u * b_mag
        gamma = float(rng.uniform(3.0, 9.0))
        d0 = float(rng.uniform(2867.5, 2872.5))
        contrast = float(rng.uniform(0.02, 0.04))
        clean = simulate_clean_odmr(
            CANONICAL_FREQS,
            ODMRParams(
                b_field_mt=tuple(b_vec.tolist()),
                linewidth_mhz=gamma,
                d0_mhz=d0,
                base_contrast=contrast,
            ),
        )["spectrum"]
        deepest_d, outer_d = compute_dip_depths(
            CANONICAL_FREQS, b_vec, d0, gamma, contrast, eta=0.35, clean_spec=clean
        )
        items.append(
            {
                "idx": i,
                "b_vec": b_vec,
                "b_canon": canonicalize_b_vec(b_vec),
                "b_mag": b_mag,
                "gamma": gamma,
                "d0": d0,
                "contrast": contrast,
                "clean": clean,
                "deepest_depth": deepest_d,
                "outer_depth": outer_d,
            }
        )
    return items


def summarize_metrics(
    errs_ut: np.ndarray,
    crbs_ut: np.ndarray,
    wedge_dists_mt: np.ndarray,
    mean_rt_ms: float,
    is_naive: bool = False,
) -> Dict[str, Any]:
    """Compute Task 2 full-sample and conditional error statistics."""
    rms_all_ut = float(np.sqrt(np.mean(errs_ut ** 2)))
    p50_ut = float(np.percentile(errs_ut, 50))
    p90_ut = float(np.percentile(errs_ut, 90))
    p95_ut = float(np.percentile(errs_ut, 95))
    p99_ut = float(np.percentile(errs_ut, 99))
    mae_ut = float(np.mean(errs_ut))

    rel_fail_mask = errs_ut > (5.0 * crbs_ut)
    fail_5x_pct = float(np.mean(rel_fail_mask) * 100.0)
    fail_100_pct = float(np.mean(errs_ut > 100.0) * 100.0)
    wrong_basin_pct = float(np.mean(wedge_dists_mt > WRONG_BASIN_THRESH_MT) * 100.0)

    core_mask = ~rel_fail_mask
    if np.any(core_mask):
        sigma_core_ut = float(np.sqrt(np.mean(errs_ut[core_mask] ** 2)))
    else:
        sigma_core_ut = rms_all_ut

    med_crb_ut = float(np.median(crbs_ut))
    rms_over_crb_med = rms_all_ut / max(med_crb_ut, 1e-9)

    return {
        "rms_all_ut": rms_all_ut,
        "med_ut": p50_ut,
        "p90_ut": p90_ut,
        "p95_ut": p95_ut,
        "p99_ut": p99_ut,
        "mae_ut": mae_ut,
        "sigma_core_ut": sigma_core_ut,
        "fail_5x_pct": fail_5x_pct,
        "wrong_basin_pct": wrong_basin_pct,
        "fail_100_pct": fail_100_pct,
        "mean_rt_ms": mean_rt_ms,
        "rms_over_crb_med": rms_over_crb_med,
        "med_crb_ut": med_crb_ut,
        "is_naive": is_naive,
    }


def print_table_header(title: str, subtitle: str) -> None:
    print("\n" + "=" * 168)
    print(title)
    print(subtitle)
    print("=" * 168)
    print(
        f"{'SNR':>3} | {'EffOut':>6} | {'Method':<44} | {'RMS(uT)':>8} | {'P50(uT)':>8} | "
        f"{'P90(uT)':>8} | {'P95(uT)':>8} | {'P99(uT)':>8} | {'MAE(uT)':>8} | "
        f"{'Fail>5xC':>8} | {'WrongBas':>8} | {'Fail>100':>8} | {'sig_core+':>9} | "
        f"{'RMS/CRBm':>8} | {'Time(ms)':>8}"
    )
    print("-" * 168)


def print_table_row(snr: float, eff_snr: float, label: str, s: Dict[str, Any]) -> None:
    wb_str = f"{s['wrong_basin_pct']:>7.1f}%" if not s["is_naive"] else f"{'N/A(1D)':>8}"
    print(
        f"{snr:>3.0f} | {eff_snr:>6.2f} | {label:<44} | {s['rms_all_ut']:>8.2f} | {s['med_ut']:>8.2f} | "
        f"{s['p90_ut']:>8.2f} | {s['p95_ut']:>8.2f} | {s['p99_ut']:>8.2f} | {s['mae_ut']:>8.2f} | "
        f"{s['fail_5x_pct']:>7.1f}% | {wb_str} | {s['fail_100_pct']:>7.1f}% | "
        f"{s['sigma_core_ut']:>9.2f} | {s['rms_over_crb_med']:>7.2f}x | {s['mean_rt_ms']:>8.1f}"
    )


def main() -> None:
    regimes: List[SNRRegime] = [
        SNRRegime(
            label="High SNR",
            demo_eff_snr_outer=4.07,
            noise_cfg=HardwareNoiseConfig(
                target_snr=15.0,
                dwell_time_ms=0.8,
                mains_freq_hz=50.0,
                hum_amp=0.0015,
                baseline_drift_amp=0.005,
                pink_noise_ratio=0.25,
            ),
        ),
        SNRRegime(
            label="Medium COTS",
            demo_eff_snr_outer=1.51,
            noise_cfg=HardwareNoiseConfig(
                target_snr=4.0,
                dwell_time_ms=0.8,
                mains_freq_hz=50.0,
                hum_amp=0.0030,
                baseline_drift_amp=0.012,
                pink_noise_ratio=0.25,
            ),
        ),
        SNRRegime(
            label="Low COTS",
            demo_eff_snr_outer=0.93,
            noise_cfg=HardwareNoiseConfig(
                target_snr=2.0,
                dwell_time_ms=0.8,
                mains_freq_hz=50.0,
                hum_amp=0.0035,
                baseline_drift_amp=0.015,
                pink_noise_ratio=0.25,
            ),
        ),
    ]

    method_specs = [
        ("naive_frozen", "naive (frozen: biased hs_ub bug)"),
        ("naive_fixed", "naive (fixed: 4-pair, model-mismatched)"),
        ("plain_deg2", "physical, plain (deg-2)"),
        ("plain_deg3", "physical, plain (deg-3: cubic drift match)"),
        ("prior_deg2", "physical, with_prior (deg-2, lam=2.5e-4)"),
        ("prior_deg3", "physical, with_prior (deg-3, lam=2.5e-4)"),
        ("inf_unk", "physical, informed_unknown_freq (45-65Hz)"),
        ("inf_harm", "physical, informed_harmonics (50/60Hz 1-3f)"),
        ("inf_deg3", "physical, informed (50/60Hz: inv crime UB)"),
        ("oracle_drift", "physical, oracle (known drift only)"),
        ("oracle_full", "physical, oracle (true drift + hum)"),
    ]

    import sys as _sys
    demo_only = "--demo-only" in _sys.argv
    n_workers = max(1, min(os.cpu_count() or 4, 8))

    if not demo_only:
        final_items = generate_final_evaluation_set(200, FINAL_SEED)

        # Collect FINAL set results across all 3 nominal SNR regimes
        final_by_snr: Dict[float, List[Dict[str, Any]]] = {}
        rt_by_snr: Dict[float, Dict[str, float]] = {}

        for regime in regimes:
            snr_val = regime.noise_cfg.target_snr
            tasks_final = [(item, regime.noise_cfg) for item in final_items]
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                final_by_snr[snr_val] = list(ex.map(_final_set_worker, tasks_final, chunksize=6))
            rt_by_snr[snr_val] = {
                k: float(np.median([float(r[k]["runtime_ms"]) for r in final_by_snr[snr_val]]))
                for k, _ in method_specs
            }

        # =========================================================================
        # TASK 2d: Print CRB Percentiles on FINAL Set (Reference & True-Param CRB)
        # =========================================================================
        print("\n" + "=" * 126)
        print("TASK 2d: Cramér-Rao Lower Bound (CRB) Distribution on Held-Out FINAL Set (N = 200, FINAL_SEED = 500001)")
        print("=" * 126)
        print(
            f"{'Nom SNR':>7} | {'CRB Definition':<42} | {'P10(uT)':>8} | {'P50(uT)':>8} | "
            f"{'P90(uT)':>8} | {'P99(uT)':>8} | {'Mean(uT)':>8} | {'%>100uT':>8} | {'%>220uT':>8}"
        )
        print("-" * 126)
        for regime in regimes:
            snr_val = regime.noise_cfg.target_snr
            res_list = final_by_snr[snr_val]
            for key, desc in [
                ("crb_ref_ut", "Standardized Ref (HWHM=4.5 MHz, C=0.03)"),
                ("crb_true_ut", "Per-Spectrum True (HWHM_i in [3,9], C_i)"),
            ]:
                c_arr = np.array([float(r[key]) for r in res_list], dtype=np.float64)
                print(
                    f"{snr_val:>7.1f} | {desc:<42} | {np.percentile(c_arr, 10):>8.2f} | "
                    f"{np.percentile(c_arr, 50):>8.2f} | {np.percentile(c_arr, 90):>8.2f} | "
                    f"{np.percentile(c_arr, 99):>8.2f} | {np.mean(c_arr):>8.2f} | "
                    f"{np.mean(c_arr > 100.0)*100:>7.1f}% | {np.mean(c_arr > 220.0)*100:>7.1f}%"
                )
        print("-" * 126)

        # =========================================================================
        # TASK 7: Print Effective-SNR Distribution & Stage 2 Regime Fractions on FINAL
        # =========================================================================
        print("\n" + "=" * 126)
        print("TASK 7: Effective-SNR Distribution & Proposed Stage-2 Regimes on FINAL Set (N = 200, FINAL_SEED = 500001)")
        print("=" * 126)
        print(
            f"{'Nom SNR':>7} | {'Deepest Dip EffSNR (P10 / Med / P90)':<38} | "
            f"{'Outer Informative Dip EffSNR (P10 / Med / P90)':<46} | {'Easy(>3)':>8} | {'Med(1.5-3)':>10} | {'Hard(<1.5)':>10}"
        )
        print("-" * 126)
        for regime in regimes:
            snr_val = regime.noise_cfg.target_snr
            res_list = final_by_snr[snr_val]
            deep_arr = np.array([float(r["eff_snr_deepest"]) for r in res_list], dtype=np.float64)
            out_arr = np.array([float(r["eff_snr_outer"]) for r in res_list], dtype=np.float64)
            easy_pct = float(np.mean(out_arr > 3.0) * 100.0)
            med_pct = float(np.mean((out_arr >= 1.5) & (out_arr <= 3.0)) * 100.0)
            hard_pct = float(np.mean(out_arr < 1.5) * 100.0)
            deep_str = f"{np.percentile(deep_arr, 10):5.2f} / {np.median(deep_arr):5.2f} / {np.percentile(deep_arr, 90):5.2f}"
            out_str = f"{np.percentile(out_arr, 10):5.2f} / {np.median(out_arr):5.2f} / {np.percentile(out_arr, 90):5.2f}"
            print(
                f"{snr_val:>7.1f} | {deep_str:<38} | {out_str:<46} | "
                f"{easy_pct:>7.1f}% | {med_pct:>9.1f}% | {hard_pct:>9.1f}%"
            )
        print("-" * 126)

        # =========================================================================
        # TABLE 1: Held-Out FINAL Set Benchmark (FINAL_SEED = 500001, N = 200)
        # =========================================================================
        print_table_header(
            "TABLE 1: Held-Out FINAL Set Benchmark (Primary Representative Evaluation, N = 200 Spectra on S^2, FINAL_SEED = 500001)",
            "EffOut = Median Effective SNR of Outer Informative Dip | Fail>5xC uses per-spectrum CRB_true_i | RMS/CRBm = Full-Sample RMS / Median(CRB_true_i)",
        )
        for regime in regimes:
            snr_val = regime.noise_cfg.target_snr
            res_list = final_by_snr[snr_val]
            crbs_arr = np.array([float(r["crb_true_ut"]) for r in res_list], dtype=np.float64)
            eff_out_med = float(np.median([float(r["eff_snr_outer"]) for r in res_list]))

            for key, label in method_specs:
                errs = np.array(
                    [abs(float(r[key]["b_mag"]) - float(r["b_mag"])) * 1000.0 for r in res_list],
                    dtype=np.float64,
                )
                w_dists = np.array(
                    [float(np.linalg.norm(r[key]["b_vec"] - r["b_canon"])) for r in res_list],
                    dtype=np.float64,
                )
                stats = summarize_metrics(
                    errs,
                    crbs_arr,
                    w_dists,
                    rt_by_snr[snr_val][key],
                    is_naive=key.startswith("naive"),
                )
                print_table_row(snr_val, eff_out_med, label, stats)
            print("-" * 168)
        print(
            "Footnote (+): sig_core is the RMS of non-failed runs (|err_i| <= 5 * CRB_true_i) ONLY; it is "
            "conditional on success and biased low by construction, and is NEVER divided by CRB.\n"
            "Footnote (RMS/CRBm): Ratio of full-sample RMS (all 200 runs, no exclusions) to the dataset median "
            "per-spectrum white-noise CRB_true (7.58 uT at SNR=15, 28.44 uT at SNR=4, 56.88 uT at SNR=2)."
        )

    # =========================================================================
    # TABLE 2: Single Degenerate-Axis Demo Vector (TEST_SEED = 90001, 50 Trials)
    # =========================================================================
    params = ODMRParams(
        b_field_mt=(1.15, 0.75, 0.40),
        base_contrast=0.03,
        linewidth_mhz=4.5,
        voigt_eta=0.35,
        include_hyperfine=True,
        single_axis=False,
    )
    sim_out = simulate_clean_odmr(CANONICAL_FREQS, params)
    clean_spectrum = sim_out["spectrum"]
    b_true_mt = float(sim_out["b_magnitude_mt"])
    b_true_vec = np.asarray(params.b_field_mt, dtype=np.float64)
    b_true_canon = canonicalize_b_vec(b_true_vec)
    n_trials = 50

    print_table_header(
        "TABLE 2: Single Degenerate-Axis Demo Vector B = (1.15, 0.75, 0.40) mT (Single Degenerate-Axis Vector; Not Representative)",
        f"|B| = {b_true_mt:.4f} mT (B . n_2 = 0) | MC Trials = {n_trials}/regime | TEST_SEED = {TEST_SEED} | CRB = 7.32 / 27.44 / 54.88 uT",
    )

    panel_items: List[Dict[str, Any]] = []

    for regime in regimes:
        snr_val = regime.noise_cfg.target_snr
        eff_snr = regime.demo_eff_snr_outer
        crb_ut = compute_crb_b_mag_ut(CANONICAL_FREQS, b_true_vec, regime.noise_cfg)
        crb_vec = np.full(n_trials, crb_ut, dtype=np.float64)

        tasks_demo = [(t_i, clean_spectrum, regime.noise_cfg) for t_i in range(n_trials)]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            trial_results = list(ex.map(_demo_trial_worker, tasks_demo, chunksize=4))

        regime_summary: Dict[str, Dict[str, Any]] = {}
        for key, label in method_specs:
            errs = np.array(
                [abs(float(tr[key]["b_mag"]) - b_true_mt) * 1000.0 for tr in trial_results],
                dtype=np.float64,
            )
            w_dists = np.array(
                [float(np.linalg.norm(tr[key]["b_vec"] - b_true_canon)) for tr in trial_results],
                dtype=np.float64,
            )
            med_rt = float(np.median([float(tr[key]["runtime_ms"]) for tr in trial_results]))
            stats = summarize_metrics(
                errs,
                crb_vec,
                w_dists,
                med_rt,
                is_naive=key.startswith("naive"),
            )
            regime_summary[key] = stats
            print_table_row(snr_val, eff_snr, label, stats)
        print("-" * 168)

        plain_errs = np.array(
            [abs(float(tr["plain_deg2"]["b_mag"]) - b_true_mt) * 1000.0 for tr in trial_results],
            dtype=np.float64,
        )
        median_rank_idx = int(np.argsort(plain_errs)[len(plain_errs) // 2])
        rep_trial = trial_results[median_rank_idx]
        rep_plain = rep_trial["plain_deg2"]
        rep_inf = rep_trial["inf_deg3"]
        rep_ora = rep_trial["oracle_full"]
        rep_naive = rep_trial["naive_fixed"]
        true_noise_rms = float(rep_trial["total_noise_rms"])

        panel_items.append(
            {
                "regime": regime,
                "crb_ut": crb_ut,
                "rep_noisy": rep_trial["noisy"],
                "rep_plain": rep_plain,
                "rep_inf": rep_inf,
                "rep_ora": rep_ora,
                "rep_naive": rep_naive,
                "rep_plain_err_ut": float(plain_errs[median_rank_idx]),
                "rep_inf_err_ut": abs(float(rep_inf["b_mag"]) - b_true_mt) * 1000.0,
                "rep_ora_err_ut": abs(float(rep_ora["b_mag"]) - b_true_mt) * 1000.0,
                "rep_naive_err_ut": abs(float(rep_naive["b_mag"]) - b_true_mt) * 1000.0,
                "true_noise_rms": true_noise_rms,
                "summary": regime_summary,
            }
        )

    # Generate high-resolution 3-panel SNR comparison figure with dual SNR axes
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6), sharey=True)
    for ax, item in zip(axes, panel_items):
        regime = item["regime"]
        assert isinstance(regime, SNRRegime)
        s_plain = item["summary"]["plain_deg2"]
        s_ora = item["summary"]["oracle_full"]
        rep_plain = item["rep_plain"]
        rep_inf = item["rep_inf"]
        rep_ora = item["rep_ora"]
        rep_naive = item["rep_naive"]

        ax.plot(
            CANONICAL_FREQS,
            item["rep_noisy"],
            color="#b0b0b0",
            linewidth=0.80,
            alpha=0.65,
            label="Raw Noisy COTS",
        )
        ax.plot(
            CANONICAL_FREQS,
            clean_spectrum,
            color="#1f77b4",
            linewidth=1.8,
            label="Ground Truth (4-Axis Quantum)",
        )
        ax.plot(
            CANONICAL_FREQS,
            rep_naive["fitted_curve"],
            color="#9467bd",
            linestyle=":",
            linewidth=1.25,
            alpha=0.85,
            label="Naive 4-Pair (Model-Mismatched)",
        )
        ax.plot(
            CANONICAL_FREQS,
            rep_plain["fitted_curve"],
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label="Physical (Plain, deg-2)",
        )
        ax.plot(
            CANONICAL_FREQS,
            rep_inf["fitted_curve"],
            color="#2ca02c",
            linestyle="-.",
            linewidth=1.35,
            label="Physical (Informed: Inv. Crime UB)",
        )
        ax.plot(
            CANONICAL_FREQS,
            rep_ora["fitted_curve"],
            color="#17becf",
            linestyle="-",
            linewidth=1.15,
            alpha=0.85,
            label="Physical (Oracle: True Drift+Hum)",
        )
        ax.plot(
            CANONICAL_FREQS,
            rep_plain["fitted_baseline"],
            color="#ff7f0e",
            linestyle=":",
            linewidth=1.3,
            label="Fitted Baseline (Plain, deg-2)",
        )

        ax.set_title(
            f"{regime.label} | Nominal SNR = {regime.noise_cfg.target_snr:.0f} "
            f"(Eff. Outer-Dip SNR = {regime.demo_eff_snr_outer:.2f}, CRB = {item['crb_ut']:.1f} uT)\n"
            f"Plain(deg-2) RMS={s_plain['rms_all_ut']:.1f}uT, Med={s_plain['med_ut']:.1f}uT | "
            f"Oracle RMS={s_ora['rms_all_ut']:.1f}uT, Med={s_ora['med_ut']:.1f}uT",
            fontsize=9.0,
            fontweight="bold",
        )
        annot_text = (
            f"Median-Error Realization (Plain deg-2):\n"
            f"Naive (Fixed, Mismatched): |dB|={item['rep_naive_err_ut']:.1f} uT, RMS={rep_naive['residual_rms']:.5f}\n"
            f"Physical (Plain, deg-2)  : |dB|={item['rep_plain_err_ut']:.1f} uT, RMS={rep_plain['residual_rms']:.5f}\n"
            f"Physical (Informed UB)   : |dB|={item['rep_inf_err_ut']:.1f} uT, RMS={rep_inf['residual_rms']:.5f}\n"
            f"Physical (Oracle Dr+Hum) : |dB|={item['rep_ora_err_ut']:.1f} uT, RMS={rep_ora['residual_rms']:.5f}\n"
            f"True Noise RMS={item['true_noise_rms']:.5f} | "
            f"WrongBasin(Pl/Or): {s_plain['wrong_basin_pct']:.0f}%/{s_ora['wrong_basin_pct']:.0f}%"
        )
        ax.text(
            0.025,
            0.035,
            annot_text,
            transform=ax.transAxes,
            fontsize=7.5,
            verticalalignment="bottom",
            bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "alpha": 0.90, "edgecolor": "#cccccc"},
        )
        ax.set_xlabel("Microwave Frequency (MHz)", fontsize=10)
        ax.set_xlim(CANONICAL_FREQS[0], CANONICAL_FREQS[-1])
        ax.grid(True, linestyle=":", alpha=0.45)

    axes[0].set_ylabel("Normalized Fluorescence", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=7.3, framealpha=0.92)

    fig.suptitle(
        f"OpenNV-Denoise Stage 1.5 Physical Baseline — Single Degenerate-Axis Demo Vector "
        f"B_true = {params.b_field_mt} mT (|B| = {b_true_mt:.4f} mT, Not Representative)",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    output_path = Path("assets") / "week1_snr_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SNR comparison figure (median-error realizations) to {output_path.as_posix()}")


if __name__ == "__main__":
    main()
