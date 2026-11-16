"""Stage 1.5 Final Benchmark: Fair Variable-Projection Spin-1 Hamiltonian Baseline Report.

Strict Three-Split Seed Policy:
  1. DEV_SEED   = 10001  - Used strictly for Part 0 diagnostics and hyperparameter selection.
  2. TEST_SEED  = 90001  - Used strictly for the single degenerate-axis demo vector
                           B_true = (1.15, 0.75, 0.40) mT (50 MC trials/regime; not representative).
  3. FINAL_SEED = 500001 - Held-out final evaluation set (200 spectra on S^2 including near-degenerate
                           directions, |B| in [0.1, 3.2] mT, never used for any tuning decision).

Evaluated Rows:
  - (naive, 4-symmetric-pair)                     : Uncoupled 4-pair symmetric Lorentzian (no hyperfine/Hamiltonian)
  - (physical, plain)                             : Variable-projection Spin-1 Hamiltonian + deg-3 wing-anchored baseline
  - (physical, informed: inverse crime, upper bd) : Adds 50/60 Hz mains sin/cos basis (uses simulator hum model)
  - (physical, oracle: known drift only)          : Fixed baseline = 1.0 + true drift (mains hum unmodelled)
  - (physical, oracle: true drift + hum)          : Fixed baseline = 1.0 + true drift + true hum (true baseline oracle)

Failure Criteria & Error Statistics (Task 4):
  - Fail>5xCRB : Relative failure rate (|err| > 5 * CRB_per_spectrum).
  - WrongBasin : Canonical-wedge 3D distance ||B_fit_canon - B_true_canon||_2 > 0.15 mT (150 uT).
  - Fail>100uT : Legacy absolute failure rate (|err| > 100 uT) kept for continuity.
  - sigma_core : RMS of |err| across non-failed runs (|err| <= 5 * CRB).
  - sig_core/CRB: Ratio sigma_core / CRB (annotated with '*' when Fail>5xCRB > 30%, where threshold
                 breakdown invalidates the local Cramér-Rao bound).
  Note: CRB assumes white intensity-scaled Gaussian noise, fixed HWHM = 4.5 MHz, and contrast = 0.03.
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
from src.config import CANONICAL_FREQS, D0_MHZ
from src.noise import HardwareNoiseConfig, apply_realistic_noise
from src.physics import ODMRParams, canonicalize_b_vec, simulate_clean_odmr

DEV_SEED: int = 10001
TEST_SEED: int = 90001
FINAL_SEED: int = 500001
WRONG_BASIN_THRESH_MT: float = 0.15


@dataclass(frozen=True)
class SNRRegime:
    """Configuration for a single COTS SNR benchmark regime."""

    label: str
    eff_snr_outer: float
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
    """Compute white-noise Cramér-Rao lower bound (in uT) for |B| at fixed reference HWHM/contrast."""
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


def _run_all_methods_on_spectrum(
    clean_spectrum: np.ndarray,
    noise_cfg: HardwareNoiseConfig,
    seed: int,
) -> Dict[str, Any]:
    """Generate noisy spectrum via apply_realistic_noise(return_components=True) and run all 5 fitters."""
    rng = np.random.default_rng(seed)
    noisy_out = apply_realistic_noise(clean_spectrum, noise_cfg, rng=rng, return_components=True)
    assert isinstance(noisy_out, tuple)
    noisy, comps = noisy_out

    true_bl_drift_only = 1.0 + comps["drift"]
    true_bl_full = 1.0 + comps["drift"] + comps["hum"]

    r_naive = fit_lorentzian_naive(CANONICAL_FREQS, noisy)
    r_plain = fit_physical_odmr(CANONICAL_FREQS, noisy, variant="plain")
    r_inf = fit_physical_odmr(
        CANONICAL_FREQS,
        noisy,
        dwell_time=noise_cfg.dwell_time_ms * 1e-3,
        variant="informed",
    )
    r_oracle_drift = fit_physical_odmr(
        CANONICAL_FREQS,
        noisy,
        oracle_baseline=true_bl_drift_only,
        variant="oracle_drift",
    )
    r_oracle_full = fit_physical_odmr(
        CANONICAL_FREQS,
        noisy,
        oracle_baseline=true_bl_full,
        variant="oracle_baseline",
    )
    return {
        "noisy": noisy,
        "true_bl_drift_only": true_bl_drift_only,
        "true_bl_full": true_bl_full,
        "naive": r_naive,
        "plain": r_plain,
        "informed": r_inf,
        "oracle_drift": r_oracle_drift,
        "oracle_full": r_oracle_full,
    }


def _demo_trial_worker(args: Tuple[int, np.ndarray, HardwareNoiseConfig]) -> Dict[str, Any]:
    """Worker for the single-vector demo benchmark (TEST_SEED = 90001)."""
    trial_idx, clean_spectrum, noise_cfg = args
    seed = TEST_SEED + int(noise_cfg.target_snr * 1000) + trial_idx * 17
    out = _run_all_methods_on_spectrum(clean_spectrum, noise_cfg, seed)
    out["trial_idx"] = trial_idx
    return out


def _final_set_worker(args: Tuple[Dict[str, Any], HardwareNoiseConfig]) -> Dict[str, Any]:
    """Worker for the 200-spectrum held-out FINAL set (FINAL_SEED = 500001)."""
    item, noise_cfg = args
    idx = int(item["idx"])
    seed = FINAL_SEED + int(noise_cfg.target_snr * 1000) + idx * 29
    out = _run_all_methods_on_spectrum(item["clean"], noise_cfg, seed)
    # Compute per-spectrum white-noise CRB (standardized at HWHM=4.5 MHz, contrast=0.03 per Task 4)
    crb_ut = compute_crb_b_mag_ut(CANONICAL_FREQS, item["b_vec"], noise_cfg)
    out["idx"] = idx
    out["b_mag"] = item["b_mag"]
    out["b_canon"] = item["b_canon"]
    out["crb_ut"] = crb_ut
    return out


def generate_final_evaluation_set(n_spectra: int = 200, seed: int = FINAL_SEED) -> List[Dict[str, Any]]:
    """Generate the 200-spectrum held-out FINAL set on S^2 (including 20% near-degenerate directions)."""
    rng = np.random.default_rng(seed)
    items: List[Dict[str, Any]] = []
    for i in range(n_spectra):
        if i % 10 == 0:
            # Near B . n_2 = 0 perpendicular-axis plane (Bx + By ~ Bz)
            t = float(rng.uniform(0.10, 0.48))
            u = np.array([t, 1.0 - t + float(rng.normal(0.0, 0.02)), 1.0], dtype=np.float64)
            u = rng.choice([-1.0, 1.0], size=3) * rng.permutation(u)
        elif i % 10 == 1:
            # Near symmetry boundary plane Bx ~ By
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
    """Compute Task 4 error statistics and failure rates."""
    med_ut = float(np.median(errs_ut))
    p90_ut = float(np.percentile(errs_ut, 90))
    mae_ut = float(np.mean(errs_ut))

    rel_fail_mask = errs_ut > (5.0 * crbs_ut)
    fail_5x_pct = float(np.mean(rel_fail_mask) * 100.0)
    fail_100_pct = float(np.mean(errs_ut > 100.0) * 100.0)
    wrong_basin_pct = float(np.mean(wedge_dists_mt > WRONG_BASIN_THRESH_MT) * 100.0)

    core_mask = ~rel_fail_mask
    if np.any(core_mask):
        sigma_core_ut = float(np.sqrt(np.mean(errs_ut[core_mask] ** 2)))
        core_crb_ref = float(np.median(crbs_ut[core_mask]))
    else:
        sigma_core_ut = float(np.sqrt(np.mean(errs_ut ** 2)))
        core_crb_ref = float(np.median(crbs_ut))

    med_crb_ut = float(np.median(crbs_ut))
    sig_over_crb = sigma_core_ut / max(core_crb_ref, 1e-9)
    in_threshold_regime = fail_5x_pct > 30.0

    return {
        "med_ut": med_ut,
        "p90_ut": p90_ut,
        "mae_ut": mae_ut,
        "sigma_core_ut": sigma_core_ut,
        "fail_5x_pct": fail_5x_pct,
        "wrong_basin_pct": wrong_basin_pct,
        "fail_100_pct": fail_100_pct,
        "mean_rt_ms": mean_rt_ms,
        "sig_over_crb": sig_over_crb,
        "med_crb_ut": med_crb_ut,
        "in_threshold_regime": in_threshold_regime,
        "is_naive": is_naive,
    }


def print_table_header(title: str, subtitle: str) -> None:
    print("\n" + "=" * 142)
    print(title)
    print(subtitle)
    print("=" * 142)
    print(
        f"{'SNR':>4} | {'EffSNR':>6} | {'Method':<43} | {'Med(uT)':>8} | {'P90(uT)':>8} | "
        f"{'MAE(uT)':>8} | {'sig_core':>8} | {'Fail>5xCRB':>10} | {'WrongBasin':>10} | "
        f"{'Fail>100uT':>10} | {'Time(ms)':>8} | {'sig/CRB':>8}"
    )
    print("-" * 142)


def print_table_row(snr: float, eff_snr: float, label: str, s: Dict[str, Any]) -> None:
    wb_str = f"{s['wrong_basin_pct']:>9.1f}%" if not s["is_naive"] else f"{'N/A(1D)':>10}"
    flag = "*" if s["in_threshold_regime"] else " "
    ratio_str = f"{s['sig_over_crb']:>6.2f}x{flag}"
    print(
        f"{snr:>4.0f} | {eff_snr:>6.2f} | {label:<43} | {s['med_ut']:>8.2f} | {s['p90_ut']:>8.2f} | "
        f"{s['mae_ut']:>8.2f} | {s['sigma_core_ut']:>8.2f} | {s['fail_5x_pct']:>9.1f}% | "
        f"{wb_str} | {s['fail_100_pct']:>9.1f}% | {s['mean_rt_ms']:>8.1f} | {ratio_str:>8}"
    )


def main() -> None:
    """Run Stage 1.5 benchmark on Held-Out FINAL-200 Set and Single-Vector Demo Set."""
    regimes: List[SNRRegime] = [
        SNRRegime(
            label="High SNR",
            eff_snr_outer=4.07,
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
            eff_snr_outer=1.51,
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
            eff_snr_outer=0.93,
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
        ("naive", "(naive, 4-symmetric-pair Lorentzian)"),
        ("plain", "(physical, plain)"),
        ("informed", "(physical, informed: inverse crime, UB)"),
        ("oracle_drift", "(physical, oracle: known drift only)"),
        ("oracle_full", "(physical, oracle: true drift + hum)"),
    ]

    n_workers = max(1, min(os.cpu_count() or 4, 8))

    # =========================================================================
    # PART A: HELD-OUT FINAL SET (FINAL_SEED = 500001, N = 200 Spectra on S^2)
    # =========================================================================
    final_items = generate_final_evaluation_set(200, FINAL_SEED)
    print_table_header(
        "TABLE 1: OpenNV-Denoise Stage 1.5 Held-Out FINAL Set Benchmark (Primary Representative Evaluation)",
        f"N = 200 Spectra on S^2 (incl. 20% near-degenerate) | |B| in [0.1, 3.2] mT | FINAL_SEED = {FINAL_SEED} (Untuned)",
    )

    for regime in regimes:
        snr_val = regime.noise_cfg.target_snr
        eff_snr = regime.eff_snr_outer

        # Uncontended single-process wall-clock runtime sample (10 spectra)
        rt_single: Dict[str, List[float]] = {k: [] for k, _ in method_specs}
        for s_i in range(10):
            res_1 = _final_set_worker((final_items[s_i], regime.noise_cfg))
            for key, _ in method_specs:
                rt_single[key].append(float(res_1[key]["runtime_ms"]))

        tasks_final = [(item, regime.noise_cfg) for item in final_items]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            final_results = list(ex.map(_final_set_worker, tasks_final, chunksize=6))

        crbs_arr = np.array([float(r["crb_ut"]) for r in final_results], dtype=np.float64)
        for key, label in method_specs:
            errs = np.array(
                [abs(float(r[key]["b_mag"]) - float(r["b_mag"])) * 1000.0 for r in final_results],
                dtype=np.float64,
            )
            w_dists = np.array(
                [float(np.linalg.norm(r[key]["b_vec"] - r["b_canon"])) for r in final_results],
                dtype=np.float64,
            )
            stats = summarize_metrics(
                errs, crbs_arr, w_dists, float(np.mean(rt_single[key])), is_naive=(key == "naive")
            )
            print_table_row(snr_val, eff_snr, label, stats)
        print("-" * 142)

    # =========================================================================
    # PART B: SINGLE DEGENERATE-AXIS DEMO VECTOR (TEST_SEED = 90001, 50 Trials)
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
        "TABLE 2: Single Degenerate-Axis Demo Vector (Single Degenerate-Axis Vector; Not Representative)",
        f"B_true = {params.b_field_mt} mT (|B| = {b_true_mt:.4f} mT, B . n_2 = 0) | MC Trials = {n_trials}/regime | TEST_SEED = {TEST_SEED}",
    )

    panel_items: List[Dict[str, Any]] = []

    for regime in regimes:
        snr_val = regime.noise_cfg.target_snr
        eff_snr = regime.eff_snr_outer
        crb_ut = compute_crb_b_mag_ut(CANONICAL_FREQS, b_true_vec, regime.noise_cfg)
        crb_vec = np.full(n_trials, crb_ut, dtype=np.float64)

        rt_single_demo: Dict[str, List[float]] = {k: [] for k, _ in method_specs}
        for t_i in range(10):
            res_1 = _demo_trial_worker((t_i, clean_spectrum, regime.noise_cfg))
            for key, _ in method_specs:
                rt_single_demo[key].append(float(res_1[key]["runtime_ms"]))

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
            stats = summarize_metrics(
                errs, crb_vec, w_dists, float(np.mean(rt_single_demo[key])), is_naive=(key == "naive")
            )
            regime_summary[key] = stats
            print_table_row(snr_val, eff_snr, label, stats)
        print("-" * 142)

        plain_errs = np.array(
            [abs(float(tr["plain"]["b_mag"]) - b_true_mt) * 1000.0 for tr in trial_results],
            dtype=np.float64,
        )
        median_rank_idx = int(np.argsort(plain_errs)[len(plain_errs) // 2])
        rep_trial = trial_results[median_rank_idx]
        rep_plain = rep_trial["plain"]
        rep_inf = rep_trial["informed"]
        rep_ora = rep_trial["oracle_full"]
        rep_naive = rep_trial["naive"]
        true_noise_rms = float(np.sqrt(np.mean((rep_trial["noisy"] - clean_spectrum) ** 2)))

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

    print(
        "Note: '*' indicates threshold breakdown regime (Fail>5xCRB > 30%) where local white-noise CRB "
        "(computed at fixed HWHM=4.5 MHz, contrast=0.03) is not a valid lower bound."
    )

    # Generate high-resolution 3-panel SNR comparison figure using median-error realizations
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.5), sharey=True)
    for ax, item in zip(axes, panel_items):
        regime = item["regime"]
        assert isinstance(regime, SNRRegime)
        s_plain = item["summary"]["plain"]
        s_inf = item["summary"]["informed"]
        s_ora = item["summary"]["oracle_full"]
        s_naive = item["summary"]["naive"]
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
            linewidth=1.2,
            alpha=0.85,
            label="Naive 4-Pair Lorentzian",
        )
        ax.plot(
            CANONICAL_FREQS,
            rep_plain["fitted_curve"],
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label="Physical (Plain)",
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
            label="Fitted Baseline (Plain)",
        )

        ax.set_title(
            f"{regime.label} (Nom SNR={regime.noise_cfg.target_snr:.0f}, "
            f"Eff Outer SNR={regime.eff_snr_outer:.2f}, CRB={item['crb_ut']:.1f} uT)\n"
            f"Plain Med={s_plain['med_ut']:.1f}uT (sig/CRB={s_plain['sig_over_crb']:.2f}x) | "
            f"Oracle Med={s_ora['med_ut']:.1f}uT ({s_ora['sig_over_crb']:.2f}x)",
            fontsize=9.3,
            fontweight="bold",
        )
        annot_text = (
            f"Median-Error Realization (Plain):\n"
            f"Naive 4-Pair : |dB|={item['rep_naive_err_ut']:.0f} uT, RMS={rep_naive['residual_rms']:.5f}\n"
            f"Phys (Plain) : |dB|={item['rep_plain_err_ut']:.1f} uT, RMS={rep_plain['residual_rms']:.5f}\n"
            f"Phys (Inf UB): |dB|={item['rep_inf_err_ut']:.1f} uT, RMS={rep_inf['residual_rms']:.5f}\n"
            f"Phys (Oracle): |dB|={item['rep_ora_err_ut']:.1f} uT, RMS={rep_ora['residual_rms']:.5f}\n"
            f"True Noise RMS={item['true_noise_rms']:.5f} | "
            f"Fail>5xCRB(Pl/Or): {s_plain['fail_5x_pct']:.0f}%/{s_ora['fail_5x_pct']:.0f}%"
        )
        ax.text(
            0.025,
            0.035,
            annot_text,
            transform=ax.transAxes,
            fontsize=7.6,
            verticalalignment="bottom",
            bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "alpha": 0.90, "edgecolor": "#cccccc"},
        )
        ax.set_xlabel("Microwave Frequency (MHz)", fontsize=10)
        ax.set_xlim(CANONICAL_FREQS[0], CANONICAL_FREQS[-1])
        ax.grid(True, linestyle=":", alpha=0.45)

    axes[0].set_ylabel("Normalized Fluorescence", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=7.4, framealpha=0.92)

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
