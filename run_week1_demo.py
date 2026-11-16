"""Stage 1.5 benchmark script: Fair Variable-Projection Spin-1 Hamiltonian Baseline Report.

Evaluates four baseline variants across COTS SNR regimes (SNR = 15, 4, 2) using
TEST_SEED = 90001 (strictly disjoint from DEV_SEED = 10001 used in Part 0 diagnostics):
  1. (naive Lorentzian)                - Uncoupled 8-free-Lorentzian fitter
  2. (physical, plain)                 - Variable-projection Spin-1 Hamiltonian + wing-anchored baseline
  3. (physical, informed with mains)   - Variable-projection Spin-1 Hamiltonian + 50/60 Hz mains hum basis
  4. (physical, oracle baseline)       - Variable-projection Spin-1 Hamiltonian with true (1.0 + drift) baseline

Regenerates assets/week1_snr_comparison.png using the exact median-error realization
for each SNR regime and annotates every panel with numeric residual RMS and CRB ratio.
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
from src.noise import HardwareNoiseConfig, generate_pink_noise
from src.physics import ODMRParams, simulate_clean_odmr

DEV_SEED: int = 10001
TEST_SEED: int = 90001


@dataclass(frozen=True)
class SNRRegime:
    """Configuration for a single COTS SNR benchmark regime."""

    label: str
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
    """Compute Cramer-Rao lower bound (in uT) for |B| under white Gaussian shot-modulated noise."""
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


def generate_noisy_and_true_drift(
    clean: np.ndarray, cfg: HardwareNoiseConfig, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate noisy ODMR spectrum identically to apply_realistic_noise, returning true drift."""
    n_points = clean.shape[0]
    time_sec = np.arange(n_points, dtype=np.float64) * (cfg.dwell_time_ms * 1e-3)
    u = np.linspace(-1.0, 1.0, n_points, dtype=np.float64)
    raw_poly = (
        rng.normal(0.0, 0.5) * u
        + rng.normal(0.0, 0.5) * (u * u - 1.0 / 3.0)
        + rng.normal(0.0, 0.3) * (u * u * u)
    )
    max_poly = float(np.max(np.abs(raw_poly)))
    if cfg.baseline_drift_amp > 0.0 and max_poly > 1e-12:
        drift = cfg.baseline_drift_amp * (raw_poly / max_poly) * rng.uniform(0.6, 1.0)
    else:
        drift = np.zeros(n_points, dtype=np.float64)

    phase = rng.uniform(0.0, 2.0 * np.pi)
    hum = cfg.hum_amp * np.sin(2.0 * np.pi * cfg.mains_freq_hz * time_sec + phase)

    signal_depth = max(float(np.max(clean) - np.min(clean)), 1e-6)
    noise_std = signal_depth / float(cfg.target_snr)
    shot_factor = np.sqrt(np.clip(clean, 0.5, 1.5))
    white = rng.standard_normal(n_points)
    pink_ratio = float(np.clip(cfg.pink_noise_ratio, 0.0, 1.0))
    if pink_ratio > 0.0:
        pink = generate_pink_noise(n_points, rng)
        combined = np.sqrt(1.0 - pink_ratio) * white + np.sqrt(pink_ratio) * pink
    else:
        combined = white
    stochastic = noise_std * shot_factor * combined
    return clean + drift + hum + stochastic, drift


def _eval_trial_worker(args: Tuple[int, np.ndarray, HardwareNoiseConfig]) -> Dict[str, Any]:
    """Run all 4 baseline methods on a single Monte Carlo realization."""
    trial_idx, clean_spectrum, noise_cfg = args
    seed = TEST_SEED + int(noise_cfg.target_snr * 1000) + trial_idx * 17
    rng = np.random.default_rng(seed)
    noisy, drift = generate_noisy_and_true_drift(clean_spectrum, noise_cfg, rng)
    true_baseline = 1.0 + drift

    r_naive = fit_lorentzian_naive(CANONICAL_FREQS, noisy)
    r_plain = fit_physical_odmr(CANONICAL_FREQS, noisy, variant="plain")
    r_inf = fit_physical_odmr(
        CANONICAL_FREQS,
        noisy,
        dwell_time=noise_cfg.dwell_time_ms * 1e-3,
        variant="informed",
    )
    r_oracle = fit_physical_odmr(
        CANONICAL_FREQS,
        noisy,
        oracle_baseline=true_baseline,
        variant="oracle_baseline",
    )
    return {
        "trial_idx": trial_idx,
        "noisy": noisy,
        "true_baseline": true_baseline,
        "naive": r_naive,
        "plain": r_plain,
        "informed": r_inf,
        "oracle": r_oracle,
    }


def main() -> None:
    """Run Stage 1.5 benchmark across SNR=15, 4, 2 and save representative comparison figure."""
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

    regimes: List[SNRRegime] = [
        SNRRegime(
            label="High SNR",
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

    n_trials = 50
    method_specs = [
        ("naive", "(naive Lorentzian)"),
        ("plain", "(physical, plain)"),
        ("informed", "(physical, informed with mains terms)"),
        ("oracle", "(physical, oracle baseline)"),
    ]

    print("=" * 108)
    print("OpenNV-Denoise Stage 1.5: Variable-Projection Spin-1 Hamiltonian Baseline Benchmark")
    print(
        f"4-Axis Ensemble | B_true = {params.b_field_mt} mT (|B| = {b_true_mt:.4f} mT) | "
        f"N = {len(CANONICAL_FREQS)} pts | MC Trials = {n_trials}/regime | TEST_SEED = {TEST_SEED}"
    )
    print("=" * 108)
    print(
        f"{'SNR':>5} | {'Method':<37} | {'Median(uT)':>10} | {'P90(uT)':>9} | "
        f"{'MAE(uT)':>9} | {'Fail>100uT':>10} | {'Time(ms)':>8} | {'Med/CRB':>7}"
    )
    print("-" * 108)

    panel_items: List[Dict[str, Any]] = []
    n_workers = max(1, min(os.cpu_count() or 4, 8))

    for regime in regimes:
        snr_val = regime.noise_cfg.target_snr
        crb_ut = compute_crb_b_mag_ut(CANONICAL_FREQS, b_true_vec, regime.noise_cfg)

        # Measure uncontended single-process wall-clock runtime on 10 trials
        rt_single: Dict[str, List[float]] = {k: [] for k, _ in method_specs}
        for t_i in range(10):
            res_1 = _eval_trial_worker((t_i, clean_spectrum, regime.noise_cfg))
            for key, _ in method_specs:
                rt_single[key].append(float(res_1[key]["runtime_ms"]))

        tasks = [(t_i, clean_spectrum, regime.noise_cfg) for t_i in range(n_trials)]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            trial_results = list(ex.map(_eval_trial_worker, tasks, chunksize=4))

        regime_summary: Dict[str, Dict[str, float]] = {}
        for key, label in method_specs:
            errs = np.array(
                [abs(float(tr[key]["b_mag"]) - b_true_mt) * 1000.0 for tr in trial_results],
                dtype=np.float64,
            )
            med_ut = float(np.median(errs))
            p90_ut = float(np.percentile(errs, 90))
            mae_ut = float(np.mean(errs))
            fail_pct = float(np.mean(errs > 100.0) * 100.0)
            mean_rt = float(np.mean(rt_single[key]))
            crb_ratio = med_ut / max(crb_ut, 1e-9)
            regime_summary[key] = {
                "med_ut": med_ut,
                "p90_ut": p90_ut,
                "mae_ut": mae_ut,
                "fail_pct": fail_pct,
                "mean_rt": mean_rt,
                "crb_ratio": crb_ratio,
                "crb_ut": crb_ut,
            }
            print(
                f"{snr_val:>5.1f} | {label:<37} | {med_ut:>10.2f} | {p90_ut:>9.2f} | "
                f"{mae_ut:>9.2f} | {fail_pct:>9.1f}% | {mean_rt:>8.1f} | {crb_ratio:>6.2f}x"
            )
        print("-" * 108)

        # Select the exact median-error realization of (physical, plain) for non-cherry-picked plotting
        plain_errs = np.array(
            [abs(float(tr["plain"]["b_mag"]) - b_true_mt) * 1000.0 for tr in trial_results],
            dtype=np.float64,
        )
        median_rank_idx = int(np.argsort(plain_errs)[len(plain_errs) // 2])
        rep_trial = trial_results[median_rank_idx]
        rep_plain = rep_trial["plain"]
        rep_inf = rep_trial["informed"]
        true_noise_rms = float(np.sqrt(np.mean((rep_trial["noisy"] - clean_spectrum) ** 2)))

        panel_items.append(
            {
                "regime": regime,
                "crb_ut": crb_ut,
                "rep_noisy": rep_trial["noisy"],
                "rep_plain": rep_plain,
                "rep_inf": rep_inf,
                "rep_plain_err_ut": float(plain_errs[median_rank_idx]),
                "rep_inf_err_ut": abs(float(rep_inf["b_mag"]) - b_true_mt) * 1000.0,
                "true_noise_rms": true_noise_rms,
                "summary": regime_summary,
            }
        )

    # Generate high-resolution 3-panel SNR comparison figure using median-error realizations
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), sharey=True)
    for ax, item in zip(axes, panel_items):
        regime = item["regime"]
        assert isinstance(regime, SNRRegime)
        s_plain = item["summary"]["plain"]
        s_inf = item["summary"]["informed"]
        rep_plain = item["rep_plain"]
        rep_inf = item["rep_inf"]

        ax.plot(
            CANONICAL_FREQS,
            item["rep_noisy"],
            color="#9e9e9e",
            linewidth=0.85,
            alpha=0.70,
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
            rep_plain["fitted_curve"],
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label="Physical Fit (Plain)",
        )
        ax.plot(
            CANONICAL_FREQS,
            rep_inf["fitted_curve"],
            color="#2ca02c",
            linestyle="-.",
            linewidth=1.4,
            label="Physical Fit (Informed Mains)",
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
            f"{regime.label} (SNR = {regime.noise_cfg.target_snr:.0f}, CRB = {item['crb_ut']:.1f} uT)\n"
            f"Plain Med = {s_plain['med_ut']:.1f} uT ({s_plain['crb_ratio']:.2f}x CRB) | "
            f"Informed Med = {s_inf['med_ut']:.1f} uT ({s_inf['crb_ratio']:.2f}x CRB)",
            fontsize=10.0,
            fontweight="bold",
        )
        # Annotate panel with numeric residual RMS and median-realization error
        annot_text = (
            f"Median-Error Realization:\n"
            f"Plain |dB| = {item['rep_plain_err_ut']:.1f} uT, "
            f"Resid RMS = {rep_plain['residual_rms']:.5f}\n"
            f"Informed |dB| = {item['rep_inf_err_ut']:.1f} uT, "
            f"Resid RMS = {rep_inf['residual_rms']:.5f}\n"
            f"True Noise RMS = {item['true_noise_rms']:.5f} | "
            f"Fail(>100uT): {s_plain['fail_pct']:.1f}% / {s_inf['fail_pct']:.1f}%"
        )
        ax.text(
            0.03,
            0.04,
            annot_text,
            transform=ax.transAxes,
            fontsize=8.2,
            verticalalignment="bottom",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
        )
        ax.set_xlabel("Microwave Frequency (MHz)", fontsize=10)
        ax.set_xlim(CANONICAL_FREQS[0], CANONICAL_FREQS[-1])
        ax.grid(True, linestyle=":", alpha=0.45)

    axes[0].set_ylabel("Normalized Fluorescence", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=8.0, framealpha=0.92)

    fig.suptitle(
        f"OpenNV-Denoise Stage 1.5 Variable-Projection Physical Baseline "
        f"(B_true = {params.b_field_mt} mT, |B| = {b_true_mt:.4f} mT, Median-Error Realizations)",
        fontsize=12.0,
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
