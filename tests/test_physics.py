"""Comprehensive pytest suite for Stage 1 & 1.5 OpenNV-Denoise physics, noise, and baseline.

Seed Separation Policy:
- DEV_SEED = 10001 is reserved strictly for development diagnostics (Part 0) and algorithm tuning.
- TEST_SEED = 90001 (and deterministic per-test offsets) is reserved strictly for unit tests and
  final benchmark evaluation. No hyperparameters or thresholds were tuned on TEST_SEED.
"""

from __future__ import annotations

from itertools import permutations, product
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baseline import fit_lorentzian_naive, fit_physical_odmr
from src.config import (
    A_HYPERFINE_MHZ,
    B_MAG_MAX_MT,
    CANONICAL_FREQS,
    D0_MHZ,
    GAMMA_E_MHZ_MT,
    NV_AXES,
)
from src.noise import (
    HardwareNoiseConfig,
    apply_realistic_noise,
    generate_pink_noise,
    resample_spectrum_pchip,
)
from src.physics import (
    ODMRParams,
    b_mag_from_splittings,
    canonicalize_b_vec,
    pseudo_voigt_dip,
    simulate_batch_clean_odmr,
    simulate_clean_odmr,
    solve_axis_transitions,
)

DEV_SEED: int = 10001
TEST_SEED: int = 90001


def _compute_white_gaussian_crb_ut(
    freqs: np.ndarray,
    b_vec: np.ndarray,
    target_snr: float = 15.0,
    d0_mhz: float = D0_MHZ,
    gamma_mhz: float = 4.5,
    contrast: float = 0.03,
    eta: float = 0.35,
) -> float:
    """Compute Cramer-Rao lower bound (in uT) for |B| under white shot-modulated Gaussian noise."""
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
    noise_std = signal_depth / float(target_snr)
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


def test_zero_field_convergence() -> None:
    """At B=(0, 0, 0), all transitions must converge to D0_MHZ with 0 splitting."""
    f_minus, f_plus, contrast_factor = solve_axis_transitions(0.0, 0.0, d0=D0_MHZ)
    assert abs(f_minus - D0_MHZ) < 1e-6
    assert abs(f_plus - D0_MHZ) < 1e-6
    assert abs(contrast_factor - 1.0) < 1e-12

    params = ODMRParams(b_field_mt=(0.0, 0.0, 0.0))
    res = simulate_clean_odmr(CANONICAL_FREQS, params)
    assert np.allclose(res["splittings_mhz"], 0.0, atol=1e-6)
    assert np.allclose(res["dip_pairs"], D0_MHZ, atol=1e-6)
    assert abs(res["b_magnitude_mt"]) < 1e-12
    assert abs(res["b_From_splittings_mt"]) < 1e-6


def test_tetrahedral_projection_invariant_and_symmetry() -> None:
    """Verify tetrahedral projection sum invariant, B/-B symmetry, and |B| recovery."""
    rng = np.random.default_rng(TEST_SEED + 1)
    directions = rng.normal(size=(50, 3))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    magnitudes = rng.uniform(0.1, 3.0, size=50)
    b_vecs = directions * magnitudes[:, None]

    for b_vec, b_mag in zip(b_vecs, magnitudes):
        b_par = NV_AXES @ b_vec
        proj_sum_sq = float(np.sum(b_par * b_par))
        assert abs(proj_sum_sq - (4.0 / 3.0) * (b_mag ** 2)) < 1e-10

        res_pos = simulate_clean_odmr(
            CANONICAL_FREQS, ODMRParams(b_field_mt=tuple(b_vec.tolist()))
        )
        res_neg = simulate_clean_odmr(
            CANONICAL_FREQS, ODMRParams(b_field_mt=tuple((-b_vec).tolist()))
        )
        assert np.allclose(res_pos["spectrum"], res_neg["spectrum"], atol=1e-12)
        assert np.allclose(res_pos["splittings_mhz"], res_neg["splittings_mhz"], atol=1e-12)
        assert abs(res_pos["b_From_splittings_mt"] - b_mag) < 5e-3

    # Verify linear Zeeman accuracy at low transverse / weak-field regime (<= 0.35 mT)
    low_mags = rng.uniform(0.05, 0.35, size=20)
    low_vecs = directions[:20] * low_mags[:, None]
    for b_vec, b_mag in zip(low_vecs, low_mags):
        res = simulate_clean_odmr(
            CANONICAL_FREQS, ODMRParams(b_field_mt=tuple(b_vec.tolist()))
        )
        assert abs(res["b_From_splittings_mt"] - b_mag) < 5e-5

    # Single-axis linear Zeeman check with pure parallel field
    b_par_only = 0.5
    f_m, f_p, _ = solve_axis_transitions(b_par_only, 0.0, d0=D0_MHZ)
    recovered_1ax = b_mag_from_splittings(np.array([f_p - f_m]))
    assert abs(recovered_1ax - b_par_only) < 1e-12


def test_transverse_field_contrast_reduction() -> None:
    """Increasing B_perp at fixed B_par must monotonically decrease contrast_factor."""
    b_par = 0.5
    b_perp_vals = np.linspace(0.0, 5.0, 25)
    _, _, contrast_factors = solve_axis_transitions(
        np.full_like(b_perp_vals, b_par), b_perp_vals, d0=D0_MHZ
    )
    assert contrast_factors[0] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(contrast_factors) < 0.0)


def test_peak_containment_in_window() -> None:
    """For |B| = 3.20 mT in any direction, all transitions stay inside [2780, 2960] MHz."""
    rng = np.random.default_rng(TEST_SEED + 2)
    dirs = rng.normal(size=(60, 3))
    special = np.vstack(
        [
            NV_AXES,
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]) / np.sqrt(2.0),
        ]
    )
    all_dirs = np.vstack([special, dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)])
    b_vecs = all_dirs * B_MAG_MAX_MT

    for b_vec in b_vecs:
        res = simulate_clean_odmr(
            CANONICAL_FREQS, ODMRParams(b_field_mt=tuple(b_vec.tolist()))
        )
        dip_pairs = res["dip_pairs"]
        assert np.all(dip_pairs > 2780.0)
        assert np.all(dip_pairs < 2960.0)


def test_pchip_resampler_and_padding() -> None:
    """Test PCHIP downsampling, safe narrow-window edge padding, and range validation."""
    rng = np.random.default_rng(TEST_SEED + 3)
    params = ODMRParams(b_field_mt=(1.15, 0.75, 0.40))

    # 1. Downsampling from 1024 points to 512 points
    freqs_dense = np.linspace(2750.0, 2990.0, 1024)
    spec_dense = simulate_clean_odmr(freqs_dense, params)["spectrum"]
    resampled_dense, meta_dense = resample_spectrum_pchip(
        freqs_dense, spec_dense, target_freqs=CANONICAL_FREQS, rng=rng
    )
    assert resampled_dense.shape == (512,)
    assert not meta_dense["window_padded"]
    assert meta_dense["original_points"] == 1024
    assert np.all(np.isfinite(resampled_dense))
    spec_canonical = simulate_clean_odmr(CANONICAL_FREQS, params)["spectrum"]
    assert np.max(np.abs(resampled_dense - spec_canonical)) < 5e-4

    # 2. Narrow window [2800.0, 2940.0] triggers edge padding without blowup
    freqs_narrow = np.linspace(2800.0, 2940.0, 300)
    spec_narrow = simulate_clean_odmr(freqs_narrow, params)["spectrum"]
    spec_narrow_noisy = apply_realistic_noise(
        spec_narrow, HardwareNoiseConfig(target_snr=15.0, baseline_drift_amp=0.0, hum_amp=0.0), rng=rng
    )
    resampled_pad, meta_pad = resample_spectrum_pchip(
        freqs_narrow, spec_narrow_noisy, target_freqs=CANONICAL_FREQS, rng=rng
    )
    assert meta_pad["window_padded"] is True
    assert meta_pad["original_min_mhz"] == pytest.approx(2800.0)
    assert meta_pad["original_max_mhz"] == pytest.approx(2940.0)
    assert np.all(np.isfinite(resampled_pad))
    left_pad = resampled_pad[CANONICAL_FREQS < 2800.0]
    right_pad = resampled_pad[CANONICAL_FREQS > 2940.0]
    assert np.all((left_pad > 0.9) & (left_pad < 1.1))
    assert np.all((right_pad > 0.9) & (right_pad < 1.1))

    # 3. Invalid window [2900.0, 2980.0] raises ValueError
    freqs_invalid = np.linspace(2900.0, 2980.0, 200)
    spec_invalid = np.ones_like(freqs_invalid)
    with pytest.raises(ValueError, match="2830, 2910"):
        resample_spectrum_pchip(freqs_invalid, spec_invalid)

    freqs_invalid_upper = np.linspace(2760.0, 2890.0, 200)
    with pytest.raises(ValueError, match="2830, 2910"):
        resample_spectrum_pchip(freqs_invalid_upper, np.ones_like(freqs_invalid_upper))


def test_batch_simulator_consistency() -> None:
    """Verify batched forward model matches scalar loop reference to <= 1e-10."""
    rng = np.random.default_rng(TEST_SEED + 4)
    batch_size = 32
    b_vecs = rng.uniform(-2.2, 2.2, size=(batch_size, 3))
    d0_arr = rng.uniform(2865.0, 2875.0, size=batch_size)
    gamma_arr = rng.uniform(3.0, 9.0, size=batch_size)
    contrast_arr = rng.uniform(0.015, 0.05, size=batch_size)
    eta_arr = rng.uniform(0.0, 1.0, size=batch_size)

    for include_hyperfine in (True, False):
        batch_spectra = simulate_batch_clean_odmr(
            CANONICAL_FREQS,
            b_vecs,
            d0_arr,
            gamma_arr,
            contrast_arr,
            eta_arr,
            include_hyperfine=include_hyperfine,
        )
        assert batch_spectra.shape == (batch_size, len(CANONICAL_FREQS))

        hf_offsets = [-A_HYPERFINE_MHZ, 0.0, A_HYPERFINE_MHZ] if include_hyperfine else [0.0]
        hf_scale = (1.0 / 3.0) if include_hyperfine else 1.0

        for i in range(batch_size):
            # Explicit scalar reference computed axis-by-axis and hyperfine-line-by-line
            bv = b_vecs[i]
            b_sq = float(np.dot(bv, bv))
            scalar_spec = np.ones_like(CANONICAL_FREQS, dtype=np.float64)
            for k in range(4):
                b_par = float(abs(np.dot(bv, NV_AXES[k])))
                b_perp = float(np.sqrt(max(0.0, b_sq - b_par * b_par)))
                fm, fp, cf = solve_axis_transitions(b_par, b_perp, d0=float(d0_arr[i]))
                eff_amp = float(contrast_arr[i]) * 0.28 * cf * hf_scale
                for f_center in (fm, fp):
                    for dhf in hf_offsets:
                        scalar_spec -= pseudo_voigt_dip(
                            CANONICAL_FREQS,
                            f_center + dhf,
                            eff_amp,
                            float(gamma_arr[i]),
                            float(eta_arr[i]),
                        )
            max_diff = float(np.max(np.abs(batch_spectra[i] - scalar_spec)))
            assert max_diff <= 1e-10

            single = simulate_clean_odmr(
                CANONICAL_FREQS,
                ODMRParams(
                    b_field_mt=tuple(bv.tolist()),
                    base_contrast=float(contrast_arr[i]),
                    linewidth_mhz=float(gamma_arr[i]),
                    voigt_eta=float(eta_arr[i]),
                    d0_mhz=float(d0_arr[i]),
                    include_hyperfine=include_hyperfine,
                ),
            )
            assert np.max(np.abs(batch_spectra[i] - single["spectrum"])) <= 1e-10


def test_b_mag_from_splittings_orders_on_exact_eigh() -> None:
    """Validate order=1 and order=2 b_mag_from_splittings against exact eigh splittings."""
    rng = np.random.default_rng(TEST_SEED + 5)
    n_samples = 2000
    dirs = rng.normal(size=(n_samples, 3))
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    mags = rng.uniform(0.1, 3.2, size=n_samples)
    b_vecs = dirs * mags[:, None]

    b_par = np.abs(b_vecs @ NV_AXES.T)
    b_perp = np.sqrt(np.maximum(0.0, mags[:, None] ** 2 - b_par ** 2))
    fm, fp, _ = solve_axis_transitions(b_par, b_perp, d0=D0_MHZ)
    splittings = fp - fm

    b_ord1 = b_mag_from_splittings(splittings, order=1)
    b_ord2 = b_mag_from_splittings(splittings, order=2, d0_mhz=D0_MHZ)

    err1_ut = np.abs(b_ord1 - mags) * 1000.0
    err2_ut = np.abs(b_ord2 - mags) * 1000.0

    rms1_ut = float(np.sqrt(np.mean(err1_ut ** 2)))
    max1_ut = float(np.max(err1_ut))
    rms2_ut = float(np.sqrt(np.mean(err2_ut ** 2)))
    max2_ut = float(np.max(err2_ut))

    # Tolerances derived from Part 1 10,000-direction validation:
    # order=1: RMS ~ 0.088 uT, Max ~ 0.51 uT; order=2: RMS ~ 0.23 uT, Max ~ 0.70 uT
    assert rms1_ut < 0.15
    assert max1_ut < 0.65
    assert rms2_ut < 0.30
    assert max2_ut < 0.85
    assert rms2_ut < 0.002
    assert max2_ut < 0.005
    assert rms2_ut < rms1_ut


def test_wedge_invariance_48_symmetries_and_strain_guard() -> None:
    """All 48 O_h sign/permutation variants of B must give identical |B| and canonical b_vec."""
    base_b = np.array([1.15, 0.75, 0.40], dtype=np.float64)
    expected_canon = canonicalize_b_vec(base_b)
    expected_mag = float(np.linalg.norm(base_b))

    variants_48 = []
    for perm in permutations((0, 1, 2)):
        p_vec = base_b[list(perm)]
        for signs in product((-1.0, 1.0), repeat=3):
            variants_48.append(p_vec * np.asarray(signs, dtype=np.float64))
    assert len(variants_48) == 48

    b_mags = []
    b_canons = []
    for v in variants_48:
        spec = simulate_clean_odmr(CANONICAL_FREQS, ODMRParams(b_field_mt=tuple(v.tolist())))["spectrum"]
        fit = fit_physical_odmr(CANONICAL_FREQS, spec)
        assert fit["success"]
        assert fit["status"] == "converged"
        b_mags.append(float(fit["b_mag"]))
        b_canons.append(fit["b_vec"])

    b_mags_arr = np.asarray(b_mags)
    b_canons_arr = np.asarray(b_canons)
    # All 48 symmetry variants produce identical |B| to < 0.01 uT and identical canonical B_vec
    assert np.max(np.abs(b_mags_arr - expected_mag)) * 1000.0 < 0.01
    assert np.max(b_mags_arr) - np.min(b_mags_arr) < 1e-8
    assert np.allclose(b_canons_arr, expected_canon[None, :], atol=1e-5)

    # Non-zero strain E != 0 breaks tetrahedral wedge symmetry and must raise ValueError
    with pytest.raises(ValueError, match="strain"):
        canonicalize_b_vec(base_b, e_mhz=1.5)
    with pytest.raises(ValueError, match="strain"):
        fit_physical_odmr(CANONICAL_FREQS, spec, e_mhz=1.5)


def test_noise_and_baseline_fitter_sanity() -> None:
    """Verify pink noise statistics, heavy-tail flag, oracle start, and 1-axis/naive fitters."""
    rng = np.random.default_rng(TEST_SEED + 6)
    pink = generate_pink_noise(4096, rng=rng)
    assert abs(np.mean(pink)) < 1e-10
    assert abs(np.std(pink) - 1.0) < 1e-10

    params = ODMRParams(b_field_mt=(1.15, 0.75, 0.40), include_hyperfine=True)
    clean = simulate_clean_odmr(CANONICAL_FREQS, params)["spectrum"]
    noisy = apply_realistic_noise(
        clean, HardwareNoiseConfig(target_snr=15.0, use_heavy_tails=True), rng=rng
    )
    assert noisy.shape == clean.shape
    assert np.all(np.isfinite(noisy))

    # Noiseless oracle-start error must be < 1 uT
    fit_oracle = fit_physical_odmr(CANONICAL_FREQS, clean, warm_start_b_vec=np.array([1.15, 0.75, 0.40]))
    assert fit_oracle["success"]
    assert abs(fit_oracle["b_mag"] - np.linalg.norm([1.15, 0.75, 0.40])) * 1000.0 < 1.0

    # Single-axis fit check
    clean_1ax = simulate_clean_odmr(
        CANONICAL_FREQS, ODMRParams(b_field_mt=(1.0, 0.0, 0.0), single_axis=True)
    )["spectrum"]
    fit_1ax = fit_physical_odmr(CANONICAL_FREQS, clean_1ax, single_axis=True)
    expected_b_par = float(abs(np.dot(np.array([1.0, 0.0, 0.0]), NV_AXES[0])))
    assert abs(fit_1ax["b_mag"] - expected_b_par) < 2e-3

    # Naive 8-Lorentzian fitter check
    naive_res = fit_lorentzian_naive(CANONICAL_FREQS, clean)
    assert np.isfinite(naive_res["b_mag"])
    assert np.all(np.isfinite(naive_res["fitted_curve"]))


def test_baseline_statistical_accuracy_snr15() -> None:
    """Statistical accuracy test: >= 50 realizations at SNR=15 against CRB-derived thresholds."""
    test_vectors = np.array(
        [
            [1.15, 0.75, 0.40],  # Degenerate central axis (B . n_2 = 0)
            [0.35, 0.80, 1.65],  # General non-degenerate 4-axis direction
            [0.20, 0.55, 1.10],  # Moderate field
            [0.60, 0.60, 1.40],  # Symmetry-plane degenerate pair (Bx = By)
            [0.45, 1.05, 2.15],  # Higher field
        ],
        dtype=np.float64,
    )
    noise_cfg = HardwareNoiseConfig(
        target_snr=15.0,
        dwell_time_ms=0.8,
        mains_freq_hz=50.0,
        hum_amp=0.0015,
        baseline_drift_amp=0.005,
        pink_noise_ratio=0.25,
    )

    crb_vals = [
        _compute_white_gaussian_crb_ut(CANONICAL_FREQS, bv, target_snr=15.0)
        for bv in test_vectors
    ]
    median_crb_ut = float(np.median(crb_vals))

    rng = np.random.default_rng(TEST_SEED + 100)
    n_realizations = 50
    errors_plain_ut = []
    errors_informed_ut = []

    for i in range(n_realizations):
        b_vec = test_vectors[i % len(test_vectors)]
        b_true_mag = float(np.linalg.norm(b_vec))
        clean = simulate_clean_odmr(
            CANONICAL_FREQS, ODMRParams(b_field_mt=tuple(b_vec.tolist()))
        )["spectrum"]
        noisy = apply_realistic_noise(clean, noise_cfg, rng=rng)

        res_plain = fit_physical_odmr(CANONICAL_FREQS, noisy, variant="plain")
        assert np.isfinite(res_plain["b_mag"])
        assert np.all(np.isfinite(res_plain["fitted_curve"]))
        errors_plain_ut.append(abs(float(res_plain["b_mag"]) - b_true_mag) * 1000.0)

        res_inf = fit_physical_odmr(
            CANONICAL_FREQS, noisy, dwell_time=noise_cfg.dwell_time_ms * 1e-3, variant="informed"
        )
        assert np.isfinite(res_inf["b_mag"])
        errors_informed_ut.append(abs(float(res_inf["b_mag"]) - b_true_mag) * 1000.0)

    arr_plain = np.asarray(errors_plain_ut, dtype=np.float64)
    arr_inf = np.asarray(errors_informed_ut, dtype=np.float64)

    # Thresholds derived directly from CRB (Part 0d):
    # Median error within 3.0x CRB, failure rate (|err| > 100 uT) < 2%
    assert float(np.median(arr_plain)) <= 3.0 * median_crb_ut
    assert float(np.mean(arr_plain > 100.0)) < 0.02

    assert float(np.median(arr_inf)) <= 2.5 * median_crb_ut
    assert float(np.mean(arr_inf > 100.0)) < 0.02


def test_noise_paths_identical_and_components() -> None:
    """Verify apply_realistic_noise return_components=False vs True produce identical outputs."""
    clean = simulate_clean_odmr(
        CANONICAL_FREQS, ODMRParams(b_field_mt=(1.15, 0.75, 0.40))
    )["spectrum"]
    for use_heavy in (False, True):
        cfg = HardwareNoiseConfig(
            target_snr=4.0,
            dwell_time_ms=0.8,
            mains_freq_hz=50.0,
            hum_amp=0.003,
            baseline_drift_amp=0.012,
            pink_noise_ratio=0.25,
            use_heavy_tails=use_heavy,
        )
        rng_a = np.random.default_rng(TEST_SEED + 201)
        rng_b = np.random.default_rng(TEST_SEED + 201)
        noisy_plain = apply_realistic_noise(clean, cfg, rng=rng_a, return_components=False)
        noisy_tuple = apply_realistic_noise(clean, cfg, rng=rng_b, return_components=True)
        assert isinstance(noisy_tuple, tuple)
        noisy_comp, comps = noisy_tuple
        assert np.array_equal(noisy_plain, noisy_comp)
        assert np.array_equal(noisy_plain, comps["total"])
        reconstructed = (
            comps["clean"] + comps["drift"] + comps["hum"] + comps["pink"] + comps["white"]
        )
        assert np.allclose(reconstructed, comps["total"], atol=1e-14)


def test_resampler_valid_mask_end_to_end_and_out_of_range() -> None:
    """Test resampler valid_mask, end-to-end 2800-2940 MHz fit, out_of_range flag, and 1001->512 cell-averaging."""
    rng = np.random.default_rng(TEST_SEED + 202)
    params_in_range = ODMRParams(b_field_mt=(1.15, 0.75, 0.40))
    b_true = float(np.linalg.norm(params_in_range.b_field_mt))

    # 1. End-to-end 2800-2940 MHz input -> resampler -> fit_physical_odmr with valid_mask
    freqs_2800_2940 = np.linspace(2800.0, 2940.0, 350)
    clean_sub = simulate_clean_odmr(freqs_2800_2940, params_in_range)["spectrum"]
    noisy_sub = apply_realistic_noise(
        clean_sub,
        HardwareNoiseConfig(target_snr=15.0, hum_amp=0.001, baseline_drift_amp=0.003),
        rng=rng,
    )
    resampled, valid_mask, meta = resample_spectrum_pchip(
        freqs_2800_2940, noisy_sub, target_freqs=CANONICAL_FREQS, return_valid_mask=True
    )
    assert valid_mask.dtype == bool
    assert np.array_equal(valid_mask, meta["valid_mask"])
    assert np.all(valid_mask == ((CANONICAL_FREQS >= 2800.0) & (CANONICAL_FREQS <= 2940.0)))
    assert meta["out_of_range"] is False

    fit_sub = fit_physical_odmr(CANONICAL_FREQS, resampled, valid_mask=valid_mask, variant="plain")
    assert fit_sub["success"] is True
    assert fit_sub["out_of_range"] is False
    assert fit_sub["status"] == "converged"
    assert abs(float(fit_sub["b_mag"]) - b_true) * 1000.0 < 35.0

    # 2. High-field |B| = 2.50 mT on narrow [2830, 2910] MHz window must trigger out_of_range
    params_high_b = ODMRParams(b_field_mt=(0.40, 0.90, 2.30))
    freqs_narrow = np.linspace(2830.0, 2910.0, 250)
    clean_high = simulate_clean_odmr(freqs_narrow, params_high_b)["spectrum"]
    res_high, mask_high, meta_high = resample_spectrum_pchip(
        freqs_narrow, clean_high, target_freqs=CANONICAL_FREQS, return_valid_mask=True
    )
    fit_high = fit_physical_odmr(CANONICAL_FREQS, res_high, valid_mask=mask_high, variant="plain")
    assert fit_high["out_of_range"] is True
    assert fit_high["status"] == "out_of_range"
    assert fit_high["success"] is False

    # 3. 1001 -> 512 cell-averaging downsampling accuracy
    freqs_1001 = np.linspace(2750.0, 2990.0, 1001)
    clean_1001 = simulate_clean_odmr(freqs_1001, params_in_range)["spectrum"]
    clean_512 = simulate_clean_odmr(CANONICAL_FREQS, params_in_range)["spectrum"]
    down_512, meta_1001 = resample_spectrum_pchip(freqs_1001, clean_1001, target_freqs=CANONICAL_FREQS)
    assert np.all(meta_1001["valid_mask"])
    assert float(np.sqrt(np.mean((down_512 - clean_512) ** 2))) < 5e-6


def test_naive_fitter_sanity_snr15() -> None:
    """Verify rewritten 4-symmetric-pair naive Lorentzian achieves median |B| error << 1 mT at SNR=15."""
    rng = np.random.default_rng(TEST_SEED + 203)
    test_vecs = np.array(
        [
            [1.15, 0.75, 0.40],
            [0.35, 0.80, 1.65],
            [0.20, 0.55, 1.10],
            [0.60, 0.60, 1.40],
            [0.45, 1.05, 2.15],
        ],
        dtype=np.float64,
    )
    cfg = HardwareNoiseConfig(
        target_snr=15.0,
        dwell_time_ms=0.8,
        mains_freq_hz=50.0,
        hum_amp=0.0015,
        baseline_drift_amp=0.005,
        pink_noise_ratio=0.25,
    )
    errs_ut = []
    for i in range(25):
        bv = test_vecs[i % len(test_vecs)]
        b_true = float(np.linalg.norm(bv))
        clean = simulate_clean_odmr(CANONICAL_FREQS, ODMRParams(b_field_mt=tuple(bv.tolist())))["spectrum"]
        noisy = apply_realistic_noise(clean, cfg, rng=rng)
        res = fit_lorentzian_naive(CANONICAL_FREQS, noisy)
        assert np.isfinite(res["b_mag"])
        assert np.all(np.isfinite(res["fitted_curve"]))
        errs_ut.append(abs(float(res["b_mag"]) - b_true) * 1000.0)

    med_ut = float(np.median(errs_ut))
    # Acceptance criterion (Task 1c): median |B| error at SNR=15 well below 1 mT (1000 uT)
    assert med_ut < 500.0

