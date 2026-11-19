"""Deterministic unit test suite for Stage 2 dataset generation, split protocol, stats, and real-data loader."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pytest

from src.config import (
    B_MAG_MAX_MT,
    B_MAG_MIN_MT,
    CANONICAL_FREQS,
    D0_MHZ,
    GAMMA_E_MHZ_MT,
    N_POINTS,
)
from src.dataset_gen import (
    DEFAULT_SYNTHETIC_DIR,
    LOCKED_TEST_SPLITS,
    MANIFEST_PATH,
    SPLIT_REGISTRY,
    TEST_ACCESS_LOG_PATH,
    NumpyODMRDataset,
    OnTheFlyODMRIterableDataset,
    SplitConfig,
    compute_file_sha256,
    derive_sample_seed,
    generate_single_sample,
    generate_split_arrays,
    get_stage3_sim2real_train_config,
    get_test_access_counts,
    load_synthetic_split,
    regenerate_sample_components,
    select_12_unique_example_indices,
    target_regime_for_index,
)
from src.noise import (
    HardwareNoiseConfig,
    apply_realistic_noise,
    generate_pink_noise,
    resample_spectrum_pchip,
)
from src.noise_models import (
    DRIFT_F3_EXP_OOD,
    DRIFT_F5_OOD,
    GRID_PERTURBED_CROP,
    ID_DRIFT_FAMILIES,
    NOISE_GAUSSIAN_ID,
    NOISE_STUDENT_T_SPIKES_OOD,
    OOD_DRIFT_FAMILIES,
    REGIME_EASY,
    REGIME_HARD,
    REGIME_MEDIUM,
    compute_depth_outer_and_flags,
    evaluate_stage15_eff_snr_consistency,
)
from src.physics import ODMRParams, b_mag_from_splittings, simulate_clean_odmr
from src.real_data import (
    DEFAULT_REAL_MANIFEST_PATH,
    load_and_validate_real_trace,
    load_csv_or_npz_odmr_file,
    normalize_frequency_to_mhz,
    normalize_odmr_spectrum,
)
from src.stats import (
    bootstrap_metric_ci,
    bootstrap_summary_cis,
    mcnemar_paired_test,
    paired_bootstrap_diff_ci,
)


def test_stats_paired_bootstrap_and_mcnemar_synthetic() -> None:
    """Part 0b: Unit-test paired bootstrap CI helpers and McNemar test on synthetic data."""
    rng = np.random.default_rng(202600)
    base_difficulty = rng.exponential(scale=20.0, size=300)
    err_a = base_difficulty + rng.uniform(8.0, 14.0, size=300)
    err_b = base_difficulty + rng.uniform(1.0, 4.0, size=300)

    fail_a = err_a > 35.0
    fail_b = err_b > 35.0

    diff_cis = paired_bootstrap_diff_ci(
        err_a, err_b, fail_a=fail_a, fail_b=fail_b, n_boot=1000, ci=0.95, seed=77
    )
    for k in ("median_diff", "rms_diff", "mae_diff", "p90_diff", "fail_rate_diff"):
        pt, lo, hi = diff_cis[k]
        assert lo <= pt <= hi
        assert lo > 0.0, f"Expected strictly positive difference for {k}, got CI [{lo}, {hi}]"

    mcn = mcnemar_paired_test(fail_a, fail_b)
    assert mcn["n_samples"] == 300
    assert mcn["n_a_only"] > mcn["n_b_only"]
    assert mcn["fail_rate_diff"] > 0.0
    assert 0.0 <= mcn["p_value"] < 0.05

    # Identical failure arrays must yield p_value == 1.0 and 0 discordant pairs
    mcn_eq = mcnemar_paired_test(fail_a, fail_a)
    assert mcn_eq["n_discordant"] == 0
    assert mcn_eq["p_value"] == 1.0
    assert mcn_eq["chi2_cc"] == 0.0

    # Single-method summary CIs
    sum_cis = bootstrap_summary_cis(
        err_a, fail_flags=fail_a, wrong_basin_flags=fail_a, n_boot=500, seed=11
    )
    for k in ("median", "p90", "rms", "mae", "fail_rate", "wrong_basin_rate"):
        pt, lo, hi = sum_cis[k]
        assert lo <= pt <= hi

    pt_m, lo_m, hi_m = bootstrap_metric_ci(
        err_a, lambda x: np.median(x, axis=-1), n_boot=500, seed=11
    )
    assert lo_m <= pt_m <= hi_m


def test_sample_i_determinism_and_batch_order_independence() -> None:
    """Verify sample i is bit-identical regardless of batch size, offset, or iterable sharding."""
    cfg = SPLIT_REGISTRY["val_id"]
    target_indices = [0, 7, 1332, 1333, 2666, 3999]

    for idx in target_indices:
        s_single_1 = generate_single_sample(idx, cfg)
        s_single_2 = generate_single_sample(idx, cfg)
        assert np.array_equal(s_single_1["X_noisy"], s_single_2["X_noisy"])
        assert np.array_equal(s_single_1["Y_clean"], s_single_2["Y_clean"])
        assert np.array_equal(s_single_1["Y_splittings"], s_single_2["Y_splittings"])
        assert s_single_1["Y_b_mag"] == s_single_2["Y_b_mag"]
        assert s_single_1["aux_sample_seed"] == s_single_2["aux_sample_seed"]

    # Compare batch [5..12) vs batch [0..15) for index 7
    batch_a = generate_split_arrays(cfg, n_samples=7, start_idx=5, n_workers=1)
    batch_b = generate_split_arrays(cfg, n_samples=15, start_idx=0, n_workers=1)
    assert np.array_equal(batch_a["X_noisy"][2], batch_b["X_noisy"][7])
    assert np.array_equal(batch_a["Y_clean"][2], batch_b["Y_clean"][7])
    assert np.array_equal(batch_a["valid_mask"][2], batch_b["valid_mask"][7])

    # Compare OnTheFlyODMRIterableDataset with saved val_id.npz
    saved_val = load_synthetic_split("val_id")
    for idx in target_indices:
        s_single = generate_single_sample(idx, cfg)
        assert np.array_equal(s_single["X_noisy"], saved_val["X_noisy"][idx])
        assert np.array_equal(s_single["Y_clean"], saved_val["Y_clean"][idx])

    stream = list(OnTheFlyODMRIterableDataset("train", start_idx=10, max_samples=4, batch_size=2))
    assert len(stream) == 2
    s10 = generate_single_sample(10, SPLIT_REGISTRY["train"])
    s13 = generate_single_sample(13, SPLIT_REGISTRY["train"])
    assert np.array_equal(stream[0]["X_noisy"][0], s10["X_noisy"])
    assert np.array_equal(stream[1]["X_noisy"][1], s13["X_noisy"])

    ds_map = NumpyODMRDataset("val_id", in_memory_arrays=saved_val, as_torch=False)
    assert len(ds_map) == 4000
    item_7 = ds_map[7]
    assert np.array_equal(item_7["X_noisy"], saved_val["X_noisy"][7])


def test_seed_disjointness_and_no_identical_parameters_across_splits() -> None:
    """Part 3: Verify master split seeds, per-sample seeds, and parameter vectors are strictly disjoint."""
    master_seeds = [cfg.split_seed for cfg in SPLIT_REGISTRY.values()]
    assert len(master_seeds) == len(set(master_seeds)), "Master split seeds are not unique!"

    inspectable_splits = [
        "train_snapshot",
        "val_id",
        "val_id_natural",
        "val_ood",
        "ood_hwhm",
        "ood_contrast",
        "ood_d0",
        "ood_noise",
        "ood_drift",
        "ood_hum",
        "ood_grid",
        "ood_high_field",
        "ood_asym",
    ]

    all_sample_seeds: set[int] = set()
    all_param_tuples: set[tuple[float, float, float, float, float]] = set()
    total_count = 0

    for sname in inspectable_splits:
        data = load_synthetic_split(sname)
        seeds = data["aux_sample_seed"].tolist()
        n_s = len(seeds)
        total_count += n_s
        for s_val in seeds:
            assert int(s_val) not in all_sample_seeds, f"Duplicate sample_seed {s_val} in {sname}"
            all_sample_seeds.add(int(s_val))

        b_arr = data["Y_b_mag"]
        d0_arr = data["Y_d0"]
        hw_arr = data["Y_hwhm"]
        c_arr = data["aux_contrast"]
        eta_arr = data["aux_eta"]
        for i in range(n_s):
            pt = (
                float(b_arr[i]),
                float(d0_arr[i]),
                float(hw_arr[i]),
                float(c_arr[i]),
                float(eta_arr[i]),
            )
            assert pt not in all_param_tuples, f"Duplicate parameter vector {pt} in {sname}"
            all_param_tuples.add(pt)

    # Also verify derived sample seeds for test_id and test_ood (without reading test files!)
    for tname in LOCKED_TEST_SPLITS:
        tcfg = SPLIT_REGISTRY[tname]
        for idx in range(tcfg.n_samples):
            tseed = derive_sample_seed(tcfg.split_seed, idx)
            assert tseed not in all_sample_seeds
            all_sample_seeds.add(tseed)
            total_count += 1

    assert len(all_sample_seeds) == total_count


def test_per_split_range_checks_and_disjoint_ood_supports() -> None:
    """Part 3: Verify per-split min/max bounds and 0% overlap between ID and OOD shifted axes."""
    id_splits = ["train_snapshot", "val_id"]
    for sname in id_splits:
        d = load_synthetic_split(sname)
        assert float(np.min(d["Y_hwhm"])) >= 3.0 and float(np.max(d["Y_hwhm"])) <= 9.0
        assert float(np.min(d["aux_contrast"])) >= 0.015 and float(np.max(d["aux_contrast"])) <= 0.040
        assert float(np.min(d["Y_d0"])) >= -3.0 and float(np.max(d["Y_d0"])) <= 3.0
        assert float(np.min(d["Y_b_mag"])) >= B_MAG_MIN_MT and float(np.max(d["Y_b_mag"])) <= B_MAG_MAX_MT
        assert float(np.min(d["aux_eta"])) >= 0.0 and float(np.max(d["aux_eta"])) <= 1.0
        assert float(np.max(d["aux_hum_rel_harmonics"])) <= 0.50 + 1e-6
        assert bool(np.all(d["aux_e_mhz"] == 0.0))

    # Combined val_ood and single-factor OOD checks
    for sname in ("val_ood", "ood_hwhm"):
        d = load_synthetic_split(sname)
        assert float(np.min(d["Y_hwhm"])) >= 9.5 - 1e-5
        assert float(np.max(d["Y_hwhm"])) <= 13.5 + 1e-5

    for sname in ("val_ood", "ood_contrast"):
        d = load_synthetic_split(sname)
        c = d["aux_contrast"]
        in_lo = (c >= 0.010 - 1e-6) & (c <= 0.0145 + 1e-6)
        in_hi = (c >= 0.0405 - 1e-6) & (c <= 0.050 + 1e-6)
        assert bool(np.all(in_lo | in_hi))
        assert not bool(np.any((c > 0.0145 + 1e-6) & (c < 0.0405 - 1e-6)))

    for sname in ("val_ood", "ood_d0"):
        d = load_synthetic_split(sname)
        d0 = d["Y_d0"]
        in_lo = (d0 >= -5.0 - 1e-5) & (d0 <= -3.1 + 1e-5)
        in_hi = (d0 >= 3.1 - 1e-5) & (d0 <= 5.0 + 1e-5)
        assert bool(np.all(in_lo | in_hi))
        assert not bool(np.any((d0 > -3.1 + 1e-5) & (d0 < 3.1 - 1e-5)))

    for sname in ("val_ood", "ood_hum"):
        d = load_synthetic_split(sname)
        rel_h = d["aux_hum_rel_harmonics"]
        assert float(np.min(rel_h)) > 0.50
        assert float(np.max(rel_h)) <= 0.85 + 1e-5
    d_hum = load_synthetic_split("ood_hum")
    assert bool(np.all(d_hum["aux_hum_freq_hz"] > 40.0))


def test_ood_only_families_and_noise_absent_from_id_splits() -> None:
    """Part 2 & 3: Assert OOD drift families (F3, F5_ood) and Student-t noise never appear in ID splits."""
    id_drift_splits = [
        "train_snapshot",
        "val_id",
        "ood_hwhm",
        "ood_contrast",
        "ood_d0",
        "ood_noise",
        "ood_hum",
        "ood_grid",
    ]
    for sname in id_drift_splits:
        d = load_synthetic_split(sname)
        fams = set(np.unique(d["aux_drift_family_id"]).tolist())
        assert fams.issubset(set(ID_DRIFT_FAMILIES)), f"Unexpected drift families {fams} in {sname}"
        assert DRIFT_F3_EXP_OOD not in fams
        assert DRIFT_F5_OOD not in fams

    for sname in ("val_ood", "ood_drift"):
        d = load_synthetic_split(sname)
        fams = set(np.unique(d["aux_drift_family_id"]).tolist())
        assert fams == set(OOD_DRIFT_FAMILIES)

    id_noise_splits = [
        "train_snapshot",
        "val_id",
        "ood_hwhm",
        "ood_contrast",
        "ood_d0",
        "ood_drift",
        "ood_hum",
        "ood_grid",
    ]
    for sname in id_noise_splits:
        d = load_synthetic_split(sname)
        assert bool(np.all(d["aux_noise_type"] == NOISE_GAUSSIAN_ID))

    for sname in ("val_ood", "ood_noise"):
        d = load_synthetic_split(sname)
        assert bool(np.all(d["aux_noise_type"] == NOISE_STUDENT_T_SPIKES_OOD))


def test_component_identity_via_regeneration() -> None:
    """Part 2: Verify Y_clean + drift + hum + pink + white + other == X_noisy_f64 to 1e-12 (1e-6 in f32)."""
    for sname in ("val_id", "val_ood", "ood_grid"):
        d = load_synthetic_split(sname)
        rng = np.random.default_rng(918273)
        indices = rng.choice(d["X_noisy"].shape[0], size=40, replace=False)
        for idx in indices:
            s_idx = int(d["aux_sample_idx"][idx])
            comps = regenerate_sample_components(sname, s_idx)
            total_f64 = (
                comps["Y_clean"]
                + comps["drift"]
                + comps["hum"]
                + comps["pink"]
                + comps["white"]
                + comps["other"]
            )
            diff_f64 = float(np.max(np.abs(total_f64 - comps["X_noisy_f64"])))
            diff_f32 = float(np.max(np.abs(total_f64.astype(np.float32) - d["X_noisy"][idx])))
            assert diff_f64 < 1e-12, f"float64 identity failed ({diff_f64}) for {sname}[{s_idx}]"
            assert diff_f32 < 1e-6, f"float32 identity failed ({diff_f32}) for {sname}[{s_idx}]"


def test_target_consistency_clean_odmr_and_exact_eigh_splittings() -> None:
    """Part 6: Verify Y_clean == simulate_clean_odmr to 1e-10, Y_splittings == eigh, and order-2 |B| RMS."""
    d = load_synthetic_split("val_id")
    rng = np.random.default_rng(445566)
    indices = rng.choice(d["X_noisy"].shape[0], size=30, replace=False)

    for idx in indices:
        s_idx = int(d["aux_sample_idx"][idx])
        comps = regenerate_sample_components("val_id", s_idx)
        b_vec = d["aux_b_vec_canon"][idx].astype(np.float64)
        d0_val = D0_MHZ + float(d["Y_d0"][idx])
        hwhm_val = float(d["Y_hwhm"][idx])
        contrast_val = float(d["aux_contrast"][idx])
        eta_val = float(d["aux_eta"][idx])

        sim = simulate_clean_odmr(
            CANONICAL_FREQS,
            ODMRParams(
                b_field_mt=(float(b_vec[0]), float(b_vec[1]), float(b_vec[2])),
                base_contrast=contrast_val,
                linewidth_mhz=hwhm_val,
                voigt_eta=eta_val,
                d0_mhz=d0_val,
                e_mhz=0.0,
            ),
        )
        clean_err_f64 = float(np.max(np.abs(comps["Y_clean"] - sim["spectrum"])))
        split_err = float(
            np.max(np.abs(d["Y_splittings"][idx].astype(np.float64) - sim["splittings_mhz"]))
        )
        assert clean_err_f64 < 1e-10
        assert split_err < 1e-5
        assert np.all(np.diff(d["Y_splittings"][idx]) >= -1e-6)

    # Check b_mag_from_splittings(order=2) vs true Y_b_mag across all 4000 val_id spectra
    b_o2 = np.array(
        [
            float(
                b_mag_from_splittings(
                    d["Y_splittings"][i].astype(np.float64),
                    order=2,
                    d0_mhz=float(D0_MHZ + d["Y_d0"][i]),
                )
            )
            for i in range(d["Y_b_mag"].shape[0])
        ],
        dtype=np.float64,
    )
    rms_ut = float(np.sqrt(np.mean(((b_o2 - d["Y_b_mag"].astype(np.float64)) * 1000.0) ** 2)))
    assert rms_ut < 0.5, f"Expected order=2 splitting inversion RMS < 0.5 uT, got {rms_ut:.4f} uT"


def test_resample_path_valid_mask_and_out_of_range() -> None:
    """Verify grid realism produces valid_mask, deterministic edge fill, and out_of_range detection."""
    d_grid = load_synthetic_split("ood_grid")
    assert bool(np.all(d_grid["aux_grid_variant_id"] == GRID_PERTURBED_CROP))
    assert float(np.min(d_grid["aux_orig_min_mhz"])) >= 2789.5
    assert float(np.max(d_grid["aux_orig_max_mhz"])) <= 2950.5

    # Every sample in ood_grid is cropped narrower than [2750, 2990], so outer points have valid_mask == False
    valid_counts = np.sum(d_grid["valid_mask"], axis=1)
    assert bool(np.all(valid_counts < N_POINTS))
    assert bool(np.all(valid_counts > 150))
    # High-field samples in ood_grid must trigger out_of_range
    assert int(np.sum(d_grid["aux_out_of_range"])) > 50


def test_regime_stratification_1333_1333_1334() -> None:
    """Part 1.3 & Part 3: Verify exact 1333/1333/1334 regime stratification on val_id and val_ood."""
    for sname in ("val_id", "val_ood"):
        d = load_synthetic_split(sname)
        reg = d["aux_regime_id"]
        snr_w = d["aux_eff_snr_white"]
        assert int(np.sum(reg == REGIME_HARD)) == 1333
        assert int(np.sum(reg == REGIME_MEDIUM)) == 1333
        assert int(np.sum(reg == REGIME_EASY)) == 1334
        assert bool(np.all(snr_w[reg == REGIME_HARD] < 1.5))
        assert bool(np.all((snr_w[reg == REGIME_MEDIUM] >= 1.5) & (snr_w[reg == REGIME_MEDIUM] <= 3.0)))
        assert bool(np.all(snr_w[reg == REGIME_EASY] > 3.0))

    # Also verify target_regime_for_index deterministic helper for 4000 samples
    t_counts = [0, 0, 0]
    for i in range(4000):
        r = target_regime_for_index(i, 4000, stratified=True)
        assert r is not None
        t_counts[r] += 1
    assert tuple(t_counts) == (1333, 1333, 1334)


def test_depth_outer_fallback_for_weak_b_and_weak_splitting_flag() -> None:
    """Part 1.1: Verify depth_outer fallback when max_k |B_par,k| <= 0.15 mT (|B| < 0.26 mT) and weak_splitting."""
    # For B = (0, 0, 0.12) mT, all 4 axes have |B_par,k| = 0.12 / sqrt(3) = 0.06928 mT < 0.15 mT
    b_weak = np.array([0.0, 0.0, 0.12], dtype=np.float64)
    info_weak = compute_depth_outer_and_flags(
        b_weak, d0_mhz=D0_MHZ, hwhm_mhz=5.0, contrast=0.03, eta=0.35
    )
    assert info_weak["used_fallback"] is True
    assert info_weak["depth_outer"] > 0.002
    assert bool(np.all(info_weak["axis_degenerate"]))
    assert info_weak["weak_splitting"] is True

    # For B = (0.20, 0.10, 0.05) mT (|B| = 0.229 mT), max |B_par| = 0.35 / sqrt(3) = 0.202 mT > 0.15 mT
    # 2 * gamma_e * 0.202 = 11.33 MHz < 2 * HWHM (when HWHM = 7.0 MHz -> 14.0 MHz) -> weak_splitting = True
    b_mid = np.array([0.20, 0.10, 0.05], dtype=np.float64)
    info_mid = compute_depth_outer_and_flags(
        b_mid, d0_mhz=D0_MHZ, hwhm_mhz=7.0, contrast=0.03, eta=0.35
    )
    assert info_mid["used_fallback"] is False
    assert info_mid["depth_outer"] > 0.002
    assert info_mid["weak_splitting"] is True

    # Strong field B = (1.0, 0.5, 0.2) mT with narrow HWHM = 3.5 MHz -> weak_splitting = False
    b_strong = np.array([1.0, 0.5, 0.2], dtype=np.float64)
    info_strong = compute_depth_outer_and_flags(
        b_strong, d0_mhz=D0_MHZ, hwhm_mhz=3.5, contrast=0.03, eta=0.35
    )
    assert info_strong["used_fallback"] is False
    assert info_strong["weak_splitting"] is False


def test_eff_snr_consistency_with_stage15_numbers() -> None:
    """Part 1.2: Verify Stage 2 eff_snr_total reproduces Stage 1.5 FINAL medians (3.17 / 1.10 / 0.68) within 5%."""
    res = evaluate_stage15_eff_snr_consistency()
    for nom_snr, expected_ref in ((15.0, 3.17), (4.0, 1.10), (2.0, 0.68)):
        entry = res[nom_snr]
        assert entry["stage15_ref_median"] == expected_ref
        assert entry["relative_diff"] < 0.05, (
            f"Nominal SNR {nom_snr}: Stage 2 median {entry['stage2_eff_snr_total_median']:.4f} "
            f"differs from {expected_ref} by {entry['relative_diff']*100:.2f}% >= 5%"
        )


def test_tract_clean_fraction_and_independence_from_regime() -> None:
    """Part 1.4 & Part 6: Verify ~20% tract_clean fraction, clean amplitude bounds, and near-zero dirt correlation."""
    for sname in ("train_snapshot", "val_id", "val_ood"):
        d = load_synthetic_split(sname)
        tc = d["aux_tract_clean"]
        frac = float(np.mean(tc == 1))
        assert 0.175 <= frac <= 0.225, f"Expected ~20% tract_clean in {sname}, got {frac:.4f}"

        clean_mask = tc == 1
        depth_c = d["aux_depth_outer"][clean_mask]
        drift_c = d["aux_drift_amp"][clean_mask]
        assert bool(np.all(drift_c <= 0.30 * depth_c + 1e-6))

        # Check independence of dirt_drift and eff_snr_white on non-clean-tract samples
        std_mask = tc == 0
        corr = float(
            np.corrcoef(
                d["aux_dirt_drift"][std_mask].astype(np.float64),
                d["aux_eff_snr_white"][std_mask].astype(np.float64),
            )[0, 1]
        )
        assert abs(corr) < 0.08, f"Expected near-zero correlation on tract_clean==0, got {corr:.4f}"


def test_locked_test_files_zero_data_reads_and_hash_check_logged() -> None:
    """Hard Constraint 3: Verify test_id.npz & test_ood.npz hashes match manifest and data_read count == 0."""
    assert MANIFEST_PATH.exists(), "data/synthetic/manifest.json does not exist!"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for tname in LOCKED_TEST_SPLITS:
        tpath = DEFAULT_SYNTHETIC_DIR / f"{tname}.npz"
        assert tpath.exists(), f"Locked test file {tpath} is missing!"
        expected_sha = manifest["splits"][tname]["sha256"]
        actual_sha = compute_file_sha256(
            tpath,
            split_name=tname,
            log_as_hash_check=True,
            log_path=TEST_ACCESS_LOG_PATH,
        )
        assert actual_sha == expected_sha, f"SHA256 mismatch for {tname}!"

    counts = get_test_access_counts(TEST_ACCESS_LOG_PATH)
    assert counts["data_read"] == 0, f"LOCK VIOLATION: {counts['data_read']} data reads of test files logged!"
    assert counts["hash_check"] >= 2
    assert counts["write_once"] == 2

    # Verify that any attempt to call load_synthetic_split on a test file raises PermissionError
    # and logs "data_read" (tested against a temporary log file so real test_access.log stays at 0!)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_log = Path(tmp_dir) / "dummy_access.log"
        with pytest.raises(PermissionError):
            _ = load_synthetic_split("test_id", log_path=tmp_log, allow_test_read=False)
        dummy_counts = get_test_access_counts(tmp_log)
        assert dummy_counts["data_read"] == 1

    # Re-verify real test_access.log still has 0 data reads
    assert get_test_access_counts(TEST_ACCESS_LOG_PATH)["data_read"] == 0


def test_old_noise_py_outputs_bit_identical_for_fixed_seeds() -> None:
    """Hard Constraint 2: Prove src/noise.py and src/physics.py outputs remain bit-identical for fixed seeds."""
    params = ODMRParams(
        b_field_mt=(1.15, 0.75, 0.40),
        base_contrast=0.03,
        linewidth_mhz=4.5,
        voigt_eta=0.35,
        d0_mhz=2870.0,
    )
    clean_1 = simulate_clean_odmr(CANONICAL_FREQS, params)["spectrum"]
    clean_2 = simulate_clean_odmr(CANONICAL_FREQS, params)["spectrum"]
    assert np.array_equal(clean_1, clean_2)
    assert np.isclose(float(clean_1[0]), 0.9999235645513142, atol=1e-15)
    assert np.isclose(float(clean_1[256]), 0.9835099703385413, atol=1e-15)

    cfg = HardwareNoiseConfig(
        target_snr=15.0,
        dwell_time_ms=0.8,
        mains_freq_hz=50.0,
        hum_amp=0.0015,
        baseline_drift_amp=0.005,
        pink_noise_ratio=0.25,
    )
    p1 = generate_pink_noise(512, rng=np.random.default_rng(12345))
    p2 = generate_pink_noise(512, rng=np.random.default_rng(12345))
    assert np.array_equal(p1, p2)

    n1, c1 = apply_realistic_noise(
        clean_1, cfg, rng=np.random.default_rng(90001), return_components=True
    )
    n2, c2 = apply_realistic_noise(
        clean_1, cfg, rng=np.random.default_rng(90001), return_components=True
    )
    assert np.array_equal(n1, n2)
    for k in c1:
        assert np.array_equal(c1[k], c2[k])


def test_real_data_importer_unit_normalization_and_narrow_window_rejection() -> None:
    """Part 5: Verify frequency unit normalization, narrow window rejection (never padded), and CSV/NPZ fixture loading."""
    f_ghz = np.array([2.750, 2.870, 2.990], dtype=np.float64)
    f_hz = np.array([2.750e9, 2.870e9, 2.990e9], dtype=np.float64)
    f_mhz = np.array([2750.0, 2870.0, 2990.0], dtype=np.float64)

    out_g, u_g = normalize_frequency_to_mhz(f_ghz)
    out_h, u_h = normalize_frequency_to_mhz(f_hz)
    out_m, u_m = normalize_frequency_to_mhz(f_mhz)
    assert u_g == "GHz" and np.allclose(out_g, f_mhz)
    assert u_h == "Hz" and np.allclose(out_h, f_mhz)
    assert u_m == "MHz" and np.allclose(out_m, f_mhz)

    # Narrow window [2.850, 2.890] GHz fails [2830, 2910] MHz coverage and must be flagged incompatible
    narrow_ghz = np.linspace(2.850, 2.890, 201)
    narrow_spec = np.ones_like(narrow_ghz) - 0.02 * np.exp(-((narrow_ghz - 2.870) / 0.005) ** 2)
    rec_narrow = load_and_validate_real_trace(
        narrow_ghz,
        narrow_spec,
        trace_id="fixture_narrow_ghz",
        source_name="Unit Test Fixture",
        license_id="CC-BY-4.0",
        citation="Synthetic test",
        nv_type="single NV",
    )
    assert rec_narrow["original_frequency_unit"] == "GHz"
    assert rec_narrow["compatible_central_2830_2910"] is False
    assert rec_narrow["resampled_spectrum"] is None
    assert rec_narrow["valid_mask"] is None

    # Test CSV and NPZ fixture loader on a valid 4-axis resolved trace covering [2760, 2980] MHz
    valid_freqs_ghz = np.linspace(2.760, 2.980, 400)
    clean_valid = simulate_clean_odmr(
        valid_freqs_ghz * 1e3,
        ODMRParams(
            b_field_mt=(1.80, 1.10, 0.35),
            base_contrast=0.05,
            linewidth_mhz=2.5,
            include_hyperfine=False,
        ),
    )["spectrum"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "fixture_trace.csv"
        np.savetxt(
            csv_path,
            np.column_stack([valid_freqs_ghz, clean_valid]),
            delimiter=",",
            header="Freq [GHz],Normalized readout",
            comments="",
        )
        rec_csv = load_csv_or_npz_odmr_file(csv_path, nv_type="ensemble")
        assert rec_csv["compatible_central_2830_2910"] is True
        assert rec_csv["resampled_spectrum"] is not None
        assert rec_csv["resampled_spectrum"].shape == (N_POINTS,)
        assert rec_csv["qa_checks_supported"]["pair_centre_symmetry_about_d0"] is True

        npz_path = Path(tmp_dir) / "fixture_trace.npz"
        np.savez(npz_path, freqs=valid_freqs_ghz * 1e9, spectrum=1.0 - clean_valid)
        rec_npz = load_csv_or_npz_odmr_file(
            npz_path, nv_type="ensemble", spectrum_convention="positive_contrast"
        )
        assert rec_npz["original_frequency_unit"] == "Hz"
        assert rec_npz["compatible_central_2830_2910"] is True
        assert np.allclose(rec_csv["resampled_spectrum"], rec_npz["resampled_spectrum"], atol=1e-5)

    assert DEFAULT_REAL_MANIFEST_PATH.exists()


def test_stage21_val_id_natural_and_ood_high_field_splits() -> None:
    """Stage 2.1 Task 2 & Task 3.1: Verify val_id_natural and ood_high_field properties and dip bounds."""
    from src.config import NV_AXES
    from src.physics import solve_axis_transitions

    d_nat = load_synthetic_split("val_id_natural")
    assert d_nat["X_noisy"].shape == (1500, N_POINTS)
    reg_nat = d_nat["aux_regime_id"]
    frac_hard = float(np.mean(reg_nat == REGIME_HARD))
    frac_med = float(np.mean(reg_nat == REGIME_MEDIUM))
    frac_easy = float(np.mean(reg_nat == REGIME_EASY))
    assert 0.24 <= frac_hard <= 0.33, f"Unexpected natural hard fraction: {frac_hard}"
    assert 0.16 <= frac_med <= 0.24, f"Unexpected natural medium fraction: {frac_med}"
    assert 0.46 <= frac_easy <= 0.56, f"Unexpected natural easy fraction: {frac_easy}"
    assert float(np.min(d_nat["Y_b_mag"])) >= B_MAG_MIN_MT
    assert float(np.max(d_nat["Y_b_mag"])) <= B_MAG_MAX_MT

    d_hf = load_synthetic_split("ood_high_field")
    assert d_hf["X_noisy"].shape == (1000, N_POINTS)
    assert float(np.min(d_hf["Y_b_mag"])) >= 3.25 - 1e-4
    assert float(np.max(d_hf["Y_b_mag"])) <= 4.60 + 1e-4

    # Verify all 8 dip centers for every sample in ood_high_field fall strictly within [2760, 2980] MHz
    b_vecs = d_hf["aux_b_vec_canon"].astype(np.float64)
    d0_arr = (D0_MHZ + d_hf["Y_d0"].astype(np.float64))[:, None]
    b_par = np.abs(b_vecs @ NV_AXES.T)
    b_mag_sq = np.sum(b_vecs * b_vecs, axis=1, keepdims=True)
    b_perp = np.sqrt(np.maximum(0.0, b_mag_sq - b_par * b_par))
    fm, fp, _ = solve_axis_transitions(b_par, b_perp, d0_arr)
    assert float(np.min(fm)) >= 2760.0 - 1e-3, f"Min dip {float(np.min(fm))} < 2760 MHz"
    assert float(np.max(fp)) <= 2980.0 + 1e-3, f"Max dip {float(np.max(fp))} > 2980 MHz"


def test_stage21_per_axis_contrast_jitter_and_stage2v1_hash_preservation() -> None:
    """Stage 2.1 Task 2.2: Verify per_axis_contrast_jitter behavior and bit-identical v1 hashes when disabled."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # All 12 Stage 2 v1 split hashes must remain bit-identical
    expected_v1_hashes = {
        "train_snapshot": "37761c3d40349ae3c0336f208074be2fc33035b1420642f2a2b38b28af4791e8",
        "val_id": "294efabc236b362340078d789282654085915e3bb68648f6165573c4f41a48cc",
        "val_ood": "014df8d74cab853b09771a114750db2fce5afd194c48e96d3d4098d3e4026dfa",
        "ood_hwhm": "643cb17ffd93f0d41fe45fac48dfc88a3e404822846f611a075ef910a4838b19",
        "ood_contrast": "9396e1f3efc8b3ee7fb264c0564f4cf0c7ebbd127389b787f226a52f8b4e30a6",
        "ood_d0": "501e98151a698d6e97cf012828d65e99328ce2b46975a1a6f108367304b2645a",
        "ood_noise": "9c6bbd457165dca0aa73d7c5a190461047d2b62a850ecd3484f59a2c24735c8b",
        "ood_drift": "c75b7c630e7ffe975c23f397708ea4c56af1408af6017c50eaf4d55f058181c4",
        "ood_hum": "9b57ed282f1a5320da09dd0373d145a118e7660a3f76f72b7479e5b7be67c340",
        "ood_grid": "f7998fbb6d7da6fdba218b0fb0b03838d3fca4371a68fac4648bc08fd6dad0c7",
        "test_id": "819e007a1a3fbc45d52df4334fd0ce6906f7e59e31cd22204b023383d63cbf79",
        "test_ood": "fcdce8033fb0d04cfd5cb7c5ea92b1b9ee7fbfd457556bba00c8a3e02ce627e7",
    }
    for sname, exp_sha in expected_v1_hashes.items():
        assert manifest["splits"][sname]["sha256"] == exp_sha, f"Hash changed for {sname}!"

    # Test per_axis_contrast_jitter > 0 produces normalized non-uniform axis weights without affecting Y_splittings
    cfg_no_jit = SplitConfig(
        name="tmp_no_jit",
        split_seed=20269991,
        n_samples=10,
        stratified_regimes=False,
        factor_ablation_id=0,
        per_axis_contrast_jitter=0.0,
    )
    cfg_jit = SplitConfig(
        name="tmp_jit",
        split_seed=20269991,
        n_samples=10,
        stratified_regimes=False,
        factor_ablation_id=0,
        per_axis_contrast_jitter=0.15,
    )
    s0 = generate_single_sample(0, cfg_no_jit)
    s1 = generate_single_sample(0, cfg_jit)
    assert np.allclose(s0["aux_axis_weights"], np.ones(4, dtype=np.float32))
    assert float(s0["aux_axis_asym_ratio"]) == 1.0
    assert np.isclose(float(np.mean(s1["aux_axis_weights"])), 1.0, atol=1e-6)
    assert float(np.std(s1["aux_axis_weights"])) > 0.02
    assert float(np.max(s1["aux_axis_weights"]) / np.min(s1["aux_axis_weights"])) > 1.05
    assert np.array_equal(s0["Y_splittings"], s1["Y_splittings"])
    assert s0["Y_b_mag"] == s1["Y_b_mag"]
    assert not np.allclose(s0["Y_clean"], s1["Y_clean"])


def test_stage21_unique_12_example_selection_and_artifacts() -> None:
    """Stage 2.1 Task 1, Task 3.2 & Task 4: Verify 12 unique example indices and generated Stage 2.1 artifacts."""
    for sname in ("val_id", "val_ood"):
        d = load_synthetic_split(sname)
        selected = select_12_unique_example_indices(d)
        indices = [idx for idx, _ in selected]
        assert len(indices) == 12
        assert len(set(indices)) == 12, f"Duplicate indices in {sname}: {indices}"

    assert (ROOT_DIR / "docs" / "trianam_sim_comparison.md").exists()
    assert (ROOT_DIR / "assets" / "stage2_trianam_vs_sim.png").exists()
    assert (ROOT_DIR / "assets" / "stage2_worst_cases.png").exists()

    val_ref_path = DEFAULT_SYNTHETIC_DIR / "val_baseline_ref.npz"
    assert val_ref_path.exists()
    with np.load(val_ref_path, allow_pickle=False) as ref:
        assert ref["val_id_abs_err_ut"].shape == (3000,)
        assert ref["val_id_natural_abs_err_ut"].shape == (1500,)
        assert ref["val_ood_abs_err_ut"].shape == (1500,)
        assert ref["ood_high_field_abs_err_ut"].shape == (100,)
        assert float(np.mean(ref["ood_high_field_wrong_basin"])) >= 0.90
        assert float(np.mean(ref["ood_high_field_out_of_range"])) >= 0.30


def test_stage22_train_sim2real_and_ood_asym_invariants() -> None:
    """Stage 2.2 Tasks 1–4: Verify train_sim2real, ood_asym.npz, Y_clean target semantics, and v1/v2 hash preservation."""
    from src.noise_models import fast_simulate_single_clean_odmr

    # 1. Verify all existing stage2-data-v1 and stage2-data-v2 SHA256 hashes remain bit-identical
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected_v1_v2_hashes = {
        "train_snapshot": "37761c3d40349ae3c0336f208074be2fc33035b1420642f2a2b38b28af4791e8",
        "val_id": "294efabc236b362340078d789282654085915e3bb68648f6165573c4f41a48cc",
        "val_id_natural": "20c84af464043f344c6b4518580ca2c81c8f8d63f018c6f01a610a242fc391ad",
        "val_ood": "014df8d74cab853b09771a114750db2fce5afd194c48e96d3d4098d3e4026dfa",
        "ood_hwhm": "643cb17ffd93f0d41fe45fac48dfc88a3e404822846f611a075ef910a4838b19",
        "ood_contrast": "9396e1f3efc8b3ee7fb264c0564f4cf0c7ebbd127389b787f226a52f8b4e30a6",
        "ood_d0": "501e98151a698d6e97cf012828d65e99328ce2b46975a1a6f108367304b2645a",
        "ood_noise": "9c6bbd457165dca0aa73d7c5a190461047d2b62a850ecd3484f59a2c24735c8b",
        "ood_drift": "c75b7c630e7ffe975c23f397708ea4c56af1408af6017c50eaf4d55f058181c4",
        "ood_hum": "9b57ed282f1a5320da09dd0373d145a118e7660a3f76f72b7479e5b7be67c340",
        "ood_grid": "f7998fbb6d7da6fdba218b0fb0b03838d3fca4371a68fac4648bc08fd6dad0c7",
        "ood_high_field": "432b6e668a4fefe4c1a93cdb0aad3a115fabb63de0d8f07ce0a50c9f20614576",
        "test_id": "819e007a1a3fbc45d52df4334fd0ce6906f7e59e31cd22204b023383d63cbf79",
        "test_ood": "fcdce8033fb0d04cfd5cb7c5ea92b1b9ee7fbfd457556bba00c8a3e02ce627e7",
    }
    for sname, exp_sha in expected_v1_v2_hashes.items():
        assert manifest["splits"][sname]["sha256"] == exp_sha, f"Hash changed for {sname}!"

    assert (
        manifest["reference_artifacts"]["val_baseline_ref"]["sha256"]
        == "906b78e739b17e74e016e89e2b9a66c98d558186238e142e14f99d8737e087a8"
    )
    assert get_test_access_counts(TEST_ACCESS_LOG_PATH)["data_read"] == 0

    # 2. Verify backward-compatible defaults on older splits loaded via load_synthetic_split
    d_vid = load_synthetic_split("val_id")
    assert d_vid["aux_axis_weights"].shape == (4000, 4)
    assert np.allclose(d_vid["aux_axis_weights"], 1.0)
    assert np.allclose(d_vid["aux_axis_asym_ratio"], 1.0)
    assert np.allclose(d_vid["aux_mw_power_tilt_amp"], 0.0)

    # 3. Verify train_sim2real preset invariants on 250 samples
    cfg_s2r = get_stage3_sim2real_train_config(allow_high_field_in_window=False)
    s2r_batch = generate_split_arrays(cfg_s2r, n_samples=250, n_workers=1)
    w_s2r = s2r_batch["aux_axis_weights"].astype(np.float64)
    r_s2r = s2r_batch["aux_axis_asym_ratio"].astype(np.float64)
    hw_s2r = s2r_batch["Y_hwhm"].astype(np.float64)
    tilt_s2r = s2r_batch["aux_mw_power_tilt_amp"].astype(np.float64)

    assert float(np.max(np.abs(np.mean(w_s2r, axis=1) - 1.0))) < 1e-6
    assert float(np.min(r_s2r)) >= 1.0 - 1e-6 and float(np.max(r_s2r)) <= 2.0 + 1e-6
    assert 0.12 <= float(np.mean(np.isclose(r_s2r, 1.0, atol=1e-5))) <= 0.30
    assert 0.60 <= float(np.mean(np.isclose(tilt_s2r, 0.12, atol=1e-5))) <= 0.80
    assert float(np.min(hw_s2r)) >= 2.0 and float(np.max(hw_s2r)) <= 9.0
    assert bool(np.any(hw_s2r < 3.0)), "Expected samples with HWHM < 3.0 MHz in train_sim2real"

    # Strict OOD exclusion checks on train_sim2real
    fams_s2r = set(np.unique(s2r_batch["aux_drift_family_id"]).tolist())
    assert fams_s2r.issubset(set(ID_DRIFT_FAMILIES))
    assert DRIFT_F3_EXP_OOD not in fams_s2r and DRIFT_F5_OOD not in fams_s2r
    assert bool(np.all(s2r_batch["aux_noise_type"] == NOISE_GAUSSIAN_ID))
    assert float(np.min(s2r_batch["Y_d0"])) >= -3.0 and float(np.max(s2r_batch["Y_d0"])) <= 3.0
    assert float(np.min(s2r_batch["aux_contrast"])) >= 0.015 and float(np.max(s2r_batch["aux_contrast"])) <= 0.040
    assert float(np.max(s2r_batch["aux_hum_rel_harmonics"])) <= 0.50 + 1e-6

    # 4. Verify ood_asym.npz properties, Y_clean target semantics, and float64 additive identity
    d_asym = load_synthetic_split("ood_asym")
    assert d_asym["X_noisy"].shape == (1000, N_POINTS)
    w_asym = d_asym["aux_axis_weights"].astype(np.float64)
    r_asym = d_asym["aux_axis_asym_ratio"].astype(np.float64)
    hw_asym = d_asym["Y_hwhm"].astype(np.float64)
    assert float(np.max(np.abs(np.mean(w_asym, axis=1) - 1.0))) < 1e-6
    assert float(np.min(r_asym)) >= 1.35 - 1e-5 and float(np.max(r_asym)) <= 2.20 + 1e-5
    assert np.allclose(np.max(w_asym, axis=1) / np.min(w_asym, axis=1), r_asym, atol=1e-5)
    assert float(np.min(hw_asym)) >= 2.0 and float(np.max(hw_asym)) <= 9.0
    assert np.allclose(d_asym["aux_mw_power_tilt_amp"], 0.12, atol=1e-5)

    for idx in (0, 5, 17, 42, 128):
        comps = regenerate_sample_components("ood_asym", idx)
        total_f64 = (
            comps["Y_clean"]
            + comps["drift"]
            + comps["hum"]
            + comps["pink"]
            + comps["white"]
            + comps["other"]
        )
        assert float(np.max(np.abs(total_f64 - comps["X_noisy_f64"]))) < 1e-12
        assert float(np.max(np.abs(total_f64.astype(np.float32) - d_asym["X_noisy"][idx]))) < 1e-6

        # Prove Y_clean includes both per-axis weights w and RF power tilt m(f) (differs from equal-axis clean)
        eq_clean, *_ = fast_simulate_single_clean_odmr(
            CANONICAL_FREQS,
            d_asym["aux_b_vec_canon"][idx].astype(np.float64),
            D0_MHZ + float(d_asym["Y_d0"][idx]),
            float(d_asym["Y_hwhm"][idx]),
            float(d_asym["aux_contrast"][idx]),
            float(d_asym["aux_eta"][idx]),
            e_mhz=0.0,
            axis_weights=None,
            mw_tilt_coeffs=None,
        )
        asym_clean, *_ = fast_simulate_single_clean_odmr(
            CANONICAL_FREQS,
            d_asym["aux_b_vec_canon"][idx].astype(np.float64),
            D0_MHZ + float(d_asym["Y_d0"][idx]),
            float(d_asym["Y_hwhm"][idx]),
            float(d_asym["aux_contrast"][idx]),
            float(d_asym["aux_eta"][idx]),
            e_mhz=0.0,
            axis_weights=d_asym["aux_axis_weights"][idx].astype(np.float64),
            mw_tilt_coeffs=d_asym["aux_mw_power_tilt_coeffs"][idx].astype(np.float64),
        )
        assert float(np.max(np.abs(comps["Y_clean"] - asym_clean))) < 1e-12
        assert float(np.max(np.abs(comps["Y_clean"] - eq_clean))) > 1e-4

    assert (DEFAULT_SYNTHETIC_DIR / "ood_asym_baseline_ref.npz").exists()
    assert (ROOT_DIR / "assets" / "stage2_ood_asym_examples.png").exists()


