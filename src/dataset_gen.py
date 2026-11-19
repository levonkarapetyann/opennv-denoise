"""Stage 2 deterministic synthetic ODMR dataset generator, split protocol, and locked test-set loader."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.config import (
    B_MAG_MAX_MT,
    B_MAG_MIN_MT,
    CANONICAL_FREQS,
    D0_MHZ,
    N_POINTS,
    NV_AXES,
    OOD_LINEWIDTH_RANGE,
    TRAIN_LINEWIDTH_RANGE,
)
from src.noise import resample_spectrum_pchip
from src.noise_models import (
    DRIFT_F1_POLY,
    DRIFT_F2_SINE,
    DRIFT_F3_EXP_OOD,
    DRIFT_F4_GP,
    DRIFT_F5_ID,
    DRIFT_F5_OOD,
    GRID_CANONICAL,
    ID_DRIFT_FAMILIES,
    NATURAL_REGIME_PROBS,
    NOISE_GAUSSIAN_ID,
    NOISE_STUDENT_T_SPIKES_OOD,
    OOD_DRIFT_FAMILIES,
    REGIME_BOUNDS,
    REGIME_EASY,
    REGIME_HARD,
    REGIME_MEDIUM,
    compute_depth_outer_and_flags,
    evaluate_transitions_on_grid,
    fast_simulate_single_clean_odmr,
    generate_drift_component,
    generate_hum_component,
    generate_simulated_grid,
    generate_stochastic_and_gain_components,
    regime_id_from_eff_snr_white,
    sample_axis_weights_with_target_ratio,
    sample_mw_power_tilt_coeffs,
)
from src.physics import b_mag_from_splittings, canonicalize_b_vec, simulate_batch_clean_odmr

DEFAULT_SYNTHETIC_DIR: Path = Path("data") / "synthetic"
TEST_ACCESS_LOG_PATH: Path = DEFAULT_SYNTHETIC_DIR / "test_access.log"
MANIFEST_PATH: Path = DEFAULT_SYNTHETIC_DIR / "manifest.json"

# Factor-ablation identifiers (Part 3)
FACTOR_ID_STANDARD: int = 0
FACTOR_OOD_HWHM: int = 1
FACTOR_OOD_CONTRAST: int = 2
FACTOR_OOD_D0: int = 3
FACTOR_OOD_NOISE: int = 4
FACTOR_OOD_DRIFT: int = 5
FACTOR_OOD_HUM: int = 6
FACTOR_OOD_GRID: int = 7
FACTOR_OOD_COMBINED: int = 8
FACTOR_OOD_HIGH_FIELD: int = 9
FACTOR_OOD_ASYM: int = 10

FACTOR_ABLATION_NAMES: Dict[int, str] = {
    FACTOR_ID_STANDARD: "id_standard",
    FACTOR_OOD_HWHM: "ood_hwhm",
    FACTOR_OOD_CONTRAST: "ood_contrast",
    FACTOR_OOD_D0: "ood_d0",
    FACTOR_OOD_NOISE: "ood_noise",
    FACTOR_OOD_DRIFT: "ood_drift",
    FACTOR_OOD_HUM: "ood_hum",
    FACTOR_OOD_GRID: "ood_grid",
    FACTOR_OOD_COMBINED: "ood_combined",
    FACTOR_OOD_HIGH_FIELD: "ood_high_field",
    FACTOR_OOD_ASYM: "ood_asym",
}

LOCKED_TEST_SPLITS: Tuple[str, ...] = ("test_id", "test_ood")


@dataclass(frozen=True)
class SplitConfig:
    """Complete deterministic specification for a Stage 2 / 2.1 / 2.2 dataset split."""

    name: str
    split_seed: int
    n_samples: int
    stratified_regimes: bool
    factor_ablation_id: int
    ood_hwhm: bool = False
    ood_contrast: bool = False
    ood_d0: bool = False
    ood_noise: bool = False
    ood_drift: bool = False
    ood_hum: bool = False
    ood_grid_aggressive: bool = False
    grid_perturb_prob: float = 0.30
    is_locked_test: bool = False
    allow_high_field_in_window: bool = False
    b_mag_range_mt: Tuple[float, float] = (B_MAG_MIN_MT, B_MAG_MAX_MT)
    in_window_freq_bounds_mhz: Tuple[float, float] = (2760.0, 2980.0)
    per_axis_contrast_jitter: float = 0.0
    hwhm_range_mhz: Optional[Tuple[float, float]] = None
    axis_asym_range: Optional[Tuple[float, float]] = None
    axis_equal_prob: float = 0.0
    mw_power_tilt_amp: float = 0.0
    mw_power_tilt_prob: float = 1.0


def get_stage3_sim2real_train_config(allow_high_field_in_window: bool = False) -> SplitConfig:
    """Return the Stage 3 Sim-to-Real on-the-fly training preset (``train_sim2real``).

    Implements Stage 2.2 Task 2:
      - ``hwhm_range_mhz = (2.0, 9.0)`` MHz (covers real ``trianam`` ``2.75–2.96 MHz``).
      - ``axis_asym_range = (1.0, 2.0)`` with ``axis_equal_prob = 0.20`` (20% exact equal-axis
        ``1.0x`` + 80% uniform max/min ratio in ``[1.0, 2.0]``).
      - ``mw_power_tilt_amp = 0.12`` with ``mw_power_tilt_prob = 0.70`` (applied with prob 0.7).
      - All OOD exclusions strictly preserved (ID drift families ONLY, Gaussian white/pink noise ONLY,
        ``d0_offset`` in ``[-3, +3]`` MHz, ``contrast`` in ``[0.015, 0.040]``, ``hum_harmonics <= 0.50``).
    """
    b_range = (B_MAG_MIN_MT, 4.60) if allow_high_field_in_window else (B_MAG_MIN_MT, B_MAG_MAX_MT)
    return SplitConfig(
        name="train_sim2real",
        split_seed=20260103,
        n_samples=0,
        stratified_regimes=False,
        factor_ablation_id=FACTOR_ID_STANDARD,
        ood_hwhm=False,
        ood_contrast=False,
        ood_d0=False,
        ood_noise=False,
        ood_drift=False,
        ood_hum=False,
        ood_grid_aggressive=False,
        grid_perturb_prob=0.30,
        allow_high_field_in_window=bool(allow_high_field_in_window),
        b_mag_range_mt=b_range,
        in_window_freq_bounds_mhz=(2760.0, 2980.0),
        hwhm_range_mhz=(2.0, 9.0),
        axis_asym_range=(1.0, 2.0),
        axis_equal_prob=0.20,
        mw_power_tilt_amp=0.12,
        mw_power_tilt_prob=0.70,
    )


SPLIT_REGISTRY: Dict[str, SplitConfig] = {
    "train": SplitConfig(
        name="train",
        split_seed=20260101,
        n_samples=0,  # Infinite on-the-fly stream
        stratified_regimes=False,
        factor_ablation_id=FACTOR_ID_STANDARD,
        grid_perturb_prob=0.30,
    ),
    "train_sim2real": get_stage3_sim2real_train_config(allow_high_field_in_window=False),
    "train_snapshot": SplitConfig(
        name="train_snapshot",
        split_seed=20260102,
        n_samples=20000,
        stratified_regimes=False,
        factor_ablation_id=FACTOR_ID_STANDARD,
        grid_perturb_prob=0.30,
    ),
    "val_id": SplitConfig(
        name="val_id",
        split_seed=20260201,
        n_samples=4000,
        stratified_regimes=True,
        factor_ablation_id=FACTOR_ID_STANDARD,
        grid_perturb_prob=0.30,
    ),
    "val_id_natural": SplitConfig(
        name="val_id_natural",
        split_seed=20260203,
        n_samples=1500,
        stratified_regimes=False,
        factor_ablation_id=FACTOR_ID_STANDARD,
        grid_perturb_prob=0.30,
    ),
    "val_ood": SplitConfig(
        name="val_ood",
        split_seed=20260202,
        n_samples=4000,
        stratified_regimes=True,
        factor_ablation_id=FACTOR_OOD_COMBINED,
        ood_hwhm=True,
        ood_contrast=True,
        ood_d0=True,
        ood_noise=True,
        ood_drift=True,
        ood_hum=True,
        ood_grid_aggressive=False,
        grid_perturb_prob=0.40,
    ),
    "ood_hwhm": SplitConfig(
        name="ood_hwhm",
        split_seed=20260301,
        n_samples=1000,
        stratified_regimes=False,
        factor_ablation_id=FACTOR_OOD_HWHM,
        ood_hwhm=True,
        grid_perturb_prob=0.30,
    ),
    "ood_contrast": SplitConfig(
        name="ood_contrast",
        split_seed=20260302,
        n_samples=1000,
        stratified_regimes=False,
        factor_ablation_id=FACTOR_OOD_CONTRAST,
        ood_contrast=True,
        grid_perturb_prob=0.30,
    ),
    "ood_d0": SplitConfig(
        name="ood_d0",
        split_seed=20260303,
        n_samples=1000,
        stratified_regimes=False,
        factor_ablation_id=FACTOR_OOD_D0,
        ood_d0=True,
        grid_perturb_prob=0.30,
    ),
    "ood_noise": SplitConfig(
        name="ood_noise",
        split_seed=20260304,
        n_samples=1000,
        stratified_regimes=False,
        factor_ablation_id=FACTOR_OOD_NOISE,
        ood_noise=True,
        grid_perturb_prob=0.30,
    ),
    "ood_drift": SplitConfig(
        name="ood_drift",
        split_seed=20260305,
        n_samples=1000,
        stratified_regimes=False,
        factor_ablation_id=FACTOR_OOD_DRIFT,
        ood_drift=True,
        grid_perturb_prob=0.30,
    ),
    "ood_hum": SplitConfig(
        name="ood_hum",
        split_seed=20260306,
        n_samples=1000,
        stratified_regimes=False,
        factor_ablation_id=FACTOR_OOD_HUM,
        ood_hum=True,
        grid_perturb_prob=0.30,
    ),
    "ood_grid": SplitConfig(
        name="ood_grid",
        split_seed=20260307,
        n_samples=1000,
        stratified_regimes=False,
        factor_ablation_id=FACTOR_OOD_GRID,
        ood_grid_aggressive=True,
        grid_perturb_prob=1.00,
    ),
    "ood_high_field": SplitConfig(
        name="ood_high_field",
        split_seed=20260308,
        n_samples=1000,
        stratified_regimes=False,
        factor_ablation_id=FACTOR_OOD_HIGH_FIELD,
        grid_perturb_prob=0.30,
        allow_high_field_in_window=True,
        b_mag_range_mt=(3.25, 4.60),
        in_window_freq_bounds_mhz=(2760.0, 2980.0),
    ),
    "ood_asym": SplitConfig(
        name="ood_asym",
        split_seed=20260309,
        n_samples=1000,
        stratified_regimes=False,
        factor_ablation_id=FACTOR_OOD_ASYM,
        grid_perturb_prob=0.30,
        b_mag_range_mt=(B_MAG_MIN_MT, B_MAG_MAX_MT),
        hwhm_range_mhz=(2.0, 9.0),
        axis_asym_range=(1.35, 2.20),
        axis_equal_prob=0.0,
        mw_power_tilt_amp=0.12,
        mw_power_tilt_prob=1.0,
    ),
    "test_id": SplitConfig(
        name="test_id",
        split_seed=20260901,
        n_samples=4000,
        stratified_regimes=True,
        factor_ablation_id=FACTOR_ID_STANDARD,
        grid_perturb_prob=0.30,
        is_locked_test=True,
    ),
    "test_ood": SplitConfig(
        name="test_ood",
        split_seed=20260902,
        n_samples=4000,
        stratified_regimes=True,
        factor_ablation_id=FACTOR_OOD_COMBINED,
        ood_hwhm=True,
        ood_contrast=True,
        ood_d0=True,
        ood_noise=True,
        ood_drift=True,
        ood_hum=True,
        ood_grid_aggressive=False,
        grid_perturb_prob=0.40,
        is_locked_test=True,
    ),
}

# Precompute orthogonal basis pairs (e1, e2) perpendicular to each of the 4 NV_AXES
_AXIS_ORTHO_BASIS: np.ndarray = np.zeros((4, 2, 3), dtype=np.float64)
for _k in range(4):
    _n = NV_AXES[_k]
    _ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    _e1 = np.cross(_n, _ref)
    _e1 /= np.linalg.norm(_e1)
    _e2 = np.cross(_n, _e1)
    _e2 /= np.linalg.norm(_e2)
    _AXIS_ORTHO_BASIS[_k, 0] = _e1
    _AXIS_ORTHO_BASIS[_k, 1] = _e2


def derive_sample_seed(split_seed: int, sample_idx: int) -> int:
    """Derive a collision-free 64-bit unsigned integer seed via ``SHA256(split_seed, sample_idx)``.

    Guarantees that sample ``i`` is regenerable independently of batch size,
    worker count, or execution order.
    """
    payload = struct.pack(
        "<QQ",
        int(split_seed) & 0xFFFFFFFFFFFFFFFF,
        int(sample_idx) & 0xFFFFFFFFFFFFFFFF,
    )
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def target_regime_for_index(sample_idx: int, n_samples: int, stratified: bool) -> Optional[int]:
    """Return the stratified regime id (0=Hard, 1=Medium, 2=Easy) for ``sample_idx`` or None."""
    if not stratified:
        return None
    if n_samples == 4000:
        n_hard, n_med = 1333, 1333
    else:
        n_hard = n_samples // 3
        n_med = n_samples // 3
    if sample_idx < n_hard:
        return REGIME_HARD
    if sample_idx < n_hard + n_med:
        return REGIME_MEDIUM
    return REGIME_EASY


def compute_stratified_importance_weights(regime_ids: np.ndarray) -> np.ndarray:
    """Compute per-sample importance weights restoring natural log-uniform train regime proportions."""
    r_arr = np.asarray(regime_ids, dtype=np.int64)
    n_total = max(r_arr.size, 1)
    weights = np.ones(n_total, dtype=np.float64)
    for r_id, p_nat in NATURAL_REGIME_PROBS.items():
        mask = r_arr == r_id
        count = int(np.sum(mask))
        if count > 0:
            p_emp = float(count) / float(n_total)
            weights[mask] = p_nat / p_emp
    return weights / max(float(np.mean(weights)), 1e-12)


def _sample_direction_s2(rng: np.random.Generator) -> Tuple[np.ndarray, bool, float]:
    """Sample a 3D unit vector on ``S^2`` with 20% in the near-degenerate belt ``min_k |cos(theta_k)| < 0.05``."""
    is_near_deg = bool(float(rng.random()) < 0.20)
    if is_near_deg:
        axis_idx = int(rng.integers(0, 4))
        cos_theta = float(rng.uniform(-0.0499, 0.0499))
        sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)))
        phi = float(rng.uniform(0.0, 2.0 * np.pi))
        n_vec = NV_AXES[axis_idx]
        e1 = _AXIS_ORTHO_BASIS[axis_idx, 0]
        e2 = _AXIS_ORTHO_BASIS[axis_idx, 1]
        u_hat = cos_theta * n_vec + sin_theta * (np.cos(phi) * e1 + np.sin(phi) * e2)
        u_hat /= np.linalg.norm(u_hat)
    else:
        raw = rng.standard_normal(3)
        norm = float(np.linalg.norm(raw))
        u_hat = (raw / norm) if norm > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

    min_abs_cos = float(np.min(np.abs(NV_AXES @ u_hat)))
    return u_hat, is_near_deg, min_abs_cos


def generate_single_sample(
    sample_idx: int,
    split_cfg: SplitConfig,
    return_f64_components: bool = False,
) -> Dict[str, Any]:
    """Generate a single deterministic ODMR sample for ``(split_cfg, sample_idx)``.

    All random draws derive exclusively from ``sample_seed = derive_sample_seed(split_cfg.split_seed, sample_idx)``.
    """
    sample_seed = derive_sample_seed(split_cfg.split_seed, sample_idx)
    rng = np.random.default_rng(sample_seed)
    target_regime = target_regime_for_index(
        sample_idx, split_cfg.n_samples, split_cfg.stratified_regimes
    )

    # 1. Physical parameters (quantized to float32 representation so stored float32 params
    #    in .npz files reproduce float64 Y_clean and Y_splittings to < 1e-14 without f32->f64 drift)
    if split_cfg.hwhm_range_mhz is not None:
        hw_lo, hw_hi = float(split_cfg.hwhm_range_mhz[0]), float(split_cfg.hwhm_range_mhz[1])
        hwhm_mhz = float(
            np.clip(
                np.float32(rng.uniform(hw_lo, hw_hi)),
                hw_lo,
                hw_hi,
            )
        )
    elif split_cfg.ood_hwhm:
        hwhm_mhz = float(
            np.clip(
                np.float32(rng.uniform(OOD_LINEWIDTH_RANGE[0], OOD_LINEWIDTH_RANGE[1])),
                OOD_LINEWIDTH_RANGE[0],
                OOD_LINEWIDTH_RANGE[1],
            )
        )
    else:
        hwhm_mhz = float(
            np.clip(
                np.float32(rng.uniform(TRAIN_LINEWIDTH_RANGE[0], TRAIN_LINEWIDTH_RANGE[1])),
                TRAIN_LINEWIDTH_RANGE[0],
                TRAIN_LINEWIDTH_RANGE[1],
            )
        )

    eta = float(np.clip(np.float32(rng.uniform(0.0, 1.0)), 0.0, 1.0))

    if split_cfg.ood_contrast:
        # [0.010, 0.0145] U [0.0405, 0.050] weighted by interval lengths (0.0045 vs 0.0095)
        if float(rng.random()) < (0.0045 / (0.0045 + 0.0095)):
            contrast = float(np.clip(np.float32(rng.uniform(0.010, 0.0145)), 0.010, 0.0145))
        else:
            contrast = float(np.clip(np.float32(rng.uniform(0.0405, 0.050)), 0.0405, 0.050))
    else:
        contrast = float(np.clip(np.float32(rng.uniform(0.015, 0.040)), 0.015, 0.040))

    if split_cfg.ood_d0:
        if float(rng.random()) < 0.5:
            d0_offset_mhz = float(np.clip(np.float32(rng.uniform(-5.0, -3.1)), -5.0, -3.1))
        else:
            d0_offset_mhz = float(np.clip(np.float32(rng.uniform(3.1, 5.0)), 3.1, 5.0))
    else:
        d0_offset_mhz = float(np.clip(np.float32(rng.uniform(-3.0, 3.0)), -3.0, 3.0))
    d0_mhz = D0_MHZ + d0_offset_mhz

    e_mhz = 0.0
    axis_weights: Optional[np.ndarray] = None
    axis_weights_f32: np.ndarray = np.ones(4, dtype=np.float32)
    axis_asym_ratio: float = 1.0
    mw_tilt_coeffs: Optional[np.ndarray] = None
    mw_tilt_coeffs_f32: np.ndarray = np.zeros(2, dtype=np.float32)
    realized_mw_tilt_amp: float = 0.0

    if (
        split_cfg.axis_asym_range is not None
        or split_cfg.per_axis_contrast_jitter > 0.0
        or split_cfg.mw_power_tilt_amp > 0.0
    ):
        rng_sim2real = np.random.default_rng(int(sample_seed) ^ 0xA5157E1687501234)
        if split_cfg.axis_asym_range is not None:
            axis_weights, axis_asym_ratio = sample_axis_weights_with_target_ratio(
                rng_sim2real,
                split_cfg.axis_asym_range,
                equal_prob=split_cfg.axis_equal_prob,
            )
            axis_weights_f32 = axis_weights.astype(np.float32)
        elif split_cfg.per_axis_contrast_jitter > 0.0:
            raw_w = np.exp(rng_sim2real.normal(0.0, float(split_cfg.per_axis_contrast_jitter), size=4))
            axis_weights_f32 = (raw_w / np.mean(raw_w)).astype(np.float32)
            axis_weights_f32 = (axis_weights_f32 / np.float32(np.mean(axis_weights_f32))).astype(np.float32)
            axis_weights = axis_weights_f32.astype(np.float64)
            axis_asym_ratio = float(np.max(axis_weights_f32) / np.min(axis_weights_f32))

        if split_cfg.mw_power_tilt_amp > 0.0:
            mw_tilt_coeffs, realized_mw_tilt_amp = sample_mw_power_tilt_coeffs(
                rng_sim2real,
                split_cfg.mw_power_tilt_amp,
                mw_power_tilt_prob=split_cfg.mw_power_tilt_prob,
            )
            mw_tilt_coeffs_f32 = mw_tilt_coeffs.astype(np.float32)

    if not split_cfg.allow_high_field_in_window:
        # |B| in [0.05, 3.20] mT (50% log-uniform, 50% uniform) — exact Stage 2 v1 RNG sequence
        if float(rng.random()) < 0.5:
            b_mag_draw = float(np.exp(rng.uniform(np.log(B_MAG_MIN_MT), np.log(B_MAG_MAX_MT))))
        else:
            b_mag_draw = float(rng.uniform(B_MAG_MIN_MT, B_MAG_MAX_MT))
        b_mag_draw = float(np.clip(b_mag_draw, B_MAG_MIN_MT + 1e-5, B_MAG_MAX_MT - 1e-5))

        u_hat, sampler_near_deg, min_abs_cos = _sample_direction_s2(rng)
        b_vec = (b_mag_draw * u_hat).astype(np.float32).astype(np.float64)
        b_mag = float(np.linalg.norm(b_vec))
        b_vec_canon = canonicalize_b_vec(b_vec, e_mhz=e_mhz)

        # 2. Simulate clean canonical spectrum & compute depth_outer / flags in one pass
        y_clean_f64, splittings_sorted, b_par, _, fm_arr, fp_arr, cf_arr, axis_depths = (
            fast_simulate_single_clean_odmr(
                CANONICAL_FREQS,
                b_vec_canon,
                d0_mhz,
                hwhm_mhz,
                contrast,
                eta,
                e_mhz=e_mhz,
                axis_weights=axis_weights,
                mw_tilt_coeffs=mw_tilt_coeffs,
            )
        )
    else:
        # Stage 2.1 orientation-filtered high-field mode: |B| in b_mag_range_mt (up to 4.60 mT)
        # with all 4 axis transition pairs strictly inside in_window_freq_bounds_mhz ([2760, 2980] MHz)
        b_lo, b_hi = float(split_cfg.b_mag_range_mt[0]), float(split_cfg.b_mag_range_mt[1])
        f_win_lo, f_win_hi = (
            float(split_cfg.in_window_freq_bounds_mhz[0]),
            float(split_cfg.in_window_freq_bounds_mhz[1]),
        )
        if b_lo < 0.20 and float(rng.random()) < 0.5:
            b_mag_draw = float(np.exp(rng.uniform(np.log(b_lo), np.log(b_hi))))
        else:
            b_mag_draw = float(rng.uniform(b_lo, b_hi))
        b_mag_draw = float(np.clip(b_mag_draw, b_lo + 1e-4, b_hi - 1e-4))

        for attempt in range(256):
            if attempt < 250:
                u_hat, sampler_near_deg, min_abs_cos = _sample_direction_s2(rng)
            else:
                # Near-<100> direction where max_k |B_par,k| ~= |B|/sqrt(3) <= 2.66 mT (< 75 MHz half-splitting)
                raw_100 = np.array([1.0, float(rng.uniform(-0.18, 0.18)), float(rng.uniform(-0.18, 0.18))])
                u_hat = raw_100 / np.linalg.norm(raw_100)
                sampler_near_deg = False
                min_abs_cos = float(np.min(np.abs(NV_AXES @ u_hat)))

            b_vec = (b_mag_draw * u_hat).astype(np.float32).astype(np.float64)
            b_mag = float(np.linalg.norm(b_vec))
            if not (b_lo <= b_mag <= b_hi):
                b_vec = ((b_mag_draw / max(b_mag, 1e-9)) * b_vec).astype(np.float32).astype(np.float64)
                b_mag = float(np.linalg.norm(b_vec))
            b_vec_canon = canonicalize_b_vec(b_vec, e_mhz=e_mhz)

            y_clean_f64, splittings_sorted, b_par, _, fm_arr, fp_arr, cf_arr, axis_depths = (
                fast_simulate_single_clean_odmr(
                    CANONICAL_FREQS,
                    b_vec_canon,
                    d0_mhz,
                    hwhm_mhz,
                    contrast,
                    eta,
                    e_mhz=e_mhz,
                    axis_weights=axis_weights,
                    mw_tilt_coeffs=mw_tilt_coeffs,
                )
            )
            if float(np.min(fm_arr)) >= f_win_lo and float(np.max(fp_arr)) <= f_win_hi:
                break
    depth_info = compute_depth_outer_and_flags(
        b_vec,
        d0_mhz,
        hwhm_mhz,
        contrast,
        eta,
        e_mhz=e_mhz,
        precomputed_transitions=(b_par, fm_arr, cf_arr),
        precomputed_axis_depths=axis_depths,
    )
    depth_outer = float(depth_info["depth_outer"])
    axis_degenerate = depth_info["axis_degenerate"]
    weak_splitting = bool(depth_info["weak_splitting"])

    # 3. Sample target eff_snr_white and determine regime_id
    if target_regime is None:
        eff_snr_white = float(np.exp(rng.uniform(np.log(0.5), np.log(20.0))))
        regime_id = regime_id_from_eff_snr_white(eff_snr_white)
    else:
        lo_snr, hi_snr = REGIME_BOUNDS[target_regime]
        eff_snr_white = float(np.exp(rng.uniform(np.log(lo_snr), np.log(hi_snr))))
        regime_id = int(target_regime)

    sigma_white_eff = depth_outer / eff_snr_white

    # 4. Noise budget construction & 20% clean-tract flag (Part 1.4)
    tract_clean = 1 if float(rng.random()) < 0.20 else 0
    if tract_clean == 1:
        drift_amp = float(rng.uniform(0.0, 0.30 * depth_outer))
        if split_cfg.factor_ablation_id == FACTOR_OOD_HUM:
            hum_present = True
        else:
            hum_present = bool(float(rng.random()) >= 0.20)
        hum_amp = float(rng.uniform(0.0002, 0.30 * depth_outer)) if hum_present else 0.0
    else:
        drift_amp = float(rng.uniform(0.002, 0.020))
        if split_cfg.factor_ablation_id == FACTOR_OOD_HUM:
            hum_present = True
        else:
            hum_present = bool(float(rng.random()) >= 0.20)
        hum_amp = float(rng.uniform(0.0004, 0.006)) if hum_present else 0.0

    # 5. Grid realism (Part 2)
    freqs_sim, grid_variant_id = generate_simulated_grid(
        grid_prob=split_cfg.grid_perturb_prob,
        aggressive_crop=split_cfg.ood_grid_aggressive,
        rng=rng,
    )
    n_sim = int(freqs_sim.shape[0])
    if grid_variant_id == GRID_CANONICAL:
        clean_sim = y_clean_f64
    else:
        clean_sim, _ = evaluate_transitions_on_grid(
            freqs_sim,
            fm_arr,
            fp_arr,
            cf_arr,
            hwhm_mhz,
            contrast,
            eta,
            axis_weights=axis_weights,
            mw_tilt_coeffs=mw_tilt_coeffs,
        )

    # 6. Drift, Hum, Stochastic, and Gain components on freqs_sim
    drift_pool = OOD_DRIFT_FAMILIES if split_cfg.ood_drift else ID_DRIFT_FAMILIES
    drift_family_id = int(rng.choice(drift_pool))
    drift_sim, drift_params = generate_drift_component(
        n_sim, drift_family_id, drift_amp, rng
    )

    dwell_time_ms = float(rng.uniform(0.3, 2.0))
    hum_out = generate_hum_component(
        n_sim,
        hum_amp=hum_amp,
        dwell_time_ms=dwell_time_ms,
        is_ood_harmonics=split_cfg.ood_hum,
        rng=rng,
    )
    hum_sim = hum_out["hum"]

    noise_type = NOISE_STUDENT_T_SPIKES_OOD if split_cfg.ood_noise else NOISE_GAUSSIAN_ID
    pink_ratio = float(rng.uniform(0.0, 0.50))
    gain_amp = float(rng.uniform(0.0, 0.005)) if float(rng.random()) < 0.70 else 0.0

    stoch_out = generate_stochastic_and_gain_components(
        clean_sim=clean_sim,
        sigma_white_eff=sigma_white_eff,
        pink_ratio=pink_ratio,
        noise_type=noise_type,
        gain_amp=gain_amp,
        rng=rng,
        drift_sim=drift_sim,
        hum_sim=hum_sim,
    )
    white_sim = stoch_out["white_sim"]
    pink_sim = stoch_out["pink_sim"]
    gain_mod_sim = stoch_out["gain_mod_sim"]

    noisy_sim = clean_sim + drift_sim + hum_sim + pink_sim + white_sim + gain_mod_sim

    # 7. Resample onto canonical grid if grid_variant_id != 0
    if grid_variant_id == GRID_CANONICAL:
        x_noisy_f64 = noisy_sim
        valid_mask = np.ones(N_POINTS, dtype=bool)
        out_of_range = False
        orig_min_mhz = float(CANONICAL_FREQS[0])
        orig_max_mhz = float(CANONICAL_FREQS[-1])
        orig_points = N_POINTS
        drift_canon = drift_sim
        hum_canon = hum_sim
        pink_canon = pink_sim
        white_canon = white_sim
        other_canon = gain_mod_sim
        sigma_stoch_canonical = sigma_white_eff
    else:
        res_out = resample_spectrum_pchip(
            freqs_sim,
            noisy_sim,
            target_freqs=CANONICAL_FREQS,
            method="pchip",
            return_valid_mask=True,
            assumed_hwhm_mhz=hwhm_mhz,
        )
        assert len(res_out) == 3
        x_noisy_f64, valid_mask, meta = res_out
        out_of_range = bool(meta["out_of_range"])
        orig_min_mhz = float(meta["original_min_mhz"])
        orig_max_mhz = float(meta["original_max_mhz"])
        orig_points = int(meta["original_points"])

        drift_canon = np.interp(CANONICAL_FREQS, freqs_sim, drift_sim)
        hum_canon = np.interp(CANONICAL_FREQS, freqs_sim, hum_sim)
        pink_canon = np.interp(CANONICAL_FREQS, freqs_sim, pink_sim)
        white_canon = np.interp(CANONICAL_FREQS, freqs_sim, white_sim)
        other_canon = x_noisy_f64 - (
            y_clean_f64 + drift_canon + hum_canon + pink_canon + white_canon
        )
        stoch_resid_valid = (
            x_noisy_f64 - y_clean_f64 - drift_canon - hum_canon
        )[valid_mask]
        sigma_stoch_canonical = float(np.std(stoch_resid_valid))

    rms_drift = float(np.sqrt(np.dot(drift_canon, drift_canon) / float(N_POINTS)))
    rms_hum = float(np.sqrt(np.dot(hum_canon, hum_canon) / float(N_POINTS)))
    eff_snr_total = depth_outer / float(
        np.sqrt(sigma_white_eff * sigma_white_eff + rms_drift * rms_drift + rms_hum * rms_hum)
    )
    dirt_drift = rms_drift / depth_outer
    dirt_hum = rms_hum / depth_outer
    clean_ptp = max(float(y_clean_f64.max() - y_clean_f64.min()), 1e-12)
    nominal_snr_legacy = clean_ptp / sigma_white_eff
    stoch_std_ratio = sigma_stoch_canonical / max(sigma_white_eff, 1e-15)

    sample_dict: Dict[str, Any] = {
        "X_noisy": x_noisy_f64.astype(np.float32),
        "valid_mask": valid_mask,
        "Y_clean": y_clean_f64.astype(np.float32),
        "Y_splittings": splittings_sorted.astype(np.float32),
        "Y_b_mag": np.float32(b_mag),
        "Y_d0": np.float32(d0_offset_mhz),
        "Y_hwhm": np.float32(hwhm_mhz),
        "aux_b_vec_canon": b_vec_canon.astype(np.float32),
        "aux_b_vec_raw": b_vec.astype(np.float32),
        "aux_b_par": b_par.astype(np.float32),
        "aux_axis_degenerate": axis_degenerate,
        "aux_min_abs_cos_theta": np.float32(min_abs_cos),
        "aux_sampler_near_degenerate": bool(sampler_near_deg),
        "aux_weak_splitting": bool(weak_splitting),
        "aux_depth_outer": np.float32(depth_outer),
        "aux_eta": np.float32(eta),
        "aux_contrast": np.float32(contrast),
        "aux_tract_clean": np.int8(tract_clean),
        "aux_sigma_white_eff": np.float32(sigma_white_eff),
        "aux_sigma_stoch_canonical": np.float32(sigma_stoch_canonical),
        "aux_stoch_std_ratio": np.float32(stoch_std_ratio),
        "aux_eff_snr_white": np.float32(eff_snr_white),
        "aux_eff_snr_total": np.float32(eff_snr_total),
        "aux_dirt_drift": np.float32(dirt_drift),
        "aux_dirt_hum": np.float32(dirt_hum),
        "aux_nominal_snr_legacy": np.float32(nominal_snr_legacy),
        "aux_regime_id": np.int8(regime_id),
        "aux_drift_family_id": np.int8(drift_family_id),
        "aux_drift_amp": np.float32(drift_amp),
        "aux_drift_rms": np.float32(rms_drift),
        "aux_drift_params": drift_params.astype(np.float32),
        "aux_hum_freq_hz": np.float32(hum_out["freq_hz"]),
        "aux_hum_phase_rad": hum_out["phases_rad"].astype(np.float32),
        "aux_hum_harmonics_amp": hum_out["harmonics_amp"].astype(np.float32),
        "aux_hum_rel_harmonics": hum_out["rel_harmonics"].astype(np.float32),
        "aux_hum_rms": np.float32(rms_hum),
        "aux_dwell_time_ms": np.float32(dwell_time_ms),
        "aux_noise_type": np.int8(noise_type),
        "aux_pink_ratio": np.float32(pink_ratio),
        "aux_gain_amp": np.float32(gain_amp),
        "aux_grid_variant_id": np.int8(grid_variant_id),
        "aux_orig_points": np.int16(orig_points),
        "aux_orig_min_mhz": np.float32(orig_min_mhz),
        "aux_orig_max_mhz": np.float32(orig_max_mhz),
        "aux_out_of_range": bool(out_of_range),
        "aux_sample_seed": np.uint64(sample_seed),
        "aux_sample_idx": np.int32(sample_idx),
        "aux_e_mhz": np.float32(e_mhz),
        "aux_factor_ablation_id": np.int8(split_cfg.factor_ablation_id),
        "aux_axis_weights": axis_weights_f32,
        "aux_axis_asym_ratio": np.float32(axis_asym_ratio),
        "aux_mw_power_tilt_amp": np.float32(realized_mw_tilt_amp),
        "aux_mw_power_tilt_coeffs": mw_tilt_coeffs_f32,
    }

    if return_f64_components:
        sample_dict["f64_components"] = {
            "Y_clean": y_clean_f64,
            "drift": drift_canon,
            "hum": hum_canon,
            "pink": pink_canon,
            "white": white_canon,
            "other": other_canon,
            "X_noisy_f64": x_noisy_f64,
        }

    return sample_dict


def regenerate_sample_components(
    split_cfg: Union[str, SplitConfig],
    sample_idx: int,
) -> Dict[str, np.ndarray]:
    """Regenerate float64 additive components for ``(split_cfg, sample_idx)`` (Part 2 verification)."""
    cfg = SPLIT_REGISTRY[split_cfg] if isinstance(split_cfg, str) else split_cfg
    out = generate_single_sample(sample_idx, cfg, return_f64_components=True)
    return out["f64_components"]


def collate_samples_to_arrays(
    samples: Sequence[Dict[str, Any]],
    split_name: str,
) -> Dict[str, np.ndarray]:
    """Collate a list of single-sample dicts into stacked NumPy arrays."""
    keys = [k for k in samples[0].keys() if k != "f64_components"]
    batched: Dict[str, np.ndarray] = {}
    for k in keys:
        batched[k] = np.stack([s[k] for s in samples], axis=0)
    batched["aux_split_name"] = np.array(split_name)
    return batched


def _worker_generate_chunk(args: Tuple[SplitConfig, int, int]) -> Dict[str, np.ndarray]:
    """Worker entry point to generate sample indices ``[start_idx, end_idx)`` for ``split_cfg``."""
    split_cfg, start_idx, end_idx = args
    samples = [
        generate_single_sample(idx, split_cfg, return_f64_components=False)
        for idx in range(start_idx, end_idx)
    ]
    return collate_samples_to_arrays(samples, split_cfg.name)


def generate_split_arrays(
    split_cfg: Union[str, SplitConfig],
    n_samples: Optional[int] = None,
    start_idx: int = 0,
    n_workers: int = 1,
    executor: Optional[ProcessPoolExecutor] = None,
) -> Dict[str, np.ndarray]:
    """Generate a dataset split of ``n_samples`` deterministically using 1 or ``n_workers`` processes.

    Also verifies batch clean simulation via ``simulate_batch_clean_odmr`` when running
    in single-process batch mode.
    """
    cfg = SPLIT_REGISTRY[split_cfg] if isinstance(split_cfg, str) else split_cfg
    total = int(n_samples) if n_samples is not None else int(cfg.n_samples)
    if total <= 0:
        raise ValueError(f"n_samples must be > 0, got {total}.")

    if (n_workers <= 1 and executor is None) or total < 128:
        samples = [
            generate_single_sample(idx, cfg, return_f64_components=False)
            for idx in range(start_idx, start_idx + total)
        ]
        return collate_samples_to_arrays(samples, cfg.name)

    n_chunks = min(max(n_workers, 2), total)
    edges = np.linspace(start_idx, start_idx + total, n_chunks + 1, dtype=int)
    tasks = [
        (cfg, int(edges[i]), int(edges[i + 1]))
        for i in range(n_chunks)
        if int(edges[i + 1]) > int(edges[i])
    ]
    if executor is not None:
        chunk_dicts = list(executor.map(_worker_generate_chunk, tasks))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            chunk_dicts = list(ex.map(_worker_generate_chunk, tasks))

    merged: Dict[str, np.ndarray] = {}
    array_keys = [k for k in chunk_dicts[0].keys() if k != "aux_split_name"]
    for k in array_keys:
        merged[k] = np.concatenate([cd[k] for cd in chunk_dicts], axis=0)
    merged["aux_split_name"] = np.array(cfg.name)
    return merged


# =============================================================================
# Test-File Lockdown & Access Audit Logging (Hard Constraint 3)
# =============================================================================


def log_test_file_event(
    split_name: str,
    file_path: Path,
    event_type: str,
    caller: str = "",
    log_path: Path = TEST_ACCESS_LOG_PATH,
) -> None:
    """Append an access audit entry to ``data/synthetic/test_access.log``.

    Args:
        split_name: Split name (e.g., ``"test_id"`` or ``"test_ood"``).
        file_path: Path to the test file.
        event_type: ``"write_once"``, ``"hash_check"``, or ``"data_read"``.
        caller: Optional caller context.
        log_path: Audit log path.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split": split_name,
        "file": file_path.name,
        "event": event_type,
        "caller": caller,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def compute_file_sha256(
    file_path: Path,
    split_name: Optional[str] = None,
    log_as_hash_check: bool = True,
    log_path: Path = TEST_ACCESS_LOG_PATH,
) -> str:
    """Compute SHA256 digest of ``file_path`` by streaming raw bytes only (never parsing NPZ).

    When ``file_path`` corresponds to a locked test file and ``log_as_hash_check=True``,
    logs ``"event": "hash_check"`` to ``test_access.log``.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    digest = h.hexdigest()

    sname = split_name if split_name is not None else file_path.stem
    if log_as_hash_check and sname in LOCKED_TEST_SPLITS:
        log_test_file_event(
            split_name=sname,
            file_path=file_path,
            event_type="hash_check",
            caller="compute_file_sha256",
            log_path=log_path,
        )
    return digest


def get_test_access_records(log_path: Path = TEST_ACCESS_LOG_PATH) -> List[Dict[str, Any]]:
    """Read all JSON records from ``data/synthetic/test_access.log``."""
    if not log_path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                records.append(json.loads(s))
    return records


def get_test_access_counts(log_path: Path = TEST_ACCESS_LOG_PATH) -> Dict[str, int]:
    """Count ``data_read``, ``hash_check``, and ``write_once`` events in ``test_access.log``."""
    records = get_test_access_records(log_path)
    counts = {"data_read": 0, "hash_check": 0, "write_once": 0}
    for rec in records:
        ev = str(rec.get("event", ""))
        if ev in counts:
            counts[ev] += 1
    return counts


def load_synthetic_split(
    split_name: str,
    data_dir: Path = DEFAULT_SYNTHETIC_DIR,
    allow_test_read: bool = False,
    log_path: Path = TEST_ACCESS_LOG_PATH,
) -> Dict[str, np.ndarray]:
    """Load a saved synthetic ``.npz`` split into a dictionary of NumPy arrays.

    CRITICAL LOCKDOWN POLICY:
      If ``split_name`` is ``"test_id"`` or ``"test_ood"`` (or filename starts with ``test_``),
      this function ALWAYS logs ``"event": "data_read"`` to ``test_access.log`` BEFORE
      checking permissions or reading arrays, so any illicit test-file read is permanently
      recorded and caught by ``pytest``. Unless ``allow_test_read=True`` is explicitly passed,
      a ``PermissionError`` is raised immediately.
    """
    file_path = data_dir / f"{split_name}.npz"
    is_test = (split_name in LOCKED_TEST_SPLITS) or file_path.stem.startswith("test_")
    if is_test:
        log_test_file_event(
            split_name=split_name,
            file_path=file_path,
            event_type="data_read",
            caller="load_synthetic_split",
            log_path=log_path,
        )
        if not allow_test_read:
            raise PermissionError(
                f"LOCKED TEST SPLIT VIOLATION: Parsing '{file_path}' is strictly forbidden in Stage 2."
            )

    with np.load(file_path, allow_pickle=False) as npz_data:
        out_dict = {k: npz_data[k] for k in npz_data.files}
    n_s = int(out_dict["X_noisy"].shape[0])
    if "aux_axis_weights" not in out_dict:
        out_dict["aux_axis_weights"] = np.ones((n_s, 4), dtype=np.float32)
    if "aux_axis_asym_ratio" not in out_dict:
        out_dict["aux_axis_asym_ratio"] = np.ones(n_s, dtype=np.float32)
    if "aux_mw_power_tilt_amp" not in out_dict:
        out_dict["aux_mw_power_tilt_amp"] = np.zeros(n_s, dtype=np.float32)
    if "aux_mw_power_tilt_coeffs" not in out_dict:
        out_dict["aux_mw_power_tilt_coeffs"] = np.zeros((n_s, 2), dtype=np.float32)
    return out_dict


def save_split_npz(
    arrays: Dict[str, np.ndarray],
    file_path: Path,
    split_name: str,
    is_locked_test: bool = False,
    log_path: Path = TEST_ACCESS_LOG_PATH,
) -> str:
    """Save a split dictionary as an uncompressed ``.npz`` and return its raw-byte SHA256."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if is_locked_test and file_path.exists():
        raise FileExistsError(
            f"Locked test file '{file_path}' already exists and must never be overwritten."
        )
    np.savez(file_path, **arrays)
    if is_locked_test:
        log_test_file_event(
            split_name=split_name,
            file_path=file_path,
            event_type="write_once",
            caller="save_split_npz",
            log_path=log_path,
        )
    return compute_file_sha256(
        file_path,
        split_name=split_name,
        log_as_hash_check=is_locked_test,
        log_path=log_path,
    )


def validate_split_arrays_in_memory(
    arrays: Dict[str, np.ndarray],
    split_cfg: SplitConfig,
    n_regen_check: int = 500,
) -> Dict[str, Any]:
    """Run all Part 3 & Part 6 numerical consistency and disjoint-support checks on an in-memory split."""
    n = int(arrays["X_noisy"].shape[0])
    assert arrays["X_noisy"].shape == (n, N_POINTS)
    assert arrays["Y_clean"].shape == (n, N_POINTS)
    assert arrays["Y_splittings"].shape == (n, 4)
    assert arrays["valid_mask"].shape == (n, N_POINTS)

    # 1. No NaN or Inf in any floating-point array
    for k, v in arrays.items():
        if np.issubdtype(v.dtype, np.floating):
            if not np.all(np.isfinite(v)):
                raise AssertionError(f"NaN/Inf detected in split '{split_cfg.name}' array '{k}'.")

    # 2. Range checks on physical parameters
    hwhm = arrays["Y_hwhm"]
    contrast = arrays["aux_contrast"]
    d0_off = arrays["Y_d0"]
    b_mag = arrays["Y_b_mag"]
    eta = arrays["aux_eta"]
    drift_fams = arrays["aux_drift_family_id"]
    noise_types = arrays["aux_noise_type"]
    hum_rel = arrays["aux_hum_rel_harmonics"]

    b_lo_chk, b_hi_chk = float(split_cfg.b_mag_range_mt[0]), float(split_cfg.b_mag_range_mt[1])
    assert float(np.min(b_mag)) >= b_lo_chk - 1e-4
    assert float(np.max(b_mag)) <= b_hi_chk + 1e-4
    assert float(np.min(eta)) >= 0.0 and float(np.max(eta)) <= 1.0

    if split_cfg.hwhm_range_mhz is not None:
        hw_lo, hw_hi = float(split_cfg.hwhm_range_mhz[0]), float(split_cfg.hwhm_range_mhz[1])
        assert float(np.min(hwhm)) >= hw_lo - 1e-5 and float(np.max(hwhm)) <= hw_hi + 1e-5
    elif split_cfg.ood_hwhm:
        assert float(np.min(hwhm)) >= 9.5 - 1e-5 and float(np.max(hwhm)) <= 13.5 + 1e-5
    else:
        assert float(np.min(hwhm)) >= 3.0 - 1e-5 and float(np.max(hwhm)) <= 9.0 + 1e-5

    if split_cfg.axis_asym_range is not None:
        r_lo, r_hi = float(split_cfg.axis_asym_range[0]), float(split_cfg.axis_asym_range[1])
        ratios = arrays["aux_axis_asym_ratio"].astype(np.float64)
        w_arr = arrays["aux_axis_weights"].astype(np.float64)
        assert float(np.max( np.abs(np.mean(w_arr, axis=1) - 1.0) )) < 1e-5
        if split_cfg.axis_equal_prob > 0.0:
            non_eq = ratios > 1.0 + 1e-5
            assert bool(np.all((ratios[~non_eq] >= 1.0 - 1e-5) & (ratios[~non_eq] <= 1.0 + 1e-5)))
            if np.any(non_eq):
                assert float(np.min(ratios[non_eq])) >= r_lo - 1e-4
                assert float(np.max(ratios[non_eq])) <= r_hi + 1e-4
        else:
            assert float(np.min(ratios)) >= r_lo - 1e-4
            assert float(np.max(ratios)) <= r_hi + 1e-4

    if split_cfg.ood_contrast:
        in_lo = (contrast >= 0.010 - 1e-6) & (contrast <= 0.0145 + 1e-6)
        in_hi = (contrast >= 0.0405 - 1e-6) & (contrast <= 0.050 + 1e-6)
        assert bool(np.all(in_lo | in_hi))
    else:
        assert float(np.min(contrast)) >= 0.015 - 1e-6 and float(np.max(contrast)) <= 0.040 + 1e-6

    if split_cfg.ood_d0:
        in_lo = (d0_off >= -5.0 - 1e-5) & (d0_off <= -3.1 + 1e-5)
        in_hi = (d0_off >= 3.1 - 1e-5) & (d0_off <= 5.0 + 1e-5)
        assert bool(np.all(in_lo | in_hi))
    else:
        assert float(np.min(d0_off)) >= -3.0 - 1e-5 and float(np.max(d0_off)) <= 3.0 + 1e-5

    if split_cfg.ood_drift:
        assert set(np.unique(drift_fams).tolist()).issubset(set(OOD_DRIFT_FAMILIES))
    else:
        assert set(np.unique(drift_fams).tolist()).issubset(set(ID_DRIFT_FAMILIES))

    if split_cfg.ood_noise:
        assert bool(np.all(noise_types == NOISE_STUDENT_T_SPIKES_OOD))
    else:
        assert bool(np.all(noise_types == NOISE_GAUSSIAN_ID))

    if split_cfg.ood_hum:
        assert float(np.min(hum_rel)) > 0.50
        assert float(np.max(hum_rel)) <= 0.85 + 1e-5
    else:
        assert float(np.max(hum_rel)) <= 0.50 + 1e-5

    # 3. Stratification check
    regimes = arrays["aux_regime_id"]
    c0 = int(np.sum(regimes == REGIME_HARD))
    c1 = int(np.sum(regimes == REGIME_MEDIUM))
    c2 = int(np.sum(regimes == REGIME_EASY))
    if split_cfg.stratified_regimes and n == 4000:
        assert (c0, c1, c2) == (1333, 1333, 1334), f"Expected (1333,1333,1334), got {(c0,c1,c2)}"

    # 4. Regeneration identity check on random subset of size min(n_regen_check, n)
    rng_sub = np.random.default_rng(split_cfg.split_seed ^ 0x5A5A5A5A)
    k_check = min(int(n_regen_check), n)
    sub_indices = rng_sub.choice(n, size=k_check, replace=False)

    max_f64_id_err = 0.0
    max_f32_id_err = 0.0
    max_clean_sim_err = 0.0
    max_splittings_eigh_err = 0.0

    b_vecs_sub = arrays["aux_b_vec_canon"][sub_indices].astype(np.float64)
    d0_sub = D0_MHZ + arrays["Y_d0"][sub_indices].astype(np.float64)
    hwhm_sub = arrays["Y_hwhm"][sub_indices].astype(np.float64)
    contrast_sub = arrays["aux_contrast"][sub_indices].astype(np.float64)
    eta_sub = arrays["aux_eta"][sub_indices].astype(np.float64)

    batch_sim = simulate_batch_clean_odmr(
        CANONICAL_FREQS,
        b_vecs=b_vecs_sub,
        d0_arr=d0_sub,
        gamma_arr=hwhm_sub,
        contrast_arr=contrast_sub,
        eta_arr=eta_sub,
        return_details=True,
    )
    assert isinstance(batch_sim, dict)

    for j_pos, idx in enumerate(sub_indices):
        s_idx = int(arrays["aux_sample_idx"][idx])
        comps = regenerate_sample_components(split_cfg, s_idx)
        sum_f64 = (
            comps["Y_clean"]
            + comps["drift"]
            + comps["hum"]
            + comps["pink"]
            + comps["white"]
            + comps["other"]
        )
        err_f64 = float(np.max(np.abs(sum_f64 - comps["X_noisy_f64"])))
        err_f32 = float(np.max(np.abs(sum_f64.astype(np.float32) - arrays["X_noisy"][idx])))
        if (
            split_cfg.axis_asym_range is not None
            or split_cfg.per_axis_contrast_jitter > 0.0
            or split_cfg.mw_power_tilt_amp > 0.0
        ):
            ref_clean, *_ = fast_simulate_single_clean_odmr(
                CANONICAL_FREQS,
                b_vecs_sub[j_pos],
                float(d0_sub[j_pos]),
                float(hwhm_sub[j_pos]),
                float(contrast_sub[j_pos]),
                float(eta_sub[j_pos]),
                e_mhz=0.0,
                axis_weights=arrays["aux_axis_weights"][idx].astype(np.float64),
                mw_tilt_coeffs=arrays["aux_mw_power_tilt_coeffs"][idx].astype(np.float64),
            )
            err_clean = float(np.max(np.abs(comps["Y_clean"] - ref_clean)))
        else:
            err_clean = float(np.max(np.abs(comps["Y_clean"] - batch_sim["spectra"][j_pos])))
        err_split = float(
            np.max(
                np.abs(
                    arrays["Y_splittings"][idx].astype(np.float64)
                    - batch_sim["splittings_mhz"][j_pos]
                )
            )
        )
        max_f64_id_err = max(max_f64_id_err, err_f64)
        max_f32_id_err = max(max_f32_id_err, err_f32)
        max_clean_sim_err = max(max_clean_sim_err, err_clean)
        max_splittings_eigh_err = max(max_splittings_eigh_err, err_split)

    # b_mag_from_splittings(order=2) vs Y_b_mag RMS (uT) across the full split
    b_from_split_o2 = np.array(
        [
            float(
                b_mag_from_splittings(
                    arrays["Y_splittings"][i].astype(np.float64),
                    order=2,
                    d0_mhz=float(D0_MHZ + arrays["Y_d0"][i]),
                )
            )
            for i in range(n)
        ],
        dtype=np.float64,
    )
    b_o2_err_ut = (b_from_split_o2 - arrays["Y_b_mag"].astype(np.float64)) * 1000.0
    b_o2_rms_ut = float(np.sqrt(np.mean(b_o2_err_ut * b_o2_err_ut)))

    assert max_f64_id_err < 1e-12, f"float64 component identity error {max_f64_id_err} >= 1e-12"
    assert max_f32_id_err < 1e-6, f"float32 component identity error {max_f32_id_err} >= 1e-6"
    if split_cfg.per_axis_contrast_jitter <= 0.0:
        assert max_clean_sim_err < 1e-10, f"Y_clean vs simulate_batch_clean_odmr error {max_clean_sim_err} >= 1e-10"
    assert max_splittings_eigh_err < 1e-5, f"Y_splittings f32 vs eigh error {max_splittings_eigh_err} >= 1e-5"

    if split_cfg.allow_high_field_in_window:
        f_win_lo, f_win_hi = (
            float(split_cfg.in_window_freq_bounds_mhz[0]),
            float(split_cfg.in_window_freq_bounds_mhz[1]),
        )
        dip_pairs = batch_sim["dip_pairs"]
        assert float(np.min(dip_pairs[..., 0])) >= f_win_lo - 1e-3
        assert float(np.max(dip_pairs[..., 1])) <= f_win_hi + 1e-3

    return {
        "split": split_cfg.name,
        "n_samples": n,
        "regime_counts": (c0, c1, c2),
        "max_f64_identity_err": max_f64_id_err,
        "max_f32_identity_err": max_f32_id_err,
        "max_clean_sim_err": max_clean_sim_err,
        "max_splittings_eigh_err": max_splittings_eigh_err,
        "b_mag_order2_vs_true_rms_ut": b_o2_rms_ut,
    }


def run_dry_run_twin_and_generate_all_splits(
    output_dir: Path = DEFAULT_SYNTHETIC_DIR,
    n_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute the full Part 3 & Part 4 dataset generation pipeline including dry-run twin verification.

    1. Generates non-test splits: ``train_snapshot`` (20,000), ``val_id`` (4,000),
       ``val_ood`` (4,000), and the 7 single-factor ``ood_*`` splits (1,000 each).
    2. Runs the dry-run twin protocol for ``test_id`` and ``test_ood`` using throwaway
       seeds ``999901`` and ``999902`` inside a temporary directory, executes all
       Part 6 consistency checks on the twins, and deletes the temporary directory.
    3. Only after the twins pass 100% of checks, generates ``test_id.npz`` and
       ``test_ood.npz`` ONCE, computes their raw-byte SHA256 hashes (logged as
       ``"hash_check"``), and never parses them again.
    4. Writes ``data/synthetic/manifest.json``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "test_access.log"
    if log_path.exists():
        log_path.unlink()

    workers = n_workers if n_workers is not None else max(1, min(os.cpu_count() or 4, 8))

    non_test_splits = [
        "train_snapshot",
        "val_id",
        "val_ood",
        "ood_hwhm",
        "ood_contrast",
        "ood_d0",
        "ood_noise",
        "ood_drift",
        "ood_hum",
        "ood_grid",
    ]

    manifest_files: Dict[str, Dict[str, Any]] = {}
    validation_summaries: Dict[str, Any] = {}
    twin_report: Dict[str, Any] = {}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for sname in non_test_splits:
            cfg = SPLIT_REGISTRY[sname]
            t0 = time.perf_counter()
            arrays = generate_split_arrays(cfg, n_workers=workers, executor=pool)
            gen_sec = time.perf_counter() - t0
            val_summary = validate_split_arrays_in_memory(arrays, cfg, n_regen_check=500)
            validation_summaries[sname] = val_summary

            fpath = output_dir / f"{sname}.npz"
            sha256_hex = save_split_npz(
                arrays, fpath, split_name=sname, is_locked_test=False, log_path=log_path
            )
            manifest_files[sname] = {
                "file": fpath.name,
                "split_seed": cfg.split_seed,
                "n_samples": cfg.n_samples,
                "stratified_regimes": cfg.stratified_regimes,
                "factor_ablation_id": cfg.factor_ablation_id,
                "factor_ablation_name": FACTOR_ABLATION_NAMES[cfg.factor_ablation_id],
                "sha256": sha256_hex,
                "generation_time_sec": round(gen_sec, 3),
                "throughput_spectra_per_sec": round(cfg.n_samples / max(gen_sec, 1e-6), 1),
            }

        # Step 2: Dry-run twin validation in temporary directory with throwaway seeds
        twin_configs = [
            SplitConfig(
                name="twin_test_id",
                split_seed=999901,
                n_samples=4000,
                stratified_regimes=True,
                factor_ablation_id=FACTOR_ID_STANDARD,
                grid_perturb_prob=0.30,
                is_locked_test=False,
            ),
            SplitConfig(
                name="twin_test_ood",
                split_seed=999902,
                n_samples=4000,
                stratified_regimes=True,
                factor_ablation_id=FACTOR_OOD_COMBINED,
                ood_hwhm=True,
                ood_contrast=True,
                ood_d0=True,
                ood_noise=True,
                ood_drift=True,
                ood_hum=True,
                ood_grid_aggressive=False,
                grid_perturb_prob=0.40,
                is_locked_test=False,
            ),
        ]

        with tempfile.TemporaryDirectory(prefix="opennv_twin_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            for t_cfg in twin_configs:
                t_arrays = generate_split_arrays(t_cfg, n_workers=workers, executor=pool)
                t_file = tmp_path / f"{t_cfg.name}.npz"
                t_sha = save_split_npz(
                    t_arrays, t_file, split_name=t_cfg.name, is_locked_test=False, log_path=log_path
                )
                with np.load(t_file, allow_pickle=False) as loaded_twin:
                    reloaded = {k: loaded_twin[k] for k in loaded_twin.files}
                t_val = validate_split_arrays_in_memory(reloaded, t_cfg, n_regen_check=500)
                t_val["temp_sha256"] = t_sha
                t_val["deleted_after_check"] = True
                twin_report[t_cfg.name] = t_val

        # Step 3: Now generate the locked test_id.npz and test_ood.npz ONCE
        for test_name in LOCKED_TEST_SPLITS:
            t_cfg = SPLIT_REGISTRY[test_name]
            fpath = output_dir / f"{test_name}.npz"
            if fpath.exists():
                fpath.unlink()
            t0 = time.perf_counter()
            test_arrays = generate_split_arrays(t_cfg, n_workers=workers, executor=pool)
            gen_sec = time.perf_counter() - t0
            sha256_hex = save_split_npz(
                test_arrays,
                fpath,
                split_name=test_name,
                is_locked_test=True,
                log_path=log_path,
            )
            # Immediately release in-memory test arrays without running any post-hoc analysis on them
            del test_arrays
            manifest_files[test_name] = {
                "file": fpath.name,
                "split_seed": t_cfg.split_seed,
                "n_samples": t_cfg.n_samples,
                "stratified_regimes": t_cfg.stratified_regimes,
                "factor_ablation_id": t_cfg.factor_ablation_id,
                "factor_ablation_name": FACTOR_ABLATION_NAMES[t_cfg.factor_ablation_id],
                "sha256": sha256_hex,
                "locked_test_set": True,
                "generation_time_sec": round(gen_sec, 3),
                "throughput_spectra_per_sec": round(t_cfg.n_samples / max(gen_sec, 1e-6), 1),
            }

    manifest_payload = {
        "stage": "Stage 2 (stage2-data-v1)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_grid": {
            "freq_min_mhz": 2750.0,
            "freq_max_mhz": 2990.0,
            "n_points": N_POINTS,
            "d0_mhz": D0_MHZ,
            "e_mhz": 0.0,
        },
        "on_the_fly_train_seed": SPLIT_REGISTRY["train"].split_seed,
        "dry_run_twin_verification": twin_report,
        "splits": manifest_files,
        "test_access_counts": get_test_access_counts(log_path),
    }

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    return {
        "manifest": manifest_payload,
        "validation_summaries": validation_summaries,
        "twin_report": twin_report,
    }


def generate_stage21_splits_and_update_manifest(
    output_dir: Path = DEFAULT_SYNTHETIC_DIR,
    n_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate Stage 2.1 additive splits (``val_id_natural`` and ``ood_high_field``) without modifying existing splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "test_access.log"
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_payload = json.load(f)

    workers = n_workers if n_workers is not None else max(1, min(os.cpu_count() or 4, 8))
    new_splits = ["val_id_natural", "ood_high_field"]
    val_summaries: Dict[str, Any] = {}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for sname in new_splits:
            cfg = SPLIT_REGISTRY[sname]
            t0 = time.perf_counter()
            arrays = generate_split_arrays(cfg, n_workers=workers, executor=pool)
            gen_sec = time.perf_counter() - t0
            val_sum = validate_split_arrays_in_memory(arrays, cfg, n_regen_check=500)
            val_summaries[sname] = val_sum

            fpath = output_dir / f"{sname}.npz"
            sha256_hex = save_split_npz(
                arrays, fpath, split_name=sname, is_locked_test=False, log_path=log_path
            )
            manifest_payload["splits"][sname] = {
                "file": fpath.name,
                "split_seed": cfg.split_seed,
                "n_samples": cfg.n_samples,
                "stratified_regimes": cfg.stratified_regimes,
                "factor_ablation_id": cfg.factor_ablation_id,
                "factor_ablation_name": FACTOR_ABLATION_NAMES[cfg.factor_ablation_id],
                "b_mag_range_mt": list(cfg.b_mag_range_mt),
                "allow_high_field_in_window": cfg.allow_high_field_in_window,
                "sha256": sha256_hex,
                "generation_time_sec": round(gen_sec, 3),
                "throughput_spectra_per_sec": round(cfg.n_samples / max(gen_sec, 1e-6), 1),
            }

    manifest_payload["stage"] = "Stage 2.1 (stage2-data-v2)"
    manifest_payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_payload["test_access_counts"] = get_test_access_counts(log_path)

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    return {
        "manifest": manifest_payload,
        "validation_summaries": val_summaries,
    }


def measure_generator_throughput(
    n_samples: int = 2000,
    n_workers: Optional[int] = None,
    split_cfg: Union[str, SplitConfig] = "train",
) -> Dict[str, float]:
    """Benchmark single-process and multi-worker dataset generation throughput (Part 4 & Task 2)."""
    cfg = SPLIT_REGISTRY[split_cfg] if isinstance(split_cfg, str) else split_cfg
    workers = n_workers if n_workers is not None else max(2, min(os.cpu_count() or 4, 8))

    # Warm-up single process
    _ = generate_split_arrays(cfg, n_samples=20, start_idx=0, n_workers=1)

    t0 = time.perf_counter()
    _ = generate_split_arrays(cfg, n_samples=n_samples, start_idx=100, n_workers=1)
    dt_single = time.perf_counter() - t0
    rate_single = float(n_samples) / max(dt_single, 1e-9)

    n_multi = n_samples * workers
    edges_warm = np.linspace(0, workers * 10, workers + 1, dtype=int)
    warm_tasks = [(cfg, int(edges_warm[i]), int(edges_warm[i + 1])) for i in range(workers)]
    edges_bench = np.linspace(5000, 5000 + n_multi, workers + 1, dtype=int)
    bench_tasks = [(cfg, int(edges_bench[i]), int(edges_bench[i + 1])) for i in range(workers)]

    with ProcessPoolExecutor(max_workers=workers) as ex:
        # Warm up worker processes (one-time Windows spawn + import)
        _ = list(ex.map(_worker_generate_chunk, warm_tasks))
        t1 = time.perf_counter()
        chunk_dicts = list(ex.map(_worker_generate_chunk, bench_tasks))
        array_keys = [k for k in chunk_dicts[0].keys() if k != "aux_split_name"]
        _ = {k: np.concatenate([cd[k] for cd in chunk_dicts], axis=0) for k in array_keys}
        dt_multi = time.perf_counter() - t1

    rate_multi = float(n_multi) / max(dt_multi, 1e-9)

    return {
        "n_samples_single": float(n_samples),
        "single_process_sec": dt_single,
        "single_process_spectra_per_sec": rate_single,
        "n_workers": float(workers),
        "n_samples_multi": float(n_multi),
        "multi_worker_sec": dt_multi,
        "multi_worker_spectra_per_sec": rate_multi,
    }


# =============================================================================
# Lazy PyTorch Dataset / IterableDataset Wrappers (Part 4)
# =============================================================================


class NumpyODMRDataset:
    """Framework-agnostic map-style dataset wrapping an in-memory or on-the-fly split.

    Works without PyTorch installed. When ``as_torch=True``, lazily imports
    ``torch`` and converts numeric arrays to ``torch.Tensor``.
    """

    def __init__(
        self,
        split_name: str = "val_id",
        data_dir: Path = DEFAULT_SYNTHETIC_DIR,
        in_memory_arrays: Optional[Dict[str, np.ndarray]] = None,
        as_torch: bool = False,
    ) -> None:
        self.split_name = split_name
        self.as_torch = as_torch
        if in_memory_arrays is not None:
            self.arrays = in_memory_arrays
        else:
            self.arrays = load_synthetic_split(split_name, data_dir=data_dir, allow_test_read=False)
        self._length = int(self.arrays["X_noisy"].shape[0])

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range for split '{self.split_name}' of size {self._length}.")
        item = {
            k: v[idx]
            for k, v in self.arrays.items()
            if k != "aux_split_name" and v.ndim >= 1
        }
        if not self.as_torch:
            return item
        import torch  # Lazy import

        return {k: torch.as_tensor(v) for k, v in item.items()}


class OnTheFlyODMRIterableDataset:
    """Infinite or bounded deterministic on-the-fly stream for ``train`` (Part 3 & Part 4).

    Supports multi-worker sharding via ``worker_id`` / ``num_workers`` (or automatic
    detection via ``torch.utils.data.get_worker_info()`` when PyTorch is active).
    Works cleanly without PyTorch installed.
    """

    def __init__(
        self,
        split_name: Union[str, SplitConfig] = "train",
        start_idx: int = 0,
        max_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
        as_torch: bool = False,
    ) -> None:
        self.split_cfg = SPLIT_REGISTRY[split_name] if isinstance(split_name, str) else split_name
        self.start_idx = int(start_idx)
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.as_torch = as_torch

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_id = 0
        num_workers = 1
        if self.as_torch:
            try:
                import torch.utils.data as tud  # Lazy import

                winfo = tud.get_worker_info()
                if winfo is not None:
                    worker_id = int(winfo.id)
                    num_workers = int(winfo.num_workers)
            except ImportError:
                pass

        idx = self.start_idx + worker_id
        emitted = 0
        batch_buf: List[Dict[str, Any]] = []

        while self.max_samples is None or emitted < self.max_samples:
            sample = generate_single_sample(idx, self.split_cfg, return_f64_components=False)
            idx += num_workers
            emitted += 1

            if self.batch_size is None:
                if self.as_torch:
                    import torch

                    yield {
                        k: torch.as_tensor(v)
                        for k, v in sample.items()
                        if isinstance(v, (np.ndarray, np.generic, bool, int, float))
                    }
                else:
                    yield sample
            else:
                batch_buf.append(sample)
                if len(batch_buf) == self.batch_size:
                    collated = collate_samples_to_arrays(batch_buf, self.split_cfg.name)
                    batch_buf.clear()
                    if self.as_torch:
                        import torch

                        yield {
                            k: torch.as_tensor(v)
                            for k, v in collated.items()
                            if k != "aux_split_name"
                        }
                    else:
                        yield collated

        if batch_buf:
            collated = collate_samples_to_arrays(batch_buf, self.split_cfg.name)
            if self.as_torch:
                import torch

                yield {
                    k: torch.as_tensor(v)
                    for k, v in collated.items()
                    if k != "aux_split_name"
                }
            else:
                yield collated


def select_12_unique_example_indices(arrays: Dict[str, np.ndarray]) -> List[Tuple[int, str]]:
    """Select 12 strictly unique representative spectrum indices across Easy/Medium/Hard regimes.

    For each regime (Easy, Medium, Hard), selects 4 distinct panels:
    1. 4-Axis Resolved (non-weak, non-degenerate, non-cropped)
    2. Axis-Degenerate (any |B_par,k| < 0.15 mT, non-weak)
    3. Weak Splitting (2*gamma_e*max|B_par,k| < 2*HWHM)
    4. Grid-Cropped (grid_variant_id == 2)

    Guarantees ``len(set(idx for idx, _ in selected)) == 12``.
    """
    reg = arrays["aux_regime_id"]
    weak = arrays["aux_weak_splitting"]
    deg_any = np.any(arrays["aux_axis_degenerate"], axis=1)
    cropped = arrays["aux_grid_variant_id"] == 2

    selected: List[Tuple[int, str]] = []
    used: set[int] = set()

    def _pick_first_unused(cands: np.ndarray, fallback: np.ndarray) -> int:
        for c in cands:
            ci = int(c)
            if ci not in used:
                used.add(ci)
                return ci
        for c in fallback:
            ci = int(c)
            if ci not in used:
                used.add(ci)
                return ci
        raise RuntimeError("Could not find an unused spectrum index.")

    for r_id, r_name in ((REGIME_EASY, "Easy"), (REGIME_MEDIUM, "Medium"), (REGIME_HARD, "Hard")):
        r_mask = reg == r_id
        r_all = np.flatnonzero(r_mask)
        idx1 = np.flatnonzero(r_mask & (~weak) & (~deg_any) & (~cropped))
        idx2 = np.flatnonzero(r_mask & deg_any & (~weak) & (~cropped))
        if idx2.size == 0:
            idx2 = np.flatnonzero(r_mask & deg_any & (~weak))
        idx3 = np.flatnonzero(r_mask & weak & (~cropped))
        if idx3.size == 0:
            idx3 = np.flatnonzero(r_mask & weak)
        idx4 = np.flatnonzero(r_mask & cropped)

        cands = [
            (_pick_first_unused(idx1, r_all), f"{r_name} (4-Axis Resolved)"),
            (_pick_first_unused(idx2, r_all), f"{r_name} (Axis-Degenerate)"),
            (_pick_first_unused(idx3, r_all), f"{r_name} (Weak Splitting)"),
            (_pick_first_unused(idx4, r_all), f"{r_name} (Grid-Cropped)"),
        ]
        selected.extend(cands)

    assert len(set(idx for idx, _ in selected)) == 12
    return selected

