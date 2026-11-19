"""Stage 2 noise, baseline drift, AC mains hum, and grid-realism models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.config import (
    A_HYPERFINE_MHZ,
    CANONICAL_FREQS,
    CENTRAL_REQUIRED_MAX_MHZ,
    CENTRAL_REQUIRED_MIN_MHZ,
    D0_MHZ,
    FREQ_MAX_MHZ,
    FREQ_MIN_MHZ,
    GAMMA_E_MHZ_MT,
    N_POINTS,
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
    canonicalize_b_vec,
    pseudo_voigt_dip,
    simulate_clean_odmr,
    solve_axis_transitions,
)

# Drift family integer identifiers
DRIFT_F1_POLY: int = 1
DRIFT_F2_SINE: int = 2
DRIFT_F3_EXP_OOD: int = 3
DRIFT_F4_GP: int = 4
DRIFT_F5_ID: int = 5
DRIFT_F5_OOD: int = 6

ID_DRIFT_FAMILIES: Tuple[int, ...] = (
    DRIFT_F1_POLY,
    DRIFT_F2_SINE,
    DRIFT_F4_GP,
    DRIFT_F5_ID,
)
OOD_DRIFT_FAMILIES: Tuple[int, ...] = (
    DRIFT_F3_EXP_OOD,
    DRIFT_F5_OOD,
)

DRIFT_FAMILY_NAMES: Dict[int, str] = {
    DRIFT_F1_POLY: "F1_poly_deg3",
    DRIFT_F2_SINE: "F2_lowfreq_sines",
    DRIFT_F3_EXP_OOD: "F3_exp_warmup_ood",
    DRIFT_F4_GP: "F4_smooth_gp",
    DRIFT_F5_ID: "F5_id_poly_plus_sine",
    DRIFT_F5_OOD: "F5_ood_exp_plus_gp",
}

# Stochastic noise type identifiers
NOISE_GAUSSIAN_ID: int = 0
NOISE_STUDENT_T_SPIKES_OOD: int = 1

# Grid realism variant identifiers
GRID_CANONICAL: int = 0
GRID_PERTURBED_FULL: int = 1
GRID_PERTURBED_CROP: int = 2

# Regime identifiers (by eff_snr_white)
REGIME_HARD: int = 0    # eff_snr_white in [0.5, 1.5)
REGIME_MEDIUM: int = 1  # eff_snr_white in [1.5, 3.0]
REGIME_EASY: int = 2    # eff_snr_white in (3.0, 20.0]

REGIME_BOUNDS: Dict[int, Tuple[float, float]] = {
    REGIME_HARD: (0.5, 1.5 - 1e-7),
    REGIME_MEDIUM: (1.5, 3.0),
    REGIME_EASY: (3.0 + 1e-7, 20.0),
}

# Natural log-uniform regime probabilities on [0.5, 20.0]
_LN_40: float = float(np.log(20.0 / 0.5))
NATURAL_REGIME_PROBS: Dict[int, float] = {
    REGIME_HARD: float(np.log(1.5 / 0.5) / _LN_40),
    REGIME_MEDIUM: float(np.log(3.0 / 1.5) / _LN_40),
    REGIME_EASY: float(np.log(20.0 / 3.0) / _LN_40),
}

_STUDENT_T_DF: float = 3.5
_STUDENT_T_SCALE: float = float(np.sqrt(_STUDENT_T_DF / (_STUDENT_T_DF - 2.0)))
_LN_2: float = float(np.log(2.0))
_SQRT_2: float = float(np.sqrt(2.0))
_DEGENERATE_B_PAR_THRESH_MT: float = 0.15

# Precomputed grids and FFT scales for the canonical 512-point sweep
_U_512: np.ndarray = np.linspace(-1.0, 1.0, N_POINTS, dtype=np.float64)
_U2_512: np.ndarray = _U_512 * _U_512 - (1.0 / 3.0)
_U3_512: np.ndarray = _U_512 * _U_512 * _U_512 - 0.6 * _U_512
_TAU_512: np.ndarray = np.linspace(0.0, 1.0, N_POINTS, dtype=np.float64)
_RFFTFREQ_512: np.ndarray = np.fft.rfftfreq(N_POINTS)
_PINK_SCALE_512: np.ndarray = np.zeros_like(_RFFTFREQ_512)
_PINK_SCALE_512[1:] = 1.0 / np.sqrt(_RFFTFREQ_512[1:])


def _fast_generate_pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Bit-identical fast version of ``src.noise.generate_pink_noise`` using cached 512-pt FFT scale."""
    if n != N_POINTS:
        return generate_pink_noise(n, rng=rng)
    white = rng.standard_normal(N_POINTS)
    pink = np.fft.irfft(np.fft.rfft(white) * _PINK_SCALE_512, n=N_POINTS)
    pink -= float(pink.mean())
    std = float(np.sqrt(np.dot(pink, pink) / float(N_POINTS)))
    if std > 1e-12:
        pink /= std
    return pink


def regime_id_from_eff_snr_white(eff_snr_white: float) -> int:
    """Map scalar ``eff_snr_white`` to regime id ``0`` (hard), ``1`` (medium), or ``2`` (easy)."""
    val = float(eff_snr_white)
    if val < 1.5:
        return REGIME_HARD
    if val <= 3.0:
        return REGIME_MEDIUM
    return REGIME_EASY


def evaluate_transitions_on_grid(
    freqs: np.ndarray,
    fm_arr: np.ndarray,
    fp_arr: np.ndarray,
    cf_arr: np.ndarray,
    hwhm_mhz: float,
    contrast: float,
    eta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate 4-axis clean ODMR spectrum and per-axis single-branch dip depths in one pass."""
    eff_amp = (contrast * 0.28 / 3.0) * cf_arr  # (4,)
    f0 = np.empty((4, 6), dtype=np.float64)
    f0[:, 0] = fm_arr - A_HYPERFINE_MHZ
    f0[:, 1] = fm_arr
    f0[:, 2] = fm_arr + A_HYPERFINE_MHZ
    f0[:, 3] = fp_arr - A_HYPERFINE_MHZ
    f0[:, 4] = fp_arr
    f0[:, 5] = fp_arr + A_HYPERFINE_MHZ

    diff = freqs[None, None, :] - f0[:, :, None]
    u_sq = (diff * diff) * (1.0 / (hwhm_mhz * hwhm_mhz))
    comp = (1.0 - eta) / (u_sq + 1.0) + eta * np.exp(-_LN_2 * u_sq)  # (4, 6, M)
    branch_minus = eff_amp[:, None] * (comp[:, 0, :] + comp[:, 1, :] + comp[:, 2, :])
    branch_plus = eff_amp[:, None] * (comp[:, 3, :] + comp[:, 4, :] + comp[:, 5, :])
    per_axis_depths = np.max(branch_minus, axis=1)
    spectrum = 1.0 - np.sum(branch_minus + branch_plus, axis=0)
    return spectrum, per_axis_depths


def fast_simulate_single_clean_odmr(
    freqs: np.ndarray,
    b_vec: np.ndarray,
    d0_mhz: float,
    hwhm_mhz: float,
    contrast: float,
    eta: float,
    e_mhz: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast single-spectrum exact spin-1 ODMR evaluator matching ``simulate_clean_odmr`` to <1e-14.

    Args:
        freqs: 1D frequency grid in MHz of shape ``(M,)``.
        b_vec: 3D magnetic field vector in mT of shape ``(3,)``.
        d0_mhz: Zero-field splitting in MHz.
        hwhm_mhz: Hyperfine line HWHM in MHz.
        contrast: Base optical contrast.
        eta: Pseudo-Voigt mixing factor in ``[0, 1]``.
        e_mhz: Transverse strain splitting in MHz (0.0 in Stage 2).

    Returns:
        Tuple ``(spectrum, splittings_sorted, b_par, b_perp, f_minus, f_plus, contrast_factor)``.
    """
    b_par = np.abs(NV_AXES @ b_vec)
    b_sq = float(b_vec[0] * b_vec[0] + b_vec[1] * b_vec[1] + b_vec[2] * b_vec[2])
    b_perp = np.sqrt(np.maximum(0.0, b_sq - b_par * b_par))

    bz = GAMMA_E_MHZ_MT * b_par
    bx = GAMMA_E_MHZ_MT * b_perp / _SQRT_2
    h = np.zeros((4, 3, 3), dtype=np.float64)
    h[:, 0, 0] = d0_mhz + bz
    h[:, 0, 1] = bx
    h[:, 1, 0] = bx
    if e_mhz != 0.0:
        h[:, 0, 2] = e_mhz
        h[:, 2, 0] = e_mhz
    h[:, 1, 2] = bx
    h[:, 2, 1] = bx
    h[:, 2, 2] = d0_mhz - bz

    eigvals, eigvecs = np.linalg.eigh(h)
    prob_ms0 = eigvecs[:, 1, :] * eigvecs[:, 1, :]
    if (
        prob_ms0[0, 0] >= prob_ms0[0, 1]
        and prob_ms0[1, 0] >= prob_ms0[1, 1]
        and prob_ms0[2, 0] >= prob_ms0[2, 1]
        and prob_ms0[3, 0] >= prob_ms0[3, 1]
    ):
        purity_ms0 = prob_ms0[:, 0]
        e0 = eigvals[:, 0]
        fm_arr = eigvals[:, 1] - e0
        fp_arr = eigvals[:, 2] - e0
    else:
        idx0 = np.argmax(prob_ms0, axis=-1)
        purity_ms0 = np.take_along_axis(prob_ms0, idx0[:, None], axis=-1)[:, 0]
        e0 = np.take_along_axis(eigvals, idx0[:, None], axis=-1)[:, 0]
        mask = np.arange(3)[None, :] != idx0[:, None]
        other = eigvals[mask].reshape(4, 2)
        fm_arr = other[:, 0] - e0
        fp_arr = other[:, 1] - e0

    cf_arr = purity_ms0 * purity_ms0

    spectrum, per_axis_depths = evaluate_transitions_on_grid(
        freqs, fm_arr, fp_arr, cf_arr, hwhm_mhz, contrast, eta
    )
    splittings_sorted = np.sort(fp_arr - fm_arr)
    return spectrum, splittings_sorted, b_par, b_perp, fm_arr, fp_arr, cf_arr, per_axis_depths


def compute_depth_outer_and_flags(
    b_vec: np.ndarray,
    d0_mhz: float,
    hwhm_mhz: float,
    contrast: float,
    eta: float,
    e_mhz: float = 0.0,
    freqs: np.ndarray = CANONICAL_FREQS,
    precomputed_transitions: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    precomputed_axis_depths: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute ``depth_outer`` (with fallback), ``weak_splitting``, and ``axis_degenerate`` flags.

    Implements Part 1.1:
      - ``depth_outer`` is the deepest clean single-axis dip among axes with
        ``|B_par,k| > 0.15`` mT (evaluating a single transition branch so the
        central merged dip at D0 is never used).
      - FALLBACK: if no axis has ``|B_par,k| > 0.15`` mT (possible when
        ``|B| <= 0.15 * sqrt(3) ~= 0.2598`` mT), uses the axis with the largest
        ``|B_par,k|``. ``depth_outer`` is strictly ``> 0``.
      - ``weak_splitting = (2 * gamma_e * max_k |B_par,k| < 2 * HWHM)``.
      - ``axis_degenerate = (|B_par,k| < 0.15 mT)`` per axis ``k in {0,1,2,3}``.
    """
    b_arr = np.asarray(b_vec, dtype=np.float64)
    if precomputed_transitions is not None:
        b_par, fm_arr, cf_arr = precomputed_transitions
    else:
        b_par = np.abs(NV_AXES @ b_arr)
        b_sq = float(np.dot(b_arr, b_arr))
        b_perp = np.sqrt(np.maximum(0.0, b_sq - b_par * b_par))
        fm, _, cf = solve_axis_transitions(b_par, b_perp, d0=d0_mhz, e_mhz=e_mhz)
        fm_arr = np.asarray(fm, dtype=np.float64)
        cf_arr = np.asarray(cf, dtype=np.float64)

    axis_degenerate = b_par < _DEGENERATE_B_PAR_THRESH_MT
    eligible_indices = np.flatnonzero(~axis_degenerate)
    used_fallback = False
    if eligible_indices.size == 0:
        eligible_indices = np.array([int(np.argmax(b_par))], dtype=np.int64)
        used_fallback = True

    if precomputed_axis_depths is not None:
        per_axis_depths = precomputed_axis_depths
    else:
        inv_gamma_sq = 1.0 / (hwhm_mhz * hwhm_mhz)
        eff_amp = (contrast * 0.28 / 3.0) * cf_arr
        f0 = np.empty((4, 3), dtype=np.float64)
        f0[:, 0] = fm_arr - A_HYPERFINE_MHZ
        f0[:, 1] = fm_arr
        f0[:, 2] = fm_arr + A_HYPERFINE_MHZ
        diff = freqs[None, None, :] - f0[:, :, None]
        u_sq = (diff * diff) * inv_gamma_sq
        comp = (1.0 - eta) / (u_sq + 1.0) + eta * np.exp(-_LN_2 * u_sq)
        branch_minus = eff_amp[:, None] * (comp[:, 0, :] + comp[:, 1, :] + comp[:, 2, :])
        per_axis_depths = np.max(branch_minus, axis=1)

    depth_outer = float(np.max(per_axis_depths[eligible_indices]))
    max_b_par = float(np.max(b_par))
    weak_splitting = bool((2.0 * GAMMA_E_MHZ_MT * max_b_par) < (2.0 * float(hwhm_mhz)))

    return {
        "depth_outer": depth_outer,
        "per_axis_depths": per_axis_depths,
        "b_par": b_par,
        "axis_degenerate": axis_degenerate,
        "weak_splitting": weak_splitting,
        "used_fallback": used_fallback,
    }


def generate_drift_component(
    n_points: int,
    family_id: int,
    drift_amp: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesize baseline drift on ``n_points`` for the specified ``family_id``.

    Args:
        n_points: Number of frequency steps on the simulated measurement grid.
        family_id: Drift family identifier in ``{1, 2, 3, 4, 5, 6}``.
        drift_amp: Target peak absolute drift amplitude ``max(|drift|)``.
        rng: Random number generator.

    Returns:
        Tuple ``(drift_curve, drift_params_4d)`` where ``drift_params_4d`` records
        up to 4 compact generative parameters.
    """
    if drift_amp <= 0.0 or n_points <= 0:
        return np.zeros(n_points, dtype=np.float64), np.zeros(4, dtype=np.float64)

    if n_points == N_POINTS:
        u = _U_512
        tau = _TAU_512
    else:
        u = np.linspace(-1.0, 1.0, n_points, dtype=np.float64)
        tau = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    params = np.zeros(4, dtype=np.float64)
    params[0] = float(drift_amp)

    def _norm_unit(curve: np.ndarray) -> np.ndarray:
        pk = float(np.abs(curve).max())
        return curve / pk if pk > 1e-12 else np.zeros_like(curve)

    if family_id == DRIFT_F1_POLY:
        c1 = float(rng.normal(0.0, 0.5))
        c2 = float(rng.normal(0.0, 0.5))
        c3 = float(rng.normal(0.0, 0.3))
        if abs(c1) + abs(c2) + abs(c3) < 1e-9:
            c1 = 0.5
        if n_points == N_POINTS:
            raw = c1 * _U_512 + c2 * _U2_512 + c3 * _U3_512
        else:
            raw = c1 * u + c2 * (u * u - 1.0 / 3.0) + c3 * (u * u * u - 0.6 * u)
        params[1], params[2], params[3] = c1, c2, c3
        unit = _norm_unit(raw)

    elif family_id == DRIFT_F2_SINE:
        n_sines = int(rng.integers(1, 4))
        raw = np.zeros(n_points, dtype=np.float64)
        first_period = 1.0
        first_phase = 0.0
        for s_i in range(n_sines):
            period_win = float(rng.uniform(0.5, 4.0))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            amp_s = float(rng.uniform(0.35, 1.0))
            raw += amp_s * np.sin((2.0 * np.pi / period_win) * tau + phase)
            if s_i == 0:
                first_period = period_win
                first_phase = phase
        raw -= float(raw.mean())
        params[1], params[2], params[3] = float(n_sines), first_period, first_phase
        unit = _norm_unit(raw)

    elif family_id == DRIFT_F3_EXP_OOD:
        tau_0 = float(rng.uniform(0.12, 0.45))
        sign_exp = float(rng.choice([-1.0, 1.0]))
        lin_slope = float(rng.uniform(-0.40, 0.40))
        raw = sign_exp * (1.0 - np.exp(-tau / tau_0)) + lin_slope * (tau - 0.5)
        raw -= float(raw.mean())
        params[1], params[2], params[3] = tau_0, sign_exp, lin_slope
        unit = _norm_unit(raw)

    elif family_id == DRIFT_F4_GP:
        corr_frac = float(rng.uniform(0.12, 0.35))
        sigma_pts = max(3.0, corr_frac * float(n_points))
        wn = rng.standard_normal(n_points)
        freqs_fft = _RFFTFREQ_512 if n_points == N_POINTS else np.fft.rfftfreq(n_points)
        kernel_fft = np.exp(-2.0 * (np.pi * sigma_pts * freqs_fft) ** 2)
        raw = np.fft.irfft(np.fft.rfft(wn) * kernel_fft, n=n_points)
        raw -= float(raw.mean())
        params[1], params[2] = corr_frac, sigma_pts
        unit = _norm_unit(raw)

    elif family_id == DRIFT_F5_ID:
        d1, p1 = generate_drift_component(n_points, DRIFT_F1_POLY, 1.0, rng)
        d2, p2 = generate_drift_component(n_points, DRIFT_F2_SINE, 1.0, rng)
        w1 = float(rng.uniform(0.35, 0.65))
        raw = w1 * d1 + (1.0 - w1) * d2
        params[1], params[2], params[3] = w1, p1[1], p2[2]
        unit = _norm_unit(raw)

    elif family_id == DRIFT_F5_OOD:
        d3, p3 = generate_drift_component(n_points, DRIFT_F3_EXP_OOD, 1.0, rng)
        d4, p4 = generate_drift_component(n_points, DRIFT_F4_GP, 1.0, rng)
        w3 = float(rng.uniform(0.35, 0.65))
        raw = w3 * d3 + (1.0 - w3) * d4
        params[1], params[2], params[3] = w3, p3[1], p4[1]
        unit = _norm_unit(raw)

    else:
        raise ValueError(f"Unknown drift family_id={family_id}.")

    return float(drift_amp) * unit, params


def generate_hum_component(
    n_points: int,
    hum_amp: float,
    dwell_time_ms: float,
    is_ood_harmonics: bool,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Synthesize time-domain AC mains hum with fundamental and 2nd/3rd harmonics.

    Args:
        n_points: Number of frequency steps on the simulated measurement grid.
        hum_amp: Peak hum amplitude (``0.0`` when hum is absent).
        dwell_time_ms: Dwell time per point in milliseconds (``[0.3, 2.0]`` ms).
        is_ood_harmonics: If False (ID), relative harmonic ratios ``r2, r3 in [0.0, 0.50]``.
            If True (OOD), relative harmonic ratios ``r2, r3 in (0.50, 0.85]``.
        rng: Random number generator.

    Returns:
        Dictionary with ``"hum"``, ``"freq_hz"``, ``"phases_rad"`` (3,),
        ``"rel_harmonics"`` (2,), and ``"harmonics_amp"`` (3,).
    """
    base_freq = float(rng.choice([50.0, 60.0]))
    jitter = float(rng.uniform(-0.5, 0.5))
    freq_hz = base_freq + jitter
    phases = rng.uniform(0.0, 2.0 * np.pi, size=3).astype(np.float64)

    if is_ood_harmonics:
        r2 = float(rng.uniform(0.501, 0.850))
        r3 = float(rng.uniform(0.501, 0.850))
    else:
        r2 = float(rng.uniform(0.0, 0.500)) if float(rng.random()) < 0.65 else 0.0
        r3 = float(rng.uniform(0.0, 0.500)) if float(rng.random()) < 0.65 else 0.0

    rel_harmonics = np.array([r2, r3], dtype=np.float64)

    if hum_amp <= 0.0 or n_points <= 0:
        return {
            "hum": np.zeros(n_points, dtype=np.float64),
            "freq_hz": 0.0,
            "phases_rad": phases,
            "rel_harmonics": rel_harmonics,
            "harmonics_amp": np.zeros(3, dtype=np.float64),
        }

    t_sec = np.arange(n_points, dtype=np.float64) * (float(dwell_time_ms) * 1e-3)
    omega = 2.0 * np.pi * freq_hz
    raw = (
        np.sin(omega * t_sec + phases[0])
        + r2 * np.sin(2.0 * omega * t_sec + phases[1])
        + r3 * np.sin(3.0 * omega * t_sec + phases[2])
    )
    peak = max(float(np.max(np.abs(raw))), 1e-12)
    scale = float(hum_amp) / peak
    hum = scale * raw
    harmonics_amp = np.array([scale, scale * r2, scale * r3], dtype=np.float64)

    return {
        "hum": hum,
        "freq_hz": freq_hz,
        "phases_rad": phases,
        "rel_harmonics": rel_harmonics,
        "harmonics_amp": harmonics_amp,
    }


def generate_stochastic_and_gain_components(
    clean_sim: np.ndarray,
    sigma_white_eff: float,
    pink_ratio: float,
    noise_type: int,
    gain_amp: float,
    rng: np.random.Generator,
    drift_sim: np.ndarray,
    hum_sim: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Synthesize white, pink, and slow multiplicative gain components on the simulated grid.

    Normalizes ``white_sim + pink_sim`` so that ``std(white_sim + pink_sim) == sigma_white_eff``
    holds to machine precision on the simulated grid.
    """
    n_points = clean_sim.shape[0]
    intensity_scale = np.sqrt(np.clip(clean_sim, 0.5, 1.5))

    if noise_type == NOISE_STUDENT_T_SPIKES_OOD:
        w_unit = rng.standard_t(df=_STUDENT_T_DF, size=n_points) / _STUDENT_T_SCALE
        n_spikes = int(rng.integers(2, 5))
        spike_idx = rng.choice(n_points, size=min(n_spikes, n_points), replace=False)
        spike_signs = rng.choice([-1.0, 1.0], size=spike_idx.size)
        spike_mags = rng.uniform(5.0, 8.5, size=spike_idx.size)
        w_unit[spike_idx] += spike_signs * spike_mags
    elif noise_type == NOISE_GAUSSIAN_ID:
        w_unit = rng.standard_normal(n_points)
    else:
        raise ValueError(f"Unknown noise_type={noise_type}.")

    r_p = float(np.clip(pink_ratio, 0.0, 0.5))
    w_centered = w_unit - float(w_unit.mean())
    w_raw = intensity_scale * np.sqrt(1.0 - r_p) * w_centered

    if r_p > 1e-9:
        p_unit = _fast_generate_pink_noise(n_points, rng=rng)
        p_centered = p_unit - float(p_unit.mean())
        p_raw = intensity_scale * np.sqrt(r_p) * p_centered
    else:
        p_raw = np.zeros(n_points, dtype=np.float64)

    w_raw -= float(w_raw.mean())
    p_raw -= float(p_raw.mean())
    stoch_raw = w_raw + p_raw
    raw_std = float(np.sqrt(np.dot(stoch_raw, stoch_raw) / float(n_points)))
    scale = float(sigma_white_eff) / max(raw_std, 1e-15)

    white_sim = scale * w_raw
    pink_sim = scale * p_raw

    if gain_amp > 0.0:
        g1 = float(rng.normal(0.0, 1.0))
        g2 = float(rng.normal(0.0, 0.5))
        if n_points == N_POINTS:
            g_raw = g1 * _U_512 + g2 * _U2_512
        else:
            u = np.linspace(-1.0, 1.0, n_points, dtype=np.float64)
            g_raw = g1 * u + g2 * (u * u - 1.0 / 3.0)
        g_pk = float(np.abs(g_raw).max())
        g_curve = (float(gain_amp) * (g_raw / g_pk)) if g_pk > 1e-12 else np.zeros(n_points, dtype=np.float64)
        gain_mod_sim = g_curve * (clean_sim + drift_sim + hum_sim + pink_sim + white_sim)
    else:
        gain_mod_sim = np.zeros(n_points, dtype=np.float64)

    return {
        "white_sim": white_sim,
        "pink_sim": pink_sim,
        "gain_mod_sim": gain_mod_sim,
    }


def generate_simulated_grid(
    grid_prob: float,
    aggressive_crop: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    """Generate the measurement frequency grid ``freqs_sim`` and ``grid_variant_id``.

    Args:
        grid_prob: Probability of activating grid perturbation (``0.30`` ID, ``0.40`` OOD,
            ``1.00`` for ``ood_grid``).
        aggressive_crop: If True (``ood_grid``), always crop to ``[2790, 2950]`` MHz
            covering ``[2830, 2910]`` MHz.
        rng: Random number generator.

    Returns:
        Tuple ``(freqs_sim, grid_variant_id)``.
    """
    u_draw = float(rng.random())
    if not aggressive_crop and u_draw >= float(grid_prob):
        return CANONICAL_FREQS, GRID_CANONICAL

    n_pts = int(rng.integers(200, 1002))
    offset_lo = float(rng.uniform(-0.30, 0.30))
    offset_hi = float(rng.uniform(-0.30, 0.30))

    if aggressive_crop:
        variant_id = GRID_PERTURBED_CROP
        f_min = float(rng.uniform(2790.0, 2827.5)) + offset_lo
        f_max = float(rng.uniform(2912.5, 2950.0)) + offset_hi
    else:
        if float(rng.random()) < 0.5:
            variant_id = GRID_PERTURBED_FULL
            f_min = FREQ_MIN_MHZ + offset_lo
            f_max = FREQ_MAX_MHZ + offset_hi
        else:
            variant_id = GRID_PERTURBED_CROP
            f_min = float(rng.uniform(2760.0, 2827.5)) + offset_lo
            f_max = float(rng.uniform(2912.5, 2980.0)) + offset_hi

    f_min = min(f_min, CENTRAL_REQUIRED_MIN_MHZ - 0.5)
    f_max = max(f_max, CENTRAL_REQUIRED_MAX_MHZ + 0.5)
    freqs_sim = np.linspace(f_min, f_max, n_pts, dtype=np.float64)
    return freqs_sim, variant_id


def evaluate_stage15_eff_snr_consistency() -> Dict[float, Dict[str, float]]:
    """Verify consistency between Stage 2 ``eff_snr_total`` and Stage 1.5 FINAL medians (Part 1.2).

    Reproduces the Stage 1.5 FINAL dataset configuration (``FINAL_SEED = 500001``,
    ``N = 200`` spectra on ``S^2``) across nominal SNRs ``15.0``, ``4.0``, and ``2.0``,
    comparing Stage 2 ``eff_snr_total`` against the reference medians recorded in
    ``docs/snr_regimes.md`` (``3.17``, ``1.10``, ``0.68``).
    """
    final_seed = 500001
    n_spectra = 200
    rng = np.random.default_rng(final_seed)

    spectra_items = []
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
        clean, _, b_par, _, fm_arr, _, cf_arr, axis_depths = fast_simulate_single_clean_odmr(
            CANONICAL_FREQS, b_vec, d0, gamma, contrast, eta=0.35
        )
        depth_info = compute_depth_outer_and_flags(
            b_vec,
            d0,
            gamma,
            contrast,
            eta=0.35,
            precomputed_transitions=(b_par, fm_arr, cf_arr),
            precomputed_axis_depths=axis_depths,
        )
        spectra_items.append(
            {
                "idx": i,
                "clean": clean,
                "depth_outer": float(depth_info["depth_outer"]),
            }
        )

    stage15_configs = [
        (
            15.0,
            3.17,
            HardwareNoiseConfig(
                target_snr=15.0,
                dwell_time_ms=0.8,
                mains_freq_hz=50.0,
                hum_amp=0.0015,
                baseline_drift_amp=0.005,
                pink_noise_ratio=0.25,
            ),
        ),
        (
            4.0,
            1.10,
            HardwareNoiseConfig(
                target_snr=4.0,
                dwell_time_ms=0.8,
                mains_freq_hz=50.0,
                hum_amp=0.0030,
                baseline_drift_amp=0.012,
                pink_noise_ratio=0.25,
            ),
        ),
        (
            2.0,
            0.68,
            HardwareNoiseConfig(
                target_snr=2.0,
                dwell_time_ms=0.8,
                mains_freq_hz=50.0,
                hum_amp=0.0035,
                baseline_drift_amp=0.015,
                pink_noise_ratio=0.25,
            ),
        ),
    ]

    results: Dict[float, Dict[str, float]] = {}
    for nom_snr, ref_med, cfg in stage15_configs:
        eff_totals = []
        for item in spectra_items:
            idx = int(item["idx"])
            trial_seed = final_seed + int(nom_snr * 1000) + idx * 29
            trial_rng = np.random.default_rng(trial_seed)
            out = apply_realistic_noise(item["clean"], cfg, rng=trial_rng, return_components=True)
            assert isinstance(out, tuple)
            _, comps = out
            sigma_w = float(np.std(comps["white"] + comps["pink"]))
            rms_d = float(np.sqrt(np.mean(comps["drift"] ** 2)))
            rms_h = float(np.sqrt(np.mean(comps["hum"] ** 2)))
            denom = float(np.sqrt(sigma_w * sigma_w + rms_d * rms_d + rms_h * rms_h))
            eff_totals.append(float(item["depth_outer"]) / max(denom, 1e-12))

        med_val = float(np.median(eff_totals))
        rel_err = abs(med_val - ref_med) / ref_med
        results[nom_snr] = {
            "stage2_eff_snr_total_median": med_val,
            "stage15_ref_median": ref_med,
            "relative_diff": rel_err,
        }

    return results
