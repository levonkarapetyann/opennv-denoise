"""Realistic COTS hardware noise simulation and safe PCHIP resampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

from src.config import (
    A_HYPERFINE_MHZ,
    CANONICAL_FREQS,
    CENTRAL_REQUIRED_MAX_MHZ,
    CENTRAL_REQUIRED_MIN_MHZ,
    D0_MHZ,
    GAMMA_E_MHZ_MT,
)

_STUDENT_T_DF: float = 3.5
_STUDENT_T_SCALE: float = float(np.sqrt(_STUDENT_T_DF / (_STUDENT_T_DF - 2.0)))


@dataclass
class HardwareNoiseConfig:
    """Noise and artifact parameters for COTS NV-ODMR hardware simulation.

    Attributes:
        target_snr: Signal-to-noise ratio defined as ``signal_depth / noise_std``.
        dwell_time_ms: Integration dwell time per microwave frequency step (ms).
        mains_freq_hz: AC power-line interference frequency (Hz).
        hum_amp: Sinusoidal mains hum amplitude.
        baseline_drift_amp: Peak thermal laser baseline drift amplitude.
        pink_noise_ratio: Fraction of stochastic noise variance from 1/f noise.
        use_heavy_tails: If True, sample Student-t (df=3.5) white noise for OOD tests.
    """

    target_snr: float = 4.0
    dwell_time_ms: float = 0.8
    mains_freq_hz: float = 50.0
    hum_amp: float = 0.003
    baseline_drift_amp: float = 0.012
    pink_noise_ratio: float = 0.25
    use_heavy_tails: bool = False


def generate_pink_noise(
    n: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Generate zero-mean, unit-variance 1/f flicker noise via FFT scaling.

    Args:
        n: Number of samples to generate.
        rng: Optional NumPy random generator.

    Returns:
        1D array of shape ``(n,)`` with approximately 1/f power spectral density.
    """
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng()
    if n == 1:
        return rng.standard_normal(1)

    white = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n)
    scale = np.zeros_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])

    spectrum = np.fft.rfftfreq(n)
    spectrum_fft = np.fft.rfft(white) * scale
    pink = np.fft.irfft(spectrum_fft, n=n)
    pink = pink - np.mean(pink)
    std = float(np.std(pink))
    if std > 1e-12:
        pink = pink / std
    return pink


def apply_realistic_noise(
    clean_spectrum: np.ndarray,
    cfg: HardwareNoiseConfig,
    rng: Optional[np.random.Generator] = None,
    return_components: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """Corrupt a clean ODMR spectrum with realistic COTS hardware artifacts.

    Includes smooth quadratic/cubic thermal laser baseline drift, time-domain
    AC mains hum, 1/f flicker noise, intensity-dependent noise scaling
    (``sqrt(clip(clean, 0.5, 1.5))``), and optional heavy-tailed Student-t
    fluctuations.

    Args:
        clean_spectrum: 1D clean ODMR spectrum normalized near 1.0.
        cfg: ``HardwareNoiseConfig`` specifying noise and drift levels.
        rng: Optional NumPy random generator.
        return_components: If True, return ``(noisy_spectrum, components_dict)``
            containing ``"clean"``, ``"drift"``, ``"hum"``, ``"pink"``,
            ``"white"``, ``"stochastic"``, and ``"total"``.

    Returns:
        Noisy 1D ODMR spectrum when ``return_components=False``, or a tuple
        ``(noisy_spectrum, components_dict)`` when ``return_components=True``.
    """
    if rng is None:
        rng = np.random.default_rng()

    clean = np.asarray(clean_spectrum, dtype=np.float64)
    n_points = clean.shape[0]
    time_sec = np.arange(n_points, dtype=np.float64) * (cfg.dwell_time_ms * 1e-3)

    # Smooth quadratic/cubic thermal laser baseline drift across the sweep
    u = np.linspace(-1.0, 1.0, n_points, dtype=np.float64)
    raw_poly = (
        rng.normal(0.0, 0.5) * u
        + rng.normal(0.0, 0.5) * (u * u - 1.0 / 3.0)
        + rng.normal(0.0, 0.3) * (u * u * u)
    )
    max_poly = float(np.max(np.abs(raw_poly)))
    if cfg.baseline_drift_amp > 0.0 and max_poly > 1e-12:
        drift = (
            cfg.baseline_drift_amp
            * (raw_poly / max_poly)
            * rng.uniform(0.6, 1.0)
        )
    else:
        drift = np.zeros(n_points, dtype=np.float64)

    # Time-domain AC mains hum
    phase = rng.uniform(0.0, 2.0 * np.pi)
    hum = cfg.hum_amp * np.sin(2.0 * np.pi * cfg.mains_freq_hz * time_sec + phase)

    # Stochastic white + 1/f noise with intensity-dependent noise scaling
    signal_depth = max(float(np.max(clean) - np.min(clean)), 1e-6)
    noise_std = signal_depth / float(cfg.target_snr)
    intensity_scale = np.sqrt(np.clip(clean, 0.5, 1.5))

    if cfg.use_heavy_tails:
        white_unit = rng.standard_t(df=_STUDENT_T_DF, size=n_points) / _STUDENT_T_SCALE
    else:
        white_unit = rng.standard_normal(n_points)

    pink_ratio = float(np.clip(cfg.pink_noise_ratio, 0.0, 1.0))
    if pink_ratio > 0.0:
        pink_unit = generate_pink_noise(n_points, rng)
        white_comp = noise_std * intensity_scale * np.sqrt(1.0 - pink_ratio) * white_unit
        pink_comp = noise_std * intensity_scale * np.sqrt(pink_ratio) * pink_unit
        combined = np.sqrt(1.0 - pink_ratio) * white_unit + np.sqrt(pink_ratio) * pink_unit
    else:
        pink_comp = np.zeros(n_points, dtype=np.float64)
        white_comp = noise_std * intensity_scale * white_unit
        combined = white_unit

    stochastic_noise = noise_std * intensity_scale * combined
    total = clean + drift + hum + stochastic_noise

    if not return_components:
        return total

    components = {
        "clean": clean.copy(),
        "drift": drift,
        "hum": hum,
        "pink": pink_comp,
        "white": white_comp,
        "stochastic": stochastic_noise,
        "total": total,
    }
    return total, components


def resample_spectrum_pchip(
    freqs_in: np.ndarray,
    spectrum_in: np.ndarray,
    target_freqs: np.ndarray = CANONICAL_FREQS,
    method: str = "pchip",
    rng: Optional[np.random.Generator] = None,
    return_valid_mask: bool = False,
    assumed_hwhm_mhz: float = 4.5,
) -> Union[Tuple[np.ndarray, Dict[str, Any]], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """Safely resample an arbitrary ODMR frequency grid onto ``target_freqs``.

    Validates central NV resonance coverage, deduplicates and sorts input
    frequencies, applies exact cell-averaging integration prior to downsampling
    (eliminating high-frequency aliasing without Savitzky-Golay phase/peak
    distortion), interpolates without polynomial extrapolation, and computes a
    boolean ``valid_mask`` (``False`` where ``target_freqs`` lies outside the
    input frequency range) plus an ``out_of_range`` flag when resonances approach
    within ``2 * HWHM + 2 * A_par`` of the input window edges. Out-of-bounds
    target points are held at the deterministic median of the outermost 5% edge
    points (no random noise injection).

    Args:
        freqs_in: Input microwave frequencies in MHz.
        spectrum_in: Input ODMR intensity values matching ``freqs_in``.
        target_freqs: Target canonical frequency grid in MHz.
        method: Interpolation method, ``"pchip"`` or ``"linear"``.
        rng: Optional random generator (accepted for backward compatibility;
            unused since out-of-bounds edges use deterministic median fill).
        return_valid_mask: If True, return a 3-tuple
            ``(resampled_spectrum, valid_mask, meta_dict)``. Default False
            returns ``(resampled_spectrum, meta_dict)`` where ``valid_mask`` is
            stored in ``meta_dict["valid_mask"]``.
        assumed_hwhm_mhz: Reference HWHM in MHz used to evaluate the edge
            guard margin ``2 * HWHM + 2 * A_par`` for ``out_of_range`` detection.

    Returns:
        Tuple ``(resampled_spectrum, meta_dict)`` (or
        ``(resampled_spectrum, valid_mask, meta_dict)`` when
        ``return_valid_mask=True``), where ``meta_dict`` contains:
        ``valid_mask``, ``out_of_range``, ``max_untruncated_b_mt``,
        ``window_padded``, ``original_min_mhz``, ``original_max_mhz``, and
        ``original_points``.

    Raises:
        ValueError: If input arrays mismatch in length, ``method`` is unknown,
            or ``freqs_in`` fails to cover ``[2830.0, 2910.0]`` MHz.
    """
    _ = rng
    freqs_arr = np.asarray(freqs_in, dtype=np.float64)
    spec_arr = np.asarray(spectrum_in, dtype=np.float64)
    target_arr = np.asarray(target_freqs, dtype=np.float64)

    if freqs_arr.ndim != 1 or spec_arr.ndim != 1 or freqs_arr.shape[0] != spec_arr.shape[0]:
        raise ValueError("freqs_in and spectrum_in must be 1D arrays of equal length.")
    if freqs_arr.shape[0] < 2:
        raise ValueError("At least 2 frequency points are required for resampling.")
    if method not in ("pchip", "linear"):
        raise ValueError(f"Unsupported resampling method '{method}'. Expected 'pchip' or 'linear'.")

    orig_min = float(np.min(freqs_arr))
    orig_max = float(np.max(freqs_arr))
    orig_points = int(freqs_arr.shape[0])

    if orig_min > CENTRAL_REQUIRED_MIN_MHZ or orig_max < CENTRAL_REQUIRED_MAX_MHZ:
        raise ValueError(
            "Input frequency range does not cover the required central NV resonance region [2830, 2910] MHz."
        )

    # Sort ascending and average any duplicate frequency coordinates
    order = np.argsort(freqs_arr)
    freqs_sorted = freqs_arr[order]
    spec_sorted = spec_arr[order]

    unique_freqs, inverse = np.unique(freqs_sorted, return_inverse=True)
    if unique_freqs.shape[0] < freqs_sorted.shape[0]:
        counts = np.bincount(inverse)
        spec_unique = np.bincount(inverse, weights=spec_sorted) / counts
        freqs_sorted = unique_freqs
        spec_sorted = spec_unique

    f_min = float(freqs_sorted[0])
    f_max = float(freqs_sorted[-1])
    valid_mask = (target_arr >= f_min) & (target_arr <= f_max)
    left_mask = target_arr < f_min
    right_mask = target_arr > f_max
    window_padded = bool(np.any(left_mask) or np.any(right_mask))

    n_in = freqs_sorted.shape[0]
    n_target = target_arr.shape[0]
    df_target = float(np.median(np.diff(target_arr))) if n_target > 1 else 0.47

    if method == "pchip":
        interpolator = PchipInterpolator(freqs_sorted, spec_sorted, extrapolate=False)
        if n_in > n_target and n_target > 1:
            # Cell-averaging anti-aliasing: integrate the continuous PCHIP spline over each target bin
            antideriv = interpolator.antiderivative()
            resampled = np.full(n_target, np.nan, dtype=np.float64)
            v_targets = target_arr[valid_mask]
            lo_edges = np.maximum(f_min, v_targets - 0.5 * df_target)
            hi_edges = np.minimum(f_max, v_targets + 0.5 * df_target)
            widths = np.maximum(hi_edges - lo_edges, 1e-12)
            resampled[valid_mask] = (antideriv(hi_edges) - antideriv(lo_edges)) / widths
        else:
            resampled = interpolator(target_arr)
    else:
        if n_in > n_target and n_target > 1:
            # Piecewise-linear cell-averaging via cumulative trapezoidal antiderivative
            cum_trapz = np.concatenate(
                [[0.0], np.cumsum(0.5 * (spec_sorted[1:] + spec_sorted[:-1]) * np.diff(freqs_sorted))]
            )
            resampled = np.full(n_target, np.nan, dtype=np.float64)
            v_targets = target_arr[valid_mask]
            lo_edges = np.maximum(f_min, v_targets - 0.5 * df_target)
            hi_edges = np.minimum(f_max, v_targets + 0.5 * df_target)
            widths = np.maximum(hi_edges - lo_edges, 1e-12)
            int_hi = np.interp(hi_edges, freqs_sorted, cum_trapz)
            int_lo = np.interp(lo_edges, freqs_sorted, cum_trapz)
            resampled[valid_mask] = (int_hi - int_lo) / widths
        else:
            resampled = np.interp(target_arr, freqs_sorted, spec_sorted, left=np.nan, right=np.nan)

    # Deterministic constant edge fill for out-of-bounds points (valid_mask is False there)
    if window_padded:
        n_edge = min(n_in, max(3, int(np.ceil(0.05 * n_in))))
        if np.any(left_mask):
            resampled[left_mask] = float(np.median(spec_sorted[:n_edge]))
        if np.any(right_mask):
            resampled[right_mask] = float(np.median(spec_sorted[-n_edge:]))

    # Check if the input window is too narrow for the observed resonance span (Task 6b):
    # Guard margin: outermost dip within 2 * HWHM + 2 * A_par of [f_min, f_max]
    edge_margin_mhz = 2.0 * float(assumed_hwhm_mhz) + 2.0 * A_HYPERFINE_MHZ
    half_window_mhz = min(D0_MHZ - f_min, f_max - D0_MHZ)
    max_untruncated_b_mt = max(0.0, (half_window_mhz - edge_margin_mhz) / GAMMA_E_MHZ_MT)

    # Detect if absorption dips in the input spectrum encroach on the guard zone near f_min or f_max
    df_in = float(np.median(np.diff(freqs_sorted))) if n_in > 1 else 0.5
    smoothed_in = gaussian_filter1d(spec_sorted, sigma=max(1.0, 2.5 / max(df_in, 0.05)), mode="nearest")
    baseline_level = float(np.percentile(smoothed_in, 85))
    dip_signal = np.maximum(0.0, baseline_level - smoothed_in)
    max_dip = float(np.max(dip_signal))
    out_of_range = False
    if max_dip > 1e-3:
        guard_mask = (freqs_sorted < (f_min + edge_margin_mhz)) | (freqs_sorted > (f_max - edge_margin_mhz))
        if np.any(guard_mask) and float(np.max(dip_signal[guard_mask])) > 0.35 * max_dip:
            out_of_range = True

    meta_dict: Dict[str, Any] = {
        "valid_mask": valid_mask,
        "out_of_range": bool(out_of_range),
        "max_untruncated_b_mt": float(max_untruncated_b_mt),
        "window_padded": window_padded,
        "original_min_mhz": orig_min,
        "original_max_mhz": orig_max,
        "original_points": orig_points,
    }
    if return_valid_mask:
        return resampled, valid_mask, meta_dict
    return resampled, meta_dict
