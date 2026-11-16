"""Realistic COTS hardware noise simulation and safe PCHIP resampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

from src.config import (
    CANONICAL_FREQS,
    CENTRAL_REQUIRED_MAX_MHZ,
    CENTRAL_REQUIRED_MIN_MHZ,
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

    spectrum = np.fft.rfft(white) * scale
    pink = np.fft.irfft(spectrum, n=n)
    pink = pink - np.mean(pink)
    std = float(np.std(pink))
    if std > 1e-12:
        pink = pink / std
    return pink


def apply_realistic_noise(
    clean_spectrum: np.ndarray,
    cfg: HardwareNoiseConfig,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Corrupt a clean ODMR spectrum with realistic COTS hardware artifacts.

    Includes smooth quadratic/cubic thermal laser baseline drift, time-domain
    AC mains hum, 1/f flicker noise, Poisson shot-noise modulation, and
    optional heavy-tailed Student-t fluctuations.

    Args:
        clean_spectrum: 1D clean ODMR spectrum normalized near 1.0.
        cfg: ``HardwareNoiseConfig`` specifying noise and drift levels.
        rng: Optional NumPy random generator.

    Returns:
        Noisy 1D ODMR spectrum with the same shape as ``clean_spectrum``.
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

    # Stochastic white + 1/f noise scaled by dip depth / target SNR and shot factor
    signal_depth = max(float(np.max(clean) - np.min(clean)), 1e-6)
    noise_std = signal_depth / float(cfg.target_snr)
    shot_factor = np.sqrt(np.clip(clean, 0.5, 1.5))

    if cfg.use_heavy_tails:
        white = rng.standard_t(df=_STUDENT_T_DF, size=n_points) / _STUDENT_T_SCALE
    else:
        white = rng.standard_normal(n_points)

    pink_ratio = float(np.clip(cfg.pink_noise_ratio, 0.0, 1.0))
    if pink_ratio > 0.0:
        pink = generate_pink_noise(n_points, rng)
        combined = np.sqrt(1.0 - pink_ratio) * white + np.sqrt(pink_ratio) * pink
    else:
        combined = white

    stochastic_noise = noise_std * shot_factor * combined
    return clean + drift + hum + stochastic_noise


def _edge_baseline_pad(
    edge_values: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pad out-of-bounds frequency regions from robust edge baseline statistics."""
    edge_median = float(np.median(edge_values))
    mad = float(np.median(np.abs(edge_values - edge_median)))
    edge_std = 1.4826 * mad
    if edge_std <= 0.0 and len(edge_values) > 1:
        edge_std = float(np.std(edge_values))
    if edge_std > 0.0:
        return edge_median + rng.normal(0.0, edge_std, size=count)
    return np.full(count, edge_median, dtype=np.float64)


def resample_spectrum_pchip(
    freqs_in: np.ndarray,
    spectrum_in: np.ndarray,
    target_freqs: np.ndarray = CANONICAL_FREQS,
    method: str = "pchip",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Safely resample an arbitrary ODMR frequency grid onto ``target_freqs``.

    Validates central NV resonance coverage, deduplicates and sorts input
    frequencies, applies Savitzky-Golay anti-aliasing prior to downsampling,
    interpolates without polynomial extrapolation, and pads any out-of-bounds
    edges using outermost 5% baseline statistics.

    Args:
        freqs_in: Input microwave frequencies in MHz.
        spectrum_in: Input ODMR intensity values matching ``freqs_in``.
        target_freqs: Target canonical frequency grid in MHz.
        method: Interpolation method, ``"pchip"`` or ``"linear"``.
        rng: Optional NumPy random generator for edge padding noise.

    Returns:
        Tuple ``(resampled_spectrum, meta_dict)`` where ``meta_dict`` contains
        ``window_padded``, ``original_min_mhz``, ``original_max_mhz``, and
        ``original_points``.

    Raises:
        ValueError: If input arrays mismatch in length, ``method`` is unknown,
            or ``freqs_in`` fails to cover ``[2830.0, 2910.0]`` MHz.
    """
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

    if rng is None:
        rng = np.random.default_rng()

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

    # Anti-aliasing filter prior to downsampling
    n_in = freqs_sorted.shape[0]
    n_target = target_arr.shape[0]
    if n_in > n_target and n_in >= 5:
        raw_win = max(5, (n_in // max(n_target, 1)) * 2 + 1)
        win_len = min(n_in - 1, raw_win)
        if win_len % 2 == 0:
            win_len -= 1
        if win_len >= 5:
            filtered_spec = savgol_filter(spec_sorted, window_length=win_len, polyorder=2)
        else:
            filtered_spec = spec_sorted
    else:
        filtered_spec = spec_sorted

    if method == "pchip":
        interpolator = PchipInterpolator(freqs_sorted, filtered_spec, extrapolate=False)
        resampled = interpolator(target_arr)
    else:
        resampled = np.interp(target_arr, freqs_sorted, filtered_spec, left=np.nan, right=np.nan)

    f_min = float(freqs_sorted[0])
    f_max = float(freqs_sorted[-1])
    left_mask = target_arr < f_min
    right_mask = target_arr > f_max
    window_padded = bool(np.any(left_mask) or np.any(right_mask))

    if window_padded:
        n_edge = min(n_in, max(3, int(np.ceil(0.05 * n_in))))
        if np.any(left_mask):
            resampled[left_mask] = _edge_baseline_pad(
                spec_sorted[:n_edge], int(np.sum(left_mask)), rng
            )
        if np.any(right_mask):
            resampled[right_mask] = _edge_baseline_pad(
                spec_sorted[-n_edge:], int(np.sum(right_mask)), rng
            )

    meta_dict = {
        "window_padded": window_padded,
        "original_min_mhz": orig_min,
        "original_max_mhz": orig_max,
        "original_points": orig_points,
    }
    return resampled, meta_dict

