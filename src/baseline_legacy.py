"""Strong physical baseline: Multi-start spin-1 Hamiltonian least-squares fitter."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, savgol_filter

from src.config import D0_MHZ, GAMMA_E_MHZ_MT, NV_AXES
from src.physics import evaluate_axes_odmr

_DEFAULT_ETA: float = 0.35

# Diverse symmetry-broken directions inside the canonical octant wedge 0 < Bx < By < Bz
_WEDGE_DIRECTIONS: np.ndarray = np.array(
    [
        [0.08, 0.16, 0.98],
        [0.08, 0.66, 0.75],
        [0.48, 0.56, 0.68],
        [0.18, 0.44, 0.88],
        [0.28, 0.52, 0.81],
        [0.12, 0.58, 0.80],
        [0.36, 0.54, 0.76],
        [0.22, 0.34, 0.91],
    ],
    dtype=np.float64,
)
_WEDGE_DIRECTIONS = _WEDGE_DIRECTIONS / np.linalg.norm(
    _WEDGE_DIRECTIONS, axis=-1, keepdims=True
)


def _detrend_smooth_baseline(freqs: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    """Estimate smooth thermal laser drift without fitting downward ODMR dips.

    Uses regularized, wing-anchored, iteratively reweighted degree-3 polynomial
    regression in normalized frequency coordinates ``u = (freqs - D0) / 120``.
    Returns the zero-mean drift shift ``bl_shift`` to subtract before fitting.
    """
    n_pts = freqs.shape[0]
    if n_pts < 12:
        return np.zeros_like(spectrum)

    u = (freqs - D0_MHZ) / 120.0
    vander = np.vander(u, N=4, increasing=True)
    # Second-order difference MAD isolates stochastic noise from smooth drift and slopes
    d2 = np.diff(spectrum, n=2)
    raw_sigma = 1.4826 * float(np.median(np.abs(d2))) / np.sqrt(6.0)
    sigma = max(raw_sigma, 1e-5)
    reg = np.diag([0.0, 0.0, 4.0, 8.0])

    n_wing = max(3, int(0.10 * n_pts))
    wing_mask = np.zeros(n_pts, dtype=bool)
    wing_mask[:n_wing] = True
    wing_mask[-n_wing:] = True

    dip_w = 1e-4 if sigma < 5e-4 else 0.02
    weights = np.ones(n_pts, dtype=np.float64)
    baseline = np.full(n_pts, float(np.median(spectrum)), dtype=np.float64)
    for _ in range(8):
        w_eff = np.where(wing_mask, 2.0, weights)
        vw = vander * w_eff[:, None]
        coef = np.linalg.solve(vw.T @ vander + reg, vw.T @ spectrum)
        baseline = vander @ coef
        resid = spectrum - baseline
        weights = np.where(
            resid < -1.8 * sigma,
            dip_w,
            np.where(resid > 2.5 * sigma, 0.2, 1.0),
        )

    bl_shift = baseline - float(np.mean(baseline))
    if raw_sigma < 1e-4 and float(np.max(np.abs(bl_shift))) < 3e-4:
        return np.zeros_like(spectrum)
    return bl_shift


def _estimate_initial_observables(
    freqs: np.ndarray,
    noisy_spectrum: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """Estimate linear baseline, contrast, D0, and initial |B| from peak prominence."""
    n_pts = freqs.shape[0]
    u = (freqs - D0_MHZ) / 120.0
    n_edge = max(4, int(0.08 * n_pts))

    left_u = float(np.median(u[:n_edge]))
    left_y = float(np.median(noisy_spectrum[:n_edge]))
    right_u = float(np.median(u[-n_edge:]))
    right_y = float(np.median(noisy_spectrum[-n_edge:]))

    slope0 = float(np.clip((right_y - left_y) / max(right_u - left_u, 1e-6), -0.15, 0.15))
    offset0 = float(
        np.clip(0.5 * ((left_y - slope0 * left_u) + (right_y - slope0 * right_u)), 0.7, 1.3)
    )
    baseline0 = offset0 + slope0 * u

    # Smooth spectrum to suppress shot noise and 50 Hz ripple before peak detection
    win_len = min(n_pts - 1 if n_pts % 2 == 0 else n_pts, 15)
    if win_len % 2 == 0:
        win_len -= 1
    if win_len >= 5:
        smoothed = savgol_filter(noisy_spectrum, window_length=win_len, polyorder=2)
    else:
        smoothed = noisy_spectrum

    absorption = baseline0 - smoothed
    # Robust noise scale estimate from first differences of raw spectrum
    raw_diff_mad = float(np.median(np.abs(np.diff(noisy_spectrum))))
    noise_sigma = max(1.4826 * raw_diff_mad / np.sqrt(2.0), 1e-5)

    min_dist = max(3, int(0.02 * n_pts))
    peaks, _ = find_peaks(
        absorption,
        prominence=max(0.001, 1.2 * noise_sigma),
        distance=min_dist,
    )
    positive_peaks = [p for p in peaks if absorption[p] > 0.0]

    max_abs = float(np.max(absorption)) if np.max(absorption) > 0 else 0.015
    contrast0 = float(np.clip(max_abs / 0.35, 0.005, 0.15))

    if len(positive_peaks) >= 2:
        f_low = float(freqs[positive_peaks[0]])
        f_high = float(freqs[positive_peaks[-1]])
        d0_est = float(np.clip(0.5 * (f_low + f_high), 2858.0, 2882.0))
        max_dev = max(abs(f_high - d0_est), abs(d0_est - f_low))
        b_est = float(np.clip((max_dev / GAMMA_E_MHZ_MT) / 0.85, 0.1, 3.2))
    elif len(positive_peaks) == 1:
        f_pk = float(freqs[positive_peaks[0]])
        d0_est = D0_MHZ
        b_est = float(np.clip((abs(f_pk - D0_MHZ) / GAMMA_E_MHZ_MT) / 0.85, 0.1, 3.2))
    else:
        d0_est = D0_MHZ
        b_est = 0.6

    return offset0, slope0, contrast0, d0_est, b_est


def _build_4axis_starts(
    n_starts: int,
    b_est: float,
    warm_start_b_vec: Optional[np.ndarray],
    rng: Optional[np.random.Generator],
) -> List[np.ndarray]:
    """Construct diverse 3D magnetic-field starting guesses for multi-start TRF."""
    starts: List[np.ndarray] = []

    if warm_start_b_vec is not None:
        starts.append(np.asarray(warm_start_b_vec, dtype=np.float64).copy())
    else:
        dir_0 = np.array([0.6, 0.3, 0.1], dtype=np.float64)
        dir_0 = dir_0 / np.linalg.norm(dir_0)
        starts.append(dir_0 * b_est)

    n_extra = max(0, n_starts - 1)
    if n_extra > 0:
        norms = np.linspace(0.25, 2.75, n_extra)
        if rng is not None:
            # Shuffle norm-direction pairing slightly while preserving [0.2, 2.8] mT coverage
            perm = rng.permutation(n_extra)
            norms = norms[perm]
        for idx in range(n_extra):
            base_dir = _WEDGE_DIRECTIONS[idx % len(_WEDGE_DIRECTIONS)]
            if rng is not None:
                # Small jitter sorted inside canonical wedge 0 < Bx < By < Bz
                jitter = np.sort(np.clip(base_dir + rng.normal(0.0, 0.04, size=3), 0.04, 1.0))
                direction = jitter / np.linalg.norm(jitter)
            else:
                direction = base_dir
            starts.append(direction * float(norms[idx]))

    return starts


def _fit_4axis_odmr(
    freqs: np.ndarray,
    noisy_spectrum: np.ndarray,
    u: np.ndarray,
    n_starts: int,
    warm_start_b_vec: Optional[np.ndarray],
    rng: Optional[np.random.Generator],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Run multi-start TRF optimization for the 4-axis NV ensemble."""
    bl_shift = _detrend_smooth_baseline(freqs, noisy_spectrum)
    target_spectrum = noisy_spectrum - bl_shift

    offset0, slope0, contrast0, d0_est, b_est = _estimate_initial_observables(
        freqs, target_spectrum
    )
    b_starts = _build_4axis_starts(n_starts, b_est, warm_start_b_vec, rng)

    lower_full = np.array([-3.35, -3.35, -3.35, 2855.0, 2.0, 0.002, 0.5, -0.2], dtype=np.float64)
    upper_full = np.array([3.35, 3.35, 3.35, 2885.0, 14.0, 0.25, 1.5, 0.2], dtype=np.float64)

    # Coarse basin-capture bounds (moderately broadened linewidth merges hyperfine triplets)
    lower_coarse = np.array([-3.35, -3.35, -3.35, 0.002, 0.5, -0.2], dtype=np.float64)
    upper_coarse = np.array([3.35, 3.35, 3.35, 0.25, 1.5, 0.2], dtype=np.float64)
    coarse_gamma = np.array([7.5], dtype=np.float64)
    coarse_d0 = np.array([d0_est], dtype=np.float64)
    eta_one = np.array([_DEFAULT_ETA], dtype=np.float64)

    def _eval_4axis(
        b_vec: np.ndarray,
        d0_val: float,
        gamma_val: float,
        contrast_val: float,
        offset_val: float,
        slope_val: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        b_par = np.abs(NV_AXES @ b_vec)[None, :]
        b_sq = float(b_vec[0] * b_vec[0] + b_vec[1] * b_vec[1] + b_vec[2] * b_vec[2])
        b_perp = np.sqrt(np.maximum(0.0, b_sq - b_par * b_par))
        spec, f_minus, f_plus = evaluate_axes_odmr(
            freqs=freqs,
            b_par=b_par,
            b_perp=b_perp,
            d0_arr=np.array([d0_val], dtype=np.float64),
            gamma_arr=np.array([gamma_val], dtype=np.float64),
            contrast_arr=np.array([contrast_val], dtype=np.float64),
            eta_arr=eta_one,
            include_hyperfine=True,
            single_axis=False,
        )
        baseline_shift = (offset_val - 1.0) + slope_val * u
        return spec[0] + baseline_shift, f_minus[0], f_plus[0]

    def _radial_penalty(b_vec: np.ndarray) -> float:
        b_norm = float(np.sqrt(b_vec[0] * b_vec[0] + b_vec[1] * b_vec[1] + b_vec[2] * b_vec[2]))
        return 0.5 * max(0.0, b_norm - 3.35)

    def residual_coarse(theta_c: np.ndarray) -> np.ndarray:
        model, _, _ = _eval_4axis(
            theta_c[:3], coarse_d0[0], coarse_gamma[0], theta_c[3], theta_c[4], theta_c[5]
        )
        return np.append(model - target_spectrum, _radial_penalty(theta_c[:3]))

    def residual_full(theta: np.ndarray) -> np.ndarray:
        model, _, _ = _eval_4axis(
            theta[:3], theta[3], theta[4], theta[5], theta[6], theta[7]
        )
        return np.append(model - target_spectrum, _radial_penalty(theta[:3]))

    # Start 0 goes straight to full polish; exploratory starts 1..K-1 undergo coarse screening
    b0_clip = np.clip(b_starts[0], -3.2, 3.2)
    polish_seeds: List[np.ndarray] = [
        np.array([b0_clip[0], b0_clip[1], b0_clip[2], d0_est, 4.5, contrast0, offset0, slope0], dtype=np.float64)
    ]

    coarse_candidates: List[Tuple[float, np.ndarray]] = []
    for b_init in b_starts[1:]:
        b_init_c = np.clip(b_init, -3.2, 3.2)
        x0_coarse = np.array(
            [b_init_c[0], b_init_c[1], b_init_c[2], contrast0, offset0, slope0],
            dtype=np.float64,
        )
        res_c = least_squares(
            residual_coarse,
            x0_coarse,
            bounds=(lower_coarse, upper_coarse),
            method="trf",
            x_scale="jac",
            max_nfev=25,
            ftol=1e-3,
            xtol=1e-3,
        )
        coarse_candidates.append((float(res_c.cost), res_c.x))

    coarse_candidates.sort(key=lambda item: item[0])
    for _, x_c in coarse_candidates[:3]:
        polish_seeds.append(
            np.array(
                [
                    x_c[0],
                    x_c[1],
                    x_c[2],
                    d0_est,
                    4.5,
                    np.clip(x_c[3], 0.005, 0.2),
                    np.clip(x_c[4], 0.6, 1.4),
                    np.clip(x_c[5], -0.18, 0.18),
                ],
                dtype=np.float64,
            )
        )

    best_cost = np.inf
    best_theta: Optional[np.ndarray] = None
    best_success = False

    for x0_full in polish_seeds:
        x0_clipped = np.clip(x0_full, lower_full + 1e-6, upper_full - 1e-6)
        res = least_squares(
            residual_full,
            x0_clipped,
            bounds=(lower_full, upper_full),
            method="trf",
            x_scale="jac",
            max_nfev=80,
            ftol=1e-6,
            xtol=1e-6,
        )
        if res.cost < best_cost:
            best_cost = float(res.cost)
            best_theta = res.x.copy()
            best_success = bool(res.success)

    assert best_theta is not None
    fitted_model, f_minus, f_plus = _eval_4axis(
        best_theta[:3],
        best_theta[3],
        best_theta[4],
        best_theta[5],
        best_theta[6],
        best_theta[7],
    )
    fitted_curve = fitted_model + bl_shift
    splittings = np.sort(f_plus - f_minus)
    return best_theta, fitted_curve, splittings, best_cost, best_success


def _fit_1axis_odmr(
    freqs: np.ndarray,
    noisy_spectrum: np.ndarray,
    u: np.ndarray,
    n_starts: int,
    warm_start_b_vec: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Run multi-start TRF optimization for single-axis [111] ODMR."""
    bl_shift = _detrend_smooth_baseline(freqs, noisy_spectrum)
    target_spectrum = noisy_spectrum - bl_shift

    offset0, slope0, contrast0, d0_est, b_est = _estimate_initial_observables(
        freqs, target_spectrum
    )
    b_par_starts: List[float] = []
    if warm_start_b_vec is not None:
        b_par_starts.append(
            float(np.clip(abs(np.dot(np.asarray(warm_start_b_vec, dtype=np.float64), NV_AXES[0])), 0.0, 3.4))
        )
    else:
        b_par_starts.append(float(np.clip(b_est * 0.85, 0.05, 3.4)))

    if n_starts > 1:
        b_par_starts.extend(np.linspace(0.15, 3.0, n_starts - 1).tolist())

    lower = np.array([0.0, 2855.0, 2.0, 0.002, 0.5, -0.2], dtype=np.float64)
    upper = np.array([3.5, 2885.0, 14.0, 0.25, 1.5, 0.2], dtype=np.float64)
    eta_one = np.array([_DEFAULT_ETA], dtype=np.float64)
    b_perp_zero = np.zeros((1, 1), dtype=np.float64)

    def _eval_1axis(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        b_par_arr = np.array([[theta[0]]], dtype=np.float64)
        spec, f_minus, f_plus = evaluate_axes_odmr(
            freqs=freqs,
            b_par=b_par_arr,
            b_perp=b_perp_zero,
            d0_arr=np.array([theta[1]], dtype=np.float64),
            gamma_arr=np.array([theta[2]], dtype=np.float64),
            contrast_arr=np.array([theta[3]], dtype=np.float64),
            eta_arr=eta_one,
            include_hyperfine=True,
            single_axis=True,
        )
        baseline_shift = (theta[4] - 1.0) + theta[5] * u
        return spec[0] + baseline_shift, f_minus[0], f_plus[0]

    def residual(theta: np.ndarray) -> np.ndarray:
        model, _, _ = _eval_1axis(theta)
        return model - target_spectrum

    best_cost = np.inf
    best_theta: Optional[np.ndarray] = None
    best_success = False

    for b_par0 in b_par_starts:
        x0 = np.clip(
            np.array([b_par0, d0_est, 4.5, contrast0, offset0, slope0], dtype=np.float64),
            lower + 1e-6,
            upper - 1e-6,
        )
        res = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            max_nfev=80,
            ftol=1e-5,
            xtol=1e-5,
        )
        if res.cost < best_cost:
            best_cost = float(res.cost)
            best_theta = res.x.copy()
            best_success = bool(res.success)

    assert best_theta is not None
    fitted_model, f_minus, f_plus = _eval_1axis(best_theta)
    fitted_curve = fitted_model + bl_shift
    splittings = np.sort(f_plus - f_minus)
    return best_theta, fitted_curve, splittings, best_cost, best_success


def fit_physical_odmr(
    freqs: np.ndarray,
    noisy_spectrum: np.ndarray,
    single_axis: bool = False,
    n_starts: int = 8,
    warm_start_b_vec: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Fit a physical spin-1 NV Hamiltonian ODMR model using multi-start TRF.

    Args:
        freqs: 1D microwave frequency grid in MHz.
        noisy_spectrum: 1D observed ODMR spectrum corresponding to ``freqs``.
        single_axis: If True, fit the 1-axis ``[B_par, D0, gamma, contrast,
            offset, slope]`` model; else fit the 4-axis 3D-vector ensemble.
        n_starts: Number of multi-start initializations.
        warm_start_b_vec: Optional 3D magnetic field vector (mT) used for Start 0.
        rng: Optional NumPy random generator for multi-start jittering.

    Returns:
        Dictionary containing ``b_field_mt``, ``b_vec_mt``,
        ``splittings_mhz``, ``d0_mhz``, ``fitted_curve``, ``runtime_ms``,
        ``cost``, and ``success``.
    """
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1.")

    freqs_arr = np.asarray(freqs, dtype=np.float64)
    spec_arr = np.asarray(noisy_spectrum, dtype=np.float64)
    u = (freqs_arr - D0_MHZ) / 120.0

    t_start = time.perf_counter()

    if single_axis:
        theta, fitted_curve, splittings, cost, success = _fit_1axis_odmr(
            freqs_arr, spec_arr, u, n_starts, warm_start_b_vec
        )
        b_par = float(theta[0])
        b_vec = b_par * NV_AXES[0]
        b_field_mt = abs(b_par)
        d0_mhz = float(theta[1])
    else:
        theta, fitted_curve, splittings, cost, success = _fit_4axis_odmr(
            freqs_arr, spec_arr, u, n_starts, warm_start_b_vec, rng
        )
        b_vec = theta[:3].astype(np.float64)
        b_field_mt = float(np.linalg.norm(b_vec))
        d0_mhz = float(theta[3])

    runtime_ms = (time.perf_counter() - t_start) * 1000.0
    return {
        "b_field_mt": b_field_mt,
        "b_vec_mt": b_vec,
        "splittings_mhz": splittings,
        "d0_mhz": d0_mhz,
        "fitted_curve": fitted_curve,
        "runtime_ms": float(runtime_ms),
        "cost": float(cost),
        "success": bool(success),
    }

