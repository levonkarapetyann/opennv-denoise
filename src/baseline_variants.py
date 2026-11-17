"""Additional Stage 1.5 baseline variants calling frozen src/baseline.py functions only.

Implements the reporting-honesty variants without modifying frozen src/baseline.py:
  1. ``fit_lorentzian_naive_fixed`` (Task 3c): Fixes the outer-wing threshold and
     ``hs_ub`` box-bound truncation bug in the frozen 4-symmetric-pair naive Lorentzian
     fitter while remaining strictly uncoupled (no hyperfine, no spin-1 Hamiltonian).
  2. ``fit_informed_unknown_freq`` (Task 5a): Scans mains frequency over ``[45.0, 65.0]``
     Hz in steps of ``<= 0.25`` Hz and keeps the lowest residual.
  3. ``fit_informed_harmonics`` (Task 5b): Projects out fundamental + 2nd + 3rd mains
     harmonics (either at 50/60 Hz or scanned across 45-65 Hz).
  4. ``fit_physical_with_prior`` (Task 6a/b): Post-hoc re-ranking of K=5 Stage-B refined
     candidates from frozen ``_stage_a_coarse_screen`` and ``_run_varproj_pipeline`` using
     ``cost + lambda * (-log prior(|B|, HWHM, contrast))`` with ``lambda = 2.5e-4``
     fixed strictly on DEV_SEED = 10001.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from src.baseline import (
    _DEFAULT_ETA,
    _VarProjDesign,
    _check_window_truncation,
    _fast_single_4axis_dip_profile,
    _run_varproj_pipeline,
    _stage_a_coarse_screen,
)
from src.config import (
    B_MAG_MAX_MT,
    B_MAG_MIN_MT,
    D0_MHZ,
    TRAIN_LINEWIDTH_RANGE,
)
from src.physics import b_mag_from_splittings

# Prior penalty weight fixed strictly on DEV_SEED = 10001 (Task 6a)
DEFAULT_PRIOR_LAMBDA: float = 2.5e-4


class HarmonicVarProjDesign(_VarProjDesign):
    """Variable-Projection linear baseline & mains solver supporting harmonics (1f, 2f, 3f)."""

    def __init__(
        self,
        freqs: np.ndarray,
        delta_y: np.ndarray,
        valid_mask: np.ndarray,
        poly_degree: int = 2,
        mains_freq_hz: Optional[float] = None,
        dwell_time_sec: Optional[float] = None,
        include_harmonics: bool = False,
    ) -> None:
        if not include_harmonics or mains_freq_hz is None or dwell_time_sec is None:
            super().__init__(
                freqs,
                delta_y,
                valid_mask,
                poly_degree=poly_degree,
                mains_freq_hz=mains_freq_hz,
                dwell_time_sec=dwell_time_sec,
                oracle_delta_bl=None,
            )
            return

        self.freqs = freqs
        self.valid_mask = valid_mask
        self.oracle_delta_bl = None
        n_pts = freqs.shape[0]
        w_vec = valid_mask.astype(np.float64)
        self.w_vec = w_vec

        u = (freqs - D0_MHZ) / 120.0
        cols = [np.ones(n_pts, dtype=np.float64), u]
        reg_diag = [0.25, 0.8]
        if poly_degree >= 2:
            cols.append(u * u - (1.0 / 3.0))
            reg_diag.append(18.0)
        if poly_degree >= 3:
            cols.append(u * u * u - 0.6 * u)
            reg_diag.append(36.0)

        t_sec = np.arange(n_pts, dtype=np.float64) * float(dwell_time_sec)
        for h_mult in (1.0, 2.0, 3.0):
            omega_t = 2.0 * np.pi * (h_mult * float(mains_freq_hz)) * t_sec
            cols.append(np.sin(omega_t))
            cols.append(np.cos(omega_t))
            reg_diag.extend([8.0 * h_mult, 8.0 * h_mult])

        phi = np.column_stack(cols)
        wing_mask = ((freqs <= 2775.0) | (freqs >= 2965.0)) & valid_mask
        phi_wing = phi[wing_mask].copy()
        phi_wing[:, -6:] = 0.0

        wing_weight = 0.20
        reg_mat = np.diag(np.asarray(reg_diag, dtype=np.float64))
        penalty_mat = wing_weight * (phi_wing.T @ phi_wing) + reg_mat

        phi_w = phi * w_vec[:, None]
        gram = phi_w.T @ phi + penalty_mat
        self.phi = phi
        self.k_mat = np.linalg.solve(gram, phi_w.T)
        self.beta0 = self.k_mat @ delta_y
        self.r0 = (delta_y - (phi @ self.beta0)) * w_vec
        eig_w, eig_v = np.linalg.eigh(penalty_mat)
        self.chol_reg = (eig_v * np.sqrt(np.maximum(eig_w, 0.0))) @ eig_v.T


def fit_informed_unknown_freq(
    freqs: np.ndarray,
    noisy_spectrum: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    warm_start_b_vec: Optional[np.ndarray] = None,
    dwell_time: float = 0.8e-3,
    *,
    poly_degree: int = 2,
    freq_step_hz: float = 0.25,
    freq_bounds_hz: Tuple[float, float] = (45.0, 65.0),
    include_harmonics: bool = False,
    hwhm_bounds: Tuple[float, float] = TRAIN_LINEWIDTH_RANGE,
    eta: float = _DEFAULT_ETA,
    precomputed_basin: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fit physical ODMR when mains frequency is unknown by scanning [45, 65] Hz (step <= 0.25 Hz)."""
    t_start = time.perf_counter()
    freqs_arr = np.asarray(freqs, dtype=np.float64)
    spec_arr = np.asarray(noisy_spectrum, dtype=np.float64)
    n_pts = freqs_arr.shape[0]
    mask_arr = (
        np.isfinite(spec_arr)
        if valid_mask is None
        else (np.asarray(valid_mask, dtype=bool) & np.isfinite(spec_arr))
    )
    clean_spec_arr = np.where(mask_arr, spec_arr, 1.0)
    delta_y = clean_spec_arr - 1.0

    dt_val = float(dwell_time)
    dwell_sec = dt_val * 1e-3 if dt_val >= 0.05 else dt_val
    eta_val = float(np.clip(eta, 0.0, 1.0))

    if precomputed_basin is not None:
        out_basin = precomputed_basin
    else:
        vp_plain = _VarProjDesign(
            freqs_arr,
            delta_y,
            mask_arr,
            poly_degree=poly_degree,
            mains_freq_hz=None,
            dwell_time_sec=None,
            oracle_delta_bl=None,
        )
        out_basin = _run_varproj_pipeline(
            freqs_arr,
            clean_spec_arr,
            vp_plain,
            warm_start_b_vec=warm_start_b_vec,
            e_mhz=0.0,
            max_polish_starts=3,
            hwhm_bounds=hwhm_bounds,
            eta_val=eta_val,
        )
    dip_basin = _fast_single_4axis_dip_profile(
        freqs_arr, out_basin["b_vec"], out_basin["d0_mhz"], out_basin["hwhm"], eta_val, 0.0
    )

    step = min(float(freq_step_hz), 0.25)
    freq_grid = np.arange(float(freq_bounds_hz[0]), float(freq_bounds_hz[1]) + 1e-9, step)

    # Fast vectorized screening of the 81 candidate frequencies on the polynomial-detrended residual
    resid_poly = (clean_spec_arr - out_basin["fitted_curve"]) * mask_arr.astype(np.float64)
    t_sec = np.arange(n_pts, dtype=np.float64) * dwell_sec
    best_f_hz = 50.0
    max_proj_energy = -1.0
    for f_hz in freq_grid:
        omega_t = 2.0 * np.pi * float(f_hz) * t_sec
        s1 = np.sin(omega_t) * mask_arr
        c1 = np.cos(omega_t) * mask_arr
        proj_e = float(np.dot(resid_poly, s1) ** 2 + np.dot(resid_poly, c1) ** 2)
        if include_harmonics:
            s2 = np.sin(2.0 * omega_t) * mask_arr
            c2 = np.cos(2.0 * omega_t) * mask_arr
            s3 = np.sin(3.0 * omega_t) * mask_arr
            c3 = np.cos(3.0 * omega_t) * mask_arr
            proj_e += 0.5 * float(np.dot(resid_poly, s2) ** 2 + np.dot(resid_poly, c2) ** 2)
            proj_e += 0.33 * float(np.dot(resid_poly, s3) ** 2 + np.dot(resid_poly, c3) ** 2)
        if proj_e > max_proj_energy:
            max_proj_energy = proj_e
            best_f_hz = float(f_hz)

    best_vp = HarmonicVarProjDesign(
        freqs_arr,
        delta_y,
        mask_arr,
        poly_degree=poly_degree,
        mains_freq_hz=best_f_hz,
        dwell_time_sec=dwell_sec,
        include_harmonics=include_harmonics,
    )
    best_out = _run_varproj_pipeline(
        freqs_arr,
        clean_spec_arr,
        best_vp,
        warm_start_b_vec=out_basin["b_vec"],
        e_mhz=0.0,
        max_polish_starts=1,
        skip_stage_a=True,
        init_d0_mhz=float(out_basin["d0_mhz"]),
        init_hwhm_mhz=float(out_basin["hwhm"]),
        b_trust_radius_mt=0.035,
        hwhm_bounds=hwhm_bounds,
        eta_val=eta_val,
    )
    best_out["selected_mains_hz"] = best_f_hz
    best_out["nfev"] = int(out_basin["nfev"]) + int(best_out["nfev"])
    best_out["runtime_ms"] = float((time.perf_counter() - t_start) * 1000.0) + (
        float(out_basin.get("runtime_ms", 0.0)) if precomputed_basin is not None else 0.0
    )
    return best_out


def fit_informed_harmonics(
    freqs: np.ndarray,
    noisy_spectrum: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    warm_start_b_vec: Optional[np.ndarray] = None,
    dwell_time: float = 0.8e-3,
    *,
    poly_degree: int = 2,
    candidate_freqs_hz: Sequence[float] = (50.0, 60.0),
    hwhm_bounds: Tuple[float, float] = TRAIN_LINEWIDTH_RANGE,
    eta: float = _DEFAULT_ETA,
    precomputed_basin: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fit physical ODMR projecting out fundamental + 2nd + 3rd mains harmonics."""
    t_start = time.perf_counter()
    freqs_arr = np.asarray(freqs, dtype=np.float64)
    spec_arr = np.asarray(noisy_spectrum, dtype=np.float64)
    mask_arr = (
        np.isfinite(spec_arr)
        if valid_mask is None
        else (np.asarray(valid_mask, dtype=bool) & np.isfinite(spec_arr))
    )
    clean_spec_arr = np.where(mask_arr, spec_arr, 1.0)
    delta_y = clean_spec_arr - 1.0

    dt_val = float(dwell_time)
    dwell_sec = dt_val * 1e-3 if dt_val >= 0.05 else dt_val
    eta_val = float(np.clip(eta, 0.0, 1.0))

    if precomputed_basin is not None:
        out_basin = precomputed_basin
    else:
        vp_plain = _VarProjDesign(
            freqs_arr,
            delta_y,
            mask_arr,
            poly_degree=poly_degree,
            mains_freq_hz=None,
            dwell_time_sec=None,
            oracle_delta_bl=None,
        )
        out_basin = _run_varproj_pipeline(
            freqs_arr,
            clean_spec_arr,
            vp_plain,
            warm_start_b_vec=warm_start_b_vec,
            e_mhz=0.0,
            max_polish_starts=3,
            hwhm_bounds=hwhm_bounds,
            eta_val=eta_val,
        )
    dip_basin = _fast_single_4axis_dip_profile(
        freqs_arr, out_basin["b_vec"], out_basin["d0_mhz"], out_basin["hwhm"], eta_val, 0.0
    )

    best_vp = None
    best_r_sq = np.inf
    best_f_hz = 50.0
    for f_hz in candidate_freqs_hz:
        vp_cand = HarmonicVarProjDesign(
            freqs_arr,
            delta_y,
            mask_arr,
            poly_degree=poly_degree,
            mains_freq_hz=float(f_hz),
            dwell_time_sec=dwell_sec,
            include_harmonics=True,
        )
        _, _, r_cand, _ = vp_cand.solve_single(dip_basin)
        r_sq = float(np.dot(r_cand, r_cand))
        if r_sq < best_r_sq:
            best_r_sq = r_sq
            best_vp = vp_cand
            best_f_hz = float(f_hz)

    assert best_vp is not None
    best_out = _run_varproj_pipeline(
        freqs_arr,
        clean_spec_arr,
        best_vp,
        warm_start_b_vec=out_basin["b_vec"],
        e_mhz=0.0,
        max_polish_starts=1,
        skip_stage_a=True,
        init_d0_mhz=float(out_basin["d0_mhz"]),
        init_hwhm_mhz=float(out_basin["hwhm"]),
        b_trust_radius_mt=0.035,
        hwhm_bounds=hwhm_bounds,
        eta_val=eta_val,
    )
    best_out["selected_mains_hz"] = best_f_hz
    best_out["nfev"] = int(out_basin["nfev"]) + int(best_out["nfev"])
    best_out["runtime_ms"] = float((time.perf_counter() - t_start) * 1000.0) + (
        float(out_basin.get("runtime_ms", 0.0)) if precomputed_basin is not None else 0.0
    )
    return best_out


def compute_train_neg_log_prior(
    b_mag: float,
    hwhm: float,
    contrast: float,
    b_bounds: Tuple[float, float] = (B_MAG_MIN_MT, B_MAG_MAX_MT),
    hwhm_bounds: Tuple[float, float] = TRAIN_LINEWIDTH_RANGE,
    contrast_bounds: Tuple[float, float] = (0.02, 0.04),
) -> float:
    """Evaluate negative log-prior matching the TRAIN parameter sampling distribution.

    Penalizes candidates that slam HWHM or |B| against box boundaries to fit noise
    spikes or whose amplitude collapses below the training contrast support [0.02, 0.04]
    due to partial dip overlap in a wrong 3D basin.
    """
    h_lo, h_hi = float(hwhm_bounds[0]), float(hwhm_bounds[1])
    h_span = max(h_hi - h_lo, 1e-6)
    u_h = float(np.clip((hwhm - h_lo) / h_span, 1e-3, 1.0 - 1e-3))
    nlp_hwhm = -0.25 * (np.log(u_h) + np.log(1.0 - u_h)) - 0.25 * np.log(4.0)

    b_lo, b_hi = float(b_bounds[0]), float(b_bounds[1])
    b_span = max(b_hi - b_lo, 1e-6)
    u_b = float(np.clip((b_mag - b_lo) / b_span, 1e-3, 1.0 - 1e-3))
    nlp_b = -0.15 * (np.log(u_b) + np.log(1.0 - u_b)) - 0.15 * np.log(4.0)

    c_lo, c_hi = float(contrast_bounds[0]), float(contrast_bounds[1])
    c_deficit = max(0.0, c_lo - float(contrast)) / 0.010
    c_excess = max(0.0, float(contrast) - (c_hi + 0.002)) / 0.015
    nlp_c = 1.5 * (c_deficit * c_deficit + c_excess * c_excess)

    return float(nlp_hwhm + nlp_b + nlp_c)


def fit_physical_with_prior(
    freqs: np.ndarray,
    noisy_spectrum: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    warm_start_b_vec: Optional[np.ndarray] = None,
    *,
    n_starts: int = 5,
    poly_degree: int = 2,
    prior_lambda: float = DEFAULT_PRIOR_LAMBDA,
    hwhm_bounds: Tuple[float, float] = TRAIN_LINEWIDTH_RANGE,
    eta: float = _DEFAULT_ETA,
) -> Dict[str, Any]:
    """Fit physical ODMR and re-rank Stage-B refined candidates by cost + lambda * (-log prior).

    Calls frozen ``_stage_a_coarse_screen`` and ``_run_varproj_pipeline`` on each of the
    top ``n_starts`` coarse candidates and selects the candidate minimizing the MAP
    objective ``cost + prior_lambda * compute_train_neg_log_prior(|B|, HWHM, contrast)``.
    """
    t_start = time.perf_counter()
    freqs_arr = np.asarray(freqs, dtype=np.float64)
    spec_arr = np.asarray(noisy_spectrum, dtype=np.float64)
    mask_arr = (
        np.isfinite(spec_arr)
        if valid_mask is None
        else (np.asarray(valid_mask, dtype=bool) & np.isfinite(spec_arr))
    )
    clean_spec_arr = np.where(mask_arr, spec_arr, 1.0)
    delta_y = clean_spec_arr - 1.0
    eta_val = float(np.clip(eta, 0.0, 1.0))

    vp = _VarProjDesign(
        freqs_arr,
        delta_y,
        mask_arr,
        poly_degree=poly_degree,
        mains_freq_hz=None,
        dwell_time_sec=None,
        oracle_delta_bl=None,
    )
    candidates, fc_est = _stage_a_coarse_screen(
        freqs_arr,
        vp,
        warm_start_b_vec=warm_start_b_vec,
        e_mhz=0.0,
        top_k=max(1, int(n_starts)),
        eta_val=eta_val,
    )

    best_out: Optional[Dict[str, Any]] = None
    best_penalized_cost = np.inf
    total_nfev = 0
    candidate_records: List[Dict[str, Any]] = []

    for b_cand in candidates:
        out_c = _run_varproj_pipeline(
            freqs_arr,
            clean_spec_arr,
            vp,
            warm_start_b_vec=b_cand,
            e_mhz=0.0,
            max_polish_starts=1,
            skip_stage_a=True,
            init_d0_mhz=fc_est,
            hwhm_bounds=hwhm_bounds,
            eta_val=eta_val,
        )
        total_nfev += int(out_c["nfev"])
        nlp = compute_train_neg_log_prior(
            float(out_c["b_mag"]),
            float(out_c["hwhm"]),
            float(out_c["contrast"]),
            hwhm_bounds=hwhm_bounds,
        )
        pen_cost = float(out_c["cost"]) + float(prior_lambda) * nlp
        out_c["neg_log_prior"] = nlp
        out_c["penalized_cost"] = pen_cost
        candidate_records.append(out_c)
        if pen_cost < best_penalized_cost:
            best_penalized_cost = pen_cost
            best_out = out_c

    assert best_out is not None
    best_out["nfev"] = total_nfev
    best_out["runtime_ms"] = float((time.perf_counter() - t_start) * 1000.0)
    best_out["all_candidates"] = candidate_records
    return best_out


def fit_lorentzian_naive_fixed(
    freqs: np.ndarray,
    noisy_spectrum: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    *,
    hwhm_bounds: Tuple[float, float] = (2.0, 18.0),
    warm_start_hs_mhz: Optional[np.ndarray] = None,
    warm_start_d0_mhz: Optional[float] = None,
) -> Dict[str, Any]:
    """Bug-fixed 4-symmetric-pair naive Lorentzian fitter (still model-mismatched: no hyperfine/Hamiltonian).

    Fixes the two initialization/bound bugs identified in Task 3a/b of the frozen
    ``fit_lorentzian_naive``:
      1. Computes the wing noise threshold strictly from the dip-free outer wing
         (``df_fold >= 80 MHz``) and lowers the peak prominence threshold to
         ``0.025 * max_f`` so resolved outer single dips are not rejected when a
         merged central dip is present.
      2. Keeps the half-splitting upper bound at ``90.0 MHz`` (instead of clamping
         ``hs_ub`` to ``max_active_df + 6 MHz``, which trapped outer pairs at ~15 MHz)
         and screens a small multi-start set of mirror-folded peak initializations.
    """
    t_start = time.perf_counter()
    freqs_arr = np.asarray(freqs, dtype=np.float64)
    spec_arr = np.asarray(noisy_spectrum, dtype=np.float64)
    n_pts = freqs_arr.shape[0]
    mask_arr = (
        np.isfinite(spec_arr)
        if valid_mask is None
        else (np.asarray(valid_mask, dtype=bool) & np.isfinite(spec_arr))
    )
    clean_y = np.where(mask_arr, spec_arr, 1.0)
    w_vec = mask_arr.astype(np.float64)

    u = (freqs_arr - D0_MHZ) / 120.0
    delta_y = clean_y - 1.0

    wing_mask = ((freqs_arr <= 2775.0) | (freqs_arr >= 2965.0)) & mask_arr
    phi_bl = np.column_stack([np.ones(n_pts, dtype=np.float64), u])
    phi_w = phi_bl * w_vec[:, None]
    if np.any(wing_mask):
        gram_bl = (
            phi_w.T @ phi_bl
            + 0.50 * (phi_bl[wing_mask].T @ phi_bl[wing_mask])
            + np.diag([0.25, 0.8])
        )
        rhs_bl = phi_w.T @ delta_y + 0.50 * (phi_bl[wing_mask].T @ delta_y[wing_mask])
    else:
        gram_bl = phi_w.T @ phi_bl + np.diag([0.25, 0.8])
        rhs_bl = phi_w.T @ delta_y
    beta_init = np.linalg.solve(gram_bl, rhs_bl)
    detrended_abs = -(delta_y - phi_bl @ beta_init) * w_vec

    df_grid = float(freqs_arr[1] - freqs_arr[0]) if n_pts > 1 else 0.47
    sigma_pts = max(1.5, 3.2 / df_grid)
    smoothed = gaussian_filter1d(detrended_abs, sigma=sigma_pts, mode="nearest")

    if warm_start_d0_mhz is not None:
        best_fc = float(warm_start_d0_mhz)
    else:
        fc_grid = np.linspace(2865.0, 2875.0, 41)
        delta_f = np.linspace(2.0, 95.0, 120)
        best_fc = D0_MHZ
        best_corr = -np.inf
        for fc in fc_grid:
            right = np.interp(fc + delta_f, freqs_arr, smoothed)
            left = np.interp(fc - delta_f, freqs_arr, smoothed)
            r_zm = right - np.mean(right)
            l_zm = left - np.mean(left)
            denom = float(np.linalg.norm(r_zm) * np.linalg.norm(l_zm))
            if denom > 1e-12:
                corr = float(np.dot(r_zm, l_zm)) / denom
                if corr > best_corr:
                    best_corr = corr
                    best_fc = float(fc)

    df_fold = np.linspace(0.0, 95.0, 191)
    folded = 0.5 * (
        np.interp(best_fc + df_fold, freqs_arr, smoothed)
        + np.interp(best_fc - df_fold, freqs_arr, smoothed)
    )
    sigma_sharp = max(1.0, 1.8 / df_grid)
    smoothed_sharp = gaussian_filter1d(detrended_abs, sigma=sigma_sharp, mode="nearest")
    folded_sharp = 0.5 * (
        np.interp(best_fc + df_fold, freqs_arr, smoothed_sharp)
        + np.interp(best_fc - df_fold, freqs_arr, smoothed_sharp)
    )

    wing_slice = folded_sharp[df_fold >= 80.0]
    wing_mad = (
        1.4826 * float(np.median(np.abs(wing_slice - np.median(wing_slice))))
        if len(wing_slice) > 4
        else 1e-4
    )
    wing_max = float(np.max(np.abs(wing_slice))) if len(wing_slice) > 0 else 1e-4
    max_f = max(float(np.max(folded)), 1e-5)
    wing_thresh = max(1.30 * wing_max, 3.0 * wing_mad, 0.14 * max_f)

    pks_f, _ = find_peaks(folded, distance=10, height=wing_thresh, prominence=0.025 * max_f)
    pks_fs, _ = find_peaks(folded_sharp, distance=8, height=wing_thresh, prominence=0.035 * max_f)

    all_pks = sorted(
        list(set(list(pks_f) + list(pks_fs))),
        key=lambda idx: float(folded[idx]),
        reverse=True,
    )
    detected_nonzero: List[float] = []
    for idx in all_pks:
        val = float(df_fold[idx])
        if 3.0 <= val <= 90.0 and not any(abs(val - prev) < 4.5 for prev in detected_nonzero):
            detected_nonzero.append(val)
        if len(detected_nonzero) >= 5:
            break

    active_mask = folded > wing_thresh
    max_active_df = float(np.max(df_fold[active_mask])) if np.any(active_mask) else 40.0
    max_active_df = float(np.clip(max_active_df, 8.0, 88.0))

    def complete_4_hs(pks: List[float], span_df: float) -> np.ndarray:
        out = list(pks)[:4]
        grid_fill = np.linspace(0.20 * span_df, 0.95 * span_df, 6)
        for g_val in grid_fill:
            if len(out) >= 4:
                break
            if not any(abs(float(g_val) - prev) < 4.0 for prev in out):
                out.append(float(g_val))
        while len(out) < 4:
            out.append(float(0.5 * span_df))
        return np.sort(np.asarray(out[:4], dtype=np.float64))

    cand_inits: List[np.ndarray] = []
    if warm_start_hs_mhz is not None:
        cand_inits.append(np.sort(np.asarray(warm_start_hs_mhz, dtype=np.float64)[:4]))
    else:
        if float(np.max(folded[:5])) >= 0.35 * max_f:
            cand_inits.append(complete_4_hs([0.3] + detected_nonzero[:3], max_active_df))
        cand_inits.append(complete_4_hs(detected_nonzero[:4], max_active_df))
        cand_inits.append(np.array([0.12, 0.38, 0.66, 0.94], dtype=np.float64) * max_active_df)
        cand_inits.append(np.array([0.02, 0.35, 0.65, 0.96], dtype=np.float64) * max_active_df)

    w_min, w_max = float(hwhm_bounds[0]), float(hwhm_bounds[1])
    w0 = float(np.clip(4.8, w_min + 0.2, w_max - 0.2))
    amp_est = float(np.clip(0.55 * max_f, 0.002, 0.035))
    b0_init = float(1.0 + beta_init[0])
    b1_init = float(beta_init[1])

    def eval_sym4(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d0_p = p[0]
        hs_p = p[1:5]
        w_p = p[5]
        amps_p = p[6:10]
        bl = p[10] + p[11] * u

        w_sq = w_p * w_p
        diff_m = freqs_arr[:, None] - (d0_p - hs_p)[None, :]
        diff_p = freqs_arr[:, None] - (d0_p + hs_p)[None, :]
        pair_lor = (w_sq / (diff_m * diff_m + w_sq)) + (w_sq / (diff_p * diff_p + w_sq))
        dips = pair_lor @ amps_p
        return bl - dips, bl

    def resid_fn(p: np.ndarray) -> np.ndarray:
        curve, bl = eval_sym4(p)
        data_r = (curve - clean_y)[mask_arr]
        wing_r = 0.50 * (bl[wing_mask] - clean_y[wing_mask]) if np.any(wing_mask) else np.zeros(0)
        amp_r = 0.35 * (p[6:10] - np.mean(p[6:10]))
        return np.concatenate([data_r, wing_r, amp_r])

    lb = np.concatenate([[2865.0], np.zeros(4), [w_min], np.full(4, 0.10 * amp_est), [0.85, -0.08]])
    ub = np.concatenate([[2875.0], np.full(4, 90.0), [w_max], np.full(4, 0.06), [1.15, 0.08]])

    mad_sigma = max(
        1.4826 * float(np.median(np.abs(detrended_abs - np.median(detrended_abs)))), 1e-4
    )

    scored_starts: List[Tuple[float, np.ndarray]] = []
    for hs_c in cand_inits:
        x_c = np.clip(
            np.concatenate([[best_fc], hs_c, [w0], np.full(4, amp_est), [b0_init, b1_init]]),
            lb + 1e-6,
            ub - 1e-6,
        )
        r_c = resid_fn(x_c)
        scored_starts.append((float(np.dot(r_c, r_c)), x_c))
    scored_starts.sort(key=lambda item: item[0])

    best_res = None
    best_l2 = np.inf
    total_nfev = 0
    n_to_refine = 1 if warm_start_hs_mhz is not None else min(2, len(scored_starts))

    for _, x0_start in scored_starts[:n_to_refine]:
        res = least_squares(
            resid_fn,
            x0_start,
            bounds=(lb, ub),
            method="trf",
            loss="soft_l1",
            f_scale=2.0 * mad_sigma,
            x_scale="jac",
            max_nfev=60,
            ftol=1e-6,
            xtol=1e-6,
        )
        total_nfev += int(res.nfev)
        r_final = resid_fn(res.x)
        l2_val = 0.5 * float(np.dot(r_final, r_final))
        if l2_val < best_l2:
            best_l2 = l2_val
            best_res = res

    assert best_res is not None
    fitted_curve, fitted_bl = eval_sym4(best_res.x)
    d0_fit = float(best_res.x[0])
    hs_fit = np.sort(np.abs(best_res.x[1:5]))
    splittings = 2.0 * hs_fit
    b_mag = float(b_mag_from_splittings(splittings, order=1))
    hwhm_fit = float(best_res.x[5])
    resid_valid = (fitted_curve - clean_y)[mask_arr]
    resid_rms = float(np.sqrt(np.mean(resid_valid * resid_valid)))
    runtime_ms = float((time.perf_counter() - t_start) * 1000.0)

    outer_f_min = d0_fit - float(hs_fit[-1])
    outer_f_max = d0_fit + float(hs_fit[-1])
    window_trunc = _check_window_truncation(
        freqs_arr, mask_arr, outer_f_min, outer_f_max, hwhm_fit, detrended_abs, include_hyperfine=False
    )
    in_range = (B_MAG_MIN_MT <= b_mag <= B_MAG_MAX_MT) and not window_trunc
    status = "converged" if best_res.success else ("max_nfev" if best_res.status == 0 else "failed")
    if not in_range:
        status = "out_of_range"

    return {
        "b_mag": b_mag,
        "b_vec": np.array([0.0, 0.0, b_mag], dtype=np.float64),
        "b_field_mt": b_mag,
        "b_vec_mt": np.array([0.0, 0.0, b_mag], dtype=np.float64),
        "splittings_mhz": splittings,
        "d0_mhz": d0_fit,
        "hwhm": hwhm_fit,
        "amplitudes": best_res.x[6:10].copy(),
        "fitted_curve": fitted_curve,
        "fitted_baseline": fitted_bl,
        "residual_rms": resid_rms,
        "cost": float(best_l2),
        "nfev": int(total_nfev),
        "runtime_ms": runtime_ms,
        "out_of_range": bool(not in_range),
        "success": bool((best_res.success or best_res.status == 0) and in_range),
        "status": status,
    }
