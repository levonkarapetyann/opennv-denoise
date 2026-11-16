"""Physical baseline ODMR fitters: Variable-Projection Spin-1 Hamiltonian & Naive 8-Lorentzian."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from src.config import (
    A_HYPERFINE_MHZ,
    B_MAG_MAX_MT,
    B_MAG_MIN_MT,
    D0_MHZ,
    GAMMA_E_MHZ_MT,
    NV_AXES,
    TRAIN_LINEWIDTH_RANGE,
)
from src.physics import (
    b_mag_from_splittings,
    canonicalize_b_vec,
    evaluate_axes_odmr,
    solve_axis_transitions,
)

_DEFAULT_ETA: float = 0.35
_SQRT_2: float = float(np.sqrt(2.0))
_LN_2: float = float(np.log(2.0))


def _build_canonical_wedge_directions() -> np.ndarray:
    """Build 15 deterministic unit directions spanning the canonical wedge 0 <= Bx <= By <= Bz."""
    raw = np.array(
        [
            # Corners and symmetry-plane boundaries (including B . n_k = 0 degenerate directions)
            [0.00, 0.00, 1.00],  # [001]
            [0.00, 0.7071, 0.7071],  # [011]
            [0.5774, 0.5774, 0.5774],  # [111]
            [0.00, 0.4472, 0.8944],  # [0, 1, 2]
            [0.2673, 0.5345, 0.8018],  # [1, 2, 3] (B . n_2 = 0 degenerate plane)
            [0.3482, 0.6528, 0.6728],  # near (0.40, 0.75, 1.15) degenerate plane
            [0.1690, 0.5071, 0.8452],  # [1, 3, 5] (B . n_2 = 0 degenerate plane)
            [0.4082, 0.4082, 0.8165],  # [1, 1, 2]
            # Interior directions covering the wedge uniformly
            [0.12, 0.25, 0.96],
            [0.12, 0.62, 0.77],
            [0.25, 0.38, 0.89],
            [0.25, 0.68, 0.69],
            [0.38, 0.52, 0.76],
            [0.46, 0.58, 0.67],
            [0.18, 0.72, 0.67],
        ],
        dtype=np.float64,
    )
    raw = np.sort(np.abs(raw), axis=-1)
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    return raw / np.maximum(norms, 1e-12)


_WEDGE_LATTICE_DIRS: np.ndarray = _build_canonical_wedge_directions()

# Sign patterns (s1, s2, s3, +1) for analytic 4-axis reconstruction
_SIGN_PATTERNS: np.ndarray = np.array(
    [
        [1.0, 1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0, 1.0],
        [-1.0, -1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0, 1.0],
    ],
    dtype=np.float64,
)


class _VarProjDesign:
    """Precomputed Variable-Projection linear baseline & amplitude solver.

    Solves jointly for non-negative dip amplitude ``c >= 0`` and baseline
    coefficients ``beta`` minimizing:
        ||delta_y + c * d - Phi @ beta||_W^2
        + w_wing * ||Phi_wing @ beta||^2
        + beta^T @ Reg @ beta
    """

    def __init__(
        self,
        freqs: np.ndarray,
        delta_y: np.ndarray,
        valid_mask: np.ndarray,
        poly_degree: int = 2,
        mains_freq_hz: Optional[float] = None,
        dwell_time_sec: Optional[float] = None,
        oracle_delta_bl: Optional[np.ndarray] = None,
    ) -> None:
        self.freqs = freqs
        self.valid_mask = valid_mask
        self.oracle_delta_bl = oracle_delta_bl
        n_pts = freqs.shape[0]
        w_vec = valid_mask.astype(np.float64)
        self.w_vec = w_vec

        if oracle_delta_bl is not None:
            self.phi = np.zeros((n_pts, 0), dtype=np.float64)
            self.k_mat = np.zeros((0, n_pts), dtype=np.float64)
            self.beta0 = np.zeros(0, dtype=np.float64)
            self.r0 = (delta_y - oracle_delta_bl) * w_vec
            self.chol_reg = np.zeros((0, 0), dtype=np.float64)
            return

        u = (freqs - D0_MHZ) / 120.0
        cols = [np.ones(n_pts, dtype=np.float64), u]
        reg_diag = [0.25, 0.8]

        if poly_degree >= 2:
            cols.append(u * u - (1.0 / 3.0))
            # Strong curvature penalty prevents degree-2 baseline from absorbing central dips
            reg_diag.append(18.0)

        if mains_freq_hz is not None and dwell_time_sec is not None and dwell_time_sec > 0.0:
            t_sec = np.arange(n_pts, dtype=np.float64) * float(dwell_time_sec)
            omega_t = 2.0 * np.pi * float(mains_freq_hz) * t_sec
            cols.append(np.sin(omega_t))
            cols.append(np.cos(omega_t))
            # Gram diagonal of sin/cos over 512 pts is ~256; reg=8.0 gives ~3% shrinkage on true hum
            # while preventing sin/cos (11.7 MHz wavelength) from absorbing ODMR dips in false basins
            reg_diag.extend([8.0, 8.0])

        phi = np.column_stack(cols)
        n_params = phi.shape[1]

        # Wing prior: pull polynomial baseline toward 1.0 (delta_bl -> 0) on [2750, 2775] U [2965, 2990] MHz
        wing_mask = ((freqs <= 2775.0) | (freqs >= 2965.0)) & valid_mask
        phi_wing = phi[wing_mask].copy()
        if mains_freq_hz is not None and n_params >= 2:
            # Wing prior applies to the smooth polynomial drift, not zero-mean AC mains hum
            phi_wing[:, -2:] = 0.0

        wing_weight = 0.20
        reg_mat = np.diag(np.asarray(reg_diag, dtype=np.float64))
        penalty_mat = wing_weight * (phi_wing.T @ phi_wing) + reg_mat

        phi_w = phi * w_vec[:, None]
        gram = phi_w.T @ phi + penalty_mat
        self.phi = phi
        self.k_mat = np.linalg.solve(gram, phi_w.T)  # (P, N)
        self.beta0 = self.k_mat @ delta_y  # (P,)
        self.r0 = (delta_y - (phi @ self.beta0)) * w_vec  # (N,)
        # Symmetric square root of penalty_mat for exact residual augmentation if needed
        eig_w, eig_v = np.linalg.eigh(penalty_mat)
        self.chol_reg = (eig_v * np.sqrt(np.maximum(eig_w, 0.0))) @ eig_v.T

    def solve_batch(self, dips_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve for optimal c >= 0 and baseline on a batch of dip profiles (M, N).

        Args:
            dips_batch: Positive absorption profiles ``1.0 - S_clean(c=1)`` of shape ``(M, N)``.

        Returns:
            Tuple ``(c_opt, costs, resid_matrix)`` where ``c_opt`` has shape ``(M,)``,
            ``costs`` has shape ``(M,)``, and ``resid_matrix`` has shape ``(M, N)``.
        """
        d_w = dips_batch * self.w_vec[None, :]
        if self.oracle_delta_bl is not None:
            num = -(d_w @ self.r0)
            den = np.sum(d_w * d_w, axis=-1) + 1e-12
            c_opt = np.clip(num / den, 0.0, 0.25)
            resid = self.r0[None, :] + c_opt[:, None] * d_w
            costs = 0.5 * np.sum(resid * resid, axis=-1)
            return c_opt, costs, resid

        beta_d = d_w @ self.k_mat.T  # (M, P)
        d_tilde = d_w - (beta_d @ self.phi.T) * self.w_vec[None, :]  # (M, N)
        num = -(d_w @ self.r0)  # (M,)
        den = np.sum(d_w * d_tilde, axis=-1) + 1e-12  # (M,)
        c_opt = np.clip(num / den, 0.0, 0.25)
        resid = self.r0[None, :] + c_opt[:, None] * d_tilde
        beta_tot = self.beta0[None, :] + c_opt[:, None] * beta_d
        reg_resid = beta_tot @ self.chol_reg.T
        costs = 0.5 * (np.sum(resid * resid, axis=-1) + np.sum(reg_resid * reg_resid, axis=-1))
        return c_opt, costs, resid

    def solve_single(
        self, dip_profile: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Solve for optimal amplitude c >= 0 and baseline beta for a single dip profile (N,).

        Returns:
            Tuple ``(c_opt, beta_opt, data_resid, full_aug_resid)``.
        """
        d_w = dip_profile * self.w_vec
        if self.oracle_delta_bl is not None:
            den = float(np.dot(d_w, d_w)) + 1e-12
            c_opt = float(np.clip(-float(np.dot(d_w, self.r0)) / den, 0.0, 0.25))
            data_resid = self.r0 + c_opt * d_w
            return c_opt, np.zeros(0, dtype=np.float64), data_resid, data_resid

        beta_d = self.k_mat @ d_w  # (P,)
        d_tilde = d_w - (self.phi @ beta_d) * self.w_vec
        den = float(np.dot(d_w, d_tilde)) + 1e-12
        c_opt = float(np.clip(-float(np.dot(d_w, self.r0)) / den, 0.0, 0.25))
        beta_opt = self.beta0 + c_opt * beta_d
        data_resid = self.r0 + c_opt * d_tilde
        reg_resid = self.chol_reg @ beta_opt
        full_resid = np.concatenate([data_resid, reg_resid])
        return c_opt, beta_opt, data_resid, full_resid

    def reconstruct_baseline(self, beta_opt: np.ndarray) -> np.ndarray:
        """Reconstruct 1D baseline curve 1.0 + delta_baseline."""
        if self.oracle_delta_bl is not None:
            return 1.0 + self.oracle_delta_bl
        return 1.0 + (self.phi @ beta_opt)


def _fast_single_4axis_dip_profile(
    freqs: np.ndarray,
    b_vec: np.ndarray,
    d0_val: float,
    gamma_val: float,
    eta_val: float = _DEFAULT_ETA,
    e_mhz: float = 0.0,
) -> np.ndarray:
    """Zero-overhead 1-sample unit-contrast dip profile ``d = 1 - S_clean(c=1)``."""
    b_par = np.abs(NV_AXES @ b_vec)
    b_sq = float(b_vec[0] * b_vec[0] + b_vec[1] * b_vec[1] + b_vec[2] * b_vec[2])
    b_perp = np.sqrt(np.maximum(0.0, b_sq - b_par * b_par))

    bz = GAMMA_E_MHZ_MT * b_par
    bx = GAMMA_E_MHZ_MT * b_perp / _SQRT_2

    h = np.zeros((4, 3, 3), dtype=np.float64)
    h[:, 0, 0] = d0_val + bz
    h[:, 0, 1] = bx
    h[:, 1, 0] = bx
    if e_mhz != 0.0:
        h[:, 0, 2] = e_mhz
        h[:, 2, 0] = e_mhz
    h[:, 1, 2] = bx
    h[:, 2, 1] = bx
    h[:, 2, 2] = d0_val - bz

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
        fm = eigvals[:, 1] - e0
        fp = eigvals[:, 2] - e0
    else:
        idx0 = np.argmax(prob_ms0, axis=-1)
        purity_ms0 = np.take_along_axis(prob_ms0, idx0[:, None], axis=-1)[:, 0]
        e0 = np.take_along_axis(eigvals, idx0[:, None], axis=-1)[:, 0]
        mask = np.arange(3)[None, :] != idx0[:, None]
        other = eigvals[mask].reshape(4, 2)
        fm = other[:, 0] - e0
        fp = other[:, 1] - e0

    contrast_factor = purity_ms0 * purity_ms0
    eff_weight = (0.28 / 3.0) * contrast_factor  # (4,)

    f0_axis = np.empty((4, 6), dtype=np.float64)
    f0_axis[:, 0] = fm - A_HYPERFINE_MHZ
    f0_axis[:, 1] = fm
    f0_axis[:, 2] = fm + A_HYPERFINE_MHZ
    f0_axis[:, 3] = fp - A_HYPERFINE_MHZ
    f0_axis[:, 4] = fp
    f0_axis[:, 5] = fp + A_HYPERFINE_MHZ

    diff = freqs[None, None, :] - f0_axis[:, :, None]
    inv_gamma_sq = 1.0 / (gamma_val * gamma_val)
    u_sq = (diff * diff) * inv_gamma_sq
    lor_sum = np.sum(1.0 / (u_sq + 1.0), axis=1)
    gau_sum = np.sum(np.exp(-_LN_2 * u_sq), axis=1)
    axis_prof = (1.0 - eta_val) * lor_sum + eta_val * gau_sum
    return eff_weight @ axis_prof


def _mirror_fold_and_analytic_candidates(
    freqs: np.ndarray,
    detrended_absorption: np.ndarray,
    e_mhz: float = 0.0,
) -> Tuple[float, np.ndarray, List[float]]:
    """Find symmetry center in [2865, 2875] MHz via mirror correlation and build analytic B candidates.

    Smooths the detrended absorption to ~line FWHM, finds ``f_c`` in ``[2865, 2875]``
    MHz maximizing mirror correlation, folds the spectrum around ``f_c``, extracts
    pair offsets ``|gamma_e * B_k|`` (treating a near-D0 degenerate central dip
    explicitly), and solves ``N @ B = s * p`` via least squares for sign patterns
    satisfying ``sum_k s_k * B_k ~ 0``.
    """
    if abs(e_mhz) > 0.0:
        raise ValueError(f"Canonical wedge screening requires e_mhz == 0, got {e_mhz}.")

    df_grid = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.47
    sigma_pts = max(1.5, 3.8 / df_grid)  # ~9 MHz FWHM smoothing
    smoothed = gaussian_filter1d(detrended_absorption, sigma=sigma_pts, mode="nearest")

    # 1. Scan f_c in [2865, 2875] MHz maximizing mirror correlation
    fc_grid = np.linspace(2865.0, 2875.0, 41)
    delta_f = np.linspace(2.0, 95.0, 120)
    best_fc = D0_MHZ
    best_corr = -np.inf

    for fc in fc_grid:
        right = np.interp(fc + delta_f, freqs, smoothed)
        left = np.interp(fc - delta_f, freqs, smoothed)
        r_zm = right - np.mean(right)
        l_zm = left - np.mean(left)
        denom = float(np.linalg.norm(r_zm) * np.linalg.norm(l_zm))
        if denom > 1e-12:
            corr = float(np.dot(r_zm, l_zm)) / denom
            if corr > best_corr:
                best_corr = corr
                best_fc = float(fc)

    # 2. Fold spectrum around best_fc at two smoothing scales (sharp ~5 MHz and FWHM ~9 MHz)
    sigma_sharp = max(1.0, 2.0 / df_grid)
    smoothed_sharp = gaussian_filter1d(detrended_absorption, sigma=sigma_sharp, mode="nearest")
    df_fold = np.linspace(0.0, 96.0, 193)  # 0.5 MHz resolution
    folded_fwhm = 0.5 * (
        np.interp(best_fc + df_fold, freqs, smoothed)
        + np.interp(best_fc - df_fold, freqs, smoothed)
    )
    folded_sharp = 0.5 * (
        np.interp(best_fc + df_fold, freqs, smoothed_sharp)
        + np.interp(best_fc - df_fold, freqs, smoothed_sharp)
    )

    # 3. Extract peak offsets |gamma_e * B_k| from folded spectra
    peaks_sharp, _ = find_peaks(folded_sharp, distance=6, prominence=max(1e-4, 0.08 * float(np.max(folded_sharp))))
    peaks_fwhm, _ = find_peaks(folded_fwhm, distance=8, prominence=max(1e-4, 0.06 * float(np.max(folded_fwhm))))

    peak_b_vals: List[float] = []
    # Explicit near-D0 degenerate pair check: local maximum at or near df = 0
    if folded_fwhm[0] >= folded_fwhm[2] or folded_sharp[0] >= 0.25 * float(np.max(folded_sharp)):
        peak_b_vals.append(0.0)

    all_pk_indices = list(peaks_sharp) + list(peaks_fwhm)
    # Sort peaks by folded amplitude descending
    all_pk_indices.sort(key=lambda idx: float(folded_fwhm[idx]), reverse=True)
    for idx in all_pk_indices:
        b_proj = float(df_fold[idx]) / GAMMA_E_MHZ_MT
        if 0.0 <= b_proj <= 3.25:
            if not any(abs(b_proj - prev) < 0.06 for prev in peak_b_vals):
                peak_b_vals.append(b_proj)
        if len(peak_b_vals) >= 7:
            break

    # Ensure 0.0 is always available as an explicit degenerate-axis candidate
    if not any(abs(x) < 0.04 for x in peak_b_vals):
        peak_b_vals.append(0.0)

    # Estimate characteristic |B| scales from folded energy distribution
    pos_fold = np.maximum(folded_fwhm, 0.0)
    max_pos = float(np.max(pos_fold))
    b_mag_estimates: List[float] = []
    if max_pos > 1e-6:
        # Outer 90% and 95% energy quantiles and strongest peak norm
        significant = df_fold[pos_fold > 0.18 * max_pos]
        if len(significant) > 0:
            max_df = float(np.max(significant))
            for factor in (1.02, 1.15, 1.30):
                b_est = float(np.clip((max_df / GAMMA_E_MHZ_MT) * factor, 0.12, 3.15))
                b_mag_estimates.append(b_est)

    # 4. Build analytic 3D candidate vectors from 4-tuples of peak_b_vals
    n_pks = len(peak_b_vals)
    tuples_to_try: List[Tuple[float, float, float, float]] = []

    # 4 distinct peaks (includes 3 non-zero peaks + 0.0 since 0.0 is in peak_b_vals)
    for i in range(n_pks):
        for j in range(i + 1, n_pks):
            for k in range(j + 1, n_pks):
                for m in range(k + 1, n_pks):
                    tuples_to_try.append((peak_b_vals[i], peak_b_vals[j], peak_b_vals[k], peak_b_vals[m]))

    # Degenerate symmetry plane: 3 peaks with one duplicated
    for i in range(n_pks):
        for j in range(i + 1, n_pks):
            for k in range(j + 1, n_pks):
                tuples_to_try.append((peak_b_vals[i], peak_b_vals[i], peak_b_vals[j], peak_b_vals[k]))
                tuples_to_try.append((peak_b_vals[i], peak_b_vals[j], peak_b_vals[j], peak_b_vals[k]))
                tuples_to_try.append((peak_b_vals[i], peak_b_vals[j], peak_b_vals[k], peak_b_vals[k]))

    # High-symmetry directions: 2 peaks with duplications
    for i in range(min(n_pks, 5)):
        for j in range(i + 1, min(n_pks, 5)):
            tuples_to_try.append((peak_b_vals[i], peak_b_vals[i], peak_b_vals[j], peak_b_vals[j]))
            tuples_to_try.append((peak_b_vals[i], peak_b_vals[j], peak_b_vals[j], peak_b_vals[j]))
            tuples_to_try.append((peak_b_vals[i], peak_b_vals[i], peak_b_vals[i], peak_b_vals[j]))

    scored_analytic: List[Tuple[float, np.ndarray]] = []
    for tup in tuples_to_try:
        p_vec = np.sort(np.asarray(tup, dtype=np.float64))
        p_norm = float(np.linalg.norm(p_vec))
        if p_norm < 0.04:
            continue
        # Check sign patterns with |sum s_k * p_k| small
        signed_p = _SIGN_PATTERNS * p_vec[None, :]  # (8, 4)
        sums = np.abs(np.sum(signed_p, axis=-1))
        order_s = np.argsort(sums)
        for s_idx in order_s[:2]:
            rel_sum = float(sums[s_idx]) / p_norm
            if sums[s_idx] <= max(0.30 * p_norm, 0.12):
                q_vec = signed_p[s_idx]
                b_ls = 0.75 * (NV_AXES.T @ q_vec)
                b_canon = canonicalize_b_vec(b_ls, e_mhz=e_mhz)
                b_mag = float(np.linalg.norm(b_canon))
                if B_MAG_MIN_MT <= b_mag <= B_MAG_MAX_MT:
                    scored_analytic.append((rel_sum, b_canon))

    # Rank analytic candidates by how closely they satisfy the tetrahedral zero-sum law
    scored_analytic.sort(key=lambda item: item[0])
    analytic_vecs: List[np.ndarray] = []
    for _, b_canon in scored_analytic:
        if not any(float(np.linalg.norm(b_canon - prev)) < 0.04 for prev in analytic_vecs):
            analytic_vecs.append(b_canon)
            b_mag = float(np.linalg.norm(b_canon))
            if len(b_mag_estimates) < 6 and not any(abs(b_mag - m) < 0.06 for m in b_mag_estimates):
                b_mag_estimates.append(b_mag)
        if len(analytic_vecs) >= 40:
            break

    if not b_mag_estimates:
        b_mag_estimates = [0.5, 1.2, 2.0]

    arr_analytic = (
        np.asarray(analytic_vecs, dtype=np.float64)
        if analytic_vecs
        else np.zeros((0, 3), dtype=np.float64)
    )
    return best_fc, arr_analytic, b_mag_estimates


def _stage_a_coarse_screen(
    freqs: np.ndarray,
    vp: _VarProjDesign,
    warm_start_b_vec: Optional[np.ndarray] = None,
    e_mhz: float = 0.0,
    top_k: int = 3,
) -> Tuple[List[np.ndarray], float]:
    """Run Stage A batched band-limited coarse screening and select top K=3 distinct candidates."""
    if abs(e_mhz) > 0.0:
        raise ValueError(f"Stage A canonical wedge screening requires e_mhz == 0, got {e_mhz}.")

    detrended_abs = -vp.r0
    fc_est, analytic_cands, b_mag_ests = _mirror_fold_and_analytic_candidates(
        freqs, detrended_abs, e_mhz=e_mhz
    )

    # Deterministic fallback lattice in canonical wedge:
    # 15 canonical directions x (11 uniform magnitudes + adaptive magnitudes near b_mag_ests)
    uniform_mags = np.linspace(0.15, 3.05, 11, dtype=np.float64)
    adaptive_mags = []
    for bm in b_mag_ests[:3]:
        for scale in (0.92, 1.00, 1.08):
            m_val = float(np.clip(bm * scale, 0.08, 3.18))
            if not any(abs(m_val - prev) < 0.04 for prev in adaptive_mags):
                adaptive_mags.append(m_val)
    all_mags = np.unique(np.concatenate([uniform_mags, np.asarray(adaptive_mags, dtype=np.float64)]))

    lattice_vecs = (
        _WEDGE_LATTICE_DIRS[:, None, :] * all_mags[None, :, None]
    ).reshape(-1, 3)

    candidates_list = [lattice_vecs]
    if analytic_cands.shape[0] > 0:
        candidates_list.append(analytic_cands)
    if warm_start_b_vec is not None:
        ws_canon = canonicalize_b_vec(warm_start_b_vec, e_mhz=e_mhz).reshape(1, 3)
        candidates_list.append(ws_canon)

    all_b = np.vstack(candidates_list)
    n_cands = all_b.shape[0]

    # Evaluate all candidates in ONE batched call at unit contrast (c = 1.0)
    b_par = np.abs(all_b @ NV_AXES.T)
    b_sq = np.sum(all_b * all_b, axis=-1, keepdims=True)
    b_perp = np.sqrt(np.maximum(0.0, b_sq - b_par * b_par))

    # Fast band-limited envelope evaluation across full lattice + exact hyperfine on top candidates
    specs_env, _, _ = evaluate_axes_odmr(
        freqs=freqs,
        b_par=b_par,
        b_perp=b_perp,
        d0_arr=np.full(n_cands, fc_est, dtype=np.float64),
        gamma_arr=np.full(n_cands, 5.4, dtype=np.float64),
        contrast_arr=np.ones(n_cands, dtype=np.float64),
        eta_arr=np.full(n_cands, _DEFAULT_ETA, dtype=np.float64),
        include_hyperfine=False,
        single_axis=False,
        e_mhz=e_mhz,
    )
    dips_env = 1.0 - specs_env  # (M, N)

    # Smooth both data residual and model dips to ~line FWHM (~8 MHz) before computing coarse cost
    df_grid = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.47
    sigma_fwhm = max(1.5, 3.2 / df_grid)
    dips_smooth = gaussian_filter1d(dips_env, sigma=sigma_fwhm, axis=-1, mode="nearest")
    r0_smooth = gaussian_filter1d(vp.r0, sigma=sigma_fwhm, mode="nearest") * vp.w_vec

    num_s = -(dips_smooth @ r0_smooth)
    den_s = np.sum(dips_smooth * dips_smooth, axis=-1) + 1e-12
    c_s = np.clip(num_s / den_s, 0.0, 0.25)
    resid_s = r0_smooth[None, :] + c_s[:, None] * dips_smooth
    cost_smooth = 0.5 * np.sum(resid_s * resid_s, axis=-1)

    _, cost_env, _ = vp.solve_batch(dips_env)
    env_order = np.argsort(cost_env + 0.75 * cost_smooth)

    # Evaluate exact hyperfine structure on the top 36 envelope + analytic candidates in one batch
    n_top_hf = min(n_cands, 36)
    top_indices = env_order[:n_top_hf]
    specs_hf, _, _ = evaluate_axes_odmr(
        freqs=freqs,
        b_par=b_par[top_indices],
        b_perp=b_perp[top_indices],
        d0_arr=np.full(n_top_hf, fc_est, dtype=np.float64),
        gamma_arr=np.full(n_top_hf, 4.5, dtype=np.float64),
        contrast_arr=np.ones(n_top_hf, dtype=np.float64),
        eta_arr=np.full(n_top_hf, _DEFAULT_ETA, dtype=np.float64),
        include_hyperfine=True,
        single_axis=False,
        e_mhz=e_mhz,
    )
    _, cost_hf, _ = vp.solve_batch(1.0 - specs_hf)
    hf_score = cost_hf + 0.12 * cost_smooth[top_indices]
    order = top_indices[np.argsort(hf_score)]

    selected: List[np.ndarray] = []
    if warm_start_b_vec is not None:
        ws_arr = np.asarray(warm_start_b_vec, dtype=np.float64).copy()
        selected.append(ws_arr)

    distinct_cands: List[np.ndarray] = []
    for idx in order:
        cand = all_b[idx]
        cand_canon = canonicalize_b_vec(cand, e_mhz=e_mhz)
        cand_norm = max(float(np.linalg.norm(cand_canon)), 1e-9)
        cand_dir = cand_canon / cand_norm
        # Require canonical distance > 0.15 mT and avoid crowding all 3 starts into the same radial |B| shell
        is_distinct = True
        for prev in distinct_cands:
            dist = float(np.linalg.norm(cand_canon - prev))
            prev_norm = max(float(np.linalg.norm(prev)), 1e-9)
            prev_dir = prev / prev_norm
            cos_sim = float(np.clip(np.dot(cand_dir, prev_dir), -1.0, 1.0))
            ang = float(np.arccos(cos_sim))
            if dist <= 0.15 or (ang < 0.08 and dist < 0.45) or (abs(cand_norm - prev_norm) < 0.10 and dist < 0.30):
                is_distinct = False
                break
        if is_distinct:
            distinct_cands.append(cand_canon)
            if len(distinct_cands) >= top_k:
                break

    for dc in distinct_cands:
        if all(
            float(np.linalg.norm(dc - canonicalize_b_vec(s, e_mhz=e_mhz))) > 1e-6
            for s in selected
        ):
            selected.append(dc)

    if not selected:
        selected.append(all_b[order[0]])

    return selected, fc_est


def _fit_1axis_varproj(
    freqs: np.ndarray,
    noisy_spectrum: np.ndarray,
    vp: _VarProjDesign,
    warm_start_b_vec: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fit single-axis [111] ODMR spectrum using Variable Projection."""
    b_grid = np.linspace(0.05, 3.15, 32, dtype=np.float64)
    if warm_start_b_vec is not None:
        ws_bpar = float(abs(np.dot(np.asarray(warm_start_b_vec, dtype=np.float64), NV_AXES[0])))
        b_grid = np.unique(np.append(b_grid, np.clip(ws_bpar, 0.02, 3.2)))

    n_c = len(b_grid)
    specs_u, _, _ = evaluate_axes_odmr(
        freqs=freqs,
        b_par=b_grid[:, None],
        b_perp=np.zeros((n_c, 1), dtype=np.float64),
        d0_arr=np.full(n_c, D0_MHZ, dtype=np.float64),
        gamma_arr=np.full(n_c, 4.5, dtype=np.float64),
        contrast_arr=np.ones(n_c, dtype=np.float64),
        eta_arr=np.full(n_c, _DEFAULT_ETA, dtype=np.float64),
        include_hyperfine=True,
        single_axis=True,
    )
    _, costs, _ = vp.solve_batch(1.0 - specs_u)
    best_bpar = float(b_grid[int(np.argmin(costs))])

    lb = np.array([0.0, -5.0, TRAIN_LINEWIDTH_RANGE[0]], dtype=np.float64)
    ub = np.array([3.3, 5.0, TRAIN_LINEWIDTH_RANGE[1]], dtype=np.float64)
    eta_one = np.array([_DEFAULT_ETA], dtype=np.float64)
    bperp_zero = np.zeros((1, 1), dtype=np.float64)

    def eval_1ax_dip(theta: np.ndarray) -> np.ndarray:
        sp, _, _ = evaluate_axes_odmr(
            freqs=freqs,
            b_par=np.array([[theta[0]]], dtype=np.float64),
            b_perp=bperp_zero,
            d0_arr=np.array([D0_MHZ + theta[1]], dtype=np.float64),
            gamma_arr=np.array([theta[2]], dtype=np.float64),
            contrast_arr=np.ones(1, dtype=np.float64),
            eta_arr=eta_one,
            include_hyperfine=True,
            single_axis=True,
        )
        return 1.0 - sp[0]

    d0_init = eval_1ax_dip(np.array([best_bpar, 0.0, 4.5]))
    _, _, r_init, _ = vp.solve_single(d0_init)
    mad_sigma = max(1.4826 * float(np.median(np.abs(r_init - np.median(r_init)))), 1e-4)

    x0 = np.clip(np.array([best_bpar, 0.0, 4.5], dtype=np.float64), lb + 1e-6, ub - 1e-6)
    res = least_squares(
        lambda th: vp.solve_single(eval_1ax_dip(th))[3],
        x0,
        bounds=(lb, ub),
        method="trf",
        loss="soft_l1",
        f_scale=2.0 * mad_sigma,
        x_scale="jac",
        diff_step=np.array([1e-4, 1e-3, 1e-3]),
        max_nfev=45,
        ftol=1e-6,
        xtol=1e-6,
    )

    dip_final = eval_1ax_dip(res.x)
    c_opt, beta_opt, _, _ = vp.solve_single(dip_final)
    baseline_curve = vp.reconstruct_baseline(beta_opt)
    fitted_curve = baseline_curve - c_opt * dip_final
    fm, fp, _ = solve_axis_transitions(float(res.x[0]), 0.0, d0=D0_MHZ + float(res.x[1]))
    splittings = np.array([abs(float(fp) - float(fm))], dtype=np.float64)
    b_mag = float(abs(res.x[0]))
    b_vec = canonicalize_b_vec(b_mag * NV_AXES[0], e_mhz=0.0)
    resid_valid = (fitted_curve - noisy_spectrum)[vp.valid_mask]
    resid_rms = float(np.sqrt(np.mean(resid_valid * resid_valid)))
    in_range = B_MAG_MIN_MT <= b_mag <= B_MAG_MAX_MT
    status = "converged" if res.success else ("max_nfev" if res.status == 0 else "failed")
    if not in_range:
        status = "out_of_range"

    return {
        "b_mag": b_mag,
        "b_vec": b_vec,
        "b_field_mt": b_mag,
        "b_vec_mt": b_vec,
        "splittings_mhz": splittings,
        "d0_mhz": float(D0_MHZ + res.x[1]),
        "hwhm": float(res.x[2]),
        "contrast": float(c_opt),
        "fitted_curve": fitted_curve,
        "fitted_baseline": baseline_curve,
        "residual_rms": resid_rms,
        "cost": float(res.cost),
        "nfev": int(res.nfev),
        "success": bool(res.success and in_range),
        "status": status,
    }


def _run_varproj_pipeline(
    freqs: np.ndarray,
    noisy_spectrum: np.ndarray,
    vp: _VarProjDesign,
    warm_start_b_vec: Optional[np.ndarray],
    e_mhz: float,
    max_polish_starts: int = 3,
    stage_a_vp: Optional[_VarProjDesign] = None,
    skip_stage_a: bool = False,
    init_d0_mhz: Optional[float] = None,
    init_hwhm_mhz: float = 4.5,
    b_trust_radius_mt: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute Stage A coarse screening + Stage B TRF refinement for a given VarProj design."""
    if skip_stage_a and warm_start_b_vec is not None:
        candidates = [np.asarray(warm_start_b_vec, dtype=np.float64).copy()]
        fc_est = float(init_d0_mhz) if init_d0_mhz is not None else D0_MHZ
    else:
        vp_screen = stage_a_vp if stage_a_vp is not None else vp
        candidates, fc_est = _stage_a_coarse_screen(
            freqs, vp_screen, warm_start_b_vec=warm_start_b_vec, e_mhz=e_mhz, top_k=max_polish_starts
        )
    delta_d0_init = float(np.clip(fc_est - D0_MHZ, -4.5, 4.5))

    # Nonlinear parameter vector: [Bx, By, Bz, delta_d0, hwhm]
    # Note: NO wedge bounds during optimization; canonicalize after fit.
    lb = np.array([-3.30, -3.30, -3.30, -5.0, TRAIN_LINEWIDTH_RANGE[0]], dtype=np.float64)
    ub = np.array([3.30, 3.30, 3.30, 5.0, TRAIN_LINEWIDTH_RANGE[1]], dtype=np.float64)
    if b_trust_radius_mt is not None and warm_start_b_vec is not None:
        ws_b = np.asarray(warm_start_b_vec, dtype=np.float64)
        lb[:3] = np.maximum(lb[:3], ws_b - float(b_trust_radius_mt))
        ub[:3] = np.minimum(ub[:3], ws_b + float(b_trust_radius_mt))
    diff_steps = np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-3], dtype=np.float64)

    # Estimate noise MAD from best Stage A candidate residuals for soft_l1 f_scale
    d_init0 = _fast_single_4axis_dip_profile(freqs, candidates[0], D0_MHZ + delta_d0_init, 4.5, _DEFAULT_ETA, e_mhz)
    _, _, r_init0, _ = vp.solve_single(d_init0)
    valid_r0 = r_init0[vp.valid_mask]
    mad_sigma = max(1.4826 * float(np.median(np.abs(valid_r0 - np.median(valid_r0)))), 1e-4)
    f_scale_val = 2.0 * mad_sigma

    def residual_fn(theta: np.ndarray) -> np.ndarray:
        dip = _fast_single_4axis_dip_profile(
            freqs, theta[:3], D0_MHZ + float(theta[3]), float(theta[4]), _DEFAULT_ETA, e_mhz
        )
        _, _, _, full_res = vp.solve_single(dip)
        b_norm = float(np.sqrt(theta[0] * theta[0] + theta[1] * theta[1] + theta[2] * theta[2]))
        rad_pen = 2.0 * max(0.0, b_norm - B_MAG_MAX_MT) + 2.0 * max(0.0, B_MAG_MIN_MT - b_norm)
        return np.append(full_res, rad_pen)

    best_obj = np.inf
    best_res = None
    total_nfev = 0

    for idx, b_cand in enumerate(candidates):
        x0 = np.array(
            [b_cand[0], b_cand[1], b_cand[2], delta_d0_init, float(init_hwhm_mhz)], dtype=np.float64
        )
        x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)
        nfev_cap = 22 if skip_stage_a else (35 if idx == 0 else 22)
        res = least_squares(
            residual_fn,
            x0,
            bounds=(lb, ub),
            method="trf",
            loss="soft_l1",
            f_scale=f_scale_val,
            x_scale="jac",
            diff_step=diff_steps,
            max_nfev=nfev_cap,
            ftol=1e-6 if idx == 0 else 2e-5,
            xtol=1e-6 if idx == 0 else 2e-5,
        )
        total_nfev += int(res.nfev)
        # Evaluate true penalized L2 cost for fair comparison across starts
        dip_c = _fast_single_4axis_dip_profile(
            freqs, res.x[:3], D0_MHZ + float(res.x[3]), float(res.x[4]), _DEFAULT_ETA, e_mhz
        )
        _, _, _, full_r = vp.solve_single(dip_c)
        l2_cost = 0.5 * float(np.dot(full_r, full_r))
        if stage_a_vp is not None and stage_a_vp is not vp:
            _, _, _, screen_r = stage_a_vp.solve_single(dip_c)
            l2_cost = 0.5 * l2_cost + 0.25 * float(np.dot(screen_r, screen_r))
        if l2_cost < best_obj:
            best_obj = l2_cost
            best_res = res

    assert best_res is not None
    # Polish winner if it hit iteration cap
    if not best_res.success and best_res.status == 0:
        res_polish = least_squares(
            residual_fn,
            np.clip(best_res.x, lb + 1e-6, ub - 1e-6),
            bounds=(lb, ub),
            method="trf",
            loss="soft_l1",
            f_scale=f_scale_val,
            x_scale="jac",
            diff_step=diff_steps,
            max_nfev=25,
            ftol=1e-6,
            xtol=1e-6,
        )
        total_nfev += int(res_polish.nfev)
        dip_p = _fast_single_4axis_dip_profile(
            freqs, res_polish.x[:3], D0_MHZ + float(res_polish.x[3]), float(res_polish.x[4]), _DEFAULT_ETA, e_mhz
        )
        _, _, _, full_rp = vp.solve_single(dip_p)
        l2_cost_p = 0.5 * float(np.dot(full_rp, full_rp))
        if l2_cost_p <= best_obj:
            best_obj = l2_cost_p
            best_res = res_polish

    theta_opt = best_res.x
    b_raw = theta_opt[:3].astype(np.float64)
    b_canon = canonicalize_b_vec(b_raw, e_mhz=e_mhz)
    b_mag = float(np.linalg.norm(b_canon))
    d0_opt = float(D0_MHZ + theta_opt[3])
    hwhm_opt = float(theta_opt[4])

    dip_final = _fast_single_4axis_dip_profile(
        freqs, b_canon, d0_opt, hwhm_opt, _DEFAULT_ETA, e_mhz
    )
    c_opt, beta_opt, _, _ = vp.solve_single(dip_final)
    baseline_curve = vp.reconstruct_baseline(beta_opt)
    fitted_curve = baseline_curve - c_opt * dip_final

    b_par = np.abs(NV_AXES @ b_canon)
    b_perp = np.sqrt(np.maximum(0.0, b_mag * b_mag - b_par * b_par))
    fm, fp, _ = solve_axis_transitions(b_par, b_perp, d0=d0_opt, e_mhz=e_mhz)
    splittings = np.sort(fp - fm)

    resid_valid = (fitted_curve - noisy_spectrum)[vp.valid_mask]
    resid_rms = float(np.sqrt(np.mean(resid_valid * resid_valid)))

    in_range = (B_MAG_MIN_MT - 1e-4) <= b_mag <= (B_MAG_MAX_MT + 1e-4)
    if not np.all(np.isfinite(fitted_curve)) or not np.isfinite(b_mag):
        status = "failed"
        success = False
    elif not in_range:
        status = "out_of_range"
        success = False
    elif best_res.success or best_res.status > 0:
        status = "converged"
        success = True
    elif best_res.status == 0:
        status = "max_nfev"
        success = True
    else:
        status = "failed"
        success = False

    return {
        "b_mag": b_mag,
        "b_vec": b_canon,
        "b_field_mt": b_mag,
        "b_vec_mt": b_canon,
        "splittings_mhz": splittings,
        "d0_mhz": d0_opt,
        "hwhm": hwhm_opt,
        "contrast": float(c_opt),
        "fitted_curve": fitted_curve,
        "fitted_baseline": baseline_curve,
        "residual_rms": resid_rms,
        "cost": float(best_obj),
        "nfev": int(total_nfev),
        "success": bool(success),
        "status": status,
    }


def fit_physical_odmr(
    freqs: np.ndarray,
    noisy_spectrum: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    warm_start_b_vec: Optional[np.ndarray] = None,
    dwell_time: Optional[float] = None,
    *,
    single_axis: bool = False,
    n_starts: int = 3,
    rng: Optional[np.random.Generator] = None,
    variant: str = "plain",
    oracle_baseline: Optional[np.ndarray] = None,
    e_mhz: float = 0.0,
    poly_degree: int = 2,
) -> Dict[str, Any]:
    """Fit a physical spin-1 NV Hamiltonian ODMR model using Variable Projection.

    Linear parameters (dip amplitude scale ``c >= 0`` and polynomial/hum baseline)
    are solved in closed form inside the residual via regularized linear least
    squares with a wing prior pulling the baseline toward 1.0 on ``[2750, 2775]``
    and ``[2965, 2990]`` MHz. Nonlinear parameters ``(Bx, By, Bz, delta_D0, HWHM)``
    are initialized by Stage A batched mirror-fold + canonical-wedge screening and
    refined in Stage B via ``scipy.optimize.least_squares(method="trf", loss="soft_l1")``.

    Args:
        freqs: 1D microwave frequency grid in MHz of shape ``(N,)``.
        noisy_spectrum: 1D observed ODMR spectrum of shape ``(N,)``.
        valid_mask: Optional boolean mask of valid frequency points.
        warm_start_b_vec: Optional 3D magnetic field vector (mT) included in Stage B.
        dwell_time: Optional sweep dwell time per point (seconds if ``< 0.05``,
            milliseconds if ``>= 0.05``). When provided (or when ``variant="informed"``),
            tries 50 Hz and 60 Hz sinusoidal mains-hum basis terms and keeps the
            lower residual.
        single_axis: If True, fit a single [111] axis; else fit the 4-axis ensemble.
        n_starts: Number of distinct coarse candidates to refine in Stage B (default 3).
        rng: Optional random generator (accepted for API compatibility).
        variant: Baseline variant: ``"plain"``, ``"informed"``, or ``"oracle_baseline"``.
        oracle_baseline: True baseline array ``1.0 + drift`` when running the
            oracle-baseline diagnostic/benchmark variant.
        e_mhz: Transverse zero-field strain E in MHz (must be 0.0 when using
            canonical wedge symmetry).
        poly_degree: Polynomial baseline degree (``<= 1`` linear, or ``2`` with
            curvature penalty).

    Returns:
        Dictionary containing ``b_mag``, ``b_vec`` (canonical ``0 <= Bx <= By <= Bz``),
        ``fitted_curve``, ``fitted_baseline``, ``hwhm``, ``success``, ``status``
        (``"converged"`` / ``"max_nfev"`` / ``"out_of_range"`` / ``"failed"``),
        ``nfev``, ``runtime_ms``, ``residual_rms``, plus Stage 1 aliases
        ``b_field_mt``, ``b_vec_mt``, ``splittings_mhz``, ``d0_mhz``, and ``cost``.
    """
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1.")
    if abs(float(e_mhz)) > 0.0:
        raise ValueError(
            f"Canonical tetrahedral wedge 0 <= Bx <= By <= Bz is invalid when strain e_mhz={e_mhz} != 0."
        )

    t_start = time.perf_counter()
    freqs_arr = np.asarray(freqs, dtype=np.float64)
    spec_arr = np.asarray(noisy_spectrum, dtype=np.float64)
    n_pts = freqs_arr.shape[0]

    if valid_mask is None:
        mask_arr = np.isfinite(spec_arr)
    else:
        mask_arr = np.asarray(valid_mask, dtype=bool) & np.isfinite(spec_arr)

    if int(np.sum(mask_arr)) < 16:
        runtime_ms = (time.perf_counter() - t_start) * 1000.0
        zero_vec = np.zeros(3, dtype=np.float64)
        ones_curve = np.ones_like(freqs_arr)
        return {
            "b_mag": 0.0,
            "b_vec": zero_vec,
            "b_field_mt": 0.0,
            "b_vec_mt": zero_vec,
            "splittings_mhz": np.zeros(1 if single_axis else 4, dtype=np.float64),
            "d0_mhz": D0_MHZ,
            "hwhm": 4.5,
            "contrast": 0.0,
            "fitted_curve": ones_curve,
            "fitted_baseline": ones_curve,
            "residual_rms": 0.0,
            "cost": 0.0,
            "nfev": 0,
            "runtime_ms": float(runtime_ms),
            "success": False,
            "status": "failed",
        }

    clean_spec_arr = np.where(mask_arr, spec_arr, 1.0)
    delta_y = clean_spec_arr - 1.0

    oracle_delta_bl: Optional[np.ndarray] = None
    if oracle_baseline is not None or variant == "oracle_baseline":
        if oracle_baseline is not None:
            oracle_delta_bl = np.asarray(oracle_baseline, dtype=np.float64) - 1.0
        else:
            oracle_delta_bl = np.zeros(n_pts, dtype=np.float64)

    # Normalize dwell_time to seconds (if >= 0.05, caller passed milliseconds like 0.8)
    dwell_sec: Optional[float] = None
    if dwell_time is not None and float(dwell_time) > 0.0:
        dt_val = float(dwell_time)
        dwell_sec = dt_val * 1e-3 if dt_val >= 0.05 else dt_val
    elif variant == "informed":
        dwell_sec = 0.8e-3

    use_informed = (
        oracle_delta_bl is None
        and (variant == "informed" or (dwell_sec is not None and variant == "plain"))
    )

    k_polish = min(max(1, n_starts), 3)

    if single_axis:
        vp_1ax = _VarProjDesign(
            freqs_arr,
            delta_y,
            mask_arr,
            poly_degree=poly_degree,
            mains_freq_hz=50.0 if use_informed else None,
            dwell_time_sec=dwell_sec,
            oracle_delta_bl=oracle_delta_bl,
        )
        out = _fit_1axis_varproj(freqs_arr, clean_spec_arr, vp_1ax, warm_start_b_vec)
        out["runtime_ms"] = float((time.perf_counter() - t_start) * 1000.0)
        return out

    if not use_informed:
        vp = _VarProjDesign(
            freqs_arr,
            delta_y,
            mask_arr,
            poly_degree=poly_degree,
            mains_freq_hz=None,
            dwell_time_sec=None,
            oracle_delta_bl=oracle_delta_bl,
        )
        out = _run_varproj_pipeline(
            freqs_arr, clean_spec_arr, vp, warm_start_b_vec, e_mhz, max_polish_starts=k_polish
        )
        out["runtime_ms"] = float((time.perf_counter() - t_start) * 1000.0)
        return out

    # Informed variant: try 50 Hz and 60 Hz mains basis terms, keep lower residual.
    # Step 1: Lock into the 3D magnetic-field basin using the polynomial-only vp_plain
    # (which has no 11.7 MHz / 9.8 MHz periodic grating side-lobes from sin/cos projection).
    # Step 2: Polish inside the winning basin jointly with 50 Hz and 60 Hz mains basis terms
    # and keep the lower residual.
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
        warm_start_b_vec,
        e_mhz,
        max_polish_starts=k_polish,
    )

    vp_50 = _VarProjDesign(
        freqs_arr,
        delta_y,
        mask_arr,
        poly_degree=poly_degree,
        mains_freq_hz=50.0,
        dwell_time_sec=dwell_sec,
        oracle_delta_bl=None,
    )
    vp_60 = _VarProjDesign(
        freqs_arr,
        delta_y,
        mask_arr,
        poly_degree=poly_degree,
        mains_freq_hz=60.0,
        dwell_time_sec=dwell_sec,
        oracle_delta_bl=None,
    )

    dip_basin = _fast_single_4axis_dip_profile(
        freqs_arr, out_basin["b_vec"], out_basin["d0_mhz"], out_basin["hwhm"], _DEFAULT_ETA, e_mhz
    )
    _, _, r_50, _ = vp_50.solve_single(dip_basin)
    _, _, r_60, _ = vp_60.solve_single(dip_basin)
    vp_best_mains = vp_50 if float(np.dot(r_50, r_50)) <= float(np.dot(r_60, r_60)) else vp_60

    best_out = _run_varproj_pipeline(
        freqs_arr,
        clean_spec_arr,
        vp_best_mains,
        warm_start_b_vec=out_basin["b_vec"],
        e_mhz=e_mhz,
        max_polish_starts=1,
        skip_stage_a=True,
        init_d0_mhz=float(out_basin["d0_mhz"]),
        init_hwhm_mhz=float(out_basin["hwhm"]),
        b_trust_radius_mt=0.035,
    )
    best_out["nfev"] = int(out_basin["nfev"]) + int(best_out["nfev"])
    best_out["runtime_ms"] = float((time.perf_counter() - t_start) * 1000.0)
    return best_out


def fit_lorentzian_naive(
    freqs: np.ndarray,
    noisy_spectrum: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fit an uncoupled 8-free-Lorentzian model for the naive baseline benchmark row.

    Finds up to 8 absorption peaks (4 lower-branch, 4 upper-branch around D0),
    fits 8 independent Lorentzian dips plus a linear baseline without Hamiltonian
    axis coupling or 14N hyperfine structure, pairs the 4 resonances symmetrically
    around D0 to form 4 splittings ``df_k``, and estimates ``|B|`` via
    ``b_mag_from_splittings``.
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

    u = (freqs_arr - D0_MHZ) / 120.0
    n_edge = max(4, int(0.08 * n_pts))
    left_y = float(np.median(clean_y[:n_edge]))
    right_y = float(np.median(clean_y[-n_edge:]))
    slope0 = float(np.clip((right_y - left_y) / 1.84, -0.15, 0.15))
    offset0 = float(np.clip(0.5 * (left_y + right_y), 0.8, 1.2))
    bl0 = offset0 + slope0 * u

    smoothed = gaussian_filter1d(clean_y, sigma=4.0, mode="nearest")
    abs_sig = bl0 - smoothed
    peaks, _ = find_peaks(abs_sig, distance=8, prominence=max(5e-4, 0.08 * float(np.max(abs_sig))))
    pk_freqs = freqs_arr[peaks]
    pk_amps = abs_sig[peaks]

    low_mask = pk_freqs < D0_MHZ
    high_mask = pk_freqs >= D0_MHZ
    low_pks = pk_freqs[low_mask][np.argsort(pk_amps[low_mask])[::-1]][:4]
    high_pks = pk_freqs[high_mask][np.argsort(pk_amps[high_mask])[::-1]][:4]

    default_offsets = np.array([8.0, 20.0, 32.0, 46.0], dtype=np.float64)
    f_low_init = np.zeros(4, dtype=np.float64)
    f_high_init = np.zeros(4, dtype=np.float64)
    for k in range(4):
        f_low_init[k] = float(low_pks[k]) if k < len(low_pks) else (D0_MHZ - default_offsets[k])
        f_high_init[k] = float(high_pks[k]) if k < len(high_pks) else (D0_MHZ + default_offsets[k])
    f_low_init = np.sort(f_low_init)
    f_high_init = np.sort(f_high_init)

    centers0 = np.concatenate([f_low_init, f_high_init])  # (8,)
    widths0 = np.full(8, 5.5, dtype=np.float64)
    amps0 = np.full(8, 0.006, dtype=np.float64)
    x0 = np.concatenate([centers0, widths0, amps0, [offset0, slope0]])

    lb = np.concatenate(
        [
            np.full(8, freqs_arr[0] + 5.0),
            np.full(8, 1.5),
            np.zeros(8),
            [0.7, -0.2],
        ]
    )
    ub = np.concatenate(
        [
            np.full(8, freqs_arr[-1] - 5.0),
            np.full(8, 18.0),
            np.full(8, 0.08),
            [1.3, 0.2],
        ]
    )
    x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)

    def eval_8lor(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c_f = p[:8]
        g_f = p[8:16]
        a_f = p[16:24]
        bl = p[24] + p[25] * u
        diff = freqs_arr[:, None] - c_f[None, :]
        g_sq = g_f * g_f
        dips = np.sum(a_f[None, :] * (g_sq[None, :] / (diff * diff + g_sq[None, :])), axis=-1)
        return bl - dips, bl

    res = least_squares(
        lambda p: (eval_8lor(p)[0] - clean_y)[mask_arr],
        x0,
        bounds=(lb, ub),
        method="trf",
        x_scale="jac",
        max_nfev=60,
        ftol=1e-5,
        xtol=1e-5,
    )

    fitted_curve, fitted_bl = eval_8lor(res.x)
    c_sorted = np.sort(res.x[:8])
    # Pair outermost to innermost resonances symmetrically: (c7 - c0), (c6 - c1), (c5 - c2), (c4 - c3)
    splittings = np.sort(np.abs(c_sorted[4:][::-1] - c_sorted[:4]))
    b_mag = float(b_mag_from_splittings(splittings, order=1))
    resid_valid = (fitted_curve - clean_y)[mask_arr]
    resid_rms = float(np.sqrt(np.mean(resid_valid * resid_valid)))
    runtime_ms = float((time.perf_counter() - t_start) * 1000.0)

    return {
        "b_mag": b_mag,
        "b_vec": np.array([0.0, 0.0, b_mag], dtype=np.float64),
        "b_field_mt": b_mag,
        "b_vec_mt": np.array([0.0, 0.0, b_mag], dtype=np.float64),
        "splittings_mhz": splittings,
        "d0_mhz": float(0.5 * (np.mean(c_sorted[:4]) + np.mean(c_sorted[4:]))),
        "hwhm": float(np.median(res.x[8:16])),
        "fitted_curve": fitted_curve,
        "fitted_baseline": fitted_bl,
        "residual_rms": resid_rms,
        "cost": float(res.cost),
        "nfev": int(res.nfev),
        "runtime_ms": runtime_ms,
        "success": bool(res.success),
        "status": "converged" if res.success else "max_nfev",
    }
