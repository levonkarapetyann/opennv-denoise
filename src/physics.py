"""Physical forward model of the spin-1 NV-center diamond ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import numpy as np

from src.config import A_HYPERFINE_MHZ, D0_MHZ, GAMMA_E_MHZ_MT, NV_AXES

_HYPERFINE_OFFSETS: np.ndarray = np.array([-1.0, 0.0, 1.0], dtype=np.float64) * A_HYPERFINE_MHZ
_SQRT_2: float = float(np.sqrt(2.0))
_SQRT_3: float = float(np.sqrt(3.0))
_LN_2: float = float(np.log(2.0))
_BATCH_CHUNK_SIZE: int = 32


@dataclass
class ODMRParams:
    """Parameters governing a continuous-wave NV-center ODMR spectrum.

    Attributes:
        b_field_mt: 3D magnetic field vector (Bx, By, Bz) in millitesla.
        base_contrast: Optical spin readout contrast (fractional dip depth).
        linewidth_mhz: HWHM of each individual hyperfine line in MHz.
        voigt_eta: Pseudo-Voigt mixing factor (0.0 = Lorentzian, 1.0 = Gaussian).
        d0_mhz: Zero-field splitting D0 in MHz.
        include_hyperfine: Whether to split each transition into the 14N triplet.
        single_axis: If True, simulate only the first [111] axis; else all 4 axes.
        e_mhz: Transverse zero-field strain parameter E in MHz (0.0 in Stage 1/1.5).
    """

    b_field_mt: Tuple[float, float, float]
    base_contrast: float = 0.03
    linewidth_mhz: float = 4.5
    voigt_eta: float = 0.35
    d0_mhz: float = D0_MHZ
    include_hyperfine: bool = True
    single_axis: bool = False
    e_mhz: float = 0.0


def canonicalize_b_vec(b_vec: Union[np.ndarray, list, tuple], e_mhz: float = 0.0) -> np.ndarray:
    """Canonicalize 3D magnetic field vector(s) into the tetrahedral wedge 0 <= Bx <= By <= Bz.

    For an unstrained 4-axis NV ensemble (E = 0), the set of 4 axis projections
    ``{|B . n_k|}`` is invariant under the 48-element octahedral group O_h
    (signed coordinate permutations), so every vector has a unique representative
    in ``0 <= Bx <= By <= Bz``. Non-zero transverse strain ``E != 0`` breaks this
    symmetry unless per-axis transverse frame orientations are specified.

    Args:
        b_vec: 3D magnetic field vector of shape ``(3,)`` or batch ``(..., 3)``.
        e_mhz: Transverse zero-field strain parameter E in MHz. Must be 0.0.

    Returns:
        Canonicalized magnetic field vector(s) of the same shape with
        ``0 <= Bx <= By <= Bz``.

    Raises:
        ValueError: If ``e_mhz != 0.0``, since strain breaks octahedral wedge symmetry.
    """
    if abs(float(e_mhz)) > 0.0:
        raise ValueError(
            f"Canonical tetrahedral wedge 0 <= Bx <= By <= Bz is invalid when strain e_mhz={e_mhz} != 0."
        )
    arr = np.asarray(b_vec, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected last dimension of size 3 for b_vec, got shape {arr.shape}.")
    return np.sort(np.abs(arr), axis=-1)


def pseudo_voigt_dip(
    freqs: Union[float, np.ndarray],
    f0: Union[float, np.ndarray],
    amp: Union[float, np.ndarray],
    gamma: Union[float, np.ndarray],
    eta: Union[float, np.ndarray],
) -> np.ndarray:
    """Evaluate a pseudo-Voigt dip profile supporting 1D and batched broadcasting.

    Args:
        freqs: Microwave frequency array in MHz.
        f0: Resonance center frequency (or broadcastable array) in MHz.
        amp: Dip amplitude (or broadcastable array).
        gamma: Half-width at half-maximum (HWHM) in MHz.
        eta: Mixing parameter in [0, 1] (0 = Lorentzian, 1 = Gaussian).

    Returns:
        Non-negative dip absorption profile matching the broadcasted shape.
    """
    freqs_arr = np.asarray(freqs, dtype=np.float64)
    f0_arr = np.asarray(f0, dtype=np.float64)
    amp_arr = np.asarray(amp, dtype=np.float64)
    gamma_arr = np.asarray(gamma, dtype=np.float64)
    eta_arr = np.asarray(eta, dtype=np.float64)

    diff = freqs_arr - f0_arr
    diff_sq = diff * diff
    gamma_sq = gamma_arr * gamma_arr
    lorentzian = gamma_sq / (diff_sq + gamma_sq)
    gaussian = np.exp(-_LN_2 * diff_sq / gamma_sq)
    return amp_arr * ((1.0 - eta_arr) * lorentzian + eta_arr * gaussian)


def solve_axis_transitions(
    b_par: Union[float, np.ndarray],
    b_perp: Union[float, np.ndarray],
    d0: Union[float, np.ndarray] = D0_MHZ,
    e_mhz: Union[float, np.ndarray] = 0.0,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Diagonalize the exact 3x3 spin-1 NV ground-state Hamiltonian.

    Constructs the Hamiltonian in the ``{|+1>, |0>, |-1>}`` basis for single or
    batched inputs, diagonalizes via ``np.linalg.eigh``, identifies the ``|0>``
    eigenstate by maximum projection onto ``|ms=0>``, and computes transition
    frequencies and transverse-field contrast reduction.

    Args:
        b_par: Parallel magnetic field component along the NV axis (mT).
        b_perp: Transverse magnetic field component perpendicular to the axis (mT).
        d0: Zero-field splitting D0 in MHz.
        e_mhz: Transverse strain splitting E in MHz (couples ``|+1>`` and ``|-1>``).

    Returns:
        Tuple ``(f_minus, f_plus, contrast_factor)`` where ``contrast_factor``
        equals ``purity_ms0 ** 2``. Scalar floats are returned when all inputs
        are scalar; otherwise NumPy arrays of the broadcasted batch shape.
    """
    is_scalar = (
        np.ndim(b_par) == 0
        and np.ndim(b_perp) == 0
        and np.ndim(d0) == 0
        and np.ndim(e_mhz) == 0
    )
    b_par_b, b_perp_b, d0_b, e_b = np.broadcast_arrays(
        np.asarray(b_par, dtype=np.float64),
        np.asarray(b_perp, dtype=np.float64),
        np.asarray(d0, dtype=np.float64),
        np.asarray(e_mhz, dtype=np.float64),
    )
    batch_shape = b_par_b.shape

    bz = GAMMA_E_MHZ_MT * b_par_b
    bx = GAMMA_E_MHZ_MT * b_perp_b / _SQRT_2

    hamiltonian = np.zeros((*batch_shape, 3, 3), dtype=np.float64)
    hamiltonian[..., 0, 0] = d0_b + bz
    hamiltonian[..., 0, 1] = bx
    hamiltonian[..., 1, 0] = bx
    hamiltonian[..., 0, 2] = e_b
    hamiltonian[..., 2, 0] = e_b
    hamiltonian[..., 1, 2] = bx
    hamiltonian[..., 2, 1] = bx
    hamiltonian[..., 2, 2] = d0_b - bz

    eigvals, eigvecs = np.linalg.eigh(hamiltonian)

    # Index 1 along the state-basis axis corresponds to |ms=0>.
    prob_ms0 = eigvecs[..., 1, :] * eigvecs[..., 1, :]
    idx0 = np.argmax(prob_ms0, axis=-1)

    if np.all(idx0 == 0):
        purity_ms0 = prob_ms0[..., 0]
        e0 = eigvals[..., 0]
        f_minus = eigvals[..., 1] - e0
        f_plus = eigvals[..., 2] - e0
    else:
        purity_ms0 = np.take_along_axis(prob_ms0, idx0[..., None], axis=-1)[..., 0]
        e0 = np.take_along_axis(eigvals, idx0[..., None], axis=-1)[..., 0]
        state_indices = np.arange(3)
        non_zero_mask = state_indices != idx0[..., None]
        other_eigvals = eigvals[non_zero_mask].reshape(*batch_shape, 2)
        f_minus = other_eigvals[..., 0] - e0
        f_plus = other_eigvals[..., 1] - e0

    contrast_factor = purity_ms0 * purity_ms0

    if is_scalar:
        return float(f_minus), float(f_plus), float(contrast_factor)
    return f_minus, f_plus, contrast_factor


def b_mag_from_splittings(
    splittings_mhz: Union[np.ndarray, list, tuple],
    order: int = 1,
    d0_mhz: float = D0_MHZ,
) -> Union[float, np.ndarray]:
    """Compute scalar |B| in mT from 4-axis or 1-axis ODMR splittings.

    Order 1 (default) uses the exact tetrahedral sum-of-squares formula
    ``|B| = sqrt(3) / (4 * gamma_e) * sqrt(sum_i df_i^2)``. Because the
    positive degenerate-axis level repulsion (``+b_i^4 / D0^2``) and negative
    non-degenerate transverse shift (``-4 a_i^2 b_i^2 / (D0^2 - a_i^2)``)
    carry opposite signs, their sum across the 4 tetrahedral axes cancels to
    ~0.088 uT RMS across ``[0.1, 3.2]`` mT.

    Order 2 applies 5 fixed-point iterations of the uniform second-order
    eigenvalue relation (including transverse level repulsion at ``a_i -> 0``):
    ``df_i ~= 2 * sqrt([a_i * (1 - b_i^2 / (2 * (D0^2 - a_i^2)))]^2 + (b_i^2 / (2 * D0))^2)``,
    inverting ``a_i = gamma_e * B_par_i`` per axis and reconstructing
    ``|B| = sqrt(3) / (2 * gamma_e) * sqrt(sum_i a_i^2)`` (~0.0002 uT RMS).

    Args:
        splittings_mhz: Splitting values ``f_plus - f_minus`` in MHz. Supports
            shape ``(4,)`` or ``(1,)`` for single spectra, or ``(..., 4)`` /
            ``(..., 1)`` for batches.
        order: Perturbation order (``1`` for first-order default, ``2`` for
            iterative second-order inversion).
        d0_mhz: Zero-field splitting D0 in MHz (used when ``order == 2``).

    Returns:
        Scalar magnetic field magnitude in mT (float for 1D input, ndarray for
        batched input).
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}.")

    arr = np.asarray(splittings_mhz, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError("splittings_mhz must have at least 1 dimension (length 1 or 4).")

    n_axes = arr.shape[-1]
    if n_axes == 4:
        if order == 1:
            b_mag = (_SQRT_3 / (4.0 * GAMMA_E_MHZ_MT)) * np.sqrt(
                np.sum(arr * arr, axis=-1)
            )
        else:
            half_df_sq = (0.5 * np.abs(arr)) ** 2
            a = 0.5 * np.abs(arr)
            d0_val = float(d0_mhz)
            d0_sq = d0_val * d0_val
            for _ in range(5):
                g2_b2 = (3.0 / 4.0) * np.sum(a * a, axis=-1, keepdims=True)
                b_sq = np.maximum(0.0, g2_b2 - a * a)
                repulsion_sq = (b_sq / (2.0 * d0_val)) ** 2
                rem_sq = np.maximum(0.0, half_df_sq - repulsion_sq)
                denom = np.maximum(d0_sq - a * a, 1e-6)
                corr = np.clip(1.0 - b_sq / (2.0 * denom), 0.5, 1.0)
                a = np.sqrt(rem_sq) / corr
            b_mag = (_SQRT_3 / (2.0 * GAMMA_E_MHZ_MT)) * np.sqrt(
                np.sum(a * a, axis=-1)
            )
    elif n_axes == 1:
        b_mag = np.abs(arr[..., 0]) / (2.0 * GAMMA_E_MHZ_MT)
    else:
        raise ValueError(f"Expected 1 or 4 axis splittings, got shape {arr.shape}.")

    if arr.ndim == 1:
        return float(b_mag)
    return b_mag


def _accumulate_hyperfine_spectra(
    freqs_1d: np.ndarray,
    f_minus: np.ndarray,
    f_plus: np.ndarray,
    eff_amp: np.ndarray,
    gamma_arr: np.ndarray,
    eta_arr: np.ndarray,
    include_hyperfine: bool,
) -> np.ndarray:
    """Compute clean spectra from transition frequencies using L2-cached chunking.

    Exploits the fact that all hyperfine components of a given axis share the
    same amplitude ``eff_amp[b, k] / 3.0``, linewidth ``gamma_arr[b]``, and
    mixing factor ``eta_arr[b]``, summing the unscaled Lorentzian and Gaussian
    basis profiles per axis before applying amplitude and ``eta`` weights.
    """
    batch_size, n_axes = f_minus.shape
    n_freqs = freqs_1d.shape[0]
    spectra = np.empty((batch_size, n_freqs), dtype=np.float64)

    for start in range(0, batch_size, _BATCH_CHUNK_SIZE):
        end = min(start + _BATCH_CHUNK_SIZE, batch_size)
        fm_c = f_minus[start:end]
        fp_c = f_plus[start:end]
        amp_c = eff_amp[start:end]
        gamma_c = gamma_arr[start:end]
        eta_c = eta_arr[start:end]
        inv_gamma_sq = 1.0 / (gamma_c * gamma_c)

        if include_hyperfine:
            # Group the 6 hyperfine line centers of each axis: shape (B_c, K, 6)
            f0_c = np.empty((end - start, n_axes, 6), dtype=np.float64)
            f0_c[:, :, 0] = fm_c - A_HYPERFINE_MHZ
            f0_c[:, :, 1] = fm_c
            f0_c[:, :, 2] = fm_c + A_HYPERFINE_MHZ
            f0_c[:, :, 3] = fp_c - A_HYPERFINE_MHZ
            f0_c[:, :, 4] = fp_c
            f0_c[:, :, 5] = fp_c + A_HYPERFINE_MHZ

            diff = freqs_1d[None, None, None, :] - f0_c[:, :, :, None]
            u_sq = (diff * diff) * inv_gamma_sq[:, None, None, None]
            lor_sum = np.sum(1.0 / (u_sq + 1.0), axis=2)
            gau_sum = np.sum(np.exp(-_LN_2 * u_sq), axis=2)
            axis_prof = (
                (1.0 - eta_c)[:, None, None] * lor_sum
                + eta_c[:, None, None] * gau_sum
            )
            spectra[start:end] = 1.0 - np.einsum(
                "bk,bkn->bn", amp_c * (1.0 / 3.0), axis_prof
            )
        else:
            f0_c = np.stack([fm_c, fp_c], axis=2)  # (B_c, K, 2)
            diff = freqs_1d[None, None, None, :] - f0_c[:, :, :, None]
            u_sq = (diff * diff) * inv_gamma_sq[:, None, None, None]
            lor_sum = np.sum(1.0 / (u_sq + 1.0), axis=2)
            gau_sum = np.sum(np.exp(-_LN_2 * u_sq), axis=2)
            axis_prof = (
                (1.0 - eta_c)[:, None, None] * lor_sum
                + eta_c[:, None, None] * gau_sum
            )
            spectra[start:end] = 1.0 - np.einsum("bk,bkn->bn", amp_c, axis_prof)

    return spectra


def evaluate_axes_odmr(
    freqs: np.ndarray,
    b_par: np.ndarray,
    b_perp: np.ndarray,
    d0_arr: np.ndarray,
    gamma_arr: np.ndarray,
    contrast_arr: np.ndarray,
    eta_arr: np.ndarray,
    include_hyperfine: bool = True,
    single_axis: bool = False,
    e_mhz: Union[float, np.ndarray] = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate clean ODMR spectra and transition pairs from axis field components.

    Shared vectorized core used by single-sample simulation, batched data
    generation, and the multi-start physical baseline fitter.

    Args:
        freqs: 1D frequency grid in MHz of shape ``(N,)``.
        b_par: Parallel field magnitudes of shape ``(B, K)`` in mT.
        b_perp: Transverse field magnitudes of shape ``(B, K)`` in mT.
        d0_arr: Zero-field splittings of shape ``(B,)`` in MHz.
        gamma_arr: HWHM linewidths of shape ``(B,)`` in MHz.
        contrast_arr: Base contrasts of shape ``(B,)``.
        eta_arr: Pseudo-Voigt mixing factors of shape ``(B,)``.
        include_hyperfine: Whether to expand each transition into 3 14N lines.
        single_axis: Whether K=1 (single axis) or K=4 (4-axis ensemble).
        e_mhz: Transverse strain splitting E in MHz (default 0.0).

    Returns:
        Tuple ``(spectra, f_minus, f_plus)`` with shapes ``(B, N)``,
        ``(B, K)``, and ``(B, K)``.
    """
    freqs_1d = np.asarray(freqs, dtype=np.float64)
    e_arr = np.asarray(e_mhz, dtype=np.float64)
    if e_arr.ndim == 1 and e_arr.shape[0] == b_par.shape[0]:
        e_arg: Union[float, np.ndarray] = e_arr[:, None]
    else:
        e_arg = e_arr

    f_minus, f_plus, contrast_factor = solve_axis_transitions(
        b_par, b_perp, d0=d0_arr[:, None], e_mhz=e_arg
    )

    axis_weight = 1.0 if single_axis else 0.28
    eff_amp = contrast_arr[:, None] * axis_weight * contrast_factor  # (B, K)
    spectra = _accumulate_hyperfine_spectra(
        freqs_1d=freqs_1d,
        f_minus=f_minus,
        f_plus=f_plus,
        eff_amp=eff_amp,
        gamma_arr=gamma_arr,
        eta_arr=eta_arr,
        include_hyperfine=include_hyperfine,
    )
    return spectra, f_minus, f_plus


def simulate_batch_clean_odmr(
    freqs: np.ndarray,
    b_vecs: np.ndarray,
    d0_arr: Union[float, np.ndarray],
    gamma_arr: Union[float, np.ndarray],
    contrast_arr: Union[float, np.ndarray],
    eta_arr: Union[float, np.ndarray],
    include_hyperfine: bool = True,
    single_axis: bool = False,
    return_details: bool = False,
    e_mhz: Union[float, np.ndarray] = 0.0,
) -> Union[np.ndarray, Dict[str, Any]]:
    """Simulate a batch of clean ODMR spectra using vectorized ``(B, K, 3, 3)`` eigh.

    Args:
        freqs: 1D frequency grid in MHz of shape ``(N,)``.
        b_vecs: Magnetic field vectors of shape ``(B, 3)`` in mT.
        d0_arr: Zero-field splitting(s) in MHz, scalar or shape ``(B,)``.
        gamma_arr: HWHM linewidth(s) in MHz, scalar or shape ``(B,)``.
        contrast_arr: Base contrast(s), scalar or shape ``(B,)``.
        eta_arr: Pseudo-Voigt mixing factor(s), scalar or shape ``(B,)``.
        include_hyperfine: Whether to include 14N triplet hyperfine structure.
        single_axis: If True, simulate only axis 0; otherwise all 4 [111] axes.
        return_details: If True, return a dictionary including splittings and
            transition pairs alongside ``"spectra"``.
        e_mhz: Transverse strain splitting E in MHz, scalar or shape ``(B,)``.

    Returns:
        2D array of clean spectra with shape ``(B, N)`` when
        ``return_details=False``, or a dictionary of batched outputs when
        ``return_details=True``.
    """
    b_vecs_arr = np.asarray(b_vecs, dtype=np.float64)
    if b_vecs_arr.ndim != 2 or b_vecs_arr.shape[1] != 3:
        raise ValueError(f"Expected b_vecs of shape (B, 3), got {b_vecs_arr.shape}.")

    batch_size = b_vecs_arr.shape[0]
    d0_b = np.broadcast_to(np.asarray(d0_arr, dtype=np.float64), (batch_size,))
    gamma_b = np.broadcast_to(np.asarray(gamma_arr, dtype=np.float64), (batch_size,))
    contrast_b = np.broadcast_to(np.asarray(contrast_arr, dtype=np.float64), (batch_size,))
    eta_b = np.broadcast_to(np.asarray(eta_arr, dtype=np.float64), (batch_size,))
    e_b = np.broadcast_to(np.asarray(e_mhz, dtype=np.float64), (batch_size,))

    axes = NV_AXES[:1] if single_axis else NV_AXES
    b_par = np.abs(b_vecs_arr @ axes.T)
    b_sq = np.sum(b_vecs_arr * b_vecs_arr, axis=-1, keepdims=True)
    b_perp = np.sqrt(np.maximum(0.0, b_sq - b_par * b_par))

    spectra, f_minus, f_plus = evaluate_axes_odmr(
        freqs=freqs,
        b_par=b_par,
        b_perp=b_perp,
        d0_arr=d0_b,
        gamma_arr=gamma_b,
        contrast_arr=contrast_b,
        eta_arr=eta_b,
        include_hyperfine=include_hyperfine,
        single_axis=single_axis,
        e_mhz=e_b,
    )

    if not return_details:
        return spectra

    splittings = np.sort(f_plus - f_minus, axis=-1)
    b_mags = np.linalg.norm(b_vecs_arr, axis=-1)
    b_from_split = b_mag_from_splittings(splittings)
    dip_pairs = np.stack([f_minus, f_plus], axis=-1)
    return {
        "spectra": spectra,
        "splittings_mhz": splittings,
        "b_magnitude_mt": b_mags,
        "b_From_splittings_mt": b_from_split,
        "b_from_splittings_mt": b_from_split,
        "dip_pairs": dip_pairs,
    }


def simulate_clean_odmr(freqs: np.ndarray, params: ODMRParams) -> Dict[str, Any]:
    """Simulate a single clean ODMR spectrum from physical parameters.

    Args:
        freqs: 1D frequency grid in MHz.
        params: ``ODMRParams`` specification.

    Returns:
        Dictionary with ``"spectrum"``, ``"splittings_mhz"``,
        ``"b_magnitude_mt"``, ``"b_From_splittings_mt"``, ``"d0_mhz"``,
        and ``"dip_pairs"``.
    """
    b_vec = np.asarray(params.b_field_mt, dtype=np.float64).reshape(1, 3)
    batch_res = simulate_batch_clean_odmr(
        freqs=freqs,
        b_vecs=b_vec,
        d0_arr=np.array([params.d0_mhz], dtype=np.float64),
        gamma_arr=np.array([params.linewidth_mhz], dtype=np.float64),
        contrast_arr=np.array([params.base_contrast], dtype=np.float64),
        eta_arr=np.array([params.voigt_eta], dtype=np.float64),
        include_hyperfine=params.include_hyperfine,
        single_axis=params.single_axis,
        return_details=True,
        e_mhz=np.array([params.e_mhz], dtype=np.float64),
    )
    b_from_split = float(batch_res["b_From_splittings_mt"][0])
    return {
        "spectrum": batch_res["spectra"][0],
        "splittings_mhz": batch_res["splittings_mhz"][0],
        "b_magnitude_mt": float(batch_res["b_magnitude_mt"][0]),
        "b_From_splittings_mt": b_from_split,
        "b_from_splittings_mt": b_from_split,
        "d0_mhz": float(params.d0_mhz),
        "dip_pairs": batch_res["dip_pairs"][0],
    }
