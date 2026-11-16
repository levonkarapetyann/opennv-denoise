"""OpenNV-Denoise Stage 1 physical simulation, hardware noise, and baseline package."""

from src.baseline import fit_lorentzian_naive, fit_physical_odmr
from src.config import (
    A_HYPERFINE_MHZ,
    B_MAG_MAX_MT,
    B_MAG_MIN_MT,
    CANONICAL_FREQS,
    CENTRAL_REQUIRED_MAX_MHZ,
    CENTRAL_REQUIRED_MIN_MHZ,
    D0_MHZ,
    FREQ_MAX_MHZ,
    FREQ_MIN_MHZ,
    GAMMA_E_MHZ_MT,
    N_POINTS,
    NV_AXES,
    OOD_LINEWIDTH_RANGE,
    TRAIN_LINEWIDTH_RANGE,
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

__all__ = [
    "A_HYPERFINE_MHZ",
    "B_MAG_MAX_MT",
    "B_MAG_MIN_MT",
    "CANONICAL_FREQS",
    "CENTRAL_REQUIRED_MAX_MHZ",
    "CENTRAL_REQUIRED_MIN_MHZ",
    "D0_MHZ",
    "FREQ_MAX_MHZ",
    "FREQ_MIN_MHZ",
    "GAMMA_E_MHZ_MT",
    "N_POINTS",
    "NV_AXES",
    "OOD_LINEWIDTH_RANGE",
    "TRAIN_LINEWIDTH_RANGE",
    "ODMRParams",
    "HardwareNoiseConfig",
    "apply_realistic_noise",
    "b_mag_from_splittings",
    "canonicalize_b_vec",
    "fit_lorentzian_naive",
    "fit_physical_odmr",
    "generate_pink_noise",
    "pseudo_voigt_dip",
    "resample_spectrum_pchip",
    "simulate_batch_clean_odmr",
    "simulate_clean_odmr",
    "solve_axis_transitions",
]

