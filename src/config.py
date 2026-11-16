"""Physical constants and canonical grid parameters for OpenNV-Denoise."""

from __future__ import annotations

import numpy as np

# Spin-1 NV-center ground-state Hamiltonian constants (no rounding)
D0_MHZ: float = 2870.0  # Zero-field splitting in MHz
GAMMA_E_MHZ_MT: float = 28.025  # Electron gyromagnetic ratio in MHz/mT
A_HYPERFINE_MHZ: float = 2.16  # 14N hyperfine splitting in MHz, m_I in {-1, 0, +1}

# Four tetrahedral [111] unit vectors of the diamond NV lattice
NV_AXES: np.ndarray = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ],
    dtype=np.float64,
) / np.sqrt(3.0)

# Canonical microwave frequency sweep grid (MHz)
FREQ_MIN_MHZ: float = 2750.0
FREQ_MAX_MHZ: float = 2990.0
N_POINTS: int = 512
CANONICAL_FREQS: np.ndarray = np.linspace(
    FREQ_MIN_MHZ, FREQ_MAX_MHZ, N_POINTS, dtype=np.float64
)

# Minimum required central resonance frequency coverage for valid resampling
CENTRAL_REQUIRED_MIN_MHZ: float = 2830.0
CENTRAL_REQUIRED_MAX_MHZ: float = 2910.0

# Magnetic field and linewidth training / OOD ranges
B_MAG_MIN_MT: float = 0.05
B_MAG_MAX_MT: float = 3.20
TRAIN_LINEWIDTH_RANGE: tuple[float, float] = (3.0, 9.0)
OOD_LINEWIDTH_RANGE: tuple[float, float] = (9.5, 13.5)

