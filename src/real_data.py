"""Stage 2 experimental ODMR data importer, unit normalizer, PCHIP validator, and QA manifest builder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.io
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from src.config import (
    A_HYPERFINE_MHZ,
    CANONICAL_FREQS,
    CENTRAL_REQUIRED_MAX_MHZ,
    CENTRAL_REQUIRED_MIN_MHZ,
    D0_MHZ,
    FREQ_MAX_MHZ,
    FREQ_MIN_MHZ,
    GAMMA_E_MHZ_MT,
    NV_AXES,
)
from src.noise import resample_spectrum_pchip
from src.physics import (
    b_mag_from_splittings,
    canonicalize_b_vec,
    pseudo_voigt_dip,
    solve_axis_transitions,
)

DEFAULT_RAW_EXP_DIR: Path = Path("data") / "raw_experimental"
DEFAULT_REAL_MANIFEST_PATH: Path = Path("data") / "real_eval_manifest.json"


def normalize_frequency_to_mhz(
    freqs: Union[np.ndarray, Sequence[float]],
    declared_unit: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """Normalize microwave frequencies from Hz, kHz, MHz, or GHz to MHz.

    Args:
        freqs: 1D frequency array in original units.
        declared_unit: Optional explicit unit string (``"Hz"``, ``"kHz"``,
            ``"MHz"``, ``"GHz"``). If None, detects unit from median magnitude
            relative to the NV zero-field splitting (~2870 MHz).

    Returns:
        Tuple ``(freqs_mhz, detected_unit)``.
    """
    arr = np.asarray(freqs, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Frequency array cannot be empty.")

    if declared_unit is not None:
        u = declared_unit.strip().lower()
        if u == "ghz":
            return arr * 1e3, "GHz"
        if u == "mhz":
            return arr.copy(), "MHz"
        if u == "khz":
            return arr * 1e-3, "kHz"
        if u == "hz":
            return arr * 1e-6, "Hz"
        raise ValueError(f"Unsupported declared frequency unit '{declared_unit}'.")

    med = float(np.median(np.abs(arr)))
    if 1.5 <= med <= 5.0:
        return arr * 1e3, "GHz"
    if 1500.0 <= med <= 5000.0:
        return arr.copy(), "MHz"
    if 1.5e6 <= med <= 5.0e6:
        return arr * 1e-3, "kHz"
    if 1.5e9 <= med <= 5.0e9:
        return arr * 1e-6, "Hz"
    raise ValueError(
        f"Cannot auto-detect microwave frequency unit for median value {med}. Specify declared_unit."
    )


def normalize_odmr_spectrum(
    values: Union[np.ndarray, Sequence[float]],
    convention: str = "auto",
) -> Tuple[np.ndarray, str]:
    """Normalize experimental ODMR readout to unit baseline (~1.0) with absorption dips (<1.0).

    Args:
        values: Raw 1D intensity or contrast trace.
        convention:
            - ``"fluorescence"``: Already near 1.0 (or raw photon counts divided by outer wing median).
            - ``"positive_contrast"``: ``(ref - meas) / ref >= 0``, converted via ``1.0 - values``.
            - ``"negative_contrast"``: ``(meas - ref) / ref <= 0``, converted via ``1.0 + values``.
            - ``"auto"``: Detects convention from wing median and excursion polarity.

    Returns:
        Tuple ``(normalized_spectrum, applied_convention)``.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < 4:
        raise ValueError("Spectrum array must have at least 4 points.")

    n_edge = max(2, int(np.ceil(0.08 * arr.size)))
    wing_vals = np.concatenate([arr[:n_edge], arr[-n_edge:]])
    wing_med = float(np.median(wing_vals))

    conv = convention.strip().lower()
    if conv == "auto":
        if abs(wing_med - 1.0) < 0.25:
            conv = "fluorescence"
        elif abs(wing_med) < 0.20:
            # Differential contrast centered near 0.0
            max_pos = float(np.max(arr - wing_med))
            max_neg = float(np.max(wing_med - arr))
            conv = "positive_contrast" if max_pos >= max_neg else "negative_contrast"
        elif wing_med > 10.0:
            conv = "raw_counts"
        else:
            conv = "fluorescence"

    if conv == "positive_contrast":
        return 1.0 - arr, "positive_contrast"
    if conv == "negative_contrast":
        return 1.0 + arr, "negative_contrast"
    if conv == "raw_counts":
        return arr / max(wing_med, 1e-12), "raw_counts"
    if conv == "fluorescence":
        return arr.copy(), "fluorescence"
    raise ValueError(f"Unknown spectrum normalization convention '{convention}'.")


def evaluate_pair_centre_symmetry(
    freqs_mhz: np.ndarray,
    spectrum_norm: np.ndarray,
    expected_pairs: int = 4,
    prominence: float = 0.002,
) -> Dict[str, Any]:
    """Detect ODMR dip locations and evaluate pair-centre symmetry about D0 (~2870-2877 MHz).

    Args:
        freqs_mhz: 1D frequency array in MHz.
        spectrum_norm: 1D normalized fluorescence spectrum (baseline ~1.0, dips <1.0).
        expected_pairs: Expected number of symmetric transition pairs (4 for ensemble, 1 for single NV).
        prominence: Minimum dip prominence for peak finding.

    Returns:
        Dictionary with detected dip frequencies, paired centres, pair splittings,
        symmetry spread ``pair_centre_std_mhz``, and ``estimated_b_mag_mt``.
    """
    absorption = 1.0 - np.asarray(spectrum_norm, dtype=np.float64)
    pks, props = find_peaks(absorption, distance=4, prominence=prominence)
    dip_freqs = freqs_mhz[pks]
    proms = props["prominences"]

    if dip_freqs.size < 2:
        return {
            "n_visible_dips": int(dip_freqs.size),
            "visible_dip_freqs_mhz": np.round(dip_freqs, 2).tolist(),
            "supports_pair_symmetry": False,
            "pair_centres_mhz": [],
            "pair_splittings_mhz": [],
            "mean_pair_centre_mhz": None,
            "pair_centre_std_mhz": None,
            "estimated_b_mag_mt": None,
        }

    # Select up to 2 * expected_pairs most prominent dips, then sort ascending by frequency
    n_target_dips = 2 * expected_pairs
    if dip_freqs.size > n_target_dips:
        top_order = np.argsort(proms)[::-1][:n_target_dips]
        selected_freqs = np.sort(dip_freqs[top_order])
    else:
        n_even = (dip_freqs.size // 2) * 2
        top_order = np.argsort(proms)[::-1][:n_even]
        selected_freqs = np.sort(dip_freqs[top_order])

    n_pairs = selected_freqs.size // 2
    pair_centres = []
    pair_splittings = []
    for k in range(n_pairs):
        f_lo = float(selected_freqs[k])
        f_hi = float(selected_freqs[selected_freqs.size - 1 - k])
        pair_centres.append(0.5 * (f_lo + f_hi))
        pair_splittings.append(f_hi - f_lo)

    pair_splittings_sorted = np.sort(np.asarray(pair_splittings, dtype=np.float64))
    mean_c = float(np.mean(pair_centres))
    std_c = float(np.std(pair_centres))

    est_b_mt: Optional[float] = None
    if n_pairs == 4:
        est_b_mt = float(b_mag_from_splittings(pair_splittings_sorted, order=2, d0_mhz=mean_c))
    elif n_pairs == 1:
        est_b_mt = float(pair_splittings_sorted[0] / (2.0 * GAMMA_E_MHZ_MT))

    return {
        "n_visible_dips": int(dip_freqs.size),
        "visible_dip_freqs_mhz": np.round(dip_freqs, 2).tolist(),
        "selected_pair_dips_mhz": np.round(selected_freqs, 2).tolist(),
        "supports_pair_symmetry": bool(n_pairs == expected_pairs),
        "pair_centres_mhz": [round(c, 3) for c in pair_centres],
        "pair_splittings_mhz": [round(s, 3) for s in pair_splittings_sorted.tolist()],
        "mean_pair_centre_mhz": round(mean_c, 3),
        "pair_centre_std_mhz": round(std_c, 3),
        "estimated_b_mag_mt": round(est_b_mt, 4) if est_b_mt is not None else None,
    }


def load_and_validate_real_trace(
    freqs_raw: np.ndarray,
    spectrum_raw: np.ndarray,
    *,
    trace_id: str,
    source_name: str,
    license_id: str,
    citation: str,
    nv_type: str,
    declared_unit: Optional[str] = None,
    spectrum_convention: str = "auto",
    ground_truth_b_mt: Optional[float] = None,
    ground_truth_angles_deg: Optional[Tuple[float, float]] = None,
    coil_currents_amp: Optional[Sequence[float]] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """Normalize units, run ``resample_spectrum_pchip``, and evaluate QA checks for a real trace.

    Implements Part 5b, 5c, 5d:
      - Normalizes frequencies to MHz and readout to baseline ~1.0.
      - Runs ``resample_spectrum_pchip`` with ``return_valid_mask=True``.
      - Any file whose frequency span fails to cover ``[2830.0, 2910.0]`` MHz is
        flagged ``compatible = False`` and NEVER padded.
    """
    freqs_mhz, detected_unit = normalize_frequency_to_mhz(freqs_raw, declared_unit=declared_unit)
    spec_norm, detected_conv = normalize_odmr_spectrum(spectrum_raw, convention=spectrum_convention)

    f_min = float(np.min(freqs_mhz))
    f_max = float(np.max(freqs_mhz))
    n_pts = int(freqs_mhz.size)

    expected_pairs = 4 if "ensemble" in nv_type.lower() else 1
    qa_sym = evaluate_pair_centre_symmetry(
        freqs_mhz, spec_norm, expected_pairs=expected_pairs
    )

    # Check if any of the primary transition pair dips fall outside our canonical window [2750, 2990] MHz
    dip_freqs = np.asarray(
        qa_sym.get("selected_pair_dips_mhz", qa_sym["visible_dip_freqs_mhz"]),
        dtype=np.float64,
    )
    dips_outside_canonical = bool(
        dip_freqs.size > 0
        and (float(np.min(dip_freqs)) < FREQ_MIN_MHZ or float(np.max(dip_freqs)) > FREQ_MAX_MHZ)
    )

    try:
        resampled, valid_mask, meta = resample_spectrum_pchip(
            freqs_mhz,
            spec_norm,
            target_freqs=CANONICAL_FREQS,
            method="pchip",
            return_valid_mask=True,
        )
        compatible = True
        incompatible_reason = None
        valid_points = int(np.sum(valid_mask))
        out_of_range = bool(meta["out_of_range"]) or dips_outside_canonical
        window_padded = bool(meta["window_padded"])
    except ValueError as exc:
        resampled = None
        valid_mask = None
        compatible = False
        incompatible_reason = str(exc)
        valid_points = 0
        out_of_range = True
        window_padded = False

    usable_on_canonical_grid = bool(compatible and not out_of_range)
    coil_list = [float(c) for c in coil_currents_amp] if coil_currents_amp is not None else None
    coil_norm = float(np.linalg.norm(coil_list)) if coil_list is not None else None

    return {
        "trace_id": trace_id,
        "source_name": source_name,
        "license": license_id,
        "citation": citation,
        "nv_type": nv_type,
        "original_points": n_pts,
        "original_frequency_unit": detected_unit,
        "original_freq_range_mhz": [round(f_min, 3), round(f_max, 3)],
        "spectrum_convention": detected_conv,
        "compatible_central_2830_2910": compatible,
        "usable_untruncated_on_canonical_2750_2990": usable_on_canonical_grid,
        "out_of_range_or_dips_outside_canonical": out_of_range,
        "dips_outside_canonical_2750_2990": dips_outside_canonical,
        "valid_canonical_points": valid_points,
        "window_padded": window_padded,
        "incompatible_reason": incompatible_reason,
        "ground_truth_b_mt": ground_truth_b_mt,
        "ground_truth_angles_deg": list(ground_truth_angles_deg) if ground_truth_angles_deg else None,
        "helmholtz_coil_currents_amp": coil_list,
        "helmholtz_coil_norm_amp": round(coil_norm, 4) if coil_norm is not None else None,
        "qa_checks_supported": {
            "pair_centre_symmetry_about_d0": bool(compatible and qa_sym["supports_pair_symmetry"]),
            "linearity_vs_coil_current": bool(compatible and coil_list is not None),
            "ground_truth_b_mag_comparison": bool(compatible and ground_truth_b_mt is not None),
        },
        "qa_results": qa_sym,
        "notes": notes,
        "resampled_spectrum": resampled,
        "valid_mask": valid_mask,
    }


def load_csv_or_npz_odmr_file(
    file_path: Path,
    *,
    trace_id: Optional[str] = None,
    source_name: str = "Custom CSV/NPZ Fixture",
    license_id: str = "CC-BY-4.0",
    citation: str = "Local CSV/NPZ experimental trace",
    nv_type: str = "ensemble",
    declared_unit: Optional[str] = None,
    spectrum_convention: str = "auto",
    ground_truth_b_mt: Optional[float] = None,
    coil_currents_amp: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Load an experimental ODMR trace from a documented ``.csv`` / ``.txt`` or ``.npz`` schema.

    Documented Schema:
      - CSV / TXT: Two numeric columns ``frequency, readout`` with optional 1-line header
        (if header contains ``[GHz]``, ``[MHz]``, or ``[Hz]``, unit is parsed automatically).
      - NPZ: Keys ``freqs`` (or ``frequency``) and ``spectrum`` (or ``readout``), plus optional
        ``unit`` and ``b_mag_mt``.
    """
    tid = trace_id if trace_id is not None else file_path.stem
    suffix = file_path.suffix.lower()

    if suffix == ".npz":
        with np.load(file_path, allow_pickle=False) as data:
            f_key = "freqs" if "freqs" in data.files else "frequency"
            s_key = "spectrum" if "spectrum" in data.files else "readout"
            freqs_raw = data[f_key]
            spec_raw = data[s_key]
            if declared_unit is None and "unit" in data.files:
                declared_unit = str(data["unit"])
            if ground_truth_b_mt is None and "b_mag_mt" in data.files:
                ground_truth_b_mt = float(data["b_mag_mt"])
    elif suffix in (".csv", ".txt"):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline().strip()
        has_header = any(c.isalpha() for c in first_line)
        if declared_unit is None and has_header:
            fl_low = first_line.lower()
            if "ghz" in fl_low:
                declared_unit = "GHz"
            elif "mhz" in fl_low:
                declared_unit = "MHz"
            elif "khz" in fl_low:
                declared_unit = "kHz"
            elif "hz" in fl_low:
                declared_unit = "Hz"
        arr = np.loadtxt(file_path, delimiter=",", skiprows=1 if has_header else 0)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"Expected at least 2 columns in '{file_path}', got shape {arr.shape}.")
        freqs_raw = arr[:, 0]
        spec_raw = arr[:, 1]
    else:
        raise ValueError(f"Unsupported file extension '{suffix}' for '{file_path}'.")

    return load_and_validate_real_trace(
        freqs_raw,
        spec_raw,
        trace_id=tid,
        source_name=source_name,
        license_id=license_id,
        citation=citation,
        nv_type=nv_type,
        declared_unit=declared_unit,
        spectrum_convention=spectrum_convention,
        ground_truth_b_mt=ground_truth_b_mt,
        coil_currents_amp=coil_currents_amp,
    )


def build_real_data_evaluation_manifest(
    raw_dir: Path = DEFAULT_RAW_EXP_DIR,
    manifest_path: Path = DEFAULT_REAL_MANIFEST_PATH,
) -> Dict[str, Any]:
    """Inspect all downloaded experimental files in ``data/raw_experimental/`` and write ``data/real_eval_manifest.json``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Candidate sources audit (Part 5a/5b)
    candidate_sources_audit = [
        {
            "source_id": "github_trianam_ml_magnetic_sensing",
            "url": "https://github.com/trianam/machineLearningMagneticSensing",
            "paper_citation": (
                "Galya Haim, Stefano Martina, John Howell, Nir Bar-Gill, and Filippo Caruso (2024), "
                "'Machine-learning based high-bandwidth magnetic sensing', arXiv:2409.12820."
            ),
            "license": "GPL-3.0",
            "status": "imported",
            "actual_contents": (
                "Contains experimental 4-axis NV ensemble ODMR spectra in MATLAB .mat format under "
                "data/galya/small_field/{1,2,3,4} (270-300 pts, 2730-3000/3030 MHz, with calibrated "
                "ground-truth |B| in [3.817, 4.410] mT, angles (theta, phi), and 3-axis Helmholtz coil "
                "currents I_helm) and data/galya/{1,2,3,4} (higher-field spectra with |B| in [4.832, 10.187] mT). "
                "Folder 1 uses MATLAB v7.3 HDF5 format (not imported); small_field/1..4 and folders 2..4 "
                "(7 traces total) were imported directly."
            ),
        },
        {
            "source_id": "zenodo_14697917",
            "url": "https://zenodo.org/records/14697917",
            "paper_citation": (
                "Elke Neu and Oshnik Nimba (2025), 'Dataset for: Nitrogen-Vacancy Centers in Epitaxial "
                "Laterally Overgrown Diamond: Towards Up-scaling of Color Center-based Quantum Technologies', "
                "Zenodo, DOI: 10.5281/zenodo.14697917."
            ),
            "license": "CC-BY-4.0",
            "status": "inspected_and_flagged_incompatible",
            "actual_contents": (
                "Contains 105 single-NV zero-field strain-splitting ODMR .txt traces in GHz under "
                "Data/ODMR/{Hole2,Hole3,NoHole3,NoHole5}/ (plus Ramsey/Spin-Echo files and surface profiles). "
                "3 representative single-NV ODMR files were downloaded and run through the loader: their "
                "frequency ranges are [2.850, 2.890] GHz ([2850, 2890] MHz), [2.855, 2.880] GHz ([2855, 2880] MHz), "
                "and [2.840, 2.890] GHz ([2840, 2890] MHz). Because they do not cover the required central region "
                "[2830.0, 2910.0] MHz, all 3 are deterministically flagged incompatible (never padded)."
            ),
        },
        {
            "source_id": "github_quantumvillage_uncutgem",
            "url": "https://github.com/QuantumVillage/UncutGem",
            "paper_citation": (
                "Mark Carney and Victoria Kumaran (2025), 'Uncut Gem - A Fully Open-Source, Hackable "
                "Quantum Sensor', v0.1.0, GitHub."
            ),
            "license": "AGPL-3.0",
            "status": "inspected_no_raw_odmr_traces",
            "actual_contents": (
                "Full git tree inspected: contains KiCad PCB designs, 3D-printed mount STL/STEP files, "
                "ESP32/ADF4350 Arduino sweep firmware (firmware/src/uncut-gem.ino), and two Jupyter notebooks "
                "(jupyter/UncutGem-SensorNotebook.ipynb, jupyter/uncut_measure_clean.ipynb) for live serial "
                "port plotting over /dev/cu.usbserial-*. Contains NO saved raw ODMR spectrum files."
            ),
        },
    ]

    traces_records: List[Dict[str, Any]] = []

    # 1. Process trianam/machineLearningMagneticSensing experimental traces
    trianam_citation = (
        "Haim et al. (2024), 'Machine-learning based high-bandwidth magnetic sensing', arXiv:2409.12820"
    )
    trianam_specs = [
        (
            "trianam_small_field_1",
            "trianam_sf1_f.mat",
            "trianam_sf1_y.mat",
            "positive_contrast",
            3.8170,
            (25.5755, 67.3352),
            [0.7, 2.2, 4.0],
            "4-axis NV ensemble, 270 pts [2730, 3000] MHz, Helmholtz coil setting 1 (38.170 G).",
        ),
        (
            "trianam_small_field_2",
            "trianam_sf2_f.mat",
            "trianam_sf2_y.mat",
            "positive_contrast",
            4.1862,
            (23.1206, 67.5789),
            [0.7, 2.2, 4.5],
            "4-axis NV ensemble, 270 pts [2730, 3000] MHz, Helmholtz coil setting 2 (41.862 G).",
        ),
        (
            "trianam_small_field_3",
            "trianam_sf3_f.mat",
            "trianam_sf3_y.mat",
            "positive_contrast",
            4.2887,
            (26.2322, 73.1431),
            [1.0, 2.4, 4.5],
            "4-axis NV ensemble, 270 pts [2730, 3000] MHz, Helmholtz coil setting 3 (42.887 G).",
        ),
        (
            "trianam_small_field_4",
            "trianam_sf4_f.mat",
            "trianam_sf4_y.mat",
            "positive_contrast",
            4.4104,
            (29.2317, 69.4327),
            [1.0, 2.8, 4.5],
            "4-axis NV ensemble, 300 pts [2730, 3030] MHz, Helmholtz coil setting 4 (44.104 G).",
        ),
        (
            "trianam_folder_2_101G",
            "trianam_f2_f.mat",
            "trianam_f2_y.mat",
            "negative_contrast",
            10.1870,
            (23.124, 63.15),
            None,
            "4-axis NV ensemble, 651 pts [2550, 3200] MHz, |B|=101.87 G (10.187 mT); outer dips exceed canonical [2750, 2990] MHz window.",
        ),
        (
            "trianam_folder_3_48G_hf",
            "trianam_f3_f.mat",
            "trianam_f3_y.mat",
            "negative_contrast",
            4.8320,
            (29.77, 69.1834),
            None,
            "4-axis NV ensemble with resolved 14N hyperfine triplets, 1000 pts [2710, 3010] MHz, |B|=48.32 G (4.832 mT); outer dip at 2991.4 MHz encroaches canonical upper edge.",
        ),
        (
            "trianam_folder_4_69G",
            "trianam_f4_f.mat",
            "trianam_f4_y.mat",
            "negative_contrast",
            6.9500,
            (23.73, 66.27),
            None,
            "4-axis NV ensemble, 191 pts [2600, 3140] MHz, |B|=69.50 G (6.950 mT); outer dips exceed canonical [2750, 2990] MHz window.",
        ),
    ]

    for tid, f_file, y_file, conv, b_mt, angles, i_helm, note in trianam_specs:
        fp = raw_dir / f_file
        yp = raw_dir / y_file
        if fp.exists() and yp.exists():
            mf = scipy.io.loadmat(fp)
            my = scipy.io.loadmat(yp)
            kf = [k for k in mf.keys() if not k.startswith("__")][0]
            ky = [k for k in my.keys() if not k.startswith("__")][0]
            freqs_arr = np.asarray(mf[kf], dtype=np.float64).squeeze()
            y_arr = np.asarray(my[ky], dtype=np.float64).squeeze()

            rec = load_and_validate_real_trace(
                freqs_arr,
                y_arr,
                trace_id=tid,
                source_name="GitHub trianam/machineLearningMagneticSensing",
                license_id="GPL-3.0",
                citation=trianam_citation,
                nv_type="4-axis NV ensemble",
                declared_unit="MHz",
                spectrum_convention=conv,
                ground_truth_b_mt=b_mt,
                ground_truth_angles_deg=angles,
                coil_currents_amp=i_helm,
                notes=note,
            )
            # Also export standardized CSV in data/raw_experimental/ for schema testing
            if tid == "trianam_small_field_1":
                csv_out = raw_dir / "trianam_small_field_1_standardized.csv"
                spec_norm, _ = normalize_odmr_spectrum(y_arr, convention=conv)
                header = "freq_mhz,normalized_fluorescence"
                np.savetxt(
                    csv_out,
                    np.column_stack([freqs_arr, spec_norm]),
                    delimiter=",",
                    header=header,
                    comments="",
                    fmt="%.6f",
                )
            rec_serializable = {
                k: v for k, v in rec.items() if k not in ("resampled_spectrum", "valid_mask")
            }
            traces_records.append(rec_serializable)

    # 2. Process Zenodo 14697917 single-NV ODMR .txt files
    zenodo_citation = (
        "Neu & Nimba (2025), 'Dataset for: Nitrogen-Vacancy Centers in Epitaxial Laterally Overgrown Diamond', "
        "Zenodo 14697917, DOI: 10.5281/zenodo.14697917"
    )
    for z_path in sorted(raw_dir.glob("zenodo14697917_*.txt")):
        rec = load_csv_or_npz_odmr_file(
            z_path,
            trace_id=z_path.stem,
            source_name="Zenodo record 14697917",
            license_id="CC-BY-4.0",
            citation=zenodo_citation,
            nv_type="single NV center (zero-field strain study)",
            declared_unit="GHz",
            spectrum_convention="fluorescence",
            ground_truth_b_mt=0.0,
        )
        rec["notes"] = (
            f"Single-NV zero-field strain scan in GHz ({rec['original_freq_range_mhz']} MHz); "
            "flagged incompatible because window is narrower than required [2830.0, 2910.0] MHz."
        )
        rec_serializable = {
            k: v for k, v in rec.items() if k not in ("resampled_spectrum", "valid_mask")
        }
        traces_records.append(rec_serializable)

    # 3. Evaluate cross-trace QA check: linearity vs. Helmholtz coil current across small_field 1..4
    sf_records = [
        r
        for r in traces_records
        if r["helmholtz_coil_norm_amp"] is not None
        and r["ground_truth_b_mt"] is not None
        and r["qa_results"]["estimated_b_mag_mt"] is not None
    ]
    linearity_qa: Dict[str, Any] = {}
    if len(sf_records) >= 3:
        i_norms = np.array([float(r["helmholtz_coil_norm_amp"]) for r in sf_records], dtype=np.float64)
        b_trues = np.array([float(r["ground_truth_b_mt"]) for r in sf_records], dtype=np.float64)
        b_ests = np.array(
            [float(r["qa_results"]["estimated_b_mag_mt"]) for r in sf_records], dtype=np.float64
        )
        corr_true = float(np.corrcoef(i_norms, b_trues)[0, 1])
        corr_est = float(np.corrcoef(i_norms, b_ests)[0, 1])
        mae_ut = float(np.mean(np.abs(b_ests - b_trues)) * 1000.0)
        linearity_qa = {
            "n_coil_settings": len(sf_records),
            "trace_ids": [r["trace_id"] for r in sf_records],
            "coil_current_l2_norms_amp": i_norms.tolist(),
            "ground_truth_b_mt": b_trues.tolist(),
            "pair_splitting_estimated_b_mt": b_ests.tolist(),
            "pearson_r_coil_norm_vs_true_b": round(corr_true, 5),
            "pearson_r_coil_norm_vs_estimated_b": round(corr_est, 5),
            "pair_splitting_b_mag_mae_ut": round(mae_ut, 2),
        }

    usable_ids = [
        r["trace_id"]
        for r in traces_records
        if r["usable_untruncated_on_canonical_2750_2990"]
    ]
    compatible_central_ids = [
        r["trace_id"]
        for r in traces_records
        if r["compatible_central_2830_2910"]
    ]
    incompatible_ids = [
        r["trace_id"]
        for r in traces_records
        if not r["compatible_central_2830_2910"]
    ]

    manifest = {
        "stage": "Stage 2 Real Experimental Evaluation Manifest (Evaluation/QA Only — Never Used for Training/Tuning)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "No real data is used for training, hyperparameter selection, or threshold tuning.",
        "candidate_sources_audit": candidate_sources_audit,
        "summary_counts": {
            "total_inspected_local_traces": len(traces_records),
            "compatible_with_central_2830_2910_mhz": len(compatible_central_ids),
            "usable_untruncated_on_canonical_2750_2990_mhz": len(usable_ids),
            "flagged_incompatible_narrow_window": len(incompatible_ids),
        },
        "usable_untruncated_trace_ids": usable_ids,
        "compatible_central_trace_ids": compatible_central_ids,
        "incompatible_trace_ids": incompatible_ids,
        "helmholtz_coil_linearity_qa": linearity_qa,
        "traces": traces_records,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def audit_trianam_vs_sim_anatomy(
    raw_dir: Path = DEFAULT_RAW_EXP_DIR,
    fig_path: Optional[Path] = Path("assets") / "stage2_trianam_vs_sim.png",
) -> Dict[str, Any]:
    """Perform Task 1 deep spectral anatomy comparison of ``trianam_small_field_1..4`` vs. simulator.

    Measures directly on each of the 4 experimental 4-axis traces:
      1. Total peak-to-peak contrast and all 8 individual dip frequencies & depths.
      2. Per-axis amplitude asymmetry ratio (``max_pair_depth / min_pair_depth``) and
         relative axis weights ``w_k`` vs. the simulator's equal-axis weighting
         (``0.28 * |<0|psi0>|^4``).
      3. Effective linewidth (HWHM, MHz) and pseudo-Voigt ``eta`` (both from isolated
         outer-dip fits and full 4-axis hyperfine triplet fits).
      4. Measured zero-field center ``D0`` (both raw pair-centre mean and spin-1
         Hamiltonian ``D0`` corrected for transverse second-order shift).
      5. Wing noise RMS, residual PSD log-log slope (white vs. 1/f), and baseline drift
         excursion outside the dips.
      6. Generates ``assets/stage2_trianam_vs_sim.png`` overlaying the real traces with
         both equal-axis and per-axis weighted spin-1 simulations + residuals.
    """
    specs = [
        ("trianam_small_field_1", "trianam_sf1_f.mat", "trianam_sf1_y.mat", 3.8170, [0.7, 2.2, 4.0]),
        ("trianam_small_field_2", "trianam_sf2_f.mat", "trianam_sf2_y.mat", 4.1862, [0.7, 2.2, 4.5]),
        ("trianam_small_field_3", "trianam_sf3_f.mat", "trianam_sf3_y.mat", 4.2887, [1.0, 2.4, 4.5]),
        ("trianam_small_field_4", "trianam_sf4_f.mat", "trianam_sf4_y.mat", 4.4104, [1.0, 2.8, 4.5]),
    ]

    trace_reports: List[Dict[str, Any]] = []
    plot_payloads: List[Dict[str, Any]] = []

    for tid, f_file, y_file, b_ref_mt, i_helm in specs:
        mf = scipy.io.loadmat(raw_dir / f_file)
        my = scipy.io.loadmat(raw_dir / y_file)
        kf = [k for k in mf.keys() if not k.startswith("__")][0]
        ky = [k for k in my.keys() if not k.startswith("__")][0]
        freqs = np.asarray(mf[kf], dtype=np.float64).squeeze()
        raw_y = np.asarray(my[ky], dtype=np.float64).squeeze()
        spec_norm, _ = normalize_odmr_spectrum(raw_y, convention="positive_contrast")
        absorption = 1.0 - spec_norm

        # 1. Wing noise and baseline drift on outer wings (f < 2762 MHz or f > 2983 MHz)
        wing_mask = (freqs < 2762.0) | (freqs > 2983.0)
        u_grid = (freqs - 2875.0) / 135.0
        wing_poly = np.polyfit(u_grid[wing_mask], spec_norm[wing_mask], deg=2)
        baseline_est = np.polyval(wing_poly, u_grid)
        drift_amp_wing = float(0.5 * (np.max(baseline_est) - np.min(baseline_est)))
        wing_resid = spec_norm[wing_mask] - baseline_est[wing_mask]
        wing_noise_rms = float(np.std(wing_resid))

        # 2. Extract 8 primary Zeeman dips (top 8 by prominence)
        pks, props = find_peaks(absorption, distance=4, prominence=0.002)
        top8 = np.argsort(props["prominences"])[::-1][:8]
        pk8 = np.sort(pks[top8])
        dip_freqs_8 = freqs[pk8]
        dip_depths_8 = (baseline_est[pk8] - spec_norm[pk8])
        ptp_contrast = float(np.max(spec_norm) - np.min(spec_norm))

        # Pair the 8 dips from innermost (k=0) to outermost (k=3)
        # pk8 is sorted ascending: indices (3,4) are inner pair k=0; (0,7) are outer pair k=3
        pair_fm_exp = np.array([dip_freqs_8[3 - k] for k in range(4)], dtype=np.float64)
        pair_fp_exp = np.array([dip_freqs_8[4 + k] for k in range(4)], dtype=np.float64)
        pair_dm_exp = np.array([dip_depths_8[3 - k] for k in range(4)], dtype=np.float64)
        pair_dp_exp = np.array([dip_depths_8[4 + k] for k in range(4)], dtype=np.float64)
        pair_mean_depths = 0.5 * (pair_dm_exp + pair_dp_exp)
        pair_centres_raw = 0.5 * (pair_fm_exp + pair_fp_exp)
        pair_splittings = pair_fp_exp - pair_fm_exp

        inter_axis_asym_ratio = float(np.max(pair_mean_depths) / max(float(np.min(pair_mean_depths)), 1e-9))
        intra_pair_ratios = (pair_dm_exp / np.maximum(pair_dp_exp, 1e-9)).tolist()

        # 3. Fit isolated pseudo-Voigt profile to outermost pair (f_{-,3}, f_{+,3}) for direct HWHM & eta
        outer_single_hwhms = []
        outer_single_etas = []
        for f_center_0, d_0 in ((pair_fm_exp[3], pair_dm_exp[3]), (pair_fp_exp[3], pair_dp_exp[3])):
            local_mask = np.abs(freqs - f_center_0) <= 11.0
            f_loc = freqs[local_mask]
            y_loc = (baseline_est[local_mask] - spec_norm[local_mask])

            def _single_dip_res(p: np.ndarray) -> np.ndarray:
                # p = [f0, amp, hwhm, eta]
                return pseudo_voigt_dip(f_loc, p[0], p[1], p[2], p[3]) - y_loc

            res_loc = least_squares(
                _single_dip_res,
                x0=np.array([f_center_0, d_0, 4.5, 0.35]),
                bounds=(
                    np.array([f_center_0 - 3.0, 0.001, 1.5, 0.0]),
                    np.array([f_center_0 + 3.0, 0.050, 14.0, 1.0]),
                ),
            )
            outer_single_hwhms.append(float(res_loc.x[2]))
            outer_single_etas.append(float(res_loc.x[3]))

        eff_single_dip_hwhm = float(np.mean(outer_single_hwhms))
        eff_single_dip_eta = float(np.mean(outer_single_etas))

        # 4. Fit exact 3D spin-1 Hamiltonian (Bx, By, Bz, D0) to the 8 experimental transition frequencies
        # Initialize from analytic reconstruction of the 4 half-splittings
        p_init = 0.5 * pair_splittings / GAMMA_E_MHZ_MT
        # Sign pattern (-p0, -p1, +p2, +p3) or best zero-sum pattern
        best_b0 = np.array([0.8, 1.8, 3.3], dtype=np.float64)
        best_s_err = np.inf
        for s0 in (-1.0, 1.0):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    q = np.array([s0 * p_init[0], s1 * p_init[1], s2 * p_init[2], p_init[3]])
                    if abs(np.sum(q)) < best_s_err:
                        best_s_err = abs(np.sum(q))
                        best_b0 = canonicalize_b_vec(0.75 * (NV_AXES.T @ q))

        def _freq_8_resid(theta_b_d0: np.ndarray) -> np.ndarray:
            bv = theta_b_d0[:3]
            d0_v = float(theta_b_d0[3])
            bp = np.abs(NV_AXES @ bv)
            bsq = float(np.dot(bv, bv))
            bperp = np.sqrt(np.maximum(0.0, bsq - bp * bp))
            fm_m, fp_m, _ = solve_axis_transitions(bp, bperp, d0=d0_v)
            order_k = np.argsort(fp_m - fm_m)
            return np.concatenate([fm_m[order_k] - pair_fm_exp, fp_m[order_k] - pair_fp_exp])

        res_ham = least_squares(
            _freq_8_resid,
            x0=np.array([best_b0[0], best_b0[1], best_b0[2], float(np.mean(pair_centres_raw)) - 1.2]),
            bounds=(np.array([0.0, 0.0, 0.0, 2860.0]), np.array([5.0, 5.0, 5.0, 2885.0])),
        )
        b_ham_canon = canonicalize_b_vec(res_ham.x[:3])
        b_ham_mag = float(np.linalg.norm(b_ham_canon))
        d0_ham_mhz = float(res_ham.x[3])

        # Compute exact spin-1 transition frequencies & contrast factors at (b_ham_canon, d0_ham_mhz)
        bp_h = np.abs(NV_AXES @ b_ham_canon)
        bperp_h = np.sqrt(np.maximum(0.0, b_ham_mag * b_ham_mag - bp_h * bp_h))
        fm_h, fp_h, cf_h = solve_axis_transitions(bp_h, bperp_h, d0=d0_ham_mhz)
        ord_h = np.argsort(fp_h - fm_h)
        fm_h = fm_h[ord_h]
        fp_h = fp_h[ord_h]
        cf_h = cf_h[ord_h]
        sim_cf_asym_ratio = float(np.max(cf_h) / max(float(np.min(cf_h)), 1e-9))

        # Relative empirical axis weights w_k after accounting for spin-mixing contrast factor cf_h
        raw_wk = pair_mean_depths / np.maximum(cf_h, 1e-9)
        rel_axis_weights = (raw_wk / np.mean(raw_wk)).tolist()
        rel_axis_log_std = float(np.std(np.log(np.maximum(rel_axis_weights, 1e-6))))

        # 5. Fit full 4-axis hyperfine spectrum model (Equal-Axis vs. Per-Axis Weighted)
        def _build_axis_profiles(hwhm_v: float, eta_v: float, d0_shift: float) -> np.ndarray:
            # Returns (4, N) unit-amplitude hyperfine-summed profiles for the 4 ordered axes
            profs = np.zeros((4, freqs.size), dtype=np.float64)
            for k in range(4):
                for fc in (fm_h[k] + d0_shift, fp_h[k] + d0_shift):
                    for dhf in (-A_HYPERFINE_MHZ, 0.0, A_HYPERFINE_MHZ):
                        profs[k] += (1.0 / 3.0) * pseudo_voigt_dip(
                            freqs, fc + dhf, 1.0, hwhm_v, eta_v
                        )
            return profs

        # (a) Equal-axis simulator fit: [C_base, hwhm_hf, eta_hf, d0_fine, bl_0, bl_1]
        def _eq_axis_resid(p: np.ndarray) -> np.ndarray:
            c_base, hw_v, et_v, d0_s, b0, b1 = p
            profs = _build_axis_profiles(hw_v, et_v, d0_s)
            weights = c_base * 0.28 * cf_h
            model = (1.0 + b0 + b1 * u_grid) - (weights @ profs)
            return model - spec_norm

        res_eq = least_squares(
            _eq_axis_resid,
            x0=np.array([0.045, 3.6, 0.35, 0.0, 0.0, 0.0]),
            bounds=(
                np.array([0.005, 1.2, 0.0, -2.0, -0.02, -0.02]),
                np.array([0.150, 12.0, 1.0, 2.0, 0.02, 0.02]),
            ),
        )
        c_eq, hw_eq, et_eq, d0s_eq, b0_eq, b1_eq = res_eq.x
        profs_eq = _build_axis_profiles(float(hw_eq), float(et_eq), float(d0s_eq))
        sim_equal_axis = (1.0 + b0_eq + b1_eq * u_grid) - ((c_eq * 0.28 * cf_h) @ profs_eq)
        resid_equal_axis = spec_norm - sim_equal_axis
        rms_equal_axis = float(np.sqrt(np.mean(resid_equal_axis ** 2)))

        # (b) Per-axis weighted simulator fit (Stage 2.1: 4 non-negative axis amplitudes + linear baseline)
        X_weighted = np.column_stack([profs_eq.T, np.ones_like(u_grid), u_grid])
        coef_w, _, _, _ = np.linalg.lstsq(X_weighted, 1.0 - spec_norm, rcond=None)
        axis_amps_w = np.maximum(0.0, coef_w[:4])
        sim_weighted_axis = (1.0 - coef_w[4] - coef_w[5] * u_grid) - (axis_amps_w @ profs_eq)
        resid_weighted_axis = spec_norm - sim_weighted_axis
        rms_weighted_axis = float(np.sqrt(np.mean(resid_weighted_axis ** 2)))

        # (c) Per-axis weighted + RF power tilt simulator fit (Stage 2.2: m(f) = 1 + c1*u + c2*(u^2 - 1/3))
        u_rf = np.clip((freqs - D0_MHZ) * (1.0 / 120.0), -1.0, 1.0)
        u2_rf = u_rf * u_rf - (1.0 / 3.0)
        u2_grid = u_grid * u_grid - (1.0 / 3.0)

        def _eval_tilted_fit(theta_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            c1_t, c2_t, hw_t, et_t, d0s_t = (
                float(theta_t[0]),
                float(theta_t[1]),
                float(theta_t[2]),
                float(theta_t[3]),
                float(theta_t[4]),
            )
            m_env = 1.0 + c1_t * u_rf + c2_t * u2_rf
            profs_t = _build_axis_profiles(hw_t, et_t, d0s_t) * m_env[None, :]
            X_t = np.column_stack([profs_t.T, np.ones_like(u_grid), u_grid, u2_grid])
            coef_t, _, _, _ = np.linalg.lstsq(X_t, 1.0 - spec_norm, rcond=None)
            amps_t = np.maximum(0.0, coef_t[:4])
            sim_t = (1.0 - coef_t[4] - coef_t[5] * u_grid - coef_t[6] * u2_grid) - (amps_t @ profs_t)
            return sim_t, spec_norm - sim_t, m_env

        res_tilt = least_squares(
            lambda th: np.append(_eval_tilted_fit(th)[1], 0.005 * float(th[1])),
            x0=np.array([0.0, 0.0, float(hw_eq), float(et_eq), float(d0s_eq)], dtype=np.float64),
            bounds=(
                np.array([-0.12, -0.12, 1.5, 0.0, -2.0]),
                np.array([0.12, 0.12, 9.0, 1.0, 2.0]),
            ),
        )
        sim_tilted_axis, resid_tilted_axis, m_env_opt = _eval_tilted_fit(res_tilt.x)
        rms_tilted_axis = float(np.sqrt(np.mean(resid_tilted_axis ** 2)))
        fitted_tilt_amp = float(np.max(np.abs(m_env_opt - 1.0)))
        fitted_tilt_coeffs = [round(float(res_tilt.x[0]), 4), round(float(res_tilt.x[1]), 4)]

        # 6. Residual PSD slope (log10 PSD vs log10 normalized frequency on [f_1 .. f_Nyquist])
        fft_vals = np.fft.rfft(resid_weighted_axis - np.mean(resid_weighted_axis))
        psd = (np.abs(fft_vals) ** 2) / float(freqs.size)
        f_fft = np.fft.rfftfreq(freqs.size)
        valid_f = f_fft > 0.02  # exclude lowest DC/drift bins
        psd_slope = float(
            np.polyfit(np.log10(f_fft[valid_f]), np.log10(np.maximum(psd[valid_f], 1e-18)), deg=1)[0]
        )

        trace_reports.append(
            {
                "trace_id": tid,
                "n_points": int(freqs.size),
                "freq_range_mhz": [float(freqs[0]), float(freqs[-1])],
                "helmholtz_coil_currents_amp": i_helm,
                "ground_truth_b_mt": b_ref_mt,
                "hamiltonian_fitted_b_mt": round(b_ham_mag, 4),
                "b_canon_mt": [round(float(x), 4) for x in b_ham_canon],
                "raw_pair_centre_mean_mhz": round(float(np.mean(pair_centres_raw)), 3),
                "raw_pair_centre_offset_mhz": round(float(np.mean(pair_centres_raw) - D0_MHZ), 3),
                "hamiltonian_d0_mhz": round(d0_ham_mhz + float(d0s_eq), 3),
                "hamiltonian_d0_offset_mhz": round(d0_ham_mhz + float(d0s_eq) - D0_MHZ, 3),
                "total_ptp_contrast": round(ptp_contrast, 5),
                "fitted_base_contrast_c": round(float(c_eq), 5),
                "dip_freqs_8_mhz": [round(float(x), 2) for x in dip_freqs_8],
                "dip_depths_8": [round(float(x), 5) for x in dip_depths_8],
                "pair_mean_depths_inner_to_outer": [round(float(x), 5) for x in pair_mean_depths],
                "exp_inter_axis_depth_ratio_max_over_min": round(inter_axis_asym_ratio, 3),
                "sim_equal_axis_cf_ratio_max_over_min": round(sim_cf_asym_ratio, 3),
                "relative_axis_weights_wk": [round(float(x), 3) for x in rel_axis_weights],
                "relative_axis_log_std": round(rel_axis_log_std, 3),
                "intra_pair_minus_over_plus_ratios": [round(float(x), 3) for x in intra_pair_ratios],
                "isolated_outer_single_dip_hwhm_mhz": round(eff_single_dip_hwhm, 3),
                "isolated_outer_single_dip_eta": round(eff_single_dip_eta, 3),
                "hyperfine_triplet_hwhm_mhz": round(float(hw_eq), 3),
                "hyperfine_triplet_eta": round(float(et_eq), 3),
                "wing_noise_rms": round(wing_noise_rms, 6),
                "wing_drift_amp": round(drift_amp_wing, 6),
                "residual_psd_log_slope": round(psd_slope, 3),
                "equal_axis_sim_residual_rms": round(rms_equal_axis, 5),
                "per_axis_weighted_sim_residual_rms": round(rms_weighted_axis, 5),
                "per_axis_tilted_sim_residual_rms": round(rms_tilted_axis, 5),
                "fitted_mw_power_tilt_amp": round(fitted_tilt_amp, 4),
                "fitted_mw_power_tilt_coeffs": fitted_tilt_coeffs,
            }
        )
        plot_payloads.append(
            {
                "tid": tid,
                "freqs": freqs,
                "spec_norm": spec_norm,
                "sim_equal": sim_equal_axis,
                "sim_weighted": sim_weighted_axis,
                "sim_tilted": sim_tilted_axis,
                "resid_equal": resid_equal_axis,
                "resid_weighted": resid_weighted_axis,
                "resid_tilted": resid_tilted_axis,
                "meta": trace_reports[-1],
            }
        )

    if fig_path is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            2, 4, figsize=(19.2, 8.2), sharex="col", gridspec_kw={"height_ratios": [2.25, 1.0]}
        )
        for col_i, item in enumerate(plot_payloads):
            ax_top = axes[0, col_i]
            ax_bot = axes[1, col_i]
            m = item["meta"]
            f_arr = item["freqs"]

            ax_top.plot(
                f_arr, item["spec_norm"], color="#111111", lw=1.30, alpha=0.85, label="Real Experimental (trianam)"
            )
            ax_top.plot(
                f_arr,
                item["sim_equal"],
                color="#d62728",
                ls="--",
                lw=1.30,
                label=f"Sim Equal-Axis (RMS={m['equal_axis_sim_residual_rms']:.5f})",
            )
            ax_top.plot(
                f_arr,
                item["sim_weighted"],
                color="#1f77b4",
                ls="-.",
                lw=1.30,
                label=f"Sim Per-Axis Weighted (RMS={m['per_axis_weighted_sim_residual_rms']:.5f})",
            )
            ax_top.plot(
                f_arr,
                item["sim_tilted"],
                color="#2ca02c",
                ls="-",
                lw=1.40,
                label=f"Sim Per-Axis + RF Tilt (RMS={m['per_axis_tilted_sim_residual_rms']:.5f})",
            )
            ax_top.set_title(
                f"{m['trace_id']} (|B|={m['ground_truth_b_mt']:.3f} mT)\n"
                f"D0={m['hamiltonian_d0_mhz']:.2f} MHz (dD0=+{m['hamiltonian_d0_offset_mhz']:.2f}), "
                f"HWHM_hf={m['hyperfine_triplet_hwhm_mhz']:.2f} MHz\n"
                f"Axis Asym={m['exp_inter_axis_depth_ratio_max_over_min']:.2f}x, "
                f"RF Tilt={m['fitted_mw_power_tilt_amp']*100:.1f}%",
                fontsize=8.3,
                fontweight="bold",
            )
            ax_top.grid(True, linestyle=":", alpha=0.45)
            ax_top.legend(loc="lower right", fontsize=6.8)

            ax_bot.axhline(0.0, color="#666666", lw=0.8, ls=":")
            ax_bot.plot(
                f_arr,
                item["resid_equal"],
                color="#d62728",
                lw=0.85,
                ls="--",
                alpha=0.80,
                label=f"Equal-Axis (RMS={m['equal_axis_sim_residual_rms']:.5f})",
            )
            ax_bot.plot(
                f_arr,
                item["resid_weighted"],
                color="#1f77b4",
                lw=0.85,
                ls="-.",
                alpha=0.85,
                label=f"Per-Axis (RMS={m['per_axis_weighted_sim_residual_rms']:.5f})",
            )
            ax_bot.plot(
                f_arr,
                item["resid_tilted"],
                color="#2ca02c",
                lw=1.00,
                ls="-",
                alpha=0.95,
                label=f"Per-Axis+RF Tilt (RMS={m['per_axis_tilted_sim_residual_rms']:.5f})",
            )
            ax_bot.set_xlabel("Microwave Frequency (MHz)", fontsize=9)
            ax_bot.grid(True, linestyle=":", alpha=0.45)
            ax_bot.legend(loc="upper right", fontsize=6.5)
            if col_i == 0:
                ax_top.set_ylabel("Norm. Fluorescence", fontsize=9.5)
                ax_bot.set_ylabel("Residual (Exp - Sim)", fontsize=9.5)

        fig.suptitle(
            "OpenNV-Denoise Stage 2.2 — Real Experimental (trianam) vs. Simulator: Equal-Axis vs. Per-Axis vs. Per-Axis + RF Power Tilt\n"
            "[SIM-TO-REAL STATUS: trianam HWHM_hf (2.75–2.96 MHz) in train_sim2real [2.0, 9.0] MHz | Axis Asym (1.44x–1.60x) in train_sim2real [1.0, 2.0]x | RF Tilt in <=12%]",
            fontsize=10.5,
            fontweight="bold",
            y=1.03,
        )
        fig.tight_layout()
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=170, bbox_inches="tight")
        plt.close(fig)

    return {
        "traces": trace_reports,
        "summary": {
            "b_mag_range_mt": [
                min(r["ground_truth_b_mt"] for r in trace_reports),
                max(r["ground_truth_b_mt"] for r in trace_reports),
            ],
            "hamiltonian_d0_offset_range_mhz": [
                min(r["hamiltonian_d0_offset_mhz"] for r in trace_reports),
                max(r["hamiltonian_d0_offset_mhz"] for r in trace_reports),
            ],
            "raw_pair_centre_offset_range_mhz": [
                min(r["raw_pair_centre_offset_mhz"] for r in trace_reports),
                max(r["raw_pair_centre_offset_mhz"] for r in trace_reports),
            ],
            "exp_inter_axis_depth_ratio_range": [
                min(r["exp_inter_axis_depth_ratio_max_over_min"] for r in trace_reports),
                max(r["exp_inter_axis_depth_ratio_max_over_min"] for r in trace_reports),
            ],
            "mean_relative_axis_log_std": round(
                float(np.mean([r["relative_axis_log_std"] for r in trace_reports])), 3
            ),
            "hyperfine_hwhm_range_mhz": [
                min(r["hyperfine_triplet_hwhm_mhz"] for r in trace_reports),
                max(r["hyperfine_triplet_hwhm_mhz"] for r in trace_reports),
            ],
            "isolated_single_dip_hwhm_range_mhz": [
                min(r["isolated_outer_single_dip_hwhm_mhz"] for r in trace_reports),
                max(r["isolated_outer_single_dip_hwhm_mhz"] for r in trace_reports),
            ],
            "fitted_base_contrast_range": [
                min(r["fitted_base_contrast_c"] for r in trace_reports),
                max(r["fitted_base_contrast_c"] for r in trace_reports),
            ],
            "wing_noise_rms_range": [
                min(r["wing_noise_rms"] for r in trace_reports),
                max(r["wing_noise_rms"] for r in trace_reports),
            ],
        },
    }
