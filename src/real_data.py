"""Stage 2 experimental ODMR data importer, unit normalizer, PCHIP validator, and QA manifest builder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.io
from scipy.signal import find_peaks

from src.config import (
    CANONICAL_FREQS,
    CENTRAL_REQUIRED_MAX_MHZ,
    CENTRAL_REQUIRED_MIN_MHZ,
    D0_MHZ,
    FREQ_MAX_MHZ,
    FREQ_MIN_MHZ,
    GAMMA_E_MHZ_MT,
)
from src.noise import resample_spectrum_pchip
from src.physics import b_mag_from_splittings

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
