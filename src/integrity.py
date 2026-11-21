"""Frozen-asset SHA256 hash manifest builder and verification helper (Stage 3 Part 0c)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.dataset_gen import (
    DEFAULT_SYNTHETIC_DIR,
    LOCKED_TEST_SPLITS,
    TEST_ACCESS_LOG_PATH,
    compute_file_sha256,
)

DEFAULT_FROZEN_HASHES_PATH: Path = Path("data") / "frozen_hashes.json"

FROZEN_SOURCE_FILES: List[str] = [
    "src/baseline.py",
    "src/physics.py",
    "src/noise.py",
    "tests/test_physics.py",
]


def build_frozen_hashes_manifest(
    root_dir: Path = Path("."),
    out_path: Path = DEFAULT_FROZEN_HASHES_PATH,
    log_path: Path = TEST_ACCESS_LOG_PATH,
) -> Dict[str, Any]:
    """Compute SHA256 digests of frozen source files and Stage 2/2.1/2.2 synthetic `.npz` archives.

    Reads raw bytes only (never opens or parses `.npz` contents). Locked test files
    (`test_id.npz`, `test_ood.npz`) are logged as ``hash_check``, keeping ``data_read == 0``.
    """
    source_hashes: Dict[str, str] = {}
    for rel in FROZEN_SOURCE_FILES:
        fpath = root_dir / rel
        if not fpath.exists():
            raise FileNotFoundError(f"Frozen source file missing: {fpath}")
        source_hashes[rel] = compute_file_sha256(fpath, log_as_hash_check=False, log_path=log_path)

    synthetic_dir = root_dir / DEFAULT_SYNTHETIC_DIR
    archive_hashes: Dict[str, Dict[str, Any]] = {}
    for npz_path in sorted(synthetic_dir.glob("*.npz")):
        rel = npz_path.relative_to(root_dir).as_posix()
        sname = npz_path.stem
        is_locked = sname in LOCKED_TEST_SPLITS
        digest = compute_file_sha256(
            npz_path,
            split_name=sname,
            log_as_hash_check=is_locked,
            log_path=log_path,
        )
        archive_hashes[rel] = {
            "sha256": digest,
            "size_bytes": int(npz_path.stat().st_size),
            "locked_test": bool(is_locked),
        }

    payload: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_source_files": source_hashes,
        "frozen_synthetic_archives": archive_hashes,
    }
    full_out = root_dir / out_path if not out_path.is_absolute() else out_path
    full_out.parent.mkdir(parents=True, exist_ok=True)
    with open(full_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def verify_frozen_hashes(
    manifest_path: Path = DEFAULT_FROZEN_HASHES_PATH,
    root_dir: Path = Path("."),
    log_path: Path = TEST_ACCESS_LOG_PATH,
) -> Dict[str, Any]:
    """Verify every file recorded in ``data/frozen_hashes.json`` against its current SHA256 digest.

    Returns a summary dictionary with ``ok: bool`` and ``mismatches: List[str]``.
    Raises ``AssertionError`` only if callers assert ``result["ok"]``.
    """
    full_manifest = root_dir / manifest_path if not manifest_path.is_absolute() else manifest_path
    if not full_manifest.exists():
        return {"ok": False, "checked": 0, "mismatches": [f"Missing manifest: {full_manifest}"]}

    with open(full_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    mismatches: List[str] = []
    checked = 0

    for rel, expected_sha in manifest.get("frozen_source_files", {}).items():
        fpath = root_dir / rel
        checked += 1
        if not fpath.exists():
            mismatches.append(f"MISSING source file: {rel}")
            continue
        actual_sha = compute_file_sha256(fpath, log_as_hash_check=False, log_path=log_path)
        if actual_sha != expected_sha:
            mismatches.append(f"HASH MISMATCH source {rel}: expected {expected_sha}, got {actual_sha}")

    for rel, entry in manifest.get("frozen_synthetic_archives", {}).items():
        fpath = root_dir / rel
        checked += 1
        if not fpath.exists():
            mismatches.append(f"MISSING archive: {rel}")
            continue
        sname = fpath.stem
        is_locked = bool(entry.get("locked_test", sname in LOCKED_TEST_SPLITS))
        actual_sha = compute_file_sha256(
            fpath,
            split_name=sname,
            log_as_hash_check=is_locked,
            log_path=log_path,
        )
        expected_sha = str(entry["sha256"])
        if actual_sha != expected_sha:
            mismatches.append(f"HASH MISMATCH archive {rel}: expected {expected_sha}, got {actual_sha}")

    return {
        "ok": len(mismatches) == 0,
        "checked": checked,
        "mismatches": mismatches,
    }
