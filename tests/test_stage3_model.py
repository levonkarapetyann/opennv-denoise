"""Deterministic unit & integrity test suite for Stage 3 model, physics layer, hybrid, and leakage checks."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pytest
import torch

from src.config import CANONICAL_FREQS, D0_MHZ, N_POINTS
from src.dataset_gen import TEST_ACCESS_LOG_PATH, get_test_access_counts
from src.integrity import verify_frozen_hashes
from src.losses import LossWeights, OpenNVMultiTaskLoss
from src.model_cnn import (
    ModelConfig,
    OpenNVResUNet1D,
    b_mag_from_splittings_torch,
    canonical_b_vec_from_splittings,
    compute_tetra_residual,
    invert_splittings_order2_torch,
)
from src.physics import ODMRParams, b_mag_from_splittings, canonicalize_b_vec, simulate_clean_odmr


def test_stage3_model_shapes_monotonicity_masking_and_finiteness() -> None:
    """Verify output shapes, df_1 <= df_2 <= df_3 <= df_4, masked dip_hat, and finiteness on cropped/spiked inputs."""
    torch.manual_seed(42)
    for unc_mode in ("quantile", "laplace", "student_t"):
        cfg = ModelConfig(base_channels=16, use_ch3=True, uncertainty_mode=unc_mode)
        model = OpenNVResUNet1D(cfg)
        model.eval()

        # Construct realistic + adversarial inputs: normal, cropped (valid_mask edges 0), spiked (+50/-50), flat
        x = torch.ones((4, N_POINTS), dtype=torch.float32)
        x[0] -= 0.015 * torch.exp(-((torch.linspace(-1, 1, N_POINTS)) / 0.1) ** 2)
        x[1] += 0.005 * torch.randn(N_POINTS)
        x[2, 100] += 50.0  # extreme spike
        x[2, 300] -= 50.0
        x[3] = 1.0  # completely flat

        vm = torch.ones((4, N_POINTS), dtype=torch.bool)
        vm[1, :80] = False
        vm[1, -80:] = False
        vm[2, :50] = False

        out = model(x, vm)
        assert out["y_clean_hat"].shape == (4, N_POINTS)
        assert out["dip_hat"].shape == (4, N_POINTS)
        assert out["dip_hat_masked"].shape == (4, N_POINTS)
        assert out["splittings_hat"].shape == (4, 4)
        assert out["b_mag_phys"].shape == (4,)
        assert out["b_vec_canon"].shape == (4, 3)
        assert out["q10"].shape == (4,) and out["q50"].shape == (4,) and out["q90"].shape == (4,)
        assert out["tetra_residual"].shape == (4,)

        # All outputs strictly finite
        for k, v in out.items():
            assert bool(torch.all(torch.isfinite(v))), f"Non-finite output in {k} ({unc_mode})"

        # Strict monotonicity 0 <= df_1 <= df_2 <= df_3 <= df_4
        sp = out["splittings_hat"]
        assert bool(torch.all(sp >= 0.0))
        assert bool(torch.all(sp[:, 1:] >= sp[:, :-1] - 1e-6))

        # No quantile crossing: q10 < q50 < q90
        assert bool(torch.all(out["q10"] < out["q50"]))
        assert bool(torch.all(out["q50"] < out["q90"]))

        # Masked dip_hat must be strictly zero outside valid_mask
        invalid_entries = out["dip_hat_masked"][~vm]
        assert bool(torch.all(invalid_entries == 0.0))

        # Eval-mode determinism
        out2 = model(x, vm)
        for k in ("y_clean_hat", "splittings_hat", "b_mag_phys", "q10", "q90"):
            assert torch.equal(out[k], out2[k]), f"Non-deterministic eval output in {k}"


def test_stage3_physics_layer_float64_parity_and_gradients() -> None:
    """Verify <1e-3 uT float64 parity vs src.physics order=2 and finite non-zero gradients at edge cases."""
    rng = np.random.default_rng(20263001)
    max_err_f64_ut = 0.0
    max_err_f32_ut = 0.0

    for _ in range(100):
        b_mag_true = float(rng.uniform(0.05, 3.20))
        u = rng.standard_normal(3)
        u /= np.linalg.norm(u)
        bv = b_mag_true * u
        d0_val = float(D0_MHZ + rng.uniform(-3.0, 3.0))
        sim = simulate_clean_odmr(
            CANONICAL_FREQS,
            ODMRParams(b_field_mt=(float(bv[0]), float(bv[1]), float(bv[2])), d0_mhz=d0_val),
        )
        sp_np = sim["splittings_mhz"].astype(np.float64)
        b_ref = float(b_mag_from_splittings(sp_np, order=2, d0_mhz=d0_val))

        sp_t64 = torch.tensor(sp_np, dtype=torch.float64)
        b_t64 = float(b_mag_from_splittings_torch(sp_t64, d0_mhz=d0_val).item())
        sp_t32 = torch.tensor(sp_np, dtype=torch.float32)
        b_t32 = float(b_mag_from_splittings_torch(sp_t32, d0_mhz=d0_val).item())

        max_err_f64_ut = max(max_err_f64_ut, abs(b_t64 - b_ref) * 1000.0)
        max_err_f32_ut = max(max_err_f32_ut, abs(b_t32 - b_ref) * 1000.0)

    assert max_err_f64_ut < 1e-3, f"float64 parity error {max_err_f64_ut:.6f} uT >= 1e-3 uT"
    assert max_err_f32_ut < 5.0, f"float32 parity error {max_err_f32_ut:.6f} uT unexpectedly large"

    # Gradient check at edge cases: splitting -> 0, |B| -> 0.05 mT, |B| -> 3.20 mT, degenerate axes
    edge_splittings = torch.tensor(
        [
            [0.0, 1e-4, 2e-4, 3e-4],          # splitting -> 0
            [1.6, 1.6, 1.6, 1.6],             # |B| ~= 0.05 mT along <100>
            [0.0, 20.0, 40.0, 60.0],          # degenerate axis (p1 = 0)
            [59.8, 59.8, 59.8, 179.4],        # |B| ~= 3.20 mT along [111]
            [0.05, 65.0, 65.0, 130.0],        # high field + degenerate axis
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    d0_t = torch.full((5,), D0_MHZ, dtype=torch.float32, requires_grad=True)
    b_out = b_mag_from_splittings_torch(edge_splittings, d0_mhz=d0_t)
    loss = b_out.sum()
    loss.backward()

    assert edge_splittings.grad is not None and d0_t.grad is not None
    assert bool(torch.all(torch.isfinite(edge_splittings.grad)))
    assert bool(torch.all(torch.isfinite(d0_t.grad)))
    # At least one non-zero gradient per sample row
    row_norms = torch.linalg.norm(edge_splittings.grad, dim=-1)
    assert bool(torch.all(row_norms > 1e-6)), f"Vanishing gradients at edge cases: {row_norms}"


def test_stage3_canonical_b_vec_inversion_and_tetra_residual() -> None:
    """Verify canonical_b_vec_from_splittings (<1 uT on resolved axes) and tetra_residual == 0 on exact eigh."""
    rng = np.random.default_rng(20263002)
    resolved_errs_ut = []
    tetra_res_ut = []

    for _ in range(100):
        b_mag_true = float(rng.uniform(0.30, 3.20))
        u = rng.standard_normal(3)
        u /= np.linalg.norm(u)
        bv = b_mag_true * u
        b_canon_true = canonicalize_b_vec(bv)
        sim = simulate_clean_odmr(
            CANONICAL_FREQS,
            ODMRParams(b_field_mt=(float(bv[0]), float(bv[1]), float(bv[2])), d0_mhz=D0_MHZ),
        )
        sp_t64 = torch.tensor(sim["splittings_mhz"], dtype=torch.float64)
        _, p_mt = invert_splittings_order2_torch(sp_t64, d0_mhz=D0_MHZ)
        b_vec_inv = canonical_b_vec_from_splittings(sp_t64, d0_mhz=D0_MHZ, p_long_mt=p_mt).numpy()
        t_res = float(compute_tetra_residual(sp_t64, d0_mhz=D0_MHZ, p_long_mt=p_mt).item()) * 1000.0
        tetra_res_ut.append(t_res)

        if float(np.min(sim["splittings_mhz"])) > 5.0:  # all 4 axes resolved
            err_ut = float(np.linalg.norm(b_vec_inv - b_canon_true) * 1000.0)
            resolved_errs_ut.append(err_ut)

    assert len(resolved_errs_ut) > 50
    assert float(np.median(resolved_errs_ut)) < 1.0, f"Median resolved B_vec inversion error {np.median(resolved_errs_ut):.3f} uT >= 1.0 uT"
    assert float(np.max(tetra_res_ut)) < 1.0, f"Max exact eigh tetra_residual {max(tetra_res_ut):.4f} uT >= 1.0 uT"


def test_stage3_loss_backward_and_frozen_integrity_and_no_leakage() -> None:
    """Verify loss backward pass, frozen hash integrity, 0 test reads, and static zero-leakage grep on src/train.py."""
    # 1. Loss backward pass
    cfg = ModelConfig(base_channels=16, use_ch3=True, uncertainty_mode="quantile")
    model = OpenNVResUNet1D(cfg)
    criterion = OpenNVMultiTaskLoss(weights=LossWeights(w_tetra=0.05), uncertainty_mode="quantile")
    bsz = 4
    batch = {
        "X_noisy": torch.ones((bsz, N_POINTS), dtype=torch.float32) + 0.005 * torch.randn(bsz, N_POINTS),
        "valid_mask": torch.ones((bsz, N_POINTS), dtype=torch.bool),
        "Y_clean": torch.ones((bsz, N_POINTS), dtype=torch.float32) - 0.01 * torch.exp(-torch.linspace(-2, 2, N_POINTS) ** 2),
        "Y_splittings": torch.tensor([[10.0, 20.0, 30.0, 40.0]] * bsz, dtype=torch.float32),
        "Y_b_mag": torch.full((bsz,), 0.85, dtype=torch.float32),
        "Y_d0": torch.zeros((bsz,), dtype=torch.float32),
        "Y_hwhm": torch.full((bsz,), 4.5, dtype=torch.float32),
    }
    pred = model(batch["X_noisy"], batch["valid_mask"])
    loss, terms = criterion(pred, batch)
    assert torch.isfinite(loss)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert bool(torch.all(torch.isfinite(p.grad)))

    # 2. Frozen asset SHA256 integrity (no git diff in pytest)
    ver = verify_frozen_hashes(root_dir=ROOT_DIR)
    assert ver["ok"], f"Frozen hash verification failed: {ver['mismatches']}"

    # 3. Locked test files MUST have 0 data reads
    counts = get_test_access_counts(ROOT_DIR / TEST_ACCESS_LOG_PATH)
    assert counts["data_read"] == 0, f"Test access lock violation: {counts}"

    # 4. Static grep leakage check on src/train.py
    train_src = (ROOT_DIR / "src" / "train.py").read_text(encoding="utf-8")
    forbidden_patterns = [r"\bval_ood\b", r"\bood_[a-zA-Z0-9_]*", r"\btest_id\b", r"\btest_ood\b"]
    for pat in forbidden_patterns:
        matches = re.findall(pat, train_src)
        assert len(matches) == 0, f"Leakage violation in src/train.py matching {pat}: {matches}"
