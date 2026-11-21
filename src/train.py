"""Stage 3 training pipeline for OpenNV-Denoise (Runs 3a & 3b).

Strictly uses on-the-fly synthetic training streams (with disjoint master seeds)
and the held-out selection subset ``val_sel`` (1,000 spectra of ``val_id.npz``
disjoint from the 3,000 reporting indices in ``val_baseline_ref.npz``).
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.config import B_MAG_MAX_MT, B_MAG_MIN_MT, D0_MHZ, N_POINTS
from src.dataset_gen import (
    DEFAULT_SYNTHETIC_DIR,
    FACTOR_ID_STANDARD,
    SPLIT_REGISTRY,
    SplitConfig,
    generate_split_arrays,
    load_synthetic_split,
)
from src.losses import LossWeights, OpenNVMultiTaskLoss, compute_per_term_grad_norms
from src.model_cnn import ModelConfig, OpenNVResUNet1D


@dataclass
class TrainConfig:
    """Training recipe and step/wall-clock budget configuration."""

    run_name: str = "model_a_seed0"
    seed: int = 3001
    train_stream_seed: int = 20260101
    total_steps: int = 800
    batch_size: int = 32
    lr: float = 1.5e-3
    min_lr: float = 5.0e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 80
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.99
    eval_every_steps: int = 100
    max_wall_clock_min: float = 20.0
    n_gen_workers: int = 4
    stream_chunk_batches: int = 8
    warm_start_ckpt: Optional[str] = None
    # Optional Sim2Real training stream knobs (Model B only)
    per_axis_contrast_jitter: float = 0.0
    allow_high_field_in_window: bool = False
    b_mag_max_mt: float = B_MAG_MAX_MT
    hwhm_min_mhz: float = 3.0
    hwhm_max_mhz: float = 9.0
    mw_power_tilt_amp: float = 0.0
    mw_power_tilt_prob: float = 0.0
    save_dir: str = "models/checkpoints"
    history_dir: str = "models"


def get_val_sel_indices(data_dir: Path = DEFAULT_SYNTHETIC_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """Return strictly disjoint ``(val_rep_indices, val_sel_indices)`` within ``val_id.npz``."""
    ref_path = data_dir / "val_baseline_ref.npz"
    with np.load(ref_path, allow_pickle=False) as ref:
        val_rep_idx = np.sort(ref["val_id_indices"].astype(np.int64))
    n_val_id = SPLIT_REGISTRY["val_id"].n_samples
    rep_set = set(val_rep_idx.tolist())
    val_sel_idx = np.array([i for i in range(n_val_id) if i not in rep_set], dtype=np.int64)
    assert len(set(val_rep_idx.tolist()) & set(val_sel_idx.tolist())) == 0
    return val_rep_idx, val_sel_idx


def load_val_sel_arrays(data_dir: Path = DEFAULT_SYNTHETIC_DIR) -> Dict[str, np.ndarray]:
    """Load the 1,000-spectrum ``val_sel`` subset from ``val_id.npz`` for model selection only."""
    _, sel_idx = get_val_sel_indices(data_dir=data_dir)
    full_val = load_synthetic_split("val_id", data_dir=data_dir, allow_test_read=False)
    sel_arrays: Dict[str, np.ndarray] = {}
    for k, v in full_val.items():
        if v.ndim >= 1 and v.shape[0] == full_val["X_noisy"].shape[0]:
            sel_arrays[k] = v[sel_idx]
        else:
            sel_arrays[k] = v
    sel_arrays["val_sel_indices"] = sel_idx
    return sel_arrays


def build_train_split_config(train_cfg: TrainConfig) -> SplitConfig:
    """Build the on-the-fly training ``SplitConfig`` (strictly ID drifts/noise only)."""
    hwhm_rng = (
        (float(train_cfg.hwhm_min_mhz), float(train_cfg.hwhm_max_mhz))
        if (train_cfg.hwhm_min_mhz != 3.0 or train_cfg.hwhm_max_mhz != 9.0)
        else None
    )
    return SplitConfig(
        name=f"stream_{train_cfg.run_name}",
        split_seed=int(train_cfg.train_stream_seed),
        n_samples=0,
        stratified_regimes=False,
        factor_ablation_id=FACTOR_ID_STANDARD,
        grid_perturb_prob=0.30,
        allow_high_field_in_window=bool(train_cfg.allow_high_field_in_window),
        b_mag_range_mt=(B_MAG_MIN_MT, float(train_cfg.b_mag_max_mt)),
        in_window_freq_bounds_mhz=(2760.0, 2980.0),
        per_axis_contrast_jitter=float(train_cfg.per_axis_contrast_jitter),
        hwhm_range_mhz=hwhm_rng,
        mw_power_tilt_amp=float(train_cfg.mw_power_tilt_amp),
        mw_power_tilt_prob=float(train_cfg.mw_power_tilt_prob),
    )


def _arrays_to_tensor_batch(
    arrays: Dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    keys = ("X_noisy", "valid_mask", "Y_clean", "Y_splittings", "Y_b_mag", "Y_d0", "Y_hwhm")
    return {k: torch.from_numpy(arrays[k][indices]).to(device) for k in keys}


def _update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.mul_(decay).add_(p, alpha=1.0 - decay)
        for b_ema, b in zip(ema_model.buffers(), model.buffers()):
            b_ema.copy_(b)


def _cosine_lr(step: int, total_steps: int, warmup_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * float(step + 1) / float(max(warmup_steps, 1))
    progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_on_arrays(
    model: OpenNVResUNet1D,
    criterion: OpenNVMultiTaskLoss,
    arrays: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 128,
) -> Dict[str, float]:
    """Evaluate loss terms, |B| error quantiles, wrong-basin rate, dip RMSE, and 80% coverage."""
    model.eval()
    n = int(arrays["X_noisy"].shape[0])
    loss_sums: Dict[str, float] = {}
    b_pred_all: List[np.ndarray] = []
    b_vec_pred_all: List[np.ndarray] = []
    q10_all: List[np.ndarray] = []
    q90_all: List[np.ndarray] = []
    unc_all: List[np.ndarray] = []
    tetra_all: List[np.ndarray] = []
    y_hat_all: List[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = np.arange(start, end)
        batch = _arrays_to_tensor_batch(arrays, idx, device)
        pred = model(batch["X_noisy"], batch["valid_mask"])
        _, terms = criterion(pred, batch)
        bs = float(end - start)
        for k, v in terms.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + float(v.item()) * bs
        b_pred_all.append(pred["b_mag_phys"].cpu().numpy())
        b_vec_pred_all.append(pred["b_vec_canon"].cpu().numpy())
        q10_all.append(pred["q10"].cpu().numpy())
        q90_all.append(pred["q90"].cpu().numpy())
        unc_all.append(pred["uncertainty_mt"].cpu().numpy())
        tetra_all.append(pred["tetra_residual"].cpu().numpy())
        y_hat_all.append(pred["y_clean_hat"].cpu().numpy())

    b_pred = np.concatenate(b_pred_all, axis=0).astype(np.float64)
    b_vec_pred = np.concatenate(b_vec_pred_all, axis=0).astype(np.float64)
    q10 = np.concatenate(q10_all, axis=0).astype(np.float64)
    q90 = np.concatenate(q90_all, axis=0).astype(np.float64)
    unc = np.concatenate(unc_all, axis=0).astype(np.float64)
    tetra = np.concatenate(tetra_all, axis=0).astype(np.float64)
    y_hat = np.concatenate(y_hat_all, axis=0).astype(np.float64)

    b_true = arrays["Y_b_mag"].astype(np.float64)
    b_vec_true = arrays["aux_b_vec_canon"].astype(np.float64)
    y_clean = arrays["Y_clean"].astype(np.float64)
    vm = arrays["valid_mask"].astype(bool)

    err_ut = np.abs(b_pred - b_true) * 1000.0
    wedge_mt = np.linalg.norm(b_vec_pred - b_vec_true, axis=1)
    cov_80 = float(np.mean((b_true >= q10) & (b_true <= q90)) * 100.0)

    dip_mask = vm & ((1.0 - y_clean) > 0.0015)
    dip_rmse = float(np.sqrt(np.mean((y_hat[dip_mask] - y_clean[dip_mask]) ** 2))) if np.any(dip_mask) else 0.0
    valid_rmse = float(np.sqrt(np.mean((y_hat[vm] - y_clean[vm]) ** 2)))

    metrics: Dict[str, float] = {f"loss_{k}": round(v / max(n, 1), 6) for k, v in loss_sums.items()}
    metrics.update(
        {
            "median_ut": round(float(np.median(err_ut)), 2),
            "p90_ut": round(float(np.percentile(err_ut, 90)), 2),
            "mae_ut": round(float(np.mean(err_ut)), 2),
            "rms_ut": round(float(np.sqrt(np.mean(err_ut * err_ut))), 2),
            "fail_100_pct": round(float(np.mean(err_ut > 100.0) * 100.0), 2),
            "wrong_basin_pct": round(float(np.mean(wedge_mt > 0.15) * 100.0), 2),
            "cov80_pct": round(cov_80, 2),
            "mean_unc_ut": round(float(np.mean(unc) * 1000.0), 2),
            "mean_tetra_ut": round(float(np.mean(tetra) * 1000.0), 2),
            "dip_rmse": round(dip_rmse, 6),
            "valid_rmse": round(valid_rmse, 6),
        }
    )

    if "aux_regime_id" in arrays:
        reg = arrays["aux_regime_id"]
        for r_id, r_name in ((0, "hard"), (1, "medium"), (2, "easy")):
            rm = reg == r_id
            if np.any(rm):
                metrics[f"{r_name}_median_ut"] = round(float(np.median(err_ut[rm])), 2)
                metrics[f"{r_name}_wrong_basin_pct"] = round(float(np.mean(wedge_mt[rm] > 0.15) * 100.0), 2)
    return metrics


def run_overfit_sanity_check(
    model_cfg: ModelConfig,
    loss_weights: Optional[LossWeights] = None,
    n_samples: int = 64,
    n_steps: int = 320,
    lr: float = 2.5e-3,
    seed: int = 3000,
    mini_batch_size: int = 16,
) -> Dict[str, Any]:
    """Run 64-sample fixed-batch overfit gate (Run 3a): total loss MUST drop by > 90%."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stream_cfg = SplitConfig(
        name="overfit_sanity_64",
        split_seed=20260199,
        n_samples=n_samples,
        stratified_regimes=False,
        factor_ablation_id=FACTOR_ID_STANDARD,
        grid_perturb_prob=0.30,
    )
    arrays = generate_split_arrays(stream_cfg, n_samples=n_samples, start_idx=0, n_workers=1)
    idx_all = np.arange(n_samples)
    full_batch = _arrays_to_tensor_batch(arrays, idx_all, device)

    model = OpenNVResUNet1D(model_cfg).to(device)
    criterion = OpenNVMultiTaskLoss(
        weights=loss_weights if loss_weights is not None else LossWeights(),
        uncertainty_mode=model_cfg.uncertainty_mode,
        use_denoiser_head=model_cfg.use_denoiser_head,
        student_t_df=model_cfg.student_t_df,
    )
    grad_cal = compute_per_term_grad_norms(model, criterion, full_batch)
    with torch.no_grad():
        model.eval()
        init_loss = float(criterion(model(full_batch["X_noisy"], full_batch["valid_mask"]), full_batch)[0].item())

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    rng_b = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for step in range(n_steps):
        cur_lr = _cosine_lr(step, n_steps, warmup_steps=20, lr=lr, min_lr=lr * 0.1)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        mb_idx = rng_b.choice(n_samples, size=min(mini_batch_size, n_samples), replace=False)
        mb = _arrays_to_tensor_batch(arrays, mb_idx, device)
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(mb["X_noisy"], mb["valid_mask"])
        loss, _ = criterion(pred, mb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    dt_sec = time.perf_counter() - t0

    eval_m = evaluate_on_arrays(model, criterion, arrays, device, batch_size=64)
    final_loss = float(eval_m["loss_total"])
    drop_pct = (1.0 - final_loss / max(init_loss, 1e-12)) * 100.0

    return {
        "device": str(device),
        "n_params": model.count_parameters(),
        "n_samples": n_samples,
        "n_steps": n_steps,
        "mini_batch_size": mini_batch_size,
        "wall_sec": round(dt_sec, 2),
        "ms_per_step": round((dt_sec * 1000.0) / n_steps, 2),
        "init_loss": round(init_loss, 5),
        "final_loss": round(final_loss, 5),
        "loss_drop_pct": round(drop_pct, 2),
        "passed_90pct_drop": bool(drop_pct > 90.0),
        "final_metrics": eval_m,
        "grad_norm_calibration": grad_cal,
    }


def train_model(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    loss_weights: Optional[LossWeights] = None,
    val_sel_arrays: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Train ``OpenNVResUNet1D`` with on-the-fly streaming and ``val_sel`` checkpoint selection."""
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    if val_sel_arrays is None:
        val_sel_arrays = load_val_sel_arrays()

    model = OpenNVResUNet1D(model_cfg).to(device)
    if train_cfg.warm_start_ckpt is not None:
        ckpt = torch.load(train_cfg.warm_start_ckpt, map_location=device, weights_only=False)
        state = ckpt.get("ema_state_dict", ckpt["model_state_dict"])
        model.load_state_dict(state, strict=True)

    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()

    lw = loss_weights if loss_weights is not None else LossWeights()
    criterion = OpenNVMultiTaskLoss(
        weights=lw,
        uncertainty_mode=model_cfg.uncertainty_mode,
        use_denoiser_head=model_cfg.use_denoiser_head,
        student_t_df=model_cfg.student_t_df,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    stream_cfg = build_train_split_config(train_cfg)
    chunk_size = train_cfg.batch_size * train_cfg.stream_chunk_batches
    sample_cursor = 0
    buf_arrays: Optional[Dict[str, np.ndarray]] = None
    buf_pos = 0

    # Keep a single persistent worker pool (if n_gen_workers > 1) to avoid Windows spawn overhead per chunk
    from concurrent.futures import ProcessPoolExecutor

    gen_pool = ProcessPoolExecutor(max_workers=train_cfg.n_gen_workers) if train_cfg.n_gen_workers > 1 else None

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_step = 0
    best_ema_state = copy.deepcopy(ema_model.state_dict())
    best_model_state = copy.deepcopy(model.state_dict())

    t_start = time.perf_counter()
    max_sec = train_cfg.max_wall_clock_min * 60.0
    train_loss_acc: Dict[str, float] = {}
    acc_steps = 0
    gen_time_sec = 0.0
    opt_time_sec = 0.0
    steps_completed = 0

    try:
        for step in range(1, train_cfg.total_steps + 1):
            elapsed = time.perf_counter() - t_start
            if elapsed >= max_sec and step > train_cfg.warmup_steps:
                break

            if buf_arrays is None or (buf_pos + train_cfg.batch_size) > int(buf_arrays["X_noisy"].shape[0]):
                t_g0 = time.perf_counter()
                buf_arrays = generate_split_arrays(
                    stream_cfg,
                    n_samples=chunk_size,
                    start_idx=sample_cursor,
                    n_workers=train_cfg.n_gen_workers,
                    executor=gen_pool,
                )
                gen_time_sec += time.perf_counter() - t_g0
                sample_cursor += chunk_size
                buf_pos = 0

            idx_b = np.arange(buf_pos, buf_pos + train_cfg.batch_size)
            buf_pos += train_cfg.batch_size
            batch = _arrays_to_tensor_batch(buf_arrays, idx_b, device)

            cur_lr = _cosine_lr(
                step - 1, train_cfg.total_steps, train_cfg.warmup_steps, train_cfg.lr, train_cfg.min_lr
            )
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            t_o0 = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda"):
                    pred = model(batch["X_noisy"], batch["valid_mask"])
                    loss, terms = criterion(pred, batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(batch["X_noisy"], batch["valid_mask"])
                loss, terms = criterion(pred, batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
                optimizer.step()

            ema_d = min(train_cfg.ema_decay, (1.0 + step) / (10.0 + step))
            _update_ema(ema_model, model, ema_d)
            opt_time_sec += time.perf_counter() - t_o0
            steps_completed = step

            for k, v in terms.items():
                train_loss_acc[k] = train_loss_acc.get(k, 0.0) + float(v.detach().item())
            acc_steps += 1

            is_last = (step == train_cfg.total_steps) or ((time.perf_counter() - t_start) >= max_sec)
            if (step % train_cfg.eval_every_steps == 0) or is_last:
                val_metrics = evaluate_on_arrays(ema_model, criterion, val_sel_arrays, device)
                train_avg = {f"train_loss_{k}": round(v / max(acc_steps, 1), 6) for k, v in train_loss_acc.items()}
                train_loss_acc.clear()
                acc_steps = 0

                rec: Dict[str, Any] = {
                    "step": step,
                    "spectra_seen": step * train_cfg.batch_size,
                    "elapsed_sec": round(time.perf_counter() - t_start, 2),
                    "lr": round(cur_lr, 7),
                    **train_avg,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
                history.append(rec)

                cur_val_loss = float(val_metrics["loss_total"])
                if cur_val_loss < best_val_loss:
                    best_val_loss = cur_val_loss
                    best_step = step
                    best_ema_state = copy.deepcopy(ema_model.state_dict())
                    best_model_state = copy.deepcopy(model.state_dict())

                print(
                    f"[{train_cfg.run_name}] step={step}/{train_cfg.total_steps} "
                    f"({rec['elapsed_sec']:.1f}s) train={rec['train_loss_total']:.4f} "
                    f"val_sel={cur_val_loss:.4f} med={val_metrics['median_ut']:.1f}uT "
                    f"p90={val_metrics['p90_ut']:.1f}uT wb={val_metrics['wrong_basin_pct']:.1f}% "
                    f"cov80={val_metrics['cov80_pct']:.1f}%",
                    flush=True,
                )
                if is_last:
                    break
    finally:
        if gen_pool is not None:
            gen_pool.shutdown(wait=True)

    total_wall_sec = time.perf_counter() - t_start
    still_decreasing = (
        len(history) >= 2 and history[-1]["val_loss_total"] < history[-2]["val_loss_total"]
    )

    save_dir = Path(train_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"{train_cfg.run_name}.pt"

    ema_model.load_state_dict(best_ema_state)
    best_val_metrics = evaluate_on_arrays(ema_model, criterion, val_sel_arrays, device)

    torch.save(
        {
            "run_name": train_cfg.run_name,
            "model_config": model_cfg.to_dict(),
            "train_config": asdict(train_cfg),
            "loss_weights": lw.to_dict(),
            "best_step": best_step,
            "steps_completed": steps_completed,
            "spectra_seen": steps_completed * train_cfg.batch_size,
            "best_val_sel_metrics": best_val_metrics,
            "model_state_dict": best_model_state,
            "ema_state_dict": best_ema_state,
        },
        ckpt_path,
    )

    hist_dir = Path(train_cfg.history_dir)
    hist_dir.mkdir(parents=True, exist_ok=True)
    hist_path = hist_dir / f"train_history_{train_cfg.run_name}.json"
    summary_payload = {
        "run_name": train_cfg.run_name,
        "device": str(device),
        "n_params": model.count_parameters(),
        "model_config": model_cfg.to_dict(),
        "train_config": asdict(train_cfg),
        "loss_weights": lw.to_dict(),
        "steps_completed": steps_completed,
        "spectra_seen": steps_completed * train_cfg.batch_size,
        "best_step": best_step,
        "total_wall_sec": round(total_wall_sec, 2),
        "gen_time_sec": round(gen_time_sec, 2),
        "opt_time_sec": round(opt_time_sec, 2),
        "ms_per_opt_step": round((opt_time_sec * 1000.0) / max(steps_completed, 1), 2),
        "val_sel_still_decreasing_at_end": bool(still_decreasing),
        "best_val_sel_metrics": best_val_metrics,
        "history": history,
        "checkpoint_path": str(ckpt_path),
    }
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    return summary_payload
