"""Stage 3 multi-task physics-guided loss functions and gradient-norm calibration utilities."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    """Loss term weights for Stage 3 training (Part 3)."""

    w_denoise: float = 1.0
    w_split: float = 0.5
    w_bmag: float = 1.0
    w_aux: float = 0.1
    w_tetra: float = 0.0
    bmag_l1_aux: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, q: float) -> torch.Tensor:
    """Quantile (pinball) loss for quantile ``q in (0, 1)``."""
    err = y_true - y_pred
    return torch.maximum(q * err, (q - 1.0) * err).mean()


def laplace_nll_loss(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
) -> torch.Tensor:
    """Laplace negative log-likelihood ``log(2*sigma) + |y - mu| / sigma``."""
    sigma = torch.exp(log_sigma)
    nll = log_sigma + math.log(2.0) + torch.abs(y_true - mu) / sigma
    return nll.mean()


def student_t_nll_loss(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    df: float = 4.0,
) -> torch.Tensor:
    """Student-t negative log-likelihood with degrees of freedom ``df``."""
    sigma = torch.exp(log_sigma)
    z = (y_true - mu) / sigma
    const = (
        math.lgamma(0.5 * df)
        + 0.5 * math.log(df * math.pi)
        - math.lgamma(0.5 * (df + 1.0))
    )
    nll = log_sigma + const + 0.5 * (df + 1.0) * torch.log1p((z * z) / df)
    return nll.mean()


class OpenNVMultiTaskLoss(nn.Module):
    """Multi-task physics-guided ODMR loss (denoising, splittings, |B| uncertainty, aux, tetra)."""

    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        uncertainty_mode: str = "quantile",
        use_denoiser_head: bool = True,
        student_t_df: float = 4.0,
    ) -> None:
        super().__init__()
        self.weights = weights if weights is not None else LossWeights()
        self.uncertainty_mode = uncertainty_mode
        self.use_denoiser_head = use_denoiser_head
        self.student_t_df = float(student_t_df)

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        y_clean = batch["Y_clean"].to(dtype=torch.float32)
        valid_mask = batch["valid_mask"].to(dtype=torch.float32)
        y_split = batch["Y_splittings"].to(dtype=torch.float32)
        y_bmag = batch["Y_b_mag"].to(dtype=torch.float32)
        y_d0 = batch["Y_d0"].to(dtype=torch.float32)
        y_hwhm = batch["Y_hwhm"].to(dtype=torch.float32)

        # 1. L_denoise: dip-weighted SmoothL1(beta=0.05) on y_clean / 0.02 over valid_mask == 1
        if self.use_denoiser_head and self.weights.w_denoise > 0.0:
            elem_loss = F.smooth_l1_loss(
                pred["y_clean_hat"] / 0.02,
                y_clean / 0.02,
                beta=0.05,
                reduction="none",
            )
            dip_weights = 1.0 + 6.0 * torch.clamp((1.0 - y_clean) / 0.01, 0.0, 1.0)
            w_eff = dip_weights * valid_mask
            l_denoise = (elem_loss * w_eff).sum() / torch.clamp(w_eff.sum(), min=1.0)
        else:
            l_denoise = torch.zeros((), dtype=torch.float32, device=y_bmag.device)

        # 2. L_split: SmoothL1(beta=0.1) on splittings / 10.0 (beta=0.1 <-> 1.0 MHz)
        l_split = F.smooth_l1_loss(
            pred["splittings_hat"] / 10.0,
            y_split / 10.0,
            beta=0.1,
            reduction="mean",
        )

        # 3. L_bmag: Physics-coupled uncertainty loss centered on b_mag_phys + small L1 aux
        b_phys = pred["b_mag_phys"]
        if self.uncertainty_mode == "quantile":
            l_q10 = pinball_loss(y_bmag, pred["q10"], 0.10)
            l_q50 = pinball_loss(y_bmag, pred["q50"], 0.50)
            l_q90 = pinball_loss(y_bmag, pred["q90"], 0.90)
            l_unc = l_q10 + l_q50 + l_q90
        elif self.uncertainty_mode == "laplace":
            l_unc = laplace_nll_loss(y_bmag, b_phys, pred["log_sigma"])
        elif self.uncertainty_mode == "student_t":
            l_unc = student_t_nll_loss(y_bmag, b_phys, pred["log_sigma"], df=self.student_t_df)
        else:
            raise ValueError(f"Unsupported uncertainty_mode: {self.uncertainty_mode}")

        l_bmag_l1 = F.l1_loss(b_phys, y_bmag, reduction="mean")
        l_bmag = l_unc + self.weights.bmag_l1_aux * l_bmag_l1

        # 4. L_aux: SmoothL1 on d0_offset (ID span 6.0 MHz) and hwhm (ID span 6.0 MHz)
        l_d0 = F.smooth_l1_loss(pred["d0_offset_hat"] / 3.0, y_d0 / 3.0, beta=0.1, reduction="mean")
        l_hwhm = F.smooth_l1_loss(
            (pred["hwhm_hat"] - 6.0) / 3.0,
            (y_hwhm - 6.0) / 3.0,
            beta=0.1,
            reduction="mean",
        )
        l_aux = 0.5 * (l_d0 + l_hwhm)

        # 5. L_tetra: L1 penalty on tetrahedral consistency residual (mT)
        l_tetra = pred["tetra_residual"].mean()

        total = (
            self.weights.w_denoise * l_denoise
            + self.weights.w_split * l_split
            + self.weights.w_bmag * l_bmag
            + self.weights.w_aux * l_aux
            + self.weights.w_tetra * l_tetra
        )

        terms = {
            "total": total,
            "denoise": l_denoise,
            "split": l_split,
            "bmag": l_bmag,
            "bmag_l1": l_bmag_l1,
            "aux": l_aux,
            "tetra": l_tetra,
        }
        return total, terms


def compute_per_term_grad_norms(
    model: nn.Module,
    criterion: OpenNVMultiTaskLoss,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    """Measure unweighted & weighted loss values and global gradient norms per loss term on a fixed batch."""
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    weight_map = {
        "denoise": criterion.weights.w_denoise,
        "split": criterion.weights.w_split,
        "bmag": criterion.weights.w_bmag,
        "aux": criterion.weights.w_aux,
        "tetra": criterion.weights.w_tetra,
    }
    pred = model(batch["X_noisy"], batch["valid_mask"])
    _, terms = criterion(pred, batch)

    out: Dict[str, Dict[str, float]] = {}
    for name in ("denoise", "split", "bmag", "aux", "tetra"):
        loss_val = terms[name]
        w_val = float(weight_map[name])
        grads = torch.autograd.grad(
            loss_val, params, retain_graph=True, allow_unused=True
        )
        sq_sum = 0.0
        for g in grads:
            if g is not None:
                sq_sum += float(torch.sum(g.detach() * g.detach()).item())
        gnorm = math.sqrt(sq_sum)
        out[name] = {
            "raw_loss": round(float(loss_val.detach().item()), 6),
            "weight": w_val,
            "weighted_loss": round(w_val * float(loss_val.detach().item()), 6),
            "raw_grad_norm": round(gnorm, 6),
            "weighted_grad_norm": round(w_val * gnorm, 6),
        }
    return out
