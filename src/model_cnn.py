"""Stage 3 1D ResUNet denoiser with physics-guided parameter heads and calibrated uncertainty."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CANONICAL_FREQS, D0_MHZ, GAMMA_E_MHZ_MT, N_POINTS

_SQRT_2: float = math.sqrt(2.0)
_SQRT_3: float = math.sqrt(3.0)
_SQRT_3_OVER_2: float = _SQRT_3 / 2.0


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and ablation configuration for ``OpenNVResUNet1D``."""

    base_channels: int = 32
    use_ch3: bool = True
    use_denoiser_head: bool = True
    stop_grad_dip: bool = False
    uncertainty_mode: str = "quantile"  # "quantile" | "laplace" | "student_t"
    student_t_df: float = 4.0
    n_points: int = N_POINTS

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def prepare_model_inputs(
    x_noisy: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    use_ch3: bool = True,
    u_coord: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct normalized 3- or 4-channel 1D ResUNet inputs from raw spectra.

    Channels:
      - ``ch0 = clamp((x - median_valid) / 0.02, -15, 15) * valid_mask``
      - ``ch1 = valid_mask`` (float32, 1.0 inside measured window, 0.0 outside)
      - ``ch2 = CoordConv`` ``u = linspace(-1, 1, 512)``
      - ``ch3`` (optional): ``clamp((x - median_valid) / robust_sigma_valid, -15, 15) * valid_mask``
        where ``robust_sigma_valid = max(1.4826 * MAD(diff(x_valid)) / sqrt(2), 1e-4)``.

    Returns:
        Tuple ``(x_in, vm_f, u_expanded)`` with shapes ``(B, C_in, L)``, ``(B, L)``, ``(B, L)``.
    """
    if x_noisy.ndim == 1:
        x = x_noisy.unsqueeze(0)
    else:
        x = x_noisy
    x = x.to(dtype=torch.float32)
    bsz, length = x.shape
    device = x.device

    if valid_mask is None:
        vm = torch.ones((bsz, length), dtype=torch.float32, device=device)
    else:
        vm = valid_mask.unsqueeze(0) if valid_mask.ndim == 1 else valid_mask
        vm = vm.to(device=device, dtype=torch.float32)
    vm_bool = vm >= 0.5
    vm = vm_bool.to(dtype=torch.float32)

    # Compute per-sample median strictly over valid points
    x_masked_nan = torch.where(vm_bool, x, torch.full_like(x, float("nan")))
    med_valid = torch.nanmedian(x_masked_nan, dim=-1, keepdim=True).values
    med_valid = torch.nan_to_num(med_valid, nan=1.0)

    centered = x - med_valid
    ch0 = torch.clamp(centered / 0.02, -15.0, 15.0) * vm
    ch1 = vm

    if u_coord is None:
        u_1d = torch.linspace(-1.0, 1.0, length, dtype=torch.float32, device=device)
    else:
        u_1d = u_coord.to(device=device, dtype=torch.float32)
    u_exp = u_1d.unsqueeze(0).expand(bsz, length)
    ch2 = u_exp

    if not use_ch3:
        x_in = torch.stack([ch0, ch1, ch2], dim=1)
        return x_in, vm, u_exp

    # Robust noise scale from MAD of first differences on valid adjacent points
    dx = x[:, 1:] - x[:, :-1]
    vm_d = vm_bool[:, 1:] & vm_bool[:, :-1]
    dx_nan = torch.where(vm_d, dx, torch.full_like(dx, float("nan")))
    med_dx = torch.nanmedian(dx_nan, dim=-1, keepdim=True).values
    med_dx = torch.nan_to_num(med_dx, nan=0.0)
    mad_dx = torch.nanmedian((dx_nan - med_dx).abs(), dim=-1, keepdim=True).values
    mad_dx = torch.nan_to_num(mad_dx, nan=1e-3)
    robust_sigma = torch.clamp(1.4826 * mad_dx / _SQRT_2, min=1e-4)
    ch3 = torch.clamp(centered / robust_sigma, -15.0, 15.0) * vm

    x_in = torch.stack([ch0, ch1, ch2, ch3], dim=1)
    return x_in, vm, u_exp


def _smooth_pos(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Smooth approximation to ``max(0, x)`` with strictly positive finite derivative everywhere."""
    return 0.5 * (x + torch.sqrt(x * x + eps * eps))


def invert_splittings_order2_torch(
    splittings_mhz: torch.Tensor,
    d0_mhz: torch.Tensor | float = D0_MHZ,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable float64 order-2 transverse-field inversion from 4-axis splittings.

    Mirrors ``src.physics.b_mag_from_splittings(..., order=2)`` internally in ``float64``
    using smooth positive guards (no hard min/max cliffs) so gradients remain finite and
    non-zero at ``splitting -> 0``, ``|B| -> 0.05/3.20 mT``, and degenerate axes.

    Args:
        splittings_mhz: Tensor of shape ``(..., 4)`` in MHz.
        d0_mhz: Scalar float or Tensor broadcastable to ``splittings_mhz.shape[:-1]`` in MHz.

    Returns:
        Tuple ``(b_mag_mt, p_long_mt)`` in original input dtype:
          - ``b_mag_mt``: shape ``(...)`` scalar field magnitude in mT.
          - ``p_long_mt``: shape ``(..., 4)`` order-2 transverse-corrected longitudinal
            projections ``|B_par, k|`` in mT.
    """
    orig_dtype = splittings_mhz.dtype
    s64 = splittings_mhz.to(dtype=torch.float64)
    if isinstance(d0_mhz, torch.Tensor):
        d0_64 = d0_mhz.to(device=s64.device, dtype=torch.float64)
    else:
        d0_64 = torch.full(s64.shape[:-1], float(d0_mhz), device=s64.device, dtype=torch.float64)
    d0_val = d0_64.unsqueeze(-1)
    d0_sq = d0_val * d0_val

    half_df = 0.5 * torch.sqrt(s64 * s64 + 1e-18)
    half_df_sq = half_df * half_df
    a = half_df

    for _ in range(5):
        g2_b2 = 0.75 * torch.sum(a * a, dim=-1, keepdim=True)
        b_sq = _smooth_pos(g2_b2 - a * a, eps=1e-9)
        repulsion_sq = (b_sq / (2.0 * d0_val)) ** 2
        rem_sq = _smooth_pos(half_df_sq - repulsion_sq, eps=1e-9)
        denom = 1.0 + _smooth_pos(d0_sq - a * a - 1.0, eps=1e-6)
        raw_corr = 1.0 - b_sq / (2.0 * denom)
        # Smooth clamp of corr to [0.5, 1.0]
        corr_lo = 0.5 + _smooth_pos(raw_corr - 0.5, eps=1e-8)
        corr = 1.0 - _smooth_pos(1.0 - corr_lo, eps=1e-8)
        a = torch.sqrt(rem_sq + 1e-18) / corr

    p_mt = a / GAMMA_E_MHZ_MT
    b_mag_64 = _SQRT_3_OVER_2 * torch.sqrt(torch.sum(p_mt * p_mt, dim=-1) + 1e-18)
    return b_mag_64.to(dtype=orig_dtype), p_mt.to(dtype=orig_dtype)


def b_mag_from_splittings_torch(
    splittings_mhz: torch.Tensor,
    d0_mhz: torch.Tensor | float = D0_MHZ,
) -> torch.Tensor:
    """Differentiable order-2 ``|B|`` (mT) from 4-axis ODMR splittings (MHz)."""
    b_mag, _ = invert_splittings_order2_torch(splittings_mhz, d0_mhz=d0_mhz)
    return b_mag


def canonical_b_vec_from_splittings(
    splittings_mhz: torch.Tensor,
    d0_mhz: torch.Tensor | float = D0_MHZ,
    p_long_mt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Analytical tetrahedral inversion into canonical wedge ``0 <= Bx <= By <= Bz`` (mT).

    From sorted order-2 corrected longitudinal projections ``p1 <= p2 <= p3 <= p4`` (mT):
      ``Bx = (sqrt(3)/2) * (p4 - p3)``
      ``By = (sqrt(3)/2) * (p4 - p2)``
      ``Bz = (sqrt(3)/2) * (p3 + p2)``
    clamped non-negative and sorted into ``0 <= Bx <= By <= Bz``.
    """
    if p_long_mt is None:
        _, p_mt = invert_splittings_order2_torch(splittings_mhz, d0_mhz=d0_mhz)
    else:
        p_mt = p_long_mt

    p_sorted, _ = torch.sort(p_mt, dim=-1)
    p2 = p_sorted[..., 1]
    p3 = p_sorted[..., 2]
    p4 = p_sorted[..., 3]

    bx = torch.clamp(_SQRT_3_OVER_2 * (p4 - p3), min=0.0)
    by = torch.clamp(_SQRT_3_OVER_2 * (p4 - p2), min=0.0)
    bz = torch.clamp(_SQRT_3_OVER_2 * (p3 + p2), min=0.0)
    b_vec = torch.stack([bx, by, bz], dim=-1)
    b_vec_sorted, _ = torch.sort(b_vec, dim=-1)
    return b_vec_sorted


def compute_tetra_residual(
    splittings_mhz: torch.Tensor,
    d0_mhz: torch.Tensor | float = D0_MHZ,
    p_long_mt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute tetrahedral consistency residual ``| |p3 + p2 - p4| - p1 |`` in mT.

    Evaluated strictly on the order-2 transverse-corrected longitudinal projections
    ``p1 <= p2 <= p3 <= p4``. Equals ``0.0`` on exact ``eigh`` data for all 3D orientations
    (both ``Bx + By <= Bz`` and ``Bx + By > Bz``).
    """
    if p_long_mt is None:
        _, p_mt = invert_splittings_order2_torch(splittings_mhz, d0_mhz=d0_mhz)
    else:
        p_mt = p_long_mt

    p_sorted, _ = torch.sort(p_mt, dim=-1)
    p1 = p_sorted[..., 0]
    p2 = p_sorted[..., 1]
    p3 = p_sorted[..., 2]
    p4 = p_sorted[..., 3]

    inner = p3 + p2 - p4
    abs_inner = torch.sqrt(inner * inner + 1e-12)
    diff = abs_inner - p1
    return torch.sqrt(diff * diff + 1e-12) - 1e-6


class ResBlock1D(nn.Module):
    """Residual 1D convolution block using GroupNorm(8) + SiLU (no BatchNorm)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        g1 = min(8, in_ch)
        g2 = min(8, out_ch)
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class OpenNVResUNet1D(nn.Module):
    """1D ResUNet denoiser with physics-guided splitting/field/uncertainty heads."""

    def __init__(self, cfg: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else ModelConfig()
        c = int(self.cfg.base_channels)
        in_ch = 4 if self.cfg.use_ch3 else 3

        self.register_buffer(
            "u_coord",
            torch.linspace(-1.0, 1.0, int(self.cfg.n_points), dtype=torch.float32),
            persistent=False,
        )

        # Encoder (512 -> 256 -> 128 -> 64 -> 32)
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, c, kernel_size=7, padding=3),
            ResBlock1D(c, c),
        )
        self.down1 = nn.Sequential(
            nn.Conv1d(c, 2 * c, kernel_size=4, stride=2, padding=1),
            ResBlock1D(2 * c, 2 * c),
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(2 * c, 3 * c, kernel_size=4, stride=2, padding=1),
            ResBlock1D(3 * c, 3 * c),
        )
        self.down3 = nn.Sequential(
            nn.Conv1d(3 * c, 4 * c, kernel_size=4, stride=2, padding=1),
            ResBlock1D(4 * c, 4 * c),
        )
        self.down4 = nn.Conv1d(4 * c, 4 * c, kernel_size=4, stride=2, padding=1)

        # Dilated bottleneck (d = 1, 2, 4 at length 32 -> covers full 240 MHz window)
        self.bottleneck = nn.Sequential(
            ResBlock1D(4 * c, 4 * c, kernel_size=3, dilation=1),
            ResBlock1D(4 * c, 4 * c, kernel_size=3, dilation=2),
            ResBlock1D(4 * c, 4 * c, kernel_size=3, dilation=4),
        )

        # Spatial bottleneck projection preserving exact frequency-bin positions (B, 8, 32) -> 256
        self.bottleneck_spatial = nn.Sequential(
            nn.GroupNorm(min(8, 4 * c), 4 * c),
            nn.SiLU(),
            nn.Conv1d(4 * c, 8, kernel_size=1),
        )
        spatial_b_dim = 8 * (int(self.cfg.n_points) // 16)  # 8 * 32 = 256

        # Decoder (32 -> 64 -> 128 -> 256 -> 512)
        if self.cfg.use_denoiser_head:
            self.up3_fuse = nn.Sequential(
                nn.Conv1d(4 * c + 4 * c, 4 * c, kernel_size=3, padding=1),
                ResBlock1D(4 * c, 4 * c),
            )
            self.up2_fuse = nn.Sequential(
                nn.Conv1d(4 * c + 3 * c, 3 * c, kernel_size=3, padding=1),
                ResBlock1D(3 * c, 3 * c),
            )
            self.up1_fuse = nn.Sequential(
                nn.Conv1d(3 * c + 2 * c, 2 * c, kernel_size=3, padding=1),
                ResBlock1D(2 * c, 2 * c),
            )
            self.up0_fuse = nn.Sequential(
                nn.Conv1d(2 * c + c, c, kernel_size=3, padding=1),
                ResBlock1D(c, c),
            )
            self.denoise_out = nn.Sequential(
                nn.GroupNorm(min(8, c), c),
                nn.SiLU(),
                nn.Conv1d(c, 1, kernel_size=1),
            )

            # Lightweight 1D Conv encoder over [dip_hat_masked / 0.02, u, valid_mask]
            # Downsamples 512 -> 128 -> 32 -> 16, preserving explicit spatial dip locations!
            self.dip_encoder = nn.Sequential(
                nn.Conv1d(3, c, kernel_size=7, stride=4, padding=3),
                nn.GroupNorm(min(8, c), c),
                nn.SiLU(),
                nn.Conv1d(c, 2 * c, kernel_size=5, stride=4, padding=2),
                nn.GroupNorm(min(8, 2 * c), 2 * c),
                nn.SiLU(),
                nn.Conv1d(2 * c, 2 * c, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(min(8, 2 * c), 2 * c),
                nn.SiLU(),
            )
            self.dip_spatial = nn.Conv1d(2 * c, 16, kernel_size=1)
            spatial_dip_dim = 16 * (int(self.cfg.n_points) // 32)  # 16 * 16 = 256
            param_in_dim = 8 * c + spatial_b_dim + 4 * c + spatial_dip_dim
        else:
            self.dip_encoder = None
            self.dip_spatial = None
            param_in_dim = 8 * c + spatial_b_dim

        # Shared parameter MLP
        hidden_dim = 4 * c
        self.param_mlp = nn.Sequential(
            nn.Linear(param_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Sorted splittings head: inc = 10.0 * F.softplus(raw, beta=4.0) -> cumsum (MHz)
        # Scaling by 10 MHz matches L_split normalization (/ 10 MHz) and keeps logits O(1)
        # while smoothly allowing degenerate axes (df_1 -> 0) for negative raw logits.
        self.split_head = nn.Linear(hidden_dim, 4)
        with torch.no_grad():
            self.split_head.bias.copy_(torch.tensor([0.2, 0.8, 1.1, 1.5], dtype=torch.float32))

        # Auxiliary heads: d0_offset (MHz) and hwhm (MHz) scaled to O(1) logits
        self.d0_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.d0_head.bias)
        self.hwhm_head = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            self.hwhm_head.bias.fill_(1.2)

        # Physics-coupled uncertainty head
        if self.cfg.uncertainty_mode == "quantile":
            self.unc_head = nn.Linear(hidden_dim, 2)
            with torch.no_grad():
                self.unc_head.bias.fill_(-1.5)  # initial wings ~0.20 mT
        elif self.cfg.uncertainty_mode in ("laplace", "student_t"):
            self.unc_head = nn.Linear(hidden_dim, 1)
            with torch.no_grad():
                self.unc_head.bias.fill_(-2.0)  # initial sigma ~0.135 mT
        else:
            raise ValueError(f"Unknown uncertainty_mode: {self.cfg.uncertainty_mode}")

    def count_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(
        self,
        x_noisy: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x_in, vm, u_exp = prepare_model_inputs(
            x_noisy,
            valid_mask=valid_mask,
            use_ch3=self.cfg.use_ch3,
            u_coord=self.u_coord,
        )

        # Encoder
        s0 = self.stem(x_in)      # (B, C, 512)
        s1 = self.down1(s0)       # (B, 2C, 256)
        s2 = self.down2(s1)       # (B, 3C, 128)
        s3 = self.down3(s2)       # (B, 4C, 64)
        b_in = self.down4(s3)     # (B, 4C, 32)
        b_feat = self.bottleneck(b_in)  # (B, 4C, 32)

        pool_b = torch.cat(
            [
                b_feat.mean(dim=-1),
                b_feat.amax(dim=-1),
                self.bottleneck_spatial(b_feat).flatten(1),
            ],
            dim=-1,
        )

        if self.cfg.use_denoiser_head:
            d3 = F.interpolate(b_feat, size=s3.shape[-1], mode="linear", align_corners=False)
            d3 = self.up3_fuse(torch.cat([d3, s3], dim=1))
            d2 = F.interpolate(d3, size=s2.shape[-1], mode="linear", align_corners=False)
            d2 = self.up2_fuse(torch.cat([d2, s2], dim=1))
            d1 = F.interpolate(d2, size=s1.shape[-1], mode="linear", align_corners=False)
            d1 = self.up1_fuse(torch.cat([d1, s1], dim=1))
            d0 = F.interpolate(d1, size=s0.shape[-1], mode="linear", align_corners=False)
            d0 = self.up0_fuse(torch.cat([d0, s0], dim=1))

            z = self.denoise_out(d0).squeeze(1)  # (B, 512)
            dip_hat = 0.02 * F.softplus(z - 4.6)
            y_clean_hat = 1.0 - dip_hat
            dip_hat_masked = dip_hat * vm

            dip_for_param = dip_hat_masked.detach() if self.cfg.stop_grad_dip else dip_hat_masked
            dip_feat_in = torch.stack([dip_for_param / 0.02, u_exp, vm], dim=1)
            dip_enc = self.dip_encoder(dip_feat_in)
            pool_dip = torch.cat(
                [
                    dip_enc.mean(dim=-1),
                    dip_enc.amax(dim=-1),
                    self.dip_spatial(dip_enc).flatten(1),
                ],
                dim=-1,
            )
            h_in = torch.cat([pool_b, pool_dip], dim=-1)
        else:
            dip_hat = torch.zeros_like(vm)
            dip_hat_masked = dip_hat
            y_clean_hat = torch.ones_like(vm)
            h_in = pool_b

        h = self.param_mlp(h_in)

        # Strictly non-decreasing splittings 0 <= df_1 <= df_2 <= df_3 <= df_4 (MHz)
        raw_inc = self.split_head(h)
        inc = 10.0 * F.softplus(raw_inc, beta=4.0)
        splittings_hat = torch.cumsum(inc, dim=-1)

        d0_offset_hat = 3.0 * self.d0_head(h).squeeze(-1)
        hwhm_hat = 1.0 + 3.0 * F.softplus(self.hwhm_head(h).squeeze(-1))

        # Differentiable order-2 physics layer (internally float64)
        d0_total = D0_MHZ + d0_offset_hat
        b_mag_phys, p_long_mt = invert_splittings_order2_torch(splittings_hat, d0_mhz=d0_total)
        b_vec_canon = canonical_b_vec_from_splittings(
            splittings_hat, d0_mhz=d0_total, p_long_mt=p_long_mt
        )
        tetra_residual = compute_tetra_residual(
            splittings_hat, d0_mhz=d0_total, p_long_mt=p_long_mt
        )

        out: Dict[str, torch.Tensor] = {
            "dip_hat": dip_hat,
            "dip_hat_masked": dip_hat_masked,
            "y_clean_hat": y_clean_hat,
            "splittings_hat": splittings_hat,
            "d0_offset_hat": d0_offset_hat,
            "hwhm_hat": hwhm_hat,
            "b_mag_phys": b_mag_phys,
            "b_vec_canon": b_vec_canon,
            "p_long_mt": p_long_mt,
            "tetra_residual": tetra_residual,
        }

        if self.cfg.uncertainty_mode == "quantile":
            raw_wings = self.unc_head(h)
            delta_lo = F.softplus(raw_wings[:, 0]) + 1e-4
            delta_hi = F.softplus(raw_wings[:, 1]) + 1e-4
            q50 = b_mag_phys
            q10 = q50 - delta_lo
            q90 = q50 + delta_hi
            interval_width = delta_lo + delta_hi
            out["q10"] = q10
            out["q50"] = q50
            out["q90"] = q90
            out["uncertainty_mt"] = interval_width
            out["sigma_mt"] = interval_width / 2.5631031311  # approx Gaussian 80% -> 1-sigma eq
        else:
            log_sigma = torch.clamp(self.unc_head(h).squeeze(-1), min=-7.0, max=2.0)
            sigma = torch.exp(log_sigma)
            if self.cfg.uncertainty_mode == "laplace":
                # 80% central interval for Laplace(mu, b=sigma): +/- b * ln(1 / 0.2) = +/- 1.609438 * sigma
                half_80 = 1.6094379124341003 * sigma
            else:
                # 80% central interval for Student-t(df=4.0): t_{0.90, df=4} = 1.533206
                half_80 = 1.533206274 * sigma
            out["log_sigma"] = log_sigma
            out["sigma_mt"] = sigma
            out["q10"] = b_mag_phys - half_80
            out["q50"] = b_mag_phys
            out["q90"] = b_mag_phys + half_80
            out["uncertainty_mt"] = 2.0 * half_80

        return out
