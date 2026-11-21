"""Stage 3 selective hybrid predictor combining OpenNVResUNet1D with frozen src/baseline.py."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.baseline import fit_physical_odmr
from src.config import CANONICAL_FREQS
from src.model_cnn import ModelConfig, OpenNVResUNet1D


@dataclass
class SelectiveHybridConfig:
    """Thresholds tuned strictly on val_sel for selective fallback to frozen fit_physical_odmr."""

    unc_threshold_mt: float = 0.12
    tetra_threshold_mt: float = 0.08
    disagreement_threshold_mt: float = 0.30
    fallback_mode: str = "hybrid_warm_start"  # "hybrid_warm_start" or "physical_plain"
    fallback_n_starts: int = 3
    fallback_poly_degree: int = 3
    hwhm_bounds: Tuple[float, float] = (3.0, 9.0)


class OpenNVPredictor:
    """Unified Stage 3 inference wrapper supporting NN-direct and hybrid modes.

    Calls frozen ``src.baseline.fit_physical_odmr`` strictly untouched.
    """

    def __init__(
        self,
        model: OpenNVResUNet1D,
        device: Optional[torch.device] = None,
        hybrid_cfg: Optional[SelectiveHybridConfig] = None,
    ) -> None:
        self.device = device if device is not None else torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.hybrid_cfg = hybrid_cfg if hybrid_cfg is not None else SelectiveHybridConfig()

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        device: Optional[torch.device] = None,
        use_ema: bool = True,
        hybrid_cfg: Optional[SelectiveHybridConfig] = None,
    ) -> "OpenNVPredictor":
        dev = device if device is not None else torch.device("cpu")
        payload = torch.load(ckpt_path, map_location=dev, weights_only=False)
        m_cfg = ModelConfig(**payload["model_config"])
        model = OpenNVResUNet1D(m_cfg)
        state = payload["ema_state_dict"] if (use_ema and "ema_state_dict" in payload) else payload["model_state_dict"]
        model.load_state_dict(state)
        return cls(model=model, device=dev, hybrid_cfg=hybrid_cfg)

    @torch.no_grad()
    def predict_nn_batch(
        self,
        x_noisy: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        batch_size: int = 128,
    ) -> Dict[str, np.ndarray]:
        """Run batched ``nn_direct`` inference on a NumPy array ``(N, 512)``."""
        x_arr = np.asarray(x_noisy, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        n = x_arr.shape[0]
        if valid_mask is None:
            vm_arr = np.ones_like(x_arr, dtype=bool)
        else:
            vm_arr = np.asarray(valid_mask, dtype=bool)
            if vm_arr.ndim == 1:
                vm_arr = vm_arr[None, :]

        outs: Dict[str, List[np.ndarray]] = {}
        t0 = time.perf_counter()
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.from_numpy(x_arr[start:end]).to(self.device)
            vmb = torch.from_numpy(vm_arr[start:end]).to(self.device)
            pred = self.model(xb, vmb)
            for k, v in pred.items():
                outs.setdefault(k, []).append(v.detach().cpu().numpy())
        total_ms = (time.perf_counter() - t0) * 1000.0
        stacked = {k: np.concatenate(v_list, axis=0) for k, v_list in outs.items()}
        stacked["nn_runtime_ms_per_spec"] = np.full(n, total_ms / max(n, 1), dtype=np.float32)
        return stacked

    @torch.no_grad()
    def predict_single(
        self,
        noisy_spectrum: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        mode: str = "nn_direct",
        freqs: np.ndarray = CANONICAL_FREQS,
        hwhm_bounds: Optional[Tuple[float, float]] = None,
        n_starts: Optional[int] = None,
        poly_degree: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Predict on a single spectrum in one of four modes.

        Modes:
          - ``"nn_direct"``
          - ``"hybrid_warm_start"``
          - ``"hybrid_denoised_fit"``
          - ``"selective_hybrid"``
        """
        x_np = np.asarray(noisy_spectrum, dtype=np.float32)
        vm_np = np.ones_like(x_np, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)

        t_nn0 = time.perf_counter()
        xb = torch.from_numpy(x_np[None, :]).to(self.device)
        vmb = torch.from_numpy(vm_np[None, :]).to(self.device)
        pred_t = self.model(xb, vmb)
        nn_ms = (time.perf_counter() - t_nn0) * 1000.0

        y_clean_hat = pred_t["y_clean_hat"][0].detach().cpu().numpy().astype(np.float64)
        splittings_hat = pred_t["splittings_hat"][0].detach().cpu().numpy().astype(np.float64)
        b_mag_nn = float(pred_t["b_mag_phys"][0].item())
        b_vec_nn = pred_t["b_vec_canon"][0].detach().cpu().numpy().astype(np.float64)
        unc_mt = float(pred_t["uncertainty_mt"][0].item())
        sigma_mt = float(pred_t["sigma_mt"][0].item())
        q10 = float(pred_t["q10"][0].item())
        q90 = float(pred_t["q90"][0].item())
        tetra_res = float(pred_t["tetra_residual"][0].item())
        d0_off_hat = float(pred_t["d0_offset_hat"][0].item())
        hwhm_hat = float(pred_t["hwhm_hat"][0].item())

        result: Dict[str, Any] = {
            "mode": mode,
            "y_clean_hat": y_clean_hat,
            "splittings_hat": splittings_hat,
            "b_mag_nn": b_mag_nn,
            "b_vec_nn": b_vec_nn,
            "b_mag": b_mag_nn,
            "b_vec": b_vec_nn,
            "uncertainty_mt": unc_mt,
            "sigma_mt": sigma_mt,
            "q10": q10,
            "q90": q90,
            "tetra_residual_mt": tetra_res,
            "d0_offset_hat": d0_off_hat,
            "hwhm_hat": hwhm_hat,
            "used_fitter": False,
            "undetermined": False,
            "nn_runtime_ms": nn_ms,
            "fitter_runtime_ms": 0.0,
            "total_runtime_ms": nn_ms,
        }

        if mode == "nn_direct":
            return result

        hw_b = hwhm_bounds if hwhm_bounds is not None else self.hybrid_cfg.hwhm_bounds

        if mode == "hybrid_warm_start":
            ns = int(n_starts) if n_starts is not None else int(self.hybrid_cfg.fallback_n_starts)
            pd = int(poly_degree) if poly_degree is not None else int(self.hybrid_cfg.fallback_poly_degree)
            t_f0 = time.perf_counter()
            fit = fit_physical_odmr(
                freqs,
                x_np.astype(np.float64),
                valid_mask=vm_np,
                warm_start_b_vec=b_vec_nn,
                n_starts=ns,
                variant="plain",
                poly_degree=pd,
                hwhm_bounds=hw_b,
            )
            fit_ms = (time.perf_counter() - t_f0) * 1000.0
            result["b_mag"] = float(fit["b_mag"])
            result["b_vec"] = np.asarray(fit["b_vec"], dtype=np.float64)
            result["used_fitter"] = True
            result["fitter_status"] = str(fit["status"])
            result["fitter_runtime_ms"] = fit_ms
            result["total_runtime_ms"] = nn_ms + fit_ms
            return result

        if mode == "hybrid_denoised_fit":
            ns = int(n_starts) if n_starts is not None else 1
            pd = int(poly_degree) if poly_degree is not None else 1
            t_f0 = time.perf_counter()
            fit = fit_physical_odmr(
                freqs,
                y_clean_hat,
                valid_mask=vm_np,
                warm_start_b_vec=b_vec_nn,
                n_starts=ns,
                variant="plain",
                poly_degree=pd,
                hwhm_bounds=hw_b,
            )
            fit_ms = (time.perf_counter() - t_f0) * 1000.0
            result["b_mag"] = float(fit["b_mag"])
            result["b_vec"] = np.asarray(fit["b_vec"], dtype=np.float64)
            result["used_fitter"] = True
            result["fitter_status"] = str(fit["status"])
            result["fitter_runtime_ms"] = fit_ms
            result["total_runtime_ms"] = nn_ms + fit_ms
            return result

        if mode == "selective_hybrid":
            is_uncertain = (
                (unc_mt > self.hybrid_cfg.unc_threshold_mt)
                or (tetra_res > self.hybrid_cfg.tetra_threshold_mt)
            )
            if not is_uncertain:
                return result

            ns = int(n_starts) if n_starts is not None else int(self.hybrid_cfg.fallback_n_starts)
            pd = int(poly_degree) if poly_degree is not None else int(self.hybrid_cfg.fallback_poly_degree)
            ws_vec = b_vec_nn if self.hybrid_cfg.fallback_mode == "hybrid_warm_start" else None
            t_f0 = time.perf_counter()
            fit = fit_physical_odmr(
                freqs,
                x_np.astype(np.float64),
                valid_mask=vm_np,
                warm_start_b_vec=ws_vec,
                n_starts=ns,
                variant="plain",
                poly_degree=pd,
                hwhm_bounds=hw_b,
            )
            fit_ms = (time.perf_counter() - t_f0) * 1000.0
            b_fit = float(fit["b_mag"])
            result["b_mag"] = b_fit
            result["b_vec"] = np.asarray(fit["b_vec"], dtype=np.float64)
            result["used_fitter"] = True
            result["fitter_status"] = str(fit["status"])
            result["undetermined"] = bool(abs(b_fit - b_mag_nn) > self.hybrid_cfg.disagreement_threshold_mt)
            result["fitter_runtime_ms"] = fit_ms
            result["total_runtime_ms"] = nn_ms + fit_ms
            return result

        raise ValueError(f"Unknown prediction mode: {mode}")


def tune_selective_hybrid_thresholds(
    nn_b_mag: np.ndarray,
    nn_unc_mt: np.ndarray,
    nn_tetra_mt: np.ndarray,
    fit_b_mag: np.ndarray,
    b_true_mt: np.ndarray,
    nn_ms_per_spec: float,
    fit_ms_per_spec: float,
    max_fallback_frac: float = 0.50,
) -> Dict[str, Any]:
    """Grid-search selective-hybrid gating thresholds strictly on val_sel.

    Returns the selected ``(unc_threshold_mt, tetra_threshold_mt)`` minimizing a joint
    P90 + failure-rate objective while respecting ``fallback_frac <= max_fallback_frac``,
    along with the full accuracy-vs-coverage Pareto curve.
    """
    err_nn_ut = np.abs(nn_b_mag - b_true_mt) * 1000.0
    err_fit_ut = np.abs(fit_b_mag - b_true_mt) * 1000.0

    # Sweep quantile grids over NN uncertainty and tetra_residual
    q_grid_unc = [50, 60, 70, 75, 80, 85, 90, 95, 100]
    q_grid_tet = [70, 80, 90, 95, 100]

    best_score = float("inf")
    best_cfg: Dict[str, Any] = {}
    curve_points: List[Dict[str, float]] = []

    for q_u in q_grid_unc:
        t_u = float(np.percentile(nn_unc_mt, q_u)) if q_u < 100 else float("inf")
        for q_t in q_grid_tet:
            t_t = float(np.percentile(nn_tetra_mt, q_t)) if q_t < 100 else float("inf")
            use_fit = (nn_unc_mt > t_u) | (nn_tetra_mt > t_t)
            fb_frac = float(np.mean(use_fit))
            err_sel = np.where(use_fit, err_fit_ut, err_nn_ut)
            med_ut = float(np.median(err_sel))
            p90_ut = float(np.percentile(err_sel, 90))
            fail_pct = float(np.mean(err_sel > 100.0) * 100.0)
            mean_ms = nn_ms_per_spec + fb_frac * fit_ms_per_spec

            if q_t == 100:
                curve_points.append(
                    {
                        "nn_coverage_pct": round((1.0 - fb_frac) * 100.0, 1),
                        "fallback_pct": round(fb_frac * 100.0, 1),
                        "unc_thresh_mt": round(t_u, 5) if np.isfinite(t_u) else 999.0,
                        "median_ut": round(med_ut, 2),
                        "p90_ut": round(p90_ut, 2),
                        "fail_100_pct": round(fail_pct, 2),
                        "mean_runtime_ms": round(mean_ms, 2),
                    }
                )

            if fb_frac <= max_fallback_frac + 1e-9:
                # Objective balances tail accuracy (P90) and failure rate (> 100 uT)
                score = p90_ut + 5.0 * fail_pct
                if score < best_score:
                    best_score = score
                    best_cfg = {
                        "unc_threshold_mt": float(t_u),
                        "tetra_threshold_mt": float(t_t),
                        "q_unc": q_u,
                        "q_tetra": q_t,
                        "fallback_pct": round(fb_frac * 100.0, 2),
                        "val_sel_median_ut": round(med_ut, 2),
                        "val_sel_p90_ut": round(p90_ut, 2),
                        "val_sel_fail_100_pct": round(fail_pct, 2),
                        "val_sel_mean_runtime_ms": round(mean_ms, 2),
                    }

    return {"best": best_cfg, "coverage_curve": curve_points}
