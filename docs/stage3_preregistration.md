# OpenNV-Denoise Stage 3 Preregistration & Preflight Audit

**Preregistration Timestamp (UTC)**: `2026-11-21T08:56:00Z` (written strictly **before** Run 3a, Run 3b, or Run 3c training).  
**Immutability Policy**: The hypotheses, subsets, test statistics, multiple-comparison family, and verdict rules below MUST NOT be edited after training starts. Any post-hoc analyses or deviations must be appended as dated updates and explicitly labelled *exploratory*.

---

## 1. Part 0 Preflight Audit Summary (Measured Before Training)

### 1.1 Environment & Hardware
- **Python**: `3.14.2` (`win32`, AMD64) inside `.venv`.
- **PyTorch**: `2.15.0+cpu` (installed cleanly from `https://download.pytorch.org/whl/cpu`; `requirements.txt` updated with `torch>=2.0.0`).
- **Compute Device**: CPU-only (`torch.cuda.is_available() == False`), `4` logical CPU cores (`torch.set_num_threads(4)`).

### 1.2 Repository, Data & License Audit
- **Git Tags Verified**: `baseline-frozen`, `baseline-report-v2`, `stage2-data-v1`, `stage2-data-v2`, `stage2-data-v3`.
- **GPL-3.0 Hygiene**: Untracked all 18 `data/raw_experimental/trianam_*` files from git via `git rm --cached` and added `data/raw_experimental/trianam_*` to `.gitignore`, leaving them on local disk for local QA/evaluation while retaining only the download script (`src/real_data.py`), checksums (`data/real_eval_manifest.json`), and CC-BY-4.0 Zenodo `.txt` fixtures in git.
- **Frozen Asset Hash Manifest**: Created [`data/frozen_hashes.json`](../data/frozen_hashes.json) via [`src/integrity.py`](../src/integrity.py) covering `4` frozen source files (`src/baseline.py`, `src/physics.py`, `src/noise.py`, `tests/test_physics.py`) and `17` frozen synthetic `.npz` archives (`21/21` verified `ok: True`, `0` mismatches).
- **Locked Test Vault**: `test_id.npz` (`19,369,132` bytes) and `test_ood.npz` (`19,369,136` bytes) were NEVER opened (`data_read = 0` in [`data/synthetic/test_access.log`](../data/synthetic/test_access.log)).
- **Regime Label Definition in `val_id.npz`**: Verified empirically that `aux_regime_id` is defined strictly by **`aux_eff_snr_white`** (NOT `aux_eff_snr_total`):
  - `Hard` (`regime_id = 0`): `aux_eff_snr_white` $\in [0.5019, 1.4985]$ (`aux_eff_snr_total` $\in [0.2476, 1.4845]$).
  - `Medium` (`regime_id = 1`): `aux_eff_snr_white` $\in [1.5001, 2.9971]$ (`aux_eff_snr_total` $\in [0.2593, 2.9278]$).
  - `Easy` (`regime_id = 2`): `aux_eff_snr_white` $\in [3.0028, 19.9388]$ (`aux_eff_snr_total` $\in [0.2537, 15.7155]$).
- **`per_axis_contrast_jitter = 0.18` Distribution Check (`N = 1000`)**:
  - `max(w)/min(w)` quantiles: `P5 = 1.145, P10 = 1.198, P25 = 1.295, P50 = 1.431, P75 = 1.604, P90 = 1.791, P95 = 1.941, max = 2.665`.
  - `48.4%` of draws have `max(w)/min(w) >= 1.44` and `23.2%` fall directly in `[1.44, 1.60]`, covering the real `trianam` crystal asymmetry (`1.44x–1.60x`).

### 1.3 Frozen Fitter Inspection (`src/baseline.py`)
- Exact public signature of `fit_physical_odmr`:
  `(freqs, noisy_spectrum, valid_mask=None, warm_start_b_vec=None, dwell_time=None, *, single_axis=False, n_starts=3, rng=None, variant='plain', oracle_baseline=None, e_mhz=0.0, poly_degree=3, hwhm_bounds=(3.0, 9.0), eta=0.35)`.
- **Stage A Behavior**: Inside frozen `fit_physical_odmr`, `_run_varproj_pipeline` is called with `skip_stage_a=False` (default), so passing `warm_start_b_vec` with `n_starts=1` **does NOT skip Stage A coarse screening**; it injects `warm_start_b_vec` into `_stage_a_coarse_screen(..., top_k=1)` and polishes only `1` candidate in Stage B.
- **Measured Median Runtime (`N = 10` on `val_sel`)**:
  - Default (`n_starts=3, poly_degree=3`): **`244.8 ms/spectrum`**
  - Warm start (`warm_start_b_vec, n_starts=1, poly_degree=3`): **`212.8 ms/spectrum`**
  - Warm start (`warm_start_b_vec, n_starts=1, poly_degree=1`): **`181.3 ms/spectrum`**

### 1.4 Step-Cost Benchmark & CPU Training Budget
Measured on CPU (`PyTorch 2.15.0+cpu`, `bs = 32`):
- `base_channels = 24` (`632,529` params, `threads=2`): `313.0 ms/step`, single-spectrum inference `12.30 ms`.
- `base_channels = 32` (`1,120,873` params, `threads=2`): `465.4 ms/step`; **`threads=4`: `423.7 ms/step` (`75.5 spectra/s`)**, single-spectrum inference `~15–25 ms`.
- `base_channels = 48` (`2,513,817` params, `threads=2`): `1191.3 ms/step`.

**Selected Model Width**: **`base_channels = 32`** (`1,120,873` parameters) with `torch.set_num_threads(4)` and `batch_size = 32`.

**Documented CPU-Bounded Step & Wall-Clock Budgets**:
- **Run 3a (Sanity & Gate)**:
  - 64-sample fixed-batch overfit check: `200` optimizer steps (`bs = 64`, cap `3 min`). Gate: total loss drop `> 90%`.
  - Smoke training run: `250` optimizer steps (`bs = 32`, `8,000` spectra, cap `3 min`). Gate: no mode collapse.
- **Run 3b (Full Training & Ablations)**:
  - **`Model A` Seed 0** (Primary Strict-ID, `seed = 3001`, `stream_seed = 20260101`): **`1,600` optimizer steps** (`51,200` spectra, cap `16 min`).
  - **`Model A` Seeds 1 & 2** (`seed = 3002, 3003`, `stream_seed = 20260111, 20260112`): **`800` optimizer steps each** (`25,600` spectra each, cap `8 min` each).
  - **5 Ablations on `val_sel`** (1 seed each, `seed = 3001`): **`500` optimizer steps each** (`16,000` spectra each, cap `5 min` each), compared against `Model A` Seed 0 at matched step `500` on `val_sel`:
    1. `abl_no_denoiser` (`use_denoiser_head=False`)
    2. `abl_stop_grad_dip` (`stop_grad_dip=True`)
    3. `abl_no_ch3` (`use_ch3=False`)
    4. `abl_tetra_005` (`w_tetra=0.05` vs `0.0`)
    5. `abl_laplace_nll` (`uncertainty_mode="laplace"` vs `"quantile"`)
  - **`Model B` (Sim2Real Extension)**: warm-started from `Model A` Seed 0 (`seed = 3010`, `stream_seed = 20260120`), **`700` optimizer steps** (`22,400` spectra, cap `8 min`) with `per_axis_contrast_jitter=0.18`, `allow_high_field_in_window=True` (`|B| in [0.05, 4.60] mT`, dips in `[2760, 2980] MHz`), `hwhm_min_mhz=2.2`, `hwhm_max_mhz=9.0`, `mw_power_tilt_amp=0.12`, `mw_power_tilt_prob=0.70` (strictly retaining all OOD exclusions).

---

## 2. Disjoint Split Protocol

1. **`val_rep` (Reporting Only — `3,000` spectra)**: Exact indices in `val_baseline_ref.npz`: `0..999` (Hard, `1,000`), `1333..2332` (Medium, `1,000`), `2666..3665` (Easy, `1,000`). NEVER used for checkpoint selection, early stopping, loss weighting, smoother tuning, or hybrid threshold tuning.
2. **`val_sel` (Selection & Tuning Only — `1,000` spectra)**: Remaining strictly disjoint indices of `val_id.npz`: `1000..1332` (Hard, `333`), `2333..2665` (Medium, `333`), `3666..3999` (Easy, `334`). Used exclusively for all checkpoint selection, ablation decisions, Savitzky-Golay/Gaussian smoother tuning, and selective-hybrid threshold tuning.
3. **`val_id_natural` (`1,500`), `val_ood` (`4,000` / `1,500` ref), and all 9 `ood_*` splits (`1,000` each)**: Reporting ONLY.
4. **Locked Test Vault (`test_id.npz`, `test_ood.npz`)**: Strictly untouched (`data_read == 0`).
5. **Real `trianam` Split**:
   - **Tuning traces** (inspected for `Model B` knob verification only): `trianam_small_field_1`, `trianam_small_field_2`.
   - **Check traces** (reported strictly separately in Run 3c): `trianam_small_field_3`, `trianam_small_field_4`, `trianam_folder_2_101G`, `trianam_folder_3_48G_hf`, `trianam_folder_4_69G`.

---

## 3. Preregistered Hypotheses & Decision Rules (`M = 4`, Holm-Bonferroni at $\alpha = 0.05$)

All 4 primary hypotheses are evaluated once in Run 3c on **`val_rep`** using primary `Model A` (Seed 0, EMA weights). Multiple comparisons across the $M = 4$ primary hypotheses are controlled via the **Holm-Bonferroni step-down procedure** at family-wise $\alpha = 0.05$ (ordered significance thresholds $0.0125, 0.0167, 0.025, 0.05$). Every additional metric, slice, `Model B` result, and OOD check is explicitly labelled **exploratory**.

### H1 — Hard-Regime Wrong-Basin Reduction (`Model A` NN Direct vs. Frozen Physical Fitter)
- **Subset**: `val_rep` Hard regime (`aux_eff_snr_white in [0.5, 1.5)`, $N = 1,000$; frozen fitter reference wrong-basin rate = `55.4%`).
- **Metric**: Paired difference in wrong-basin rate (`||b_vec_pred - b_vec_true||_2 > 0.15 mT`), tested with exact/continuity-corrected **paired McNemar test** (`src.stats.mcnemar_paired_test`).
- **Verdict Criteria**:
  - **Supported**: `wb_rate(NN) < wb_rate(fitter)` AND Holm-adjusted McNemar $p_{\text{adj}} < 0.05$.
  - **Not supported**: `wb_rate(NN) >= wb_rate(fitter)` (or 95% paired CI excludes improvement).
  - **Inconclusive**: `wb_rate(NN) < wb_rate(fitter)` but Holm-adjusted $p_{\text{adj}} \ge 0.05$.

### H2 — Easy-Regime Non-Inferiority of `Model A` NN Direct (`X = 15.0 uT`)
- **Subset**: `val_rep` Easy regime (`aux_eff_snr_white in (3.0, 20.0]`, $N = 1,000$; frozen fitter reference median $|\Delta B| = 24.28\text{ uT}$, 95% CI `[21.70, 27.95] uT`).
- **Justification of Margin $X = 15.0\text{ uT}$**: The canonical 512-point frequency step is $\Delta f = 0.4697\text{ MHz}$, corresponding to $\Delta f / (2\gamma_e) = 8.38\text{ uT}$ per grid bin; `21.1%` of Easy samples are `weak_splitting`. A pre-specified non-inferiority margin of `15.0 uT` ($\approx 1.8$ frequency bins) bounds Easy median error below `~39 uT` while running $>10\times$ faster than nonlinear optimization.
- **Statistic**: Paired bootstrap (`n_boot = 4000`, `src.stats.paired_bootstrap_diff_ci`) on `delta_med = median(|ΔB|_NN) - median(|ΔB|_fitter)`, converted to a 1-sided bootstrap p-value for $H_0: \text{delta\_med} \ge +15.0\text{ uT}$.
- **Verdict Criteria**:
  - **Supported**: `delta_med < +15.0 uT` AND Holm-adjusted 1-sided bootstrap $p_{\text{adj}} < 0.05$ (equivalently upper confidence bound `< +15.0 uT`).
  - **Not supported**: Lower 95% CI bound of `delta_med >= +15.0 uT`.
  - **Inconclusive**: Paired CI straddles `+15.0 uT` (or $p_{\text{adj}} \ge 0.05$).

### H3 — Selective Hybrid Improves Tail Accuracy / Fail Rate at Lower Mean Total Runtime
- **Subset**: Deterministic paired stratified subset of `val_rep` ($N = 600$: first `200` indices of each regime in `val_baseline_ref.npz`: `0..199`, `1333..1532`, `2666..2865`), with gating thresholds `(unc_threshold_mt, tetra_threshold_mt)` tuned strictly on `val_sel`.
- **Metrics**: Mean total wall-clock runtime (ms/spec, including NN + any fallback fitter calls), P90 $|\Delta B|$ (paired bootstrap CI), and Failure Rate `> 100 uT` (paired McNemar test).
- **Verdict Criteria**:
  - **Supported**: `mean_runtime(selective_hybrid) < mean_runtime(fitter)` AND (`fail_100(hybrid) < fail_100(fitter)` with Holm-adjusted McNemar $p_{\text{adj}} < 0.05$ OR paired bootstrap 1-sided $p_{\text{adj}} < 0.05$ for `P90(hybrid) < P90(fitter)`).
  - **Not supported**: Mean runtime is not lower than frozen fitter alone, OR both P90 and fail rate are worse than or equal to the frozen fitter.
  - **Inconclusive**: Lower runtime and point improvement in P90/fail rate, but $p_{\text{adj}} \ge 0.05$.

### H4 — NN Denoiser Outperforms Tuned Savitzky-Golay in Dip-Region RMSE Without Dip-Position Degradation
- **Subset**: All `3,000` spectra of `val_rep`, comparing `Model A` (`y_clean_hat`) against Savitzky-Golay smoothing whose `(window_length, polyorder)` is tuned strictly on `val_sel` to minimize Dip-Region RMSE.
- **Metrics**:
  - `Dip-Region RMSE`: per-spectrum RMSE evaluated over valid points where `1.0 - Y_clean > 0.0015`.
  - `Dip-Position Error (MHz)`: per-spectrum mean absolute frequency offset (MHz) between the primary detected dip minimum of the denoised curve and `Y_clean`.
- **Verdict Criteria**:
  - **Supported**: Mean `Dip-Region RMSE (NN) < Dip-Region RMSE (SG)` with Holm-adjusted paired bootstrap $p_{\text{adj}} < 0.05$ AND `median_pos_err(NN) <= median_pos_err(SG) + 0.10 MHz`.
  - **Not supported**: `Dip-Region RMSE (NN) >= Dip-Region RMSE (SG)` OR `median_pos_err(NN) > median_pos_err(SG) + 0.10 MHz`.
  - **Inconclusive**: Point `Dip-Region RMSE (NN) < Dip-Region RMSE (SG)` with equal/better dip position, but $p_{\text{adj}} \ge 0.05$.

---

## 4. Dated Post-Preregistration Updates (Appended Only — Sections 1–3 Unchanged)

### Update 1 (`2026-11-21T13:15:00Z`) — Gate 3a Pass & Part 3 Gradient-Norm Calibration
- **Architecture & Loss Calibration (Part 3, before Run 3b)**:
  - During the initial 64-sample overfit diagnostic (Run 3a), pure global avg/max pooling in the parameter head lost frequency-axis position information and stalled at `81.13%` loss drop (`40.1x` gradient-norm imbalance between `L_bmag` and `L_denoise` due to `sigmoid(-4.6) ≈ 0.01` at `dip_hat` init).
  - Added compact spatial frequency-bin projections (`bottleneck_spatial: 8×32`, `dip_spatial: 16×16`) to the parameter head (`1,184,641` total parameters at `base_channels = 32`), scaled splitting increments (`10.0 * softplus(raw, beta=4.0)`), and calibrated loss weights via Part 3 gradient-norm inspection to `w_denoise = 5.0, w_split = 0.5, w_bmag = 0.8, w_aux = 0.15, w_tetra = 0.0` (all active gradient norms within `< 10x`).
- **Gate 3a Results (PASSED)**:
  - **64-Sample Overfit Check**: `init_loss = 4.11222 -> final_loss = 0.24319` (**`94.09%` drop > `90%`**, `87.98 s`, `median |ΔB| = 10.51 uT`, `P90 = 37.05 uT`, `fail_100 = 0.0%`, `dip_rmse = 0.001506`).
  - **Smoke Training (`smoke_3a`, `val_sel`)**: Loss decreased monotonically (`1.7091 -> 1.2541 -> 1.0818`), median `|ΔB|` dropped (`173.5 -> 120.0 -> 97.8 uT`), `cov80 = 88.8%`, `no_collapse_ok = True` (`dip_hat_std = 0.007625`, `dip_hat_max = 0.073895`, `splittings_std = 27.207 MHz`, `unc_median = 674.36 uT`).
- **Hypotheses `H1–H4`, subsets (`val_rep`, `val_sel`), test statistics, and verdict criteria in Sections 2–3 remain completely unchanged.**
