# OpenNV-Denoise Stage 2 Dataset Card & Specification

This document defines the mathematical formulas, noise/artifact generative models, split support intervals, seed derivation rules, and schema for the **OpenNV-Denoise Stage 2** synthetic and experimental evaluation datasets.

---

## 1. Physical Constants & Canonical Grid

All spectra are represented on the canonical 512-point microwave frequency sweep grid:
- `FREQ_MIN_MHZ = 2750.0`, `FREQ_MAX_MHZ = 2990.0`, `N_POINTS = 512` ($\Delta f = \frac{240}{511} \approx 0.46967\text{ MHz}$).
- `D0_MHZ = 2870.0` MHz (zero-field splitting reference).
- `GAMMA_E_MHZ_MT = 28.025` MHz/mT (electron gyromagnetic ratio $\gamma_e$).
- `A_HYPERFINE_MHZ = 2.16` MHz ($^{14}\text{N}$ nuclear spin $I=1$ hyperfine triplet splitting, $m_I \in \{-1, 0, +1\}$).
- `e_mhz = 0.0` MHz (zero transverse strain $E = 0$; stored in every sample's metadata as `aux_e_mhz = 0.0`).
- `CENTRAL_REQUIRED_MIN_MHZ = 2830.0`, `CENTRAL_REQUIRED_MAX_MHZ = 2910.0` MHz (minimum central window coverage required for PCHIP resampling).

---

## 2. Core Signal, SNR, and Cleanliness Definitions (Part 1)

### 2.1 Outer Single-Axis Dip Depth (`depth_outer`) & `weak_splitting`
Given a 3D magnetic field vector $\mathbf{B} \in \mathbb{R}^3$ (in mT), zero-field splitting $D_0 = 2870.0 + \Delta D_0$, linewidth $\gamma = \text{HWHM}$ (in MHz), base optical contrast $C$, and pseudo-Voigt mixing parameter $\eta \in [0, 1]$:

1. **Per-Axis Field Projections**:
   For each tetrahedral $[111]$ NV axis $\mathbf{n}_k$ ($k \in \{0, 1, 2, 3\}$):
   $$|B_{\parallel, k}| = |\mathbf{B} \cdot \mathbf{n}_k|, \qquad B_{\perp, k} = \sqrt{\max\!\left(0,\; \|\mathbf{B}\|_2^2 - |B_{\parallel, k}|^2\right)}$$

2. **Exact Spin-1 Transition Frequencies & Contrast Factor**:
   Diagonalizing the $3 \times 3$ spin-1 ground-state Hamiltonian via `solve_axis_transitions(b_par_k, b_perp_k, d0=D0, e_mhz=0.0)` yields transition frequencies $(f_{-, k}, f_{+, k})$ and transverse spin-mixing contrast factor $c_k = |\langle m_s = 0 | \psi_0^{(k)} \rangle|^4 \in (0, 1]$.

3. **Single-Axis Hyperfine Triplet Dip Depth ($d_k$)**:
   Each transition branch of axis $k$ in the 4-axis ensemble (`axis_weight = 0.28`) consists of 3 hyperfine lines of amplitude $a_k = \frac{0.28 \cdot C \cdot c_k}{3}$ at $f_{-, k} + m_I A_{\parallel}$ ($m_I \in \{-1, 0, +1\}$):
   $$D_{k, -}(f) = \sum_{m_I \in \{-1, 0, +1\}} \text{pseudo\_voigt\_dip}\!\left(f,\; f_{-, k} + m_I A_{\parallel},\; a_k,\; \gamma,\; \eta\right)$$
   Evaluating on `CANONICAL_FREQS` (never summing $f_{-, k}$ and $f_{+, k}$, so the merged central dip at $D_0$ is never used):
   $$d_k = \max_{f \in \text{CANONICAL\_FREQS}} D_{k, -}(f) > 0$$

4. **`depth_outer` Selection Rule & Fallback**:
   Let $\mathcal{K}_{\text{non-deg}} = \{k \in \{0, 1, 2, 3\} : |B_{\parallel, k}| > 0.15\text{ mT}\}$.
   - **Standard Rule** ($\mathcal{K}_{\text{non-deg}} \ne \emptyset$):
     $$\text{depth\_outer} = \max_{k \in \mathcal{K}_{\text{non-deg}}} d_k$$
   - **Fallback Rule** ($\mathcal{K}_{\text{non-deg}} = \emptyset$, which can occur for $\|\mathbf{B}\|_2 < 0.15\sqrt{3} \approx 0.2598\text{ mT}$ since $\max_k |B_{\parallel, k}| \ge \|\mathbf{B}\|_2 / \sqrt{3}$):
     $$k^* = \arg\max_{k \in \{0, 1, 2, 3\}} |B_{\parallel, k}|, \qquad \text{depth\_outer} = d_{k^*} > 0$$
   In all cases, $\text{depth\_outer} > 0$.

5. **`weak_splitting` Flag**:
   $$\text{weak\_splitting} = \left(2 \cdot \gamma_e \cdot \max_{k \in \{0,1,2,3\}} |B_{\parallel, k}| < 2 \cdot \text{HWHM}\right)$$
   When `weak_splitting == True`, the $f_-$ and $f_+$ transition branches of even the most strongly projected NV axis are separated by less than one full-width at half-maximum ($\text{FWHM} = 2 \cdot \text{HWHM}$), so $|\mathbf{B}|$ Fisher information is intrinsically limited by line overlap regardless of SNR.

---

### 2.2 SNR & Cleanliness Axes
- **`sigma_white_eff`**: Standard deviation of the injected zero-mean stochastic (`white + pink`) component evaluated on the measurement's own simulated sampling grid `freqs_sim`:
  $$\sigma_{\text{white\_eff}} = \text{std}\!\left(\text{white}_{\text{sim}} + \text{pink}_{\text{sim}}\right)$$
- **`eff_snr_white`** (Primary Regime Axis):
  $$\text{eff\_snr\_white} = \frac{\text{depth\_outer}}{\sigma_{\text{white\_eff}}}$$
- **`eff_snr_total`** (Total Effective Outer-Dip SNR including Baseline Drift & Hum):
  $$\text{eff\_snr\_total} = \frac{\text{depth\_outer}}{\sqrt{\sigma_{\text{white\_eff}}^2 + \text{RMS}(\text{drift})^2 + \text{RMS}(\text{hum})^2}}$$
  where $\text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{j=1}^N x_j^2}$ on the canonical grid.
- **`dirt_drift` & `dirt_hum`** (Cleanliness Axes):
  $$\text{dirt\_drift} = \frac{\text{RMS}(\text{drift})}{\text{depth\_outer}}, \qquad \text{dirt\_hum} = \frac{\text{RMS}(\text{hum})}{\text{depth\_outer}}$$
- **`nominal_snr_legacy`** (Stage 1.5 Compatibility Metric):
  $$\text{nominal\_snr\_legacy} = \frac{\max(Y_{\text{clean}}) - \min(Y_{\text{clean}})}{\sigma_{\text{white\_eff}}}$$

---

### 2.3 Regime Tiers & Stratification
Samples are categorized into three regimes strictly by `eff_snr_white`:
1. **Hard (`regime_id = 0`)**: $\text{eff\_snr\_white} \in [0.5, 1.5)$
2. **Medium (`regime_id = 1`)**: $\text{eff\_snr\_white} \in [1.5, 3.0]$
3. **Easy (`regime_id = 2`)**: $\text{eff\_snr\_white} \in (3.0, 20.0]$

- **Natural Log-Uniform Proportions** (`train`, `train_snapshot`, `ood_*` single-factor sets where $\text{eff\_snr\_white} \sim \text{LogUniform}(0.5, 20.0)$):
  $$P_{\text{nat}}(\text{Hard}) = \frac{\ln(1.5/0.5)}{\ln(20.0/0.5)} \approx 0.29788, \quad P_{\text{nat}}(\text{Medium}) = \frac{\ln(3.0/1.5)}{\ln(20.0/0.5)} \approx 0.18790, \quad P_{\text{nat}}(\text{Easy}) = \frac{\ln(20.0/3.0)}{\ln(20.0/0.5)} \approx 0.51422$$
- **Stratified Evaluation Splits** (`val_id`, `val_ood`, `test_id`, `test_ood`, $N = 4000$ each):
  Stratified into exact thirds: $1333$ Hard (`regime_id = 0`), $1333$ Medium (`regime_id = 1`), and $1334$ Easy (`regime_id = 2`), with `eff_snr_white` drawn log-uniformly inside each tier's interval. Importance weights $w_r = P_{\text{nat}}(r) / (N_r / N)$ restore the natural training distribution when computing overall unstratified validation metrics.

---

### 2.4 Noise Budget Construction & Clean-Tract Subpopulation (No Rejection Loops)
1. Given `depth_outer` and target `eff_snr_white`, set:
   $$\sigma_{\text{white\_eff}} = \frac{\text{depth\_outer}}{\text{eff\_snr\_white}}$$
   The stochastic (`white + pink`) component is scaled on `freqs_sim` to have exact standard deviation $\sigma_{\text{white\_eff}}$. Because the white-noise variance fraction is $1 - r_{\text{pink}} \ge 0.5$, shot-like white noise never vanishes.
2. With probability $20\%$, the sample is drawn from the **Clean Tract (`tract_clean = 1`)**:
   - Drift peak amplitude: $A_{\text{drift}} \sim \mathcal{U}(0.0,\; 0.3 \cdot \text{depth\_outer})$.
   - Mains hum amplitude: $A_{\text{hum}} = 0$ with probability $20\%$; otherwise $A_{\text{hum}} \sim \mathcal{U}(0.0,\; 0.3 \cdot \text{depth\_outer})$.
3. With probability $80\%$, the sample is drawn from the **Standard COTS Tract (`tract_clean = 0`)**:
   - Drift peak amplitude: $A_{\text{drift}} \sim \mathcal{U}(0.002,\; 0.020)$ (independent of `depth_outer` and `eff_snr_white`).
   - Mains hum amplitude: $A_{\text{hum}} = 0$ with probability $20\%$; otherwise $A_{\text{hum}} \sim \mathcal{U}(0.0,\; 0.006)$.

---

### 2.5 Degeneracy Definitions: Sampling vs. Metadata Labelling (Part 1.7)
Two distinct degeneracy criteria are used for two distinct purposes:
1. **Direction Sampler Near-Degenerate Branch (Angular Criterion)**:
   During 3D direction sampling on $S^2$, $20\%$ of samples are drawn from a near-degenerate angular neighborhood satisfying:
   $$\min_{k \in \{0, 1, 2, 3\}} |\cos \theta_k| = \min_{k} \frac{|\mathbf{B} \cdot \mathbf{n}_k|}{\|\mathbf{B}\|_2} < 0.05$$
   (recorded in `aux_sampler_near_degenerate` and `aux_min_abs_cos_theta`). This is scale-invariant with respect to $|\mathbf{B}|$ and ensures geometric coverage near the perpendicular planes $\mathbf{B} \cdot \mathbf{n}_k = 0$.
2. **Metadata `axis_degenerate` Flag (Physical Zeeman Criterion)**:
   In the auxiliary metadata (`aux_axis_degenerate`, shape `(N, 4)`), axis $k$ is flagged as physically degenerate when:
   $$|B_{\parallel, k}| < 0.15\text{ mT}$$
   ($2 \gamma_e |B_{\parallel, k}| < 8.41\text{ MHz}$, where the $f_{-, k}$ and $f_{+, k}$ transitions merge into the central $D_0$ region). At weak field magnitudes ($|\mathbf{B}| < 0.26\text{ mT}$), multiple axes satisfy $|B_{\parallel, k}| < 0.15\text{ mT}$ even for general non-perpendicular angles.

---

## 3. Noise, Drift, Hum, and Grid Realism Models (`src/noise_models.py`, Part 2)

### 3.1 Exact Additive Component Identity
For every sample in `float64`, the canonical 512-point arrays satisfy:
$$Y_{\text{clean}} + \text{drift} + \text{hum} + \text{pink} + \text{white} + \text{other} = X_{\text{noisy, f64}}$$
to $< 10^{-12}$ max absolute error (and $< 10^{-6}$ after casting `X_noisy` and `Y_clean` to `float32`).
Here `other` encapsulates slow multiplicative gain modulation plus the PCHIP resampling/grid-perturbation delta when grid realism is active.

### 3.2 Baseline Drift Families (`drift_family_id`)
Evaluated on normalized sweep coordinate $u \in [-1, 1]$ across `freqs_sim` and scaled to peak absolute amplitude $A_{\text{drift}}$:
- **`F1` (`id = 1`, ID)**: Orthogonal Legendre polynomial of degree $\le 3$:
  $$d_{\text{raw}}(u) = c_1 u + c_2 \left(u^2 - \tfrac{1}{3}\right) + c_3 \left(u^3 - \tfrac{3}{5} u\right), \quad c_1, c_2 \sim \mathcal{N}(0, 0.5^2),\; c_3 \sim \mathcal{N}(0, 0.3^2)$$
- **`F2` (`id = 2`, ID)**: Sum of $K \in \{1, 2, 3\}$ low-frequency sinusoids with periods $T_m \sim \mathcal{U}(0.5, 4.0) \times \text{window}$ and random phases $\phi_m \sim \mathcal{U}(0, 2\pi)$.
- **`F3` (`id = 3`, OOD ONLY)**: Exponential thermal warm-up/decay plus linear ramp on normalized time $\tau \in [0, 1]$ across the sweep:
  $$d_{\text{raw}}(\tau) = s_1 \left(1 - e^{-\tau / \tau_0}\right) + s_2 \tau, \quad \tau_0 \sim \mathcal{U}(0.12, 0.45),\; s_1 \in \{-1, +1\},\; s_2 \sim \mathcal{U}(-0.4, 0.4)$$
- **`F4` (`id = 4`, ID)**: Smooth Gaussian-process-like random curve generated by convolving white Gaussian noise with a Gaussian kernel of width $\sigma_{\text{corr}} \sim \mathcal{U}(0.12, 0.35) \times M$ points and removing the mean.
- **`F5_id` (`id = 5`, ID)**: Equal-weighted normalized sum of `F1` and `F2`, rescaled to peak $A_{\text{drift}}$.
- **`F5_ood` (`id = 6`, OOD ONLY)**: Equal-weighted normalized sum of `F3` and `F4`, rescaled to peak $A_{\text{drift}}$.

**Strict Split Invariant**: `train`, `train_snapshot`, `val_id`, `test_id`, and non-drift single-factor OOD sets sample strictly from `{F1, F2, F4, F5_id}` (`id` $\in \{1, 2, 4, 5\}$). `F3` (`id=3`) and `F5_ood` (`id=6`) appear exclusively in `val_ood`, `test_ood`, and `ood_drift`.

### 3.3 Time-Domain AC Mains Hum
- Base frequency $f_{\text{mains}} \in \{50.0, 60.0\}\text{ Hz}$ (50/50 probability) plus jitter $\delta f \sim \mathcal{U}(-0.5, +0.5)\text{ Hz}$.
- Dwell time per step $dt \sim \mathcal{U}(0.3, 2.0)\text{ ms}$, time vector $t_k = k \cdot (dt \times 10^{-3})\text{ s}$.
- Fundamental + optional 2nd and 3rd harmonics with independent phases $\phi_1, \phi_2, \phi_3 \sim \mathcal{U}(0, 2\pi)$ and relative harmonic ratios $r_2, r_3$:
  - **ID Splits**: $r_2, r_3 \sim \mathcal{U}(0.0, 0.50]$ (each active with $65\%$ probability, else $0.0$).
  - **OOD Hum Splits (`val_ood`, `test_ood`, `ood_hum`)**: when hum is present ($A_{\text{hum}} > 0$), $r_2, r_3 \sim \mathcal{U}(0.5001, 0.85]$ (strictly $> 0.50$).
- Waveform:
  $$h_{\text{raw}}(t_k) = \sin(2\pi f t_k + \phi_1) + r_2 \sin(4\pi f t_k + \phi_2) + r_3 \sin(6\pi f t_k + \phi_3)$$
  scaled so that $\max_k |h(t_k)| = A_{\text{hum}}$ (and $A_1, A_2, A_3$ recorded in `aux_hum_harmonics_amp`).

### 3.4 Stochastic Noise & Slow Multiplicative Gain Fluctuation
- **Intensity Modulation**: $s_k = \sqrt{\text{clip}(S_{\text{clean, sim}}(f_k), 0.5, 1.5)}$.
- **Pink Ratio**: $r_{\text{pink}} \sim \mathcal{U}(0.0, 0.50)$.
- **White Innovation (`noise_type`)**:
  - **ID (`noise_type = 0`)**: $w_{\text{unit}} \sim \mathcal{N}(0, 1)$, zero-mean shifted.
  - **OOD (`noise_type = 1`, in `val_ood`, `test_ood`, `ood_noise`)**: Heavy-tailed Student-$t$ ($\nu = 3.5$) normalized by $\sqrt{\nu / (\nu - 2)}$ plus $n_{\text{spikes}} \in \{2, 3, 4\}$ single-point impulse spikes of magnitude $\pm \mathcal{U}(4.5, 8.0)$.
- **Exact Variance Normalization**:
  Let $w_{\text{raw}} = s \cdot \sqrt{1 - r_{\text{pink}}} \cdot w_{\text{unit}}$ (mean-centered) and $p_{\text{raw}} = s \cdot \sqrt{r_{\text{pink}}} \cdot p_{\text{unit}}$ (mean-centered). Both components are scaled by a common factor $\alpha = \sigma_{\text{white\_eff}} / \text{std}(w_{\text{raw}} + p_{\text{raw}})$ so that $\text{std}(\text{white}_{\text{sim}} + \text{pink}_{\text{sim}}) \equiv \sigma_{\text{white\_eff}}$ identically.
- **Slow Multiplicative Gain Fluctuation**:
  $g(u) = g_{\text{amp}} \cdot \frac{a_1 u + a_2 (u^2 - 1/3)}{\max |a_1 u + a_2 (u^2 - 1/3)|}$ with $g_{\text{amp}} \sim \mathcal{U}(0.0, 0.005)$, producing $\text{gain\_mod}_{\text{sim}} = g(u) \cdot (S_{\text{clean, sim}} + \text{drift}_{\text{sim}} + \text{hum}_{\text{sim}} + \text{pink}_{\text{sim}} + \text{white}_{\text{sim}})$.

### 3.5 Noise Placement & Grid Realism
All noise and artifact components are synthesized on the measurement's **simulated grid (`freqs_sim`)**, and the composite noisy spectrum is then resampled onto `CANONICAL_FREQS` via `resample_spectrum_pchip`.
- **`grid_variant_id = 0` (Canonical Unperturbed)**: $70\%$ of ID samples, $60\%$ of OOD samples (`0%` in `ood_grid`). `freqs_sim = CANONICAL_FREQS` ($512$ points).
- **`grid_variant_id = 1` (Point-Count & Calibration Offset/Stretch Perturbation)**: $15\%$ of ID, $20\%$ of OOD (`0%` in `ood_grid`). Full span $[2750, 2990]\text{ MHz}$ with random point count $M \in [200, 1001]$ and small linear calibration shift/stretch $\delta f \in [-0.3, +0.3]\text{ MHz}$ clamped so $[f_{\min}, f_{\max}] \supseteq [2830, 2910]\text{ MHz}$.
- **`grid_variant_id = 2` (Range Crop + Point-Count + Offset/Stretch)**: $15\%$ of ID, $20\%$ of OOD (`100%` in `ood_grid`).
  - For ID / `val_ood` / `test_ood`: $f_{\min} \sim \mathcal{U}(2760.0, 2828.0)\text{ MHz}$, $f_{\max} \sim \mathcal{U}(2912.0, 2980.0)\text{ MHz}$, $M \in [200, 1001]$, offset/stretch $\in [-0.3, +0.3]\text{ MHz}$ covering $[2830, 2910]\text{ MHz}$.
  - For `ood_grid` (aggressive crop): $100\%$ perturbed with $f_{\min} \sim \mathcal{U}(2790.0, 2828.0)\text{ MHz}$, $f_{\max} \sim \mathcal{U}(2912.0, 2950.0)\text{ MHz}$, $M \in [200, 1001]$, offset/stretch $\in [-0.3, +0.3]\text{ MHz}$ covering $[2830, 2910]\text{ MHz}$.

---

## 4. Split Protocol & Strictly Disjoint Support Table (Part 3)

| Parameter / Axis | ID Distribution (`train`, `val_id`, `test_id`) | Combined OOD (`val_ood`, `test_ood`) | Single-Factor OOD Diagnostic Sets (`1000` samples each, val only) | Support Overlap (ID vs. OOD Axis) |
| :--- | :--- | :--- | :--- | :---: |
| **HWHM (MHz)** | $[3.0,\; 9.0]$ | $[9.5,\; 13.5]$ | `ood_hwhm`: $[9.5,\; 13.5]$ (all else ID) | **`0.0%`** ($0.5\text{ MHz}$ gap) |
| **Base Contrast $C$** | $[0.015,\; 0.040]$ | $[0.010, 0.0145] \cup [0.0405, 0.050]$ | `ood_contrast`: $[0.010, 0.0145] \cup [0.0405, 0.050]$ | **`0.0%`** ($0.0005$ gap) |
| **$D_0$ Offset (MHz)** | $[-3.0,\; +3.0]$ | $[-5.0, -3.1] \cup [+3.1, +5.0]$ | `ood_d0`: $[-5.0, -3.1] \cup [+3.1, +5.0]$ | **`0.0%`** ($0.1\text{ MHz}$ gap) |
| **Stochastic Noise** | Gaussian (`noise_type = 0`) | Student-$t$ ($\nu=3.5$) + Spikes (`1`) | `ood_noise`: Student-$t$ ($\nu=3.5$) + Spikes (`1`) | **`0.0%`** (disjoint family) |
| **Drift Family** | `{F1, F2, F4, F5_id}` (`1, 2, 4, 5`) | `{F3, F5_ood}` (`3, 6`) | `ood_drift`: `{F3, F5_ood}` (`3, 6`) | **`0.0%`** (disjoint IDs) |
| **Hum Harmonics** | $r_2, r_3 \in [0.0,\; 0.50]$ | $r_2, r_3 \in (0.50,\; 0.85]$ (when hum $>0$) | `ood_hum`: $100\%$ hum active, $r_2, r_3 \in (0.50, 0.85]$ | **`0.0%`** (strictly $> 0.50$) |
| **Grid Realism** | $30\%$ perturbed ($[2760, 2980]$) | $40\%$ perturbed ($[2760, 2980]$) | `ood_grid`: $100\%$ cropped ($[2790, 2950]$) | Diagnostic stress shift |
| **$|\mathbf{B}|$ (mT)** | $[0.05, 3.20]$ (50% log-unif, 50% unif) | $[0.05, 3.20]$ (50% log-unif, 50% unif) | Same as ID | Shared physical target range |
| **Voigt $\eta$** | $[0.0,\; 1.0]$ uniform | $[0.0,\; 1.0]$ uniform | Same as ID | Shared line-shape range |

### Split Sizes, Master Seeds, SHA256 Hashes, and Per-Sample Deterministic Seed Derivation
Per-sample seeds are derived via `SHA256(struct.pack("<QQ", split_seed, sample_idx))` (taking the first 8 bytes in little-endian order as an unsigned 64-bit integer `uint64`), guaranteeing that sample $i$ is 100% independent of batch size, worker count, or evaluation order.

| Split Name | Sample Count $N$ | Master Split Seed | Regime Stratification | File Path | Raw-Byte SHA256 Digest |
| :--- | :---: | :---: | :--- | :--- | :--- |
| `train` (Iterable) | Infinite (on-the-fly) | `20260101` | Natural log-uniform $[0.5, 20.0]$ | On-the-fly (`OnTheFlyODMRIterableDataset`) | N/A (stream) |
| `train_snapshot` | `20,000` | `20260102` | Natural log-uniform $[0.5, 20.0]$ | `data/synthetic/train_snapshot.npz` | `37761c3d40349ae3c0336f208074be2fc33035b1420642f2a2b38b28af4791e8` |
| `val_id` | `4,000` | `20260201` | Stratified `1333 / 1333 / 1334` | `data/synthetic/val_id.npz` | `294efabc236b362340078d789282654085915e3bb68648f6165573c4f41a48cc` |
| `val_ood` | `4,000` | `20260202` | Stratified `1333 / 1333 / 1334` | `data/synthetic/val_ood.npz` | `014df8d74cab853b09771a114750db2fce5afd194c48e96d3d4098d3e4026dfa` |
| `ood_hwhm` | `1,000` | `20260301` | Natural log-uniform $[0.5, 20.0]$ | `data/synthetic/ood_hwhm.npz` | `643cb17ffd93f0d41fe45fac48dfc88a3e404822846f611a075ef910a4838b19` |
| `ood_contrast` | `1,000` | `20260302` | Natural log-uniform $[0.5, 20.0]$ | `data/synthetic/ood_contrast.npz` | `9396e1f3efc8b3ee7fb264c0564f4cf0c7ebbd127389b787f226a52f8b4e30a6` |
| `ood_d0` | `1,000` | `20260303` | Natural log-uniform $[0.5, 20.0]$ | `data/synthetic/ood_d0.npz` | `501e98151a698d6e97cf012828d65e99328ce2b46975a1a6f108367304b2645a` |
| `ood_noise` | `1,000` | `20260304` | Natural log-uniform $[0.5, 20.0]$ | `data/synthetic/ood_noise.npz` | `9c6bbd457165dca0aa73d7c5a190461047d2b62a850ecd3484f59a2c24735c8b` |
| `ood_drift` | `1,000` | `20260305` | Natural log-uniform $[0.5, 20.0]$ | `data/synthetic/ood_drift.npz` | `c75b7c630e7ffe975c23f397708ea4c56af1408af6017c50eaf4d55f058181c4` |
| `ood_hum` | `1,000` | `20260306` | Natural log-uniform $[0.5, 20.0]$ | `data/synthetic/ood_hum.npz` | `9b57ed282f1a5320da09dd0373d145a118e7660a3f76f72b7479e5b7be67c340` |
| `ood_grid` | `1,000` | `20260307` | Natural log-uniform $[0.5, 20.0]$ | `data/synthetic/ood_grid.npz` | `f7998fbb6d7da6fdba218b0fb0b03838d3fca4371a68fac4648bc08fd6dad0c7` |
| `test_id` | `4,000` | `20260901` | Stratified `1333 / 1333 / 1334` | `data/synthetic/test_id.npz` (LOCKED) | `819e007a1a3fbc45d52df4334fd0ce6906f7e59e31cd22204b023383d63cbf79` |
| `test_ood` | `4,000` | `20260902` | Stratified `1333 / 1333 / 1334` | `data/synthetic/test_ood.npz` (LOCKED) | `fcdce8033fb0d04cfd5cb7c5ea92b1b9ee7fbfd457556bba00c8a3e02ce627e7` |

---

## 5. Locked Test-Set Protocol & Access Audit (`test_access.log`)

1. **Dry-Run Twin Validation**:
   Prior to generating `test_id.npz` and `test_ood.npz`, temporary dry-run twin datasets (`twin_test_id.npz` with throwaway seed `999901` and `twin_test_ood.npz` with throwaway seed `999902`) are generated into a temporary directory, subjected to all consistency and range checks, and deleted.
2. **Write-Once Test Generation & Hashing**:
   `test_id.npz` and `test_ood.npz` are written once and never parsed by validation, plotting, reference estimators, or unit tests.
3. **Access Audit Log (`data/synthetic/test_access.log`)**:
   - Any call to `load_synthetic_split` targeting `test_id` or `test_ood` appends a JSON line with `"action": "data_read"` to `data/synthetic/test_access.log` (and raises `PermissionError` unless `allow_test_read=True` is explicitly passed in Stage 3 evaluation).
   - Raw byte-stream SHA256 verification (`verify_split_sha256`) reads raw bytes without parsing NPZ arrays and appends `"action": "hash_check"` to `data/synthetic/test_access.log`.
   - Unit tests assert that the count of `"action": "data_read"` entries in `data/synthetic/test_access.log` is strictly `0`.
