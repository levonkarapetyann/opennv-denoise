# SNR Regimes and Evaluation Taxonomy (Stage 1.5 Specification)

This document formalizes the dual-axis SNR specification, the Cramér-Rao lower bound (CRB) distribution across 3D field geometries, and the three-tier evaluation taxonomy (Easy, Medium, Hard) established on the held-out `FINAL` dataset (`FINAL_SEED = 500001`, $N = 200$ spectra).

---

## 1. Physical Motivation: Why Nominal SNR Is Deceptive in 4-Axis NV Magnetometry

In single-frequency magnetometry, Signal-to-Noise Ratio is conventionally defined as:
$$\text{SNR}_{\text{nom}} = \frac{C_{\text{base}}}{\sigma_{\text{white}}}$$
where $C_{\text{base}} \approx 0.03$ (3% base optical contrast) and $\sigma_{\text{white}}$ is the Gaussian photon-shot-noise standard deviation.

However, in full 4-axis tetrahedral diamond ensembles ($\mathbf{n}_k \in \frac{1}{\sqrt{3}} \{(\pm 1, \pm 1, \pm 1)\}$), $\text{SNR}_{\text{nom}}$ severely misrepresents the true information content and inversion difficulty of the spectrum for three fundamental physical reasons:

1. **$^{14}\text{N}$ Hyperfine Triplet Splitting ($1/3$ Factor)**:
   Each electronic transition ($m_s = 0 \leftrightarrow \pm 1$) is split into 3 hyperfine transitions ($m_I \in \{-1, 0, +1\}$) by $A_{\parallel} = 2.16\text{ MHz}$. Under thermal nuclear polarization, each hyperfine dip carries only **$1/3$** of the axis contrast ($C_{\text{dip}} = C_{\text{axis}} / 3$).

2. **Transverse-Field Contrast Quenching ($|\langle 0 | \psi_0 \rangle|^4$)**:
   The magnetic field projection perpendicular to the $k$-th NV axis ($B_{\perp, k} = \|\mathbf{B} - (\mathbf{B}\cdot\mathbf{n}_k)\mathbf{n}_k\|$) mixes the ground-state $|m_s = 0\rangle$ with $|\pm 1\rangle$. This reduces the transition optical contrast by:
   $$c_k = |\langle m_s = 0 | \psi_0^{(k)} \rangle|^4 \le 1.0$$
   For large magnetic fields or directions nearly perpendicular to an NV axis, $c_k$ drops dramatically (down to $0.40 - 0.70$), suppressing the dip depth far below $C_{\text{base}}$.

3. **Maximum Informative Splitting (Outer Dips Determine $|\mathbf{B}|$ Resolution)**:
   The total field magnitude $|\mathbf{B}|$ is primarily constrained by the largest Zeeman splitting, corresponding to the NV axis with the maximum longitudinal projection:
   $$k_{\text{outer}} = \arg\max_k |\mathbf{B} \cdot \mathbf{n}_k|$$
   If this outer dip is submerged beneath noise or thermal drift, nonlinear optimizers jump to an interior basin, failing catastrophically.

### Dual-Axis Definition: Effective Outer-Dip SNR
To capture true detection capability, we define the **Effective Outer-Dip SNR**:
$$\text{EffSNR}_{\text{outer}} = \frac{C_{\text{base}} \cdot c_{\text{outer}} / 3}{\sigma_{\text{white}}} = \frac{\text{SNR}_{\text{nom}} \cdot c_{\text{outer}}}{3}$$
where $c_{\text{outer}} = |\langle 0 | \psi_0^{(k_{\text{outer}})} \rangle|^4 \le 1.0$.

Because $c_{\text{outer}} \le 1.0$, $\text{EffSNR}_{\text{outer}}$ is **always less than $\frac{1}{3} \text{SNR}_{\text{nom}}$**:
- At nominal $\text{SNR} = 15$, the effective outer dip SNR is only $\sim 3.2 - 4.9$ (not 15).
- At nominal $\text{SNR} = 4$, the effective outer dip SNR is only $\sim 0.9 - 1.3$ (the outer dip peak amplitude is comparable to $1\sigma$ noise).
- At nominal $\text{SNR} = 2$, the effective outer dip SNR is only $\sim 0.4 - 0.7$ (outer dips are strictly sub-noise, invisible without coherent baseline/prior averaging).

---

## 2. Empirical Distributions on Held-Out FINAL Set (`FINAL_SEED = 500001`, $N = 200$)

The held-out evaluation dataset consists of $N = 200$ synthetic spectra on the unit sphere $S^2$ (with 20% sampled near high-symmetry degenerate planes), $|\mathbf{B}| \in [0.1, 3.2]\text{ mT}$, $\text{HWHM} \in [3.0, 9.0]\text{ MHz}$, and $C_{\text{base}} \in [0.02, 0.04]$.

### Table: Dual-Axis SNR Percentiles on FINAL-200
| Nominal SNR | $\sigma_{\text{white}}$ | $\text{EffSNR}_{\text{outer}}$ P10 | $\text{EffSNR}_{\text{outer}}$ P50 (Median) | $\text{EffSNR}_{\text{outer}}$ P90 | Mean $\text{EffSNR}_{\text{outer}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **15** | `0.00200` | `2.41` | `3.17` | `4.22` | `3.27` |
| **4** | `0.00750` | `0.84` | `1.10` | `1.47` | `1.14` |
| **2** | `0.01500` | `0.52` | `0.68` | `0.91` | `0.70` |

### Table: Cramér-Rao Lower Bound (CRB) Distributions on FINAL-200
The Cramér-Rao Lower Bound $\text{CRB}(|\mathbf{B}|) = \sqrt{[F^{-1}]_{00}}$ is computed from the full forward-model Jacobian with respect to $[|\mathbf{B}|, \hat{\mathbf{b}}, D_0, \text{HWHM}, C_{\text{base}}, \text{poly baseline}]$ under white Gaussian noise.

#### 1. Standardized Reference CRB ($\text{HWHM} = 4.5\text{ MHz}$, $C_{\text{base}} = 0.03$):
| Nominal SNR | P10 ($\mu\text{T}$) | P50 (Median, $\mu\text{T}$) | P90 ($\mu\text{T}$) | P99 ($\mu\text{T}$) | Mean ($\mu\text{T}$) | Fraction $>100\,\mu\text{T}$ | Fraction $>220\,\mu\text{T}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **15** | `3.81` | `5.71` | `38.78` | `146.55` | `14.94` | `3.0%` | `0.0%` |
| **4** | `14.30` | `21.43` | `145.43` | `549.56` | `56.04` | `12.0%` | `6.5%` |
| **2** | `28.61` | `42.85` | `290.87` | `1099.11` | `112.08` | `20.0%` | `11.5%` |

#### 2. Per-Spectrum True CRB (Individual True $\text{HWHM}_i \in [3, 9]\text{ MHz}$, $C_i \in [0.02, 0.04]$):
| Nominal SNR | P10 ($\mu\text{T}$) | P50 (Median, $\mu\text{T}$) | P90 ($\mu\text{T}$) | P99 ($\mu\text{T}$) | Mean ($\mu\text{T}$) | Fraction $>100\,\mu\text{T}$ | Fraction $>220\,\mu\text{T}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **15** | `4.07` | `7.58` | `53.20` | `299.65` | `24.14` | `5.0%` | `2.0%` |
| **4** | `15.25` | `28.44` | `199.48` | `1123.69` | `90.54` | `16.0%` | `10.0%` |
| **2** | `30.50` | `56.88` | `398.97` | `2247.38` | `181.08` | `29.5%` | `13.5%` |

> [!NOTE]
> **Confirmation of Previous Claims**: At nominal $\text{SNR} = 4$, the P90 CRB is **`145.43 uT`** (standardized) and **`199.48 uT`** (per-spectrum), confirming the previous finding that P90 CRB exceeds $110\,\mu\text{T}$. At nominal $\text{SNR} = 2$, the P90 CRB is **`290.87 uT`** (standardized) and **`398.97 uT`** (per-spectrum), confirming that P90 CRB exceeds $220\,\mu\text{T}$.

---

## 3. Evaluation Taxonomy for Stage 2

To prevent reporting averages over physically incompatible regimes, all Stage 2 machine learning evaluations will segment test sets into three pre-registered regimes based on $\text{EffSNR}_{\text{outer}}$:

### Definition of Regimes
1. **Easy Regime ($\text{EffSNR}_{\text{outer}} > 3.0$)**:
   - Dips are clearly distinguishable above the noise floor.
   - Non-convex optimization basin has low curvature ambiguity.
   - Standard classical NLLSVarProj achieves $< 10\,\mu\text{T}$ median accuracy.
2. **Medium Regime ($1.5 \le \text{EffSNR}_{\text{outer}} \le 3.0$)**:
   - Dips are partially obscured; peak detection without global spatial coupling fails frequently.
   - Mains hum and pink noise induce significant baseline-fitting covariance.
   - Classical NLLS exhibits $10\% - 35\%$ wrong-basin traps.
3. **Hard / Breakdown Regime ($\text{EffSNR}_{\text{outer}} < 1.5$)**:
   - Individual hyperfine peaks have amplitudes below $1.5\sigma$ Gaussian noise.
   - Severe non-convex multi-modality: classical gradient-based optimizers fail in $>50\%$ of realizations.
   - Local CRB is unachievable without inductive priors or learned regularizers.

### Spectrum Counts & Breakdown on FINAL-200
| Nominal SNR | Easy ($\text{EffSNR} > 3$) | Medium ($1.5 \le \text{EffSNR} \le 3$) | Hard ($\text{EffSNR} < 1.5$) |
| :---: | :---: | :---: | :---: |
| **SNR 15** | **`112 (56.0%)`** | **`87 (43.5%)`** | **`1 (0.5%)`** |
| **SNR 4** | **`0 (0.0%)`** | **`19 (9.5%)`** | **`181 (90.5%)`** |
| **SNR 2** | **`0 (0.0%)`** | **`0 (0.0%)`** | **`200 (100.0%)`** |

---

## 4. Architectural & Evaluation Implications for Stage 2 (Deep Learning)

1. **Loss Function Weighting**:
   - In the Hard regime ($\text{EffSNR}_{\text{outer}} < 1.5$), Mean Squared Error (MSE) on predicted $|\mathbf{B}|$ is heavily corrupted by outlier wrong-basin predictions. Stage 2 models must use smooth Huber loss or negative log-likelihood under a mixture density network (MDN) to prevent gradient collapse from outliers.

2. **Stratified Reporting**:
   - Any machine learning model evaluated on OpenNV-Denoise must report performance stratified by both Nominal SNR and the three $\text{EffSNR}_{\text{outer}}$ tiers. Comparing a neural network against classical NLLS averaged across all spectra without tier stratification creates an unfair comparison because the classical fitter operates in the threshold breakdown regime for $>90\%$ of SNR 4 and 100% of SNR 2.

3. **Inductive Priors**:
   - In the Hard regime, classical NLLS with a prior (`fit_physical_with_prior`) reduces wrong basins from $37.0\% \to 34.5\%$ at SNR 4 and $66.0\% \to 63.5\%$ at SNR 2. Deep neural networks must encode the tetrahedral lattice geometry and $O_h$ quotient symmetry directly to outperform physics-informed baselines in this regime.
