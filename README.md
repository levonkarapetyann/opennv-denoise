# OpenNV-Denoise: Physics-Informed Quantum Magnetometry & Classical Baseline (Stage 1.5)

**OpenNV-Denoise** is a scientific Python framework for simulating, diagnosing, and inverting Optically Detected Magnetic Resonance (ODMR) spectra from tetrahedral nitrogen-vacancy ($\text{NV}^-$) diamond ensembles under realistic commercial-off-the-shelf (COTS) hardware noise.

> [!IMPORTANT]
> **Code Freeze Notice**: Core physical modeling and the classical baseline in [`src/baseline.py`](src/baseline.py) are **FROZEN** at git tag `baseline-frozen` (`e9194ad`). All diagnostic variants, fixes, and realistic extensions reside in [`src/baseline_variants.py`](src/baseline_variants.py).

---

## 1. Explicit Assumptions & Physical Modelling Choices

To guarantee an honest, non-strawman comparison between classical nonlinear least-squares (NLLS) physical baselines and downstream neural-network denoisers, all physical modelling assumptions, regularization constants, and evaluation protocols are pre-registered below:

1. **Transverse-Field Contrast Factor ($|\langle 0 | \psi_0 \rangle|^4$)**:
   - In [`src/physics.py`](src/physics.py), each NV axis transition amplitude is scaled by $\text{contrast\_factor} = |\langle m_s = 0 | \psi_0 \rangle|^4$ (`purity_ms0 ** 2`), where $|\psi_0\rangle$ is the eigenstate of the $3 \times 3$ spin-1 ground-state Hamiltonian with the largest $|m_s = 0\rangle$ projection.
   - **Modelling Choice**: This fourth-power dependence is a phenomenological **modelling choice**, not a first-principles derivation. In physical $\text{NV}^-$ ensembles, transverse magnetic fields mix $|m_s = 0\rangle$ with $|m_s = \pm 1\rangle$ in both the ground and excited triplets, modifying intersystem crossing (ISC) rates and steady-state spin polarization in a manner that depends on optical pump intensity and microwave Rabi frequency.

2. **Equal $^{14}\text{N}$ Hyperfine Amplitude Split ($C / 3$)**:
   - Each electronic spin transition ($|0\rangle \leftrightarrow |-1\rangle$ and $|0\rangle \leftrightarrow |+1\rangle$) is split into a triplet of pseudo-Voigt dips separated by $A_{\parallel} = 2.16\text{ MHz}$ ($m_I \in \{-1, 0, +1\}$), with each hyperfine component assigned an equal amplitude of $C_{\text{axis}} / 3$.
   - **Modelling Choice**: This equal $C/3$ partition is a **modelling choice** assuming an unpolarized thermal nuclear spin bath and neglecting transverse hyperfine mixing ($A_{\perp}$), nuclear quadrupolar shifts, and dynamic nuclear polarization (DNP).

3. **Zero-Strain Assumption ($E = 0\text{ MHz}$) & Canonical Wedge Symmetry Guard**:
   - In Stage 1 / 1.5, the transverse crystal-field/strain splitting parameter is $E = 0.0\text{ MHz}$, while all public physics and baseline functions accept `e_mhz: float = 0.0` for forward compatibility.
   - When $E = 0$, the 4 tetrahedral $[111]$ NV axes depend only on the unordered set of unsigned projections $\{|B_{\parallel, k}|\}_{k=1}^4$, which is invariant under the 48-element octahedral group $O_h$ ($2^3 = 8$ coordinate sign flips $\times$ $3! = 6$ coordinate permutations). Thus any vector $\mathbf{B} \in \mathbb{R}^3$ has a unique canonical representative in the wedge $0 \le B_x \le B_y \le B_z$.
   - Because non-zero strain ($E \ne 0$) breaks this 48-fold octahedral symmetry, [`canonicalize_b_vec`](src/physics.py) and [`fit_physical_odmr`](src/baseline.py) explicitly raise a `ValueError` whenever `e_mhz != 0.0`.

4. **Variable-Projection Regularization & Three-Way Seed Separation**:
   - All variable-projection ridge weights in [`_VarProjDesign`](src/baseline.py)—linear baseline ridge `[0.25, 0.8]`, degree-2 Legendre curvature penalty `18.0`, degree-3 Legendre cubic penalty `36.0`, sinusoidal mains-hum ridge `8.0`, and outer wing anchor weight `wing_weight = 0.20` on $[2750, 2775] \cup [2965, 2990]\text{ MHz}$—were tuned exclusively on the **Development Set (`DEV_SEED = 10001`)**.
   - Three strictly disjoint seed sets are enforced across the project:
     - **`DEV_SEED = 10001`**: Used for diagnostic explorations, parameter tuning, and design decisions.
     - **`TEST_SEED = 90001`**: Used for unit tests and the single degenerate-axis demo vector $\mathbf{B} = (1.15, 0.75, 0.40)\text{ mT}$ (non-representative).
     - **`FINAL_SEED = 500001`**: Held-out final evaluation set of 200 random spectra on $S^2$ (including 20% near-degenerate symmetry planes), executed once at the conclusion of development.

5. **Statistical Metrics & Cramér-Rao Lower Bound (CRB) Reporting**:
   - The reference CRB reported in [`run_week1_demo.py`](run_week1_demo.py) assumes white intensity-scaled Gaussian noise, evaluated from the forward-model Jacobian with respect to all 10 free parameters ($|\mathbf{B}|, \hat{\mathbf{b}}, D_0, \text{HWHM}, C_{\text{base}}, \text{poly baseline}$).
   - **Full-Sample Reporting**: To prevent selection bias, the primary metrics reported are **all-inclusive full-sample statistics**: RMS over all $N$ trials (no exclusions), median (P50), P90, P95, P99, MAE, Failure rate ($>5\times\text{CRB}_i$), Wrong Basin rate ($>0.15\text{ mT}$ error in canonical wedge), and legacy Failure rate ($>100\,\mu\text{T}$).
   - **Auxiliary Metric**: $\sigma_{\text{core}}$ (RMS of runs with $|\Delta B| \le 5\times\text{CRB}$) is reported strictly as an auxiliary column labelled "conditional on success; biased low by construction" and is never divided by CRB in primary comparisons.

6. **HWHM Bounds & Pseudo-Voigt Mixing Factor (`_DEFAULT_ETA = 0.35`) Policy**:
   - [`fit_physical_odmr`](src/baseline.py) accepts `hwhm_bounds: Tuple[float, float] = (3.0, 9.0)` (matching the training distribution `TRAIN_LINEWIDTH_RANGE`) and `eta: float = 0.35`.
   - **Out-of-Distribution (OOD) Policy**: Any OOD linewidth evaluation (`OOD_LINEWIDTH_RANGE = (9.5, 13.5)` MHz) comparing `fit_physical_odmr` against a neural network **must** pass widened bounds `hwhm_bounds=(2.0, 16.0)` so the classical fitter is not artificially constrained.

7. **Intensity-Dependent Noise Scaling**:
   - In [`src/noise.py`](src/noise.py), stochastic noise is modulated by $\sqrt{\text{clip}(S_{\text{clean}}, 0.5, 1.5)}$. This models intensity-dependent Gaussian noise scaling (high-photon-count Gaussian approximation) rather than discrete Poisson counting.

---

## 2. Provenance of Provisional Targets & Reporting Honesty

An audit of earlier project notes revealed references to provisional "acceptance targets" (e.g., $\text{Med} \le 15\,\mu\text{T} / \text{P90} \le 35\,\mu\text{T}$ at SNR 15; $\text{Med} \le 40\,\mu\text{T} / \text{P90} \le 90\,\mu\text{T}$ at SNR 4; $\text{Med} \le 90\,\mu\text{T} / \text{P90} \le 220\,\mu\text{T}$ at SNR 2).

- **Source Audit**: A comprehensive search of the project repository, configuration files, and commit logs confirms that these numerical thresholds had **no pre-registered physical or external source**. They originated as ad-hoc prompt heuristics during intermediate debugging.
- **Reporting Policy**: All pass/fail labels based on ungrounded targets have been purged from the repository. In their place, all fitters are evaluated using pure descriptive statistics and compared against rigorous physical benchmarks: the local Cramér-Rao Lower Bound (CRB), the true-parameter oracle, and the effective outer-dip SNR.

---

## 3. Dual-Axis SNR Regimes & CRB Distributions

In 4-axis NV magnetometry, nominal SNR ($\text{SNR}_{\text{nom}} = C_{\text{base}} / \sigma_{\text{white}}$) masks the true difficulty of the inversion task. We define the **Effective Outer-Dip SNR**:
$$\text{EffSNR}_{\text{outer}} = \frac{C_{\text{base}} \cdot c_{\text{outer}} / 3}{\sigma_{\text{white}}} = \frac{\text{SNR}_{\text{nom}} \cdot |\langle 0 | \psi_0^{(k_{\text{outer}})} \rangle|^4}{3}$$
where $k_{\text{outer}} = \arg\max_k |\mathbf{B} \cdot \mathbf{n}_k|$.

See [`docs/snr_regimes.md`](docs/snr_regimes.md) for full physical derivation and Stage 2 evaluation taxonomy.

### Empirical CRB Distributions on Held-Out FINAL Set (`FINAL_SEED = 500001`, $N = 200$)

#### Standardized Reference CRB ($\text{HWHM} = 4.5\text{ MHz}$, $C_{\text{base}} = 0.03$):
| Nom SNR | Mean $\text{EffSNR}_{\text{outer}}$ | P10 ($\mu\text{T}$) | P50 (Med, $\mu\text{T}$) | P90 ($\mu\text{T}$) | P99 ($\mu\text{T}$) | Mean ($\mu\text{T}$) | Frac $>100\,\mu\text{T}$ | Frac $>220\,\mu\text{T}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **15** | `3.27` | `3.81` | `5.71` | `38.78` | `146.55` | `14.94` | `3.0%` | `0.0%` |
| **4** | `1.14` | `14.30` | `21.43` | `145.43` | `549.56` | `56.04` | `12.0%` | `6.5%` |
| **2** | `0.70` | `28.61` | `42.85` | `290.87` | `1099.11` | `112.08` | `20.0%` | `11.5%` |

#### Per-Spectrum True CRB ($\text{HWHM}_i \in [3, 9]\text{ MHz}$, $C_i \in [0.02, 0.04]$):
| Nom SNR | Mean $\text{EffSNR}_{\text{outer}}$ | P10 ($\mu\text{T}$) | P50 (Med, $\mu\text{T}$) | P90 ($\mu\text{T}$) | P99 ($\mu\text{T}$) | Mean ($\mu\text{T}$) | Frac $>100\,\mu\text{T}$ | Frac $>220\,\mu\text{T}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **15** | `3.27` | `4.07` | `7.58` | `53.20` | `299.65` | `24.14` | `5.0%` | `2.0%` |
| **4** | `1.14` | `15.25` | `28.44` | `199.48` | `1123.69` | `90.54` | `16.0%` | `10.0%` |
| **2** | `0.70` | `30.50` | `56.88` | `398.97` | `2247.38` | `181.08` | `29.5%` | `13.5%` |

> [!NOTE]
> **Confirmed**: The P90 CRB on the random sphere exceeds $100\,\mu\text{T}$ at SNR 4 (`145.43–199.48 uT`) and exceeds $220\,\mu\text{T}$ at SNR 2 (`290.87–398.97 uT`). In the threshold breakdown regime ($\text{EffSNR} < 1.5$), wide confidence intervals and wrong-basin traps are physically unavoidable properties of the forward model.

---

## 4. Methodological Findings & Diagnostics

### Task 3: Naive Lorentzian Diagnosis & Fix
In frozen `fit_lorentzian_naive`, the median error on the demo vector was $\sim 391\,\mu\text{T}$. Diagnostic investigation revealed two implementation bugs:
1. `wing_thresh` was computed across all $df$ where inner dips distorted the median absolute deviation (MAD).
2. `hs_ub` was clamped to `max_active_df + 6.0 MHz` ($\approx 15\text{ MHz}$), physically trapping the outer half-splitting inside the box.

In [`src/baseline_variants.py`](src/baseline_variants.py), `fit_lorentzian_naive_fixed` computes the wing noise threshold strictly from $df \ge 80\text{ MHz}$, requires peak prominence $\ge 0.025 \max(f)$, keeps `hs_ub = 90.0 MHz`, and uses a 4-start comb.
- **Demo Vector (SNR 15)**: Median error dropped from `390.73 uT` $\to$ **`38.90 uT`** (`Fail>100uT` reduced from $66\%$ to $18\%$).
- **FINAL-200 (SNR 15)**: Median error is **`170.43 uT`** (vs. frozen `346.84 uT`).
- Both versions are presented side-by-side in the benchmark tables, explicitly labelled as `"naive (frozen: biased hs_ub bug)"` and `"naive (fixed: 4-pair, model-mismatched)"`.

### Task 4: Polynomial Degree Policy & Drift Mismatch on DEV
- **Frozen Default (`poly_degree = 3`)**: At git tag `baseline-frozen`, the default signature of [`fit_physical_odmr`](src/baseline.py) is `poly_degree: int = 3`.
- **Inverse Crime Disclosure**: The default `poly_degree = 3` in `fit_physical_odmr` matches the simulator's polynomial drift generator (an inverse crime).
- **Exact Polynomial Degree Per Benchmark Table Row**:
  - `naive (frozen)` and `naive (fixed)`: linear baseline (`deg-1`).
  - `physical, plain (deg-2)`: quadratic Legendre baseline (`poly_degree=2`).
  - `physical, plain (deg-3: default, cubic drift match)`: cubic Legendre baseline (`poly_degree=3`, the frozen default).
  - `physical, with_prior (deg-3, lam=2.5e-4)`: cubic Legendre baseline (`poly_degree=3`).
  - `physical, informed_unknown_freq (deg-2, 45-65Hz)`: quadratic Legendre baseline (`poly_degree=2`) + scanned fundamental sinusoid.
  - `physical, informed_harmonics (deg-2, 1-3f)`: quadratic Legendre baseline (`poly_degree=2`) + 1f/2f/3f harmonic sinusoids.
  - `physical, informed (deg-3, exact 50Hz UB)`: cubic Legendre baseline (`poly_degree=3`) + exact 50/60 Hz fundamental sinusoid.
  - `physical, oracle (0-poly: known drift only)` and `physical, oracle (0-poly: true drift + hum)`: `0` fitted polynomial parameters (true baseline subtracted directly).
- **Polynomial Degree Side-by-Side**: On FINAL-200, degree-2 baseline (`deg-2`) performs essentially identically to degree-3 (`deg-3`):
  - SNR 15: `deg-2` achieves `RMS = 33.78 uT, Med = 11.95 uT` vs. `deg-3` `RMS = 34.31 uT, Med = 11.81 uT`.
  - SNR 4: `deg-2` achieves `RMS = 120.21 uT, Med = 41.42 uT` vs. `deg-3` `RMS = 119.73 uT, Med = 42.36 uT`.
  - SNR 2: `deg-2` achieves `RMS = 536.89 uT, Med = 95.33 uT` vs. `deg-3` `RMS = 514.28 uT, Med = 91.16 uT`.
- **Mismatched Drift Stress Test (DEV-200, 2 Sinusoids + Exponential Warm-Up)**:
  - SNR 15: `deg-1` `RMS = 222.4 uT` | `deg-2` `RMS = 188.3 uT` | `deg-3` `RMS = 187.7 uT`.
  - Under unmodelled non-polynomial drift, quadratic/cubic baselines prevent low-frequency drift curvature from falsely shifting dip locations.

### Task 5: Realistic Informed Variants
In field applications, mains hum frequency is not known a priori. [`src/baseline_variants.py`](src/baseline_variants.py) provides:
1. `fit_informed_unknown_freq`: Scans frequencies from 45 to 65 Hz in 0.25 Hz steps using residual projection screening before refinement.
   - On DEV-200 at SNR 15, recovers the true 50 Hz hum within $\le 0.5\text{ Hz}$ in 100% of cases and matches exact-frequency accuracy (`RMS = 23.88 uT` vs. `23.83 uT`).
2. `fit_informed_harmonics`: Projects fundamental plus 2nd and 3rd harmonics ($1f, 2f, 3f$).

### Task 6: Prior-Regularized Fitter
[`fit_physical_with_prior`](src/baseline_variants.py) re-ranks Stage-B candidates using a MAP objective:
$$\text{Cost}_{\text{MAP}} = \text{VarProjCost} + \lambda \left[ -\ln p_{\text{train}}(|\mathbf{B}|, \text{HWHM}, C_{\text{base}}) \right]$$
with $\lambda = 2.5 \times 10^{-4}$ (tuned on DEV).
On FINAL-200, MAP re-ranking reduces wrong-basin traps in challenging regimes:
- SNR 4: Wrong Basin rate drops from **$37.0\% \to 34.5\%$** (`RMS` drops from `119.73` to **`112.43 uT`**).
- SNR 2: Wrong Basin rate drops from **$66.0\% \to 63.5\%$** (`Med` drops from `91.16` to **`83.37 uT`**).

---

## 5. Benchmark Results

### Table 1: Held-Out FINAL Set (`FINAL_SEED = 500001`, $N = 200$ Spectra on $S^2$, Representative)

All-inclusive full-sample statistics over all 200 spectra (no exclusions):

| Nom SNR | Eff Outer SNR | Method | RMS ($\mu\text{T}$) | Med ($\mu\text{T}$) | P90 ($\mu\text{T}$) | P95 ($\mu\text{T}$) | P99 ($\mu\text{T}$) | MAE ($\mu\text{T}$) | Fail $>5\text{CRB}_i$ | Wrong Basin ($>0.15\text{mT}$) | Fail $>100\mu\text{T}$ | Time (ms) | $\sigma_{\text{core}}^\dagger$ | $\frac{\text{RMS}}{\text{CRB}_{\text{med}}^\ddagger}$ |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **15** | `3.17` | `naive (deg-1, frozen: biased hs_ub bug)` | `658.69` | `346.84` | `1004.50` | `1850.51` | `2656.70` | `445.62` | `71.0%` | `N/A` | `66.5%` | `55.8` | `173.07` | `86.90x*` |
| **15** | `3.17` | `naive (deg-1, fixed: 4-pair, mismatched)` | `754.07` | `170.43` | `1884.22` | `2418.59` | `2788.94` | `476.32` | `64.0%` | `N/A` | `55.0%` | `61.8` | `194.52` | `99.48x*` |
| **15** | `3.17` | `physical, plain (deg-2)` | `33.78` | `11.95` | `39.86` | `73.71` | `153.30` | `20.10` | `5.0%` | `8.0%` | `3.0%` | `241.6` | `33.58` | `4.45x` |
| **15** | `3.17` | `physical, plain (deg-3: cubic drift match)` | `34.31` | `11.81` | `39.77` | `72.63` | `168.10` | `20.07` | `5.0%` | `8.0%` | `3.0%` | `261.6` | `33.56` | `4.52x` |
| **15** | `3.17` | `physical, with_prior (deg-3, lam=2.5e-4)` | `33.90` | `11.79` | `37.04` | `72.63` | `168.10` | `19.78` | `4.5%` | `7.5%` | `2.5%` | `262.2` | `33.54` | `4.47x` |
| **15** | `3.17` | `physical, informed_unknown_freq (deg-2, 45-65Hz)` | `26.26` | `7.56` | `30.51` | `55.35` | `127.35` | `14.89` | `1.5%` | `5.0%` | `1.5%` | `321.4` | `25.59` | `3.46x` |
| **15** | `3.17` | `physical, informed_harmonics (deg-2, 1-3f)` | `26.46` | `7.55` | `30.46` | `55.70` | `127.35` | `14.89` | `1.5%` | `5.0%` | `1.5%` | `302.1` | `25.59` | `3.49x` |
| **15** | `3.17` | `physical, informed (deg-3, exact 50Hz UB)` | `27.81` | `7.52` | `30.87` | `62.90` | `132.85` | `14.96` | `1.5%` | `4.5%` | `2.0%` | `283.4` | `26.47` | `3.67x` |
| **15** | `3.17` | `physical, oracle (0-poly: known drift only)` | `26.56` | `11.48` | `36.15` | `60.85` | `102.73` | `17.41` | `3.5%` | `5.5%` | `1.0%` | `249.0` | `24.86` | `3.50x` |
| **15** | `3.17` | **`physical, oracle (0-poly: true drift + hum)`** | **`22.13`** | **`7.49`** | **`27.78`** | **`48.61`** | **`104.93`** | **`12.69`** | **`0.0%`** | **`1.5%`** | **`0.5%`** | **`261.5`** | **`20.90`** | **`2.92x`** |
| **4** | `1.10` | `naive (deg-1, frozen: biased hs_ub bug)` | `576.65` | `337.27` | `969.07` | `1479.44` | `2396.65` | `426.45` | `51.5%` | `N/A` | `74.0%` | `58.9` | `272.82` | `20.27x*` |
| **4** | `1.10` | `naive (deg-1, fixed: 4-pair, mismatched)` | `722.89` | `225.48` | `1732.18` | `2215.35` | `2821.57` | `473.06` | `50.0%` | `N/A` | `66.5%` | `61.4` | `248.16` | `25.41x*` |
| **4** | `1.10` | `physical, plain (deg-2)` | `120.21` | `41.42` | `162.20` | `308.19` | `572.80` | `70.19` | `7.5%` | `37.5%` | `22.0%` | `247.9` | `75.71` | `4.23x` |
| **4** | `1.10` | `physical, plain (deg-3: cubic drift match)` | `119.73` | `42.36` | `159.42` | `295.66` | `573.49` | `69.38` | `8.0%` | `37.0%` | `20.5%` | `268.5` | `75.65` | `4.21x` |
| **4** | `1.10` | `physical, with_prior (deg-3, lam=2.5e-4)` | `112.43` | `39.51` | `156.85` | `292.05` | `571.21` | `66.82` | `7.5%` | `34.5%` | `19.0%` | `270.8` | `75.20` | `3.95x` |
| **4** | `1.10` | `physical, informed_unknown_freq (deg-2, 45-65Hz)` | `116.91` | `35.50` | `151.34` | `288.75` | `581.42` | `64.21` | `5.0%` | `32.5%` | `18.0%` | `330.1` | `71.65` | `4.11x` |
| **4** | `1.10` | `physical, informed_harmonics (deg-2, 1-3f)` | `117.15` | `35.51` | `151.20` | `288.75` | `581.42` | `64.25` | `5.0%` | `32.5%` | `18.0%` | `311.7` | `71.65` | `4.12x` |
| **4** | `1.10` | `physical, informed (deg-3, exact 50Hz UB)` | `116.39` | `35.51` | `149.98` | `288.75` | `581.42` | `63.92` | `4.0%` | `32.0%` | `18.5%` | `293.5` | `71.70` | `4.09x` |
| **4** | `1.10` | `physical, oracle (0-poly: known drift only)` | `105.82` | `39.97` | `148.99` | `260.10` | `485.12` | `59.30` | `6.5%` | `31.0%` | `18.5%` | `209.5` | `73.21` | `3.72x` |
| **4** | `1.10` | **`physical, oracle (0-poly: true drift + hum)`** | **`78.37`** | **`30.43`** | **`132.23`** | **`220.14`** | **`398.22`** | **`52.15`** | **`2.0%`** | **`22.5%`** | **`14.0%`** | **`206.6`** | **`66.50`** | **`2.76x`** |
| **2** | `0.68` | `naive (deg-1, frozen: biased hs_ub bug)` | `692.75` | `377.67` | `1228.99` | `1665.40` | `2878.11` | `507.94` | `42.5%` | `N/A` | `77.0%` | `64.5` | `384.06` | `12.18x*` |
| **2** | `0.68` | `naive (deg-1, fixed: 4-pair, mismatched)` | `747.93` | `266.54` | `1821.40` | `2298.15` | `2790.32` | `514.88` | `39.0%` | `N/A` | `73.0%` | `65.2` | `365.12` | `13.15x*` |
| **2** | `0.68` | `physical, plain (deg-2)` | `536.89` | `95.33` | `527.62` | `730.15` | `1280.45` | `246.12` | `13.5%` | `66.5%` | `48.5%` | `252.1` | `321.45` | `9.44x*` |
| **2** | `0.68` | `physical, plain (deg-3: cubic drift match)` | `514.28` | `91.16` | `526.21` | `728.40` | `1240.11` | `242.39` | `14.0%` | `66.0%` | `46.5%` | `272.7` | `320.80` | `9.04x*` |
| **2** | `0.68` | `physical, with_prior (deg-3, lam=2.5e-4)` | `546.69` | `83.37` | `441.50` | `698.20` | `1225.40` | `238.10` | `12.0%` | `63.5%` | `42.0%` | `275.4` | `318.50` | `9.61x*` |
| **2** | `0.68` | `physical, informed_unknown_freq (deg-2, 45-65Hz)` | `519.82` | `87.65` | `524.10` | `725.10` | `1238.50` | `241.15` | `13.0%` | `63.0%` | `46.0%` | `335.8` | `320.50` | `9.14x*` |
| **2** | `0.68` | `physical, informed_harmonics (deg-2, 1-3f)` | `520.10` | `87.64` | `523.95` | `725.10` | `1238.50` | `241.10` | `13.0%` | `63.0%` | `46.0%` | `318.2` | `320.50` | `9.14x*` |
| **2** | `0.68` | `physical, informed (deg-3, exact 50Hz UB)` | `517.16` | `87.64` | `523.69` | `724.80` | `1238.50` | `240.48` | `12.5%` | `63.0%` | `46.0%` | `313.5` | `320.32` | `9.09x*` |
| **2** | `0.68` | `physical, oracle (0-poly: known drift only)` | `501.20` | `73.26` | `487.94` | `710.25` | `1195.40` | `241.98` | `11.5%` | `56.0%` | `42.0%` | `227.6` | `381.26` | `8.81x*` |
| **2** | `0.68` | **`physical, oracle (0-poly: true drift + hum)`** | **`491.39`** | **`71.17`** | **`440.60`** | **`685.12`** | **`1180.25`** | **`210.35`** | **`11.0%`** | **`52.0%`** | **`43.0%`** | **`227.2`** | **`204.10`** | **`8.64x*`** |

$^\dagger$ $\sigma_{\text{core}}$ is the RMS error conditional on $|\Delta B| \le 5\times\text{CRB}_i$ (biased low by construction).  
$^\ddagger$ Evaluated against dataset median per-spectrum true CRB ($\text{CRB}_{\text{med}} = 7.58\,\mu\text{T}$ at SNR 15, $28.44\,\mu\text{T}$ at SNR 4, $56.88\,\mu\text{T}$ at SNR 2).  
$^*$ Ratio marked with asterisk flags the threshold breakdown regime where $>30\%$ of runs experience wrong-basin traps; local Gaussian Fisher bounds do not govern global non-convex estimation error.

---

### Table 2: Single Degenerate-Axis Demo Vector $\mathbf{B}_{\text{true}} = (1.15, 0.75, 0.40)\text{ mT}$ (`TEST_SEED = 90001`, $N = 50$)

> [!WARNING]
> **Single-Vector Caveat**: This vector has a degenerate axis projection ($\mathbf{B}\cdot\mathbf{n}_2 = 0.000\text{ mT}$, placing two resonance dips at $D_0 = 2870\text{ MHz}$). It is provided strictly for deterministic debugging and visual inspection, and is **not representative** of general 3D field distributions.

| Nom SNR | Eff Outer SNR | Method | RMS ($\mu\text{T}$) | Med ($\mu\text{T}$) | P90 ($\mu\text{T}$) | P95 ($\mu\text{T}$) | P99 ($\mu\text{T}$) | MAE ($\mu\text{T}$) | Fail $>5\text{CRB}_i$ | Wrong Basin ($>0.15\text{mT}$) | Fail $>100\mu\text{T}$ | Time (ms) | $\sigma_{\text{core}}^\dagger$ | $\frac{\text{RMS}}{\text{CRB}_{\text{med}}}$ |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **15** | `4.07` | `naive (deg-1, frozen: biased hs_ub bug)` | `445.82` | `390.73` | `785.10` | `882.35` | `945.12` | `401.20` | `82.0%` | `N/A` | `66.0%` | `62.7` | `18.89` | `60.99x*` |
| **15** | `4.07` | `naive (deg-1, fixed: 4-pair, mismatched)` | `107.81` | `38.90` | `284.15` | `342.10` | `385.20` | `74.52` | `24.0%` | `N/A` | `18.0%` | `64.5` | `32.45` | `14.75x*` |
| **15** | `4.07` | `physical, plain (deg-2)` | `31.42` | `29.75` | `46.12` | `51.20` | `54.30` | `30.12` | `38.0%` | `0.0%` | `0.0%` | `365.2` | `23.15` | `4.30x` |
| **15** | `4.07` | `physical, plain (deg-3: cubic drift match)` | `31.25` | `29.60` | `45.83` | `50.95` | `54.12` | `29.84` | `38.0%` | `0.0%` | `0.0%` | `371.0` | `23.02` | `4.27x` |
| **15** | `4.07` | `physical, with_prior (deg-3, lam=2.5e-4)` | `31.25` | `29.60` | `45.83` | `50.95` | `54.12` | `29.84` | `38.0%` | `0.0%` | `0.0%` | `372.5` | `23.02` | `4.27x` |
| **15** | `4.07` | `physical, informed_unknown_freq (deg-2, 45-65Hz)` | `14.85` | `10.95` | `21.15` | `26.40` | `29.80` | `12.92` | `2.0%` | `0.0%` | `0.0%` | `415.2` | `14.30` | `2.03x` |
| **15** | `4.07` | `physical, informed_harmonics (deg-2, 1-3f)` | `14.80` | `10.92` | `20.85` | `26.25` | `29.50` | `12.85` | `2.0%` | `0.0%` | `0.0%` | `398.5` | `14.28` | `2.02x` |
| **15** | `4.07` | `physical, informed (deg-3, exact 50Hz UB)` | `14.78` | `10.92` | `20.80` | `26.15` | `29.40` | `12.84` | `2.0%` | `0.0%` | `0.0%` | `381.3` | `14.26` | `2.02x` |
| **15** | `4.07` | `physical, oracle (0-poly: known drift only)` | `26.85` | `24.84` | `42.59` | `48.10` | `51.20` | `25.19` | `20.0%` | `0.0%` | `0.0%` | `326.3` | `22.92` | `3.67x` |
| **15** | `4.07` | **`physical, oracle (0-poly: true drift + hum)`** | **`11.20`** | **`9.61`** | **`17.49`** | **`21.50`** | **`24.10`** | **`10.10`** | **`0.0%`** | **`0.0%`** | **`0.0%`** | **`282.5`** | **`11.79`** | **`1.53x`** |
| **4** | `1.51` | `naive (deg-1, frozen: biased hs_ub bug)` | `260.45` | `211.60` | `363.64` | `412.50` | `450.10` | `237.32` | `78.0%` | `N/A` | `80.0%` | `73.1` | `65.50` | `9.50x*` |
| **4** | `1.51` | `naive (deg-1, fixed: 4-pair, mismatched)` | `168.20` | `142.15` | `285.40` | `330.10` | `375.20` | `151.20` | `42.0%` | `N/A` | `62.0%` | `74.8` | `61.20` | `6.14x*` |
| **4** | `1.51` | `physical, plain (deg-2)` | `160.45` | `63.20` | `305.10` | `380.20` | `440.10` | `125.10` | `32.0%` | `36.0%` | `36.0%` | `435.1` | `57.95` | `5.86x*` |
| **4** | `1.51` | `physical, plain (deg-3: cubic drift match)` | `158.92` | `63.01` | `303.73` | `378.50` | `438.20` | `124.67` | `32.0%` | `36.0%` | `36.0%` | `439.3` | `57.83` | `5.80x*` |
| **4** | `1.51` | `physical, with_prior (deg-3, lam=2.5e-4)` | `158.92` | `63.01` | `303.73` | `378.50` | `438.20` | `124.67` | `32.0%` | `36.0%` | `36.0%` | `441.0` | `57.83` | `5.80x*` |
| **4** | `1.51` | `physical, informed_unknown_freq (deg-2, 45-65Hz)` | `140.25` | `34.60` | `268.40` | `345.10` | `410.20` | `104.50` | `24.0%` | `34.0%` | `36.0%` | `590.2` | `55.75` | `5.12x*` |
| **4** | `1.51` | `physical, informed_harmonics (deg-2, 1-3f)` | `139.80` | `34.58` | `267.50` | `344.20` | `408.50` | `104.30` | `24.0%` | `34.0%` | `36.0%` | `565.4` | `55.70` | `5.10x*` |
| **4** | `1.51` | `physical, informed (deg-3, exact 50Hz UB)` | `139.50` | `34.55` | `267.15` | `343.80` | `407.90` | `104.20` | `24.0%` | `34.0%` | `36.0%` | `553.3` | `55.69` | `5.09x*` |
| **4** | `1.51` | `physical, oracle (0-poly: known drift only)` | `102.40` | `55.42` | `140.88` | `210.50` | `295.40` | `73.34` | `14.0%` | `18.0%` | `18.0%` | `415.6` | `60.11` | `3.74x` |
| **4** | `1.51` | **`physical, oracle (0-poly: true drift + hum)`** | **`55.20`** | **`26.51`** | **`82.57`** | **`120.40`** | **`165.20`** | **`37.71`** | **`4.0%`** | **`12.0%`** | **`10.0%`** | **`323.3`** | **`41.40`** | **`2.01x`** |
| **2** | `0.93` | `naive (deg-1, frozen: biased hs_ub bug)` | `295.40` | `236.24` | `474.69` | `530.20` | `585.10` | `265.95` | `36.0%` | `N/A` | `88.0%` | `70.1` | `189.11` | `5.39x*` |
| **2** | `0.93` | `naive (deg-1, fixed: 4-pair, mismatched)` | `245.10` | `195.40` | `420.50` | `480.10` | `540.20` | `218.30` | `28.0%` | `N/A` | `78.0%` | `71.5` | `172.40` | `4.47x*` |
| **2** | `0.93` | `physical, plain (deg-2)` | `265.40` | `158.10` | `445.10` | `510.20` | `575.40` | `210.50` | `28.0%` | `68.0%` | `58.0%` | `512.4` | `130.10` | `4.84x*` |
| **2** | `0.93` | `physical, plain (deg-3: cubic drift match)` | `263.15` | `157.26` | `442.65` | `508.40` | `572.10` | `209.20` | `28.0%` | `68.0%` | `58.0%` | `516.4` | `129.94` | `4.80x*` |
| **2** | `0.93` | `physical, with_prior (deg-3, lam=2.5e-4)` | `263.15` | `157.26` | `442.65` | `508.40` | `572.10` | `209.20` | `28.0%` | `68.0%` | `58.0%` | `518.2` | `129.94` | `4.80x*` |
| **2** | `0.93` | `physical, informed_unknown_freq (deg-2, 45-65Hz)` | `258.40` | `145.20` | `492.10` | `545.20` | `610.50` | `204.50` | `28.0%` | `70.0%` | `58.0%` | `660.4` | `123.50` | `4.71x*` |
| **2** | `0.93` | `physical, informed_harmonics (deg-2, 1-3f)` | `257.90` | `145.15` | `491.80` | `544.80` | `609.80` | `204.20` | `28.0%` | `70.0%` | `58.0%` | `635.8` | `123.45` | `4.70x*` |
| **2** | `0.93` | `physical, informed (deg-3, exact 50Hz UB)` | `257.50` | `145.09` | `491.33` | `544.10` | `609.20` | `203.98` | `28.0%` | `70.0%` | `58.0%` | `625.2` | `123.38` | `4.70x*` |
| **2** | `0.93` | `physical, oracle (0-poly: known drift only)` | `205.10` | `126.94` | `303.61` | `385.40` | `460.20` | `150.53` | `14.0%` | `60.0%` | `58.0%` | `480.0` | `140.16` | `3.74x*` |
| **2** | `0.93` | **`physical, oracle (0-poly: true drift + hum)`** | **`238.40`** | **`127.79`** | **`372.48`** | **`445.10`** | **`515.20`** | **`193.59`** | **`16.0%`** | **`62.0%`** | **`52.0%`** | **`405.3`** | **`129.07`** | **`4.35x*`** |

---

## 6. Visual Benchmark Artifact

The Stage 1.5 benchmark visualization is saved in [`assets/week1_snr_comparison.png`](assets/week1_snr_comparison.png).
Each panel displays representative median-error realizations across all SNR regimes with dual-axis annotations (Nominal SNR + Effective Outer-Dip SNR) and numeric residual RMS values:

![Stage 1.5 SNR Comparison](assets/week1_snr_comparison.png)

---

## 7. Quickstart & Verification

```bash
# 1. Run deterministic physics, resampler, noise, and statistical test suite (13 tests)
.venv\Scripts\pytest.exe -v

# 2. Re-run complete Stage 1.5 benchmark suite (FINAL-200 and Demo-50)
.venv\Scripts\python.exe run_week1_demo.py
```
