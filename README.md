# OpenNV-Denoise: Physics-Informed Quantum Magnetometry & Classical Baseline (Stage 1.5 — Frozen)

**OpenNV-Denoise** is a scientific Python framework for simulating, diagnosing, and inverting Optically Detected Magnetic Resonance (ODMR) spectra from tetrahedral nitrogen-vacancy ($\text{NV}^-$) diamond ensembles under realistic commercial-off-the-shelf (COTS) hardware noise.

---

## 1. Explicit Assumptions & Physical Modelling Choices

To guarantee a fair, non-strawman comparison between the classical nonlinear least-squares (NLLS) physical baseline and downstream neural-network denoisers, all physical modelling assumptions, regularization constants, and evaluation protocols are documented below:

1. **Transverse-Field Contrast Factor ($|\langle 0 | \psi_0 \rangle|^4$)**:
   - In [`src/physics.py`](src/physics.py), each NV axis transition amplitude is scaled by $\text{contrast\_factor} = |\langle m_s = 0 | \psi_0 \rangle|^4$ (`purity_ms0 ** 2`), where $|\psi_0\rangle$ is the eigenstate of the $3 \times 3$ spin-1 ground-state Hamiltonian with the largest $|m_s = 0\rangle$ projection.
   - **Modelling Choice**: This fourth-power dependence is a phenomenological **modelling choice**, not a first-principles derivation. In physical $\text{NV}^-$ ensembles, transverse magnetic fields mix $|m_s = 0\rangle$ with $|m_s = \pm 1\rangle$ in both the ground and excited triplets, modifying intersystem crossing (ISC) rates and steady-state spin polarization in a manner that depends on optical pump intensity and microwave Rabi frequency.

2. **Equal $^{14}\text{N}$ Hyperfine Amplitude Split ($C / 3$)**:
   - Each electronic spin transition ($|0\rangle \leftrightarrow |-1\rangle$ and $|0\rangle \leftrightarrow |+1\rangle$) is split into a triplet of pseudo-Voigt dips separated by $A_{\parallel} = 2.16\text{ MHz}$ ($m_I \in \{-1, 0, +1\}$), with each hyperfine component assigned an equal amplitude of $C_{\text{axis}} / 3$.
   - **Modelling Choice**: This equal $C/3$ partition is a **modelling choice** assuming an unpolarized thermal nuclear spin bath and neglecting transverse hyperfine mixing ($A_{\perp}$), nuclear quadrupolar shifts, and dynamic nuclear polarization (DNP).

3. **Zero-Strain Assumption ($E = 0\text{ MHz}$) & Canonical Wedge Symmetry Guard**:
   - In Stage 1 / 1.5, the transverse crystal-field/strain splitting parameter is $E = 0.0\text{ MHz}$, while all public physics and baseline functions accept `e_mhz: float = 0.0` for forward compatibility.
   - When $E = 0$, the 4 tetrahedral $[111]$ NV axes depend only on the unordered set of unsigned projections $\{|B_{\parallel, k}|\}_{k=1}^4$, which is invariant under the 48-element octahedral group $O_h$ ($2^3 = 8$ coordinate sign flips $\times$ $3! = 6$ coordinate permutations). Thus any vector $\mathbf{B} \in \mathbb{R}^3$ has a unique canonical representative in $0 \le B_x \le B_y \le B_z$.
   - Because non-zero strain ($E \ne 0$) breaks this 48-fold octahedral symmetry, [`canonicalize_b_vec`](src/physics.py) and [`fit_physical_odmr`](src/baseline.py) explicitly raise a `ValueError` whenever `e_mhz != 0.0`.

4. **Variable-Projection Regularization Constants & Three-Way Seed Separation**:
   - All variable-projection ridge weights in [`_VarProjDesign`](src/baseline.py)—linear baseline ridge `[0.25, 0.8]`, degree-2 Legendre curvature penalty `18.0`, degree-3 Legendre cubic penalty `36.0`, sinusoidal mains-hum ridge `8.0`, outer wing anchor weight `wing_weight = 0.20` on $[2750, 2775] \cup [2965, 2990]\text{ MHz}$, and default polynomial degree `poly_degree = 3`—were tuned exclusively on the **Development Set (`DEV_SEED = 10001`)**.
   - Three strictly disjoint seed sets are enforced:
     - **`DEV_SEED = 10001`**: Used for Part 0 diagnostics and hyperparameter/design decisions.
     - **`TEST_SEED = 90001`**: Used for deterministic unit tests and the single degenerate-axis demo vector $\mathbf{B} = (1.15, 0.75, 0.40)\text{ mT}$ (not representative of general 3D directions).
     - **`FINAL_SEED = 500001`**: Held-out final evaluation set of 200 spectra on $S^2$ (including 20% near-degenerate symmetry planes), never used for any tuning decision prior to freezing `src/baseline.py`.

5. **Cramér-Rao Lower Bound (CRB) Assumptions & Statistical Efficiency Metric (`sigma_core / CRB`)**:
   - The reference CRB reported in [`run_week1_demo.py`](run_week1_demo.py) assumes white intensity-scaled Gaussian noise (neglecting $1/f$ pink noise, cubic thermal drift, and AC mains hum), evaluated at fixed reference $\text{HWHM} = 4.5\text{ MHz}$ and contrast $C = 0.03$.
   - Because $\text{median}(|\varepsilon|) = 0.6745\,\sigma$ for a Gaussian estimator, comparing $\text{median}(|\Delta B|)$ directly against $\text{CRB}$ understates estimator variance by $1 / 0.6745 \approx 1.48\times$. We therefore report $\sigma_{\text{core}} / \text{CRB}$, where $\sigma_{\text{core}}$ is the RMS error of non-failed runs ($|\Delta B| \le 5\times\text{CRB}$).
   - Furthermore, the CRB is a local bound around the true basin and is **not valid in the threshold breakdown regime** ($>30\%$ relative failures, marked with `*` in benchmark tables).

6. **HWHM Bounds & Pseudo-Voigt Mixing Factor (`_DEFAULT_ETA = 0.35`) Policy**:
   - [`fit_physical_odmr`](src/baseline.py) accepts `hwhm_bounds: Tuple[float, float] = (3.0, 9.0)` (matching the training distribution `TRAIN_LINEWIDTH_RANGE`) and `eta: float = 0.35`.
   - **Out-of-Distribution (OOD) Policy**: Any OOD linewidth evaluation (`OOD_LINEWIDTH_RANGE = (9.5, 13.5)` MHz) comparing `fit_physical_odmr` against a neural network **must** pass `hwhm_bounds=(2.0, 16.0)` so the classical fitter is not artificially clamped at `9.0 MHz`.

7. **Intensity-Dependent Noise Scaling**:
   - In [`src/noise.py`](src/noise.py), stochastic noise is modulated by $\sqrt{\text{clip}(S_{\text{clean}}, 0.5, 1.5)}$. This models intensity-dependent Gaussian noise scaling (high-photon-count Gaussian approximation) rather than discrete Poisson photon counting.

---

## 2. Verification of Existing Unit Tests (`git diff tests/`)

No existing test threshold was weakened, loosened, or deleted:
- In `test_b_mag_from_splittings_orders_on_exact_eigh`, implementing the exact second-order level-repulsion term `(b^2 / (2*D0))^2` in Task 8 reduced `order=2` RMS error from `0.2302 uT` to **`0.0002 uT`** (435$\times$ better than `order=1`'s `0.0871 uT`), so `assert rms1_ut < rms2_ut` was updated to `assert rms2_ut < rms1_ut` alongside tightened bounds `assert rms2_ut < 0.002` and `assert max2_ut < 0.005` while preserving all original upper bounds (`rms1_ut < 0.15`, `max1_ut < 0.65`, `rms2_ut < 0.30`, `max2_ut < 0.85`).
- Three new tests (`test_noise_paths_identical_and_components`, `test_resampler_valid_mask_end_to_end_and_out_of_range`, `test_naive_fitter_sanity_snr15`) were appended at the end of [`tests/test_physics.py`](tests/test_physics.py).

---

## 3. Summary of Hypothesis Checks & Task-by-Task Measurements (Tasks 1–8)

### Task 1: Diagnosis & Rewrite of `fit_lorentzian_naive`
- **Task 1a HYPOTHESIS Check (10 Failing Realizations at SNR=15 on $\mathbf{B}=(1.15, 0.75, 0.40)\text{ mT}$, True Splittings `[0.56, 25.89, 48.54, 74.43]` MHz)**:
  - **Confirmed**: In all 10 trials of the legacy 8-free-Lorentzian fitter, 2 to 4 unconstrained Lorentzians drifted into the outer noise/drift wings (`2755–2803 MHz` and `2945–2984 MHz`) with widths slammed against the `1.5 MHz` lower bound or `18.0 MHz` upper bound. Sorting the 8 centres (`c7 - c0`, `c6 - c1`, ...) paired those wing noise spikes together:
    - `Trial 0`: `centres = [2790.6, 2803.0, 2814.9, 2844.6, 2871.4, 2892.8, 2908.4, 2945.6] MHz`, `widths = [1.5, 1.5, 1.5, 18.0, 6.53, 15.04, 2.25, 1.5] MHz`, `splittings = [26.82, 77.93, 105.35, 154.95] MHz` $\to$ `|B_est| = 3.1628 mT (err = 1732.8 uT)`.
    - `Trial 2`: `centres = [2760.8, 2832.4, 2844.6, 2856.1, 2870.7, 2895.8, 2938.3, 2972.7] MHz`, `splittings = [14.52, 51.25, 105.85, 211.87] MHz` $\to$ `|B_est| = 3.7508 mT (err = 2320.7 uT)`.
    - `Trial 5`: `centres = [2757.1, 2770.4, 2781.6, 2792.9, 2866.5, 2901.0, 2969.8, 2982.3] MHz`, `splittings = [73.60, 119.46, 199.41, 225.15] MHz` $\to$ `|B_est| = 5.1279 mT (err = 3697.8 uT)`.
- **Task 1b/1c Rewrite & Before/After**:
  - Rewritten in [`fit_lorentzian_naive`](src/baseline.py) as 4 symmetric Lorentzian pairs $d_0 \pm \text{hs}_k$ around one common center $d_0 \in [2865, 2875]\text{ MHz}$, shared HWHM $w \in [2, 18]\text{ MHz}$, 4 non-negative pair amplitudes, and a wing-regularized linear baseline, initialized from mirror-folded peaks above the outer wing noise floor (no hyperfine, no Hamiltonian coupling):
  - **Demo Vector ($\text{SNR}=15$)**: Median $|\Delta B|$ improved from **`1914.25 uT` $\to$ `390.73 uT`** (`P90: 3105.61 -> 785.10 uT`, `MAE: 1979.09 -> 401.20 uT`, `Time: 142.8 -> 62.7 ms`).
  - **Dev-200 ($\text{SNR}=15$)**: Median $|\Delta B| =$ **`261.69 uT`**; **FINAL-200 ($\text{SNR}=15$)**: Median $|\Delta B| =$ **`346.84 uT`** (well below the $1\text{ mT} = 1000\,\mu\text{T}$ acceptance criterion).

### Task 2: True Oracle Baseline (`1.0 + drift + hum` vs. `1.0 + drift`)
- Passing the complete true baseline `1.0 + drift + hum` (from `apply_realistic_noise(..., return_components=True)`) and removing the `0.25` upper amplitude clip makes the oracle a true lower-error reference across all SNR regimes:
  - **Demo Vector ($\text{SNR}=15$)**: `Known Drift Only = 24.84 uT` vs. `Informed (Inverse Crime UB) = 10.92 uT` vs. **`True Drift + Hum Oracle = 9.61 uT`** (`P90 = 17.49 uT`, `Fail>5xCRB = 0.0%`).
  - **Demo Vector ($\text{SNR}=4$)**: `Known Drift Only = 55.42 uT` vs. `Informed = 34.55 uT` vs. **`True Drift + Hum Oracle = 26.51 uT`** (`P90 = 82.57 uT`, `Fail>5xCRB = 4.0%`).
  - **FINAL-200 ($\text{SNR}=15 / 4 / 2$)**: `True Drift + Hum Oracle` achieves the lowest median (`7.49 / 30.43 / 71.17 uT`), lowest MAE (`12.69 / 52.15 / 210.35 uT`), and lowest failure rate (`3.0% / 6.0% / 14.0%`) of all methods.

### Task 3: Removing Test-Vector Leakage from Stage A (`_build_canonical_wedge_directions`)
- Replaced the 15 hand-picked directions with 16 deterministic directions in [`_build_canonical_wedge_directions`](src/baseline.py): 3 canonical wedge vertices (`[0,0,1]`, `[0,1,1]/sqrt(2)`, `[1,1,1]/sqrt(3)`), 5 geometric symmetry-plane points (`Bx=0`, `Bx=By`, `By=Bz` midpoints and two rational points $t \in \{1/4, 2/5\}$ on $\mathbf{B}\cdot\mathbf{n}_2 = 0$), and 8 equal-area 2D Halton $(2,3)$ spherical octant points canonicalized into $0 \le B_x \le B_y \le B_z$.
- **Dev-200 Before vs. After (with `poly_degree=2`)**:
  - `SNR = 15`: `Median: 12.01 -> 11.75 uT` | `P90: 42.75 -> 42.76 uT` | `MAE: 17.45 -> 17.35 uT` | `Fail>100uT: 0.5% -> 0.5%` | `WrongBasin: 10.5% -> 10.5%`
  - `SNR = 4`: `Median: 36.63 -> 36.05 uT` | `P90: 145.38 -> 145.31 uT` | `MAE: 70.38 -> 72.06 uT` | `Fail>100uT: 17.5% -> 18.5%` | `WrongBasin: 34.5% -> 36.0%`
  - `SNR = 2`: `Median: 82.96 -> 80.57 uT` | `P90: 691.14 -> 541.87 uT` | `MAE: 277.32 -> 260.00 uT` | `Fail>100uT: 45.0% -> 43.5%` | `WrongBasin: 58.0% -> 57.5%`

### Task 5: OOD HWHM Bounds (`[2, 16]` MHz) & Simulator `eta` Randomization on Dev-200
- **Task 5b (200 Dev Spectra Simulated with OOD $\text{HWHM} \sim \mathcal{U}(10.0, 13.0)\text{ MHz}$ at $\text{SNR}=15$)**:
  - Default `hwhm_bounds=(3.0, 9.0)` MHz: `|B| Median = 26.40 uT | P90 = 168.15 uT | MAE = 79.47 uT | Fail>100uT = 17.5% | HWHM MAE = 2.84 MHz`
  - Widened `hwhm_bounds=(2.0, 16.0)` MHz: **`|B| Median = 23.61 uT | P90 = 140.59 uT | MAE = 66.03 uT | Fail>100uT = 13.5% | HWHM MAE = 1.62 MHz`**
- **Task 5c (200 Dev Spectra Simulated with Randomized $\eta \sim \mathcal{U}(0.0, 1.0)$ at $\text{SNR}=15$)**:
  - Fixed fitter `eta = 0.35`: `Median = 12.28 uT | P90 = 43.43 uT | MAE = 19.72 uT | Signed Bias = +2.36 uT | Fail>100uT = 1.5%`
  - True simulator `eta`: `Median = 11.59 uT | P90 = 45.60 uT | MAE = 19.21 uT | Signed Bias = +2.75 uT | Fail>100uT = 1.5%`
  - Fixing `eta = 0.35` induces only `+0.69 uT` median difference and `-0.39 uT` signed bias shift because dip centres are symmetric with respect to line-shape mixing $\eta$.

### Task 6: Resampler `valid_mask`, `out_of_range` Detection, & Cell-Averaging Downsampling
- [`resample_spectrum_pchip`](src/noise.py) returns `valid_mask` (`False` outside `[orig_min, orig_max]`) and `out_of_range` (when resonances approach within $2\,\text{HWHM} + 2 A_{\parallel}$ of the window boundary), using deterministic edge-median fill instead of random noise.
- **Task 6c (1001 $\to$ 512 Downsampling Comparison)**:
  - Clean spectrum RMS vs. exact 512-point evaluation: `1.55e-6` (cell-averaging via exact PCHIP antiderivative integration) vs. `1.65e-7` (Savitzky-Golay).
  - Noisy spectrum ($\sigma = 0.005$, 200 trials) anti-aliasing RMS vs. clean 512-point ground truth: **`0.003231` (cell-averaging)** vs. `0.003365` (Savitzky-Golay window=5).

### Task 7b: HYPOTHESIS Check on Cubic Drift Mismatch (`poly_degree` in `{1, 2, 3}` on Dev-200)
- **Confirmed**: Because `apply_realistic_noise` generates thermal drift with linear, quadratic, and cubic terms ($u, u^2 - 1/3, u^3$), restricting the fitter baseline to `poly_degree = 1` substantially degrades accuracy, while adding orthogonal cubic Legendre $P_3(u) = u^3 - 0.6 u$ (`poly_degree = 3`, penalty `36.0`) achieves the lowest Dev-200 MAE, P90, and failure rate across all SNR regimes:
  - `SNR = 15`: `deg=1: Med=14.65, MAE=26.50, Fail=2.5%` | `deg=2: Med=11.75, MAE=17.35, Fail=0.5%` | **`deg=3: Med=11.49 uT, P90=42.68 uT, MAE=17.32 uT, Fail=0.5%`**
  - `SNR = 4`: `deg=1: Med=39.15, MAE=107.16, Fail=25.5%` | `deg=2: Med=36.05, P90=145.31, MAE=72.06, Fail=18.5%` | **`deg=3: Med=36.46 uT, P90=138.10 uT, MAE=68.48 uT, Fail=16.5%`**
  - `SNR = 2`: `deg=1: Med=104.44, MAE=321.47, Fail=50.5%` | `deg=2: Med=80.57, P90=541.87, MAE=260.00, Fail=43.5%` | **`deg=3: Med=81.90 uT, P90=513.12 uT, MAE=252.27 uT, Fail=42.0%`**
  - Selected `poly_degree = 3` on DEV.

### Task 8: `b_mag_from_splittings` Second-Order Inversion with Degenerate Level Repulsion
- Validated on **20,000 random 4-axis directions** on $S^2$ with $|\mathbf{B}| \sim \mathcal{U}(0.1, 3.2)\text{ mT}$ against exact `np.linalg.eigh` splittings:
  - Forward splitting approximation max error: **`0.0027 MHz`** for $\Delta f_i \approx 2\sqrt{[a_i(1 - b_i^2/(2(D_0^2 - a_i^2)))]^2 + (b_i^2/(2 D_0))^2}$ vs. `2.7462 MHz` for linear $2 a_i$.
  - **`order=1` (Default)**: `RMS = 0.0871 uT | MAE = 0.0492 uT | Max = 0.4955 uT | Bias = -0.0002 uT`
  - **`order=2` (5-iteration inversion)**: **`RMS = 0.0002 uT | MAE = 0.0001 uT | Max = 0.0008 uT | Bias = -0.0000 uT`** ($435\times$ lower RMS error; `order=1` kept as default per Task 8a).

---

## 4. Task 9: Final Evaluation Tables (`FINAL_SEED = 500001` & Demo Vector `TEST_SEED = 90001`)

### Table 1: Held-Out FINAL Set (`FINAL_SEED = 500001`, $N = 200$ Spectra on $S^2$ incl. 20% Near-Degenerate, $|\mathbf{B}| \in [0.1, 3.2]\text{ mT}$)

| Nom SNR | Eff Outer SNR | Method | Median ($\mu\text{T}$) | P90 ($\mu\text{T}$) | MAE ($\mu\text{T}$) | $\sigma_{\text{core}}$ ($\mu\text{T}$) | Fail $>5\times\text{CRB}$ | Wrong Basin ($>0.15\text{ mT}$) | Fail $>100\,\mu\text{T}$ | Time (ms) | $\sigma_{\text{core}} / \text{CRB}$ |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **15** | `4.07` | `(naive, 4-symmetric-pair Lorentzian)` | `346.84` | `1004.50` | `445.62` | `173.07` | `82.0%` | `N/A (1D)` | `66.5%` | `56.6` | `29.10x*` |
| **15** | `4.07` | **`(physical, plain)`** | **`11.81`** | **`39.77`** | **`20.07`** | **`33.56`** | **`9.5%`** | **`8.0%`** | **`3.0%`** | **`261.6`** | **`5.54x`** |
| **15** | `4.07` | `(physical, informed: inverse crime, UB)` | `7.52` | `30.87` | `14.96` | `26.47` | `4.0%` | `4.5%` | `2.0%` | `283.4` | `4.56x` |
| **15** | `4.07` | `(physical, oracle: known drift only)` | `11.48` | `36.15` | `17.41` | `24.86` | `7.0%` | `5.5%` | `1.0%` | `249.0` | `4.18x` |
| **15** | `4.07` | **`(physical, oracle: true drift + hum)`** | **`7.49`** | **`27.78`** | **`12.69`** | **`20.90`** | **`3.0%`** | **`1.5%`** | **`0.5%`** | **`261.5`** | **`3.64x`** |
| **4** | `1.51` | `(naive, 4-symmetric-pair Lorentzian)` | `337.27` | `969.07` | `426.45` | `272.82` | `59.0%` | `N/A (1D)` | `74.0%` | `58.9` | `8.17x*` |
| **4** | `1.51` | **`(physical, plain)`** | **`42.36`** | **`159.42`** | **`69.38`** | **`75.65`** | **`11.0%`** | **`37.0%`** | **`20.5%`** | **`268.5`** | **`3.52x`** |
| **4** | `1.51` | `(physical, informed: inverse crime, UB)` | `35.51` | `149.98` | `63.92` | `71.70` | `7.0%` | `32.0%` | `18.5%` | `293.5` | `3.35x` |
| **4** | `1.51` | `(physical, oracle: known drift only)` | `39.97` | `148.99` | `59.30` | `73.21` | `9.0%` | `31.0%` | `18.5%` | `209.5` | `3.28x` |
| **4** | `1.51` | **`(physical, oracle: true drift + hum)`** | **`30.43`** | **`132.23`** | **`52.15`** | **`66.50`** | **`6.0%`** | **`22.5%`** | **`14.0%`** | **`206.6`** | **`3.06x`** |
| **2** | `0.93` | `(naive, 4-symmetric-pair Lorentzian)` | `377.67` | `1228.99` | `507.94` | `384.06` | `55.5%` | `N/A (1D)` | `77.0%` | `64.5` | `6.97x*` |
| **2** | `0.93` | **`(physical, plain)`** | **`91.16`** | **`526.21`** | **`242.39`** | **`320.80`** | **`18.5%`** | **`66.0%`** | **`46.5%`** | **`272.7`** | **`7.06x`** |
| **2** | `0.93` | `(physical, informed: inverse crime, UB)` | `87.64` | `523.69` | `240.48` | `320.32` | `18.0%` | `63.0%` | `46.0%` | `313.5` | `7.04x` |
| **2** | `0.93` | `(physical, oracle: known drift only)` | `73.26` | `487.94` | `241.98` | `381.26` | `15.0%` | `56.0%` | `42.0%` | `227.6` | `8.90x` |
| **2** | `0.93` | **`(physical, oracle: true drift + hum)`** | **`71.17`** | **`440.60`** | **`210.35`** | **`204.10`** | **`14.0%`** | **`52.0%`** | **`43.0%`** | **`227.2`** | **`4.77x`** |

### Table 2: Single Degenerate-Axis Demo Vector $\mathbf{B}_{\text{true}} = (1.15, 0.75, 0.40)\text{ mT}$ (`TEST_SEED = 90001`, 50 MC Trials / Regime — *Single Degenerate-Axis Vector; Not Representative*)

| Nom SNR | Eff Outer SNR | Method | Median ($\mu\text{T}$) | P90 ($\mu\text{T}$) | MAE ($\mu\text{T}$) | $\sigma_{\text{core}}$ ($\mu\text{T}$) | Fail $>5\times\text{CRB}$ | Wrong Basin ($>0.15\text{ mT}$) | Fail $>100\,\mu\text{T}$ | Time (ms) | $\sigma_{\text{core}} / \text{CRB}$ |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **15** | `4.07` | `(naive, 4-symmetric-pair Lorentzian)` | `390.73` | `785.10` | `401.20` | `18.89` | `82.0%` | `N/A (1D)` | `66.0%` | `62.7` | `2.58x*` |
| **15** | `4.07` | `(physical, plain)` | `29.60` | `45.83` | `29.84` | `23.02` | `38.0%` | `0.0%` | `0.0%` | `371.0` | `3.15x*` |
| **15** | `4.07` | `(physical, informed: inverse crime, UB)` | `10.92` | `20.80` | `12.84` | `14.26` | `2.0%` | `0.0%` | `0.0%` | `381.3` | `1.95x` |
| **15** | `4.07` | `(physical, oracle: known drift only)` | `24.84` | `42.59` | `25.19` | `22.92` | `20.0%` | `0.0%` | `0.0%` | `326.3` | `3.13x` |
| **15** | `4.07` | **`(physical, oracle: true drift + hum)`** | **`9.61`** | **`17.49`** | **`10.10`** | **`11.79`** | **`0.0%`** | **`0.0%`** | **`0.0%`** | **`282.5`** | **`1.61x`** |
| **4** | `1.51` | `(naive, 4-symmetric-pair Lorentzian)` | `211.60` | `363.64` | `237.32` | `65.50` | `78.0%` | `N/A (1D)` | `80.0%` | `73.1` | `2.39x*` |
| **4** | `1.51` | `(physical, plain)` | `63.01` | `303.73` | `124.67` | `57.83` | `32.0%` | `36.0%` | `36.0%` | `439.3` | `2.11x*` |
| **4** | `1.51` | `(physical, informed: inverse crime, UB)` | `34.55` | `267.15` | `104.20` | `55.69` | `24.0%` | `34.0%` | `36.0%` | `553.3` | `2.03x` |
| **4** | `1.51` | `(physical, oracle: known drift only)` | `55.42` | `140.88` | `73.34` | `60.11` | `14.0%` | `18.0%` | `18.0%` | `415.6` | `2.19x` |
| **4** | `1.51` | **`(physical, oracle: true drift + hum)`** | **`26.51`** | **`82.57`** | **`37.71`** | **`41.40`** | **`4.0%`** | **`12.0%`** | **`10.0%`** | **`323.3`** | **`1.51x`** |
| **2** | `0.93` | `(naive, 4-symmetric-pair Lorentzian)` | `236.24` | `474.69` | `265.95` | `189.11` | `36.0%` | `N/A (1D)` | `88.0%` | `70.1` | `3.45x*` |
| **2** | `0.93` | `(physical, plain)` | `157.26` | `442.65` | `209.20` | `129.94` | `28.0%` | `68.0%` | `58.0%` | `516.4` | `2.37x` |
| **2** | `0.93` | `(physical, informed: inverse crime, UB)` | `145.09` | `491.33` | `203.98` | `123.38` | `28.0%` | `70.0%` | `58.0%` | `625.2` | `2.25x` |
| **2** | `0.93` | `(physical, oracle: known drift only)` | `126.94` | `303.61` | `150.53` | `140.16` | `14.0%` | `60.0%` | `58.0%` | `480.0` | `2.55x` |
| **2** | `0.93` | **`(physical, oracle: true drift + hum)`** | **`127.79`** | **`372.48`** | **`193.59`** | **`129.07`** | **`16.0%`** | **`62.0%`** | **`52.0%`** | **`405.3`** | **`2.35x`** |

![Stage 1.5 SNR Comparison](assets/week1_snr_comparison.png)

### Honest Report of Every Target That Was Not Met on the Held-Out FINAL Set
1. **SNR = 15 (`plain` P90 and Failure Rate on FINAL-200)**:
   - Provisional target: `Median <= 15 uT`, `P90 <= 35 uT`, `Fail < 2%`.
   - On `FINAL-200`, `informed` (`Med = 7.52 uT, P90 = 30.87 uT, Fail>100uT = 2.0%`) and `oracle (true drift + hum)` (`Med = 7.49 uT, P90 = 27.78 uT, Fail>100uT = 0.5%`) meet the targets; `plain` achieves **`Med = 11.81 uT`** (meeting `<= 15 uT`), while its **`P90 = 39.77 uT`** (vs. `35 uT`), **`Fail>100uT = 3.0%`** (vs. `< 2%`), and **`Fail>5xCRB = 9.5%`** slightly exceed the provisional targets because unmodelled 50 Hz mains hum (`wavelength = 11.72 MHz`) perturbs broad linewidth spectra (`gamma in [7.5, 9.0] MHz`) when its period matches adjacent Zeeman splittings.
2. **SNR = 4 (`plain` and `informed` P90 / Failure Rate on FINAL-200)**:
   - Provisional target: `Median <= 40 uT`, `P90 <= 90 uT`, `Fail < 10%`.
   - On `FINAL-200`, `informed` (`Med = 35.51 uT, Fail>5xCRB = 7.0%`) and `oracle` (`Med = 30.43 uT, Fail>5xCRB = 6.0%`) meet the median and relative-failure targets, while `plain` has **`Med = 42.36 uT`** (`Fail>5xCRB = 11.0%`), and all variants exceed `P90 <= 90 uT` (`P90 = 132.23–159.42 uT`, `WrongBasin = 22.5%–37.0%`) because the effective SNR of the outer informative dips at nominal SNR=4 is only **`1.51`** and the 90th-percentile white-noise CRB across random directions and linewidths is already `> 110 uT`.
3. **SNR = 2 (P90 Target `<= 220 uT` on FINAL-200)**:
   - Provisional target: `Median <= 90 uT`, `P90 <= 220 uT`.
   - `informed` (`Med = 87.64 uT`) and `oracle` (`Med = 71.17 uT`) meet `Median <= 90 uT` (`plain` is `91.16 uT`), while `P90` (`440.60–526.21 uT`) and `WrongBasin` (`52.0%–66.0%`) exceed `220 uT` across all methods—including the true drift+hum oracle—because at nominal SNR=2 the outer informative dip effective SNR is **`0.93` ($< 1$)** and the 90th-percentile white-noise CRB alone exceeds `220 uT`.

---

## 5. Quickstart & Verification

```bash
# Run the 13-test deterministic physics, resampler, noise, and statistical suite
.venv\Scripts\pytest.exe -v

# Execute the Stage 1.5 FINAL-200 + Demo-50 benchmark and regenerate assets/week1_snr_comparison.png
.venv\Scripts\python.exe run_week1_demo.py
```
