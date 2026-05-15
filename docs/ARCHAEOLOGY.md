# Fulcher Analyzer — Phase 1 Archaeology Report

> Scope: read-only mapping of the existing codebase. **No refactoring, no
> deletions, no API changes.** This report defines what the working pipeline
> is, where it lives, what is safe to extract later, and what is dangerous to
> touch without numerical regression tests.

Smoke-tested in `~/.venvs/fulcher` (Python 3.12, numpy/pandas/scipy/
matplotlib/sympy/xarray latest). The full notebook pipeline reproduces in
both isotopologues — see "Reference numerical outputs" below.

---

## 1. Repository layout at a glance

```
fulcheranalyzer/
├── fulcheranalyzer/              # OLD package — THE WORKING PIPELINE
│   ├── __init__.py               # version stub only
│   ├── _constants.py             # package_directory path helper
│   └── coronalmodel.py           # 2073 lines: everything lives here
├── src/
│   └── fulcher_analyser/         # PARTIAL REFACTOR (note: typo "analyser")
│       ├── __init__.py           # path hack, no exports
│       ├── __main__.py           # tiny CLI wrapper
│       ├── molecular_data.py     # Fulcher-α wavelength dict
│       └── spectrum.py           # xarray-based spectrum loader
├── data/                         # experimental intensity CSVs + 1 NetCDF
│   ├── 143306.nc                 # orphan, NOT used by either pipeline
│   ├── 150482_fr_7.csv (+_err)   # D2 Q-branch intensities, shot 150482
│   ├── 152478_fr_10.csv (+_err)  # H2 Q-branch intensities, shot 152478
│   └── figs7-8_h.svg             # plot output accidentally tracked
├── data_molecular/               # molecular constants + R-matrix cache
│   ├── fulcher-α_band_wavelength.txt           (H2, [nm])
│   ├── fulcher-α_band_wavenumber_D2.txt        (D2, [cm⁻¹])
│   ├── franck_condon_factor{,_D2}.txt
│   ├── vibrational_energy{,_D2}.txt
│   ├── excitation_vibrational_energy{,_D2}.txt
│   ├── Rmatrix_2_11_2_10_h.npy   shape (3,12,3,11)  ← cache, regenerable
│   └── Rmatrix_3_14_3_13_d.npy   shape (4,15,4,14)  ← cache, regenerable
├── echelle_spectra/
│   ├── data/                     # empty
│   └── out/151953_echelle_spec.txt   # raw 40-frame echelle spectrum
├── examples/
│   ├── CoronalModel-D-H.ipynb    # ← CANONICAL WORKFLOW (paper figs 7–12)
│   └── load_spectra.ipynb        # demo for the partial src/ refactor
├── docs/                         # Sphinx build of OLD package only
│   └── source/examples/CoronaModelExample.ipynb  # near-duplicate of canonical nb
├── pics/                         # SVG/PNG outputs of paper figures
├── pyproject.toml                # ⚠ project name = "cnlecture" (wrong)
├── requirements.txt              # pinned 2022 versions, incomplete
├── .readthedocs.yaml
└── LICENSE / Readme.md / .gitignore
```

Git history is short and revealing — six commits total:

```
bbef1b7  added load_spectra.ipynb, docs update     ← the abandoned refactor
1405859  Reworking and organising package
96218d4  Docs Update
caef379  Docs update
6beea00  Docs + Readme
75db391  initial commit                            ← the entire pipeline
```

Everything that runs the paper analysis was added in the **initial commit**.
The `src/fulcher_analyser/` tree (the partial refactor by A. Liptak) was
added later in `bbef1b7` and never integrated.

---

## 2. The working pipeline: `examples/CoronalModel-D-H.ipynb`

The notebook is 26 cells, structured as two symmetric blocks (Deuterium,
then Hydrogen). It does **not** load raw spectra — it consumes the already-
integrated Q-branch intensity CSVs in `data/`. Spectrum loading / Gaussian
deconvolution is *not* part of this notebook; only `data/*.csv` is read.

### Cell-by-cell trace (D2 path; H2 is identical with different shot args)

| Cell | Code | Calls in `fulcheranalyzer.coronalmodel` |
|------|------|-----------------------------------------|
| 1 | `sys.path.insert(0,"../")` + `from fulcheranalyzer import coronalmodel as fcm` | bootstrap, no install |
| 4 | `inte = fcm.read_intensities(150482, 7)` | `read_intensities` → `pd.read_csv` on `data/{shot}_fr_{frame}.csv` + `_err.csv` |
| 4 | `bp = fcm.BoltzmannPlot(inte, 'd')` | `__init__` chains: `load_wavelength_data → trim_wavelength → prep_mol → bplot_constants → calculate_boltzmann → prep_fit → set_style`. `prep_mol` instantiates `MolecularConstants()` which runs `create_dataframes / calculate_tfrac / general_constants / calculate_all_E_rot / acoeff / load_corona_constants / load_wavelength_data`. |
| 4 | `bp.autofit()` | `fit_boltzmann` (scipy `curve_fit` on `two_t_all_v`) → `calc_nd_const` → `calc_all_rot_temp`. Produces `popt = [α, β, T_rot1, T_rot2, c0…c_{v}]` and synthetic populations `nd_synth`, `nd_sc`, `nd_vibrofit`. |
| 6 | `bp.plot_popd_paper(stylename='color', ms=4)` + `fcm.set_tick_size` + `plt.savefig(figs7-8_*.svg)` | paper Figure 7/8 |
| 7 | `bp.print_fit_result(); bp.trotall.style.format(...)` | textual report |
| 8 | `como = fcm.CoronaModel(bp)` | `__init__` → `prep_constants → prep_style → prep_corona_fit(load=True)`; `prep_corona_fit` calls `set_pop_shape` + `make_rmatrix(load=True)` which **tries to `np.load` `Rmatrix_{vx}_{jx}_{vd}_{jd}_{iso}.npy`** before recomputing via `R_formula`/`branching` (sympy `wigner_3j`). |
| 8 | `como.print_pop_shape()` | printout |
| 8 | `como.coronal_autofit()` | scipy `curve_fit` on `coronal_fit_formula` → fits **T_vib** only (α, β, T_rot inherited from `bp`). Calls `calc_errorbars`. |
| 9 | `como.plot_coronal_result()` → `plt.savefig(Corona_*.png)` | paper Figure 4.24 / equivalent |
| 11 | `como.plot_popx_paper(ms=4)` → `plt.savefig(figs9-10_*.svg)` | paper Figures 9/10 |
| 13 | `como.plot_paper_compare(yerr=como.yerr_flat,ms=4)` → `plt.savefig(figs11-12_*.svg)` | paper Figures 11/12 |
| 14 | (commented out) `como.plot_contribution(7000)` | optional |

`load_spectra.ipynb` is on a **different code path entirely** — it imports
`src/fulcher_analyser/spectrum.py` and reads `echelle_spectra/out/151953_echelle_spec.txt`.
It does not feed anything into the Coronal-Model pipeline.

### Reference numerical outputs (regression targets)

Re-running the full notebook on the committed CSV data on 2026-05-15
produced:

```
=== D2 (shot 150482, frame 7) ===
intensities shape: (14, 4)
Trot1 = 279.94   Trot2 = 1782.83   alpha = 0.7609   beta = 0.4159
T_vib(D2) = 7743 ± 557 K

=== H2 (shot 152478, frame 10) ===
intensities shape: (11, 3)
Trot1 = 374.72   Trot2 = 2079.14   alpha = 0.7335   beta = 0.4473
T_vib(H2) = 6801 ± 753 K
```

These six numbers per isotope (plus full `popt` / covariance) are **the
non-negotiable regression targets** for any future refactor. They should
become the assertions in a `tests/test_paper_reproduction.py`.

---

## 3. Module / function inventory

### 3.1 `fulcheranalyzer/coronalmodel.py` (single-file, 2073 lines)

Conceptually it contains five concerns interleaved. The natural future
boundaries are drawn below; **for Phase 1 we only label them**.

| Section | Lines | Public symbols | Future module suggestion |
|---|---|---|---|
| Path constants | 5–7 | `ABSOLUTESIGMA`, `MOLECULAR_DATA_FOLDER`, `DATA_FOLDER` | `paths.py` / config |
| `MolecularConstants` | 10–312 | class with: `create_dataframes`, `tfrac`, `calculate_tfrac`, `load_wavelength_data`, `E_rot_formula`, `calculate_E_rot`, `E_vib_formula`, `calculate_E_vib`, `calculate_all_E_rot`, `acoeff`, `calculate_spin_multiplicity`, `load_corona_constants` | `molecular_constants.py` |
| Boltzmann helpers | 322–451 | `expsum`, `two_t_all_v`, `plot_n_all_v`, `fittext`, `write_intensities`, `read_intensities` | `boltzmann/` + `io/intensities.py` |
| `BoltzmannPlot` | 461–901 | `load_wavelength_data` (dup of `MolecularConstants`), `trim_wavelength`, `prep_mol`, `bplot_constants`, `calculate_boltzmann`, `plot_boltzmannn`, `prep_fit`, `plot_fit`, `print_fit_result`, `fit_boltzmann`, `calc_nd_synth`, `plot_fit_nice`, `calc_nd_const`, `plot_nd`, `calc_all_rot_temp`, `autofit`, `set_style`, `plot_popd_paper`, `about_var` | `boltzmann/d_state_fit.py` (logic) + `plotting/boltzmann.py` (plots) |
| `CoronaModel` | 911–1899 | `prep_style`, `prep_constants`, `calculate_e_cross`, `f_vibro`, `fit_vibro_ishi`, `plot_fit_ishi`, `branching`, `calc_branching_matrix`, `calc_delta`, `ccs_formula`, `calc_rtp`, `R_formula`, `prep_corona_fit`, `set_pop_shape`, `print_pop_shape`, `make_rmatrix`, `make_rmatrix_2d`, `check_rmatrix_indexing`, `calc_X_bp`, `calc_nx`, `calc_nd`, `calc_ndx`, `plot_xd`, `plot_xd_flat`, `plot_paper_compare`, `plot_rtp`, `plot_R`, `plot_fcf`, `plot_ccs`, `plot_coronal_result`, `plot_contribution`, `coronal_fit_formula`, `coronal_autofit`, `print_coronal_fit_result`, `get_nxbp`, `calc_errorbars`, `plot_popx_paper` | `coronal/model.py` (physics) + `plotting/coronal.py` |
| Free utilities | 1909–2073 | `delta_kro`, `g_as`, `g_as_vector`, `tjpo_vector`, `reshape_4d2d`, `plot_rmatrix`, `flatdf`, `flatdf_1`, `figsize`, `set_tick_size` | `utils/`, `plotting/utils.py` |

### 3.2 `src/fulcher_analyser/` (partial refactor — Liptak)

| File | Purpose | Status |
|---|---|---|
| `__init__.py` | `sys.path` hack so submodules can `from molecular_data import …` instead of relative imports | bootstrap hack, not idiomatic |
| `__main__.py` | argparse CLI: `python -m fulcher_analyser -f N -s path` → calls `read_spectrum` + `plot_spectrum` | works in isolation |
| `spectrum.py` | `read_spectrum_headers`, `read_spectrum` (returns `xr.Dataset`), `plot_spectrum` (annotates Q-branches over raw spectrum) | works for `echelle_spectra/out/*.txt`; only used by `load_spectra.ipynb` |
| `molecular_data.py` | `fulcher_alpha_wavelengths` dict, H2 (4×11) and D2 (5×15) | **duplicates** the data in `data_molecular/fulcher-α_band_wavelength.txt` and `..._wavenumber_D2.txt` |

The refactor only handles **spectrum visualization** — i.e. step 0 of the
pipeline that is *not* in the canonical notebook. It does no fitting, no
Boltzmann analysis, no coronal model. It is essentially a dead-end branch.

---

## 4. Dependency / data-flow graph

### 4.1 Module import graph

```
examples/CoronalModel-D-H.ipynb
        │
        ▼
fulcheranalyzer.coronalmodel
        │   imports: numpy, pandas, scipy.{optimize, constants},
        │            matplotlib, mpl_toolkits.axes_grid1, sympy.physics.wigner
        ▼
fulcheranalyzer._constants            (only: package_directory)

examples/load_spectra.ipynb
        │
        ▼
src.fulcher_analyser.spectrum
        │   imports: numpy, matplotlib, xarray, .molecular_data
        ▼
src.fulcher_analyser.molecular_data   (numpy only)

src.fulcher_analyser.__main__ ──▶ src.fulcher_analyser.spectrum  (CLI only)
```

The two trees never cross. There is **no shared core**.

### 4.2 Workflow data-flow (D2 example)

```
data/150482_fr_7.csv ──┐
data/150482_fr_7_err.csv ┘
        │ read_intensities()
        ▼
DataFrame inte (14×4) , interr (14×4)
        │
        │ BoltzmannPlot(inte,'d')
        ▼
┌──────────────────────────────────────────────────────────────┐
│  MolecularConstants() ─ uses data_molecular/*.txt            │
│      ├─ wld, wlh (Q-branch wavelengths)                      │
│      ├─ EdD, ExD, EdH, ExH  (rotational energies)            │
│      ├─ AD, AH, ADsum, AHsum (Einstein A-coefficients)       │
│      ├─ frac (T-ratio d3↔X)                                  │
│      └─ corona_constants_{d,h} = [E_vib, Ee_vib, fcf]        │
│                                                              │
│  bplot_constants → v2jp1, vg, fc, AevJ                       │
│  calculate_boltzmann → nd, nd_bol, nd_rel, nd_err, mask      │
│                                                              │
│  autofit()                                                   │
│    ├─ fit_boltzmann   → popt=[α,β,T1,T2,c0..]                │
│    ├─ calc_nd_synth   → nd_bol_synth, nd_synth               │
│    ├─ calc_nd_const   → c_nd, nd_sc, nd_vibrofit             │
│    └─ calc_all_rot_temp → trotall (d/X × Trot1/Trot2)        │
└──────────────────────────────────────────────────────────────┘
        │
        │ CoronaModel(bp)
        ▼
┌──────────────────────────────────────────────────────────────┐
│  prep_corona_fit                                             │
│    ├─ set_pop_shape  → popshape{vdmax,jdmax,vxmax,jxmax}     │
│    └─ make_rmatrix(load=True)                                │
│         try np.load data_molecular/Rmatrix_{...}_{iso}.npy   │
│         else build via R_formula = fcf · ccs_formula · rtp   │
│                              rtp  = branching · delta_kro    │
│              branching ← sympy.physics.wigner.wigner_3j      │
│                                                              │
│  coronal_autofit() = scipy.curve_fit on coronal_fit_formula  │
│    fits Tvib only (α,β,Trot1,Trot2 frozen from bp.popt)      │
│       → fitres=[Tvib], err=[Tviberr]                         │
│    calc_errorbars → yerr_flat, yerr_rot                      │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
pics/figs7-8_{d,h}.svg     (BoltzmannPlot.plot_popd_paper)
pics/Corona_{d,h}.png      (CoronaModel.plot_coronal_result)
pics/figs9-10_{d,h}.svg    (CoronaModel.plot_popx_paper)
pics/figs11-12_{d,h}.svg   (CoronaModel.plot_paper_compare)
```

---

## 5. Comparison: `./fulcheranalyzer` vs `./src/fulcher_analyser`

| Concern | old `fulcheranalyzer` | refactored `src/fulcher_analyser` |
|---|---|---|
| Spectrum loading | ❌ (notebook uses pre-integrated CSVs) | ✅ `spectrum.read_spectrum` (xarray) |
| Q-branch wavelength reference | ✅ `data_molecular/fulcher-α_band_*.txt` (loaded at runtime) | ✅ `molecular_data.fulcher_alpha_wavelengths` (Python dict, duplicates the .txt) |
| Gaussian fitting / line extraction | ❌ never implemented | ❌ never implemented |
| Molecular constants (we, Be, αe, De) | ✅ hard-coded strings in `MolecularConstants.create_dataframes` | ❌ |
| Franck–Condon factors | ✅ from data_molecular/*.txt | ❌ |
| Einstein A coefficients | ✅ hard-coded literals in `acoeff()` | ❌ |
| Boltzmann analysis (d-state) | ✅ `BoltzmannPlot` | ❌ |
| Coronal model + R-matrix | ✅ `CoronaModel` | ❌ |
| Plotting (paper-style) | ✅ ~20 plot methods | ✅ one diagnostic spectrum plot |
| Tests | ❌ none | ❌ none |
| CLI | ❌ | ✅ minimal argparse |
| Typing / docstrings | partial Google-style | full Google-style with annotations |
| Code quality of present parts | dense, prose docstrings, lots of `# TODO` | clean, modern, but small surface |

**Verdict.** The two packages do not overlap functionally except for the
Fulcher-α reference wavelengths (where the new one is a Python re-encoding
of the old one's text file). The refactor never reached the parts of the
pipeline that actually produced the JQSRT 2021 figures.

**Newer/better where present:** `src/fulcher_analyser/spectrum.py` for raw
spectrum I/O (xarray, typed, clean) — but it is *upstream* of where the
notebook currently starts; it is not a replacement for any old module.

**Older but irreplaceable:** the entire physics — `MolecularConstants`,
`BoltzmannPlot`, `CoronaModel`. These reproduce published numbers and must
be preserved bit-for-bit until covered by regression tests.

---

## 6. Issues / smells / hazards

### 6.1 Packaging is broken

- `pyproject.toml` declares the project as **"cnlecture"** ("complex numbers
  lecture") — copy-paste from another project. Build backend `hatchling`,
  but no `[tool.hatch.build.targets.wheel]` config, no entry for either
  `fulcheranalyzer` or `src/fulcher_analyser`. **`pip install -e .` will
  not give you either package.**
- `requirements.txt` lists only `matplotlib==3.5.2`, `numpy==1.23.1`,
  `xarray==2022.6.0`. Actually required at runtime: `pandas`, `scipy`,
  `sympy` (for `wigner_3j`), `matplotlib`, `numpy`. `xarray` is needed only
  by the refactor.
- Package on disk is `fulcheranalyzer` (top-level, not under `src/`), and
  the refactor is `src/fulcher_analyser` (typo: "analyser" vs "analyzer").
  Both `__init__.py` use a `sys.path` hack instead of being importable as
  a real installed package.
- Notebooks both rely on `sys.path.insert(0, os.path.abspath("../"))` /
  `"../src"` — i.e. only work when launched from `examples/`.
- `.readthedocs.yaml` exists; `docs/` Sphinx build references both the old
  package and the refactor (`docs/src/fulcher_analyser.*`), suggesting an
  intent that never landed.

### 6.2 Hardcoded constants / file paths inside Python source

Candidates for externalisation **(do not move yet)**:

| Where | What | Notes |
|---|---|---|
| `coronalmodel.py:42–58` `MolecularConstants.create_dataframes` | H2/D2 spectroscopic constants (we, wexe, Be, αe, De) parsed from triple-quoted whitespace-formatted strings, with provenance `"Ishihara's thesis / NIST WebBook"` in the docstring. **3 states × 5 constants × 2 isotopes = 30 numbers.** | `molecular_constants.csv` or `.yaml` |
| `coronalmodel.py:181–278` `acoeff` | Einstein A-coefficients `AD` (4×8) and `AH` (4×8), inline DataFrame literals, no source citation. | `einstein_A_{h2,d2}.csv` |
| `coronalmodel.py:563–576` `BoltzmannPlot.bplot_constants` | Franck–Condon factors hard-coded: D2 `[2.3387e7, 1.8841e7, 1.4795e7, 1.1276e7]` (looks like a re-use of `AD` diagonal, not FCF!) and H2 `[0.93016, 0.79701, 0.67139]` (annotated "Relative?"). **Suspicious — should be cross-checked against `data_molecular/franck_condon_factor*.txt`.** | flag for review, then `fcf_*.csv` |
| `coronalmodel.py:1165` `CoronaModel.branching` | rotational branching weights `Qr = [0.76, 0.122, 0.1, 0.014]` (Q1–Q4) | `branching_weights.yaml` |
| `coronalmodel.py:5` `ABSOLUTESIGMA = False` | global, used in two `curve_fit` calls | already a const; document semantics |
| `coronalmodel.py:6–7` | `MOLECULAR_DATA_FOLDER`, `DATA_FOLDER` derived from `_constants.package_directory` via `..`. Breaks if the package is ever installed site-packages-style. | use `importlib.resources` |
| `coronalmodel.py:1376` `make_rmatrix` | hardcoded cache filename pattern; writes back into `MOLECULAR_DATA_FOLDER` (i.e. into the package). | move cache to user cache dir |
| `coronalmodel.py:1242` `calc_rtp` | hardcoded `x, d = [30, 15]` — silently caps Jx, Jd. | document |

### 6.3 Duplicated logic

- `BoltzmannPlot.load_wavelength_data` (lines 493–507) is byte-for-byte
  identical to `MolecularConstants.load_wavelength_data` (lines 91–106).
  Both are kept; class TODO already notes this.
- D2 FCF "approximation" in `bplot_constants` re-uses A-coefficient
  magnitudes; the real FCFs are loaded elsewhere in `corona_constants_d`.
  This duplication is **load-bearing** — changing it would change the
  Boltzmann normalisation.
- `calculate_e_cross` (line 1006) and `ccs_formula` (line 1219) compute
  the same Boltzmann-factor matrix two different ways; the docstring of
  `calculate_e_cross` admits "Duplicate of `self.ccs_formula()`".
- `f_vibro` (line 1042, Ishihara's vibro-only fit) and
  `coronal_fit_formula` (line 1746, the rotationally-resolved fit) are
  two coexisting "the model" implementations. The notebook only calls the
  second one (`coronal_autofit`). `fit_vibro_ishi` is **never called** by
  the canonical notebook — it is reference code for comparison.

### 6.4 Unused / partially abandoned

- `src/fulcher_analyser/__main__.py` CLI: never called from any notebook.
- `data/143306.nc` (NetCDF): not opened by any code in the repo.
- `echelle_spectra/data/`: empty.
- `echelle_spectra` is in `.gitignore` (line 11), yet `echelle_spectra/out/151953_echelle_spec.txt` is committed — fragile.
- `*.csv` is in `.gitignore` yet all four intensity CSVs in `data/` are
  committed — same fragility. Any future `git clean -fdx` from a fresh
  clone would still keep them (because they are tracked), but new shots
  would silently be ignored.
- `docs/source/examples/CoronaModelExample.ipynb` is a near-duplicate of
  the canonical notebook with a different `sys.path` prefix.
- Inside `CoronaModel`: `fit_vibro_ishi`, `plot_fit_ishi`,
  `calculate_e_cross`, `plot_contribution`, `plot_rtp`, `plot_R`,
  `plot_fcf`, `plot_ccs`, `plot_xd`, `check_rmatrix_indexing`,
  `calc_branching_matrix`, `calc_delta` — none are exercised by the
  canonical notebook. They are diagnostic / alternative-formulation
  helpers; keep them, but they are *not* on the critical path.
- Inside `BoltzmannPlot`: `plot_boltzmannn`, `plot_fit`, `plot_fit_nice`,
  `plot_nd`, `about_var` — diagnostic only.

### 6.5 Python-3.10+ assumptions

- `set_pop_shape` uses `self.popshape = x | y` (dict union, 3.9+).
- f-strings everywhere (3.6+). No Python-2 fallbacks; safe.
- Many `SyntaxWarning: invalid escape sequence` (≈14 of them) from raw
  LaTeX in plot strings — harmless under 3.12 but already noisy. These
  are the only "true bugs" the interpreter complains about today.

### 6.6 Broken paths in `docs/source/conf.py`

```python
sys.path.insert(0, os.path.abspath("../../"))
print("path ../../src/fulcheranalyzer: ", os.path.abspath("../../src/fulcheranalyzer"))
```

The path `../../src/fulcheranalyzer` does not exist — the actual folder is
`src/fulcher_analyser`. Sphinx autodoc for the refactor cannot resolve.

---

## 7. What should survive / deprecate / become tests

### 7.1 SURVIVE — load-bearing, physics-correct, hard to re-derive

The following must reach the future stable package **unchanged in numerics**:

- `MolecularConstants` (entire class) — molecular spectroscopic data layer.
- `read_intensities` / `write_intensities` — CSV I/O contract that frames
  what shape downstream code expects (rows = J, cols = v).
- `BoltzmannPlot.{prep_mol, bplot_constants, calculate_boltzmann,
  fit_boltzmann, calc_nd_synth, calc_nd_const, calc_all_rot_temp}` and
  the helper `two_t_all_v` + `expsum`.
- `CoronaModel.{prep_constants, set_pop_shape, prep_corona_fit,
  make_rmatrix, make_rmatrix_2d, branching, ccs_formula, rtp,
  R_formula, calc_X_bp, calc_nx, calc_nd, calc_ndx,
  coronal_fit_formula, coronal_autofit, calc_errorbars}`.
- Free utilities used inside fits: `delta_kro`, `g_as`, `g_as_vector`,
  `tjpo_vector`, `reshape_4d2d`, `flatdf`.
- All `data_molecular/*.txt` files (verbatim).
- Both `Rmatrix_*.npy` cache files (or be willing to regenerate and accept
  ~minute startup).

### 7.2 DEPRECATE (cautiously, after the refactor is green)

- The `pyproject.toml` "cnlecture" content.
- `src/fulcher_analyser/molecular_data.py` once an authoritative single
  source-of-truth for Fulcher-α wavelengths is chosen.
- `docs/source/examples/CoronaModelExample.ipynb` (duplicate notebook).
- The pile of diagnostic `plot_*` methods inside the classes can move out
  to a `plotting/` submodule (or be dropped) only **after** the figures
  they reproduce are checkpointed.

### 7.3 BECOME TESTS — regression assertions before any refactor

Create `tests/test_paper_reproduction.py` that pins, with reasonable
tolerance:

```
D2 (shot 150482, frame 7):
    bp.trot1   ≈ 279.94  K
    bp.trot2   ≈ 1782.83 K
    bp.alpha   ≈ 0.7609
    bp.beta    ≈ 0.4159
    como.tvib  ≈ 7743    K
    como.tviberr ≈ 557   K

H2 (shot 152478, frame 10):
    bph.trot1  ≈ 374.72  K
    bph.trot2  ≈ 2079.14 K
    bph.alpha  ≈ 0.7335
    bph.beta   ≈ 0.4473
    comoh.tvib ≈ 6801    K
    comoh.tviberr ≈ 753  K
```

Also pin: shapes of `bp.nd`, `bp.nd_synth`, `como.Rm`, `como.nx`, `como.nd`;
contents of `bp.popt` and `como.fitres` (np.allclose with `rtol=1e-3` is
likely safe; `1e-6` is probably too tight given iterative `curve_fit`).

Secondary tests worth adding:

- `MolecularConstants` deterministic snapshot of `EdH`, `EdD`, `frac`,
  `AD`, `AH` (golden DataFrames).
- `branching(JX, Jd)` for a small fixed grid (depends on sympy).
- R-matrix regeneration vs. cached `.npy` (numerical equality).
- `read_intensities` round-trip with `write_intensities`.

---

## 8. High-risk areas — refactoring here could silently break physics

| Risk | Why |
|---|---|
| Index conventions of `Rm` `[vx, jx, vd, jd]` and the order of `reshape_4d2d` + `.flatten(order='f')` in `calc_nd` | A wrong reshape silently corrupts the d-state reconstruction with no exception. The 4D→2D→matmul→reshape pipeline is the most fragile single thing in the codebase. |
| `frac` normalisation (`calculate_tfrac(..., norm=True)`) called in two different places with two different `vmax` | The note in `two_t_all_v` says: "norm=True is essential! Otherwise produces T2*2." Forgetting this doubles T_rot2. |
| `J0 = 1` for d-state, `J0 = 0` for X-state in `calculate_E_rot` | A single off-by-one would shift Boltzmann fits. |
| The order in `coronal_fit_formula` of (a) overwriting `alpha/beta/Trot1/Trot2` from `bp.popt`, (b) calling `calc_ndx`, (c) flattening with `self.bp.mask` | Changing this order changes which experimental points participate in the fit. |
| `make_rmatrix(load=True)` silently uses a stale cache if the formula changes but the cache filename does not | The comment says: "CAUTION: must recalculate this matrix if the model changed." Future refactors must invalidate `Rmatrix_*.npy` whenever `branching`, `ccs_formula`, or `rtp` change. |
| The hardcoded D2 FCF `[2.3387e7, 1.8841e7, 1.4795e7, 1.1276e7]` in `BoltzmannPlot.bplot_constants` — these look like A-coefficient magnitudes, not FCFs | This is either a clever physical trick or a long-standing bug. Either way the published numbers depend on these exact values. Do not "fix" before reading the paper sections that derive `n_d/(2J+1)/g_as` from intensity. |
| `Qr = [0.76, 0.122, 0.1, 0.014]` in `branching` | Q-branch rotational weights; unsourced. Used inside `R_formula`. Replacing with re-derived numbers will shift `T_vib` by ~10%. |
| `MOLECULAR_DATA_FOLDER` resolved via `__file__/..` | Will break when the package is ever installed under `site-packages/`. |

---

## 9. Proposed Phase-2 refactor boundaries (NOT to be executed in Phase 1)

A minimal, low-risk extraction order:

1. **Stabilize packaging** — fix `pyproject.toml` (project name, `[tool.hatch.build]` pointing at `fulcheranalyzer/`), add a real `requirements.txt` (pandas, scipy, sympy), make `pip install -e .` succeed. **No code moves yet.**
2. **Add the regression test** from §7.3 against the unmodified
   `fulcheranalyzer.coronalmodel`. Make this run in CI.
3. **Extract the data layer** — move `MolecularConstants` to
   `fulcheranalyzer/molecular_constants.py`, keep a re-export from
   `coronalmodel` for backward compatibility. Re-run regression.
4. **Extract free utilities** (`delta_kro`, `g_as`, `flatdf`, …) to
   `fulcheranalyzer/_utils.py`. Re-run regression.
5. **Externalize hardcoded data tables** from §6.2 to `data_molecular/`
   (CSV/YAML), one table at a time, with the regression test guarding
   each move.
6. **Split the plotting** out of `BoltzmannPlot` / `CoronaModel` into
   `fulcheranalyzer/plotting/`. Class methods become thin wrappers.
7. **Only then** consider whether `src/fulcher_analyser/spectrum.py`
   becomes the entry point that produces the CSVs that
   `read_intensities` consumes — closing the loop from raw echelle
   spectra to T_vib.

At each step the success criterion is: `python -m pytest tests/test_paper_reproduction.py` continues to pass.

---

## 10. Quick "what touches what" reference card

```
Notebook step              Class.method                            Touches
─────────────────────────  ──────────────────────────────────────  ────────────────────────────────────
read_intensities           module fn                               data/*.csv
BoltzmannPlot(inte,iso)    __init__ → MolecularConstants()         data_molecular/fulcher-α_*.txt
                                                                   data_molecular/franck_condon_*
                                                                   data_molecular/{,excitation_}vibrational_energy*
bp.autofit                 fit_boltzmann                           scipy.optimize.curve_fit
                           calc_nd_synth / calc_nd_const           —
                           calc_all_rot_temp                       —
bp.plot_popd_paper         (plotting)                              writes pics/figs7-8_*.svg
CoronaModel(bp)            prep_corona_fit → make_rmatrix          reads/writes data_molecular/Rmatrix_*.npy
                                            ← branching            sympy.physics.wigner.wigner_3j
como.coronal_autofit       curve_fit on coronal_fit_formula        —
como.plot_coronal_result   plot_xd_flat + plot_paper_compare       writes pics/Corona_*.png
como.plot_popx_paper                                               writes pics/figs9-10_*.svg
como.plot_paper_compare                                            writes pics/figs11-12_*.svg
```

---

## 11. TL;DR for the next session

- The **only** working pipeline is `examples/CoronalModel-D-H.ipynb` →
  `fulcheranalyzer/coronalmodel.py`. It runs today on Python 3.12 with
  recent numpy/pandas/scipy/sympy/matplotlib (only `SyntaxWarning`s).
- The `src/fulcher_analyser/` tree is a half-finished, unrelated detour
  that only handles raw-spectrum visualization. It is safe to ignore for
  the physics refactor.
- Packaging (`pyproject.toml`) is wrong — pinned to the wrong project name.
  Fix this *first* and *only this* in any cleanup attempt.
- Reproduce the six reference numbers from §2 before touching anything
  else; treat them as the immovable physical truth of the codebase.
- The R-matrix indexing, the `frac` normalisation flag, and the hardcoded
  FCFs/branching-weights are the three places where a well-meaning
  refactor will silently break published results.
