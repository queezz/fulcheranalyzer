# Coronal model (X-state reconstruction)

Stage 2 of the analysis. Given the fitted d-state population from
[`BoltzmannPlot`](boltzmann.md), `CoronaModel` constructs a trial X-state
rovibrational population for a given `Tvib`, projects it onto the d-state
through an R-matrix, and fits `Tvib` against the measured d-state
population.

This page summarises the workflow exposed by `coronal_model.py`. The
detailed per-method docstrings live in the source.

## Inputs and inheritance

`CoronaModel` is constructed from a **fully-fitted** `BoltzmannPlot`:

```python
from fulcher_analyzer import BoltzmannPlot, CoronaModel, read_intensities

intensities, errors = read_intensities(150482, 7)
bp = BoltzmannPlot(intensities, "d")
bp.autofit()

cm = CoronaModel(bp)
cm.coronal_autofit()

print(cm.tvib, cm.tviberr)
```

`CoronaModel` inherits the d-state Boltzmann result through its `bp`
reference. The rotational parameters `alpha`, `beta`, `Trot1`, and
`Trot2` are *not* free in the second-stage fit — only `Tvib` is fitted.

## The R-matrix

The mapping from X-state to d-state populations is encoded in a
4-dimensional R-matrix indexed by `(vX, JX, vd, Jd)`:

```text
R[vX, JX, vd, Jd] = FCF[vX, vd] * CCS[vX, vd] * RTP[JX, Jd]
```

where

- **FCF** — Franck–Condon factor for the vibrational overlap between
  X-state level `vX` and d-state level `vd`.
- **CCS** — electron-impact cross-section factor used in the historical
  coronal model. Implemented as a Boltzmann exponential in the
  threshold energy difference between `vd` and `vX`.
- **RTP** — rotational transition probability, combining the Q-branch
  rotational branching ratio (`Qr = [0.76, 0.122, 0.10, 0.014]`) with
  the ortho/para nuclear-spin selection rule.

The trial X-state populations are arranged on the same `(vX, JX)` grid,
and the model d-state population is obtained as a matrix product against
a 2-D flattening of `R`:

```text
n_d(model) = R @ n_X
```

In code this is implemented as `Rm2d @ nx.flatten(order='f')`, with the
2-D R-matrix produced by `reshape_4d2d`. The `order="f"` (Fortran /
column-major) flattening is **load-bearing** — it must agree with the
column ordering used to construct `Rm2d`.

## Method walk-through

The key methods on `CoronaModel`, in approximately the order they are
used by `coronal_autofit`:

- `prep_constants()` — loads isotopologue-specific molecular constants
  (Einstein A's, vibrational energies, Franck–Condon factors) and
  pre-computes the rotational-transition-probability matrix `rtp` via
  `calc_rtp`.
- `prep_corona_fit(load=True)` — sets the population-shape arrays and
  builds (or loads from cache) the 4-D R-matrix.
- `calc_nx(Tvib, alpha, beta, Trot1, Trot2)` — constructs the weighted
  X-state population on the `(vX, JX)` grid for a trial `Tvib`, mixing
  the two rotational components via `alpha` and `beta`.
- `calc_nd()` — applies the flattened R-matrix to obtain the model
  d-state population.
- `coronal_fit_formula(_, Tvib, alpha, beta, Trot1, Trot2)` — returns
  the normalised, masked d-state population vector that
  `scipy.optimize.curve_fit` minimises against `bp.nd`. Normalisation
  uses `frac(..., norm=True)`; the mask removes d-state cells that are
  not measured.
- `coronal_autofit()` — runs the second-stage `curve_fit`. Only `Tvib`
  is varied; `alpha`, `beta`, `Trot1`, `Trot2` are taken from `bp`.
  Sets `self.tvib` and `self.tviberr`.

A separate legacy path (`f_vibro` / `fit_vibro_ishi`) implements
Ishihara's rotationally-integrated formula (4.5). It is preserved as a
diagnostic but is not part of the main `coronal_autofit` workflow.

## Outputs

After `cm.coronal_autofit()`:

| Attribute | Meaning |
|-----------|---------|
| `cm.tvib`    | Fitted X-state vibrational temperature (K) |
| `cm.tviberr` | 1-σ uncertainty on `Tvib` (K) |
| `cm.nx`      | Trial X-state population on the `(vX, JX)` grid |
| `cm.nd`      | Model d-state population |
| `cm.Rm`, `cm.Rm2d` | 4-D and 2-D R-matrices |

The corresponding plotting helpers live in
`fulcher_analyzer.plotting` (`plot_rmatrix`, `plot_coronal_result`,
`plot_xd`, `plot_paper_compare`, …) and are also exposed as thin method
wrappers on `CoronaModel`.

## Load-bearing implementation details

!!! warning "Do not casually change these"
    The following are deliberately preserved by regression tests and
    should not be modified without dedicated regression review:

    - R-matrix indexing `R[vX, JX, vd, Jd]` and the `reshape_4d2d`
      column ordering.
    - `flatten(order="f")` (Fortran / column-major) on `nx` when
      multiplying by `Rm2d`.
    - `frac(..., norm=True)` normalisation inside
      `coronal_fit_formula`.
    - D₂ normalisation constants used in the Boltzmann fit.
    - Q-branch rotational branching weights
      `Qr = [0.76, 0.122, 0.10, 0.014]`.
    - The R-matrix on-disk cache layout.

    These are *load-bearing historical choices*: changing any one of
    them will silently move the fitted `Tvib` away from the published
    regression values.
