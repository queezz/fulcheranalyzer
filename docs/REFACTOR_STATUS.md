# Refactor Status Checkpoint

## Scope

This document records the current state of the repository after the
mechanical cleanup and refactor that followed the initial archaeology
report (`docs/ARCHAEOLOGY.md`). It is a checkpoint, not a redesign:
no physics, fitting logic, constants, or expected regression values
were intentionally changed.

## Current package layout

The package now lives under a `src/` layout:

- `src/fulcher_analyzer/`
    - `molecular_constants.py`
    - `boltzmann.py`
    - `coronal_model.py`
    - `intensity_io.py`
    - `plotting.py`
    - `_utils.py`
    - packaged `data_molecular/` (spectroscopic constants, Franck–Condon
      factors, vibrational energies, R-matrices, Fulcher band tables)
    - packaged `example_data/intensities/` (paper-regression intensity
      arrays for D2 and H2)

## Completed cleanup/refactor steps

- migrated from old root package to `src/` layout
- removed obsolete root `fulcheranalyzer/` package
- removed legacy `coronalmodel.py` facade
- updated canonical notebook imports to the new public API
- moved bundled paper-regression intensities into package data
- removed stale `requirements.txt`
- removed orphan root `data/` artifacts
- moved canonical notebook figures to `examples/figures/`
- moved CoronaModel plotting helpers into `plotting.py`
- moved `set_tick_size` and `figsize` plotting helpers into `plotting.py`
- adopted MkDocs Material styling and a GitHub Pages deploy workflow

## Current canonical workflow

The canonical example remains:

- `examples/CoronalModel-D-H.ipynb`

It now uses canonical imports such as:

```python
from fulcher_analyzer import BoltzmannPlot, CoronaModel, read_intensities
from fulcher_analyzer.plotting import set_tick_size, figsize
```

## Regression guardrails

Numerical regression tests in `tests/test_paper_reproduction.py`
continue to pin the published D2/H2 workflow against fixed inputs:

- D2: shot 150482, frame 7
- H2: shot 152478, frame 10

Exact pinned values for `Trot1`, `Trot2`, `alpha`, `beta`, `Tvib`,
and `Tvib_err` are kept in the test module and the original numbers
are documented in `docs/ARCHAEOLOGY.md`. They are intentionally not
duplicated here.

## Current model-code status

- Plotting has been extracted from `CoronaModel` into `plotting.py`.
- `coronal_model.py` now mostly contains model and fitting logic
  (X-state population construction, R-matrix projection, masked
  d-state fit, Tvib fit on top of the Boltzmann result).
- The remaining logic is still side-effect based (results are stashed
  on the model instance rather than returned) and will need
  documentation and clarification before any deeper refactor.
- No physics behavior was intentionally changed during this checkpoint.

## Known high-risk areas still unchanged

These are deliberately left untouched and should not be modified
without dedicated regression review:

- R-matrix 4D/2D reshape and `flatten(order="f")`
- `frac(..., norm=True)`
- D2 normalization constants in Boltzmann fitting
- `Qr = [0.76, 0.122, 0.1, 0.014]`
- R-matrix cache behavior

## Current checks

At this checkpoint:

- `pytest`: 30 passed
- `mkdocs build --strict`: success (warning shown is a vendor notice
  from Material for MkDocs about a future 2.0 release; the build
  itself succeeds with no documentation warnings)
- current git short hash: `51eb51b`

## Suggested next step

The next safe step is a documentation and readability pass over
`coronal_model.py`, with no behavioral changes. In particular it would
help to clearly explain:

- how the X-state population is constructed
- how the R-matrix projection maps X-state populations onto d-state
  rovibrational populations
- how the normalized, masked d-state fit is set up
- why the second-stage fit only varies `Tvib`, while inheriting
  `alpha`, `beta`, `Trot1`, and `Trot2` from the Boltzmann fit
