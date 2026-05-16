# API Overview

All public symbols are importable directly from `fulcher_analyzer`:

```python
from fulcher_analyzer import (
    MolecularConstants,
    BoltzmannPlot,
    CoronaModel,
    read_intensities,
    write_intensities,
)
```

Plotting helpers live in the `fulcher_analyzer.plotting` submodule:

```python
from fulcher_analyzer.plotting import (
    plot_rmatrix,
    figsize,
    set_tick_size,
)
```

For a physics-oriented description of how `BoltzmannPlot` and
`CoronaModel` fit together, see the [Physics overview](physics/index.md).

---

## `MolecularConstants`

**Module:** `fulcher_analyzer.molecular_constants`

Holds spectroscopic constants for H₂ and D₂ Fulcher-α transitions: Einstein A
coefficients, transition wavelengths, upper-state energies, and degeneracy
factors.

```python
mc = MolecularConstants(isotop)   # isotop: "h" or "d"
```

---

## `BoltzmannPlot`

**Module:** `fulcher_analyzer.boltzmann`

Constructs a Boltzmann plot from measured line intensities and fits rotational
and vibrational temperatures of the d-state population.

```python
bp = BoltzmannPlot(intensities, isotop="d")
bp.autofit()
```

Key attributes after `bp.autofit()`:

- `bp.alpha`, `bp.beta` — mixing weights of the two-temperature
  rotational model.
- `bp.trot1`, `bp.trot2` — fitted rotational temperatures (K).
- `bp.popt` — full `curve_fit` parameter vector.
- `bp.nd`, `bp.nd_err` — d-state population and its error on the
  `(vd, Jd)` grid.

These quantities are consumed downstream by `CoronaModel`.

---

## `CoronaModel`

**Module:** `fulcher_analyzer.coronal_model`

Applies the coronal model to reconstruct ground (X-) state rovibrational
populations from the d-state Boltzmann fit. Only the X-state vibrational
temperature `Tvib` is fitted; `alpha`, `beta`, `Trot1`, `Trot2` are
inherited from the supplied `BoltzmannPlot` and held fixed.

```python
model = CoronaModel(bp)
model.coronal_autofit()

print(model.tvib, model.tviberr)
```

See the [Coronal model](physics/coronal_model.md) page for the workflow
and load-bearing implementation details.

---

## `read_intensities`

**Module:** `fulcher_analyzer.intensity_io`

Loads intensity and uncertainty DataFrames from the standard CSV layout.

```python
intensities, errors = read_intensities(shot, frame)
```

Parameters:

- `shot` — discharge shot number
- `frame` — frame index within the shot

Returns a tuple `(intensity_df, error_df)` of `pandas.DataFrame`.

---

## `write_intensities`

**Module:** `fulcher_analyzer.intensity_io`

Saves intensity DataFrames back to the standard CSV layout.

```python
write_intensities(intensities, errors, shot, frame)
```

---

## Plotting helpers

**Module:** `fulcher_analyzer.plotting`

### `plot_rmatrix`

Render a 2-D view of the (reshaped) R-matrix for inspection.

```python
from fulcher_analyzer.plotting import plot_rmatrix

plot_rmatrix(model.Rm2d, model.rshape, text="R-matrix")
```

### `figsize`

Convenience helper returning a `(width, height)` tuple at a fixed aspect
ratio, used by the canonical notebook figures.

```python
from fulcher_analyzer.plotting import figsize

fig = plt.figure(figsize=figsize(width=8, ratio=5/6))
```

### `set_tick_size`

Utility to set matplotlib axis tick sizes (length and width for major and
minor ticks).

```python
from fulcher_analyzer.plotting import set_tick_size

set_tick_size(ax, width_major, length_major, width_minor, length_minor)
```

Additional `CoronaModel`-specific plotting helpers (`plot_coronal_result`,
`plot_xd`, `plot_paper_compare`, `plot_contribution`, …) also live in
this module and are exposed as thin method wrappers on `CoronaModel`.
