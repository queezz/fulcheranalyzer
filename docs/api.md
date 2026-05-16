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

Key attributes after fitting: rotational temperature `T_rot`, vibrational
temperature `T_vib`, and the population distribution over (v′, J′) levels.

---

## `CoronaModel`

**Module:** `fulcher_analyzer.coronal_model`

Applies the coronal model to reconstruct ground-state vibrational populations
from the d-state Boltzmann fit.

```python
model = CoronaModel(bp)
model.coronal_autofit()
```

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

## `set_tick_size`

**Module:** `fulcher_analyzer.plotting`

Utility to set matplotlib axis tick sizes (length and width for major and minor ticks).

```python
from fulcher_analyzer.plotting import set_tick_size

set_tick_size(ax, width_major, length_major, width_minor, length_minor)
```
