# Workflow

This page describes the analysis pipeline at a high level.

## Pipeline overview

```
Intensity CSV files
        │
        ▼
  read_intensities()          ← workflow entry point
        │
        ▼
   BoltzmannPlot              ← d-state population fitting
   · autofit()
        │
        ▼
    CoronaModel               ← ground-state reconstruction
    · coronal_autofit()
        │
        ▼
  Ground-state populations / results
```

---

## Step 1 — Load intensities

Intensity data lives in CSV files that encode measured line intensities and
their uncertainties for each Fulcher-α transition.

```python
from fulcher_analyzer import read_intensities

intensities, errors = read_intensities(shot=150482, frame=7)
```

The returned DataFrames are indexed by transition label and contain one column
per measured spectral band.

---

## Step 2 — Boltzmann fit (d-state)

`BoltzmannPlot` constructs a Boltzmann plot from the measured intensities and
fits rotational and vibrational temperatures of the electronically-excited
d-state population.

```python
from fulcher_analyzer import BoltzmannPlot

bp = BoltzmannPlot(intensities, isotop="d")
bp.autofit()
```

Internally this uses `MolecularConstants` to look up Einstein A coefficients,
transition energies, and degeneracy factors for either H₂ (`"h"`) or D₂
(`"d"`).

---

## Step 3 — Coronal model (ground-state reconstruction)

`CoronaModel` takes the fitted `BoltzmannPlot` and applies the coronal model
to infer the ground-state vibrational population distribution.

```python
from fulcher_analyzer import CoronaModel

model = CoronaModel(bp)
model.coronal_autofit()
```

The coronal model relates the measured d-state populations back to the
ground-state populations through excitation cross-sections and the assumption
of coronal equilibrium (low-density limit).

---

## Regression tests

The test suite pins the full pipeline against the published D₂ and H₂
reference outputs from Kuzmin *et al.* (JQSRT 2021):

| Test file | Coverage |
|---|---|
| `tests/test_smoke.py` | Imports, data-path resolution, DataFrame shapes |
| `tests/test_paper_reproduction.py` | Numerical D₂/H₂ Boltzmann + coronal-model results |

Run the full suite with:

```bash
pytest
```

The regression tests exercise the complete pipeline end-to-end and will fail
if any physics constant, formula, or data file is accidentally modified.
