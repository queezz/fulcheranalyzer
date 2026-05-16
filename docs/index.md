# FulcherAnalyzer

**FulcherAnalyzer** analyzes Fulcher-α molecular hydrogen and deuterium emission
spectra. It extracts line intensities, fits Boltzmann distributions over the
d-state rotational/vibrational populations, and reconstructs ground-state
population distributions through a coronal model.

## Background

The code reproduces the published coronal-model workflow described in:

> Kuzmin *et al.*, *Journal of Quantitative Spectroscopy and Radiative Transfer*
> (JQSRT), 2021.

The package has been **mechanically refactored** from a monolithic script into a
structured `src`-layout Python package. All numerical behavior is preserved:
regression tests assert the published D₂ and H₂ reference outputs to floating-point
precision.

## Quick start

```python
from fulcher_analyzer import BoltzmannPlot, CoronaModel, read_intensities

intensities, errors = read_intensities(150482, 7)
bp = BoltzmannPlot(intensities, "d")
bp.autofit()

model = CoronaModel(bp)
model.coronal_autofit()
```

See the [Usage](usage.md) page for the complete import style, the
[Workflow](workflow.md) page for a high-level description of the analysis
pipeline, and the [Physics overview](physics/index.md) for a description
of the d-state Boltzmann fit and the coronal-model projection used to
recover `Tvib`.

## Key modules

| Name | Purpose |
|---|---|
| `MolecularConstants` | Spectroscopic constants for H₂/D₂ |
| `BoltzmannPlot` | d-state Boltzmann fitting |
| `CoronaModel` | Ground-state coronal-model reconstruction |
| `read_intensities` | Load intensity CSV files |
| `write_intensities` | Save intensity CSV files |

See the [API Overview](api.md) for short descriptions of each public symbol.
