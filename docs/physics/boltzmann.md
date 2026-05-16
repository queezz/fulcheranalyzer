# Boltzmann population fit (d-state)

The first stage of the analysis converts measured Q-branch line intensities
into relative rovibrational populations of the electronically-excited d-state
and fits them with a two-temperature rotational model.

## Inputs

`read_intensities(shot, frame)` loads the integrated Q-branch line
intensities and their per-line uncertainties from the bundled example
data (or from a user-supplied `data_folder=...`):

```python
from fulcher_analyzer import read_intensities

intensities, errors = read_intensities(150482, 7)
```

The returned objects are `pandas.DataFrame`s indexed by transition label,
covering the Q-branch lines used in the published workflow.

## Fitting

`BoltzmannPlot` performs the d-state fit:

```python
from fulcher_analyzer import BoltzmannPlot, read_intensities

intensities, errors = read_intensities(150482, 7)

bp = BoltzmannPlot(intensities, "d")   # "d" for D₂, "h" for H₂
bp.autofit()

print(bp.trot1, bp.trot2, bp.popt)
```

Internally, `BoltzmannPlot` uses `MolecularConstants` to look up the
Einstein A coefficients, transition energies, and degeneracy factors for
the chosen isotopologue, converts line intensities into a Boltzmann plot
of d-state level populations, and fits a two-temperature rotational
distribution.

## Fitted quantities used downstream

After `bp.autofit()` the following attributes are populated and are
consumed by the second-stage coronal-model fit:

| Attribute | Meaning |
|-----------|---------|
| `bp.alpha` | Mixing weight of the two rotational components |
| `bp.beta`  | Second mixing weight (see `popt`) |
| `bp.trot1` | First rotational temperature `Trot1` (K) |
| `bp.trot2` | Second rotational temperature `Trot2` (K) |
| `bp.popt`  | Full `curve_fit` parameter vector |

`CoronaModel` inherits these via its `bp` reference and holds them fixed
during the `Tvib` fit. See [Coronal model](coronal_model.md).

## Notes

- The two-temperature parametrisation is historical and follows the
  published workflow / Ishihara-style analysis. It is preserved by
  regression tests; do not re-parameterise without updating the
  regression values.
- `BoltzmannPlot` accepts either the `(intensity_df, error_df)` tuple
  returned by `read_intensities` or just the intensity DataFrame.
