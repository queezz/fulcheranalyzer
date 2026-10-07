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

The population and the Boltzmann ordinate use one consistent route:
`n_d ∝ I·λ/A`, followed by division by `(2N'+1) g_as`. Franck–Condon
factors belong to the downstream coronal model and are not a second
normalization of the measured d-state population.

The packaged H2 and D2 Einstein-A matrices reproduce the comparison columns
of Tables 5 and 7 in Lavrov, Pozdeev & Yakovleva (2015), transposed to
`(v', v'')`: [arXiv:1512.06306](https://doi.org/10.48550/arXiv.1512.06306).
Those columns quote the non-empirical adiabatic calculation from that paper's
reference 21; they are not the authors' recommended semiempirical H2 values.
Both complete H2 and D2 columns are available. Pass
`a_table="semiempirical"` to `BoltzmannPlot` to select the recommended values;
the default `"comparison"` preserves historical results. Semiempirical
uncertainties are exposed as `bp.mol.AH_err` and `bp.mol.AD_err`. Zeros in
that table mean below the semiempirical method's reach, not physical zeros.

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
| `bp.fit_at_bound` | Boolean vector identifying parameters fitted at a limit |

`CoronaModel` inherits these via its `bp` reference and holds them fixed
during the `Tvib` fit. See [Coronal model](coronal_model.md).

## Notes

- The two-temperature parametrisation is historical and follows the
  published workflow / Ishihara-style analysis. It is preserved by
  regression tests; do not re-parameterise without updating the
  regression values.
- The default upper bound for the hot rotational component is 6000 K.
  Treat a true value in `bp.fit_at_bound` as a limit, not a measurement.
- `BoltzmannPlot` accepts either the `(intensity_df, error_df)` tuple
  returned by `read_intensities` or just the intensity DataFrame.
