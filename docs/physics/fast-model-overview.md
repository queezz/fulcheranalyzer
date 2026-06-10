# Fast model overview

This page is a quick memory aid for the current Fulcher analyzer model.
It is intentionally shorter than the detailed Boltzmann and coronal-model
pages.

## Pipeline

```python
from fulcher_analyzer import BoltzmannPlot, CoronaModel, read_intensities

intensities, errors = read_intensities(shot, frame)

bp = BoltzmannPlot((intensities, errors), isotop="h")  # or "d"
bp.autofit()

cm = CoronaModel(bp)
cm.coronal_autofit()
```

The first stage fits the d-state Boltzmann population.  The second stage
uses that fitted shape and fits only the X-state vibrational temperature.

## What alpha and beta mean

`alpha` and `beta` are mixing weights for the hot rotational component in
the two-temperature rotational distribution:

```text
n(E) = (1 - a) exp(-E / kTrot1) + a exp(-E / kTrot2)
```

where `Trot1` is the colder component and `Trot2` is the hotter component.
The coefficient `a` is therefore the hot-component weight.

In the d-state Boltzmann fit:

| Parameter | Meaning |
|-----------|---------|
| `alpha` | hot `Trot2` weight for `v = 0` |
| `beta` | hot `Trot2` weight for `v > 0` |
| `Trot1` | cold rotational temperature |
| `Trot2` | hot rotational temperature |

The fitted vector is:

```text
bp.popt = [alpha, beta, Trot1, Trot2, c0, c1, ...]
```

The `c*` entries are per-vibrational-band scale constants for the
Boltzmann fit, not extra temperatures.

## What happens after the Boltzmann fit

`bp.autofit()` runs:

1. `fit_boltzmann()` — fit `alpha`, `beta`, `Trot1`, `Trot2`, and the
   per-band constants.
2. `calc_nd_synth()` — create a synthetic d-state ro-vibrational population,
   including values for missing lines.
3. `calc_nd_const()` — scale the synthetic d-state population to measured
   `nd` and compute `nd_vibrofit`.
4. `calc_all_rot_temp()` — prepare derived d-state and X-state rotational
   temperature tables.

`nd_vibrofit` exists because the old rotationally integrated vibro-only
fit used it.  It is still useful as a diagnostic, but it is not the main
production path.

## Current coronal fit

The current production path is:

```python
cm = CoronaModel(bp)
cm.coronal_autofit()
```

`coronal_autofit()` fits only:

```text
Tvib
```

It holds these values fixed from the preceding `BoltzmannPlot`:

```text
alpha, beta, Trot1, Trot2
```

The objective is a normalized shape comparison between the measured d-state
population and the coronal model's projected d-state population.  The same
Q-branch mask from `bp.mask` is applied before comparison.

## X-state reuse pattern

The coronal model reuses the fitted two-temperature rotational shape when
constructing the trial X-state population.  The isotope-specific mapping is:

| Isotopologue | X-state hot-component pattern |
|--------------|-------------------------------|
| D2 | `[beta, beta, alpha, alpha]` |
| H2 | `[beta, alpha, alpha]` |

For H2, the old Ishihara-style pattern was:

```text
[beta, beta, alpha]
```

That pattern is still visible in a code comment, but the current analyzer
uses `[beta, alpha, alpha]`.

## Legacy vibro-only fit

`CoronaModel.fit_vibro_ishi()` is the old rotationally integrated
Ishihara-style fit.  It fits `Tvib` from `bp.nd_vibrofit` and ignores the
rotationally resolved R-matrix path.

Use it only as a diagnostic or historical comparison.  The canonical
notebook and regression tests use `coronal_autofit()`.

## Sparse H2 notes

For sparse H2 data, including the usual limited `2-2` coverage, the intended
workflow is still:

```python
bp.autofit()
cm = CoronaModel(bp)
cm.coronal_autofit()
```

The sparse data first constrain the d-state Boltzmann shape.  Then the
coronal fit keeps that rotational shape fixed and only searches for the
best `Tvib`.
