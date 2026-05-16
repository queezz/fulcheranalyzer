# Physics overview

`fulcher_analyzer` analyses **Fulcher-α emission spectroscopy of molecular
hydrogen isotopologues** (H₂ and D₂) and infers the rovibrational population
of the electronic ground state from measured emission line intensities.

The analysis proceeds in two stages:

1. **d-state Boltzmann fit.** Measured Q-branch line intensities are converted
   into relative populations of the electronically-excited d ³Π<sub>u</sub>
   state. A two-temperature rotational model is fitted to the d-state Boltzmann
   plot, yielding the fitted parameters `alpha`, `beta`, `Trot1`, and `Trot2`.
2. **Coronal-model projection.** A trial vibrational temperature `Tvib` is
   used to construct an X ¹Σ<sub>g</sub><sup>+</sup> ground-state rovibrational
   population. This X-state population is projected onto the d-state through
   a precomputed R-matrix (Franck–Condon × electron-impact cross-section ×
   rotational transition probability) and compared to the experimental
   d-state population. A second-stage fit varies **only** `Tvib`; the
   rotational parameters from stage 1 are inherited and held fixed.

The final scientific quantity of interest is the effective X-state
**vibrational temperature** `Tvib` (with uncertainty `Tviberr`).

## Scope and provenance

This package is a refactored version of the historical implementation
underlying the published workflow (Kuzmin *et al.*, JQSRT 2021). The
numerical behaviour is preserved by regression tests against two canonical
paper-reproduction examples:

| Isotopologue | Shot   | Frame |
|--------------|--------|-------|
| D₂           | 150482 | 7     |
| H₂           | 152478 | 10    |

These pinned numbers are not just demos — they are the regression anchors
in `tests/test_paper_reproduction.py`. The intensity CSV files for both
shots ship inside the package (`fulcher_analyzer/example_data/intensities/`)
and are loaded by default through `read_intensities(shot, frame)`.

## What this documentation does *not* do

These pages summarise *what the code does*, not a full re-derivation of the
physics. Where the implementation follows the published / Ishihara-style
workflow without an in-code citation, the docs say so rather than invent a
derivation. For the full physical motivation see the cited paper and the
references in the source docstrings.

## Read next

- [Boltzmann population fit](boltzmann.md) — stage 1, d-state.
- [Coronal model](coronal_model.md) — stage 2, X-state via R-matrix.
