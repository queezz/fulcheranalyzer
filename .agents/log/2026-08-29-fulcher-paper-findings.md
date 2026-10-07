# Fulcher paper findings received — 2026-08-29

Fleet action letter `20260828-9bbf00c3-a9d36d` was read from
`code/2026-Echelle-workflow`. It reports three findings measured in the new
Fulcher paper pilot:

- `calculate_boltzmann` derives one d-state population through inconsistent
  Einstein-A and Franck–Condon paths;
- the 3000 K upper bound on `Trot2` binds in 8 of 24 band-rich frames;
- H2 analysis hard-codes three Franck–Condon entries even though the packaged
  Einstein-A table covers four diagonal bands.

The paper repository records the first finding as publication gate G1. This
session will make the Einstein-A population canonical, remove the duplicate
Franck–Condon normalization from the Boltzmann stage, support an input-sized
H2 diagonal A-vector, expose bound contact, and verify the changed numerical
baseline. The letter is recorded here before collection as required by the
Fleet mail lifecycle.

## Result

- `nd_bol` is now derived from `nd = I·λ/A` and the rotational/statistical
  degeneracies only. Franck–Condon factors remain in the coronal model, where
  they belong.
- H2 and D2 diagonal A-vectors are sliced to the input band count; a synthetic
  9×4 H2 table verifies use of the packaged (3-3) coefficient.
- The `Trot2` upper limit is 6000 K. Every fit exposes per-parameter lower,
  upper, and combined bound-contact vectors; batch CSVs carry explicit overall,
  `Trot1`, and `Trot2` flags.
- Version advanced from 0.2.0 to 0.2.1 and the corrected reproduction values
  were frozen. The canonical D2/H2 `Trot2` values changed from 1783/2079 K to
  1733/2010 K; `Tvib` changed from 7743/6801 K to 7840/7053 K.

## Verification

```text
C:\Users\queezz\.venvs\fulcher\Scripts\python.exe -m pytest -q
--basetemp <Codex machine scratch>
49 passed in 13.46s

git diff --check
passed (line-ending conversion notices only)
```

`ruff check src tests` could not run because Ruff is not installed in the
declared `fulcher` environment and is absent from the repository's `dev`
dependencies. No dependency policy was expanded in this focused fix.

## Remaining scientific boundary

This repair makes the package internally consistent; it does not complete the
paper's G1 re-anchoring to cited per-line Einstein-A coefficients. The packaged
A matrices remain uncited inherited data and require the separate literature
re-anchoring already tracked by the paper repository.

## Provenance follow-up received

Fleet action letter `20260828-d2a7d430-e52377` was read from the Fulcher paper
repository. It establishes that the packaged `AH` and `AD` matrices reproduce,
element by element and transposed to `(v', v'')`, the comparison-calculation
columns in Tables 5 and 7 of Lavrov, Pozdeev, and Yakovleva (2015),
arXiv:1512.06306, DOI 10.48550/arXiv.1512.06306. The letter also confirms the
0.2.1 population fix physically: the removed Franck–Condon substitution had
dropped the vibrational dependence of the transition moment.

The letter was recorded here before collection. This follow-up will attach the
source and column provenance to the packaged matrices, inspect the paper audit
for complete alternative data, and verify the low-confidence `(1, 7)` H2 digit
flag against the held evidence before changing any number.

### Provenance update result

- Moved the unchanged 4×8 `AH` and `AD` matrices from Python literals to
  `einstein_A_h2.csv` and `einstein_A_d2.csv` with units, orientation, DOI,
  table number, and comparison-column identity in each file.
- `MolecularConstants` now loads those resources and exposes `A_source` and
  `A_source_doi`.
- Added regression checks for both shapes, selected H2/D2 elements, and the
  independent 38.848 ns H2 `v'=0` lifetime cross-check.
- The paper audit contains only the recommended H2 diagonal, not its complete
  4×8 matrix. No selectable alternative was added: doing so would imply a
  complete branching table the supplied evidence does not contain.
- The low-confidence `(v'=1, v''=7)` reading remains unchanged at 4.9920e-3
  s^-1. The audit did not independently resolve the page-image ambiguity, and
  its contribution is physically negligible; changing it without stronger
  source evidence would reduce rather than improve provenance.

Verification after externalization: 50 tests passed in 13.99 s and
`git diff --check` passed with line-ending notices only. Numerical reproduction
values did not move.

## Complete semiempirical matrices received

Fleet action letter `20260828-41bf0367-35a62f` was read from the Fulcher paper
repository. It supplies the complete recommended semiempirical H2 and D2 4×8
branching matrices from Lavrov et al. (2015), including 1 SD uncertainties and
independent lifetime checks against Tables 3 and 4. The authoritative
machine-readable copies are in the paper audit as long-form CSVs with explicit
upper/lower vibrational labels.

This letter is recorded before collection. With full matrices now available,
the package can safely offer both Lavrov columns: retain the existing
comparison calculation as the backward-compatible default and add an explicit
semiempirical selection carrying uncertainties and caveats for below-reach
zeros.

### Dual-table implementation result

- Added full recommended semiempirical H2/D2 value and uncertainty resources.
- `MolecularConstants(a_table="comparison" | "semiempirical")` selects a
  complete matrix and exposes its matching source description; comparison
  remains the compatibility default.
- `BoltzmannPlot(..., a_table=...)` propagates the choice into `CoronaModel`.
  Batch plans/CLI accept `a_table = "semiempirical"` /
  `--a-table semiempirical`, and output summaries record table and DOI.
- Semiempirical coronal runs rebuild the R-matrix in memory. They neither load
  nor overwrite the packaged comparison-table cache.
- All eight semiempirical row-sum lifetimes reproduce the independent source
  checks within 0.02 ns. The complete suite passes: 52 tests in 11.50 s.
- A direct H2 semiempirical workflow completed with `Trot2=2010.274 K`,
  `Tvib=6791.426 K`, and R-matrix shape `(3, 12, 3, 11)`.
