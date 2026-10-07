# Changelog

## 0.2.1 - 2026-08-29

- Derive measured d-state and Boltzmann populations through one consistent
  Einstein-A path instead of mixing incompatible A and Franck–Condon tables.
- Support four-band H2 intensity tables using the packaged diagonal A values.
- Move the H2 and D2 Einstein-A matrices into cited data files and identify
  them as Lavrov et al. (2015) Tables 5/7 comparison columns.
- Add selectable complete recommended semiempirical H2/D2 matrices with
  uncertainties and independent lifetime regression checks.
- Raise the hot rotational-temperature ceiling to 6000 K and expose bound
  contact through `BoltzmannPlot.fit_at_bound` and batch-summary flags.

## 0.2.0 - 2026-06-13

- Add `fulcher-analyze-batch` for running Boltzmann and coronal analysis from extractor intensity tables.
- Add rerunnable batch controls: `--plot-kind`, `--qc-every`, `--resume`, and `--checkpoint-every`.
- Add `--workers` for process-based parallel frame analysis.
- Separate analyzer QC rendering with `--plot` from saved QC tables.
- Add Boltzmann and coronal QC plot outputs for blink/review workflows.
- Stabilize analyzer QC plot axes, legends, and coronal transition positions for blink review.
- Document analyzer-side outputs for extractor-to-analyzer runs.
