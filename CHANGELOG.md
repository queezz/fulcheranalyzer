# Changelog

## 0.2.0 - 2026-06-13

- Add `fulcher-analyze-batch` for running Boltzmann and coronal analysis from extractor intensity tables.
- Add rerunnable batch controls: `--plot-kind`, `--qc-every`, `--resume`, and `--checkpoint-every`.
- Add `--workers` for process-based parallel frame analysis.
- Separate analyzer QC rendering with `--plot-only` from saved QC tables.
- Add Boltzmann and coronal QC plot outputs for blink/review workflows.
- Stabilize analyzer QC plot axes, legends, and coronal transition positions for blink review.
- Document analyzer-side outputs for extractor-to-analyzer runs.
