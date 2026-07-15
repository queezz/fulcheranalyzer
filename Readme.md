# Molecula Hydrogen spectra analysis.

[![DOI](https://zenodo.org/badge/512373585.svg)](https://doi.org/10.5281/zenodo.21372075)

## Install package

```bash
python -m pip install -e ".[dev]"
```

Install with optional spectrum-loading support (xarray):

```bash
python -m pip install -e ".[dev,spectrum]"
```

## Running tests

```bash
pytest
```

The test suite has two layers:

| File | Purpose |
|---|---|
| `tests/test_smoke.py` | Import, data paths, and DataFrame shape checks. Fast. |
| `tests/test_paper_reproduction.py` | Full D2/H2 Boltzmann + coronal-model fit. Asserts published numerical results (Kuzmin et al., JQSRT 2021). |

The regression tests run the complete pipeline and take ~6 s on a warm
R-matrix cache. They will fail if any physics constant, formula, or
data file is accidentally changed.


## Usage

### Canonical import (recommended)

```python
from fulcher_analyzer import BoltzmannPlot, CoronaModel, read_intensities

# Uses bundled example data (shot 150482 = D2, 152478 = H2):
inte = read_intensities(150482, 7)            # (intensity_df, error_df)

# Or load from a custom directory:
# inte = read_intensities(shot, frame, data_folder="/path/to/data")

bp   = BoltzmannPlot(inte, isotop="d")        # "d" or "h"
bp.autofit()

cm   = CoronaModel(bp)
cm.coronal_autofit()
```

### Batch analysis from extracted intensities

After `fulcher-extractor` has written `intensities/` and `fit_reports/`, run
the saved tables through the analyzer without reopening SpectroCubes:

```bash
fulcher-analyze-batch --input-dir /path/to/run/dataset/intensities
```

For paper or batch work, prefer a TOML plan:

```bash
fulcher-analyze-batch --plan h2_dataset_plan.toml
```

The analyzer reads `[analyze]` from the plan. CLI flags override plan values
when supplied.

By default, summaries are written next to the `intensities/` directory:

```text
dataset/boltzmann_summary.csv
dataset/coronal_summary.csv
dataset/tables/*_boltzmann_qc_points.csv
```

By default, `fulcher-analyze-batch` fits frames only: it writes summary CSVs
plus plot-ready QC data tables, but it does not render figures. Plot rendering
is a separate pass that reads those saved tables:

```bash
fulcher-analyze-batch --plan h2_dataset_plan.toml --plot
```

The plot pass writes:

```text
dataset/plots/boltzmann/*_boltzmann_qc.png
dataset/plots/coronal/*_coronal_tvib_*K_qc.png
```

Use `--qc-every N` to thin plot output, `--plot-kind boltzmann` or
`--plot-kind coronal` for one QC stage, or `--plot-kind none` for tables only.

Frame analysis is serial by default. Use `--workers N` for process-based
parallel analysis when running large datasets:

```bash
fulcher-analyze-batch --plan h2_dataset_plan.toml --workers 11
```

Useful rerun patterns:

```bash
# Continue a stopped run, keeping successful rows from existing summaries.
fulcher-analyze-batch --plan h2_dataset_plan.toml --resume

# Regenerate only Boltzmann QC plots.
fulcher-analyze-batch --plan h2_dataset_plan.toml --plot --plot-kind boltzmann

# Write only tables and summary CSVs.
fulcher-analyze-batch --plan h2_dataset_plan.toml
```

Summary CSVs are checkpointed after each frame by default, so interrupted runs
can be continued with `--resume`.

## Documentation

Install the docs dependencies and serve locally:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

Then open <http://127.0.0.1:8000> in a browser.

Build static HTML:

```bash
mkdocs build
```

Output goes to `site/`.

## VENV

### Create virtual environment

Linux / macOS:

```bash
python3 -m venv ~/.venvs/fulcher
```

Windows PowerShell:

```powershell
python -m venv "$env:USERPROFILE/.venvs/fulcher"
```

### Activate virtual environment

Linux / macOS:

```bash
source ~/.venvs/fulcher/bin/activate
```

Windows PowerShell:

```powershell
& "$env:USERPROFILE/.venvs/fulcher/Scripts/Activate.ps1"
```
