# Molecula Hydrogen spectra analysis.

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

### Legacy compatibility import (still fully supported)

Notebooks and scripts written against the old monolithic module continue
to work without modification:

```python
from fulcher_analyzer import coronalmodel as fcm

inte = fcm.read_intensities(shot, frame)
bp   = fcm.BoltzmannPlot(inte, "d")
bp.autofit()

cm   = fcm.CoronaModel(bp)
```

All names previously available as `fcm.*` remain accessible through
`coronalmodel.py`, which is now a thin backward-compatibility facade.


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
