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
