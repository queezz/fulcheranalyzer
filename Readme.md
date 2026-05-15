# Molecula Hydrogen spectra analysis.

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

### Install package

```bash
python -m pip install -e ".[dev,docs]"
```