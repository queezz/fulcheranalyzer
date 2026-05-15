# Archive: Liptak partial refactor (`fulcher_analyser`)

This directory preserves a partial refactor contributed by A. Liptak
(committed in `bbef1b7`, 2022).

## What it was

A modernisation attempt focused on the *raw-spectrum* layer — the step
*upstream* of what the published pipeline actually uses:

| File | Purpose |
|---|---|
| `fulcher_analyser/spectrum.py` | Reads the proprietary echelle-spectrometer text format into an `xarray.Dataset` (wavelength × frame). Also provides an annotated Q-branch diagnostic plot. |
| `fulcher_analyser/molecular_data.py` | Hard-coded Python dict of Fulcher-α Q-branch wavelengths for H2 (4×11) and D2 (5×15), used only by the plot above. |
| `fulcher_analyser/__main__.py` | Minimal argparse CLI: `python -m fulcher_analyser -f <frame> -s <path>` → load + plot. |
| `fulcher_analyser/__init__.py` | `sys.path` hack so the submodules could import each other without being a proper package. |

## Why it was archived

The refactor was never integrated with the physics pipeline
(`fulcheranalyzer/coronalmodel.py`).  It only covered spectrum
visualisation — no Gaussian fitting, no Q-branch intensity extraction, no
Boltzmann analysis, no coronal model.  No notebook calls into it except
`examples/load_spectra.ipynb`.

The `molecular_data.py` wavelength dict duplicates the data already stored
in `data_molecular/fulcher-α_band_wavelength.txt` and
`data_molecular/fulcher-α_band_wavenumber_D2.txt`.

## Status

Read-only reference.  Do not import from here in production code.
When the spectrum-loading layer is eventually built, `spectrum.py` here is
a useful design reference for the xarray-based approach.
