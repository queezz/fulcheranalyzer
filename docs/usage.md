# Usage

## Installation

```bash
python -m pip install -e ".[dev]"
```

Activate the project venv first if needed:

```bash
source ~/.venvs/fulcher/bin/activate
```

---

## Canonical import (recommended)

```python
from fulcher_analyzer import BoltzmannPlot, CoronaModel, read_intensities

intensities, errors = read_intensities(150482, 7)

bp = BoltzmannPlot(intensities, "d")   # "d" for deuterium, "h" for hydrogen
bp.autofit()

model = CoronaModel(bp)
model.coronal_autofit()
```

`read_intensities` returns a tuple of `(intensity_df, error_df)` where each is
a `pandas.DataFrame` indexed by spectral line.

`BoltzmannPlot` accepts either the tuple directly or just the intensity
DataFrame; pass `isotop="h"` for hydrogen.

### Bundled example data

`read_intensities(150482, 7)` and `read_intensities(152478, 10)` load the two
bundled example shots (D₂ and H₂) that ship with the package inside
`fulcher_analyzer/example_data/intensities/`. No extra download is required.

To load intensity CSV files from your own directory:

```python
intensities, errors = read_intensities(my_shot, my_frame, data_folder="/path/to/my/data")
```

---

## Batch analysis from extractor outputs

Use `fulcher-analyze-batch` after `fulcher-extractor` has written a dataset
with `intensities/` and, when available, `fit_reports/`:

```bash
fulcher-analyze-batch --input-dir /path/to/run/dataset/intensities
```

For PSI-style runs, keep the paths in a TOML plan and add an `[analyze]`
section:

```toml
[analyze]
input_dir = "dataset/intensities"
output_dir = "dataset"
fit_report_dir = "dataset/fit_reports"
manifest = "scan/selected_frames.csv"
isotopologue = "h"
qc_every = 0
plot_kind = "none"
workers = 11
```

CLI flags override plan values. The batch command writes:

```text
dataset/boltzmann_summary.csv
dataset/coronal_summary.csv
dataset/tables/*_boltzmann_qc_points.csv
```

No figures are rendered during fitting. The plot pass reads the saved QC
tables and writes:

```text
dataset/plots/boltzmann/*_boltzmann_qc.png
dataset/plots/coronal/*_coronal_tvib_*K_qc.png
```

Rerun controls:

```bash
fulcher-analyze-batch --plan h2_dataset_plan.toml --resume
fulcher-analyze-batch --plan h2_dataset_plan.toml --plot
fulcher-analyze-batch --plan h2_dataset_plan.toml --plot --plot-kind boltzmann
fulcher-analyze-batch --plan h2_dataset_plan.toml --plot --plot-kind coronal
```

`--resume` skips frames already marked `ok` in both summary CSVs. Summaries are
checkpointed after each processed frame by default; use `--checkpoint-every N`
to reduce write frequency on very large runs.

`--plot` regenerates analyzer QC plots from saved QC tables as a separate
pass, analogous to the extractor `plot` stage. It does not refit frames or
rewrite summary CSVs. `--workers N` runs independent frames in parallel worker
processes. Summary CSVs are still merged in manifest/discovery order, so
blink-review filenames and tables remain deterministic even when frames finish
out of order.


---

## Accessing molecular constants directly

```python
from fulcher_analyzer import MolecularConstants

mc = MolecularConstants("d")   # deuterium
print(mc.transitions)
```
