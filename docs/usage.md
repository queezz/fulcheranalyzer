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

## Legacy compatibility import

Notebooks and scripts written against the original monolithic `coronalmodel`
module continue to work without modification:

```python
from fulcher_analyzer import coronalmodel as fcm

inte = fcm.read_intensities(shot, frame)
bp   = fcm.BoltzmannPlot(inte, "d")
bp.autofit()

cm = fcm.CoronaModel(bp)
cm.coronal_autofit()
```

`coronalmodel.py` is a thin backward-compatibility facade that re-exports all
names that were previously available through the old monolithic module.

---

## Accessing molecular constants directly

```python
from fulcher_analyzer import MolecularConstants

mc = MolecularConstants("d")   # deuterium
print(mc.transitions)
```
