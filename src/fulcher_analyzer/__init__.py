"""
Molecular hydrogen emission analyzer.

Identify Fulcher-alpha lines in emission spectra, deconvolute,
calculate intensities, and calculate d-state and X-state
ro-vibrational populations.

Canonical public API
--------------------

    from fulcher_analyzer import BoltzmannPlot, CoronaModel, read_intensities
"""
__version__ = "0.0.1"

from .molecular_constants import MolecularConstants
from .boltzmann import BoltzmannPlot
from .boltzmann_qc import (
    apply_boltzmann_qc_mask,
    boltzmann_qc_points,
    plot_boltzmann_qc,
)
from .coronal_model import CoronaModel
from .intensity_io import read_intensities, write_intensities

__all__ = [
    "MolecularConstants",
    "BoltzmannPlot",
    "apply_boltzmann_qc_mask",
    "boltzmann_qc_points",
    "CoronaModel",
    "plot_boltzmann_qc",
    "read_intensities",
    "write_intensities",
]
