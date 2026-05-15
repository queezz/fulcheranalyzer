"""
Backward-compatibility facade for fulcher_analyzer.coronalmodel.

All public names are re-exported from their canonical modules so that
existing code of the form:

    from fulcher_analyzer import coronalmodel as fcm
    fcm.CoronaModel(...)
    fcm.BoltzmannPlot(...)
    fcm.MolecularConstants()
    ...

continues to work without modification.
"""
import numpy as np, pandas as pd  # noqa: F401  (kept for any downstream star-imports)
from os.path import join, abspath

from ._constants import package_directory
from .molecular_constants import MolecularConstants  # noqa: F401
from .intensity_io import read_intensities, write_intensities  # noqa: F401
from .boltzmann import (  # noqa: F401
    expsum,
    two_t_all_v,
    plot_n_all_v,
    fittext,
    BoltzmannPlot,
    ABSOLUTESIGMA,
)
from ._utils import (  # noqa: F401
    delta_kro,
    g_as,
    g_as_vector,
    tjpo_vector,
    reshape_4d2d,
    flatdf,
    flatdf_1,
    figsize,
)
from .coronal_model import CoronaModel, plot_rmatrix  # noqa: F401

MOLECULAR_DATA_FOLDER = abspath(join(package_directory, "..", "..", "data_molecular"))
DATA_FOLDER = abspath(join(package_directory, "..", "..", "data"))


def set_tick_size(ax, *size):
    """ 
    Set matplotlib axis tick sizes
    For some reason the length from style is ignored.
    """
    width_major, length_major, width_minor, length_minor = size
    ax.xaxis.set_tick_params(width=width_major, length=length_major, which="major")
    ax.xaxis.set_tick_params(width=width_minor, length=length_minor, which="minor")
    ax.yaxis.set_tick_params(width=width_major, length=length_major, which="major")
    ax.yaxis.set_tick_params(width=width_minor, length=length_minor, which="minor")
