"""
Free helper functions for two-temperature Boltzmann model fitting and plotting.
"""
import numpy as np
import pandas as pd

from .molecular_constants import MolecularConstants
from ._utils import flatdf


def expsum(e, T1, T2, a, constant):
    """
    Mixing of two Boltzman distributions
    """
    from scipy.constants import Boltzmann, elementary_charge

    kb = Boltzmann / elementary_charge
    return np.log(
        constant * ((1 - a) * np.exp(-e / (kb * T1)) + a * np.exp(-e / (kb * T2)))
    )


def two_t_all_v(Erot, *param, **kws):
    """
    Constract two exponent distribution for all vibrational transitions
    populate DataFrame
    a, b - exponent multiplyers for v=0 and others, v>0 assumed to have same multiplier
    T1, T2 - temperatures of the two exponents
    cs - tuple of constants for all v.     
    """
    a, b, T1, T2 = param[:4]
    cs = param[4:]
    aps = [a] + [b for i in range(Erot.shape[1] - 1)]
    # temperature fractions
    mol = MolecularConstants()
    mol.calculate_tfrac(
        Erot.shape[1], norm=True
    )  # norm=True is essential! Otherwise produces T2*2.
    isotop = kws.get("isotop", "d")
    tfrac = mol.frac[isotop].values
    ns = pd.DataFrame(
        [
            expsum(Erot[i], tc * T1, tc * T2, a, c)
            for tc, a, i, c in zip(tfrac, aps, Erot, cs)
        ]
    ).T
    melt = kws.get("melt", True)

    if melt:
        return flatdf(ns)
    else:
        return ns


def plot_n_all_v(n, E, **kws):
    """
    input: experimental populations in [2D] DataFrame with missing values as NaN
    Energy - full DataFrame with theoretical rotational energy E
    cycles through columns in n (corresponding to vibrational level), removes nans,
    and plots
    """
    from matplotlib.pyplot import plot

    nomarker = kws.get("nomarker", False)
    noline = kws.get("noline", False)
    mkws = [
        {"label": f"v'={l}", "marker": m, "ls": "--"}
        for l, m in zip(range(4), ["s", "o", "d", "x"])
    ]
    if nomarker:
        mkws = [{"label": f"v'={l}", "ls": "--"} for l in range(4)]
    if noline:
        mkws = [
            {"label": f"v'={l}", "marker": m, "ls": ""}
            for l, m in zip(range(4), ["s", "o", "d", "x"])
        ]

    log = kws.get("log", True)
    for p, (i, j) in enumerate(zip(n, E)):
        m = ~pd.isna(n[i])
        if log:
            plot(E[j][m], np.log(n[i][m]), **mkws[p])
        else:
            plot(E[j][m], n[i][m], **mkws[p])


def fittext(p):
    """
    Return text for matplotlib plot with fitting parameters, a,b,T1,T2 from two_t_all_v()
    """
    return (
        f"$\\alpha={p[0]:.2f}$ $\\beta={p[1]:.2f}$ $T_1={p[2]:.0f}$K $T_2={p[3]:.0f}$K"
    )
