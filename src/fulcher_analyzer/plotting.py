"""
CoronaModel plotting helpers.

All functions take a ``CoronaModel`` instance as their first argument so they
can be called both as standalone functions and as thin method wrappers on the
class.  No circular imports: ``CoronaModel`` is only referenced under
``TYPE_CHECKING``.
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ._utils import flatdf, figsize

if TYPE_CHECKING:
    from .coronal_model import CoronaModel


# ---------------------------------------------------------------------------
# Module-level helper (not a method wrapper)
# ---------------------------------------------------------------------------

def plot_rmatrix(Rm, shapes, text="R-matrix"):
    """
    Plot 2-D R-matrix for inspection.
    vdl, jdl, vxl, jxl = shapes
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    vdl, jdl, vxl, jxl = shapes
    xticks = np.arange(vxl + 1) * jxl
    yticks = np.arange(vdl + 1) * jdl

    cm = plt.cm.seismic
    cm.set_under("#736200")
    cm.set_over("#15ad1a")

    im = plt.imshow(Rm, origin="lower", cmap=cm, norm=mpl.colors.LogNorm())
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(im, extend="both", cax=cax)

    plt.sca(ax)
    ax.set_aspect(1)
    ax.set_facecolor("#a8a594")
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    plt.xlabel(r"$v\cdot J$ ($X^1\Sigma^+_g$)")
    plt.ylabel(r"$v$'$\cdot (J$'$-1)$ ($d^3\Pi_{u}$)")
    plt.xlim(-1, xticks[-1])
    plt.ylim(-1, yticks[-1])
    plt.grid(c="k")
    ax.text(
        -0.12,
        0,
        rf"$vd\cdot Jd = {vdl}\cdot {jdl}$",
        transform=ax.transAxes,
        rotation=90,
    )
    ax.text(
        -0.12,
        0.75,
        f"$\\nexists Jd = 0$, $Jd>0$!!!",
        transform=ax.transAxes,
        rotation=90,
    )
    ax.text(-0.12, -0.1, rf"$vx\cdot Jx = {vxl}\cdot {jxl}$", transform=ax.transAxes)
    ax.text(0.7, -0.1, f"{text}", transform=ax.transAxes)

    [
        ax.text(vxl * jxl + 2, j + jdl / 2, i, va="center", ha="center")
        for i, j in enumerate(yticks[:-1])
    ]
    [
        ax.text(j + jxl / 2, vdl * jdl + 2, i, va="center", ha="center")
        for i, j in enumerate(xticks[:-1])
    ]


# ---------------------------------------------------------------------------
# Style helper
# ---------------------------------------------------------------------------

def prep_corona_style(model: "CoronaModel", ms: int = 8) -> None:
    """Populate ``model.style_coronaplot`` and ``model.style_rovib``."""
    model.style_coronaplot = [
        {
            "capsize": 0.8 * ms,
            "capthick": 0.4,
            "elinewidth": 0.4,
            "fmt": i,
            "ecolor": j,
            "markeredgecolor": j,
            "color": j,
            "label": k,
            "ms": l,
            "lw": 0.6,
        }
        for i, j, k, l in zip(
            ["o-.", "s-."],
            ["k", "r"],
            ["experiment", "reconstructed"],
            [ms, 0.7 * ms],
        )
    ]

    model.style_rovib = [
        {
            "lw": 0.8,
            "marker": i,
            "mfc": j,
            "c": "k",
            "ls": "-.",
            "ms": ms,
            "mew": k,
            "label": f"v={s}",
        }
        for s, (i, j, k) in enumerate(
            zip(
                ["o", "o", "x", "s"],
                ["k", "w", "k", "k"],
                np.array([1, 2, 2, 1]) * 0.5,
            )
        )
    ]


# ---------------------------------------------------------------------------
# Plotting functions (standalone, first arg = CoronaModel instance)
# ---------------------------------------------------------------------------

def plot_fit_ishi(model: "CoronaModel", **kws) -> None:
    """'Nice' plot of the vibro fit and gauge for Ishihara's fit."""
    import matplotlib.pyplot as plt

    fig = plt.gcf()
    fig.set_size_inches([7, 6])

    tvs = np.arange(1, 31) * 1000
    colors = plt.cm.jet(np.linspace(0, 1, len(tvs)))
    gls = []

    for i, t in enumerate(tvs):
        nv = model.f_vibro("", t, False)
        (ll,) = plt.plot(nv.sum(axis=0), ".-", c=colors[i], label=f"{t//1000}")
        gls.append(ll,)

    try:
        tvib = kws.get("tvib", model.tvib)
    except Exception:
        tvib = kws.get("tvib", 1000)

    nv = model.f_vibro("", tvib, False)

    (l1,) = plt.plot(
        nv.sum(axis=0), "s-", c="r", ms=10, label=f"$T_{{vib}} = {tvib:.0f}$K"
    )
    nde = model.bp.nd_vibrofit
    (l2,) = plt.plot(nde / nde[0] * nv.sum(axis=0)[0], "ko-", label="data")

    plt.locator_params(nbins=4)
    ax = plt.gca()
    ax.text(0.98, 1.02, "T/1000K", transform=ax.transAxes)
    pl0 = plt.legend(
        handles=gls, loc=1, prop={"size": 7}, bbox_to_anchor=[1.13, 1.01]
    )
    ax.add_artist(pl0)
    pl1 = plt.legend(handles=[l1, l2])
    plt.xlabel("vibrational q.n.")
    plt.ylabel(r"relative pop. of $d^3\Pi_u$")


def plot_xd(model: "CoronaModel") -> None:
    """Plot synthetic nX and nd."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches([5, 5])

    [axs[0].plot(model.nx[i], label=f"v={i}") for i in range(model.nx.shape[1])]
    [axs[1].plot(model.nd[i], label=f"v={i}") for i in range(model.nd.shape[1])]
    [ax.legend(fontsize=10) for ax in axs]
    axs[0].text(0.5, 0.8, r"$X^1\Sigma^{+}_{g}$", transform=axs[0].transAxes)
    axs[1].text(0.5, 0.8, r"$d^3\Pi_{u}$", transform=axs[1].transAxes)


def plot_xd_flat(model: "CoronaModel", **kws) -> None:
    """Plot flattened X and d populations."""
    import matplotlib.pyplot as plt
    from math import ceil

    ax = kws.get("ax", plt.gca())

    ndeflat = model.bp.nd_sc.values.flatten(order="f")
    ndflat = model.nd.values.flatten(order="f")

    mask = model.bp.mask.values.flatten(order="f")
    x_ndem = np.arange(mask.shape[0])[mask]
    ndemflat = ndeflat[mask]

    plt.sca(ax)
    plt.plot(ndeflat / ndeflat.sum(), "ko-", label="exp-synth")
    plt.plot(x_ndem, ndemflat / ndeflat.sum(), "C3s", label="data")
    plt.plot(ndflat / ndflat.sum(), "C2x-.", label="corona")
    plt.legend()
    ax.set_xlim(-1, ceil(model.nd.shape[0] * model.nd.shape[1] / 5) * 5)


def plot_paper_compare(model: "CoronaModel", **kws) -> None:
    """
    Plot comparison of experimental d-state population with Corona-Model
    reconstruction (fig. 4.24 Ishihara thesis / figs. 11–12 JQSRT-2021).
    """
    import matplotlib.pyplot as plt

    ax = kws.get("ax", plt.gca())

    ndeflat = flatdf(model.bp.nd[model.bp.mask])
    nd_err = flatdf(model.bp.nd_err)
    ndesflat = flatdf(model.bp.nd_sc[model.bp.mask])
    ndflat = flatdf(model.nd[model.bp.mask])

    ndflat = ndflat / ndflat.sum()
    nd_err = nd_err / ndeflat.sum()
    ndeflat = ndeflat / ndeflat.sum()
    ndesflat = ndesflat / ndesflat.sum()

    plt.sca(ax)
    ms = kws.get("ms", 4)
    model.prep_style(ms=ms)
    style = model.style_coronaplot
    x = np.arange(len(ndeflat))
    plt.errorbar(x, ndeflat, yerr=nd_err, **style[0])
    showbp = kws.get("showbp", False)
    if showbp:
        plt.plot(ndesflat, "C1o-.", label="BP-fit")
    yerr = kws.get("yerr", [])
    if not len(yerr):
        plt.plot(ndflat, "rs:", label="corona")
    else:
        plt.errorbar(x, ndflat, yerr=yerr, **style[1])

    plt.legend()
    plt.xticks(np.arange(0, len(ndflat), 1))
    nms = model.bp.qnames[model.bp.mask]
    ax.set_xticklabels(flatdf(nms), rotation=90)
    ax.set_xlabel("Q-branch transition [QN'(v'-v'')]")
    ax.set_ylabel(r"$\mathrm{n_{d v^{\prime} N^{\prime}}}$ [a.u.]")


def plot_rtp(model: "CoronaModel") -> None:
    """Plot rtp (radiation transition probability)."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    im = plt.imshow(
        model.rtp, origin="lower", cmap=plt.cm.seismic, norm=mpl.colors.LogNorm()
    )
    plt.colorbar(im)
    ax = plt.gca()
    ax.set_xlabel("J'-1 (d)")
    ax.set_ylabel("J (X)")


def plot_R(model: "CoronaModel") -> None:
    """Plot R-matrix."""
    import matplotlib.pyplot as plt

    shp = model.rshapelist
    plot_rmatrix(model.Rm2d, shp, text="R-matrix")
    plt.gcf().set_size_inches([9, 9])


def plot_fcf(model: "CoronaModel") -> None:
    """Plot Franck-Condon factors."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    fcf = model.fcf[:4, :4]
    im = plt.imshow(
        fcf, origin="lower", cmap=plt.cm.seismic, norm=mpl.colors.LogNorm()
    )
    plt.colorbar(im)
    ax = plt.gca()
    ax.set_xticks(range(fcf.shape[0]))
    ax.set_yticks(range(fcf.shape[1]))
    ax.set_xlabel(r"$v$'' ($a^3\Sigma_g^+$)")
    ax.set_ylabel(r"$v$' ($d^3\Pi_u$)")


def plot_ccs(model: "CoronaModel") -> None:
    """Plot collision cross-sections."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    ccs = model.ccs.values[:4, :4]
    im = plt.imshow(ccs, origin="lower", cmap=plt.cm.seismic)
    plt.colorbar(im)
    ax = plt.gca()
    ax.set_xticks(range(ccs.shape[0]))
    ax.set_yticks(range(ccs.shape[1]))
    ax.set_xlabel(r"$v$'' ($a^3\Sigma_g^+$)")
    ax.set_ylabel(r"$v$' ($d^3\Pi_u$)")


def plot_coronal_result(model: "CoronaModel") -> None:
    """Plot two panels: synthetic population and data comparison."""
    import matplotlib.pyplot as plt

    model.coronal_fit_formula([], *model.fitres)
    fig, axs = plt.subplots(2, 1)
    plot_xd_flat(model, ax=axs[0])
    plot_paper_compare(model, ax=axs[1])
    fig.set_size_inches([12, 10])


def plot_contribution(model: "CoronaModel", Tvib: int = 7000) -> None:
    """Plot contribution of X-state vibrational levels to d-state population."""
    import matplotlib.pyplot as plt
    from math import ceil
    from scipy.constants import Boltzmann, elementary_charge

    kb = Boltzmann / elementary_charge

    model.calculate_e_cross()
    ccs = model.ccs.values
    fcf = model.fcf
    E_vib = model.E_vib

    v_range = 15
    vv_range = 4
    nvv = np.zeros([v_range, vv_range])
    for v in range(v_range):
        for vv in range(vv_range):
            nvv[v, vv] = (
                ccs[v, vv]
                * fcf[v, vv]
                * np.exp(-(E_vib[v] - E_vib[0]) / (Tvib * kb))
                / model.Asum[vv]
                * 1e9
            )

    mks = ["o", "s", "^", ">"]
    for v in range(v_range):
        if v < 4:
            plt.plot(
                nvv[v],
                "-",
                color=plt.cm.hsv(v / 4),
                marker=mks[v],
                ms=10,
                label=f"v={v}",
            )
        else:
            plt.plot(nvv[v], ".--", c="gray", alpha=0.5)
    plt.plot(nvv[14], ".--", c="gray", label="v=4-14", alpha=0.5)

    plt.xticks(np.arange(0, 4, 1))
    plt.xlim(-0.1, 3.1)
    plt.ylim(-0.3, ceil(nvv.max()) + 0.5)
    plt.xlabel(r"Vibrational Number $v$'")
    plt.ylabel(r"Contribution to $n_{d}(v$'$)$ from $n_{X}(v)$")
    plt.legend(loc=1, bbox_to_anchor=[1, 0.85])
    plt.gcf().set_size_inches([8, 5])
    ax = plt.gca()
    ax.text(
        0,
        1.03,
        f"$T_{{vib}}$={Tvib}K, isotop={model.isotop}",
        transform=ax.transAxes,
    )


def plot_popx_paper(model: "CoronaModel", fontsize: int = 11, ms: int = 4) -> None:
    """Plot X-state population (figs 9–10 of the paper)."""
    import matplotlib.pylab as plt

    model.prep_style(ms=ms)
    kws = model.style_rovib
    [
        plt.fill_between(
            model.EX[i],
            model.yerr_rot["minus"][i] / model.nxbp[0][0],
            model.yerr_rot["plus"][i] / model.nxbp[0][0],
            color="gray",
            alpha=0.5,
            ec="none",
        )
        for i in model.yerr_rot["minus"]
    ]
    [
        plt.plot(model.EX[i], model.nxbp[i] / model.nxbp[0][0], **kws[i])
        for i in model.nxbp
    ]
    plt.fill_between([], [], [], color="gray", alpha=0.5, label="error", ec="none")

    plt.yscale("log")
    plt.xlabel("Rotational Energy [eV]")
    plt.ylabel(
        r"$\mathrm{\frac{n_{XvN}}{(2N+1)\,g_{\mathrm{as}}^{N}}}$ [a.u.]",
        fontsize=fontsize + 2,
    )

    plt.legend()
    ax = plt.gca()
    ax.text(model.EX.values[0, 0] + 0.02, 0.95, "N=0", fontsize=fontsize - 2)
    ax.text(
        model.EX.values[-1, 0] * 1.05,
        (model.yerr_rot["minus"][0] / model.nxbp[0][0]).values[-1] + 0.02,
        f"N={model.popshape['jxmax']}",
        ha="center",
        fontsize=fontsize - 2,
    )
    plt.xlim(-0.05, 1.0)

    plt.gcf().set_size_inches(figsize(8))
