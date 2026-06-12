"""QC plotting helpers for coronal/Tvib fit inspection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._utils import flatdf

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .coronal_model import CoronaModel


def plot_coronal_qc(
    model: "CoronaModel",
    *,
    ax: "Axes | None" = None,
    title: str | None = None,
) -> "Figure":
    """Plot measured d-state populations against the fitted coronal model."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
    else:
        fig = ax.figure

    exp_values = flatdf(model.bp.nd[model.bp.mask])
    exp_errors = flatdf(model.bp.nd_err[model.bp.mask])
    model_values = flatdf(model.nd[model.bp.mask])

    exp_sum = exp_values.sum()
    model_sum = model_values.sum()
    exp_norm = exp_values / exp_sum
    exp_err_norm = exp_errors / exp_sum
    model_norm = model_values / model_sum
    x = np.arange(len(exp_norm))

    ax.errorbar(
        x,
        exp_norm,
        yerr=exp_err_norm,
        fmt="o",
        ms=4.5,
        capsize=2.5,
        lw=0.8,
        color="#2f6fce",
        label="measured",
    )
    ax.plot(
        x,
        model_norm,
        "s-",
        ms=4.0,
        lw=1.1,
        color="#d54a2a",
        label=f"coronal Tvib={model.tvib:.0f} K",
    )

    yerr_flat = getattr(model, "yerr_flat", None)
    if yerr_flat is not None and len(yerr_flat) == len(model_norm):
        lower = np.clip(model_norm - yerr_flat, a_min=0.0, a_max=None)
        upper = model_norm + yerr_flat
        ax.fill_between(x, lower, upper, color="#d54a2a", alpha=0.18, label="Tvib +/- 1 sigma")

    labels = flatdf(model.bp.qnames[model.bp.mask])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel("Q-branch transition")
    ax.set_ylabel("Normalized d-state population")
    if title:
        ax.set_title(title)
    ax.grid(True, color="0.88", lw=0.7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
