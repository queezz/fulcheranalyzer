"""QC plotting helpers for Boltzmann population inspection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .boltzmann import BoltzmannPlot


@dataclass(frozen=True)
class BandStyle:
    color: str
    marker: str
    linestyle: str


BAND_STYLES: dict[str, BandStyle] = {
    "0-0": BandStyle("#2f6fce", "s", "-"),
    "1-1": BandStyle("#d54a2a", "o", "--"),
    "2-2": BandStyle("#1f9a55", "D", "-."),
    "3-3": BandStyle("#6f6f6f", "x", ":"),
}
DEFAULT_BAND_STYLE = BandStyle("#6f6f6f", "x", ":")


def band_style(band: str | int) -> BandStyle:
    """Return the QC style for a Fulcher diagonal band."""
    band_name = _band_name(band)
    return BAND_STYLES.get(band_name, DEFAULT_BAND_STYLE)


def boltzmann_qc_points(
    bp: "BoltzmannPlot",
    *,
    max_fit_relerr: float = 1.0,
    fit_report: str | Path | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a tidy table of finite positive Boltzmann points for QC plotting."""
    report_lookup = _fit_report_lookup(fit_report)
    rows = []
    for band_index in bp.nd_rel:
        band = _band_name(band_index)
        for row_index, value in bp.nd_rel[band_index].items():
            if not np.isfinite(value) or value <= 0.0:
                continue
            energy = bp.Ed[band_index].loc[row_index]
            relerr = bp.relerr[band_index].loc[row_index]
            if not np.isfinite(energy) or not np.isfinite(relerr):
                continue
            report_row = report_lookup.get((int(row_index) + 1, band), {})
            boltzmann_action = str(report_row.get("boltzmann_fit_action", "fit"))
            boltzmann_reason = str(report_row.get("boltzmann_fit_reason", ""))
            fit_mask = bool(relerr <= max_fit_relerr)
            if boltzmann_action == "exclude":
                fit_mask = False
            rows.append(
                {
                    "line_id": str(report_row.get("line_id", "")),
                    "band_index": int(band_index),
                    "band": band,
                    "N": int(row_index) + 1,
                    "energy_eV": float(energy),
                    "nd_rel": float(value),
                    "relerr": float(relerr),
                    "fit_mask": fit_mask,
                    "boltzmann_fit_action": boltzmann_action,
                    "boltzmann_fit_reason": boltzmann_reason,
                }
            )
    return pd.DataFrame(rows)


def apply_boltzmann_qc_mask(
    bp: "BoltzmannPlot",
    points: pd.DataFrame | None = None,
    *,
    max_fit_relerr: float = 1.0,
    fit_report: str | Path | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Mask a BoltzmannPlot in-place to keep only QC-accepted fit points."""
    if points is None:
        points = boltzmann_qc_points(
            bp,
            max_fit_relerr=max_fit_relerr,
            fit_report=fit_report,
        )

    quality_mask = bp.mask.copy()
    quality_mask.loc[:, :] = False
    for row in points.loc[points["fit_mask"]].itertuples():
        quality_mask.loc[row.N - 1, row.band_index] = True

    for attr in (
        "inte",
        "interr",
        "nd",
        "nd_bol",
        "nd_rel",
        "nd_err",
        "nd_bol_err",
        "relerr",
    ):
        setattr(bp, attr, getattr(bp, attr).where(quality_mask))
    bp.mask = quality_mask
    return points


def plot_boltzmann_qc(
    bp: "BoltzmannPlot",
    points: pd.DataFrame | None = None,
    *,
    ax: "Axes | None" = None,
    title: str | None = None,
    fit_label: str = "double exp",
    max_fit_relerr: float = 1.0,
    show_fit: bool = True,
    annotate: bool = True,
    ylim: tuple[float, float] | None = None,
    fit_report: str | Path | pd.DataFrame | None = None,
) -> "Figure":
    """Plot Boltzmann points and an optional fitted population curve for QC."""
    import matplotlib.pyplot as plt

    if points is None:
        points = boltzmann_qc_points(
            bp,
            max_fit_relerr=max_fit_relerr,
            fit_report=fit_report,
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
    else:
        fig = ax.figure

    first_excluded_label = True
    for band, group in points.groupby("band", sort=True):
        style = band_style(band)
        band_index = int(group["band_index"].iloc[0])
        fit_group = group.loc[group["fit_mask"]]
        excluded = group.loc[~group["fit_mask"]]

        if not fit_group.empty:
            ax.errorbar(
                fit_group["energy_eV"],
                fit_group["nd_rel"],
                yerr=fit_group["relerr"] * fit_group["nd_rel"],
                fmt=style.marker,
                ms=5.0,
                capsize=2.5,
                lw=0.8,
                color=style.color,
                label=band,
            )
            if annotate:
                _annotate_points(ax, fit_group, color=style.color, weight="normal")

        if not excluded.empty:
            ax.scatter(
                excluded["energy_eV"],
                excluded["nd_rel"],
                marker=style.marker,
                s=38,
                color=style.color,
                alpha=0.45,
                zorder=3,
            )
            ax.scatter(
                excluded["energy_eV"],
                excluded["nd_rel"],
                marker="x",
                s=95,
                color="black",
                lw=1.4,
                label="not fit" if first_excluded_label else None,
                zorder=4,
            )
            first_excluded_label = False
            if annotate:
                _annotate_points(ax, excluded, color="black", weight="bold", offset=(5, 5))

        if show_fit and hasattr(bp, "nd_bol_synth"):
            fit_y = bp.nd_bol_synth[band_index]
            fit_x = bp.Ed[band_index]
            ax.plot(
                fit_x,
                fit_y,
                color=style.color,
                lw=1.25,
                ls=style.linestyle,
                label=f"{band} {fit_label}",
            )

    ax.set_xlabel("Rotational Energy [eV]")
    ax.set_ylabel(
        r"$\mathrm{\frac{n_{d v' N'}}{(2N'+1)\,g_{\mathrm{as}}^{N'}}}$ [a.u.]"
    )
    if title:
        ax.set_title(title)
    ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, which="both", color="0.88", lw=0.7)
    ax.legend(title="Band", fontsize=7.5)
    fig.tight_layout()
    return fig


def _annotate_points(
    ax: "Axes",
    points: pd.DataFrame,
    *,
    color: str,
    weight: str,
    offset: tuple[int, int] = (3, 3),
) -> None:
    for row in points.itertuples():
        ax.annotate(
            f"Q{row.N:.0f}",
            (row.energy_eV, row.nd_rel),
            xytext=offset,
            textcoords="offset points",
            fontsize=8 if weight == "bold" else 7,
            color=color,
            fontweight=weight,
        )


def _band_name(band: str | int) -> str:
    if isinstance(band, str):
        return band if "-" in band else f"{band}-{band}"
    return f"{band}-{band}"


def _fit_report_lookup(
    fit_report: str | Path | pd.DataFrame | None,
) -> dict[tuple[int, str], dict[str, object]]:
    if fit_report is None:
        return {}
    if isinstance(fit_report, pd.DataFrame):
        report = fit_report
    else:
        report = pd.read_csv(fit_report)
    required = {"N", "band"}
    if not required.issubset(report.columns):
        missing = ", ".join(sorted(required - set(report.columns)))
        raise ValueError(f"Fit report is missing required columns: {missing}")

    lookup: dict[tuple[int, str], dict[str, object]] = {}
    for row in report.to_dict(orient="records"):
        try:
            key = (int(row["N"]), str(row["band"]))
        except Exception:
            continue
        lookup[key] = row
    return lookup
