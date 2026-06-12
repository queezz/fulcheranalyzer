"""Batch Boltzmann/coronal analysis for saved Fulcher intensity tables."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import pandas as pd
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal installs.
    tqdm = None

from .boltzmann import BoltzmannPlot
from .boltzmann_qc import (
    apply_boltzmann_qc_mask,
    band_style,
    boltzmann_qc_points,
)
from .coronal_model import CoronaModel
from ._utils import flatdf
from .intensity_io import read_intensities


PLAN_PATH_KEYS = {"input_dir", "output_dir", "fit_report_dir", "manifest"}
PLOT_KINDS = {"all", "boltzmann", "coronal", "none"}
BOLTZMANN_FIGSIZE = (7.4, 5.0)
BOLTZMANN_SUBPLOTS = {"left": 0.14, "right": 0.96, "bottom": 0.15, "top": 0.88}
BOLTZMANN_DEFAULT_Y_LIMITS = (1e-2, 2.0)
CORONAL_FIGSIZE = (8.2, 5.2)
CORONAL_SUBPLOTS = {"left": 0.10, "right": 0.96, "bottom": 0.22, "top": 0.92}


def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    setting = os.environ.get("FULCHER_COLOR", "").lower()
    if setting in {"1", "true", "yes", "on"}:
        return True
    if setting in {"0", "false", "no", "off"}:
        return False
    return sys.stdout.isatty()


USE_COLOR = _color_enabled()


def _c(text: str, code: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(text: str) -> str:
    return _c(text, "1")


def _dim(text: str) -> str:
    return _c(text, "2")


def _green(text: str) -> str:
    return _c(text, "32")


def _cyan(text: str) -> str:
    return _c(text, "36")


def _provided_destinations(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    provided: set[str] = set()
    actions = [action for action in parser._actions if action.dest != argparse.SUPPRESS]
    for token in argv:
        for action in actions:
            for option in action.option_strings:
                if token == option or token.startswith(f"{option}="):
                    provided.add(action.dest)
    return provided


def _plan_path(value: object, plan_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return plan_path.parent / path


def _apply_plan(args: argparse.Namespace, plan_path: Path, provided: set[str]) -> None:
    plan_path = plan_path.resolve()
    if not plan_path.is_file():
        raise SystemExit(f"Plan file not found: {plan_path}")
    with plan_path.open("rb") as fh:
        raw = tomllib.load(fh)

    for section_name in ("common", "analyze"):
        section = raw.get(section_name, {})
        if not section:
            continue
        if not isinstance(section, dict):
            raise SystemExit(f"Plan section [{section_name}] must be a TOML table.")
        for key, value in section.items():
            if key in provided:
                continue
            if not hasattr(args, key):
                if section_name == "analyze":
                    raise SystemExit(f"Unknown plan key [{section_name}].{key}")
                continue
            if key in PLAN_PATH_KEYS and value not in (None, ""):
                value = _plan_path(value, plan_path)
            setattr(args, key, value)


def _frame_key(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def _intensity_records(
    input_dir: Path,
    *,
    manifest: Path | None = None,
) -> list[dict[str, object]]:
    allowed = _manifest_frame_pairs(manifest) if manifest is not None else None
    records = []
    for path in sorted(input_dir.glob("*.csv")):
        stem = path.stem
        if stem.endswith("_err") or stem.endswith("_fit_report"):
            continue
        if "_fr_" not in stem:
            continue
        shot, frame = stem.rsplit("_fr_", maxsplit=1)
        frame = _frame_key(frame)
        if allowed is not None and (shot, frame) not in allowed:
            continue
        records.append({"shot": shot, "frame": frame, "stem": stem, "path": path})
    return records


def _manifest_frame_pairs(path: Path | None) -> set[tuple[str, str]] | None:
    if path is None:
        return None
    selected: set[tuple[str, str]] = set()
    frame_table = pd.read_csv(path)
    if not {"shot", "frame"}.issubset(frame_table.columns):
        return None
    for row in frame_table.itertuples(index=False):
        selected.add((str(getattr(row, "shot")), _frame_key(getattr(row, "frame"))))
    return selected


def _fit_report_path(
    stem: str,
    *,
    input_dir: Path,
    fit_report_dir: Path | None,
) -> Path | None:
    candidates = []
    if fit_report_dir is not None:
        candidates.append(fit_report_dir / f"{stem}_fit_report.csv")
    candidates.append(input_dir / f"{stem}_fit_report.csv")
    for path in candidates:
        if path.is_file():
            return path
    return None


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _row_key(row: dict[str, object]) -> str:
    stem = str(row.get("stem", "")).strip()
    if stem:
        return stem
    return f"{row.get('shot')}_fr_{_frame_key(row.get('frame'))}"


def _summary_maps(output_dir: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    boltzmann = {_row_key(row): row for row in _load_csv_rows(output_dir / "boltzmann_summary.csv")}
    coronal = {_row_key(row): row for row in _load_csv_rows(output_dir / "coronal_summary.csv")}
    return boltzmann, coronal


def _ordered_summary_rows(rows_by_stem: dict[str, dict], records: list[dict[str, object]]) -> list[dict]:
    ordered = []
    seen: set[str] = set()
    for record in records:
        stem = str(record["stem"])
        if stem in rows_by_stem:
            ordered.append(rows_by_stem[stem])
            seen.add(stem)
    ordered.extend(row for stem, row in sorted(rows_by_stem.items()) if stem not in seen)
    return ordered


def _write_summaries(
    output_dir: Path,
    records: list[dict[str, object]],
    boltzmann_by_stem: dict[str, dict],
    coronal_by_stem: dict[str, dict],
) -> None:
    _write_csv(
        output_dir / "boltzmann_summary.csv",
        _ordered_summary_rows(boltzmann_by_stem, records),
    )
    _write_csv(
        output_dir / "coronal_summary.csv",
        _ordered_summary_rows(coronal_by_stem, records),
    )


def _format_temperature(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(numeric):
        return "nan"
    return f"{numeric:.0f}K"


def _progress_label(stem: str, *, trot1: object = None, trot2: object = None, tvib: object = None) -> str:
    return f"{stem} T2={_format_temperature(trot2)}"


def _short_progress_label(label: str, width: int = 26) -> str:
    text = " ".join(str(label).split())
    if len(text) <= width:
        return text
    keep_left = max(8, width // 2 - 1)
    keep_right = max(8, width - keep_left - 3)
    return f"{text[:keep_left]}...{text[-keep_right:]}"


def _should_write_qc(index: int, qc_every: int) -> bool:
    return qc_every > 0 and index % qc_every == 0


def _plot_kinds(args: argparse.Namespace) -> set[str]:
    plot_kind = getattr(args, "plot_kind", "all")
    qc_every = getattr(args, "qc_every", 1)
    if qc_every <= 0 or plot_kind == "none":
        return set()
    if plot_kind == "all":
        return {"boltzmann", "coronal"}
    return {plot_kind}


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _relative_display_path(path: Path, base: Path) -> str:
    resolved_path = path.expanduser().resolve()
    resolved_base = base.expanduser().resolve()
    try:
        return resolved_path.relative_to(resolved_base).as_posix()
    except ValueError:
        return str(resolved_path)


def _print_analysis_summary(
    *,
    title: str,
    input_dir: Path,
    output_dir: Path,
    artifacts: list[tuple[str, str, Path]],
    analyzed_frames: int,
    skipped_frames: int,
    workers: int,
    workdir: Path | None = None,
) -> None:
    workdir = workdir or Path.cwd()
    print()
    print(_bold(f"=== {title} ==="))
    print(f"workdir   : {_dim(str(workdir.resolve()))}")
    print(f"input     : {_cyan(_relative_display_path(input_dir, workdir))}")
    print(f"output    : {_cyan(_relative_display_path(output_dir, workdir))}")
    print("artifacts :")
    for action, label, path in artifacts:
        action_style = _green if action == "WRITE" else _cyan
        print(f"  {action_style(f'{action:<5}')} {label:<13} {_relative_display_path(path, output_dir)}")
    print(f"frames    : {_cyan(analyzed_frames)}")
    if skipped_frames:
        print(f"skipped   : {_cyan(skipped_frames)}")
    print(f"workers   : {_cyan(workers)}")


def _analyze_record(
    index: int,
    record: dict[str, object],
    config: dict[str, object],
) -> dict[str, object]:
    input_dir = Path(str(config["input_dir"]))
    output_dir = Path(str(config["output_dir"]))
    fit_report_dir = (
        Path(str(config["fit_report_dir"]))
        if config.get("fit_report_dir") not in (None, "")
        else None
    )
    tables_dir = output_dir / "tables"
    isotopologue = str(config["isotopologue"])
    max_fit_relerr = float(config.get("max_fit_relerr", 1.0))
    show_model_output = bool(config.get("show_model_output", False))

    shot = str(record["shot"])
    frame = str(record["frame"])
    stem = str(record["stem"])
    model_stdout = io.StringIO()
    stdout_context = (
        contextlib.nullcontext()
        if show_model_output
        else contextlib.redirect_stdout(model_stdout)
    )
    try:
        with stdout_context:
            fit_report = _fit_report_path(
                stem,
                input_dir=input_dir,
                fit_report_dir=fit_report_dir,
            )
            intensities = read_intensities(shot, frame, data_folder=input_dir)
            bp = BoltzmannPlot(intensities, isotopologue)
            points = boltzmann_qc_points(
                bp,
                max_fit_relerr=max_fit_relerr,
                fit_report=fit_report,
            )
            apply_boltzmann_qc_mask(bp, points)
            bp.autofit()
            points = boltzmann_qc_points(
                bp,
                max_fit_relerr=max_fit_relerr,
                fit_report=fit_report,
            )
            qc_points_path = tables_dir / f"{stem}_boltzmann_qc_points.csv"
            points.to_csv(qc_points_path, index=False)
            boltzmann_fit_curve_path = tables_dir / f"{stem}_boltzmann_qc_fit.csv"
            _write_boltzmann_fit_curve(boltzmann_fit_curve_path, bp)
            boltzmann_row = {
                "shot": shot,
                "frame": frame,
                "stem": stem,
                "isotopologue": isotopologue,
                "alpha": bp.alpha,
                "beta": bp.beta,
                "Trot1": bp.trot1,
                "Trot2": bp.trot2,
                "alpha_stderr": bp.err[0],
                "beta_stderr": bp.err[1],
                "Trot1_stderr": bp.err[2],
                "Trot2_stderr": bp.err[3],
                "n_boltzmann_points": int(points["fit_mask"].sum()) if "fit_mask" in points else "",
                "fit_report": str(fit_report or ""),
                "qc_points": str(qc_points_path),
                "qc_fit_curve": str(boltzmann_fit_curve_path),
                "status": "ok",
            }

            cm = CoronaModel(bp)
            cm.coronal_autofit()
            progress_label = _progress_label(
                stem,
                trot1=bp.trot1,
                trot2=bp.trot2,
                tvib=cm.tvib,
            )
            coronal_row = {
                "shot": shot,
                "frame": frame,
                "stem": stem,
                "isotopologue": isotopologue,
                "Tvib": cm.tvib,
                "Tvib_stderr": cm.tviberr,
                "status": "ok",
            }
            coronal_qc_points_path = tables_dir / f"{stem}_coronal_qc_points.csv"
            _write_coronal_qc_points(coronal_qc_points_path, cm)
            coronal_row["qc_points"] = str(coronal_qc_points_path)
            return {
                "stem": stem,
                "status": "ok",
                "progress_label": progress_label,
                "boltzmann_row": boltzmann_row,
                "coronal_row": coronal_row,
            }
    except Exception as exc:
        boltzmann_row = {"shot": shot, "frame": frame, "stem": stem, "status": "failed", "error": repr(exc)}
        if not show_model_output and model_stdout.getvalue():
            boltzmann_row["captured_stdout"] = model_stdout.getvalue()
        boltzmann_row["traceback"] = traceback.format_exc()
        return {
            "stem": stem,
            "status": "failed",
            "error": repr(exc),
            "progress_label": f"{stem} failed",
            "boltzmann_row": boltzmann_row,
            "coronal_row": {"shot": shot, "frame": frame, "stem": stem, "status": "failed", "error": repr(exc)},
        }


def _plot_record(
    index: int,
    record: dict[str, object],
    config: dict[str, object],
) -> dict[str, object]:
    output_dir = Path(str(config["output_dir"]))
    tables_dir = output_dir / "tables"
    boltzmann_plot_dir = output_dir / "plots" / "boltzmann"
    coronal_plot_dir = output_dir / "plots" / "coronal"
    plot_kinds = set(config.get("plot_kinds", ()))
    isotopologue = str(config["isotopologue"])
    molecule_label = "H2" if isotopologue == "h" else "D2"
    qc_every = int(config.get("qc_every", 1))
    shot = str(record["shot"])
    frame = str(record["frame"])
    stem = str(record["stem"])

    try:
        if _should_write_qc(index, qc_every):
            if "boltzmann" in plot_kinds:
                _plot_saved_boltzmann_qc(
                    points_path=tables_dir / f"{stem}_boltzmann_qc_points.csv",
                    fit_curve_path=tables_dir / f"{stem}_boltzmann_qc_fit.csv",
                    output_path=boltzmann_plot_dir / f"{stem}_boltzmann_qc.png",
                    title=f"{molecule_label} {shot} frame {frame}: d-state Boltzmann fit",
                    y_limits=config.get("boltzmann_y_limits"),
                )
            if "coronal" in plot_kinds:
                coronal_points = tables_dir / f"{stem}_coronal_qc_points.csv"
                points = pd.read_csv(coronal_points)
                tvib = float(points["Tvib"].iloc[0]) if "Tvib" in points else float("nan")
                _plot_saved_coronal_qc(
                    points_path=coronal_points,
                    output_path=(
                        coronal_plot_dir
                        / f"{stem}_{molecule_label.lower()}_coronal_tvib_{tvib:.0f}K_qc.png"
                    ),
                    title=f"{molecule_label} {shot} frame {frame}: coronal fit Tvib={tvib:.0f} K",
                    y_limit=config.get("coronal_y_limit"),
                    labels=config.get("coronal_labels"),
                )
        return {
            "stem": stem,
            "status": "ok",
            "progress_label": f"{stem} plotted",
            "boltzmann_row": {"shot": shot, "frame": frame, "stem": stem, "status": "plotted"},
            "coronal_row": {"shot": shot, "frame": frame, "stem": stem, "status": "plotted"},
        }
    except Exception as exc:
        return {
            "stem": stem,
            "status": "failed",
            "error": repr(exc),
            "progress_label": f"{stem} failed",
            "boltzmann_row": {"shot": shot, "frame": frame, "stem": stem, "status": "failed", "error": repr(exc)},
            "coronal_row": {"shot": shot, "frame": frame, "stem": stem, "status": "failed", "error": repr(exc)},
        }


def _analysis_config(
    args: argparse.Namespace,
    *,
    input_dir: Path,
    output_dir: Path,
    fit_report_dir: Path | None,
    plot_kinds: set[str],
) -> dict[str, object]:
    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "fit_report_dir": str(fit_report_dir) if fit_report_dir is not None else "",
        "plot_kinds": sorted(plot_kinds),
        "isotopologue": args.isotopologue,
        "max_fit_relerr": args.max_fit_relerr,
        "qc_every": args.qc_every,
        "show_model_output": args.show_model_output,
    }


def _write_boltzmann_fit_curve(path: Path, bp: BoltzmannPlot) -> None:
    rows = []
    for band_index in bp.nd_bol_synth:
        band = f"{int(band_index)}-{int(band_index)}"
        for row_index, value in bp.nd_bol_synth[band_index].items():
            energy = bp.Ed[band_index].loc[row_index]
            if not np_is_finite(value) or not np_is_finite(energy):
                continue
            rows.append(
                {
                    "band": band,
                    "band_index": int(band_index),
                    "N": int(row_index) + 1,
                    "energy_eV": float(energy),
                    "nd_bol_synth": float(value),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def np_is_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _ceil_to_step(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def _positive_log_floor(values: list[float]) -> float:
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    if not positive:
        return BOLTZMANN_DEFAULT_Y_LIMITS[0]
    minimum = min(positive)
    if minimum >= BOLTZMANN_DEFAULT_Y_LIMITS[0]:
        return BOLTZMANN_DEFAULT_Y_LIMITS[0]
    return 10 ** math.floor(math.log10(minimum * 0.8))


def _boltzmann_axis_limits(points: pd.DataFrame, fit_curve: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x_values = list(points["energy_eV"].astype(float))
    if not fit_curve.empty:
        x_values.extend(fit_curve["energy_eV"].astype(float))
    x_upper = max(0.45, _ceil_to_step(max(x_values) * 1.03, 0.05)) if x_values else 0.45

    point_y = points["nd_rel"].astype(float)
    point_err = points["relerr"].astype(float) * point_y
    error_lower = point_y - point_err
    lower_values = list(
        error_lower.where(
            point_y < BOLTZMANN_DEFAULT_Y_LIMITS[0],
            BOLTZMANN_DEFAULT_Y_LIMITS[0],
        ).clip(lower=0.0)
    )
    upper_values = list(point_y + point_err)
    if not fit_curve.empty:
        curve_y = list(fit_curve["nd_bol_synth"].astype(float))
        lower_values.extend(curve_y)
        upper_values.extend(curve_y)
    y_lower = _positive_log_floor(lower_values)
    y_upper = BOLTZMANN_DEFAULT_Y_LIMITS[1]
    return (0.0, x_upper), (y_lower, y_upper)


def _coronal_y_limit(points: pd.DataFrame) -> float:
    upper_values = list(points["measured_norm"].astype(float) + points["measured_err_norm"].astype(float))
    upper_values.extend(points["model_norm"].astype(float))
    if "model_err_norm" in points:
        upper_values.extend(points["model_norm"].astype(float) + points["model_err_norm"].astype(float))
    upper = max(upper_values) if upper_values else 0.0
    return max(0.16, _ceil_to_step(upper * 1.10, 0.02))


def _shared_boltzmann_y_limits(
    *,
    tables_dir: Path,
    pending: list[tuple[int, dict[str, object]]],
    qc_every: int,
) -> tuple[float, float] | None:
    frame_limits = []
    for index, record in pending:
        if not _should_write_qc(index, qc_every):
            continue
        stem = str(record["stem"])
        points_path = tables_dir / f"{stem}_boltzmann_qc_points.csv"
        fit_curve_path = tables_dir / f"{stem}_boltzmann_qc_fit.csv"
        points = pd.read_csv(points_path)
        fit_curve = pd.read_csv(fit_curve_path)
        _, ylim = _boltzmann_axis_limits(points, fit_curve)
        frame_limits.append(ylim)
    if len(frame_limits) < 2:
        return frame_limits[0] if frame_limits else None

    shared = (min(low for low, _ in frame_limits), max(high for _, high in frame_limits))
    frame_spans = sorted(math.log10(high / low) for low, high in frame_limits if low > 0 and high > low)
    if not frame_spans or shared[0] <= 0 or shared[1] <= shared[0]:
        return None
    median_span = frame_spans[len(frame_spans) // 2]
    shared_span = math.log10(shared[1] / shared[0])
    if shared_span <= median_span + 0.70:
        return shared
    return None


def _shared_coronal_y_limit(
    *,
    tables_dir: Path,
    pending: list[tuple[int, dict[str, object]]],
    qc_every: int,
) -> float | None:
    frame_limits = []
    for index, record in pending:
        if not _should_write_qc(index, qc_every):
            continue
        stem = str(record["stem"])
        points = pd.read_csv(tables_dir / f"{stem}_coronal_qc_points.csv")
        frame_limits.append(_coronal_y_limit(points))
    if len(frame_limits) < 2:
        return frame_limits[0] if frame_limits else None

    shared = max(frame_limits)
    median = sorted(frame_limits)[len(frame_limits) // 2]
    if median > 0 and shared <= median * 2.0:
        return shared
    return None


def _shared_coronal_labels(
    *,
    tables_dir: Path,
    pending: list[tuple[int, dict[str, object]]],
    qc_every: int,
) -> list[str]:
    labels: list[str] = []
    seen = set()
    for index, record in pending:
        if not _should_write_qc(index, qc_every):
            continue
        stem = str(record["stem"])
        points = pd.read_csv(tables_dir / f"{stem}_coronal_qc_points.csv")
        sort_columns = ["index"] if "index" in points else None
        if sort_columns is not None:
            points = points.sort_values(sort_columns)
        for label in points["label"].astype(str):
            if label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def _shared_qc_plot_limits(
    *,
    tables_dir: Path,
    pending: list[tuple[int, dict[str, object]]],
    plot_kinds: set[str],
    qc_every: int,
) -> dict[str, object]:
    limits: dict[str, object] = {}
    if "boltzmann" in plot_kinds:
        boltzmann_limits = _shared_boltzmann_y_limits(
            tables_dir=tables_dir,
            pending=pending,
            qc_every=qc_every,
        )
        if boltzmann_limits is not None:
            limits["boltzmann_y_limits"] = boltzmann_limits
    if "coronal" in plot_kinds:
        coronal_limit = _shared_coronal_y_limit(
            tables_dir=tables_dir,
            pending=pending,
            qc_every=qc_every,
        )
        if coronal_limit is not None:
            limits["coronal_y_limit"] = coronal_limit
        coronal_labels = _shared_coronal_labels(
            tables_dir=tables_dir,
            pending=pending,
            qc_every=qc_every,
        )
        if coronal_labels:
            limits["coronal_labels"] = coronal_labels
    return limits


def _write_coronal_qc_points(path: Path, cm: CoronaModel) -> None:
    exp_values = flatdf(cm.bp.nd[cm.bp.mask])
    exp_errors = flatdf(cm.bp.nd_err[cm.bp.mask])
    model_values = flatdf(cm.nd[cm.bp.mask])
    labels = flatdf(cm.bp.qnames[cm.bp.mask])
    exp_sum = exp_values.sum()
    model_sum = model_values.sum()
    exp_norm = exp_values / exp_sum
    exp_err_norm = exp_errors / exp_sum
    model_norm = model_values / model_sum
    yerr_flat = getattr(cm, "yerr_flat", None)

    rows = []
    for index, (label, measured, measured_err, model) in enumerate(
        zip(labels, exp_norm, exp_err_norm, model_norm)
    ):
        row = {
            "index": index,
            "label": label,
            "measured_norm": float(measured),
            "measured_err_norm": float(measured_err),
            "model_norm": float(model),
            "Tvib": float(cm.tvib),
            "Tvib_stderr": float(cm.tviberr),
        }
        if yerr_flat is not None and len(yerr_flat) == len(model_norm):
            row["model_err_norm"] = float(yerr_flat[index])
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_saved_boltzmann_qc(
    *,
    points_path: Path,
    fit_curve_path: Path,
    output_path: Path,
    title: str,
    y_limits: tuple[float, float] | None = None,
) -> None:
    points = pd.read_csv(points_path)
    fit_curve = pd.read_csv(fit_curve_path)
    fig, ax = plt.subplots(figsize=BOLTZMANN_FIGSIZE)
    first_excluded_label = True
    band_handles: dict[str, dict[str, object]] = {}
    for band, group in points.groupby("band", sort=True):
        style = band_style(band)
        fit_group = group.loc[group["fit_mask"].astype(bool)]
        excluded = group.loc[~group["fit_mask"].astype(bool)]
        if not fit_group.empty:
            band_handles.setdefault(str(band), {})["points"] = ax.errorbar(
                fit_group["energy_eV"],
                fit_group["nd_rel"],
                yerr=fit_group["relerr"] * fit_group["nd_rel"],
                fmt=style.marker,
                ms=5.0,
                capsize=2.5,
                lw=0.8,
                color=style.color,
            )
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
        curve = fit_curve.loc[fit_curve["band"] == band]
        if not curve.empty:
            (line,) = ax.plot(
                curve["energy_eV"],
                curve["nd_bol_synth"],
                color=style.color,
                lw=1.25,
                ls=style.linestyle,
            )
            band_handles.setdefault(str(band), {})["line"] = line
    ax.set_xlabel("Rotational Energy [eV]")
    ax.set_ylabel(r"$\mathrm{\frac{n_{d v' N'}}{(2N'+1)\,g_{\mathrm{as}}^{N'}}}$ [a.u.]")
    ax.set_title(title)
    ax.set_yscale("log")
    xlim, ylim = _boltzmann_axis_limits(points, fit_curve)
    ax.set_xlim(*xlim)
    ax.set_ylim(*(y_limits or ylim))
    ax.grid(True, which="both", color="0.88", lw=0.7)
    handles = []
    labels = []
    for band, parts in band_handles.items():
        if "points" in parts and "line" in parts:
            handles.append((parts["points"], parts["line"]))
        else:
            handles.append(parts.get("points") or parts.get("line"))
        labels.append(band)
    extra_handles, extra_labels = ax.get_legend_handles_labels()
    for handle, label in zip(extra_handles, extra_labels):
        if label == "not fit":
            handles.append(handle)
            labels.append(label)
    ax.legend(
        handles,
        labels,
        title="Band",
        fontsize=7.0,
        title_fontsize=8.0,
        loc="upper right",
        framealpha=0.92,
        labelspacing=0.25,
        handlelength=1.8,
        borderpad=0.35,
        borderaxespad=0.35,
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )
    fig.subplots_adjust(**BOLTZMANN_SUBPLOTS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_saved_coronal_qc(
    *,
    points_path: Path,
    output_path: Path,
    title: str,
    y_limit: float | None = None,
    labels: list[str] | None = None,
) -> None:
    points = pd.read_csv(points_path)
    fig, ax = plt.subplots(figsize=CORONAL_FIGSIZE)
    labels = labels or list(points["label"].astype(str))
    x_positions = {label: index for index, label in enumerate(labels)}
    points = points.assign(_x=points["label"].astype(str).map(x_positions)).dropna(subset=["_x"]).sort_values("_x")
    x = points["_x"].astype(float)
    ax.errorbar(
        x,
        points["measured_norm"],
        yerr=points["measured_err_norm"],
        fmt="o",
        ms=4.5,
        capsize=2.5,
        lw=0.8,
        color="#2f6fce",
        label="measured",
    )
    ax.plot(
        x,
        points["model_norm"],
        "s-",
        ms=4.0,
        lw=1.1,
        color="#d54a2a",
        label="model",
    )
    if "model_err_norm" in points:
        lower = (points["model_norm"] - points["model_err_norm"]).clip(lower=0.0)
        upper = points["model_norm"] + points["model_err_norm"]
        ax.fill_between(x, lower, upper, color="#d54a2a", alpha=0.18, label="+/- 1 sigma")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(0.0, y_limit or _coronal_y_limit(points))
    ax.set_xlabel("Q-branch transition")
    ax.set_ylabel("Normalized d-state population")
    ax.set_title(title)
    ax.grid(True, color="0.88", lw=0.7)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
    fig.subplots_adjust(**CORONAL_SUBPLOTS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _record_done(
    result: dict[str, object],
    *,
    records: list[dict[str, object]],
    output_dir: Path,
    boltzmann_by_stem: dict[str, dict],
    coronal_by_stem: dict[str, dict],
    completed: int,
    checkpoint_every: int,
    write_summaries: bool,
    progress: object | None,
    base_desc: str,
    no_progress: bool,
    progress_every: int,
    total: int,
    start: float,
) -> None:
    stem = str(result["stem"])
    boltzmann_by_stem[stem] = result["boltzmann_row"]
    coronal_by_stem[stem] = result["coronal_row"]
    progress_label = str(result["progress_label"])
    if result.get("status") == "failed":
        message = f"failed {stem}: {result.get('error')}"
        if progress is not None:
            progress.write(message, file=sys.stderr)
        else:
            print(message, file=sys.stderr, flush=True)

    if write_summaries and completed % checkpoint_every == 0:
        _write_summaries(output_dir, records, boltzmann_by_stem, coronal_by_stem)

    if progress is not None:
        progress.update(1)
        progress.set_description_str(
            f"{base_desc} | {_short_progress_label(progress_label)}",
            refresh=False,
        )
    elif not no_progress and (completed == 1 or completed % progress_every == 0 or completed == total):
        elapsed = time.monotonic() - start
        verb = "plot" if base_desc.startswith("plot") else "fit"
        print(f"{verb} {completed}/{total}: {progress_label}, elapsed {elapsed:.1f}s", flush=True)


def analyze_batch(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir else input_dir.parent
    fit_report_dir = args.fit_report_dir.expanduser() if args.fit_report_dir else None
    manifest = args.manifest.expanduser() if args.manifest else None
    plot_only = bool(getattr(args, "plot_only", False))
    if not plot_only and (int(getattr(args, "qc_every", 0)) > 0 or getattr(args, "plot_kind", "none") != "none"):
        raise SystemExit("QC plotting is separated from fitting; use --plot-only for plot generation.")
    plot_kinds = _plot_kinds(args) if plot_only else set()
    checkpoint_every = max(1, int(getattr(args, "checkpoint_every", 1)))
    if fit_report_dir is None and (output_dir / "fit_reports").is_dir():
        fit_report_dir = output_dir / "fit_reports"

    tables_dir = output_dir / "tables"
    boltzmann_plot_dir = output_dir / "plots" / "boltzmann"
    coronal_plot_dir = output_dir / "plots" / "coronal"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if "boltzmann" in plot_kinds:
        boltzmann_plot_dir.mkdir(parents=True, exist_ok=True)
    if "coronal" in plot_kinds:
        coronal_plot_dir.mkdir(parents=True, exist_ok=True)

    records = _intensity_records(input_dir, manifest=manifest)
    if args.max_frames is not None:
        records = records[: args.max_frames]
    if not records:
        raise SystemExit(f"No intensity CSVs found in {input_dir}")

    if getattr(args, "resume", False):
        boltzmann_by_stem, coronal_by_stem = _summary_maps(output_dir)
    else:
        boltzmann_by_stem, coronal_by_stem = {}, {}
    write_summaries = not plot_only
    start = time.monotonic()
    total = len(records)
    skipped = 0
    workers = max(1, int(getattr(args, "workers", 1)))
    base_desc = "plot frames" if plot_only else "fit frames"
    config = _analysis_config(
        args,
        input_dir=input_dir,
        output_dir=output_dir,
        fit_report_dir=fit_report_dir,
        plot_kinds=plot_kinds,
    )
    verb = "plotting" if plot_only else "analyzing"
    if not args.no_progress:
        print(f"{verb} {total} frames with {workers} worker{'s' if workers != 1 else ''}", flush=True)
    progress = None
    if not args.no_progress and tqdm is not None:
        progress = tqdm(
            total=total,
            desc=base_desc,
            unit="fr",
            ncols=96,
            dynamic_ncols=False,
            bar_format="{desc:<34} {percentage:3.0f}%|{bar:10}| {n_fmt:>4}/{total_fmt:<4} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    pending: list[tuple[int, dict[str, object]]] = []
    completed = 0
    for index, record in enumerate(records, start=1):
        stem = str(record["stem"])
        if getattr(args, "resume", False) and not plot_only:
            existing_boltzmann = boltzmann_by_stem.get(stem, {})
            existing_coronal = coronal_by_stem.get(stem, {})
            if (
                existing_boltzmann.get("status") == "ok"
                and existing_coronal.get("status") == "ok"
            ):
                skipped += 1
                completed += 1
                progress_label = f"{stem} skipped"
                if progress is not None:
                    progress.update(1)
                    progress.set_description_str(
                        f"{base_desc} | {_short_progress_label(progress_label)}",
                        refresh=False,
                    )
                elif not args.no_progress and (completed == 1 or completed % args.progress_every == 0 or completed == total):
                    elapsed = time.monotonic() - start
                    print(f"fit {completed}/{total}: {progress_label}, elapsed {elapsed:.1f}s", flush=True)
                continue
        pending.append((index, record))

    if plot_only:
        config.update(
            _shared_qc_plot_limits(
                tables_dir=tables_dir,
                pending=pending,
                plot_kinds=plot_kinds,
                qc_every=args.qc_every,
            )
        )

    if workers == 1 or len(pending) <= 1:
        for index, record in pending:
            worker_fn = _plot_record if plot_only else _analyze_record
            result = worker_fn(index, record, config)
            completed += 1
            _record_done(
                result,
                records=records,
                output_dir=output_dir,
                boltzmann_by_stem=boltzmann_by_stem,
                coronal_by_stem=coronal_by_stem,
                completed=completed,
                checkpoint_every=checkpoint_every,
                write_summaries=write_summaries,
                progress=progress,
                base_desc=base_desc,
                no_progress=args.no_progress,
                progress_every=args.progress_every,
                total=total,
                start=start,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            worker_fn = _plot_record if plot_only else _analyze_record
            futures = [
                executor.submit(worker_fn, index, record, config)
                for index, record in pending
            ]
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                _record_done(
                    result,
                    records=records,
                    output_dir=output_dir,
                    boltzmann_by_stem=boltzmann_by_stem,
                    coronal_by_stem=coronal_by_stem,
                    completed=completed,
                    checkpoint_every=checkpoint_every,
                    write_summaries=write_summaries,
                    progress=progress,
                    base_desc=base_desc,
                    no_progress=args.no_progress,
                    progress_every=args.progress_every,
                    total=total,
                    start=start,
                )
    if progress is not None:
        progress.close()

    if write_summaries:
        _write_summaries(output_dir, records, boltzmann_by_stem, coronal_by_stem)
    molecule_label = "H2" if args.isotopologue == "h" else "D2"
    artifacts: list[tuple[str, str, Path]] = []
    if write_summaries:
        artifacts.extend(
            [
                ("WRITE", "Boltzmann", output_dir / "boltzmann_summary.csv"),
                ("WRITE", "coronal", output_dir / "coronal_summary.csv"),
            ]
        )
    artifacts.append((("READ" if plot_only else "WRITE"), "QC tables", tables_dir))
    if "boltzmann" in plot_kinds:
        artifacts.append(("WRITE", "Boltzmann QC", boltzmann_plot_dir))
    if "coronal" in plot_kinds:
        artifacts.append(("WRITE", "coronal QC", coronal_plot_dir))
    _print_analysis_summary(
        title=f"{molecule_label} Fulcher {'QC plotting' if plot_only else 'analysis'}",
        input_dir=input_dir,
        output_dir=output_dir,
        artifacts=artifacts,
        analyzed_frames=total,
        skipped_frames=skipped,
        workers=workers,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=None, help="TOML run plan with an [analyze] section.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory containing <shot>_fr_<frame>.csv and *_err.csv intensity tables.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for summary outputs. Defaults to the parent of --input-dir.")
    parser.add_argument("--fit-report-dir", type=Path, default=None, help="Optional directory containing <shot>_fr_<frame>_fit_report.csv files.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional selected_frames.csv used to filter frames.")
    parser.add_argument("--isotopologue", default="h", choices=["h", "d"], help="Molecule constants to use: h for H2 or d for D2.")
    parser.add_argument("--max-fit-relerr", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--qc-every", type=int, default=0, help="Write QC plots every N frames; 0 disables plots.")
    parser.add_argument("--plot-kind", choices=sorted(PLOT_KINDS), default="none", help="QC plot stage to write.")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate analyzer QC plots without rewriting summary CSVs.")
    parser.add_argument("--resume", action="store_true", help="Skip frames already marked ok in existing summary CSVs.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Rewrite summary CSVs every N processed frames.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes for frame analysis.")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--no-progress", action="store_true", help="Disable progress output.")
    parser.add_argument("--show-model-output", action="store_true", help="Show verbose per-frame model output.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    provided = _provided_destinations(parser, argv)
    args = parser.parse_args(argv)
    if args.plan:
        _apply_plan(args, args.plan, provided)
    if args.plot_only:
        if "plot_kind" not in provided:
            args.plot_kind = "all"
        if "qc_every" not in provided:
            args.qc_every = 1
    elif args.qc_every > 0 or args.plot_kind != "none":
        parser.error("QC plotting is separated from fitting; use --plot-only for plot generation.")
    if args.input_dir is None:
        parser.error("--input-dir is required unless supplied by --plan [analyze].input_dir")
    analyze_batch(args)


if __name__ == "__main__":
    main()
