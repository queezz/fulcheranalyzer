"""Batch Boltzmann/coronal analysis for saved Fulcher intensity tables."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import sys
import time
import traceback
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal installs.
    tqdm = None

from .boltzmann import BoltzmannPlot
from .boltzmann_qc import (
    apply_boltzmann_qc_mask,
    boltzmann_qc_points,
    plot_boltzmann_qc,
)
from .coronal_model import CoronaModel
from .coronal_qc import plot_coronal_qc
from .intensity_io import read_intensities


PLAN_PATH_KEYS = {"input_dir", "output_dir", "fit_report_dir", "manifest"}
PLOT_KINDS = {"all", "boltzmann", "coronal", "none"}


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
    return (
        f"{stem} "
        f"T1={_format_temperature(trot1)} "
        f"T2={_format_temperature(trot2)} "
        f"Tv={_format_temperature(tvib)}"
    )


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


def analyze_batch(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir else input_dir.parent
    fit_report_dir = args.fit_report_dir.expanduser() if args.fit_report_dir else None
    manifest = args.manifest.expanduser() if args.manifest else None
    plot_kinds = _plot_kinds(args)
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
    start = time.monotonic()
    total = len(records)
    skipped = 0
    molecule_label = "H2" if args.isotopologue == "h" else "D2"
    progress = None
    if not args.no_progress and tqdm is not None:
        progress = tqdm(records, total=total, desc="fit frames", unit="frame", dynamic_ncols=True)
        record_iterable = progress
    else:
        record_iterable = records
    for index, record in enumerate(record_iterable, start=1):
        shot = str(record["shot"])
        frame = str(record["frame"])
        stem = str(record["stem"])
        progress_label = stem
        if getattr(args, "resume", False):
            existing_boltzmann = boltzmann_by_stem.get(stem, {})
            existing_coronal = coronal_by_stem.get(stem, {})
            if (
                existing_boltzmann.get("status") == "ok"
                and existing_coronal.get("status") == "ok"
            ):
                skipped += 1
                progress_label = f"{stem} skipped"
                if progress is not None:
                    progress.set_postfix_str(progress_label, refresh=False)
                elif not args.no_progress and (index == 1 or index % args.progress_every == 0 or index == total):
                    elapsed = time.monotonic() - start
                    print(f"fit {index}/{total}: {progress_label}, elapsed {elapsed:.1f}s", flush=True)
                continue
        model_stdout = io.StringIO()
        stdout_context = (
            contextlib.nullcontext()
            if args.show_model_output
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
                bp = BoltzmannPlot(intensities, args.isotopologue)
                points = boltzmann_qc_points(
                    bp,
                    max_fit_relerr=args.max_fit_relerr,
                    fit_report=fit_report,
                )
                apply_boltzmann_qc_mask(bp, points)
                bp.autofit()
                points = boltzmann_qc_points(
                    bp,
                    max_fit_relerr=args.max_fit_relerr,
                    fit_report=fit_report,
                )
                points.to_csv(tables_dir / f"{stem}_boltzmann_qc_points.csv", index=False)
                boltzmann_row = {
                    "shot": shot,
                    "frame": frame,
                    "stem": stem,
                    "isotopologue": args.isotopologue,
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
                    "qc_points": str(tables_dir / f"{stem}_boltzmann_qc_points.csv"),
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
                    "isotopologue": args.isotopologue,
                    "Tvib": cm.tvib,
                    "Tvib_stderr": cm.tviberr,
                    "status": "ok",
                }

                if _should_write_qc(index, args.qc_every):
                    if "boltzmann" in plot_kinds:
                        boltzmann_qc_path = boltzmann_plot_dir / f"{stem}_boltzmann_qc.png"
                        fig = plot_boltzmann_qc(
                            bp,
                            points,
                            title=f"{molecule_label} {shot} frame {frame}: d-state Boltzmann fit",
                        )
                        fig.savefig(boltzmann_qc_path, dpi=180)
                        plt.close(fig)
                        boltzmann_row["boltzmann_qc_plot"] = str(boltzmann_qc_path)
                    if "coronal" in plot_kinds:
                        coronal_qc_path = (
                            coronal_plot_dir
                            / f"{stem}_{molecule_label.lower()}_coronal_tvib_{cm.tvib:.0f}K_qc.png"
                        )
                        fig = plot_coronal_qc(
                            cm,
                            title=(
                                f"{molecule_label} {shot} frame {frame}: "
                                f"coronal fit Tvib={cm.tvib:.0f} K"
                            ),
                        )
                        fig.savefig(coronal_qc_path, dpi=180)
                        plt.close(fig)
                        coronal_row["coronal_qc_plot"] = str(coronal_qc_path)
                boltzmann_by_stem[stem] = boltzmann_row
                coronal_by_stem[stem] = coronal_row
        except Exception as exc:
            boltzmann_row = {"shot": shot, "frame": frame, "stem": stem, "status": "failed", "error": repr(exc)}
            if not args.show_model_output and model_stdout.getvalue():
                boltzmann_row["captured_stdout"] = model_stdout.getvalue()
            boltzmann_row["traceback"] = traceback.format_exc()
            boltzmann_by_stem[stem] = boltzmann_row
            coronal_by_stem[stem] = {"shot": shot, "frame": frame, "stem": stem, "status": "failed", "error": repr(exc)}
            progress_label = f"{stem} failed"
            if progress is not None:
                progress.write(f"failed {stem}: {exc!r}", file=sys.stderr)
            else:
                print(f"failed {stem}: {exc!r}", file=sys.stderr, flush=True)

        if index % checkpoint_every == 0:
            _write_summaries(output_dir, records, boltzmann_by_stem, coronal_by_stem)

        if progress is not None:
            progress.set_postfix_str(progress_label, refresh=False)
        elif not args.no_progress and (index == 1 or index % args.progress_every == 0 or index == total):
            elapsed = time.monotonic() - start
            print(f"fit {index}/{total}: {progress_label}, elapsed {elapsed:.1f}s", flush=True)
    if progress is not None:
        progress.close()

    _write_summaries(output_dir, records, boltzmann_by_stem, coronal_by_stem)
    print(f"analyzed frames: {total}")
    if skipped:
        print(f"skipped completed frames: {skipped}")
    print(f"output: {_display_path(output_dir)}")


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
    parser.add_argument("--qc-every", type=int, default=1, help="Write QC plots every N frames; 0 disables plots.")
    parser.add_argument("--plot-kind", choices=sorted(PLOT_KINDS), default="all", help="QC plot stage to write.")
    parser.add_argument("--resume", action="store_true", help="Skip frames already marked ok in existing summary CSVs.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Rewrite summary CSVs every N processed frames.")
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
    if args.input_dir is None:
        parser.error("--input-dir is required unless supplied by --plan [analyze].input_dir")
    analyze_batch(args)


if __name__ == "__main__":
    main()
