"""
Smoke test — Phase 2a / 2i.

Verifies that:
1. The package imports cleanly from the src-layout.
2. Data files are found at their expected locations.
3. Both canonical datasets load and produce DataFrames of the right shape.
4. BoltzmannPlot initialises for both D2 and H2 (instantiates MolecularConstants,
   loads molecular data, runs the Boltzmann calculation — no fit yet).
5. The canonical public API is complete and consistent.

Run with:
    pip install -e .
    pytest tests/test_smoke.py
or:
    python tests/test_smoke.py
"""

import pytest


def test_import():
    import fulcher_analyzer as fa
    assert hasattr(fa, "BoltzmannPlot")
    assert hasattr(fa, "CoronaModel")
    assert hasattr(fa, "MolecularConstants")
    assert hasattr(fa, "read_intensities")


def test_public_api():
    """Top-level package exposes the canonical public names."""
    import fulcher_analyzer as fa

    assert hasattr(fa, "BoltzmannPlot")
    assert hasattr(fa, "boltzmann_qc_points")
    assert hasattr(fa, "plot_boltzmann_qc")
    assert hasattr(fa, "apply_boltzmann_qc_mask")
    assert hasattr(fa, "CoronaModel")
    assert hasattr(fa, "plot_coronal_qc")
    assert hasattr(fa, "MolecularConstants")
    assert hasattr(fa, "read_intensities")
    assert hasattr(fa, "write_intensities")


def test_data_folders_exist():
    from fulcher_analyzer.coronal_model import MOLECULAR_DATA_FOLDER
    from fulcher_analyzer.intensity_io import INTENSITY_DATA

    assert MOLECULAR_DATA_FOLDER.is_dir(), (
        f"MOLECULAR_DATA_FOLDER not found: {MOLECULAR_DATA_FOLDER}"
    )
    assert INTENSITY_DATA.is_dir(), (
        f"INTENSITY_DATA not found: {INTENSITY_DATA}"
    )


def test_molecular_data_files():
    from fulcher_analyzer.coronal_model import MOLECULAR_DATA_FOLDER

    required = [
        "franck_condon_factor.txt",
        "franck_condon_factor_D2.txt",
        "vibrational_energy.txt",
        "vibrational_energy_D2.txt",
        "excitation_vibrational_energy.txt",
        "excitation_vibrational_energy_D2.txt",
        "fulcher-\u03b1_band_wavelength.txt",
        "fulcher-\u03b1_band_wavenumber_D2.txt",
        "spectroscopic_constants.csv",
    ]
    for fname in required:
        resource = MOLECULAR_DATA_FOLDER.joinpath(fname)
        assert resource.is_file(), f"Missing molecular data file: {fname}"


def test_spectroscopic_constants_values():
    """Selected constants must match NIST/Ishihara source values exactly."""
    from fulcher_analyzer import MolecularConstants

    mc = MolecularConstants()

    # H2 d3 state
    assert mc.h2.loc["d3", "we"] == pytest.approx(2371.57)
    assert mc.h2.loc["d3", "Be"] == pytest.approx(30.364)
    assert mc.h2.loc["X", "we"] == pytest.approx(4401.21)

    # D2 d3 state
    assert mc.d2.loc["d3", "we"] == pytest.approx(1678.22)
    assert mc.d2.loc["d3", "Be"] == pytest.approx(15.200)
    assert mc.d2.loc["X", "De"] == pytest.approx(0.01141)

    # DataFrame index order must be preserved
    assert list(mc.h2.index) == ["d3", "a3", "X"]
    assert list(mc.d2.index) == ["d3", "a3", "X"]


def test_read_intensities_d2():
    from fulcher_analyzer import read_intensities

    inte, interr = read_intensities(150482, 7)
    assert inte.shape == (14, 4), f"Unexpected D2 intensity shape: {inte.shape}"
    assert interr.shape == (14, 4), f"Unexpected D2 error shape: {interr.shape}"


def test_read_intensities_h2():
    from fulcher_analyzer import read_intensities

    inte, interr = read_intensities(152478, 10)
    assert inte.shape == (11, 3), f"Unexpected H2 intensity shape: {inte.shape}"
    assert interr.shape == (11, 3), f"Unexpected H2 error shape: {interr.shape}"


def test_boltzmann_init_d2():
    from fulcher_analyzer import BoltzmannPlot, read_intensities

    inte = read_intensities(150482, 7)
    bp = BoltzmannPlot(inte, "d")
    assert bp.isotop == "d"
    assert bp.nd.shape == (14, 4)
    assert bp.nd_rel.shape == (14, 4)


def test_boltzmann_init_h2():
    from fulcher_analyzer import BoltzmannPlot, read_intensities

    inte = read_intensities(152478, 10)
    bp = BoltzmannPlot(inte, "h")
    assert bp.isotop == "h"
    assert bp.nd.shape == (11, 3)
    assert bp.nd_rel.shape == (11, 3)


def test_boltzmann_qc_plotter_renders(tmp_path):
    import matplotlib.pyplot as plt

    from fulcher_analyzer import (
        BoltzmannPlot,
        apply_boltzmann_qc_mask,
        boltzmann_qc_points,
        read_intensities,
    )

    inte = read_intensities(152478, 10)
    bp = BoltzmannPlot(inte, "h")
    points = boltzmann_qc_points(bp)

    assert {"band", "N", "energy_eV", "nd_rel", "relerr", "fit_mask"}.issubset(
        points.columns
    )
    assert not points.empty

    apply_boltzmann_qc_mask(bp, points)
    bp.autofit()
    fig = bp.plot_qc(points, title="Synthetic QC")
    output = tmp_path / "boltzmann_qc.png"
    fig.savefig(output)

    assert output.exists()
    assert output.stat().st_size > 0
    assert fig.axes[0].get_yscale() == "log"
    plt.close(fig)


def test_boltzmann_qc_points_follow_fit_report_exclusion():
    import pandas as pd

    from fulcher_analyzer import BoltzmannPlot, boltzmann_qc_points, read_intensities

    inte = read_intensities(152478, 10)
    bp = BoltzmannPlot(inte, "h")
    fit_report = pd.DataFrame(
        [
            {
                "line_id": "H2_Q3_1-1",
                "N": 3,
                "band": "1-1",
                "boltzmann_fit_action": "exclude",
                "boltzmann_fit_reason": "Peak too wide; suspected contamination.",
            }
        ]
    )

    points = boltzmann_qc_points(bp, fit_report=fit_report)
    excluded = points.loc[(points["band"] == "1-1") & (points["N"] == 3)].iloc[0]

    assert excluded["relerr"] <= 1.0
    assert bool(excluded["fit_mask"]) is False
    assert excluded["boltzmann_fit_action"] == "exclude"
    assert "wide" in excluded["boltzmann_fit_reason"]


def test_batch_cli_discovers_intensity_records(tmp_path):
    from fulcher_analyzer.batch_cli import _intensity_records

    (tmp_path / "193809_fr_9.csv").write_text("1,2,3\n", encoding="utf-8")
    (tmp_path / "193809_fr_9_err.csv").write_text("0.1,0.2,0.3\n", encoding="utf-8")
    (tmp_path / "193809_fr_9_fit_report.csv").write_text("line_id\nH2_Q1_0-0\n", encoding="utf-8")
    (tmp_path / "notes.csv").write_text("not,a,frame\n", encoding="utf-8")

    records = _intensity_records(tmp_path)

    assert records == [
        {
            "shot": "193809",
            "frame": "9",
            "stem": "193809_fr_9",
            "path": tmp_path / "193809_fr_9.csv",
        }
    ]


def test_batch_cli_qc_schedule_defaults_to_every_frame():
    from argparse import Namespace

    from fulcher_analyzer.batch_cli import _plot_kinds, _should_write_qc

    assert _should_write_qc(1, 1)
    assert _should_write_qc(10, 5)
    assert not _should_write_qc(9, 5)
    assert not _should_write_qc(1, 0)
    assert _plot_kinds(Namespace(qc_every=1, plot_kind="all")) == {"boltzmann", "coronal"}
    assert _plot_kinds(Namespace(qc_every=1, plot_kind="boltzmann")) == {"boltzmann"}
    assert _plot_kinds(Namespace(qc_every=1, plot_kind="none")) == set()
    assert _plot_kinds(Namespace(qc_every=0, plot_kind="all")) == set()


def test_batch_cli_plan_applies_analyze_section(tmp_path):
    from fulcher_analyzer.batch_cli import _apply_plan, _build_parser, _provided_destinations

    plan = tmp_path / "h2_dataset_plan.toml"
    plan.write_text(
        "\n".join(
            [
                "[common]",
                'cube_glob = "ignored/*.nc"',
                "progress_every = 7",
                "",
                "[analyze]",
                'input_dir = "dataset/intensities"',
                'output_dir = "~/Fulcher-runs/demo/dataset"',
                'fit_report_dir = "dataset/fit_reports"',
                'manifest = "scan/selected_frames.csv"',
                'isotopologue = "h"',
                "qc_every = 2",
                "max_fit_relerr = 0.5",
            ]
        ),
        encoding="utf-8",
    )

    parser = _build_parser()
    argv = ["--plan", str(plan), "--qc-every", "0"]
    args = parser.parse_args(argv)
    _apply_plan(args, plan, _provided_destinations(parser, argv))

    assert args.input_dir == tmp_path / "dataset" / "intensities"
    assert args.fit_report_dir == tmp_path / "dataset" / "fit_reports"
    assert args.manifest == tmp_path / "scan" / "selected_frames.csv"
    assert str(args.output_dir).endswith("Fulcher-runs\\demo\\dataset") or str(args.output_dir).endswith("Fulcher-runs/demo/dataset")
    assert args.progress_every == 7
    assert args.qc_every == 0
    assert args.max_fit_relerr == 0.5


def test_batch_cli_writes_coronal_qc_and_closes_figures(tmp_path):
    import shutil
    from argparse import Namespace

    import matplotlib.pyplot as plt

    from fulcher_analyzer.batch_cli import analyze_batch
    from fulcher_analyzer.intensity_io import INTENSITY_DATA

    input_dir = tmp_path / "dataset" / "intensities"
    output_dir = tmp_path / "dataset"
    input_dir.mkdir(parents=True)
    for name in ("152478_fr_10.csv", "152478_fr_10_err.csv"):
        shutil.copyfile(INTENSITY_DATA.joinpath(name), input_dir / name)

    before = set(plt.get_fignums())
    analyze_batch(
        Namespace(
            input_dir=input_dir,
            output_dir=output_dir,
            fit_report_dir=None,
            manifest=None,
            isotopologue="h",
            max_fit_relerr=1.0,
            max_frames=None,
            qc_every=1,
            plot_kind="all",
            resume=False,
            checkpoint_every=1,
            progress_every=10,
            no_progress=True,
            show_model_output=False,
        )
    )
    after = set(plt.get_fignums())

    coronal_plots = list((output_dir / "plots" / "coronal").glob("*_coronal_tvib_*K_qc.png"))
    assert len(coronal_plots) == 1
    assert coronal_plots[0].name.startswith("152478_fr_10_h2_coronal_tvib_")
    assert coronal_plots[0].stat().st_size > 0
    assert after == before


def test_batch_cli_plot_kind_limits_qc_outputs(tmp_path):
    import shutil
    from argparse import Namespace

    import pandas as pd

    from fulcher_analyzer.batch_cli import analyze_batch
    from fulcher_analyzer.intensity_io import INTENSITY_DATA

    input_dir = tmp_path / "dataset" / "intensities"
    output_dir = tmp_path / "dataset"
    input_dir.mkdir(parents=True)
    for name in ("152478_fr_10.csv", "152478_fr_10_err.csv"):
        shutil.copyfile(INTENSITY_DATA.joinpath(name), input_dir / name)

    analyze_batch(
        Namespace(
            input_dir=input_dir,
            output_dir=output_dir,
            fit_report_dir=None,
            manifest=None,
            isotopologue="h",
            max_fit_relerr=1.0,
            max_frames=None,
            qc_every=1,
            plot_kind="boltzmann",
            resume=False,
            checkpoint_every=1,
            progress_every=10,
            no_progress=True,
            show_model_output=False,
        )
    )

    assert list((output_dir / "plots" / "boltzmann").glob("*_boltzmann_qc.png"))
    assert not (output_dir / "plots" / "coronal").exists()
    boltzmann_summary = pd.read_csv(output_dir / "boltzmann_summary.csv")
    coronal_summary = pd.read_csv(output_dir / "coronal_summary.csv")
    assert "boltzmann_qc_plot" in boltzmann_summary.columns
    assert "coronal_qc_plot" not in coronal_summary.columns


def test_batch_cli_resume_skips_completed_summary_rows(tmp_path):
    from argparse import Namespace

    import pandas as pd

    from fulcher_analyzer.batch_cli import analyze_batch

    input_dir = tmp_path / "dataset" / "intensities"
    output_dir = tmp_path / "dataset"
    input_dir.mkdir(parents=True)
    (input_dir / "193809_fr_9.csv").write_text("not,a,real,intensity,table\n", encoding="utf-8")
    (input_dir / "193809_fr_9_err.csv").write_text("not,a,real,error,table\n", encoding="utf-8")
    (output_dir / "boltzmann_summary.csv").write_text(
        "shot,frame,stem,status\n193809,9,193809_fr_9,ok\n",
        encoding="utf-8",
    )
    (output_dir / "coronal_summary.csv").write_text(
        "shot,frame,stem,status\n193809,9,193809_fr_9,ok\n",
        encoding="utf-8",
    )

    analyze_batch(
        Namespace(
            input_dir=input_dir,
            output_dir=output_dir,
            fit_report_dir=None,
            manifest=None,
            isotopologue="h",
            max_fit_relerr=1.0,
            max_frames=None,
            qc_every=1,
            plot_kind="all",
            resume=True,
            checkpoint_every=1,
            progress_every=10,
            no_progress=True,
            show_model_output=False,
        )
    )

    summary = pd.read_csv(output_dir / "boltzmann_summary.csv")
    assert summary.to_dict(orient="records") == [
        {"frame": 9, "shot": 193809, "status": "ok", "stem": "193809_fr_9"}
    ]


if __name__ == "__main__":
    import sys

    tests = [
        test_import,
        test_public_api,
        test_data_folders_exist,
        test_molecular_data_files,
        test_spectroscopic_constants_values,
        test_read_intensities_d2,
        test_read_intensities_h2,
        test_boltzmann_init_d2,
        test_boltzmann_init_h2,
        test_boltzmann_qc_plotter_renders,
        test_boltzmann_qc_points_follow_fit_report_exclusion,
        test_batch_cli_discovers_intensity_records,
        test_batch_cli_qc_schedule_defaults_to_every_frame,
        test_batch_cli_plan_applies_analyze_section,
        test_batch_cli_writes_coronal_qc_and_closes_figures,
        test_batch_cli_plot_kind_limits_qc_outputs,
        test_batch_cli_resume_skips_completed_summary_rows,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
    sys.exit(failed)
