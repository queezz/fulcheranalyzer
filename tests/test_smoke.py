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

    from fulcher_analyzer.batch_cli import _build_parser, _plot_kinds, _should_write_qc

    assert _should_write_qc(1, 1)
    assert _should_write_qc(10, 5)
    assert not _should_write_qc(9, 5)
    assert not _should_write_qc(1, 0)
    assert _plot_kinds(Namespace(qc_every=1, plot_kind="all")) == {"boltzmann", "coronal"}
    assert _plot_kinds(Namespace(qc_every=1, plot_kind="boltzmann")) == {"boltzmann"}
    assert _plot_kinds(Namespace(qc_every=1, plot_kind="none")) == set()
    assert _plot_kinds(Namespace(qc_every=0, plot_kind="all")) == set()
    args = _build_parser().parse_args([])
    assert args.qc_every == 0
    assert args.plot_kind == "none"


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
                "workers = 3",
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
    assert args.workers == 3
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
            qc_every=0,
            plot_kind="none",
            plot_only=False,
            resume=False,
            workers=1,
            checkpoint_every=1,
            progress_every=10,
            no_progress=True,
            show_model_output=False,
        )
    )
    after = set(plt.get_fignums())
    assert not (output_dir / "plots").exists()
    assert (output_dir / "tables" / "152478_fr_10_coronal_qc_points.csv").exists()
    assert after == before

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
            plot_kind="coronal",
            plot_only=True,
            resume=False,
            workers=1,
            checkpoint_every=1,
            progress_every=10,
            no_progress=True,
            show_model_output=False,
        )
    )
    final = set(plt.get_fignums())

    coronal_plots = list((output_dir / "plots" / "coronal").glob("*_coronal_tvib_*K_qc.png"))
    assert len(coronal_plots) == 1
    assert coronal_plots[0].name.startswith("152478_fr_10_h2_coronal_tvib_")
    assert coronal_plots[0].stat().st_size > 0
    assert final == before


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
            qc_every=0,
            plot_kind="none",
            plot_only=False,
            resume=False,
            checkpoint_every=1,
            workers=1,
            progress_every=10,
            no_progress=True,
            show_model_output=False,
        )
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
            plot_kind="boltzmann",
            plot_only=True,
            resume=False,
            checkpoint_every=1,
            workers=1,
            progress_every=10,
            no_progress=True,
            show_model_output=False,
        )
    )

    assert list((output_dir / "plots" / "boltzmann").glob("*_boltzmann_qc.png"))
    assert not (output_dir / "plots" / "coronal").exists()
    boltzmann_summary = pd.read_csv(output_dir / "boltzmann_summary.csv")
    coronal_summary = pd.read_csv(output_dir / "coronal_summary.csv")
    assert "boltzmann_qc_plot" not in boltzmann_summary.columns
    assert "coronal_qc_plot" not in coronal_summary.columns


def test_saved_qc_plotters_use_stable_geometry(tmp_path, monkeypatch):
    import pandas as pd
    from matplotlib.figure import Figure

    from fulcher_analyzer.batch_cli import (
        _plot_saved_boltzmann_qc,
        _plot_saved_coronal_qc,
    )

    captured = []

    def capture_savefig(self, path, *args, **kwargs):
        ax = self.axes[0]
        captured.append(
            {
                "path": str(path),
                "position": tuple(round(value, 6) for value in ax.get_position().bounds),
                "xlim": tuple(round(value, 6) for value in ax.get_xlim()),
                "ylim": tuple(round(value, 6) for value in ax.get_ylim()),
            }
        )

    monkeypatch.setattr(Figure, "savefig", capture_savefig)

    def write_boltzmann(frame, scale):
        points = pd.DataFrame(
            {
                "band": ["0-0", "0-0", "1-1", "1-1", "2-2", "2-2"],
                "fit_mask": [True, True, True, True, True, True],
                "energy_eV": [0.0, 0.42, 0.02, 0.39, 0.01, 0.36],
                "nd_rel": [1.0 * scale, 0.07 * scale, 0.8 * scale, 0.05 * scale, 0.9 * scale, 0.04 * scale],
                "relerr": [0.05] * 6,
            }
        )
        curve = pd.DataFrame(
            {
                "band": ["0-0", "0-0", "1-1", "1-1", "2-2", "2-2"],
                "energy_eV": [0.0, 0.43, 0.02, 0.41, 0.01, 0.40],
                "nd_bol_synth": [0.95 * scale, 0.06 * scale, 0.75 * scale, 0.045 * scale, 0.85 * scale, 0.035 * scale],
            }
        )
        points_path = tmp_path / f"boltzmann_{frame}_points.csv"
        curve_path = tmp_path / f"boltzmann_{frame}_fit.csv"
        points.to_csv(points_path, index=False)
        curve.to_csv(curve_path, index=False)
        _plot_saved_boltzmann_qc(
            points_path=points_path,
            fit_curve_path=curve_path,
            output_path=tmp_path / f"boltzmann_{frame}.png",
            title=f"H2 frame {frame}: d-state Boltzmann fit",
        )

    def write_coronal(frame, scale):
        points = pd.DataFrame(
            {
                "index": [0, 1, 2, 3],
                "label": ["Q1(0-0)", "Q2(0-0)", "Q1(1-1)", "Q2(1-1)"],
                "measured_norm": [0.08 * scale, 0.04 * scale, 0.13 * scale, 0.06 * scale],
                "measured_err_norm": [0.004] * 4,
                "model_norm": [0.075 * scale, 0.045 * scale, 0.12 * scale, 0.065 * scale],
                "model_err_norm": [0.006] * 4,
                "Tvib": [6500 * scale] * 4,
                "Tvib_stderr": [200] * 4,
            }
        )
        points_path = tmp_path / f"coronal_{frame}_points.csv"
        points.to_csv(points_path, index=False)
        _plot_saved_coronal_qc(
            points_path=points_path,
            output_path=tmp_path / f"coronal_{frame}.png",
            title=f"H2 frame {frame}: coronal fit Tvib={6500 * scale:.0f} K",
        )

    write_boltzmann(10, 1.0)
    write_boltzmann(11, 0.9)
    write_coronal(10, 1.0)
    write_coronal(11, 0.9)

    assert captured[0]["position"] == captured[1]["position"]
    assert captured[0]["xlim"] == captured[1]["xlim"]
    assert captured[0]["ylim"] == captured[1]["ylim"]
    assert captured[2]["position"] == captured[3]["position"]
    assert captured[2]["xlim"] == captured[3]["xlim"]
    assert captured[2]["ylim"] == captured[3]["ylim"]


def test_plot_only_preflight_uses_shared_y_limits(tmp_path):
    import pandas as pd

    from fulcher_analyzer.batch_cli import _shared_qc_plot_limits

    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    pending = [
        (1, {"stem": "152478_fr_10"}),
        (2, {"stem": "152478_fr_11"}),
    ]

    for stem, scale in [("152478_fr_10", 1.0), ("152478_fr_11", 0.9)]:
        pd.DataFrame(
            {
                "band": ["0-0", "0-0", "1-1", "1-1"],
                "fit_mask": [True, True, True, True],
                "energy_eV": [0.0, 0.42, 0.02, 0.39],
                "nd_rel": [1.0 * scale, 0.07 * scale, 0.8 * scale, 0.05 * scale],
                "relerr": [0.05] * 4,
            }
        ).to_csv(tables_dir / f"{stem}_boltzmann_qc_points.csv", index=False)
        pd.DataFrame(
            {
                "band": ["0-0", "0-0", "1-1", "1-1"],
                "energy_eV": [0.0, 0.43, 0.02, 0.41],
                "nd_bol_synth": [0.95 * scale, 0.06 * scale, 0.75 * scale, 0.045 * scale],
            }
        ).to_csv(tables_dir / f"{stem}_boltzmann_qc_fit.csv", index=False)
        labels = (
            ["Q1(0-0)", "Q2(0-0)", "Q1(1-1)", "Q4(1-1)"]
            if stem.endswith("_10")
            else ["Q1(0-0)", "Q2(0-0)", "Q1(1-1)", "Q6(1-1)"]
        )
        pd.DataFrame(
            {
                "index": [0, 1, 2, 3],
                "label": labels,
                "measured_norm": [0.08 * scale, 0.04 * scale, 0.13 * scale, 0.06 * scale],
                "measured_err_norm": [0.004] * 4,
                "model_norm": [0.075 * scale, 0.045 * scale, 0.12 * scale, 0.065 * scale],
                "model_err_norm": [0.006] * 4,
                "Tvib": [6500 * scale] * 4,
                "Tvib_stderr": [200] * 4,
            }
        ).to_csv(tables_dir / f"{stem}_coronal_qc_points.csv", index=False)

    limits = _shared_qc_plot_limits(
        tables_dir=tables_dir,
        pending=pending,
        plot_kinds={"boltzmann", "coronal"},
        qc_every=1,
    )

    assert limits["boltzmann_y_limits"] == (0.01, 2.0)
    assert limits["coronal_y_limit"] == 0.16
    assert limits["coronal_labels"] == ["Q1(0-0)", "Q2(0-0)", "Q1(1-1)", "Q4(1-1)", "Q6(1-1)"]


def test_boltzmann_limits_hold_default_floor_above_one_percent():
    import pandas as pd

    from fulcher_analyzer.batch_cli import _boltzmann_axis_limits

    points = pd.DataFrame(
        {
            "energy_eV": [0.0, 0.12, 0.30],
            "nd_rel": [1.05, 0.12, 0.014],
            "relerr": [0.05, 0.10, 0.05],
        }
    )
    fit_curve = pd.DataFrame(
        {
            "energy_eV": [0.0, 0.45],
            "nd_bol_synth": [1.0, 0.02],
        }
    )

    _, ylim = _boltzmann_axis_limits(points, fit_curve)

    assert ylim == (0.01, 2.0)


def test_boltzmann_limits_hold_default_ceiling_for_high_first_points():
    import pandas as pd

    from fulcher_analyzer.batch_cli import _boltzmann_axis_limits

    points = pd.DataFrame(
        {
            "energy_eV": [0.0, 0.12, 0.30],
            "nd_rel": [1.45, 0.12, 0.014],
            "relerr": [0.05, 0.10, 0.05],
        }
    )
    fit_curve = pd.DataFrame(
        {
            "energy_eV": [0.0, 0.45],
            "nd_bol_synth": [1.0, 0.02],
        }
    )

    _, ylim = _boltzmann_axis_limits(points, fit_curve)

    assert ylim == (0.01, 2.0)


def test_boltzmann_limits_clip_errors_when_points_stay_above_floor():
    import pandas as pd

    from fulcher_analyzer.batch_cli import _boltzmann_axis_limits

    points = pd.DataFrame(
        {
            "energy_eV": [0.0, 0.14, 0.30],
            "nd_rel": [1.05, 0.07, 0.03],
            "relerr": [0.05, 0.98, 0.10],
        }
    )
    fit_curve = pd.DataFrame(
        {
            "energy_eV": [0.0, 0.45],
            "nd_bol_synth": [1.0, 0.02],
        }
    )

    _, ylim = _boltzmann_axis_limits(points, fit_curve)

    assert ylim == (0.01, 2.0)


def test_batch_cli_rejects_qc_flags_during_fit(tmp_path):
    from argparse import Namespace

    from fulcher_analyzer.batch_cli import analyze_batch

    input_dir = tmp_path / "dataset" / "intensities"
    output_dir = tmp_path / "dataset"
    input_dir.mkdir(parents=True)
    (input_dir / "193809_fr_9.csv").write_text("not,a,real,intensity,table\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="QC plotting is separated from fitting"):
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
                plot_only=False,
                resume=False,
                checkpoint_every=1,
                workers=1,
                progress_every=10,
                no_progress=True,
                show_model_output=False,
            )
        )


def test_batch_cli_plot_only_uses_saved_qc_tables(tmp_path):
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
            qc_every=0,
            plot_kind="none",
            plot_only=False,
            resume=False,
            checkpoint_every=1,
            workers=1,
            progress_every=10,
            no_progress=True,
            show_model_output=False,
        )
    )
    boltzmann_before = pd.read_csv(output_dir / "boltzmann_summary.csv")
    coronal_before = pd.read_csv(output_dir / "coronal_summary.csv")
    for path in input_dir.glob("*.csv"):
        path.write_text("corrupted,on,purpose\n", encoding="utf-8")

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
            plot_kind="coronal",
            plot_only=True,
            resume=False,
            checkpoint_every=1,
            workers=1,
            progress_every=10,
            no_progress=True,
            show_model_output=False,
        )
    )

    assert list((output_dir / "plots" / "coronal").glob("*_coronal_tvib_*K_qc.png"))
    pd.testing.assert_frame_equal(
        boltzmann_before,
        pd.read_csv(output_dir / "boltzmann_summary.csv"),
    )
    pd.testing.assert_frame_equal(
        coronal_before,
        pd.read_csv(output_dir / "coronal_summary.csv"),
    )


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
            qc_every=0,
            plot_kind="none",
            plot_only=False,
            resume=True,
            checkpoint_every=1,
            workers=1,
            progress_every=10,
            no_progress=True,
            show_model_output=False,
        )
    )

    summary = pd.read_csv(output_dir / "boltzmann_summary.csv")
    assert summary.to_dict(orient="records") == [
        {"frame": 9, "shot": 193809, "status": "ok", "stem": "193809_fr_9"}
    ]


def test_batch_cli_parallel_workers_merge_summaries(tmp_path):
    import shutil
    from argparse import Namespace

    import pandas as pd

    from fulcher_analyzer.batch_cli import analyze_batch
    from fulcher_analyzer.intensity_io import INTENSITY_DATA

    input_dir = tmp_path / "dataset" / "intensities"
    output_dir = tmp_path / "dataset"
    input_dir.mkdir(parents=True)
    for frame in ("10", "11"):
        shutil.copyfile(INTENSITY_DATA / "152478_fr_10.csv", input_dir / f"152478_fr_{frame}.csv")
        shutil.copyfile(INTENSITY_DATA / "152478_fr_10_err.csv", input_dir / f"152478_fr_{frame}_err.csv")

    analyze_batch(
        Namespace(
            input_dir=input_dir,
            output_dir=output_dir,
            fit_report_dir=None,
            manifest=None,
            isotopologue="h",
            max_fit_relerr=1.0,
            max_frames=None,
            qc_every=0,
            plot_kind="none",
            plot_only=False,
            resume=False,
            checkpoint_every=1,
            workers=2,
            progress_every=10,
            no_progress=True,
            show_model_output=False,
        )
    )

    boltzmann_summary = pd.read_csv(output_dir / "boltzmann_summary.csv")
    coronal_summary = pd.read_csv(output_dir / "coronal_summary.csv")

    assert boltzmann_summary["stem"].tolist() == ["152478_fr_10", "152478_fr_11"]
    assert coronal_summary["stem"].tolist() == ["152478_fr_10", "152478_fr_11"]
    assert boltzmann_summary["status"].tolist() == ["ok", "ok"]
    assert coronal_summary["status"].tolist() == ["ok", "ok"]


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
        test_batch_cli_plot_only_uses_saved_qc_tables,
        test_batch_cli_resume_skips_completed_summary_rows,
        test_batch_cli_parallel_workers_merge_summaries,
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
