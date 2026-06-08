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
