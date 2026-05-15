"""
Smoke test — Phase 2a / 2i.

Verifies that:
1. The package imports cleanly from the src-layout.
2. Data files are found at their expected locations.
3. Both canonical datasets load and produce DataFrames of the right shape.
4. BoltzmannPlot initialises for both D2 and H2 (instantiates MolecularConstants,
   loads molecular data, runs the Boltzmann calculation — no fit yet).
5. The new top-level public API works and is identical to the legacy facade.

Run with:
    pip install -e .
    pytest tests/test_smoke.py
or:
    python tests/test_smoke.py
"""

import pytest


def test_import():
    from fulcher_analyzer import coronalmodel as fcm  # noqa: F401
    assert hasattr(fcm, "BoltzmannPlot")
    assert hasattr(fcm, "CoronaModel")
    assert hasattr(fcm, "MolecularConstants")
    assert hasattr(fcm, "read_intensities")


def test_public_api():
    """Top-level package exposes the canonical public names."""
    import fulcher_analyzer as fa

    assert hasattr(fa, "BoltzmannPlot")
    assert hasattr(fa, "CoronaModel")
    assert hasattr(fa, "MolecularConstants")
    assert hasattr(fa, "read_intensities")
    assert hasattr(fa, "write_intensities")


def test_public_api_same_objects():
    """Top-level names are the same objects as the legacy facade names."""
    import fulcher_analyzer as fa
    from fulcher_analyzer import coronalmodel as fcm

    assert fa.BoltzmannPlot is fcm.BoltzmannPlot
    assert fa.CoronaModel is fcm.CoronaModel
    assert fa.MolecularConstants is fcm.MolecularConstants
    assert fa.read_intensities is fcm.read_intensities


def test_data_folders_exist():
    from fulcher_analyzer.coronalmodel import MOLECULAR_DATA_FOLDER, DATA_FOLDER
    import os

    assert MOLECULAR_DATA_FOLDER.is_dir(), (
        f"MOLECULAR_DATA_FOLDER not found: {MOLECULAR_DATA_FOLDER}"
    )
    assert os.path.isdir(DATA_FOLDER), (
        f"DATA_FOLDER not found: {DATA_FOLDER}"
    )


def test_molecular_data_files():
    from fulcher_analyzer.coronalmodel import MOLECULAR_DATA_FOLDER

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
    from fulcher_analyzer import coronalmodel as fcm

    inte, interr = fcm.read_intensities(150482, 7)
    assert inte.shape == (14, 4), f"Unexpected D2 intensity shape: {inte.shape}"
    assert interr.shape == (14, 4), f"Unexpected D2 error shape: {interr.shape}"


def test_read_intensities_h2():
    from fulcher_analyzer import coronalmodel as fcm

    inte, interr = fcm.read_intensities(152478, 10)
    assert inte.shape == (11, 3), f"Unexpected H2 intensity shape: {inte.shape}"
    assert interr.shape == (11, 3), f"Unexpected H2 error shape: {interr.shape}"


def test_boltzmann_init_d2():
    from fulcher_analyzer import coronalmodel as fcm

    inte = fcm.read_intensities(150482, 7)
    bp = fcm.BoltzmannPlot(inte, "d")
    assert bp.isotop == "d"
    assert bp.nd.shape == (14, 4)
    assert bp.nd_rel.shape == (14, 4)


def test_boltzmann_init_h2():
    from fulcher_analyzer import coronalmodel as fcm

    inte = fcm.read_intensities(152478, 10)
    bp = fcm.BoltzmannPlot(inte, "h")
    assert bp.isotop == "h"
    assert bp.nd.shape == (11, 3)
    assert bp.nd_rel.shape == (11, 3)


if __name__ == "__main__":
    import sys

    tests = [
        test_import,
        test_public_api,
        test_public_api_same_objects,
        test_data_folders_exist,
        test_molecular_data_files,
        test_spectroscopic_constants_values,
        test_read_intensities_d2,
        test_read_intensities_h2,
        test_boltzmann_init_d2,
        test_boltzmann_init_h2,
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
