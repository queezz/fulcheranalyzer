"""
Smoke test — Phase 2a.

Verifies that:
1. The package imports cleanly from the src-layout.
2. Data files are found at their expected locations.
3. Both canonical datasets load and produce DataFrames of the right shape.
4. BoltzmannPlot initialises for both D2 and H2 (instantiates MolecularConstants,
   loads molecular data, runs the Boltzmann calculation — no fit yet).

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


def test_data_folders_exist():
    from fulcher_analyzer.coronalmodel import MOLECULAR_DATA_FOLDER, DATA_FOLDER
    import os

    assert os.path.isdir(MOLECULAR_DATA_FOLDER), (
        f"MOLECULAR_DATA_FOLDER not found: {MOLECULAR_DATA_FOLDER}"
    )
    assert os.path.isdir(DATA_FOLDER), (
        f"DATA_FOLDER not found: {DATA_FOLDER}"
    )


def test_molecular_data_files():
    from fulcher_analyzer.coronalmodel import MOLECULAR_DATA_FOLDER
    import os

    required = [
        "franck_condon_factor.txt",
        "franck_condon_factor_D2.txt",
        "vibrational_energy.txt",
        "vibrational_energy_D2.txt",
        "excitation_vibrational_energy.txt",
        "excitation_vibrational_energy_D2.txt",
        "fulcher-\u03b1_band_wavelength.txt",
        "fulcher-\u03b1_band_wavenumber_D2.txt",
    ]
    for fname in required:
        path = os.path.join(MOLECULAR_DATA_FOLDER, fname)
        assert os.path.isfile(path), f"Missing molecular data file: {path}"


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
        test_data_folders_exist,
        test_molecular_data_files,
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
