"""
Numerical regression tests — Phase 2b.

Freezes the published workflow results from:
  Kuzmin et al., JQSRT 267, 107592 (2021)

These values were captured from a clean run of
examples/CoronalModel-D-H.ipynb after the Phase 2a src-layout migration
and must not change under any future refactoring of the package.

Reference values (archaeology report, 2026-05-15):
  D2 (shot 150482, frame 7):  Trot1=279.94 K  Trot2=1782.83 K
                               alpha=0.7609    beta=0.4159
                               Tvib=7743 K     Tviberr=557 K
  H2 (shot 152478, frame 10): Trot1=374.72 K  Trot2=2079.14 K
                               alpha=0.7335    beta=0.4473
                               Tvib=6801 K     Tviberr=753 K

Run with:
    pytest tests/test_paper_reproduction.py
or the full suite:
    pytest
"""

import numpy as np
import pytest


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def d2_workflow():
    """Run the full D2 canonical workflow once per test module."""
    import matplotlib
    matplotlib.use("Agg")
    from fulcher_analyzer import BoltzmannPlot, CoronaModel, read_intensities

    inte = read_intensities(150482, 7)
    bp = BoltzmannPlot(inte, "d")
    bp.autofit()
    como = CoronaModel(bp)
    como.coronal_autofit()
    return bp, como


@pytest.fixture(scope="module")
def h2_workflow():
    """Run the full H2 canonical workflow once per test module."""
    import matplotlib
    matplotlib.use("Agg")
    from fulcher_analyzer import BoltzmannPlot, CoronaModel, read_intensities

    inte = read_intensities(152478, 10)
    bp = BoltzmannPlot(inte, "h")
    bp.autofit()
    como = CoronaModel(bp)
    como.coronal_autofit()
    return bp, como


# ── basic import ────────────────────────────────────────────────────────────

def test_import_canonical():
    from fulcher_analyzer import BoltzmannPlot, CoronaModel, read_intensities, MolecularConstants
    assert callable(BoltzmannPlot)
    assert callable(CoronaModel)
    assert callable(read_intensities)
    assert callable(MolecularConstants)


# ── D2 data shapes ──────────────────────────────────────────────────────────

def test_d2_intensity_shape():
    from fulcher_analyzer import read_intensities
    inte, interr = read_intensities(150482, 7)
    assert inte.shape == (14, 4)
    assert interr.shape == (14, 4)


def test_d2_boltzmann_shapes(d2_workflow):
    bp, como = d2_workflow
    assert bp.nd.shape == (14, 4)
    assert bp.nd_synth.shape == (14, 4)


def test_d2_coronal_shapes(d2_workflow):
    bp, como = d2_workflow
    # R-matrix: [vx, jx, vd, jd] = [4, 15, 4, 14]
    assert como.Rm.shape == (4, 15, 4, 14)
    assert como.nx.shape == (15, 4)
    assert como.nd.shape == (14, 4)


# ── D2 Boltzmann fit values ──────────────────────────────────────────────────

def test_d2_trot1(d2_workflow):
    bp, _ = d2_workflow
    assert bp.trot1 == pytest.approx(279.94, rel=1e-3), (
        f"D2 Trot1 changed: got {bp.trot1:.4f}"
    )


def test_d2_trot2(d2_workflow):
    bp, _ = d2_workflow
    assert bp.trot2 == pytest.approx(1782.83, rel=1e-3), (
        f"D2 Trot2 changed: got {bp.trot2:.4f}"
    )


def test_d2_alpha(d2_workflow):
    bp, _ = d2_workflow
    assert bp.alpha == pytest.approx(0.7609, abs=1e-3), (
        f"D2 alpha changed: got {bp.alpha:.6f}"
    )


def test_d2_beta(d2_workflow):
    bp, _ = d2_workflow
    assert bp.beta == pytest.approx(0.4159, abs=1e-3), (
        f"D2 beta changed: got {bp.beta:.6f}"
    )


# ── D2 coronal fit values ────────────────────────────────────────────────────

def test_d2_tvib(d2_workflow):
    _, como = d2_workflow
    assert como.tvib == pytest.approx(7743, rel=1e-3), (
        f"D2 Tvib changed: got {como.tvib:.2f}"
    )


def test_d2_tviberr(d2_workflow):
    _, como = d2_workflow
    assert como.tviberr == pytest.approx(557, rel=5e-3), (
        f"D2 Tviberr changed: got {como.tviberr:.2f}"
    )


# ── H2 data shapes ──────────────────────────────────────────────────────────

def test_h2_intensity_shape():
    from fulcher_analyzer import read_intensities
    inte, interr = read_intensities(152478, 10)
    assert inte.shape == (11, 3)
    assert interr.shape == (11, 3)


def test_h2_boltzmann_shapes(h2_workflow):
    bp, como = h2_workflow
    assert bp.nd.shape == (11, 3)
    assert bp.nd_synth.shape == (11, 3)


def test_h2_coronal_shapes(h2_workflow):
    bp, como = h2_workflow
    # R-matrix: [vx, jx, vd, jd] = [3, 12, 3, 11]
    assert como.Rm.shape == (3, 12, 3, 11)
    assert como.nx.shape == (12, 3)
    assert como.nd.shape == (11, 3)


# ── H2 Boltzmann fit values ──────────────────────────────────────────────────

def test_h2_trot1(h2_workflow):
    bp, _ = h2_workflow
    assert bp.trot1 == pytest.approx(374.72, rel=1e-3), (
        f"H2 Trot1 changed: got {bp.trot1:.4f}"
    )


def test_h2_trot2(h2_workflow):
    bp, _ = h2_workflow
    assert bp.trot2 == pytest.approx(2079.14, rel=1e-3), (
        f"H2 Trot2 changed: got {bp.trot2:.4f}"
    )


def test_h2_alpha(h2_workflow):
    bp, _ = h2_workflow
    assert bp.alpha == pytest.approx(0.7335, abs=1e-3), (
        f"H2 alpha changed: got {bp.alpha:.6f}"
    )


def test_h2_beta(h2_workflow):
    bp, _ = h2_workflow
    assert bp.beta == pytest.approx(0.4473, abs=1e-3), (
        f"H2 beta changed: got {bp.beta:.6f}"
    )


# ── H2 coronal fit values ────────────────────────────────────────────────────

def test_h2_tvib(h2_workflow):
    _, como = h2_workflow
    assert como.tvib == pytest.approx(6801, rel=1e-3), (
        f"H2 Tvib changed: got {como.tvib:.2f}"
    )


def test_h2_tviberr(h2_workflow):
    _, como = h2_workflow
    assert como.tviberr == pytest.approx(753, rel=5e-3), (
        f"H2 Tviberr changed: got {como.tviberr:.2f}"
    )


# ── full popt vectors (tolerant — catches gross changes) ────────────────────

def test_d2_popt_vector(d2_workflow):
    """Full popt = [alpha, beta, Trot1, Trot2, c0, c1, c2, c3]."""
    bp, _ = d2_workflow
    expected = np.array([
        7.60906788e-01, 4.15931835e-01,
        2.79937908e+02, 1.78283197e+03,
        1.12869013e+00, 1.25887148e+00, 1.08586827e+00, 9.70839868e-01,
    ])
    np.testing.assert_allclose(bp.popt, expected, rtol=1e-3)


def test_h2_popt_vector(h2_workflow):
    """Full popt = [alpha, beta, Trot1, Trot2, c0, c1, c2]."""
    bp, _ = h2_workflow
    expected = np.array([
        7.33530273e-01, 4.47270326e-01,
        3.74721751e+02, 2.07913968e+03,
        1.07399710e+00, 1.02697855e+00, 9.37903466e-01,
    ])
    np.testing.assert_allclose(bp.popt, expected, rtol=1e-3)
