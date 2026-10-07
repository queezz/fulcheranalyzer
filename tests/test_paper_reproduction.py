"""
Numerical regression tests — Phase 2b.

Freezes the published workflow results from:
  Kuzmin et al., JQSRT 267, 107592 (2021)

These values were recaptured after the 0.2.1 population-consistency fix made
the Einstein-A route canonical. They must not change accidentally under future
refactoring of the package.

Reference values (archaeology report, 2026-05-15):
  D2 (shot 150482, frame 7):  Trot1=278.57 K  Trot2=1732.77 K
                               alpha=0.7611    beta=0.4162
                               Tvib=7840 K     Tviberr=554 K
  H2 (shot 152478, frame 10): Trot1=372.27 K  Trot2=2010.27 K
                               alpha=0.7341    beta=0.4476
                               Tvib=7053 K     Tviberr=722 K

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
    from fulcher_analyzer import (
        BoltzmannPlot,
        CoronaModel,
        MolecularConstants,
        read_intensities,
    )
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
    bp, _como = d2_workflow
    assert bp.nd.shape == (14, 4)
    assert bp.nd_synth.shape == (14, 4)


def test_d2_coronal_shapes(d2_workflow):
    _bp, como = d2_workflow
    # R-matrix: [vx, jx, vd, jd] = [4, 15, 4, 14]
    assert como.Rm.shape == (4, 15, 4, 14)
    assert como.nx.shape == (15, 4)
    assert como.nd.shape == (14, 4)


# ── D2 Boltzmann fit values ──────────────────────────────────────────────────

def test_d2_trot1(d2_workflow):
    bp, _ = d2_workflow
    assert bp.trot1 == pytest.approx(278.57, rel=1e-3), (
        f"D2 Trot1 changed: got {bp.trot1:.4f}"
    )


def test_d2_trot2(d2_workflow):
    bp, _ = d2_workflow
    assert bp.trot2 == pytest.approx(1732.77, rel=1e-3), (
        f"D2 Trot2 changed: got {bp.trot2:.4f}"
    )


def test_d2_alpha(d2_workflow):
    bp, _ = d2_workflow
    assert bp.alpha == pytest.approx(0.7611, abs=1e-3), (
        f"D2 alpha changed: got {bp.alpha:.6f}"
    )


def test_d2_beta(d2_workflow):
    bp, _ = d2_workflow
    assert bp.beta == pytest.approx(0.4162, abs=1e-3), (
        f"D2 beta changed: got {bp.beta:.6f}"
    )


# ── D2 coronal fit values ────────────────────────────────────────────────────

def test_d2_tvib(d2_workflow):
    _, como = d2_workflow
    assert como.tvib == pytest.approx(7840, rel=1e-3), (
        f"D2 Tvib changed: got {como.tvib:.2f}"
    )


def test_d2_tviberr(d2_workflow):
    _, como = d2_workflow
    assert como.tviberr == pytest.approx(554, rel=5e-3), (
        f"D2 Tviberr changed: got {como.tviberr:.2f}"
    )


# ── H2 data shapes ──────────────────────────────────────────────────────────

def test_h2_intensity_shape():
    from fulcher_analyzer import read_intensities
    inte, interr = read_intensities(152478, 10)
    assert inte.shape == (11, 3)
    assert interr.shape == (11, 3)


def test_h2_boltzmann_shapes(h2_workflow):
    bp, _como = h2_workflow
    assert bp.nd.shape == (11, 3)
    assert bp.nd_synth.shape == (11, 3)


def test_h2_coronal_shapes(h2_workflow):
    _bp, como = h2_workflow
    # R-matrix: [vx, jx, vd, jd] = [3, 12, 3, 11]
    assert como.Rm.shape == (3, 12, 3, 11)
    assert como.nx.shape == (12, 3)
    assert como.nd.shape == (11, 3)


# ── H2 Boltzmann fit values ──────────────────────────────────────────────────

def test_h2_trot1(h2_workflow):
    bp, _ = h2_workflow
    assert bp.trot1 == pytest.approx(372.27, rel=1e-3), (
        f"H2 Trot1 changed: got {bp.trot1:.4f}"
    )


def test_h2_trot2(h2_workflow):
    bp, _ = h2_workflow
    assert bp.trot2 == pytest.approx(2010.27, rel=1e-3), (
        f"H2 Trot2 changed: got {bp.trot2:.4f}"
    )


def test_h2_alpha(h2_workflow):
    bp, _ = h2_workflow
    assert bp.alpha == pytest.approx(0.7341, abs=1e-3), (
        f"H2 alpha changed: got {bp.alpha:.6f}"
    )


def test_h2_beta(h2_workflow):
    bp, _ = h2_workflow
    assert bp.beta == pytest.approx(0.4476, abs=1e-3), (
        f"H2 beta changed: got {bp.beta:.6f}"
    )


# ── H2 coronal fit values ────────────────────────────────────────────────────

def test_h2_tvib(h2_workflow):
    _, como = h2_workflow
    assert como.tvib == pytest.approx(7053, rel=1e-3), (
        f"H2 Tvib changed: got {como.tvib:.2f}"
    )


def test_h2_tviberr(h2_workflow):
    _, como = h2_workflow
    assert como.tviberr == pytest.approx(722, rel=5e-3), (
        f"H2 Tviberr changed: got {como.tviberr:.2f}"
    )


# ── full popt vectors (tolerant — catches gross changes) ────────────────────

def test_d2_popt_vector(d2_workflow):
    """Full popt = [alpha, beta, Trot1, Trot2, c0, c1, c2, c3]."""
    bp, _ = d2_workflow
    expected = np.array([
        7.61121699e-01, 4.16207087e-01,
        2.78565228e+02, 1.73277372e+03,
        1.13099261e+00, 1.21626979e+00, 1.01224646e+00, 8.73546575e-01,
    ])
    np.testing.assert_allclose(bp.popt, expected, rtol=1e-3)


def test_h2_popt_vector(h2_workflow):
    """Full popt = [alpha, beta, Trot1, Trot2, c0, c1, c2]."""
    bp, _ = h2_workflow
    expected = np.array([
        7.34073413e-01, 4.47590044e-01,
        3.72266471e+02, 2.01027337e+03,
        1.07539267e+00, 9.75955356e-01, 8.48838566e-01,
    ])
    np.testing.assert_allclose(bp.popt, expected, rtol=1e-3)
