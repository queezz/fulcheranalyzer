"""
Coronal model for two-stage Tvib fitting of H2/D2 Fulcher-alpha emission.

Workflow overview
-----------------
1. **BoltzmannPlot** (upstream) fits the d-state rotational population and
   provides the rotational parameters ``alpha``, ``beta``, ``Trot1``,
   ``Trot2`` together with the measured d-state population ``nd`` and its
   error ``nd_err``.

2. **CoronaModel** constructs a synthetic X-state population ``nx`` for a
   trial vibrational temperature ``Tvib`` via :meth:`calc_X_bp` /
   :meth:`calc_nx`.

3. The X-state population is projected to the d-state through the R-matrix::

       R[vX, JX, vd, Jd] = FCF[vX, vd] * CCS[vX, vd] * RTP[JX, Jd]

   where FCF are Franck-Condon factors, CCS is the electron-impact
   cross-section factor (Boltzmann exponential in the threshold energy), and
   RTP is the radiation transition probability (branching ratio × nuclear-spin
   selection rule).

4. The model d-state population is computed as a matrix product::

       n_d(model) = Rm2d @ nx.flatten(order='f')

5. :meth:`coronal_fit_formula` returns the normalised, masked d-state
   population that ``scipy.optimize.curve_fit`` minimises against the
   experimental data.

6. :meth:`coronal_autofit` performs the second-stage fit varying **only**
   ``Tvib``; ``alpha``, ``beta``, ``Trot1``, and ``Trot2`` are inherited
   from the preceding Boltzmann fit and are not free parameters here.

Side-effect pattern
-------------------
Many methods write intermediate arrays directly onto ``self`` (``self.nx``,
``self.nd``, ``self.nxbp``, ``self.Rm``, ``self.Rm2d``, …) rather than
returning values.  Callers must invoke methods in the order documented by the
workflow above to ensure these attributes exist before they are consumed.

Plotting helpers live in ``.plotting`` and are exposed here as thin wrappers.
"""
import numpy as np
import pandas as pd
from importlib.resources import files

from .boltzmann import BoltzmannPlot, ABSOLUTESIGMA
from ._utils import flatdf, delta_kro, g_as, g_as_vector, tjpo_vector, reshape_4d2d
from .plotting import plot_rmatrix  # re-export so existing callers still work

MOLECULAR_DATA_FOLDER = files("fulcher_analyzer.data_molecular")


class CoronaModel:
    """Two-stage coronal model for H2/D2 Fulcher-alpha vibrational temperature fitting.

    Inherits rotational parameters from a completed :class:`~fulcher_analyzer.boltzmann.BoltzmannPlot`
    and fits only the X-state vibrational temperature ``Tvib``.

    Based on the Ishihara coronal model implementation.

    Parameters
    ----------
    bp : BoltzmannPlot
        A fully fitted Boltzmann plot instance for the same discharge.
        Provides ``isotop``, ``nd``, ``nd_err``, ``mask``, ``popt``,
        ``trot1``, and ``trot2``.
    """

    def __init__(self, bp: BoltzmannPlot):
        """Initialise the coronal model from a completed BoltzmannPlot.

        Loads molecular constants for the correct isotopologue, pre-calculates
        the radiation transition probability matrix ``rtp``, and builds (or
        loads from cache) the full R-matrix.  Also pre-computes the X-state
        rotational energy grid ``EX`` used later in population calculations.
        """
        self.name = "Coronal Model for H and D"
        self.bp = bp
        self.isotop = self.bp.isotop
        self.prep_constants()
        self.prep_style()

        self.prep_corona_fit(load=True)

        # Load constants from MolecularConstants, don't keep them here.
        # self.acoeff()
        # self.load_constants()

    def prep_style(self, ms=8):
        """Prepare standard plot styles.
        Thin wrapper — logic lives in :func:`~fulcher_analyzer.plotting.prep_corona_style`.
        """
        from .plotting import prep_corona_style
        prep_corona_style(self, ms=ms)

    # ------------------------------------------------------------------
    # Initialization and constants
    # ------------------------------------------------------------------

    def prep_constants(self):
        """Load isotopologue-specific molecular constants and pre-compute ``rtp``.

        Sets on ``self``:

        - ``A``, ``Asum`` — Einstein A-coefficients for d→a transitions.
        - ``E_vib``, ``Ee_vib`` — X-state and d-state vibrational energy grids.
        - ``fcf`` — Franck-Condon factors, shape ``(vX, vd)``.
        - ``v_range_f_vibro`` — ``[vx_range, vd_range]`` used by
          :meth:`f_vibro` for the legacy Ishihara vibro-only fit.
        - ``rtp`` — radiation transition probability matrix, computed by
          :meth:`calc_rtp`.
        """
        mol = self.bp.mol
        self.mol = self.bp.mol
        if self.isotop == "d":
            self.A = mol.AD
            self.Asum = mol.ADsum
            self.E_vib, self.Ee_vib, self.fcf = mol.corona_constants_d

        if self.isotop == "h":
            self.A = mol.AH
            self.Asum = self.mol.AHsum
            self.E_vib, self.Ee_vib, self.fcf = mol.corona_constants_h

        self.v_range_f_vibro = [
            5,
            self.bp.nd.shape[1],
        ]  # [vx_range, vd_range] for fit_vibro()

        self.calc_rtp()

    # ------------------------------------------------------------------
    # Legacy Ishihara vibro-only fit (rotationally integrated)
    # ------------------------------------------------------------------

    def calculate_e_cross(self, Te=15):
        """Compute the electron-impact cross-section matrix for the legacy vibro fit.

        Duplicate of :meth:`ccs_formula` expressed as a DataFrame over all
        ``(vX, vd)`` pairs.  Used by :meth:`f_vibro` / :meth:`fit_vibro_ishi`
        which work with rotationally-summed populations (Ishihara's original
        approach) rather than the fully resolved R-matrix path.

        Sets ``self.ccs`` — cross-section DataFrame with shape matching ``fcf``.

        Parameters
        ----------
        Te : float
            Electron temperature in eV.  ``Ee_vib`` is the d-state vibrational
            energy; ``E_vib`` is the X-state vibrational energy.
        """
        from scipy.constants import Boltzmann, elementary_charge

        kb = Boltzmann / elementary_charge
        Te = Te / kb  # Convert Te from [eV] to [K]

        self.ccs = pd.DataFrame(
            [
                [
                    np.exp(
                        -(
                            (self.Ee_vib[vd] - self.Ee_vib[0])
                            - (self.E_vib[vX] - self.E_vib[0])
                        )
                        / (kb * Te)
                    )
                    for vd in range(len(self.fcf[0, :]))
                ]
                for vX in range(len(self.fcf[:, 0]))
            ]
        )

    def f_vibro(self, _, T_vib, fit=True):
        """Rotationally-integrated coronal model population (Ishihara's formula 4.5).

        This is the fit function for the legacy vibro-only path; it sums over
        all rotational levels so that only ``Tvib`` is varied.  The R-matrix is
        not rotationally resolved here and therefore does not include the
        branching ratio.

        Parameters
        ----------
        _ : ignored
            Placeholder for the independent variable required by
            ``scipy.optimize.curve_fit`` (no independent variable is needed
            in this parameterisation).
        T_vib : float
            X-state vibrational temperature in K.
        fit : bool
            If ``True`` (default) return the normalised population vector
            suitable for ``curve_fit``.  If ``False`` return the raw ``nv``
            DataFrame.
        """
        from scipy.constants import Boltzmann, elementary_charge

        kb = Boltzmann / elementary_charge
        vx_range, vd_range = self.v_range_f_vibro

        ccs = self.ccs.values
        nv = pd.DataFrame(
            [
                [
                    ccs[vx, vd]
                    * self.fcf[vx, vd]
                    * np.exp(-(self.E_vib[vx] - self.E_vib[0]) / (kb * T_vib))
                    / self.Asum[vd]
                    for vd in range(vd_range)
                ]
                for vx in range(vx_range)
            ]
        )
        if not fit:
            return nv
        return (nv.sum() / nv.values.sum()).values

    # Population formula (Ishihara eq. 4.5):
    # n_{dv'} * sum(A^{dv'}_{av''}) ∝  sum_{v=0}^{vmax} R_{Xv}^{dv'} * exp(-E_vib^X(v) / k*Tvib^X)

    def fit_vibro_ishi(self):
        r"""Fit vibrational temperature using the rotationally-integrated Ishihara model.

        This is a legacy diagnostic fit that ignores the rotational distribution
        to obtain a quick estimate of ``Tvib``.  See formula (4.5) in Ishihara's
        Master thesis (p. 36); the R-coefficient there is not rotationally
        resolved and does not include the branching ratio.

        Sets on ``self``:

        - ``tvib`` — fitted vibrational temperature in K.
        - ``tvib_cov`` — covariance matrix from ``curve_fit``.
        - ``tvib_err`` — 1-sigma uncertainty in K.

        .. math::

            n_{dv'} \sum A^{dv'}_{av''} \propto
            \sum_{v=0}^{v_{\max}} R_{Xv}^{dv'}
            \exp\!\left(-\frac{E_{\mathrm{vib}}^{X}(v)}{k T_{\mathrm{vib}}^{X}}\right)
        """
        from scipy.optimize import curve_fit

        tvib, cov = curve_fit(self.f_vibro, [], self.bp.nd_vibrofit, 10000)
        self.tvib = tvib[0]
        self.tvib_cov = cov
        self.tvib_err = np.sqrt(np.diag(self.tvib_cov))[0]
        print(f"Tvib = {tvib[0]:.0f} +- {self.tvib_err:.0f} K")

    def plot_fit_ishi(self, **kws):
        """Thin wrapper — see :func:`~fulcher_analyzer.plotting.plot_fit_ishi`."""
        from .plotting import plot_fit_ishi
        return plot_fit_ishi(self, **kws)

    # ------------------------------------------------------------------
    # Transition / rate coefficients  (Niihama & Ishihara)
    # ------------------------------------------------------------------

    def branching(self, JX, Jd):
        """Compute the rotational branching ratio for one (JX, Jd) pair.

        Evaluates the sum over branches ``r = 1…4`` weighted by the Q-branch
        intensity ratios ``Qr = [0.76, 0.122, 0.10, 0.014]``::

            branching(JX, Jd) = sum_r  Qr[r] * (2*Jd+1) * W3j(Jd, r, JX; 1,-1,0)^2

        Parameters
        ----------
        JX : int
            X-state rotational quantum number.
        Jd : int
            d-state rotational quantum number (starts from 1).

        Returns
        -------
        float
            Branching ratio for the (JX, Jd) transition.
        """
        from sympy.physics.wigner import wigner_3j

        Qr = np.array([0.76, 0.122, 0.1, 0.014])
        return float(
            sum(
                [
                    Qr[r - 1] * (2 * Jd + 1) * (wigner_3j(Jd, r, JX, 1, -1, 0)) ** 2
                    for r in range(1, 5, 1)
                ]
            )
        )

    def calc_branching_matrix(self, JXmax=10, Jdmax=5):
        """Build the full branching-ratio matrix ``axd``.

        Rows correspond to JX (index = JX, starting at 0), columns to Jd
        (index = Jd − 1, because Jd starts from 1).

        Parameters
        ----------
        JXmax : int
            Maximum X-state rotational quantum number (inclusive).
        Jdmax : int
            Maximum d-state rotational quantum number (inclusive).

        Returns
        -------
        pandas.DataFrame
            Shape ``(JXmax+1, Jdmax)``.
        """
        return pd.DataFrame(
            [
                [self.branching(JX, Jd) for Jd in range(1, Jdmax + 1)]
                for JX in range(JXmax + 1)
            ]
        )

    def calc_delta(self, axd):
        """Build the nuclear-spin selection-rule (Kronecker-delta) matrix.

        Enforces the ortho/para selection rule: only transitions where the
        nuclear-spin symmetry of JX matches that of Jd are allowed
        (``delta_kro`` returns 1 if both are ortho or both are para, 0
        otherwise).  The resulting matrix has the same shape as ``axd``.

        Parameters
        ----------
        axd : array-like
            Branching-ratio matrix (used only for shape).  Rows = JX,
            columns = Jd index (Jd − 1).

        Returns
        -------
        pandas.DataFrame
            Boolean-valued (0/1) selection-rule matrix.
        """
        return pd.DataFrame(
            [
                [
                    delta_kro(
                        g_as(JX, isotop=self.isotop), g_as(Jd + 1, isotop=self.isotop)
                    )
                    for Jd in range(axd.shape[1])
                ]
                for JX in range(axd.shape[0])
            ]
        )

    def ccs_formula(self, vX, vd, Te=15):
        """Electron-impact cross-section factor for a single (vX, vd) pair.

        Returns the Boltzmann exponential factor representing the energy
        threshold for excitation from X-state level ``vX`` to d-state
        level ``vd``::

            CCS(vX, vd) = exp( -( (E_d[vd] - E_d[0]) - (E_X[vX] - E_X[0]) ) / Te )

        This is the scalar version of :meth:`calculate_e_cross`, used when
        building the R-matrix element by element.

        Parameters
        ----------
        vX : int
            X-state vibrational quantum number.
        vd : int
            d-state vibrational quantum number.
        Te : float
            Electron temperature in eV (default 15 eV).

        Returns
        -------
        float
            Dimensionless cross-section factor in (0, 1].
        """
        evd = self.mol.calculate_E_vib(20, state="d3", isotop=self.isotop)
        evx = self.mol.calculate_E_vib(20, state="X", isotop=self.isotop)
        return np.exp(-((evd[vd] - evd[0]) - (evx[vX] - evx[0])) / Te)

    # ------------------------------------------------------------------
    # Population shape and R-matrix construction
    # ------------------------------------------------------------------

    def calc_rtp(self):
        """Pre-compute the radiation transition probability matrix ``rtp``.

        ``rtp[jx, jd_index]`` = branching(jx, jd) × delta_kro(g_as(jx), g_as(jd))

        where ``jd_index = jd − 1`` (d-state Jd starts from 1, not 0).
        The fixed grid is 30 × 15 (jx = 0…29, jd = 1…15), matching
        Ishihara's implementation.

        Sets ``self.rtp`` — ndarray of shape ``(30, 15)``.
        Called once during initialisation by :meth:`prep_constants`.
        """
        # x,d = [como.popshape['jxlen'],como.popshape['jdlen']]
        x, d = [30, 15]
        br = np.empty([x, d])
        dl = np.empty([x, d])
        for jx in range(x):
            for jd in range(1, d + 1):
                br[jx, jd - 1] = self.branching(jx, jd)
                dl[jx, jd - 1] = delta_kro(
                    g_as(jx, isotop=self.isotop), g_as(jd, isotop=self.isotop)
                )
        self.rtp = br * dl

    def R_formula(self, vX, JX, vd, Jdind):
        """Compute one element of the R-matrix (electron-impact excitation rate).

        Implements::

            R[vX, JX, vd, Jdind] = FCF[vX, vd] * CCS(vX, vd) * rtp[JX, Jdind]

        This is a scalar helper used for verification and spot-checks.
        The full matrix is built more efficiently in :meth:`make_rmatrix`
        using vectorised operations.

        Parameters
        ----------
        vX : int
            X-state vibrational quantum number.
        JX : int
            X-state rotational quantum number.
        vd : int
            d-state vibrational quantum number.
        Jdind : int
            Index for d-state rotational quantum number; ``Jdind = Jd − 1``
            because Jd starts from 1 in ``self.rtp``.

        Returns
        -------
        float
            R-matrix element for the given quantum numbers.
        """
        # FCF indices
        # If in numpy, then `fcf[d][X]`, if in pandas, then `fcf[X][d]`
        if self.isotop == "d":
            self.fcf = self.mol.corona_constants_d[-1]
        if self.isotop == "h":
            self.fcf = self.mol.corona_constants_h[-1]

        # new (Ishihara-s formula)
        return self.fcf[vX][vd] * self.ccs_formula(vX, vd) * self.rtp[JX, Jdind]

        # old formula
        """
        return (
            self.fcf[vX][vd]
            * self.ccs_formula(vX, vd)
            * self.branching(JX, Jd)
            * delta_kro(g_as(JX, isotop=self.isotop), g_as(Jd, isotop=self.isotop))
        )   
        """

        # Ishihara-s formula: branching[jx, jd] * fcf[vx, vd] * ccs[vx, vd]
        # Ishihara-s branching includes kronecker-s delta

    def prep_corona_fit(self, load=True):
        """Set up population grids and build (or load) the R-matrix.

        Orchestrates the initialisation sequence:

        1. Derives d-state and X-state grid dimensions from ``self.bp.nd``.
        2. Calls :meth:`set_pop_shape` to store dimension metadata in
           ``self.popshape``.
        3. Calls :meth:`make_rmatrix` to compute or load ``self.Rm`` and
           fold it into the 2-D matrix ``self.Rm2d``.
        4. Pre-computes the X-state rotational energy grid ``self.EX``.

        Called once during ``__init__``.

        Parameters
        ----------
        load : bool
            If ``True`` (default), attempt to load a previously cached
            R-matrix from disk to save computation time.  Set to ``False``
            to force recalculation after any change to the R-matrix formula.
        """
        dshape = self.bp.nd.shape
        self.set_pop_shape([dshape[1] - 1, dshape[0]])
        self.make_rmatrix(load=load)

        self.EX = self.mol.calculate_E_rot(
            self.popshape["vxlen"],
            self.popshape["jxlen"],
            isotop=self.isotop,
            state="X",
        )

    def set_pop_shape(self, limits):
        """Store the quantum-number grid dimensions in ``self.popshape``.

        Derives maximum indices and lengths for all four quantum numbers
        (vd, jd, vX, jX) from the measured d-state population shape and the
        supplied X-state limits.

        .. note::
            **jd is a 0-based index.**  The actual d-state rotational quantum
            number is ``Jd = jd + 1`` because Jd = 0 does not exist for the
            d-state, but NumPy indexing starts at 0.  This convention requires
            ``jxmax ≥ jdmax + 1``.

        Sets ``self.popshape`` — dict with keys ``{vd,jd,vx,jx}max`` and
        ``{vd,jd,vx,jx}len``, and ``self.rshapelist`` — ordered list of
        lengths ``[vdlen, jdlen, vxlen, jxlen]``.

        Parameters
        ----------
        limits : list
            ``[vx_max, jx_max]`` — upper bounds (inclusive) for the X-state
            vibrational and rotational quantum numbers.
        """
        dshape = self.bp.nd.shape
        limits = [dshape[1] - 1, dshape[0] - 1] + limits
        names = ["vd", "jd", "vx", "jx"]
        x = {f"{n}max": l for n, l in zip(names, limits)}
        y = {f"{n}len": l + 1 for n, l in zip(names, limits)}
        self.popshape = x | y
        self.rshapelist = [self.popshape[f"{i}len"] for i in ["vd", "jd", "vx", "jx"]]

        if self.popshape["jxmax"] < self.popshape["jdmax"] + 1:
            raise ValueError(
                f"Must be: jxmax >= {self.popshape['jdmax']+1}, now jxmax={self.popshape['jxmax']}."
            )

        if self.popshape["vxmax"] < self.popshape["vdmax"]:
            raise ValueError(
                (
                    f"X-state v range (0-{self.popshape['vxmax']}) must be"
                    f" at least equal to d-state v range (0-{self.popshape['vdmax']})"
                )
            )

    def print_pop_shape(self):
        """ 
        Print vx, jx, vd, jd
        """
        sh = (
            f"vd = 0-{self.popshape['vdmax']}\tJd = 1-{self.popshape['jdmax']+1}\n"
            f"vX = 0-{self.popshape['vxmax']}\tJx = 0-{self.popshape['jxmax']}"
        )
        print(sh)

    def make_rmatrix(self, load=True):
        """Build the 4-D R-matrix ``self.Rm`` and its 2-D projection ``self.Rm2d``.

        The 4-D array has shape ``(vxlen, jxlen, vdlen, jdlen)`` and stores::

            Rm[vX, jX, vd, jd] = FCF[vX, vd] * CCS(vX, vd) * rtp[jX, jd]

        **Important:** ``jd`` here is a 0-based index; actual ``Jd = jd + 1``
        because the d-state has no ``Jd = 0`` level.

        After building ``self.Rm`` the method calls :meth:`make_rmatrix_2d`
        to fold it into the 2-D matrix used by the matrix-product in
        :meth:`calc_nd`.

        The result is cached to disk as a ``.npy`` file in the package data
        directory.  **If you change ``R_formula``, ``ccs_formula``, or
        ``branching``, pass ``load=False`` to force recalculation.**

        Parameters
        ----------
        load : bool
            If ``True`` (default), attempt to restore a cached ``.npy`` file
            matching the current grid dimensions and isotopologue.
        """
        vxmax = self.popshape["vxmax"]
        jxmax = self.popshape["jxmax"]
        vdmax = self.popshape["vdmax"]
        jdmax = self.popshape["jdmax"]

        fname = f"Rmatrix_{vxmax}_{jxmax}_{vdmax}_{jdmax}_{self.isotop}.npy"
        rmatrix_resource = MOLECULAR_DATA_FOLDER.joinpath(fname)
        if load:
            try:
                with rmatrix_resource.open("rb") as f:
                    self.Rm = np.load(f)
                print("saved R-matrix found, loaded")
                self.make_rmatrix_2d()
                return
            except:
                print("could not load, calculating R-matrix")

        Rmatrix = np.empty([vxmax + 1, jxmax + 1, vdmax + 1, jdmax + 1])
        for vx in range(vxmax + 1):
            for jx in range(jxmax + 1):
                for vd in range(vdmax + 1):
                    for jd in range(jdmax + 1):
                        # Rmatrix[vx, jx, vd, jd] = self.R_formula(vx, jx, vd, jd)
                        Rmatrix[vx, jx, vd, jd] = (
                            self.fcf[vx, vd]
                            * self.ccs_formula(vx, vd)
                            * self.rtp[jx, jd]
                        )

        self.Rm = Rmatrix
        # TODO: move regenerated R-matrix cache to a user cache directory (e.g. platformdirs)
        # For editable installs the resource path is a real filesystem path and np.save works.
        np.save(str(rmatrix_resource), Rmatrix)
        self.make_rmatrix_2d()

    def make_rmatrix_2d(self):
        """Flatten the 4-D R-matrix into the 2-D operator ``self.Rm2d``.

        Uses :func:`~fulcher_analyzer._utils.reshape_4d2d` (Fortran column-major
        ordering) and transposes the result so that::

            nd_flat = Rm2d @ nx_flat

        where ``nx_flat = nx.values.flatten(order='f')`` and ``nd_flat`` has
        length ``vdlen * jdlen``.  The ``flatten(order='f')`` call in
        :meth:`calc_nd` **must** use the same ordering — changing either side
        silently breaks the population reconstruction.

        Sets ``self.Rm2d`` — 2-D ndarray of shape ``(vdlen*jdlen, vxlen*jxlen)``.
        """
        self.Rm2d = reshape_4d2d(self.Rm).T  # transpose for correct orientation

    def check_rmatrix_indexing(self):
        """Diagnostic: verify that the cached R-matrix matches ``R_formula`` for a spot-check.

        Picks the fixed test point ``(vX=1, jX=5, vd=2, jd=1)`` and compares
        the value stored in ``self.Rm`` against a direct call to
        :meth:`R_formula`.  Prints both values and a validity flag.
        Intended for interactive verification after regenerating the matrix.
        """
        from operator import mul
        from functools import reduce

        vx, jx, vd, jd = (1, 5, 2, 1)
        print(f"vX={vx} jX={jx} vd={vd} jd={jd}")
        a = self.Rm[vx, jx, vd, jd]
        b = self.R_formula(vx, jx, vd, jd)
        print(a)
        print(b)
        print(f"valid: {a==b}")
        print()
        print(f"matrix shape = {self.Rm.shape}")
        print(f"matrix shape sum = {reduce(mul, self.Rm.shape, 1)}")

    # ------------------------------------------------------------------
    # X-state population and d-state projection
    # ------------------------------------------------------------------

    def calc_X_bp(
        self, Tvib=5000.0, Trot1=200.0, Trot2=1000.0, alpha=0.1, beta=0.1, const=1.0
    ):
        """Compute the synthetic X-state Boltzmann population ``nxbp``.

        Constructs a two-temperature rotational distribution with a single
        vibrational temperature::

            nxbp[J, v] = exp(-Evib[v] / k*Tvib)
                         * [ (1 - ab[v]) * exp(-Erot[v,J] / k*Trot1*frac[v])
                             +    ab[v]  * exp(-Erot[v,J] / k*Trot2*frac[v]) ]
                         * const

        where ``ab[v]`` is ``alpha`` for v=0 and ``beta`` for v>0 (D2), or a
        slightly different pattern for H2 (see inline comments).  Jmax is
        limited by the Franck-Condon data; higher ``vmax`` contribute less to
        the d-state population.  All temperatures are in K.

        Sets ``self.nxbp`` — DataFrame of shape ``(jxlen, vxlen)``, scaled by
        ``const``.

        Parameters
        ----------
        Tvib : float
            X-state vibrational temperature in K.
        Trot1 : float
            Low rotational temperature in K.
        Trot2 : float
            High rotational temperature in K.
        alpha : float
            Two-temperature mixing coefficient for v = 0.
        beta : float
            Two-temperature mixing coefficient for v > 0.
        const : float
            Overall scale multiplier for the distribution.
        """
        from scipy.constants import Boltzmann, elementary_charge

        vmax = self.popshape["vxmax"]
        Jmax = self.popshape["jxmax"]

        kb = Boltzmann / elementary_charge
        Evib = self.mol.calculate_E_vib(Jmax, state="X", isotop=self.isotop)
        Erot = self.mol.calculate_E_rot(
            vmax + 1, Jmax + 1, isotop=self.isotop, state="X"
        )
        self.mol.calculate_tfrac(vmax + 1)
        frac = self.mol.frac[self.isotop].values

        self.alphabeta = [alpha for i in range(vmax + 1)]  # no mixing
        # Ishihara-s "mixing" of two exponents
        if self.isotop == "d":
            self.alphabeta = [beta, beta, alpha, alpha]
        if self.isotop == "h":
            # self.alphabeta = [beta, beta, alpha] # Ishihara-s pattern
            self.alphabeta = [beta, alpha, alpha]  # this seems to be better
        ab = self.alphabeta

        nxbp = [
            [
                np.exp(-Evib[v] / (kb * Tvib))
                * (
                    (1 - ab[v]) * np.exp(-Erot[v][J] / (kb * Trot1 * frac[v]))
                    + ab[v] * np.exp(-Erot[v][J] / (kb * Trot2 * frac[v]))
                )
                for v in range(vmax + 1)
            ]
            for J in range(Jmax + 1)
        ]

        self.nxbp = pd.DataFrame(nxbp) * const

    def calc_nx(self, arg):
        """Apply nuclear-spin and (2J+1) weights to ``nxbp`` to get the true X-state population.

        Calls :meth:`calc_X_bp` with the supplied parameters, then multiplies
        element-wise by the nuclear-spin statistical weight ``g_as`` and the
        ``(2J+1)`` degeneracy factor to produce the physical population::

            nx = nxbp * g_as_vector * tjpo_vector

        Sets ``self.nx`` — DataFrame of shape ``(jxlen, vxlen)``.

        Parameters
        ----------
        arg : list
            ``[Tvib, Trot1, Trot2, alpha, beta, const]`` passed directly to
            :meth:`calc_X_bp`.
        """
        # Tvib, Trot1, Trot2, alpha, const= arg
        self.calc_X_bp(*arg)

        gasvect = g_as_vector(self.nxbp.shape[0], transpose=1, isotop=self.isotop, j0=0)

        tjpo = tjpo_vector(self.nxbp.shape[0], transpose=1, j0=0)
        self.nx = self.nxbp * gasvect * tjpo

    def calc_nd(self):
        """Project the X-state population to the d-state via the R-matrix.

        Computes the coronal model d-state population::

            nd_flat = Rm2d @ nx.flatten(order='f')

        then reshapes the result back to the ``(jdlen, vdlen)`` grid.

        :meth:`calc_nx` must be called first to populate ``self.nx``.
        Using matrix multiplication is ~100× faster than element-wise loops
        (Niihama's original implementation).

        Sets ``self.nd`` — DataFrame of shape ``(jdlen, vdlen)``.
        """
        # Rm2d maps the flattened X-state population to the flattened d-state population.
        # flatten(order='f') uses Fortran (column-major) ordering, which must match the
        # column-major layout assumed by reshape_4d2d when building Rm2d.
        # Changing this ordering (e.g. to 'c') silently breaks the reconstruction.
        nmulti = self.Rm2d @ self.nx.values.flatten(order="f")
        self.nd = pd.DataFrame(
            nmulti.reshape(self.popshape["vdlen"], self.popshape["jdlen"])
        ).T

    def calc_ndx(
        self, Tvib=5000.0, Trot1=200.0, Trot2=1000.0, alpha=0.4, beta=0.1, const=1.0
    ):
        """Convenience wrapper: compute both ``nx`` and ``nd`` in one call.

        Calls :meth:`calc_nx` followed by :meth:`calc_nd`, updating
        ``self.nxbp``, ``self.nx``, and ``self.nd`` in place.
        """
        self.calc_nx([Tvib, Trot1, Trot2, alpha, beta, const])
        self.calc_nd()

    # ------------------------------------------------------------------
    # Plotting compatibility wrappers
    # (logic lives in fulcher_analyzer.plotting)
    # ------------------------------------------------------------------

    def plot_xd(self):
        """Thin wrapper — see :func:`~fulcher_analyzer.plotting.plot_xd`."""
        from .plotting import plot_xd
        return plot_xd(self)

    def plot_xd_flat(self, **kws):
        """Thin wrapper — see :func:`~fulcher_analyzer.plotting.plot_xd_flat`."""
        from .plotting import plot_xd_flat
        return plot_xd_flat(self, **kws)

    def plot_paper_compare(self, **kws):
        """Thin wrapper — see :func:`~fulcher_analyzer.plotting.plot_paper_compare`."""
        from .plotting import plot_paper_compare
        return plot_paper_compare(self, **kws)

    def plot_rtp(self):
        """Thin wrapper — see :func:`~fulcher_analyzer.plotting.plot_rtp`."""
        from .plotting import plot_rtp
        return plot_rtp(self)

    def plot_R(self):
        """Thin wrapper — see :func:`~fulcher_analyzer.plotting.plot_R`."""
        from .plotting import plot_R
        return plot_R(self)

    def plot_fcf(self):
        """Thin wrapper — see :func:`~fulcher_analyzer.plotting.plot_fcf`."""
        from .plotting import plot_fcf
        return plot_fcf(self)

    def plot_ccs(self):
        """Thin wrapper — see :func:`~fulcher_analyzer.plotting.plot_ccs`."""
        from .plotting import plot_ccs
        return plot_ccs(self)

    def plot_coronal_result(self):
        """Thin wrapper — see :func:`~fulcher_analyzer.plotting.plot_coronal_result`."""
        from .plotting import plot_coronal_result
        return plot_coronal_result(self)

    def plot_contribution(self, Tvib=7000):
        """Thin wrapper — see :func:`~fulcher_analyzer.plotting.plot_contribution`."""
        from .plotting import plot_contribution
        return plot_contribution(self, Tvib=Tvib)

    # ------------------------------------------------------------------
    # Tvib fitting and uncertainty
    # ------------------------------------------------------------------

    def coronal_fit_formula(
        self, _=[], Tvib=8000, alpha=0.57, beta=0.33, Trot1=200, Trot2=1000
    ):
        """Second-stage fit function: vary only ``Tvib``, inherit rotational parameters.

        This is the callable passed to ``scipy.optimize.curve_fit`` in
        :meth:`coronal_autofit`.  The signature accepts ``alpha``, ``beta``,
        ``Trot1``, and ``Trot2`` as formal parameters so that ``curve_fit``
        can call it with any number of arguments, but **they are immediately
        overwritten** from the completed Boltzmann fit stored in ``self.bp``.
        Only ``Tvib`` is a free fitting parameter in the current production
        workflow.

        The returned vector is normalised to unit sum and masked to the
        Q-branch lines available in ``self.bp.mask``, ensuring a consistent
        comparison with the normalised experimental data.

        Updates ``self.nxbp``, ``self.nx``, and ``self.nd`` as side effects.

        Parameters
        ----------
        _ : ignored
            Placeholder independent-variable array required by ``curve_fit``.
        Tvib : float
            X-state vibrational temperature in K (the sole free parameter).
        alpha : float
            Overwritten by ``self.bp.popt[0]``; kept in signature for
            ``curve_fit`` compatibility.
        beta : float
            Overwritten by ``self.bp.popt[1]``; kept in signature for
            ``curve_fit`` compatibility.
        Trot1 : float
            Overwritten by ``self.bp.trot1``; kept in signature for
            ``curve_fit`` compatibility.
        Trot2 : float
            Overwritten by ``self.bp.trot2``; kept in signature for
            ``curve_fit`` compatibility.

        Returns
        -------
        numpy.ndarray
            Normalised, masked d-state population vector (sums to 1).
        """
        # alpha, beta, Trot1, Trot2 are inherited from the Boltzmann fit;
        # the formal parameters above are never used — curve_fit may pass them
        # but they are overwritten here to keep the rotational distribution fixed.
        alpha = self.bp.popt[0]
        beta = self.bp.popt[1]
        Trot1 = self.bp.trot1
        Trot2 = self.bp.trot2
        self.calc_ndx(Tvib, Trot1, Trot2, alpha, beta, 1)
        # Apply the same Q-branch mask used on the experimental data, then
        # normalise so that the fit minimises the shape difference only.
        fit = flatdf(self.nd[self.bp.mask])
        return fit / fit.sum()

    def coronal_autofit(self):
        """Run the second-stage coronal fit, varying only ``Tvib``.

        Uses ``scipy.optimize.curve_fit`` with a single free parameter
        ``Tvib`` bounded to ``[3000, 15000]`` K.  Rotational parameters
        ``alpha``, ``beta``, ``Trot1``, and ``Trot2`` are fixed to the values
        from the preceding Boltzmann fit (see :meth:`coronal_fit_formula`).

        The experimental d-state population ``self.bp.nd`` and its error
        ``self.bp.nd_err`` are both normalised by the same sum before fitting,
        so the objective function is a shape comparison.

        Sets on ``self`` after the fit:

        - ``fitres`` — array of fitted parameters (currently only ``[Tvib]``).
        - ``err`` — 1-sigma uncertainties from the covariance diagonal.
        - ``tvib`` — best-fit vibrational temperature in K.
        - ``tviberr`` — 1-sigma uncertainty in K.

        Also calls :meth:`print_coronal_fit_result` and
        :meth:`calc_errorbars`.
        """
        from scipy.optimize import curve_fit

        # p = [3000,0.3,0.3,200,1000]
        # bounds = [[3000,0.1,0.01,150,800],[15000,0.99,0.7,1000,6000]]
        p = [3000]
        bounds = [[3000], [15000]]
        self.param = p
        self.bounds = bounds

        ndexp = flatdf(self.bp.nd)  # experimental data, normalized by the sum
        nd_err = flatdf(self.bp.nd_err)  # experimental error
        nd_err = nd_err / ndexp.sum()
        ndexp = ndexp / ndexp.sum()

        self.fitres, cov = curve_fit(
            self.coronal_fit_formula,
            [],
            ndexp,
            p0=p,
            bounds=bounds,
            sigma=nd_err,
            absolute_sigma=ABSOLUTESIGMA,
        )

        self.err = np.sqrt(np.diag(cov))
        self.tvib = self.fitres[0]
        self.tviberr = self.err[0]
        self.print_coronal_fit_result()
        self.calc_errorbars()

    def print_coronal_fit_result(self):
        """Print a formatted summary of the coronal fit results.

        Iterates over ``self.param`` (the list of free parameter names) and
        prints each fitted value from ``self.fitres`` together with its
        1-sigma uncertainty from ``self.err``.  Called automatically at the
        end of :meth:`coronal_autofit`.
        """
        names = ["Tvib", "alpha", "beta", "Trot1", "Trot2"]
        pr = [0, 3, 3, 0, 0]
        print(
            "\n".join(
                [
                    f"{names[i]} = {self.fitres[i]:.{pr[i]}f} +- {self.err[i]:.{pr[i]}f}"
                    for i in range(len(self.param))
                ]
            )
        )

    def get_nxbp(self, Tvib):
        """Return the synthetic X-state population for a given ``Tvib``.

        Re-runs :meth:`calc_ndx` with rotational parameters inherited from
        ``self.bp`` (same overwrite logic as :meth:`coronal_fit_formula`) and
        returns ``self.nxbp``.  Used by :meth:`calc_errorbars` to evaluate the
        population at ``Tvib ± tviberr``.

        Parameters
        ----------
        Tvib : float
            X-state vibrational temperature in K.

        Returns
        -------
        pandas.DataFrame
            ``self.nxbp`` — the updated synthetic X-state Boltzmann population.
        """
        alpha = self.bp.popt[0]
        beta = self.bp.popt[1]
        Trot1 = self.bp.trot1
        Trot2 = self.bp.trot2
        self.calc_ndx(Tvib, Trot1, Trot2, alpha, beta, 1)
        return self.nxbp

    def calc_errorbars(self, Tvib=8000):
        """Compute ±1-sigma error envelopes for the fitted d-state population.

        Evaluates the model at ``Tvib − tviberr``, ``Tvib + tviberr``, and
        ``Tvib`` (best fit) using :meth:`coronal_fit_formula` and
        :meth:`get_nxbp`.

        Sets on ``self``:

        - ``yerr_flat`` — absolute difference between the +1σ and −1σ
          normalised flat-population vectors; used for 1-D population plots.
        - ``yerr_rot`` — dict with keys ``'plus'``, ``'minus'``, ``'val'``
          holding the ``nxbp`` DataFrames at the ±1σ and best-fit ``Tvib``;
          used for rotational Boltzmann plot error shading.

        Called automatically at the end of :meth:`coronal_autofit`.
        """
        corer = np.array(
            [
                self.coronal_fit_formula([], *[t])
                for t in [self.tvib - self.tviberr, self.tvib + self.tviberr, self.tvib]
            ]
        )
        self.yerr_flat = np.absolute(corer[1] - corer[0])

        em = self.get_nxbp(self.tvib - self.tviberr)
        ep = self.get_nxbp(self.tvib + self.tviberr)
        val = self.get_nxbp(self.tvib)
        self.yerr_rot = {"plus": ep, "minus": em, "val": val}

    def plot_popx_paper(self, fontsize=11, ms=4):
        """Thin wrapper — see :func:`~fulcher_analyzer.plotting.plot_popx_paper`."""
        from .plotting import plot_popx_paper
        return plot_popx_paper(self, fontsize=fontsize, ms=ms)
