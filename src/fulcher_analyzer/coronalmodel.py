import numpy as np, pandas as pd
from os.path import join, abspath
from ._constants import package_directory
from .molecular_constants import MolecularConstants  # noqa: F401  (re-export)
from .intensity_io import read_intensities, write_intensities  # noqa: F401  (re-export)
from .boltzmann import (  # noqa: F401  (re-export)
    expsum,
    two_t_all_v,
    plot_n_all_v,
    fittext,
    BoltzmannPlot,
    ABSOLUTESIGMA,
)
from ._utils import (  # noqa: F401  (re-export)
    delta_kro,
    g_as,
    g_as_vector,
    tjpo_vector,
    reshape_4d2d,
    flatdf,
    flatdf_1,
    figsize,
)

MOLECULAR_DATA_FOLDER = abspath(join(package_directory, "..", "..", "data_molecular"))
DATA_FOLDER = abspath(join(package_directory, "..", "..", "data"))


# ==================================================
#
#  Coronal Model useful functions
#
# ==================================================


class CoronaModel:
    """ 
    Coronal Model class 
    Based on the Ishihara-s code.

    Parameters
    ----------
    isotop: string
         'd' or 'h' 
    """

    def __init__(self, bp: BoltzmannPlot):
        """
        Initialize
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
        """ 
        Prepare standard plot styles 
        """
        self.style_coronaplot = [
            {
                "capsize": 0.8 * ms,
                "capthick": 0.4,
                "elinewidth": 0.4,
                "fmt": i,
                "ecolor": j,
                "markeredgecolor": j,
                "color": j,
                "label": k,
                "ms": l,
                "lw": 0.6,
            }
            for i, j, k, l in zip(
                ["o-.", "s-."],
                ["k", "r"],
                ["experiment", "reconstructed"],
                [ms, 0.7 * ms],
            )
        ]

        self.style_rovib = [
            {
                "lw": 0.8,
                "marker": i,
                "mfc": j,
                "c": "k",
                "ls": "-.",
                "ms": ms,
                "mew": k,
                "label": f"v={s}",
            }
            for s, (i, j, k) in enumerate(
                zip(
                    ["o", "o", "x", "s"],
                    ["k", "w", "k", "k"],
                    np.array([1, 2, 2, 1]) * 0.5,
                )
            )
        ]

    def prep_constants(self):
        """
        Prepare the nessecary constanst
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

    def calculate_e_cross(self, Te=15):
        """ 
        Dublicate of self.ccs_formula().
        For fitting vibro population, which is calculated from simulated rot. pop. and summed up.

        Calculate electron impact crossections for given electron temperature Te [eV]
        Here Ee_vib is excited state vibrational energy,the d-state. 
        E_vib is X-state vibrational energy.
        Matrix shape is same as Franck-Condon data shape.

        Parameters
        ----------
        Te: float
            electron temperature for Coronal Model
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
        """ 
        Function for fitting vibrational population. 
        This is Coronal Model, used by Ishihara 
        (but population is summed by rotational quantum number).
        Second variable "_" is for independent variable, 
        required by scipy.optimize.curve_fit. In this case it is not needed.

        Parameters
        ----------
        T_vib: float
            vibrational temperature
        fit: bool
            True - return normalized population, False - return nv

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

    """
    :math:`n_{dv'}\sum{A^{dv'}_{av''}} \propto \sum_{v=0}^{vmax} R_{Xv}^{dv'}
         \exp{\left( -\frac{E_{vib}^{X}(v)}{kT_{vib}^{X}} \right)}`
    """

    def fit_vibro_ishi(self):
        """ 
        Ishihara-s vibro fit
        Fit vibrational population, ignoring rotational distribuiton to get Tvib.
        See formula (4.5) from Ishihara-s Master thesis, page 36. There R-coefficient 
        is not rotationally resolved and hence does not include the branching ratio.

        :math:`n_{dv\'} \sum{A^{dv\'}_{av\'\'}} \propto \sum_{v=0}^{vmax} R_{Xv}^{dv\'}
        \exp{\left( -\\frac{E_{vib}^{X}(v)}{kT_{vib}^{X}} \\right)}`
        
        """
        from scipy.optimize import curve_fit

        tvib, cov = curve_fit(self.f_vibro, [], self.bp.nd_vibrofit, 10000)
        self.tvib = tvib[0]
        self.tvib_cov = cov
        self.tvib_err = np.sqrt(np.diag(self.tvib_cov))[0]
        print(f"Tvib = {tvib[0]:.0f} +- {self.tvib_err:.0f} K")

    def plot_fit_ishi(self, **kws):
        """ 
        "Nice" plot of the vibro fit and 'gauge' for Ishihara-s fit 
        """
        import matplotlib.pyplot as plt

        fig = plt.gcf()
        fig.set_size_inches([7, 6])

        tvs = np.arange(1, 31) * 1000
        colors = plt.cm.jet(np.linspace(0, 1, len(tvs)))
        gls = []

        for i, t in enumerate(tvs):
            nv = self.f_vibro("", t, False)
            (ll,) = plt.plot(nv.sum(axis=0), ".-", c=colors[i], label=f"{t//1000}")
            gls.append(ll,)

        try:
            tvib = kws.get("tvib", self.tvib)
        except:
            tvib = kws.get("tvib", 1000)

        nv = self.f_vibro("", tvib, False)

        (l1,) = plt.plot(
            nv.sum(axis=0), "s-", c="r", ms=10, label=f"$T_{{vib}} = {tvib:.0f}$K"
        )
        nde = self.bp.nd_vibrofit
        (l2,) = plt.plot(nde / nde[0] * nv.sum(axis=0)[0], "ko-", label="data")

        plt.locator_params(nbins=4)
        ax = plt.gca()
        ax.text(0.98, 1.02, "T/1000K", transform=ax.transAxes)
        pl0 = plt.legend(
            handles=gls, loc=1, prop={"size": 7}, bbox_to_anchor=[1.13, 1.01]
        )
        ax.add_artist(pl0)  # first legned we must add manually
        pl1 = plt.legend(handles=[l1, l2])
        plt.xlabel("vibrational q.n.")
        plt.ylabel("relative pop. of $d^3\Pi_u$")

    # ==================================================
    #
    #  Coronal Model Functions from Niihama and Ishihara.
    #
    # ==================================================

    def branching(self, JX, Jd):
        """ 
        branching ratio 

        Parameters
        ----------
        JX: int
            X-state rotational quantum number
        Jd: int
            d-state rotational quantum number
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
        """
        Calculate branching ratio matrix, `axd`
        Jd = index + 1 (start from 1, not 0, range(1,Jdmax))
        Jx = index

        Parameters
        -----------
        JXmax: int
            Maximum value for X-state rotational quantum number

        Jdmax: int
            Maximum value for d-state rotational quantum number
        """
        return pd.DataFrame(
            [
                [self.branching(JX, Jd) for Jd in range(1, Jdmax + 1)]
                for JX in range(JXmax + 1)
            ]
        )

    def calc_delta(self, axd):
        """ 
        Calculate Kronecker-s delta matrix, 
        isotop = ['d','h']

        Parameters
        ----------
        axd: array
        isotop: string
            'd' or 'h'
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
        """ 
        Collisional cross section formula
        Te in [eV]

        Parameters
        ----------
        vX: int
            X-state vibrational quantum number
        vd: int
            d-state vibrational quantum number
        Te: float
            electron temperature [eV]
        """
        evd = self.mol.calculate_E_vib(20, state="d3", isotop=self.isotop)
        evx = self.mol.calculate_E_vib(20, state="X", isotop=self.isotop)
        return np.exp(-((evd[vd] - evd[0]) - (evx[vX] - evx[0])) / Te)

    def calc_rtp(self):
        """ 
        Calculate rtp (radiation transition probability), like in Ishihara-s code
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
        """ 
        Calculate electron impact excitation rate coefficient

        Parameters
        ----------

        vX: int
            X-state vibrational quantum number
        JX: int
            X-state rotational quantum number
        vd: int
            d-state vibrational quantum number
        Jdind: int
            index for d-state rotational quantum number
            Jdind = Jd-1, Jd starts from 1 in `self.rtp`
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
        """ 
        Prep proper corona.
        setup X and d states shapes, calculate R-matrix

        Parameters
        ----------
        load: bool
            True try to load previously calculated matrix to save time.
            CAUTION: must recalculate this matrix if the model changed.
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
        """ 
        Set population shape for d and X levels
        NOTE: jd here is an index. Actual JD is JD+1
        There is no Jd=0 for d-state. But numpy matrix indexing starts from 0.
        That is why max(jx) >= max(jd) + 1

        Parameters
        ----------
        limits: array
            limits = ['vx','jx']
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
        """ 
        Calculate R matrix for given isotop
        If formula changed, RECALCULATE! argument `load = False`   

        NOTE! Jd starts from 1. Index 0 in Jd axis corresponds to Jd=1

        Parameters
        -----------
        load: bool
            True - load previously calculated matrix, False - recalculate. 
        
        """
        vxmax = self.popshape["vxmax"]
        jxmax = self.popshape["jxmax"]
        vdmax = self.popshape["vdmax"]
        jdmax = self.popshape["jdmax"]

        fname = f"Rmatrix_{vxmax}_{jxmax}_{vdmax}_{jdmax}_{self.isotop}.npy"
        fpth = join(MOLECULAR_DATA_FOLDER, f"{fname}")
        if load:
            try:
                self.Rm = np.load(fpth)
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
        np.save(fpth, Rmatrix)
        self.make_rmatrix_2d()

    def make_rmatrix_2d(self):
        """ 
        Reshape R matrix (electron impact excitation rate)
        """
        self.Rm2d = reshape_4d2d(self.Rm).T  # transpose for correct orientation

    def check_rmatrix_indexing(self):
        """ 
        Check Rmatrix indexing
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

    def calc_X_bp(
        self, Tvib=5000.0, Trot1=200.0, Trot2=1000.0, alpha=0.1, beta=0.1, const=1.0
    ):
        """ 
        Calculate synthetic Boltzmann plot of the X-state (nx*g_as*(2J+1))
        with one Tvib and two Trot
        Jmax is limited by the Franck-Condon data
        higher vmax have smaller contribution to d-state population. 
        Temperatures in K

        Parameters
        ----------

        Tvib: float
            X-state vibrational temperature
        Trot1: float
            X-state low rotational temperature
        Trot2: float
            X-state high rotational temperature
        alpha: float
            exponent mixing coefficient for v=0
        beta: float
            exponent mixing coefficient for v>0
        const: float
            constant multiplyer for the distribution
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
        """
        Calculate X-state population

        Parameters
        ----------

        arg: list
            arg = [5000,200,1000,0.1,1,3,15]
            list of parameters for :func:`~self.calc_X_bp`
        """
        # Tvib, Trot1, Trot2, alpha, const= arg
        self.calc_X_bp(*arg)

        gasvect = g_as_vector(self.nxbp.shape[0], transpose=1, isotop=self.isotop, j0=0)

        tjpo = tjpo_vector(self.nxbp.shape[0], transpose=1, j0=0)
        self.nx = self.nxbp * gasvect * tjpo

    def calc_nd(self):
        """ 
        Calculate d-state population using Corona Model
        X-state population must be calculated first
        Using numpy matrix multiplication for this, same as Ishihara.
        This is ~ 100 times faster compared to using formula in loops,
        as in Niihama-kun-s code.
        """
        nmulti = self.Rm2d @ self.nx.values.flatten(order="f")
        self.nd = pd.DataFrame(
            nmulti.reshape(self.popshape["vdlen"], self.popshape["jdlen"])
        ).T

    def calc_ndx(
        self, Tvib=5000.0, Trot1=200.0, Trot2=1000.0, alpha=0.4, beta=0.1, const=1.0
    ):
        """ 
        Construct synthetic population of X-state with calc_nx()
        Calculate population of d-state from nX using Corona model calc_nd()
        """
        self.calc_nx([Tvib, Trot1, Trot2, alpha, beta, const])
        self.calc_nd()

    def plot_xd(self):
        """ plot synthetic nX and nd"""
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1)
        fig.set_size_inches([5, 5])

        [axs[0].plot(self.nx[i], label=f"v={i}") for i in range(self.nx.shape[1])]
        [axs[1].plot(self.nd[i], label=f"v={i}") for i in range(self.nd.shape[1])]
        [ax.legend(fontsize=10) for ax in axs]
        axs[0].text(0.5, 0.8, "$X^1\Sigma^{+}_{g}$", transform=axs[0].transAxes)
        axs[1].text(0.5, 0.8, "$d^3\Pi_{u}$", transform=axs[1].transAxes)

    def plot_xd_flat(self, **kws):
        """
        Plot flattened X and d populations
        """
        import matplotlib.pyplot as plt
        from math import ceil

        ax = kws.get("ax", plt.gca())

        ndeflat = self.bp.nd_sc.values.flatten(order="f")
        ndflat = self.nd.values.flatten(order="f")

        mask = self.bp.mask.values.flatten(order="f")
        x_ndem = np.arange(mask.shape[0])[mask]
        ndemflat = ndeflat[mask]

        plt.sca(ax)
        plt.plot(ndeflat / ndeflat.sum(), "ko-", label="exp-synth")
        plt.plot(x_ndem, ndemflat / ndeflat.sum(), "C3s", label="data")
        plt.plot(ndflat / ndflat.sum(), "C2x-.", label="corona")
        plt.legend()
        ax.set_xlim(-1, ceil(self.nd.shape[0] * self.nd.shape[1] / 5) * 5)

    def plot_paper_compare(self, **kws):
        """
        Plot comparison of experimental population of d-state
        with Corona-Model reconstructed one (p.39 fig. 4.24 Ishihara-s thesis)
        (or Figures 11 and 12 from JQSRT-2021 paper)
        """
        import matplotlib.pyplot as plt

        ax = kws.get("ax", plt.gca())

        "nd - n(d3), nde - nd experimental, ndes - nd experimental synthetic (fit)"
        ndeflat = flatdf(self.bp.nd[self.bp.mask])  # experimental data
        nd_err = flatdf(self.bp.nd_err)  # experimental error
        ndesflat = flatdf(self.bp.nd_sc[self.bp.mask])  # boltzmann fit of d-state
        ndflat = flatdf(self.nd[self.bp.mask])  # corona reconstruction of d-state
        # normalize bu the sum()
        ndflat = ndflat / ndflat.sum()  # corona
        nd_err = nd_err / ndeflat.sum()  # experimental error, ORDER MATTERS!
        ndeflat = ndeflat / ndeflat.sum()  # experiment
        ndesflat = ndesflat / ndesflat.sum()  # boltzmann fit

        plt.sca(ax)
        # plt.plot(ndeflat, "ko-", label="exp-data")
        ms = kws.get("ms", 4)
        self.prep_style(ms=ms)
        style = self.style_coronaplot
        x = np.arange(len(ndeflat))
        plt.errorbar(x, ndeflat, yerr=nd_err, **style[0])
        showbp = kws.get("showbp", False)
        if showbp:
            plt.plot(ndesflat, "C1o-.", label="BP-fit")
        yerr = kws.get("yerr", [])
        if not len(yerr):
            plt.plot(ndflat, "rs:", label="corona")
        else:
            plt.errorbar(x, ndflat, yerr=yerr, **style[1])

        plt.legend()

        plt.xticks(np.arange(0, len(ndflat), 1))
        nms = self.bp.qnames[self.bp.mask]
        ax.set_xticklabels(flatdf(nms), rotation=90)
        ax.set_xlabel("Q-branch transition [QN'(v'-v'')]")
        ax.set_ylabel(r"$\mathrm{n_{d v^{\prime} N^{\prime}}}$ [a.u.]")

    def plot_rtp(self):
        """ 
        Plot rtp 
        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        im = plt.imshow(
            self.rtp, origin="lower", cmap=plt.cm.seismic, norm=mpl.colors.LogNorm()
        )
        plt.colorbar(im)
        ax = plt.gca()
        ax.set_xlabel("J'-1 (d)")
        ax.set_ylabel("J (X)")

    def plot_R(self):
        """
        plot R-matrix 
        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        # shp = [self.popshape[f"{i}len"] for i in ["vd", "jd", "vx", "jx"]]
        shp = self.rshapelist
        plot_rmatrix(self.Rm2d, shp, text="R-matrix")
        plt.gcf().set_size_inches([9, 9])

    def plot_fcf(self):
        """
        Plot FCF 
        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        fcf = self.fcf[:4, :4]
        im = plt.imshow(
            fcf, origin="lower", cmap=plt.cm.seismic, norm=mpl.colors.LogNorm()
        )
        plt.colorbar(im)
        ax = plt.gca()
        ax.set_xticks(range(fcf.shape[0]))
        ax.set_yticks(range(fcf.shape[1]))
        ax.set_xlabel("$v$'' ($a^3\Sigma_g^+$)")
        ax.set_ylabel("$v$' ($d^3\Pi_u$)")

    def plot_ccs(self):
        """
        Plot ccs 
        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        ccs = self.ccs.values[:4, :4]
        im = plt.imshow(ccs, origin="lower", cmap=plt.cm.seismic)
        plt.colorbar(im)
        ax = plt.gca()
        ax.set_xticks(range(ccs.shape[0]))
        ax.set_yticks(range(ccs.shape[1]))
        ax.set_xlabel("$v$'' ($a^3\Sigma_g^+$)")
        ax.set_ylabel("$v$' ($d^3\Pi_u$)")

    def plot_coronal_result(self):
        """
        Plot two panels with synthetic population and with data only
        """
        import matplotlib.pyplot as plt
        from math import ceil

        self.coronal_fit_formula([], *self.fitres)  # update self.nxbp from self.fitres
        fig, axs = plt.subplots(2, 1)
        self.plot_xd_flat(ax=axs[0])
        self.plot_paper_compare(ax=axs[1])
        fig.set_size_inches([12, 10])

    def plot_contribution(self, Tvib=7000):
        """
        Plot contribution of X-state to d-state population
        """
        import matplotlib.pyplot as plt
        from math import ceil
        from scipy.constants import Boltzmann, elementary_charge

        kb = Boltzmann / elementary_charge

        self.calculate_e_cross()
        ccs = self.ccs.values
        fcf = self.fcf
        E_vib = self.E_vib

        v_range = 15
        vv_range = 4
        nvv = np.zeros([v_range, vv_range])
        for v in range(v_range):
            for vv in range(vv_range):
                nvv[v, vv] = (
                    ccs[v, vv]
                    * fcf[v, vv]
                    * np.exp(-(E_vib[v] - E_vib[0]) / (Tvib * kb))
                    / self.Asum[vv]
                    * 1e9
                )

        mks = ["o", "s", "^", ">"]
        for v in range(v_range):
            if v < 4:
                plt.plot(
                    nvv[v],
                    "-",
                    color=plt.cm.hsv(v / 4),
                    marker=mks[v],
                    ms=10,
                    label=f"v={v}",
                )
                # print(sum(nvv[v, :] / sum(nvv)))
            else:
                plt.plot(nvv[v], ".--", c="gray", alpha=0.5)
                # print(sum(nvv[v, :] / sum(nvv)))
        plt.plot(nvv[14], ".--", c="gray", label="v=4-14", alpha=0.5)

        plt.xticks(np.arange(0, 4, 1))
        plt.xlim(-0.1, 3.1)
        plt.ylim(-0.3, ceil(nvv.max()) + 0.5)
        plt.xlabel(r"Vibrational Number $v$'")
        plt.ylabel(r"Contribution to $n_{d}(v$'$)$ from $n_{X}(v)$")
        plt.legend(loc=1, bbox_to_anchor=[1, 0.85])
        plt.gcf().set_size_inches([8, 5])
        ax = plt.gca()
        ax.text(
            0,
            1.03,
            f"$T_{{vib}}$={Tvib}K, isotop={self.isotop}",
            transform=ax.transAxes,
        )

    def coronal_fit_formula(
        self, _=[], Tvib=8000, alpha=0.57, beta=0.33, Trot1=200, Trot2=1000
    ):
        """ 
        Fit formula for Coronal Model. 
        Updates nxbp based on the input, returns flat array with d-state population
        for avaliable Q-branch lines.

        Parameters
        ----------

        Tvib: float
            X-state vibrational temperature
        Trot1: float
            X-state low rotational temperature
        Trot2: float
            X-state high rotational temperature
        alpha: float
            exponent mixing coefficient for v=0
        beta: float
            exponent mixing coefficient for v>0
        const: float
            constant multiplyer for the distribution
        """
        alpha = self.bp.popt[0]
        beta = self.bp.popt[1]
        Trot1 = self.bp.trot1
        Trot2 = self.bp.trot2
        self.calc_ndx(Tvib, Trot1, Trot2, alpha, beta, 1)
        fit = flatdf(self.nd[self.bp.mask])
        return fit / fit.sum()

    def coronal_autofit(self):
        """
        Run all the neccessary steps with default settings
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
        """
        Print coronal fit result
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
        """ 
        Updated CoronaModelIshihara-s nxbp for given Tvib, return it
        """
        alpha = self.bp.popt[0]
        beta = self.bp.popt[1]
        Trot1 = self.bp.trot1
        Trot2 = self.bp.trot2
        self.calc_ndx(Tvib, Trot1, Trot2, alpha, beta, 1)
        return self.nxbp

    def calc_errorbars(self, Tvib=8000):
        """
        Calculate error bars for flat and full population plots
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
        """
        Plot X-state population (figs 9-10)
        """
        import matplotlib.pylab as plt

        self.prep_style(ms=ms)
        kws = self.style_rovib
        [
            plt.fill_between(
                self.EX[i],
                self.yerr_rot["minus"][i] / self.nxbp[0][0],
                self.yerr_rot["plus"][i] / self.nxbp[0][0],
                color="gray",
                alpha=0.5,
                ec="none",
            )
            for i in self.yerr_rot["minus"]
        ]
        [
            plt.plot(self.EX[i], self.nxbp[i] / self.nxbp[0][0], **kws[i])
            for i in self.nxbp
        ]
        plt.fill_between([], [], [], color="gray", alpha=0.5, label="error", ec="none")

        plt.yscale("log")
        plt.xlabel("Rotational Energy [eV]")
        plt.ylabel(
            r"$\mathrm{\frac{n_{XvN}}{(2N+1)\,g_{\mathrm{as}}^{N}}}$ [a.u.]",
            fontsize=fontsize + 2,
        )

        plt.legend()
        ax = plt.gca()
        ax.text(self.EX.values[0, 0] + 0.02, 0.95, "N=0", fontsize=fontsize - 2)
        ax.text(
            self.EX.values[-1, 0] * 1.05,
            (self.yerr_rot["minus"][0] / self.nxbp[0][0]).values[-1] + 0.02,
            f"N={self.popshape['jxmax']}",
            ha="center",
            fontsize=fontsize - 2,
        )
        plt.xlim(-0.05, 1.0)

        plt.gcf().set_size_inches(figsize(8))


# ==================================================
#
#  Coronal Model Functions without any data references
#  Put here, outside of the class.
# ==================================================


def plot_rmatrix(Rm, shapes, text="R-matrix"):
    """
    Plot 2d R matrix for inspection 
    vdl, jdl, vxl, jxl = shapes
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    vdl, jdl, vxl, jxl = shapes
    xticks = np.arange(vxl + 1) * jxl
    yticks = np.arange(vdl + 1) * jdl
    # Color Map with Under and Over defined
    cm = plt.cm.seismic
    cm.set_under("#736200")
    cm.set_over("#15ad1a")

    im = plt.imshow(Rm, origin="lower", cmap=cm, norm=mpl.colors.LogNorm())
    ax = plt.gca()
    # Colorbar in a divider axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(im, extend="both", cax=cax)

    # Text
    plt.sca(ax)
    ax.set_aspect(1)
    ax.set_facecolor("#a8a594")
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    plt.xlabel("$v\cdot J$ ($X^1\Sigma^+_g$)")
    plt.ylabel("$v$'$\cdot (J$'$-1)$ ($d^3\Pi_{u}$)")
    plt.xlim(-1, xticks[-1])
    plt.ylim(-1, yticks[-1])
    plt.grid(c="k")
    ax.text(
        -0.12,
        0,
        f"$vd\cdot Jd = {vdl}\cdot {jdl}$",
        transform=ax.transAxes,
        rotation=90,
    )
    ax.text(
        -0.12,
        0.75,
        f"$\\nexists Jd = 0$, $Jd>0$!!!",
        transform=ax.transAxes,
        rotation=90,
    )
    ax.text(-0.12, -0.1, f"$vx\cdot Jx = {vxl}\cdot {jxl}$", transform=ax.transAxes)
    ax.text(0.7, -0.1, f"{text}", transform=ax.transAxes)

    [
        ax.text(vxl * jxl + 2, j + jdl / 2, i, va="center", ha="center")
        for i, j in enumerate(yticks[:-1])
    ]
    [
        ax.text(j + jxl / 2, vdl * jdl + 2, i, va="center", ha="center")
        for i, j in enumerate(xticks[:-1])
    ]


def set_tick_size(ax, *size):
    """ 
    Set matplotlib axis tick sizes
    For some reason the length from style is ignored.
    """
    width_major, length_major, width_minor, length_minor = size
    ax.xaxis.set_tick_params(width=width_major, length=length_major, which="major")
    ax.xaxis.set_tick_params(width=width_minor, length=length_minor, which="minor")
    ax.yaxis.set_tick_params(width=width_major, length=length_major, which="major")
    ax.yaxis.set_tick_params(width=width_minor, length=length_minor, which="minor")
