"""
Free helper functions and BoltzmannPlot class for two-temperature Boltzmann analysis.
"""
import numpy as np
import pandas as pd
from os.path import join, abspath

from ._constants import package_directory
from .molecular_constants import MolecularConstants
from ._utils import flatdf, figsize

ABSOLUTESIGMA = False
MOLECULAR_DATA_FOLDER = abspath(join(package_directory, "..", "..", "data_molecular"))


def expsum(e, T1, T2, a, constant):
    """
    Mixing of two Boltzman distributions
    """
    from scipy.constants import Boltzmann, elementary_charge

    kb = Boltzmann / elementary_charge
    return np.log(
        constant * ((1 - a) * np.exp(-e / (kb * T1)) + a * np.exp(-e / (kb * T2)))
    )


def two_t_all_v(Erot, *param, **kws):
    """
    Constract two exponent distribution for all vibrational transitions
    populate DataFrame
    a, b - exponent multiplyers for v=0 and others, v>0 assumed to have same multiplier
    T1, T2 - temperatures of the two exponents
    cs - tuple of constants for all v.     
    """
    a, b, T1, T2 = param[:4]
    cs = param[4:]
    aps = [a] + [b for i in range(Erot.shape[1] - 1)]
    # temperature fractions
    mol = MolecularConstants()
    mol.calculate_tfrac(
        Erot.shape[1], norm=True
    )  # norm=True is essential! Otherwise produces T2*2.
    isotop = kws.get("isotop", "d")
    tfrac = mol.frac[isotop].values
    ns = pd.DataFrame(
        [
            expsum(Erot[i], tc * T1, tc * T2, a, c)
            for tc, a, i, c in zip(tfrac, aps, Erot, cs)
        ]
    ).T
    melt = kws.get("melt", True)

    if melt:
        return flatdf(ns)
    else:
        return ns


def plot_n_all_v(n, E, **kws):
    """
    input: experimental populations in [2D] DataFrame with missing values as NaN
    Energy - full DataFrame with theoretical rotational energy E
    cycles through columns in n (corresponding to vibrational level), removes nans,
    and plots
    """
    from matplotlib.pyplot import plot

    nomarker = kws.get("nomarker", False)
    noline = kws.get("noline", False)
    mkws = [
        {"label": f"v'={l}", "marker": m, "ls": "--"}
        for l, m in zip(range(4), ["s", "o", "d", "x"])
    ]
    if nomarker:
        mkws = [{"label": f"v'={l}", "ls": "--"} for l in range(4)]
    if noline:
        mkws = [
            {"label": f"v'={l}", "marker": m, "ls": ""}
            for l, m in zip(range(4), ["s", "o", "d", "x"])
        ]

    log = kws.get("log", True)
    for p, (i, j) in enumerate(zip(n, E)):
        m = ~pd.isna(n[i])
        if log:
            plot(E[j][m], np.log(n[i][m]), **mkws[p])
        else:
            plot(E[j][m], n[i][m], **mkws[p])


def fittext(p):
    """
    Return text for matplotlib plot with fitting parameters, a,b,T1,T2 from two_t_all_v()
    """
    return (
        f"$\\alpha={p[0]:.2f}$ $\\beta={p[1]:.2f}$ $T_1={p[2]:.0f}$K $T_2={p[3]:.0f}$K"
    )


# ==================================================
#
#  Boltzmann Plot and fit class
#
# ==================================================


class BoltzmannPlot:
    """"
    Analysis of Boltzmann distribution
    """

    def __init__(self, inte, isotop):
        """ 
        Supply intensity DataFrame

        Parameters
        ----------
        inte: 

        isotop: string
             'd' or 'h' for deuterium and hydrogen, respectively.
        """

        self.name = "Analysis of Boltzmann distribution"
        self.inte, self.interr = inte
        isotops = ["h", "d"]
        if not isotop in isotops:
            raise ValueError(f"isotop must be one of the {isotops}")
        self.isotop = isotop
        self.load_wavelength_data()
        self.trim_wavelength()
        self.prep_mol()
        self.bplot_constants()
        self.calculate_boltzmann()
        self.prep_fit()
        self.set_style()

    # TODO: Move to MolecularConstants class
    def load_wavelength_data(self):
        """ 
        Load wavelength data for Q-branch for H and D 
        """
        # Deuterium, data in the file is in [cm^{-1}], 800 is nan
        wld = np.loadtxt(
            join(MOLECULAR_DATA_FOLDER, "fulcher-α_band_wavenumber_D2.txt")
        )
        wld = pd.DataFrame(1 / (wld * 1e-7))  # wavenumber [cm-1] -> wavelength [nm]
        wld[wld > 800] = np.nan
        self.wld = wld
        wlh = pd.DataFrame(
            np.loadtxt(join(MOLECULAR_DATA_FOLDER, "fulcher-α_band_wavelength.txt"))
        )
        self.wlh = wlh

    def trim_wavelength(self):
        """
        Match wavelength DataFrame shape with Intensities Dataframe shape.
        """
        if self.isotop == "d":
            wl = self.wld
        elif self.isotop == "h":
            wl = self.wlh

        if wl.shape < self.inte.shape:
            raise ValueError(
                "Wavelength data shape is smaller than intensity data shape. "
                "Either reduce intensity DataFrame size, or add missing Wavelength data."
            )

        # drop rows
        w = wl.drop(wl.index[range(wl.shape[0] - 1, self.inte.shape[0] - 1, -1)])
        # drop columns
        self.wl = w.drop(
            columns=list(range(wl.shape[1] - 1, self.inte.shape[1] - 1, -1))
        )

    def prep_mol(self):
        """
        load molecular data class and adjust DataFrame shapes 
        """
        self.mol = MolecularConstants()
        self.mol.calculate_tfrac(self.inte.shape[1], norm=True)
        self.mol.calculate_all_E_rot(*self.inte.shape[::-1])

        if self.isotop == "d":
            self.Ed = self.mol.EdD
            self.EX = self.mol.ExD
            self.Aev = np.diag(self.mol.AD)
        elif self.isotop == "h":
            self.Ed = self.mol.EdH
            self.EX = self.mol.ExH
            self.Aev = np.diag(self.mol.AH)

    def bplot_constants(self):
        """
        Calculated 2J+1, g_as, and Franck-Condon vectors 
        """
        # D2
        # vector of  2J + 1, same for D2 and H2
        self.v2jp1 = np.array([2 * (J + 1) + 1 for J in range(self.inte.shape[0])])[
            :, None
        ]
        if self.isotop == "d":
            # vector g_as
            self.vg = np.array(
                [6 - 3 * np.mod((J + 1), 2) for J in range(self.inte.shape[0])]
            )[:, None]
            # Franck-Condon factors
            self.fc = np.array([2.3387e7, 1.8841e7, 1.4795e7, 1.1276e7])

            # So I'm just broadcasting Aev for electronic levels into AevJ for ro-vib with same vals
            self.AevJ = np.array(
                [[i for i in self.Aev] for J in range(self.inte.shape[0])]
            )
        # H2
        elif self.isotop == "h":
            # vector g_as
            self.vg = np.array(
                [np.mod((J + 1), 2) * 2 + 1 for J in range(self.inte.shape[0])]
            )[:, None]
            # Franck-Condon factors (Relative?)
            self.fc = np.array([0.93016, 0.79701, 0.67139])
            # For H2 self.Aev[:-1] because Hydrogen discharge had only v = 0,1,2.
            # TODO: automatically correct shape if intensities include values up to v = 3.
            self.AevJ = np.array(
                [[i for i in self.Aev[:-1]] for J in range(self.inte.shape[0])]
            )

    def calculate_boltzmann(self):
        """ 
        Calculate population from intensity.
        nd: population of the upper state d3
        nd_bol: nd/(2J+1)/g_as for Boltzmann plot
        nd_rel: nd_bol/nd_bol[0][0]
        """
        # nd = I*lambda / (hc) / A, so formula before is true up to a constant hc
        self.nd = self.inte * self.wl / self.AevJ
        self.nd_bol = self.inte * self.wl ** 4 / self.v2jp1 / self.vg / self.fc
        norm = self.nd_bol[0].dropna().iloc[0]
        self.nd_rel = self.nd_bol / norm
        self.mask = ~pd.isna(self.nd_rel)
        m = self.mask
        self.qnames = pd.DataFrame(
            [[f"Q{k+1}({i}-{i})" for k, j in enumerate(m[i])] for i in m]
        ).T

        self.nd_err = self.interr * self.wl / self.AevJ
        self.nd_bol_err = self.nd_err / self.v2jp1 / self.vg / self.fc
        self.relerr = self.nd_err / self.nd

    def plot_boltzmannn(self):
        """ 
        Plto Boltzmann plot 
        """
        import matplotlib.pyplot as plt

        plot_n_all_v(self.nd_rel, self.Ed[self.mask], log=False)
        plt.xlabel("E, ev")
        plt.ylabel(r"$n/(2J+1)/g_{as}$")
        plt.legend()
        plt.yscale("log")

    def prep_fit(self):
        """ 
        prep 2-temperature fit for all vibrational q.n. at the same time 
        """
        # initial guess
        self.param = [6.9e-1, 6.0e-1, 200, 1700] + [
            1 for i in range(self.inte.shape[1])
        ]
        # bounds
        self.bounds = [
            [0.05, 0.05, 100, 1000] + list(0.1 for i in range(self.inte.shape[1])),
            [1, 1, 800, 3000] + list(1.5 for i in range(self.inte.shape[1])),
        ]

    def plot_fit(self, init=True):
        """ 
        plot nd_rel and resuls for initial guess 
        
        Parameters
        ----------
        init: bool
            True - use initial data, False - use fitting result.
        """
        import matplotlib.pyplot as plt

        if init:
            param = self.param
        else:
            param = self.popt

        ns = two_t_all_v(self.Ed[self.mask], *param)
        nd = flatdf(self.nd_rel)
        plt.plot(np.log(nd), "ks-")
        plt.plot(ns, "C1o")
        ax = plt.gca()
        ax.text(0, 1.05, fittext(param), transform=ax.transAxes)
        plt.xlabel("E, ev")
        plt.ylabel(r"$\ln{\left( n/(2J+1)/g_{as}\right)}$, a.u.")

    def print_fit_result(self):
        """
        Print fit result 
        """
        err = self.err
        names = ["alpha", "beta", "Trot1", "Trot2"]
        pr = [3, 3, 0, 0]
        print(
            "\n".join(
                [
                    f"{n} = {self.popt[i]:.{pr[i]}f} +- {err[i]:.{pr[i]}f}"
                    for i, n in enumerate(names)
                ]
            )
        )

    def fit_boltzmann(self):
        """
        Fit boltzmann distribution with 2-temp and 
        all vibrational quantum numbers at onece.
        """
        from scipy.optimize import curve_fit

        # make sure that temperature ratios are normalized
        self.mol.calculate_tfrac(self.inte.shape[1], norm=True)

        self.popt, self.pcov = curve_fit(
            two_t_all_v,
            self.Ed[self.mask],
            np.log(flatdf(self.nd_rel)),
            p0=self.param,
            bounds=self.bounds,
            sigma=flatdf(self.relerr),
            absolute_sigma=ABSOLUTESIGMA,
        )
        self.calc_nd_synth()

        self.trot1 = self.popt[2]
        self.trot2 = self.popt[3]
        self.alpha = self.popt[0]
        self.beta = self.popt[1]

        self.err = np.sqrt(np.diag(self.pcov))

    def calc_nd_synth(self):
        """
        Afer successfull boltzmann plot fit, calculate synthetic ro-vib population.
        This will "fill" the missing lines intensities.
        """
        self.nd_bol_synth = np.exp(two_t_all_v(self.Ed, *self.popt, melt=False))
        self.nd_synth = self.nd_bol_synth * self.v2jp1 * self.vg

    def plot_fit_nice(self):
        """
        plot fit result 
        plot Boltzamnn distribution with preformated output using plot_n_all
        data: nd_rel(Ed) here rel for relative, defined up to a multiplier constant
        fit: ns(Ed), s for synthetic
        """
        import matplotlib.pyplot as plt

        ns = two_t_all_v(self.Ed, *self.popt, melt=False)
        plot_n_all_v(np.exp(ns), self.Ed, log=False, nomarker=True)
        plt.gca().set_prop_cycle(None)
        plot_n_all_v(self.nd_rel, self.Ed, log=False, noline=True)
        plt.yscale("log")
        plt.xlabel("E, ev")
        plt.ylabel(r"$n/(2J+1)/g_{as}$, a.u.")

    def calc_nd_const(self):
        """ 
        Calculate constant for synthetic nd_synth to match with nd
        c_nd - multiplier constant for synthetic nd
        c_nd_err - error for these constants for each vibrational q.n.
        nd_sc - nd synthetic (s) constant (c) - synthetic upper statat population with correct constant
        """
        from scipy.optimize import curve_fit

        res = [
            curve_fit(
                lambda x, c: c * x,
                self.nd_synth[self.mask][i].dropna(),
                self.nd[i].dropna(),
                1e-7,
            )
            for i in self.nd
        ]
        res = np.array([[i[0][0], i[1][0][0]] for i in res])
        self.c_nd = res[:, 0]
        self.c_nd_er = res[:, 1]
        self.nd_sc = self.nd_synth * self.c_nd

        self.nd_vibrofit = (self.nd_sc.sum() / self.nd_sc.sum().sum()).values

    def plot_nd(self, v=0):
        """
        Plot nd experimental and synthetic
        """
        import matplotlib.pyplot as plt

        plt.plot(self.nd[v], "o")
        plt.plot(self.nd_synth[v] * self.c_nd[v], "kx")
        plt.locator_params(nbins=15)
        plt.xlim(0, 14)
        plt.grid()

    def calc_all_rot_temp(self):
        """
        Calculate Trot for d- and X-states for all v
        put them into DataFrame
        """
        self.mol.calculate_tfrac(self.nd.shape[1], norm=False)
        frac = self.mol.frac[self.isotop].values
        fr = frac / frac[0]
        trotd = pd.DataFrame(
            [fr * self.trot1, fr * self.err[2], fr * self.trot2, fr * self.err[3]],
            ["d Trot1", "d er1", "d Trot2", "d er2"],
        ).T

        fr = frac
        trotx = pd.DataFrame(
            [fr * self.trot1, fr * self.err[2], fr * self.trot2, fr * self.err[3]],
            ["X Trot1", "X er1", "X Trot2", "X er2"],
        ).T
        self.trotall = pd.concat([trotd, trotx], axis=1)

    def autofit(self):
        """ 
        A shortcut function to run required fits with default parameters 
        """
        self.fit_boltzmann()
        self.calc_nd_const()
        self.calc_all_rot_temp()

    def set_style(self, size=6):
        """
        Define plot sytles for "standard" plot outputs.
        """
        self.style_popd = [
            {
                "capsize": size - 1,
                "fmt": i,
                "mec": m,
                "mfc": j,
                "color": "k",
                "mec": m,
                "label": f"(v'-v'') = ({k}-{k})",
                "ms": size,
                "mew": l,
                "lw": 1,
            }
            for i, j, k, l, m in zip(
                ["o", "o", "x", "s"],
                ["k", "w", "k", "dimgray"],
                range(4),
                [1, 1, 1, 1,],
                ["k", "k", "k", "k"],
            )
        ]

        self.style_popd_color = [
            {
                "capsize": size,
                "elinewidth": 0.5,
                "fmt": i,
                "mec": m,
                "mfc": m,
                "color": m,
                "mec": m,
                "label": f"(v'-v'') = ({k}-{k})",
                "ms": size,
                "mew": 0.5,
                "lw": 1,
            }
            for i, j, k, m in zip(
                ["s", "o", "d", "x"],
                ["k", "w", "k", "dimgray"],
                range(4),
                [f"C{i}" for i in range(4)],
            )
        ]

    def plot_popd_paper(self, stylename="bw", fontsize=11, ms=2):
        """
        Plot d-state population with errors and fit results 
        fontsize = 11, image width = 8cm (for WORD document)
        """
        from matplotlib import pyplot as plt

        relerr = self.relerr
        self.set_style(size=ms)
        if stylename == "color":
            style = self.style_popd_color
        if stylename == "bw":
            style = self.style_popd

        for i in self.nd:
            x = flatdf(self.Ed[self.mask][i])
            y = flatdf(self.nd_rel[i])
            yerr = flatdf(relerr[i]) * y
            plt.errorbar(x, y, yerr=yerr, **style[i])
            plt.plot(self.Ed[i], self.nd_bol_synth[i], ":", color=style[i]["mec"])

        plt.plot([], [], "k:", label="fitted")

        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[1:] + [handles[0]]
        labels = labels[1:] + [labels[0]]
        ax.legend(handles, labels, fontsize=fontsize - 4.5, loc=1)

        ax.text(
            self.Ed.values[-1, 0] * 0.98,
            ((flatdf(self.relerr[0]) + 1) * flatdf(self.nd_rel[0]))[-1] + 0.02,
            f"N'={self.nd.shape[0]}",
            ha="center",
            fontsize=fontsize - 2,
        )

        plt.yscale("log")
        plt.xlabel("Rotational Energy [eV]")
        plt.ylabel(
            r"$\mathrm{\frac{n_{d v' N'}}{(2N'+1)\,g_{\mathrm{as}}^{N'}}}$ [a.u.]",
            fontsize=fontsize + 2,
        )
        plt.ylim(0.01, 1.5)
        plt.xlim(-0.025, 0.5)

        plt.gcf().set_size_inches(figsize(8))

    def about_var(self):
        """ 
        Print info about variables.
        TODO: improve names to reduce confusion
        """
        s = (
            "nd: population of the upper state d3\n"
            "nd_bol: nd/(2J+1)/g_as for Boltzmann plot\n"
            "nd_rel: nd_bol/nd_bol[0][0]\n"
            "nd_bol_synth = np.exp(two_t_all_v(Ed, *popt, melt=False))\n"
            "nd_synth = nd_bol_synth * v2jp1 * vg\n"
            "nd_sc: nd synthetic (s) constant (c) - synthetic upper state population with correct constant\n"
            "nd_vibrofit = (nd_sc.sum() / nd_sc.sum().sum()).values - synthetic vibro population\n"
        )

        print(s)
