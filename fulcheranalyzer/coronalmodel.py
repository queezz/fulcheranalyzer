import numpy as np, pandas as pd
from os.path import join, abspath
from ._constants import package_directory

ABSOLUTESIGMA = False
MOLECULAR_DATA_FOLDER = abspath(join(package_directory, "..", "data_molecular"))
DATA_FOLDER = abspath(join(package_directory, "..", "data"))


class MolecularConstants:
    """
    Hydrogen Isotopolouges molecular constants
    """

    def __init__(self):
        self.name = "Molecular Constatns for Hydrogen Isotopologues"
        self.create_dataframes()
        self.calculate_tfrac(4, norm=True)
        self.general_constants()
        self.calculate_all_E_rot()
        self.acoeff()
        self.load_corona_constants()
        self.load_wavelength_data()

    def general_constants(self):
        """
        Populate class with useful constants 
        """
        self.eV_cm = 1.23984e-4  # eV/cm-1 wavenumber to eV

    def parse_data(self, s):
        """
        Parse file with rotational constants
        """
        return pd.DataFrame(
            np.array([i.strip().split(" ") for i in s.split("\n")]).astype(np.float64),
            ["d3", "a3", "X"],
            ["we", "wexe", "Be", "ae", "De"],
        )

    # TODO: Move data to datafiles.
    def create_dataframes(self):
        """ 
        Data from Ishihara-s thesis 
        [16]:  NIST Chemistry WebBook ( https://webbook.nist.gov/chemistry/form-ser/ )
        All constants have same energy unit: [1/cm].
        """

        self.h2 = self.parse_data(
            """2371.57 66.27 30.364 1.545 0.0191
               2664.83 71.65 34.216 1.671 0.0216
               4401.21 121.33 60.853 3.062 0.0471"""
        )
        self.d2 = self.parse_data(
            """1678.22 32.94 15.200 0.5520 0.0049
               1885.84 35.96 17.109 0.606 0.0055
               3115.50 61.82 30.443 1.0786 0.01141"""
        )

    def tfrac(self, v, isotop="d"):
        """
        Temperature ratio for given vibrational number for d3 level
        calculated as ratio of rotational constants
        """
        if isotop == "d":
            x = self.d2.loc["X"]
            d = self.d2.loc["d3"]
        else:
            x = self.h2.loc["X"]
            d = self.h2.loc["d3"]
        return ((x["Be"] - x["ae"] * (v + 0.5)) / (x["Be"] - x["ae"] * (0 + 0.5))) * (
            (x["Be"] - x["ae"] * (0 + 0.5)) / (d["Be"] - d["ae"] * (0 + 0.5))
        )

    def calculate_tfrac(self, vmax, norm=False):
        """
        Temperature ratios DataFrame for H and D
        """
        frac = pd.DataFrame(
            [
                np.array([self.tfrac(v, isotop="d") for v in range(vmax)]),
                np.array([self.tfrac(v, isotop="h") for v in range(vmax)]),
            ],
            ["d", "h"],
        ).T
        if norm:
            self.frac = frac / frac.loc[0]
        else:
            self.frac = frac

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
        self.wlh[self.wlh < 1] = np.nan

    def E_rot_formula(self, v, J, c, isotop="d", state="d3"):
        """
        Rotational energy formula
        """
        Be, ae, De = c
        B = Be - ae * (v + 1 / 2)
        return (B * J * (J + 1) - De * J ** 2 * (J + 1) ** 2) * self.eV_cm

    def calculate_E_rot(self, vlen, Jlen, isotop="d", state="d3"):
        """
        Fill DataFrame with rotational energies for given isotope and state
        [En] isotope [Ru] isotop
        NOTE:
        For d-state: J starts from 1.
        For X-state: J must start from 0.
        """
        if state == "d3":
            J0 = 1
        if state == "X":
            J0 = 0
        if isotop == "d":
            m = self.d2
        else:
            m = self.h2

        Be = m.loc[state, "Be"]  # rotational constant
        ae = m.loc[state, "ae"]  # ro-vib interaction constant
        De = m.loc[state, "De"]  # centrifugal distortion constant
        c = (Be, ae, De)

        return pd.DataFrame(
            [
                [
                    self.E_rot_formula(v, J + J0, c, isotop=isotop, state=state)
                    for v in range(vlen)
                ]
                for J in range(Jlen)
            ]
        )

    def E_vib_formula(self, v, c):
        """
        Vibrational energy formula
        """
        we, wexe = c
        return (we * (v + 0.5) - wexe * (v + 0.5) ** 2) * self.eV_cm

    def calculate_E_vib(self, vmax=5, state="d3", isotop="d"):
        """
        Calculate an array of vibrational energy
        vmax - maximum vibrational q.n. in the array
        state - electronic state, 'X', 'd3', 'a3'
        isotop - isotopologue, 'd' - D2, 'h' - H2
        """
        if isotop == "d":
            m = self.d2
        if isotop == "h":
            m = self.h2
        we = m.loc[state, "we"]
        wexe = m.loc[state, "wexe"]

        return np.array([self.E_vib_formula(v, [we, wexe]) for v in range(vmax + 1)])

    def calculate_all_E_rot(self, vlen=4, Jlen=14):
        """
        Calculate rotational energy arrays for X, d states for H and D
        """
        self.EdH = self.calculate_E_rot(vlen, Jlen, isotop="h", state="d3")
        self.ExH = self.calculate_E_rot(vlen, Jlen, isotop="h", state="X")
        self.EdD = self.calculate_E_rot(vlen, Jlen, isotop="d", state="d3")
        self.ExD = self.calculate_E_rot(vlen, Jlen, isotop="d", state="X")

    # TODO: Move data to datafiles. Add data reference to the source.
    def acoeff(self):
        """
        Populate Acoeff for D2 and H2
        """
        # Deuterium
        self.AD = pd.DataFrame(
            [
                [
                    2.3387e07,
                    2.3479e06,
                    3.9083e04,
                    6.9282e01,
                    6.7372e-02,
                    1.9986e-02,
                    2.9027e-04,
                    7.7599e-03,
                ],
                [
                    2.1551e06,
                    1.8841e07,
                    4.4730e06,
                    1.2747e05,
                    2.7294e02,
                    4.3313e-01,
                    3.4305e-02,
                    3.5382e-02,
                ],
                [
                    1.8763e05,
                    3.8098e06,
                    1.4795e07,
                    6.3500e06,
                    2.4539e05,
                    6.9366e02,
                    2.0363e00,
                    2.0664e-03,
                ],
                [
                    1.7678e04,
                    5.2206e05,
                    4.9835e06,
                    1.1276e07,
                    7.9698e06,
                    4.1361e05,
                    1.3613e03,
                    6.0937e00,
                ],
            ]
        )
        self.ADsum = self.AD.sum(axis=1).values
        # Hydrogen
        self.AH = pd.DataFrame(
            [
                (
                    2.4077e7,
                    1.6552e6,
                    9.2743e3,
                    7.7501e-2,
                    5.6159e-2,
                    2.1645e-5,
                    1.2126e-4,
                    2.4635e-4,
                ),
                (
                    1.5258e6,
                    2.0655e7,
                    3.2649e6,
                    2.9732e4,
                    2.8248e0,
                    1.8340e-1,
                    4.0941e-6,
                    4.9920e-3,
                ),
                (
                    1.0712e5,
                    2.8369e6,
                    1.7377e7,
                    4.7993e6,
                    6.2309e4,
                    2.1093e1,
                    6.3848e-1,
                    1.1783e-2,
                ),
                (
                    8.3952e3,
                    3.1899e5,
                    3.8874e6,
                    1.4317e7,
                    6.2363e6,
                    1.0633e5,
                    1.0579e2,
                    1.8096e0,
                ),
            ]
        )

        self.AHsum = self.AH.sum(axis=1).values

    def calculate_spin_multiplicity(self, Jmax=13):
        """
        Calculate spin multiplicity vectors for D2 and H2
        """
        self.gas_d2 = np.array([6 - 3 * np.mod((J + 1), 2) for J in range(Jmax)])
        self.gas_h2 = np.array([np.mod((J + 1), 2) * 2 + 1 for J in range(Jmax)])

    def load_corona_constants(self):
        """
        Load constants for cornal model
        """
        # Deuterium
        # vibrational energy
        E_vib = np.loadtxt(join(MOLECULAR_DATA_FOLDER, "vibrational_energy_D2.txt"))
        # excitation energy for vibrational levels
        Ee_vib = np.loadtxt(
            join(MOLECULAR_DATA_FOLDER, "excitation_vibrational_energy_D2.txt")
        )
        # Franck-Condon factors
        fcf = np.loadtxt(join(MOLECULAR_DATA_FOLDER, "franck_condon_factor_D2.txt"))
        self.corona_constants_d = [E_vib, Ee_vib, fcf]
        self.fcfd = pd.DataFrame(fcf)

        # Hydrogen
        # vibrational energy
        E_vib = np.loadtxt(join(MOLECULAR_DATA_FOLDER, "vibrational_energy.txt"))
        # excitation energy for vibrational levels
        Ee_vib = np.loadtxt(
            join(MOLECULAR_DATA_FOLDER, "excitation_vibrational_energy.txt")
        )
        # Franck-Condon factors
        fcf = np.loadtxt(join(MOLECULAR_DATA_FOLDER, "franck_condon_factor.txt"))
        self.corona_constants_h = [E_vib, Ee_vib, fcf]
        self.fcfh = pd.DataFrame(fcf)


# ==================================================
#
# Convenience functions and fitting functions for Boltzmann plot
#
# ==================================================


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
#  Work with fitted line intensities
#
# ==================================================


def write_intensities(inte, *arg):
    """
    Save intensites for a given shot and given frame
    in a `*.csv` file in `DATA_FOLDER` subfolder
    intensities in pandas.DataFrame, columns: v-v (vibrational q.n.),
    rows: J (rotational q.n.).
    """
    shot, frame, gas = arg
    fpth = join(DATA_FOLDER, f"{shot}_fr_{frame}.csv")
    # write header
    txt = (
        f"# shotnumber: {shot}\n"
        f"# frame : {frame}\n"
        f"# gas : {gas}\n"
        "# Columns: vibrational quantum number\n"
        "# Rows: rotational quantum number\n"
        "# Values: Q-branch line intensities [unit]\n"
        "# [Data]\n"
    )
    with open(fpth, "w") as f:
        f.write(txt)
    # and then data
    inte.to_csv(fpth, mode="a", index=False, header=False)


def read_intensities(shot, frame):
    """
    Read intensities
    """
    fpth = join(DATA_FOLDER, f"{shot}_fr_{frame}.csv")
    ferr = join(DATA_FOLDER, f"{shot}_fr_{frame}_err.csv")
    inte = pd.read_csv(fpth, comment="#", header=None)
    try:
        interr = pd.read_csv(ferr, comment="#", header=None)
    except:
        print("no error data was found")
        return inte, inte * 0.1
    return inte, interr


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
        plt.ylabel("$n/(2J+1)/g_{as}$")
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
        plt.ylabel("$\ln{\left( n/(2J+1)/g_{as}\\right)}$, a.u.")

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
        plt.ylabel("$n/(2J+1)/g_{as}$, a.u.")

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
            "$\mathdefault{\mathrm{\\frac{ n_{dv'N'}}{(2N'+1)g^{N'}_{as}}}}$ [a.u.]",
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
        ax.set_ylabel("$\mathdefault{\mathrm{n_{dv\mathrm{'}N\mathrm{'}}}}$ [a.u.]")

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
            "$\\mathdefault{\mathrm{\\frac{n_{XvN}}{(2N+1)g_{as}^{N}}}}}$ [a.u.]",
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


def delta_kro(a, b):
    """ 
    Kronecker-s delta  https://en.wikipedia.org/wiki/Kronecker_delta
    """
    if a == b:
        return 1
    else:
        return 0


def g_as(J, isotop="d"):
    """
    Spin multiplicity, or stat. weight, 
    formula for rotational q.n. J for H2 or D2
    """
    if isotop == "d":
        return 6 - 3 * np.mod(J, 2)
    if isotop == "h":
        return np.mod(J, 2) * 2 + 1


def g_as_vector(Jlen=15, isotop="d", transpose=False, j0=0):
    """
    calculate g_as vector
    """
    gvect = np.array([g_as(J, isotop=isotop) for J in range(j0, Jlen + j0)])
    if transpose:
        return gvect[:, None]
    else:
        return gvect


def tjpo_vector(Jlen=15, transpose=False, j0=0):
    """ 
    vector (2(Jind+1)+1)
    d-state: j0=1
    X-state: j0=0
    """
    if transpose:
        return np.array([2 * J + 1 for J in range(j0, Jlen + j0)])[:, None]
    else:
        return np.array([2 * J + 1 for J in range(j0, Jlen + j0)])


def reshape_4d2d(matrix):
    """
    Reshape 4d numpy array into 2d
    """
    a, b, c, d = matrix.shape
    return matrix.reshape(a * b, c * d)


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


# Which of two are best for flattening DataFrame containing nans?
# flatdf seems to bee 2-times faster, 600 micros vs 1.8 ms for 4x15 matrix


def flatdf(df, order="f"):
    """
    Flatten a DataFrame with nans into np.array()
   

    Parameters
    ----------

    order: list
        order = ['f','c'], see numpy.ndarray.flatten
    """
    return pd.DataFrame(df.values.flatten(order="f")).dropna().values.T[0]


def flatdf_1(df, name="val"):
    """ 
    Flatten DataFrame, remove nans, reset index. Good for fitting.
    Consistently returns 1d array. Order is Column-wise, or 'f'.
    """
    df = df.melt(value_name=name).dropna()
    del df["variable"]
    df = df.reset_index(drop=True)
    return df[name].values


def figsize(width=8, ratio=5 / 6):
    """
    Calculate image size from width in cm
    """
    cm_to_inch = 1 / 2.54
    # -0.05 - adjust for padding when saving.
    # For some reason savefig adds more padding, then savefig.pad_inches: 0.05
    pad = -0.2
    return (width * cm_to_inch + pad, (width * cm_to_inch + pad) * ratio)


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
