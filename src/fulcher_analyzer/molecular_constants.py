"""
Molecular constants for hydrogen isotopologues (H2, D2).
"""
from importlib.resources import files

import numpy as np
import pandas as pd

MOLECULAR_DATA_FOLDER = files("fulcher_analyzer.data_molecular")
A_SOURCE_DOI = "10.48550/arXiv.1512.06306"
A_SOURCE = (
    "Lavrov, Pozdeev & Yakovleva (2015), Tables 5 (H2) and 7 (D2), "
    "comparison columns from their reference 21"
)
A_SEMIEMPIRICAL_SOURCE = (
    "Lavrov, Pozdeev & Yakovleva (2015), Tables 5 (H2) and 7 (D2), "
    "recommended semiempirical columns"
)
A_TABLES = ("comparison", "semiempirical")


class MolecularConstants:
    """
    Hydrogen Isotopolouges molecular constants
    """

    def __init__(self, a_table="comparison"):
        if a_table not in A_TABLES:
            raise ValueError(f"a_table must be one of {A_TABLES}, got {a_table!r}")
        self.A_table = a_table
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

    def create_dataframes(self):
        """
        Load spectroscopic constants (we, wexe, Be, ae, De) for H2 and D2.
        Data from Ishihara thesis;
        [16]: NIST Chemistry WebBook (https://webbook.nist.gov/chemistry/form-ser/)
        All constants in [1/cm].
        """
        with MOLECULAR_DATA_FOLDER.joinpath("spectroscopic_constants.csv").open(
            "r", encoding="utf-8"
        ) as f:
            df = pd.read_csv(f, comment="#")
        cols = ["we", "wexe", "Be", "ae", "De"]
        self.h2 = (
            df[df["isotope"] == "h"].set_index("state")[cols].loc[["d3", "a3", "X"]]
        )
        self.d2 = (
            df[df["isotope"] == "d"].set_index("state")[cols].loc[["d3", "a3", "X"]]
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
        with MOLECULAR_DATA_FOLDER.joinpath("fulcher-α_band_wavenumber_D2.txt").open("r") as f:
            wld = np.loadtxt(f)
        wld = pd.DataFrame(1 / (wld * 1e-7))  # wavenumber [cm-1] -> wavelength [nm]
        wld[wld > 800] = np.nan
        self.wld = wld
        with MOLECULAR_DATA_FOLDER.joinpath("fulcher-α_band_wavelength.txt").open("r") as f:
            wlh = pd.DataFrame(np.loadtxt(f))
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

    def acoeff(self):
        """
        Load Q-branch Einstein-A coefficients for D2 and H2.

        The packaged matrices are the non-empirical adiabatic comparison
        columns in Tables 5 and 7 of Lavrov, Pozdeev & Yakovleva (2015),
        transposed to rows ``v'=0..3`` and columns ``v''=0..7``. Values are
        in s^-1 and apply to N=1; the source reports negligible N dependence.
        """
        self.A_source = (
            A_SEMIEMPIRICAL_SOURCE if self.A_table == "semiempirical" else A_SOURCE
        )
        self.A_source_doi = A_SOURCE_DOI
        with MOLECULAR_DATA_FOLDER.joinpath("einstein_A_d2.csv").open(
            "r", encoding="utf-8"
        ) as f:
            self.AD_comparison = pd.read_csv(f, comment="#", header=None)
        with MOLECULAR_DATA_FOLDER.joinpath("einstein_A_h2.csv").open(
            "r", encoding="utf-8"
        ) as f:
            self.AH_comparison = pd.read_csv(f, comment="#", header=None)
        self.AD_semiempirical = self._load_a_table("einstein_A_d2_semiempirical.csv")
        self.AH_semiempirical = self._load_a_table("einstein_A_h2_semiempirical.csv")
        self.AD_semiempirical_err = self._load_a_table(
            "einstein_A_d2_semiempirical_err.csv"
        )
        self.AH_semiempirical_err = self._load_a_table(
            "einstein_A_h2_semiempirical_err.csv"
        )
        self.AD = getattr(self, f"AD_{self.A_table}")
        self.AH = getattr(self, f"AH_{self.A_table}")
        self.AD_err = (
            self.AD_semiempirical_err if self.A_table == "semiempirical" else None
        )
        self.AH_err = (
            self.AH_semiempirical_err if self.A_table == "semiempirical" else None
        )
        self.ADsum = self.AD.sum(axis=1).values
        self.AHsum = self.AH.sum(axis=1).values

    @staticmethod
    def _load_a_table(filename):
        with MOLECULAR_DATA_FOLDER.joinpath(filename).open("r", encoding="utf-8") as f:
            return pd.read_csv(f, comment="#", header=None)

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
        with MOLECULAR_DATA_FOLDER.joinpath("vibrational_energy_D2.txt").open("r") as f:
            E_vib = np.loadtxt(f)
        # excitation energy for vibrational levels
        with MOLECULAR_DATA_FOLDER.joinpath("excitation_vibrational_energy_D2.txt").open("r") as f:
            Ee_vib = np.loadtxt(f)
        # Franck-Condon factors
        with MOLECULAR_DATA_FOLDER.joinpath("franck_condon_factor_D2.txt").open("r") as f:
            fcf = np.loadtxt(f)
        self.corona_constants_d = [E_vib, Ee_vib, fcf]
        self.fcfd = pd.DataFrame(fcf)

        # Hydrogen
        # vibrational energy
        with MOLECULAR_DATA_FOLDER.joinpath("vibrational_energy.txt").open("r") as f:
            E_vib = np.loadtxt(f)
        # excitation energy for vibrational levels
        with MOLECULAR_DATA_FOLDER.joinpath("excitation_vibrational_energy.txt").open("r") as f:
            Ee_vib = np.loadtxt(f)
        # Franck-Condon factors
        with MOLECULAR_DATA_FOLDER.joinpath("franck_condon_factor.txt").open("r") as f:
            fcf = np.loadtxt(f)
        self.corona_constants_h = [E_vib, Ee_vib, fcf]
        self.fcfh = pd.DataFrame(fcf)
