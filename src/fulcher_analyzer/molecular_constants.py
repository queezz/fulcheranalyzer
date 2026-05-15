"""
Molecular constants for hydrogen isotopologues (H2, D2).
"""
import numpy as np
import pandas as pd
from importlib.resources import files

MOLECULAR_DATA_FOLDER = files("fulcher_analyzer.data_molecular")


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
