"""
CSV I/O helpers for fitted Q-branch line intensities.
"""
import pandas as pd
from os.path import join, abspath

from ._constants import package_directory

DATA_FOLDER = abspath(join(package_directory, "..", "..", "data"))


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
