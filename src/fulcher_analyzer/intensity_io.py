"""
CSV I/O helpers for fitted Q-branch line intensities.
"""
import pandas as pd
from os.path import join
from importlib.resources import files

# Bundled example/regression intensity data shipped with the package.
INTENSITY_DATA = files("fulcher_analyzer.example_data.intensities")


def write_intensities(inte, *arg, data_folder=None):
    """
    Save intensities for a given shot and frame to a CSV file.

    Parameters
    ----------
    inte : pandas.DataFrame
        Intensity DataFrame (columns: vibrational q.n., rows: rotational q.n.).
    shot, frame, gas : positional args (unpacked from *arg)
    data_folder : str or path-like, optional
        Directory to write into. Defaults to the root ``data/`` folder kept
        alongside the original source tree.  Pass an explicit path when saving
        results to a custom location.
    """
    shot, frame, gas = arg
    if data_folder is None:
        # Fallback to the legacy root data/ path for write operations.
        # Writing into the installed package tree is not supported; callers
        # that need persistence should supply an explicit data_folder.
        from os.path import abspath
        from ._constants import package_directory
        data_folder = abspath(join(package_directory, "..", "..", "data"))

    fpth = join(str(data_folder), f"{shot}_fr_{frame}.csv")
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
    inte.to_csv(fpth, mode="a", index=False, header=False)


def read_intensities(shot, frame, data_folder=None):
    """
    Read Q-branch line intensities for a given shot and frame.

    Parameters
    ----------
    shot : int
        Discharge shot number.
    frame : int
        Frame index within the shot.
    data_folder : str, path-like, or importlib.resources Traversable, optional
        Directory to read from.  When *None* (default) the bundled example
        data shipped with the package is used.  Pass an explicit path or
        Traversable to read from a custom location.

    Returns
    -------
    tuple of (intensity_df, error_df) : pandas.DataFrame
    """
    if data_folder is None:
        folder = INTENSITY_DATA
    else:
        import pathlib
        folder = pathlib.Path(data_folder)

    def _read(resource_path):
        with resource_path.open("r") as f:
            return pd.read_csv(f, comment="#", header=None)

    inte = _read(folder.joinpath(f"{shot}_fr_{frame}.csv"))
    try:
        interr = _read(folder.joinpath(f"{shot}_fr_{frame}_err.csv"))
    except Exception:
        print("no error data was found")
        return inte, inte * 0.1
    return inte, interr
