from pathlib import Path
from typing import Dict, List, Union

from molecular_data import fulcher_alpha_wavelengths

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

################################################################################
#                               HELPER FUNCTIONS                               #
################################################################################

# fmt: off
def read_spectrum_headers(spec_path: Path) -> Dict[str, Union[str, List[str]]]:
    """
    Helper function for reading headers form echelle_spectra spectrum files
    ----------
    args:
        spec_path           pathlib.Path                            path to spectrum file
    ----------
    returns:
        headers             Dict[str: List[str], str]               dictionary of headers extracted from spectrum file
    """
    headers: Dict[str: Union[str, List[str]]] = {}          # define empty dict that will contain headers
    with open(spec_path, 'r') as spec_file:                 # open spectrum file for reading only
        for line in spec_file:                              # iterate over all lines in the file
            if line.startswith("#"):                        # if line starts with "#", we are in header region:
                if len(line) < 3 or "[" in line:            # > if line contains no data or contains section title ...
                    continue                                # > ... move on to the next line, no data can be extracted
                key, value = line[2:].split(" = ")          # > split line at "=", left is key, right is value
                value = value.rstrip().replace("'", "")     # > remove newline and quote characters from string
                if "," in value:                            # > if value contains a comma, it is probably a list ...
                    value = value.split(",")                # > ... split the string at each comma, value is now a list
                headers[key] = value                        # > add key and value to header dictionary
            else:                                           # if line DOESN'T start with a "#", we are in data region:
                return headers                              # > all headers have been read, return them


def read_spectrum(spec_path: Path, fixed_err_scale: float = 0.1) -> xr.Dataset:
    """
    Helper function for reading data from spectrum file and storing it as an xarray Dataset with headers as metadata
    ----------
    args:
        spec_path           pathlib.Path                            path to spectrum file
        fixed_err_scale     float                                   if no error file is found, use this multiplicative
                                                                        factor to calculate and error from spectrum data
    ----------
    returns:
        spectrum            xarray.Dataset                          dataset containing spectrum and error data
    """
    headers: Dict[str, Union[str, List[str]]] = read_spectrum_headers(spec_path)  # read spectrum file headers
    spectrum: np.ndarray = np.loadtxt(spec_path, delimiter=",")  # load data from spectrum file into numpy array
    try:                                                    # attempt to load spectrum error data from a neighbour file
        error: np.ndarray = np.loadtxt(spec_path.with_stem(f"{spec_path.stem}_err"), delimiter=",")
    except FileNotFoundError:                               # if this file doesn't exist, error will be estimated
        print(f"No spectrum error file found, using a fixed error scale factor of {fixed_err_scale}")
        error: np.ndarray = spectrum.copy()                 # copy the contents of the spectrum array
        error[:, 1:] *= fixed_err_scale                     # multiply each intensity value by error factor

    data, err = [xr.DataArray(array[:, 1:],                 # construct a data array containing intensity values as data
                              coords=[("wavelength", array[:, 0]),  # wavelength is first array dim, first col of array
                                      ("frame", np.arange(int(headers["DimSize"])))])  # frame number is second dim
                 for array in [spectrum, error]]            # create data array for both spectrum and error data
    return xr.Dataset({"data": data, "error": err}, attrs=headers)  # combine data arrays into a dataset and return them


def plot_spectrum(spectrum: xr.Dataset, frame: int, hspace: float = 0.05, bsize: float = 0.25) -> None:
    """
    Helper function for plotting spectrum and all available vibrational transitions
    ----------
    args:
        spectrum            xarray.Dataset                          dataset containing spectrum and error data
        frame               int                                     index of image frame that will be plotted
        hspace              float                                   horizontal buffer space between outermost wavelength
                                                                        and outermost Q-branch as a factor of data range
        bsize               float                                   vertical size of each branch tick (max: 1)
    ----------
    returns:
        None
    """
    # get an unordered list of all available Q-branch wavelengths for each isotope
    h2, d2 = [np.array(list(fulcher_alpha_wavelengths[iso].values()), dtype=float) for iso in ["H2", "D2"]]
    # compute the minimum and maximum Q-branch wavelength for both of the above lists combined
    wav_min, wav_max = min([np.nanmin(iso) for iso in [h2, d2]]), max([np.nanmax(iso) for iso in [h2, d2]])
    # compute the wavelength limits based on the horizontal buffer space requirements
    limits = wav_min - ((wav_max - wav_min) * hspace), wav_max + ((wav_max - wav_min) * hspace)
    # compute the wavelength indices of the upper (right) and lower (left) limits, data will be cropped to this limit
    left_idx, right_idx = [(np.abs(spectrum["data"]["wavelength"].data - val)).argmin() for val in limits]

    # create figure and 3 axis, from top to bottom: H2 (0) and D2 (1) vib. trans. Q-branches, (2) intensity spectrum
    fig, (ax_h2, ax_d2, ax_spec) = plt.subplots(nrows=3, sharex=True, figsize=(12, 6))
    ax_spec.plot(spectrum["data"]["wavelength"][left_idx:right_idx], spectrum["data"][left_idx:right_idx, frame],
                 "k-", lw=1)                                # plot cropped spectrum on bottom axis as a thin black line
    for isotope, ax in zip(["H2", "D2"], [ax_h2, ax_d2]):   # iterate over the two available isotopes
        ax.set_yticks(ticks=np.arange(len(fulcher_alpha_wavelengths[isotope])),  # y tick positions are vv indices
                      labels=fulcher_alpha_wavelengths[isotope].keys())  # y tick labels are in friendly format: "v-v"
        for vv, wavelengths in fulcher_alpha_wavelengths[isotope].items():  # iterate over each vibrational transition
            wl = np.array(wavelengths, dtype=float)         # coerce all non-finite wavelength to np.nan for filtering
            # plot a single horizontal line at each v-v position spanning from the lowest to highest finite Q-branch
            ax.hlines(int(vv[0]), xmin=np.nanmin(wl), xmax=np.nanmax(wl), colors="k", lw=1)
            # plot short vertical lines representing individual Q-branches attached to the horizontal line
            ax.vlines(wl[np.isreal(wl)], ymin=int(vv[0]) - bsize, ymax=int(vv[0]), colors="k", lw=1)
            # write text on the left and right of the horizontal line, labelling the first and last available Q-branch
            ax.text(np.nanmin(wl) - (1 + bsize), int(vv[0]) - bsize, f"Q{np.nanargmin(wl) + 1}")
            ax.text(np.nanmax(wl) + bsize, int(vv[0]) - bsize, f"Q{np.nanargmax(wl) + 1}")

    # set other appropriate axis labels
    ax_spec.set_xlabel(f"Wavelength [{spectrum.attrs['DimUnits']}]")
    ax_spec.set_ylabel(f"Intensity [{spectrum.attrs['ValUnit'][frame]}]")
    ax_h2.set_ylabel("H2 vib. trans.")
    ax_d2.set_ylabel("D2 vib. trans.")

    # set figure super-title, adjust layout for minimal whitespace, and show figure
    fig.suptitle(f"Shot number: {spectrum.attrs['ShotNo']}; Frame number: {frame}")
    fig.tight_layout()
    plt.show()
# fmt: on

