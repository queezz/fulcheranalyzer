"""
Small non-plotting utility functions shared across the package.
"""
import numpy as np
import pandas as pd


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
