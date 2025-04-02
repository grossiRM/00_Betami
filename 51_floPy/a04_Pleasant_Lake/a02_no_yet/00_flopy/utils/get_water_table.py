import numpy as np
import warnings

def get_water_table(heads, nodata, per_idx=None):
    """
    Get a 2D array representing the water table elevation for each
    stress period in heads array.

    Parameters
    ----------
    heads : 3 or 4-D np.ndarray
        Heads array.
    nodata : real
        HDRY value indicating dry cells.
    per_idx : int or sequence of ints
        stress periods to return. If None,
        returns all stress periods (default is None).

    Returns
    -------
    wt : 2 or 3-D np.ndarray of water table elevations
        for each stress period.

    """
    heads = np.array(heads, ndmin=4)
    nper, nlay, nrow, ncol = heads.shape
    if per_idx is None:
        per_idx = list(range(nper))
    elif np.isscalar(per_idx):
        per_idx = [per_idx]
    wt = []
    for per in per_idx:
        wt_per = []
        for i in range(nrow):
            for j in range(ncol):
                for k in range(nlay):
                    if heads[per, k, i, j] != nodata:
                        wt_per.append(heads[per, k, i, j])
                        break
                    elif k == nlay - 1:
                        wt_per.append(nodata)
        assert len(wt_per) == nrow * ncol
        wt.append(np.reshape(wt_per, (nrow, ncol)))
    return np.squeeze(wt)