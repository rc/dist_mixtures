"""
Various functions for data transformations.
"""
import numpy as np

def transform_2pi(data):
    data = np.asarray(data)

    out = 2 * data * np.pi / 180.0
    return out

def transform_pi_deg(data, neg_shift=False):
    '''transform radial on (-pi, pi) to axial degrees

    Parameters
    ----------
    data : array_like
        data in radians
    neg_shift : bool
        If False (default), then radians on full circle are converted to
        degrees on half circle (axial)
        If True, then degrees are returned for half-circle (0, 180).

    Returns
    -------
    deg : ndarray
        axial degrees
    '''
    #same as np.remainder(data * (90 / np.pi), 180) if data is in (-pi, pi)
    data = np.asarray(data)

    out = 90.0 * data / np.pi
    if neg_shift:
        out = np.where(out >= 0.0, out, out + 180.0)
    return out

def fix_range(data):
    '''transform or wrap data into [-pi, pi]

    If `data` is a 2D array, the angles are assumed to be in the first column.

    Parameters
    ----------
    data : array_like
        data in radians

    Returns
    -------
    data2 : ndarray
        data in radians wrapped to closed interval [-pi, pi]
    '''
    #almost same as np.remainder(data+np.pi, 2*np.pi) - np.pi
    #which maps to half open interval [-pi, pi)
    data = np.asarray(data)
    data = data.copy()
    if data.ndim == 2:
        angles = data[:, 0] # View!

    else:
        angles = data

    while 1:
        ii = np.where(angles < -np.pi)[0]
        angles[ii] += 2 * np.pi
        if not len(ii): break
    while 1:
        ii = np.where(angles > np.pi)[0]
        angles[ii] -= 2 * np.pi
        if not len(ii): break

    return data

def fix_increasing(data):
    """
    Sort `data` so that values in the first column are non-decreasing.
    """
    data = np.asarray(data)
    ii = np.argsort(data[:, 0])
    data = data[ii]

    return data

def get_counts_from_lengths(lengths):
    """
    Get simulated counts corresponding to lengths.
    """
    lengths = np.asarray(lengths)

    lo = lengths[lengths > 0.0].min()
    counts = ((10.0 / lo) * lengths).astype(np.int32)

    return counts

def spread_by_counts(data, counts, trivial=False):
    """
    Spread items in `data` according to `counts`.

    If `trivial` is True, only repeat n-th item of data `counts[n]`
    times. Otherwise include between `data[n]` and `data[n + 1]` a linear
    sequence with `counts[n] + 1` items from `data[n]` to `data[n + 1]` without
    the last item.
    """
    if trivial:
        out = np.repeat(data, counts)

    else:
        dd = data[-1] - data[-2]
        data = np.r_[data, data[-1] + dd]

        out = np.empty(counts.sum(), dtype=data.dtype)
        ii = 0
        for ic, count in enumerate(counts):
            out[ii:ii + count] = np.linspace(data[ic], data[ic+1],
                                             count + 1)[:-1]
            ii += count

    return out
