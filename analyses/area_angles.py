import numpy as np

from analyses.transforms import transform_pi_deg

def get_area_angles(data, mode='min', ishift=None):
    """
    Get equal area angles.

    For 'min' mode:

    First, the x-axis of the histogram is rolled so that the histogram minimum
    is at the beginning/end of the x-axis. The histogram maximum is then
    assumed to be somewhere around the middle of the x-axis.

    For 'max' mode:

    First, the x-axis of the histogram is rolled so that the histogram maximum
    is in the middle of the x-axis. The histogram minimum is then assumed to be
    somewhere around the beginning/end of the x-axis.

    The two modes should give similar results for symmetric histograms.

    Then:

    Assuming two main directions of fibres which are symmetric with respect to
    the angle of symmetry a_s and have the same probability 0.5 we obtain the
    mean value of these directions and their variation using the following
    algorithm. The integral sum of the histogram is obtained. Half of the sum
    is assigned to the angles smaller than the angle of symmetry and the second
    half to the angles greater than the angle of symmetry. Then each half is
    again divided into two equal area halves - two intervals are obtained [l_s,
    a_s], [a_s, r_s], where l_s, r_s are the dividing angles in the left and
    right half-areas, respectively. The mid-points of those intervals are taken
    as the directions of the assumed two fibre systems.
    """
    ddd0 = data.copy()
    ddd0[:, 0] = transform_pi_deg(ddd0[:, 0], neg_shift=0)

    if ishift is None:
        if mode == 'min':
            ishift = np.argmin(ddd0[:, 1])

        else:
            ishift = np.argmax(ddd0[:, 1]) - int(float(data.shape[0]) / 2)

    ddd = np.roll(ddd0, -ishift, axis=0)

    # Make angle ascending.
    for ii in xrange(1, ddd.shape[0]):
        if ddd[ii, 0] < ddd[ii - 1, 0]:
            ddd[ii, 0] += 180.0

    # Mirror the first data point.
    dx = ddd[1, 0] - ddd[0, 0]
    ddd = np.r_[ddd, [[ddd[-1, 0] + dx, ddd[0, 1]]]]

    xmin, xmax = -1000, 1000
    arh, xm = split_equal_areas(ddd, xmin, xmax)
    arh1, x0 = split_equal_areas(ddd, xmin, xm)
    arh2, x1 = split_equal_areas(ddd, xm, xmax)

    x0 = x0 if x0 >= 0.0 else x0 + 180
    xm = xm if xm >= 0.0 else xm + 180
    x1 = x1 if x1 >= 0.0 else x1 + 180

    print x0, xm, x1
    print arh, arh1, arh2, arh1 - arh2, arh - (arh1 + arh2)

    return x0, xm, x1, arh1, arh2

def split_equal_areas(data, x0, x1):
    """
    Split histogram-like `data` into two parts with equal areas between `x0`
    and `x1`.
    """
    x, y = data[:, 0].copy(), data[:, 1]
    n_data = data.shape[0]

    dx = x[1] - x[0]

    xs = x - 0.5 * dx

    i0 = np.searchsorted(xs, x0)
    if i0 == 0:
        sub0 = 0.0
        i0 = 1

    else:
        sub0 = (x0 - xs[i0 - 1]) * y[i0 - 1]

    i1 = np.searchsorted(xs, x1)
    if i1 == n_data:
        sub1 = 0.0

    else:
        sub1 = (xs[i1] - x1) * y[i1 - 1]

    yy = y[i0 - 1:i1] * dx
    area = np.sum(yy) - sub0 - sub1

    ca = np.cumsum(yy) - sub0
    ih = np.searchsorted(ca, 0.5 * area)

    da = ca[ih] - 0.5 * area
    dxh = da / y[i0 - 1 + ih]

    xh = xs[i0 + ih] - dxh

    return 0.5 * area, xh

