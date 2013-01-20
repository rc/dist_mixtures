# -*- coding: utf-8 -*-
"""

Created on Fri Oct 26 21:57:48 2012

Author: Josef Perktold
"""

import numpy as np
from scipy import stats, interpolate, integrate
from dist_mixtures.mixture_von_mises import (VonMisesMixture,
                                             normalize_params, shift_loc)

from numpy.testing import assert_almost_equal, assert_


p = [-4, -2.5*np.pi, -4, 1*np.pi, 5, 3.5*np.pi, 0, 0]
p_transformed = [4, 0.5*np.pi, 4, 0, 5, -0.5*np.pi, 0, 0]
assert_almost_equal(normalize_params(p), p_transformed, 13)

loc1 = shift_loc(np.linspace(-5, 5, 21) * np.pi)[:-1]
loc2 = np.tile(np.linspace(-1, 0.5, 4) * np.pi, 5)
assert_almost_equal(loc1, loc2, 13)


res2_params = np.array([ 1.90886275, -2.99882496,  0.38442792, -0.86952549,
                        0.33013396])

res3_params = np.array([ 0.27505697, -1.27358384,  2.38037407, -2.90256257,
                        -4.29683974, -0.48736552,  2.08776124,  1.9212298 ])

res4_params = np.array([ 1.56354196, -2.95987588, -1.19876203, -2.94677148,
                        -4.76746645,  1.87938641,  9.89761038, -0.49076411,
                        3.97260961,  2.50826187,   1.29319052])

dparams = zip(res3_params[:6:2], res3_params[1:6:2])


#res3_params causes
#Exception OverflowError: 'range() result has too many items' in 'scipy.stats.vonmises_cython.von_mises_cdf_series' ignored
vn_dist2 = VonMisesMixture(res2_params)

xgrid = np.linspace(-90, 89, 180) / 90. * np.pi
delta = xgrid[1] - xgrid[0]  #= 2 * np.pi / 180.

print vn_dist2.cdf_mix(res2_params, xgrid[:10])

components = [VonMisesMixture(params) for params in dparams]


pdf1 = vn_dist2.pdf_mix(res3_params, xgrid)
pn = normalize_params(res3_params)
pdf2 = vn_dist2.pdf_mix(pn, xgrid)
assert_almost_equal(pdf1, pdf2, decimal=13)
assert_almost_equal(pdf1.sum() * delta, 1, decimal=13)

cdf1 = vn_dist2.cdf_mix(normalize_params(res3_params), xgrid)

#integrate discretized
c = np.concatenate(([0], integrate.cumtrapz(pdf1, dx=delta)))
assert_almost_equal(c, cdf1, decimal=4)
c2 = np.concatenate(([0], pdf1.cumsum()[:-1] * delta))
assert_almost_equal(c2, cdf1, decimal=2)
assert_(np.max(np.abs(1 - c2[1:] / cdf1[1:])) < 0.009)

#plot of half circle with axial density
import matplotlib.pyplot as plt
xgrid_h = np.linspace(0, 179, 180) / 180. * np.pi
pdf_h = vn_dist2.pdf_mix(res3_params, xgrid_h*2.) * 2
a, b = 1, 1
rs = a + b * pdf_h
xs = rs * np.cos(xgrid_h)
ys = rs * np.sin(xgrid_h)
xs_base = np.cos(xgrid_h)
ys_base = np.sin(xgrid_h)
plt.plot(xs, ys)
plt.plot(xs_base, ys_base)
ax = plt.gca()
ax.set_aspect('equal')
