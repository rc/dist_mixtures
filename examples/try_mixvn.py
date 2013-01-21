# -*- coding: utf-8 -*-
"""estimating von mises mixture, initial trial version

Created on Sun Jan 20 16:36:19 2013

Author: Josef Perktold
"""

import numpy as np
from scipy import stats, optimize

from dist_mixtures.mixture_von_mises import est_von_mises, loglike, get_probs

if __name__ == '__main__':
    #original example

    #np.random.seed(8764692)
    x = stats.vonmises.rvs(2, loc=3, size=500)
    print stats.vonmises.fit(x, fscale=1)
    #doesn't work:
    #stats.vonmises.a = -np.pi
    #stats.vonmises.b = np.pi
    #print stats.vonmises.fit(x)
    print est_von_mises(x)
    print est_von_mises(x, method='corr')

    #with scale=0.5
    print stats.vonmises.fit(x/2, fscale=0.5)


    #try mixture distribution, not a good example
    ni = 100
    xx = np.concatenate((stats.vonmises.rvs(1, loc=-2, size=ni),
                         stats.vonmises.rvs(3, loc=0, size=ni),
                         stats.vonmises.rvs(5, loc=2, size=ni)))
    params = optimize.fmin(loglike, [1,-2, 1,0, 1,2, 0, 0], args=(xx,))
    print params
    k_dist = 3
    ii = np.arange(k_dist)
    print 'probabilities', get_probs(params)
    print 'shapes       ', params[ii*2]
    print 'locations    ', params[ii*2+1]

    import matplotlib.pyplot as plt
    f,b,h = plt.hist(xx, bins=(2*ni)/10, normed=True)
    plt.show()

    #some global searching based on histogram peaks
    bc = b[:-1] + np.diff(b)/2
    from scipy import ndimage
    mask = (f == ndimage.filters.maximum_filter(f, 3))
    bc_short = bc[mask]
    kfact = bc_short / (2*np.pi)
    bc_short -= np.round(kfact) * (2*np.pi)
    bc_short.sort()
    n = np.sum(mask)
    params0 = np.array([1.,-2, 1.,0, 1,2, 0, 0])
    idx = [(i,j,k) for i in range(n) for j in range(i+1,n) for k in range(j+1,n)]
    result = []
    for ini in idx:
        locs = bc_short[list(ini)]
        params0[ii+1] = locs
        res = optimize.fmin(loglike, params0, args=(xx,), maxfun=5000,
                               full_output=1)
        params = res[0]
        print params[2*ii+1],
        kfact = params[2*ii+1] / (2*np.pi)
        print params[2*ii+1] - np.round(kfact) * (2*np.pi)
        params[2*ii+1] -= np.round(kfact) * (2*np.pi)
        result.append(np.concatenate(([res[1]], params)))


    #Note: estimation of location does not restrict to interval (-pi, pi)
    #I think, shifting loc to the interval is equivalent model,
    #eg. np.cos(-4.32514427+2*np.pi) or np.cos(4.73489643-2*np.pi)