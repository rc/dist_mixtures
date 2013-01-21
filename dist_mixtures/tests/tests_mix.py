# -*- coding: utf-8 -*-
"""

Created on Thu Jan 17 20:14:26 2013

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_array_less

from scipy import stats

import dist_mixtures.mixture_von_mises as mixvn

def test_split_params():
    npar = 2
    for k_dist in range(1, 5):
        nparams = npar * k_dist + k_dist - 1
        params = np.arange(nparams)
        sh, loc, pr = mixvn._split_params(params)

        m, r = divmod(sh, 2)
        assert_equal(r, np.zeros(k_dist))
        assert_equal(m.max(), k_dist - 1)

        m, r = divmod(loc, 2)
        assert_equal(r, np.ones(k_dist))
        assert_equal(m.max(), k_dist - 1)

        assert_equal(len(pr), k_dist - 1)

def test_normalize():
    np.random.seed(123)
    npar = 2
    k = 3
    x = np.linspace(-k*np.pi, k*np.pi, 21)
    vnm = mixvn.VonMisesMixture(x)
    for k_dist in range(1, 5):
        nparams = npar * k_dist + k_dist - 1
        for _ in range(10):
            params = np.random.uniform(-5, 5, size=nparams)
            params[npar * k_dist:] /= 10
            pdf1 = vnm.pdf_mix(params)
            params2 = mixvn.normalize_params(params)
            #most params are changed
            #print np.max(np.abs(params - params2)) > 0.001,
            pdf2 = vnm.pdf_mix(params2)
            assert_almost_equal(pdf2, pdf1, decimal=14)


def test_mix1():
    np.random.seed(987789)
    nobs = 10
    params_li = [[2., np.pi/4],
                 [2., np.pi/4, 2., np.pi/4, 0],  #mix of 2 identical
                 [2., np.pi/4, 1., np.pi/8, 50], #essentially all weight on 1
                 [1., np.pi/8, 2., np.pi/4, -50], #essentially all weight on 2
                 ]
    sh, loc = 2., np.pi/4
    for params in params_li:
        mod2 = mixvn.VonMisesMixture(np.random.uniform(-np.pi, np.pi, size=nobs))
        x = np.linspace(-np.pi, np.pi, 6)

        p1 = mod2.pdf_mix(params, x)
        p2 = stats.vonmises.pdf(x, sh, loc=loc)
        assert_almost_equal(p1, p2, decimal=14, err_msg=repr(params))

        c1 = mod2.cdf_mix(params, x)
        c2 = stats.vonmises.cdf(x, sh, loc=loc)
        assert_almost_equal(c1, c2 - c2[0], decimal=14, err_msg=repr(params))
        #test normalization separately

def test_vonmisesmixture():
    #np.random.seed(987789)  #TODO: add seed later, random failure without
    #values of initialization not used
    mod2 = mixvn.VonMisesMixture(np.random.uniform(-np.pi, np.pi, size=10))
    params = [2., -0.75 * np.pi, 4., np.pi/2, 0.4]
    nobs = 50000
    rvs = mod2.rvs_mix(params, size=nobs, shuffle=True)
    assert_equal(len(rvs), nobs)

    #check withing bounds
    above = (rvs > np.pi).sum()
    below = (rvs < -np.pi).sum()
    assert_equal(above, 0)
    assert_equal(below, 0)

    #gof tests
    bins = 180
    bins = np.linspace(-np.pi, np.pi, bins+1)
    #count, bins_ = np.histogram(rvs, bins=bins, normed=True)
    #freq = count * np.diff(bins_)

    count, bins_ = np.histogram(rvs, bins=bins)
    freq = count / count.sum()
    assert_equal(count.sum(), len(rvs))

    ks = stats.kstest(rvs, lambda x: mod2.cdf_mix(params, x))
    assert_array_less(0.1, ks[1])

    c1 = mod2.cdf_mix(params, bins)
    p0 = np.diff(c1)
    chi2 = stats.chisquare(count, p0 * nobs)
    assert_array_less(0.1, chi2[1])

    mse = ((freq - p0)**2).mean()
    assert_array_less(mse, 1e-4)

    #bin_center = bins[:-1] + np.diff(bins) / 2

    #more pdf, cdf checks
    p2 = mod2.pdf_mix(params, bins)
    from scipy import integrate
    c2 = integrate.cumtrapz(p2, dx=bins[1]-bins[0])
    assert_almost_equal(c2, c1[1:], decimal=4)  #approximation error

    #check wrapping to [-np.pi, np.pi]  #TODO: open interval ?


    assert_almost_equal(mod2.cdf_mix(params, -np.pi), 0, decimal=13)
    assert_almost_equal(mod2.cdf_mix(params,  np.pi), 1, decimal=13)
    c3 = mod2.cdf_mix(params, bins + 2 * np.pi)
    assert_almost_equal(c3 - 1, c1, decimal=13)
    c3 = mod2.cdf_mix(params, bins - 2 * np.pi)
    assert_almost_equal(c3 + 1, c1, decimal=13)

    p1 = mod2.pdf_mix(params, bins)
    p3 = mod2.pdf_mix(params, bins + 2 * np.pi)
    assert_almost_equal(p3, p1, decimal=13)
    p3 = mod2.pdf_mix(params, bins - 2 * np.pi)
    assert_almost_equal(p3, p1, decimal=13)

    #check standalone functions
    p4 = mod2.pdf_mix(params, bins * 4)
    pf1 = mixvn.pdf_mix(params, bins * 4)
    assert_almost_equal(pf1, p4, decimal=13)

    pvn = mixvn.pdf_vn(bins * 4, params[0], params[1])
    psp = stats.vonmises.pdf(bins * 4, params[0], params[1])
    assert_almost_equal(pvn, psp, decimal=13)

    #periodicity of cdf with trend, origin not fixed
    cvn = mixvn.cdf_vn(bins, params[0], params[1])
    cvn3 = mixvn.cdf_vn(bins + 2 * np.pi, params[0], params[1])
    assert_almost_equal(cvn3 - 1, cvn, decimal=13)
    cvn3 = mixvn.cdf_vn(bins - 2 * np.pi, params[0], params[1])
    assert_almost_equal(cvn3 + 1, cvn, decimal=13)
    assert_almost_equal(pvn, psp, decimal=13)

    #test fit
    mod3 = mixvn.VonMisesMixture(rvs[:2000])
    #good starting values
    res3 = mod3.fit(start_params=np.array(params)*1.1)
    res3.params = mixvn.normalize_params(res3.params)
    assert_almost_equal(res3.params, params, decimal=1)

    #simple starting values, refit same model instance
    res3 = mod3.fit(start_params=0.5*np.ones(len(params)))
    res3.params = mixvn.normalize_params(res3.params)
    assert_almost_equal(res3.params, params, decimal=1)

    ##fit is not much better with full sample (but slower)
    #mod4 = mixvn.VonMisesMixture(rvs)
    #res4 = mod4.fit(start_params=0.5*np.ones(len(params)))
    #res4.params = mixvn.normalize_params(res4.params)
    #assert_almost_equal(res4.params, params, decimal=1)

    #fit with binned data, full sample, 180 bins on (-pi, pi)
    mod5 = mixvn.VonMisesMixtureBinned(count, bins)
    #good starting values
    res5 = mod5.fit(start_params=np.array(params)*1.1)
    res5.params = mixvn.normalize_params(res5.params)
    assert_almost_equal(res5.params, params, decimal=1)
    assert_almost_equal(res5.params, res3.params, decimal=1)

    #LSfit with binned data, full sample, 180 bins on (-pi, pi)
    #mod5 = mixvn.VonMisesMixtureBinned(count.astype(float), bins)

    #good starting values
    res6 = mod5.fit_ls(start_params=np.array(params)*1.1)
    res6_params = mixvn.normalize_params(res6[0])
    assert_almost_equal(res6_params, params, decimal=1)
    assert_almost_equal(res6_params, res5.params, decimal=2)

    #simple starting values
    start_params=0.5*np.ones(len(params))
    start_params[-1] = 0.05
    #TODO: optimize.leastsq can fail with maxfev, scipy cdf with ZeroDivision
    start_params = [-2.5, 4, 0.1, 4, 0.1]
    res6a = mod5.fit_ls(start_params=start_params)
    res6a_params = mixvn.normalize_params(res6a[0])
    assert_almost_equal(res6a_params, params, decimal=1)
    assert_almost_equal(res6_params, res6_params, decimal=4)


if __name__ == '__main__':
    test_split_params()
    test_mix1()
    test_vonmisesmixture()

