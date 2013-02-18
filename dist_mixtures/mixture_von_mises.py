# -*- coding: utf-8 -*-
"""

Created on Mon Aug 27 17:58:50 2012

Author: Josef Perktold
Author: Robert Cimrman (est_von_mises)

"""

import numpy as np
from scipy import stats, special, optimize

from statsmodels.base.model import (GenericLikelihoodModel,
                                    GenericLikelihoodModelResults)

from dist_mixtures.base import Struct

def _split_params(params):
    '''split mixture parameter into components

    helper function
    split out for testing
    '''
    npar=2  #hard coded: 1 shape parameter plus loc per mixture
    #number of distributions in mixture:
    k_dist = (len(params) - npar) / (npar + 1) + 1
    probs = params[npar * k_dist:]
    shapes = params[: npar * k_dist : 2]
    locs = params[1 : npar * k_dist : 2]

    return shapes, locs, probs

def shift_loc(x):
    '''shift x into interval (-pi, pi) modulo 2 pi
    '''
    return np.remainder(x + np.pi, 2*np.pi) - np.pi

def normalize_params(params):
    '''shift location and shape parameters

    location will be in interval (-pi, pi) modulo 2 pi
    shape parameter kappa will be non-negative

    This leaves a circular distribution unchanged. Returns a copy no inplace.

    Parameters
    ----------
    params : array_like
       assumes parameterization of mixture distribution
       (kappa1, loc1, kappa2, loc2, ..., logitprob1, ...)

    Returns
    -------
    params_new : ndarray
       same structure as params, with standardized domain
    '''
    n_components, rem = divmod(len(params) + 1, 3)
    if rem != 0:
        raise ValueError('wrong number of parameters for mixture')
    params_new = np.array(params, copy=True)
    #concentration kappa
    params_new[:n_components*2:2] = np.abs(params_new[:n_components*2:2])
    #loc
    params_new[1:n_components*2:2] = shift_loc(params_new[1:n_components*2:2]
                        - np.pi * (np.sign(params[:n_components*2:2])==-1))
    return params_new


def pdf_vn(x, b, loc):
    '''pdf of von mises distribution

    has fixed scale=1, otherwise same as scipy.stats.vonmises

    '''
    return np.exp(b*np.cos(x-loc)) / (2. * np.pi * special.i0(b))

def cdf_vn(x, b, loc):
    '''cdf of von mises distribution

    has fixed scale=1, otherwise same as scipy.stats.vonmises

    origin of cdf (x: cdf(x)==0) is not adjusted for loc

    '''
    return stats.vonmises._cdf(x-loc, b)

def est_von_mises(data, method='mle'):
    '''estimate kappa and mu of Von Mises distribution
    '''
    #Author: Robert Cimrman, changes Josef Perktold
    from scipy.special import i0, i1

    nobs = data.shape[0]

    zn = np.exp(data * 1j)
    avg_z = zn.mean()
    loc = np.angle(avg_z)

    z2 = (avg_z * avg_z.conjugate()).real

    z = np.sqrt(z2)
    def fun_mle(x):
        val = z - i1(x) / i0(x)
        return val

    nobs_inv = 1. / nobs
    nobs_ratio = (nobs - 1.) / nobs
    def fun_correction(x):
        val = nobs_ratio * (z - nobs_inv) - i1(x) / i0(x)
        return val

    if method == 'mle':
        fun = fun_mle
    else:
        fun = fun_correction

    kappa = optimize.newton(fun, 1.0)

    return kappa, loc


def get_probs(params):
    '''extract multinomial logit parameters and convert to probabilites

    Parameters
    ----------
    params : array_like
        full array of parameters, assuming 2 parameters per component
        distribution, and the last parameters are the probability parameters

    Returns
    -------
    probs : ndarray
        probability vector for all components, adds to 1

    '''
    k_dist = (len(params) - 2) / 3 + 1
    p_params = np.concatenate((params[2 * k_dist :], [0]))
    probs = np.exp(p_params)
    probs /= probs.sum()
    return probs

def get_probs2(params):
    '''convert multinomial logit parameters to probabilites

    Parameters
    ----------
    params : array_like
        assumes the array has length equal to the number of components minus
        one.

    Returns
    -------
    probs : ndarray
        probability vector for all components, adds to 1

    '''
    p_params = np.concatenate((params, [0]))
    probs = np.exp(p_params)
    probs /= probs.sum()
    return probs

def pdf_mix(params, x):
    '''pdf of a mixture of von mises distributions
    '''
    k_dist = (len(params) - 2) / 3 + 1  #number of distributions in mixture
    p_params = np.concatenate((params[2 * k_dist:], [0]))
    probs = np.exp(p_params)
    probs /= probs.sum()
    llf = 0
    for ii in range(k_dist):
        llf += probs[ii] * pdf_vn(x, params[ii*2], params[ii*2+1])

    return llf

def loglike(params, x):
    '''loglikelihood of a mixture of von mises distributions
    '''
    k_dist = (len(params) - 2) / 3 + 1  #number of distributions in mixture
    p_params = np.concatenate((params[2 * k_dist :], [0]))
    probs = np.exp(p_params)
    probs /= probs.sum()
    llf = 0
    for ii in range(k_dist):
        llf += probs[ii] * pdf_vn(x, params[ii*2], params[ii*2+1])

    return -np.log(llf).sum()


def fit(xx, params0):
    '''call to fmin to maximize loglikelihood

    trial version
    '''
    res = optimize.fmin(loglike, params0, args=(xx,), maxfun=5000,
                               full_output=1)
    return res[0], res[1], res[1:]


from scipy.stats.kde import gaussian_kde
class GaussianKDE(gaussian_kde, Struct):
    '''A subclass of gaussian_kde that allows flexible choice of bandwidth

    Is not necessary with scipy >= 0.10.

    Parameters
    ----------
    dataset : ndarray, 1-D
        original data
    bwtransform : tuple of 2 floats (a, b)
        affine transformation of the bandwidth define by scotts_factor:
        covariance_factor = a + b * scotts_factor

    Wtih a=0 and b=0.5, we get a smaller bandwidth that is more appropriate
    for multimodal distributions.

    '''

    def __init__(self, dataset, bwtransform=(0,1)):
        self.bwtransform = bwtransform
        super(GaussianKDE, self).__init__(dataset)

    def covariance_factor(self):
        a, b = self.bwtransform
        return a + b * self.scotts_factor()





class VonMisesMixture(GenericLikelihoodModel, Struct):
    '''class to estimate a finite mixture of Von Mises distributions

    fit and results are inherited from GenericLikelihoodModel

    This assumes we have a mixture of 2 parameter Von Mises distributions.
    The probabilities are parameterized with a multinomial logit.

    The number of components is only identified by the start_params in the
    call of the ``fit`` method. The length or number of parameters is
    2*k + (k-1) where k is the number of components, params is the vector

    (b_1, loc_1, b_2, loc_2, ..., b_k, loc_k, gamma_1,..., gamma_{k-1})

    where b_i and loc_i are the parameters of the i-th distribution, and

        p_i = exp(gamma_i) / (sum_{j}(exp(gamma_j))  with gamma_k=0, is the

    probability for component i in the mixture.

    See Also
    --------
    statsmodels.base.model.GenericLikelihoodModel
    statsmodels.base.model.GenericLikelihoodResults

    '''
    def fit(self, start_params=None, method='bfgs', maxiter=500, full_output=1,
            disp=1, callback=None, retall=0, **kwargs):
        """
        Call GenericLikelihoodModel.fit() and normalize result parameters.
        """
        result = super(VonMisesMixture, self).fit(
            start_params=start_params, method=method, maxiter=maxiter,
            full_output=full_output, disp=disp, callback=callback,
            retall=retall, **kwargs)
        result.params = normalize_params(result.params)

        #this is a bit different from the way results are updated in discrete
        #TODO: check this
        result = MixtureResult(self, result)

        return result

    def pdf_mix(self, params, x=None, return_comp=False):
        '''pdf of a mixture of von mises distributions
        '''

        if x is None:
            x = self.endog

        k_dist = (len(params) - 2) / 3 + 1  #number of distributions in mixture
        probs = get_probs2(params[2*k_dist:])

        pdf_ = np.zeros(x.shape)
        if return_comp:
            pdf_d = []
        for ii in range(k_dist):
            pdf_i = probs[ii] * pdf_vn(x, params[ii*2], params[ii*2+1])
            pdf_ += pdf_i
            if return_comp:
                pdf_d.append(pdf_i)

        if return_comp:
            return pdf_, pdf_d
        else:
            return pdf_

    def cdf_mix(self, params, x=None, return_comp=False):
        '''cdf of a mixture of von mises distributions
        '''

        params = np.asarray(params)


        if x is None:
            x = self.endog
        else:
            #x = np.atleast_1d(x)   #TODO: needed or not?
            x = np.asarray(x)

        k_dist = (len(params) - 2) / 3 + 1  #number of distributions in mixture
        probs = get_probs2(params[2*k_dist:])

        #vonmises cdf can have negative values with loc outside (-pi,pi)
        #>>> stats.vonmises.cdf(-np.pi+1e-6, 180.07281913, loc=3.6730781)
        #-0.99999999999906264
        locs = params[1:2*k_dist:2]
        kfact = locs / (2*np.pi)
        locs -= np.round(kfact) * (2*np.pi)

        cdf_ = np.zeros(x.shape)
        if return_comp:
            cdf_d = []
        for ii in range(k_dist):
            #normalize to cdf(-np.pi) = 0
            cdf_component = (cdf_vn(x, params[ii*2], locs[ii]) -
                             cdf_vn(-np.pi, params[ii*2], locs[ii]))

            cdf_i = probs[ii] * cdf_component
            cdf_ += cdf_i
            if return_comp:
                cdf_d.append(cdf_i)

        if return_comp:
            return cdf_, cdf_d
        else:
            return cdf_

    def rvs_mix(self, params, size=100, ret_sizes=False, fixed_size=False,
                shuffle=False):
        '''Random variates of the mixture distribution.
        '''
        k_dist = (len(params) - 2) / 3 + 1 #number of distributions in mixture

        probs = get_probs2(params[2 * k_dist:])
        if fixed_size:
            sizes = np.ceil(probs[:k_dist] * size).astype(np.int32)
            nr = sizes.sum()
            if  nr > size:
                sizes[-1] -= (nr - size)

        sizes = np.random.multinomial(size, probs, size=1)
        sizes = sizes[0]  #return of multinomial is 2d, need 1d

        rvs = []
        for ii in range(k_dist):
            rvs.append(stats.vonmises.rvs(params[2*ii],
                                          loc=params[2*ii+1],
                                          size=sizes[ii]))
        rvs = np.concatenate(rvs)
        if shuffle:
            np.random.shuffle(rvs)

        #shift into (-pi,pi)  use helper function
        rvs = np.remainder(rvs + np.pi, 2 * np.pi) - np.pi

        if ret_sizes:
            return rvs, sizes
        else:
            return rvs

    def loglikeobs(self, params, x=None):
        '''loglikelihood of observations of a mixture of von mises distributions
        '''
        return np.log(self.pdf_mix(params, x=x))

    def loglike(self, params, x=None):
        '''loglikelihood of a mixture of von mises distributions
        '''
        return self.loglikeobs(params, x=x).sum()

    #overwrite because default is not very good, will be changed
    def score(self, params):
        '''
        Gradient of log-likelihood evaluated at params
        '''
        from statsmodels.tools.numdiff import approx_fprime
        return approx_fprime(params, self.loglike, epsilon=1e-4,
                             centered=True).ravel()

    #overwrite because default is not very good, will be changed
    def hessian(self, params):
        '''Hessian of loglikelihood calculated by numerical differentiation

        Note: import requires statsmodels master
        '''
        from statsmodels.tools.numdiff import approx_hess1
        # need options for hess (epsilon)
        return approx_hess1(params, self.loglike)


    #The following methods should be attached to the result instance, but I
    #don't have access to it without overwriting the fit method

    def get_summary_params(self, params):
        '''helper function to return the parameters (easier to interpret)

        '''
        #temporary
        pr = get_probs(params)
        k_dist = (len(params) - 2) / 3 + 1

        out = np.array([(params[2*ii], params[2*ii+1], pr[ii])
                        for ii in range(k_dist)])
        return out

    def summary_params(self, params, name=''):
        '''helper function to print the parameters (easier to interpret)

        '''
        sparams = self.get_summary_params(params)

        if name:
            postfix = ' (%s)'%name
        else:
            postfix = ''

        print '\nEstimated distributions%s' % postfix
        for ii, pp in enumerate(sparams):
            print 'dist%1d: shape=%6.4f, loc=%6.4f, prob=%6.4f' \
                  % ((ii,) + tuple(pp))

    def plot_dist(self, params, plot_kde=False, xtransform=None, bins=None,
                  n_bins=50, data=None):
        '''plot the pdf given parameters and histogram and kernel estimate

        helper for visual evaluation of fit and of components

        '''
        import matplotlib.pyplot as plt
        x0 = np.linspace(-np.pi, np.pi, 181)
        if bins is None:
            bins = n_bins

        data = data if data is not None else self.endog

        if xtransform is None:
            x0t = x0
            ip = slice(0, len(x0))

        else:
            x0t = xtransform(x0)
            ip = np.argsort(x0t)
            x0t = x0t[ip]

        fig = plt.figure()

        pdf_m, pdf_d = self.pdf_mix(params, x0, return_comp=True)
        plt.plot(x0t, pdf_m[ip], lw=2, label='mixture')

        if plot_kde:
            kde = GaussianKDE(data, (0, 0.5))
            pdf_kde = kde(x0)
            plt.plot(x0t, pdf_kde[ip], lw=2, label='kde')

        for ii, pdf_i in enumerate(pdf_d):
            plt.plot(x0t, pdf_i[ip], lw=2, label='dist%d' % ii)

        _, _, patches = plt.hist(data, bins=bins, normed=True,
                                 alpha=0.2, color='b')
        if xtransform is not None:
            for patch in patches:
                x, w = patch.get_x(), patch.get_width()
                x1t, x2t = xtransform([x, x + w])
                if x1t > x2t:
                    x1t = x
                patch.set_x(x1t)
                patch.set_width(x2t - x1t)

        ax = plt.gca()
        ax.set_xlim([x0t[0], x0t[-1]])
        plt.legend(loc='best')

        return fig

    def plot_cdf(self, params, others=None, names=None):
        '''plot the cdf given parameters and the empirical cdf

        helper for visual evaluation of fit and comparison across different
        estimates

        Parameters
        ----------
        params : ndarray
            estimated parameters
        others : list of result instances
            only the params attribute of each result instance is used
        names : list of strings
            names used in the legend for the plot, including the name of the
            current (self) instance given by params.

        Returns
        -------
        fig : matplotlib figure instance

        '''
        import matplotlib.pyplot as plt
        x0 = np.linspace(-np.pi, np.pi, 51)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        if others is None:
            cdf_m = self.cdf_mix(params, x0, return_comp=False)
            ax.plot(x0, cdf_m, lw=2, label='mixture')
        else:
            if names is None:
                names = ['self'] + ['oth %d'% i for i in range(len(others))]

            cdf_m = self.cdf_mix(params, x0, return_comp=False)
            ax.plot(x0, cdf_m, lw=2, label='cdf-%s' % names[0])

            for name, res in zip(names[1:], others):
                cdf_m = self.cdf_mix(res.params, x0, return_comp=False)
                ax.plot(x0, cdf_m, lw=2, label='cdf-%s' % name)

        nobs = self.endog.shape[0]
        plotpos = np.arange(nobs) / float(nobs)
        ax.step(np.sort(self.endog), plotpos, lw=2, label='ecdf')
        ax.set_xlim(-np.pi, np.pi)
        plt.legend()
        return fig

class VonMisesMixtureBinned(VonMisesMixture):
    '''Maximum Likelihood model for binned data for Von Mises mixture

    TODO: names and meaning of parameters, arguments in __init__ needs to be
        changed.
        No clear pattern yet for case weights in Maximum Likelihood Models

    Parameters
    ----------
    endog : ndarray, 1-D, (nobs,)
        weights, number of obwervation in bins
    exog : ndarray, 1-D (nobs+1)
        bin edges, this is stored in attribute ``bins``
        the attribute exog will contain the bin centers


    '''

    def __init__(self, endog, exog):

        self.bins = exog
        bin_centers = 0.5 * (exog[1:] + exog[:-1])
        super(VonMisesMixtureBinned, self).__init__(endog, bin_centers)


    def loglikeobs(self, params, x=None):
        '''loglikelihood of observations of a mixture of von mises distributions

        based on binned data
        assume exog contains bins, endog counts
        problem bin-edges has one more observation
        '''

        params_ = normalize_params(params)
        cdf_bins = self.cdf_mix(params_, x=self.bins)
        pdf_bins = np.clip(np.diff(cdf_bins), 1e-20, 1)
        return self.endog * np.log(pdf_bins)

    def pmf_bins(self, params):
        '''probability mass function of binned data
        '''

        #normalized params, no negative kappa. needed for estimation
        params_ = normalize_params(params)
        cdf_bins = self.cdf_mix(params_, x=self.bins)
        pdf_bins = np.clip(np.diff(cdf_bins), 1e-20, 1)
        return pdf_bins


    def fit(self, start_params=None, method='bfgs', maxiter=500, full_output=1,
            disp=1, callback=None, retall=0, **kwargs):
        result = super(VonMisesMixtureBinned, self).fit(
                    start_params=start_params, method=method, maxiter=maxiter,
                    full_output=full_output, disp=disp, callback=callback,
                    retall=retall, **kwargs)

        return result


    #added to results instance without params, calls this
    def gof_chisquare(self, params, fac=1.):
        '''chisquare goodness-of-fit test

        try ``fac`` when we are unsure about sample size

        '''

        counts = self.endog
        pmf_bins = self.pmf_bins(params)

        return stats.chisquare(counts*fac, counts.sum() * fac * pmf_bins,
                               ddof=len(params)) #[:-1])

    def fit_ls(self, start_params=None):
        '''estimate parameters by histogram fitting

        simple least squares, does not use weights
        maybe more sensitive to starting values

        '''
        #Warning: endog can be integers
        freq = self.endog * 1. / self.endog.sum()
        func = lambda params : 100 * (freq - self.pmf_bins(params))**2
        res = optimize.leastsq(func, start_params, full_output=True,
                                #maxfev=5000
                                #epsfcn=1e-3, factor=0.1
                                )

        #TODO: warnings.warn ?
        if res[-1] not in [1]:
            print "Warning: leastsq not converged with 1"
            print res[-2]

        return res


class MixtureResult(GenericLikelihoodModelResults):
    '''incomplete trial version for results class

    mainly a stub,
    Caution: not all inherited methods will make sense in the binned model

    '''

    def gof_chisquare(self, fac=1.):
        return self.model.gof_chisquare(self.params, fac=fac)
