import numpy as np

from dist_mixtures.mixture_von_mises import VonMisesMixture

import analyses.transforms as tr

def get_start_params(n_components, params):
    start_params = np.zeros(n_components * 3 - 1)
    n2 = 2 * n_components
    if params is None:
        # Zeros for mu, twos for kappa.
        start_params[:n2:2] = 2.0

    else:
        aux = np.array([float(ii) for ii in params.split(',')])
        start_params[:n2:2] = aux[1::2] # kappa.
        start_params[1:n2:2] = tr.transform_2pi(aux[0::2]) # mu.

    return start_params

def fit(data, start_params):
    '''create VonMisesMixture instance and fit to data
    '''
    mod = VonMisesMixture(data)
    #with master of statsmodels we can use bfgs
    #because the objective function is now normalized with 1/nobs we can
    #increase the gradient tolerance gtol
    #res = mod.fit(start_params=start_params, method='nm', disp=True)
    #res = mod.fit(start_params=res.params, method='bfgs', disp=True) #False)
    res = mod.fit(start_params=start_params, method='bfgs', disp=True,
                  gtol=1e-9) #False)

    return res
