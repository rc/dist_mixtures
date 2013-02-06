import numpy as np

from dist_mixtures.base import Struct
from dist_mixtures.mixture_von_mises import VonMisesMixture

import analyses.transforms as tr
import analyses.ioutils as io

class DataSource(Struct):

    def __init__(self, get_filenames, spread_data, neg_shift=True):
        Struct.__init__(self, get_filenames=get_filenames,
                        spread_data=spread_data, neg_shift=neg_shift)
        self.current = Struct(filenames=None, dir_base=None, base_names=None)
        self.state = Struct(data=None, fdata=None, bins=None)

    def __call__(self):
        for filenames in self.get_filenames:
            dir_base, base_names = io.split_dir_base(filenames)
            print '*****'
            print 'directory base:',  dir_base

            data = io.load_data(filenames, transform=tr.transform_2pi)

            print 'data range:', data[:, 1].min(), data[:, 1].max()

            # Simulate the "random process" the histogram was done from.
            counts = tr.get_counts_from_lengths(data[:, 1])
            fdata = tr.spread_by_counts(data[:, 0], counts,
                                        trivial=self.spread_data == False)

            print 'simulated counts range:', counts.min(), counts.max()

            ddata = np.sort(tr.transform_pi_deg(data[:, 0],
                                                neg_shift=self.neg_shift))
            dd = ddata[1] - ddata[0]
            all_bins = np.r_[ddata - 1e-8, ddata[-1] + dd]
            bins = all_bins[::4]

            self.current = Struct(filenames=filenames, dir_base=dir_base,
                                  base_names=base_names)
            self.state = Struct(data=data, fdata=fdata, bins=bins)

            yield self.get_state()

    def get_state(self):
        return self.state.data, self.state.fdata, self.state.bins

def get_start_params(n_components, params=None):
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

def log_results(log, result, source):
    """
    Log the fitting results.
    """
    sparams = result.model.get_summary_params(result.params)[:, [1, 0, 2]]
    sparams[:, 0] = tr.transform_pi_deg(tr.fix_range(sparams[:, 0]),
                                        neg_shift=source.neg_shift)
    flags = ['']
    if not result.mle_retvals['converged']:
        flags[0] = '*'
    print 'flags:', flags

    fit_criteria = [-result.llf, result.aic, result.bic]

    log.write_row(source.current.dir_base, source.current.base_names,
                  sparams, flags, fit_criteria)
