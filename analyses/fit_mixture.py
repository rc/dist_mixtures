import numpy as np
import matplotlib.pyplot as plt

from dist_mixtures.base import Struct
from dist_mixtures.mixture_von_mises import VonMisesMixture

import analyses.transforms as tr
import analyses.ioutils as io
import analyses.plots as pl

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

def analyze(source, psets, options):
    """
    Analyze data provided by source using given parameter sets.

    Returns
    -------
    logs : list of CSVLog
        The fitting information and results.
    """
    from analyses.logs import create_logs

    logs = create_logs(psets)

    for data, fdata, bins in source():
        pl.plot_raw_data(psets[0].output_dir, source,
                         area_angles=options.area_angles)

        # Loop over parameter sets. Each has its own CSVLog.
        res = None
        for ii, pset in enumerate(psets):
            print pset
            if (ii > 0) and pset.parameters == 'previous':
                start_parameters = get_start_params(pset.n_components,
                                                    res.params)

            else:
                start_parameters = get_start_params(pset.n_components,
                                                    pset.parameters)
            print 'starting parameters:', start_parameters

            res = fit(fdata, start_parameters, pset.solver_conf)
            res.model.summary_params(res.params,
                                     name='%d components' % pset.n_components)

            pl.plot_estimated_dist(pset.output_dir, res, source)
            pl.plot_histogram_comparison(pset.output_dir, res, source)

            log = logs[ii]
            log_results(log, res, source)

            if options.show:
                plt.show()

    return logs

def get_start_params(n_components, params=None):
    start_params = np.zeros(n_components * 3 - 1)
    n2 = 2 * n_components
    if params is None:
        # Zeros for mu, twos for kappa.
        start_params[:n2:2] = 2.0

    else:
        if isinstance(params, str):
            params = [float(ii) for ii in params.split(',')]

        params = np.asarray(params)
        nn = params.shape[0] - 1
        if nn < n2:
            start_params[:nn:2] = params[:nn:2] # kappa.
            start_params[1:nn:2] = params[1:nn:2] # mu.
            start_params[nn::2] = params[-2] # kappa.
            start_params[nn+1::2] = params[-1] # mu.

        else:
            start_params[:n2:2] = params[:n2:2] # kappa.
            start_params[1:n2:2] = params[1:n2:2] # mu.

    return start_params

def fit(data, start_params, solver_conf):
    '''
    Create VonMisesMixture instance and fit to data.

    Parameters
    ----------
    data : array
        The data to be fitted.
    start_params : array of length 3 * n_component - 1
        The vector of starting parameters - shape plus location per component
        and probabilities for all components except the last one. Its length
        determines the number of components.
    solver_conf : (str, dict)
        Solver configuration tuple consisting of name and options dictionary.

    Returns
    -------
    res : GenericLikelihoodModelResults instance
        The results object.

    Notes
    -----
    With master of statsmodels we can use bfgs because the objective function
    is now normalized with 1/nobs we can increase the gradient tolerance gtol.
    '''
    mod = VonMisesMixture(data)
    res = mod.fit(start_params=start_params, method=solver_conf[0],
                  **solver_conf[1])

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
