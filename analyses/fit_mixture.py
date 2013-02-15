import numpy as np
import matplotlib.pyplot as plt

from dist_mixtures.base import Struct
import dist_mixtures.mixture_von_mises as mvm

import analyses.transforms as tr
import analyses.ioutils as io
import analyses.plots as pl

class DataSource(Struct):

    def __init__(self, get_filenames, spread_data, neg_shift=True):
        Struct.__init__(self, get_filenames=get_filenames,
                        spread_data=spread_data, neg_shift=neg_shift)
        self.current = Struct(filenames=None, dir_base=None, base_names=None)
        self.source_data = Struct(data=None, fdata=None, bins=None)

    def __call__(self):
        for filenames in self.get_filenames:
            dir_base, base_names = io.split_dir_base(filenames)
            print '======================================================'
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
            self.source_data = Struct(data=data, fdata=fdata, bins=bins)

            yield self.source_data

    def get_source_data(self):
        return (self.source_data.data, self.source_data.fdata,
                self.source_data.bins)

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

    for source_data in source():
        pl.plot_raw_data(psets[0].output_dir, source,
                         area_angles=options.area_angles)

        # Loop over parameter sets. Each has its own CSVLog.
        res = None
        for ii, pset in enumerate(psets):
            print '------------------------------------------------------'
            print pset
            if (ii > 0) and pset.parameters == 'previous':
                start_parameters = get_start_params(pset.n_components,
                                                    res.params)

            else:
                zz = np.zeros(pset.n_components - 1, dtype=np.float64)
                start_parameters = get_start_params(pset.n_components,
                                                    np.r_[pset.parameters, zz])
            print 'starting parameters:', start_parameters

            res = fit(source_data, start_parameters,
                      pset.model_class, pset.solver_conf)
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
        ncp = (len(params) - 2) / 3 + 1
        if ncp < n_components:
            nn = 2 * ncp
            start_params[:nn:2] = params[:nn:2] # kappa.
            start_params[1:nn:2] = params[1:nn:2] # mu.
            start_params[nn:n2:2] = 1.1 * params[nn-2] # kappa.
            start_params[nn+1:n2:2] = params[nn-1] # mu.

        else:
            start_params[:n2:2] = params[:n2:2] # kappa.
            start_params[1:n2:2] = params[1:n2:2] # mu.

    return start_params

def fit(source_data, start_params, model_class, solver_conf):
    '''
    Create VonMisesMixture instance and fit to data.

    Parameters
    ----------
    source_data : Struct with 3 array attributes
        The data to be fitted - raw data, simulated process data, histogram
        plot bins.
    start_params : array of length 3 * n_component - 1
        The vector of starting parameters - shape plus location per component
        and probabilities for all components except the last one. Its length
        determines the number of components.
    model_class : MixtureVonMises or VonMisesMixtureBinned
        The class used for the fitting.
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
    if model_class == mvm.VonMisesMixture:
        assert np.isfinite(mvm.loglike(start_params,
                                       source_data.fdata)).all()
        mod = mvm.VonMisesMixture(source_data.fdata)

    elif model_class == mvm.VonMisesMixtureBinned:
        assert np.isfinite(mvm.loglike(start_params,
                                       source_data.data[:, 1])).all()

        bins_exog = np.linspace(-np.pi, np.pi, source_data.data.shape[0] + 1)
        mod = mvm.VonMisesMixtureBinned(source_data.data[:, 1], bins_exog)

    res = mod.fit(start_params=start_params, method=solver_conf[0],
                  **solver_conf[1])

    return res

def log_results(log, result, source):
    """
    Log the fitting results.

    Notes
    -----
    The resulting mixture parameters are stored into a 2d array with rows
    [location in degrees (mu), shape (kappa), probability].
    """
    sparams = result.model.get_summary_params(result.params)[:, [1, 0, 2]]
    sparams[:, 0] = tr.transform_pi_deg(tr.fix_range(sparams[:, 0]),
                                        neg_shift=source.neg_shift)
    converged = result.mle_retvals['converged']

    fit_criteria = [-result.llf, result.aic, result.bic]

    log.write_row(source.current.dir_base, source.current.base_names,
                  sparams, converged, fit_criteria)

def print_results(psets, logs):
    """
    Print fitting results and corresponding parameter sets.
    """
    print '######################################################'
    for ii, log in enumerate(logs):
        print '======================================================'
        print psets[ii]
        print log
        for item in log.items:
            print '------------------------------------------------------'
            print item

def make_summary(logs):
    """
    Make a summary table of all results.

    The best parameter set according to each criterion is selected.

    Parameters
    ----------
    logs : list of CSVLog
        A log with fitting results for each parameter set.

    Returns
    -------
    summary : dict
        The summary dictionary with data directory base names as keys.
    """
    dir_bases = [item.dir_base for item in logs[0].items]
    summary = {}
    for ii, dir_base in enumerate(dir_bases):
        # Results for dir_base for all parameter sets.
        items = [log.items[ii] for log in logs]

        criteria = [item.fit_criteria for item in items]
        pset_ids = np.array(criteria).argsort(axis=0)
        converged = np.take([item.converged for item in items], pset_ids)

        probs = np.take([item.params[:, 2] for item in items], pset_ids,
                        axis=0)
        summary[dir_base] = zip(pset_ids, converged, probs)

    return summary

_summary_header = """
Summary of results
------------------

Sorted parameter set ids for each criterion are given. Solver convergence is
denoted with '*'. Component probabilities for each criterion are given as well.

Row format: dir_base | llf | aic | bic || llf probs | aic probs | bic probs
"""

def print_summary(summary, logs):
    """
    Print a summary table.
    """
    print _summary_header

    dir_bases = [item.dir_base for item in logs[0].items]
    max_len = reduce(max, (len(ii) for ii in dir_bases), 0)
    row = '%%%ds | %%2d%%1s | %%2d%%1s | %%2d%%1s || %%s' % max_len

    star = {False : '', True : '*'}
    for idb, dir_base in enumerate(dir_bases):
        items = summary[dir_base]

        for _item in items:
            item = zip(*_item)
            aux = []
            for ii in item:
                aux.extend([ii[0], star[ii[1]]])

            aux2 = []
            for ii in item:
                probs = ii[2]
                aux2.append((' '.join(['%.2f'] * len(probs))) % tuple(probs))
            aux2 = ' | '.join(aux2)

            print row % ((dir_base,) + tuple(aux) + (aux2,))

        if (idb + 1) < len(dir_bases):
            print '-' * 79
