import os

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.gof as gof

from dist_mixtures.base import Struct
import dist_mixtures.mixture_von_mises as mvm

import analyses.transforms as tr
import analyses.ioutils as io
import analyses.plots as pl

class DataSource(Struct):

    def __init__(self, get_filenames, n_merge_bins=None, plot_bins_step=4,
                 spread_data=False, neg_shift=True):
        Struct.__init__(self, get_filenames=get_filenames,
                        n_merge_bins=n_merge_bins,
                        plot_bins_step=plot_bins_step,
                        spread_data=spread_data, neg_shift=neg_shift)
        self.current = Struct(filenames=None, dir_base=None, base_names=None)
        self.source_data = Struct(data=None, fdata=None, bins=None)

    def __call__(self):
        for filenames in self.get_filenames:
            dir_base, base_names = io.split_dir_base(filenames)
            print '======================================================'
            print 'directory base:',  dir_base

            data = io.load_data(filenames)
            if self.n_merge_bins is not None:
                data = tr.merge_bins(data, self.n_merge_bins)
            print 'angles range:', data[:, 0].min(), data[:, 0].max()

            data = tr.fix_increasing(tr.fix_range(tr.transform_2pi(data)))

            print 'transformed angles range:', data[0, 0], data[-1, 0]
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
            bins = all_bins[::self.plot_bins_step]

            self.current = Struct(filenames=filenames, dir_base=dir_base,
                                  base_names=base_names)
            self.source_data = Struct(counts=counts,
                                      data=data, fdata=fdata, bins=bins)

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
    from analyses.logs import create_logs, AnglesCSVLog

    logs = create_logs(psets)

    if options.area_angles:
        alog = AnglesCSVLog(os.path.join(psets[0].output_dir,
                                         'log_angles.csv'))
        alog.write_header()

    else:
        alog = None

    for source_data in source():
        if options.area_angles:
            from analyses.area_angles import get_area_angles
            data, _, _ = source.get_source_data()
            area_angles = get_area_angles(data, mode='max')
            alog.write_row(source.current.dir_base, source.current.base_names,
                           area_angles)

        else:
            area_angles = None

        pl.plot_raw_data(psets[0].output_dir, source, area_angles=area_angles)

        # Loop over parameter sets. Each has its own CSVLog.
        res = None
        for ii, pset in enumerate(psets):
            print '------------------------------------------------------'
            print pset
            assert((len(pset.parameters) % 2) == 0)
            if (ii > 0) and pset.parameters == 'previous':
                start_parameters = get_start_params(pset.n_components,
                                                    res.params)

            else:
                zz = np.zeros(pset.n_components - 1, dtype=np.float64)
                start_parameters = get_start_params(pset.n_components,
                                                    np.r_[pset.parameters, zz],
                                                    len(pset.parameters) / 2)
            print 'starting parameters:', start_parameters

            res = fit(source_data, start_parameters,
                      pset.model_class, pset.solver_conf)
            res.model.summary_params(res.params,
                                     name='%d components' % pset.n_components)

            pl.plot_estimated_dist(pset.output_dir, res, source, ii)
            pl.plot_histogram_comparison(pset.output_dir, res, source, ii)

            log = logs[ii]
            log_results(log, res, source)

            if options.show:
                plt.show()

    return logs, alog

def get_start_params(n_components, params=None, ncp=None):
    start_params = np.zeros(n_components * 3 - 1)
    n2 = 2 * n_components
    if params is None:
        # Zeros for mu, twos for kappa.
        start_params[:n2:2] = 2.0

    else:
        if isinstance(params, str):
            params = [float(ii) for ii in params.split(',')]

        params = np.asarray(params)
        # Number of components in params.
        if ncp is None:
            ncp = (len(params) - 2) / 3 + 1

        if ncp < n_components:
            nn = 2 * ncp
            start_params[:nn:2] = params[:nn:2] # kappa.
            start_params[1:nn:2] = params[1:nn:2] # mu.
            start_params[nn:n2:2] = params[nn-2] # kappa.
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

    mod.source_data = source_data

    np.random.seed(12344321)
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
    chisquare = result.gof_chisquare()

    # Chisquare test with effect size.
    alpha = 0.05 # Significance level.
    data = source.source_data.data
    n_obs = data[:, 1].sum()
    rad_diff = data[1, 0] - data[0, 0]

    pdf = result.model.pdf_mix(result.params, data[:, 0])
    probs = pdf * rad_diff * n_obs
    effect_size = gof.chisquare_effectsize(data[:, 1], probs)
    chi2 = gof.chisquare(data[:, 1], probs, value=effect_size)
    power = gof.chisquare_power(effect_size, n_obs,
                                data.shape[0], alpha=alpha)

    chisquare_all = list(chisquare) + [n_obs, effect_size] \
                    + list(chi2) + [power]

    log.write_row(source.current.dir_base, source.current.base_names,
                  chisquare_all, sparams, converged, fit_criteria)

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

        summary[dir_base] = Struct(sorted=zip(pset_ids, converged, probs),
                                   criteria=criteria, pset_ids=pset_ids,
                                   chisquare=[item.chisquare for item in items])
    return summary

_summary_header = """
Summary of results
------------------

Sorted parameter set ids for each criterion are given. Solver convergence is
denoted with '*'. Component probabilities for each criterion are given as well.

For each data directory, a header is printed and then rows sorted by criteria:

- header:

  dir_base | no. of parameter sets
  id       | chi2(e) | chi2(e) p-value | chi2(e) power | llf | aic | bic values
           for each parameter set

- rows:

  id(llf) | id(aic) | id(bic) | id(llf) probs | id(aic) probs | id(bic) probs
"""

def print_summary(summary, logs):
    """
    Print a summary table.
    """
    print _summary_header

    dir_bases = [item.dir_base for item in logs[0].items]
    max_len = reduce(max, (len(ii) for ii in dir_bases), 0)
    head1 = '%%%ds | %%2d parameter sets' % max_len
    head2 = '%%%dd | %%.2e | %%.2f | %%.2f | %%.6e | %%.6e | %%.6e' % max_len
    row = '  %2d%1s | %2d%1s | %2d%1s | %s'

    star = {False : '', True : '*'}
    for idb, dir_base in enumerate(dir_bases):
        items = summary[dir_base]

        n_psets = len(logs)
        print head1 % (dir_base, n_psets)
        for ii in xrange(n_psets):
            # Prevent printing of too small numbers.
            chisquare = np.asarray(items.chisquare[ii])[4:]
            aux = np.where(chisquare < 1e-99, 0.0, chisquare)
            print head2 % ((ii,) + tuple(aux)
                           + tuple(items.criteria[ii]))

        for _item in items.sorted:
            item = zip(*_item)
            aux = []
            for ii in item:
                aux.extend([ii[0], star[ii[1]]])

            aux2 = []
            for ii in item:
                probs = ii[2]
                aux2.append((' '.join(['%.2f'] * len(probs))) % tuple(probs))
            aux2 = ' | '.join(aux2)

            print row % (tuple(aux) + (aux2,))

        if (idb + 1) < len(dir_bases):
            print '-' * 79

_angles_header = """
Area angles
-----------
dir_base, x0, xm, x1, area1, area2
"""
def print_angles(log):
    print _angles_header
    for item in log.items:
        print item.dir_base, item.x0, item.xm, item.x1, item.area1, item.area2
