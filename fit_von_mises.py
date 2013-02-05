#!/usr/bin/env python
"""
Fit data files with names matching a given pattern by a mixture of von Mises
distributions.
"""
import os
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt

from dist_mixtures.mixture_von_mises import VonMisesMixture
import analyses.ioutils as io
import analyses.transforms as tr
import analyses.plots as pl
from analyses.logs import CSVLog
from analyses.area_angles import get_area_angles

usage = '%prog [options] pattern data_dir output_dir\n' + __doc__.rstrip()

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

help = {
    'n_components' :
    'number of components of the mixture [default: %default]',
    'params' :
    'initial guess of von Mises parameters for each component as a comma'
    ' separated list, e.g., for two components: "0,1,0,1" corresponding'
    ' to mu0, kappa0, mu1, kappa1 respectively. The location parameter'
    ' mu should be given in degrees in [-90, 90[.',
    'dir_pattern' :
    'pattern that subdrectories should match [default: %default]',
    'spread_data' :
    'spread raw data using their counts instead of just repeating them',
    'area_angles' :
    'compute and draw angles of two systems of fibres determined by'
    ' equal histogram areas',
    'show' :
    'show the figures',
}

def main():
    parser = OptionParser(usage=usage, version='%prog')
    parser.add_option('-n', '--n-components', type=int, metavar='positive_int',
                      action='store', dest='n_components',
                      default=2, help=help['n_components'])
    parser.add_option('-p', '--pars', metavar='mu0,kappa0,mu1,kappa1,...',
                      action='store', dest='params',
                      default=None, help=help['params'])
    parser.add_option('-d', '--dir-pattern', metavar='pattern',
                      action='store', dest='dir_pattern',
                      default='*', help=help['dir_pattern'])
    parser.add_option('', '--spread-data',
                      action='store_true', dest='spread_data',
                      default=False, help=help['spread_data'])
    parser.add_option('-a', '--area-angles',
                      action='store_true', dest='area_angles',
                      default=False, help=help['area_angles'])
    parser.add_option('-s', '--show',
                      action='store_true', dest='show',
                      default=False, help=help['show'])
    options, args = parser.parse_args()

    if len(args) == 3:
        pattern, data_dir, output_dir = args
    else:
        parser.print_help()
        return

    start_params = np.zeros(options.n_components * 3 - 1)
    n2 = 2 * options.n_components
    if options.params is None:
        # Zeros for mu, twos for kappa.
        start_params[:n2:2] = 2.0

    else:
        aux = np.array([float(ii) for ii in options.params.split(',')])
        start_params[:n2:2] = aux[1::2] # kappa.
        start_params[1:n2:2] = tr.transform_2pi(aux[0::2]) # mu.

    print 'starting parameters:', start_params

    io.ensure_path(output_dir)

    neg_shift = True

    log_name = os.path.join(output_dir, 'log.csv')
    log = CSVLog(log_name, options.n_components)
    log.write_header()

    get_data = io.locate_files(pattern, data_dir,
                               dir_pattern=options.dir_pattern,
                               group_last_level=True)
    for filenames in get_data:
        aux = filenames[0].split(os.path.sep)
        dir_base = aux[-2] if len(aux) > 1 else ''
        base_names = [os.path.basename(ii) for ii in filenames]

        print '*****'
        print 'directory base:',  dir_base

        data = io.load_data(filenames, transform=tr.transform_2pi)

        print 'data range:', data[:, 1].min(), data[:, 1].max()

        # Simulate the "random process" the histogram was done from.
        counts = tr.get_counts_from_lengths(data[:, 1])
        fdata = tr.spread_by_counts(data[:, 0], counts,
                                    trivial=options.spread_data == False)

        print 'simulated counts range:', counts.min(), counts.max()

        ddata = np.sort(tr.transform_pi_deg(data[:, 0], neg_shift=neg_shift))
        dd = ddata[1] - ddata[0]
        all_bins = np.r_[ddata - 1e-8, ddata[-1] + dd]
        bins = all_bins[::4]

        figname = os.path.join(output_dir, dir_base + '-data.png')
        fig = pl.plot_data(data, fdata, bins, neg_shift=neg_shift)

        if options.area_angles:
            pl.draw_areas(fig.axes[0],
                          *get_area_angles(data, neg_shift=neg_shift))

        fig.savefig(figname)

        res = fit(fdata, start_params)

        res.model.summary_params(res.params,
                                 name='%d components' % options.n_components)

        xtr = lambda x: tr.transform_pi_deg(x, neg_shift=neg_shift)
        rbins = tr.transform_2pi(bins) - np.pi
        fig = res.model.plot_dist(res.params, xtransform=xtr, bins=rbins)
        fig.axes[0].set_title('Estimated distribution')

        figname = os.path.join(output_dir, dir_base + '-fit-%d.png'
                               % options.n_components)
        fig.savefig(figname)

        try:
            rvs, sizes = res.model.rvs_mix(res.params, size=fdata.shape[0],
                                           ret_sizes=True)
        except ValueError:
            pass

        else:
            rvs = tr.fix_range(rvs)

            figname = os.path.join(output_dir, dir_base + '-cmp-%d.png'
                                   % options.n_components)
            fig = pl.plot_rvs_comparison(fdata, rvs, sizes, bins,
                                         neg_shift=neg_shift)
            fig.savefig(figname)

        sparams = res.model.get_summary_params(res.params)[:, [1, 0, 2]]
        sparams[:, 0] = tr.transform_pi_deg(tr.fix_range(sparams[:, 0]),
                                            neg_shift=neg_shift)
        flags = [''] * 2
        if not (sparams[:, 1] > 0.0).all():
            flags[0] = '*'
        if not res.mle_retvals['converged']:
            flags[1] = '*'
        print 'flags:', flags

        fit_criteria = [-res.llf, res.aic, res.bic]

        log.write_row(dir_base, base_names, sparams, flags, fit_criteria)

        if options.show:
            plt.show()

if __name__ == '__main__':
    main()
