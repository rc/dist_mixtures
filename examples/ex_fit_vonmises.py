# -*- coding: utf-8 -*-
"""interactive script converted from main function of von_mises_fit

Created on Wed Oct 24 19:22:17 2012
Author: Josef Perktold
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import analyses.transforms as tr
import analyses.ioutils as io
import analyses.plots as pl
from analyses.logs import CSVLog
from analyses.fit_mixture import fit

import dist_mixtures.mixture_von_mises as mixvn
#from dist_mixtures.mixture_von_mises import VonMisesMixtureBinned

class Store(object):
    pass

if __name__ == '__main__':

    options = Store()
    #options
    #default
    options.n_components = 2     # 2 default
    options.params = None        # None default
    options.dir_pattern = '*'    # '*' default
    options.area_angles = False  # False default
    options.show = True #False         # False default

    pattern = '*ImageJ.txt'
    data_dir = '../analysis/test-data/192_09'
    #output_dir = '../analysis/tmp_bfgs_rsprob_sp_2_master_allbins_hist_'
    output_dir = '../analysis/tmp_192'

    #end options
    #options.show = True
    #options.params = '0, 5'



    start_params = np.zeros(options.n_components * 3 - 1)
    n2 = 2 * options.n_components
    if options.params is None:
        # Zeros for mu, twos for kappa.
        start_params[:n2:2] = 2.0 + np.random.uniform(-0.1, 0.1, options.n_components)
        start_params[:n2:2] = 5.0 + np.random.uniform(-0.5, 0.5, options.n_components)

    else:
        aux = np.array([float(ii) for ii in options.params.split(',')])
        start_params[:n2:2] = aux[1::2] # kappa.
        start_params[1:n2:2] = tr.transform_2pi(aux[0::2]) # mu.

    start_params[n2:] = np.random.uniform(-0.1, 0.1, options.n_components - 1)

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
        fdata = tr.spread_by_counts(data[:, 0], counts)

        print 'simulated counts range:', counts.min(), counts.max(), counts.sum()

        ddata = np.sort(tr.transform_pi_deg(data[:, 0], neg_shift=neg_shift))
        dd = ddata[1] - ddata[0]
        all_bins = np.r_[ddata - 1e-8, ddata[-1] + dd]
        bins = all_bins #[::4]

        figname = os.path.join(output_dir, dir_base + '-data.png')
        fig = pl.plot_data(data, fdata, bins, neg_shift=neg_shift)

        if options.area_angles:
            pl.draw_areas(fig.axes[0],
                          *pl.get_area_angles(data, neg_shift=neg_shift))

        fig.savefig(figname)

        aux = Store()
        aux.fdata = fdata
        res = fit(aux, start_params, mixvn.VonMisesMixture, ('bfgs', {}))
        #normalize parameters, rotate, scipy cdf breaks on negative kappa
        import ex_cdf as e
        res.params = e.normalize_params(res.params)

        res.model.summary_params(res.params,
                                 name='%d components' % options.n_components)

        xtr = lambda x: tr.transform_pi_deg(x, neg_shift=neg_shift)
        fig = res.model.plot_dist(res.params, xtransform=xtr, n_bins=180)
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
        converged = res.mle_retvals['converged']
        print 'converged:', converged

        fit_criteria = [-res.llf, res.aic, res.bic]

        #goodness-of-fit chisquare test

        #TODO: we know bins of fdata
        uni_fdata = data[:, 0]
        xx_cdf = uni_fdata + (2 * np.pi / 180) / 2
        #TODO: bug: cdf_mix doesn't work if n_components=1
        tr = lambda x: np.remainder(x + np.pi, 2*np.pi) - np.pi
        if options.n_components == 1:
            #cdf = stats.vonmises._cdf(tr(xx_cdf - res.params[1] - np.pi * (np.sign(res.params[0])==-1)), np.abs(res.params[0]))
            #looks ok, cdf > 1 or cdf < 0 to make it easier to wrap around circle
            #subtract cdf of starting point
            from scipy import stats
            cdf = stats.vonmises.cdf((xx_cdf), res.params[0], loc=res.params[1])
        else:
            cdf = res.model.cdf_mix(res.params, xx_cdf)
        #TODO: problems, cdf > 1, bin boundaries merge 1st and last ?
        pdf_bins = np.diff(np.concatenate((cdf, [1+cdf[0]])))
        from scipy import stats
        print 'chisquare test',
        fac = 1.   # try when we are unsure about sample size
        chisquare = stats.chisquare(counts*fac, counts.sum() * fac * pdf_bins)
        print chisquare

        log.write_row(dir_base, base_names, chisquare, sparams, converged,
                      fit_criteria)

        #comparison with raw length distribution
        #plot not shifted to center
        data_raw = io.load_data(filenames, transform=None)
        rad_diff = data[1,0] - data[0,0]
        plt.figure()
        plt.plot(data[:,0], data_raw[:,1] / data_raw[:,1].sum(),
                 color='b', lw=2, alpha=0.7, label='data')
        plt.plot(data[:,0], res.model.pdf_mix(res.params, data[:,0]) * rad_diff,
                 color='r', lw=2, alpha=0.7, label='estimated')
        plt.title('Length distribution - data and estimate')
        plt.legend()

        count_endog = data_raw[:, 1] / 100.
        bins_exog = np.linspace(-np.pi, np.pi, 180+1)
        modb = mixvn.VonMisesMixtureBinned(count_endog, bins_exog)
        resb = modb.fit(start_params=res.params, method='bfgs')
        resb.params = mixvn.normalize_params(resb.params)
        resb2 = modb.fit(start_params=start_params, method='bfgs')
        resb2.params = mixvn.normalize_params(resb2.params)
        print 'res.params  ', res.params
        print 'resb.params ', resb.params
        print 'resb2.params', resb2.params
        #TODO: need to standardize sequence of components in params
        print 'gof chisquare', resb.model.gof_chisquare(resb.params)

        #LS is more sensitive to start_params ?
        resbls = modb.fit_ls(start_params=res.params)
        resbls_params = mixvn.normalize_params(resbls[0])
        print 'resbls params ', resbls_params

        plt.figure()
        plt.plot(data[:,0], data_raw[:,1] / data_raw[:,1].sum(),
                 color='b', lw=2, alpha=0.7, label='data')
        plt.plot(data[:,0], resb.model.pdf_mix(resb.params, data[:,0]) * rad_diff,
                 color='r', lw=2, alpha=0.7, label='estimated')
        plt.title('Length distribution - data and binned MLE estimate')
        plt.legend()

        plt.figure()
        plt.plot(data[:,0], data_raw[:,1] / data_raw[:,1].sum(),
                 color='b', lw=2, alpha=0.7, label='data')
        plt.plot(data[:,0], resb.model.pdf_mix(resbls_params, data[:,0]) * rad_diff,
                 color='r', lw=2, alpha=0.7, label='estimated')
        plt.title('Length distribution - data and LS estimate')
        plt.legend()


        if options.show:
            plt.show()
