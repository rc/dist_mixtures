# -*- coding: utf-8 -*-
"""interactive script converted from main function of von_mises_fit

Created on Wed Oct 24 19:22:17 2012
Author: Josef Perktold
"""

import numpy as np
import matplotlib.pyplot as plt

from fit_von_mises import *

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
    options.show = False         # False default
    
    pattern = '*ImageJ.txt'
    data_dir = '../analysis/test-data'
    output_dir = '../analysis/tmp_bfgs_rsprob_sp_2_master_allbins_hist'
    
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
        start_params[1:n2:2] = transform_2pi(aux[0::2]) # mu.
    
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

        data = load_data(filenames, transform=transform_2pi)

        print 'data range:', data[:, 1].min(), data[:, 1].max()

        # Simulate the "random process" the histogram was done from.
        counts = get_counts_from_lengths(data[:, 1])
        fdata = np.repeat(data[:, 0], counts)

        print 'simulated counts range:', counts.min(), counts.max()

        ddata = np.sort(transform_pi_deg(data[:, 0], neg_shift=neg_shift))
        dd = ddata[1] - ddata[0]
        all_bins = np.r_[ddata - 1e-8, ddata[-1] + dd]
        bins = all_bins #[::4]

        figname = os.path.join(output_dir, dir_base + '-data.png')
        fig = plot_data(data, fdata, bins, neg_shift=neg_shift)

        if options.area_angles:
            draw_areas(fig.axes[0],
                       *get_area_angles(data, neg_shift=neg_shift))

        fig.savefig(figname)

        res = fit(fdata, start_params)

        res.model.summary_params(res.params,
                                 name='%d components' % options.n_components)

        xtr = lambda x: transform_pi_deg(x, neg_shift=neg_shift)
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
            rvs = fix_range(rvs)

            figname = os.path.join(output_dir, dir_base + '-cmp-%d.png'
                                   % options.n_components)
            fig = plot_rvs_comparison(fdata, rvs, sizes, bins,
                                      neg_shift=neg_shift)
            fig.savefig(figname)

        sparams = res.model.get_summary_params(res.params)[:, [1, 0, 2]]
        sparams[:, 0] = transform_pi_deg(fix_range(sparams[:, 0]),
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
