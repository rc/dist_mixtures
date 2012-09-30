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
import ioutils as io

usage = '%prog [options] pattern data_dir output_dir\n' + __doc__.rstrip()

def load_data(filenames, transform=None):
    if transform is None:
        transform = lambda x: x

    datas = []
    for filename in filenames:
        data = np.genfromtxt(filename)
        data[:, 0] = transform(data[:, 0])
        datas.append(data)

    datas = np.array(datas)

    merged_data = datas[0]
    for dd in datas[1:]:
        if not (dd[:, 0] == merged_data[:, 0]).all():
            raise ValueError('x axes do not match in %s' % filenames)
        merged_data[:, 1] += dd[:, 1]

    return merged_data

def transform_2pi(data):
    out = 2 * data * np.pi / 180.0
    return out

def transform_pi_deg(data):
    out = 90.0 * data / np.pi
    return out

def fix_range(data):
    data = data.copy()
    while 1:
        ii = np.where(data < -np.pi)[0]
        data[ii] += 2 * np.pi
        if not len(ii): break
    while 1:
        ii = np.where(data >  np.pi)[0]
        data[ii] -= 2 * np.pi
        if not len(ii): break

    return data

def get_counts_from_lengths(lengths):
    """
    Get simulated counts corresponding to lengths.
    """
    lo = lengths.min()
    counts = ((10.0 / lo) * lengths).astype(np.int32)

    return counts

def fit(data, start_params):
    mod = VonMisesMixture(data)
    res = mod.fit(start_params=start_params, method='nm', disp=False)
    res = mod.fit(start_params=start_params, disp=False)
    ## #refine optimization
    ## res = mod.fit(start_params=res.params, method='ncg')

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
        start_params[1:n2:2] = transform_2pi(aux[0::2]) # mu.

    print 'starting parameters:', start_params

    io.ensure_path(output_dir)

    log_name = os.path.join(output_dir, 'log.txt')
    fd = open(log_name, 'w')
    fd.write('# case: no. of files: filenames\n')
    fd.write('# flags ' + '; '.join(['mu%d kappa%d prob%d' % (ii, ii, ii)
                               for ii in range(options.n_components)]) + '\n')
    fd.write('# -----\n')
    fd.write('# flags:\n')
    fd.write('# ! ... line with a negative kappa\n')
    fd.write('# @ ... fit did not converge\n')
    fd.write('# mu (location) is in degrees in [-90, 90[\n')
    fd.close()

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

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        ax1.plot(data[:, 0], data[:, 1])
        ax1.set_title('raw data, transformed angles')

        # Simulate the "random process" the histogram was done from.
        counts = get_counts_from_lengths(data[:, 1])
        fdata = np.repeat(data[:, 0], counts)

        print 'simulated counts range:', counts.min(), counts.max()

        nbins = 45
        ax2.hist(fdata, bins=nbins-1, range=data[[0, -1], 0], alpha=0.5)
        ax2.set_title('raw data histogram')

        figname = os.path.join(output_dir, dir_base + '-data.png')
        fig.savefig(figname)

        res = fit(fdata, start_params)

        res.model.summary_params(res.params,
                                 name='%d components' % options.n_components)

        fig = res.model.plot_dist(res.params)
        fig.axes[0].set_title('Estimated distribution')

        figname = os.path.join(output_dir, dir_base + '-fit-%d.png'
                               % options.n_components)
        fig.savefig(figname)

        fd = open(log_name, 'a')
        fd.write('%s: %d: %s\n' % (dir_base, len(base_names),
                                 ', '.join(base_names)))
        sparams = res.model.get_summary_params(res.params)
        sparams[:, 1] = transform_pi_deg(fix_range(sparams[:, 1]))
        aux = '; '.join(['% 6.2f % 5.4f %6.4f'
                         % (pp[1], pp[0], pp[2]) for pp in sparams])
        flags = [' '] * 3
        if not (sparams[:, 0] > 0.0).all():
            flags[0] = '!'
        if not res.mle_retvals['converged']:
            flags[1] = '@'
        print 'flags:', flags
        fd.write(''.join(flags) + aux + '\n')
        fd.close()

        if options.show:
            plt.show()

if __name__ == '__main__':
    main()
