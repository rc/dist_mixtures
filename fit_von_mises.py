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
from logs import CSVLog

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
    data = np.asarray(data)

    out = 2 * data * np.pi / 180.0
    return out

def transform_pi_deg(data, neg_shift=False):
    data = np.asarray(data)

    out = 90.0 * data / np.pi
    if neg_shift:
        out = np.where(out >= 0.0, out, out + 180.0)
    return out

def fix_range(data):
    data = np.asarray(data)

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
    lengths = np.asarray(lengths)

    lo = lengths.min()
    counts = ((10.0 / lo) * lengths).astype(np.int32)

    return counts

def fit(data, start_params):
    mod = VonMisesMixture(data)
    #with master of statsmodels we can use bfgs
    #because the objective function is now normalized with 1/nobs we can
    #increase the gradient tolerance gtol
    #res = mod.fit(start_params=start_params, method='nm', disp=True)
    #res = mod.fit(start_params=res.params, method='bfgs', disp=True) #False)
    res = mod.fit(start_params=start_params, method='bfgs', disp=True,
                  gtol=1e-9) #False)

    return res

def get_area_angles(data, neg_shift=False):
    aux = transform_pi_deg(data[:, 0], neg_shift=neg_shift)
    ip = np.argsort(aux)
    aux = aux[ip]
    ddd = np.c_[aux[:, None], data[ip, 1:]]

    # Mirror the first data point.
    dx = aux[1] - aux[0]
    ddd = np.r_[ddd, [[ddd[-1, 0] + dx, ddd[0, 1]]]]

    xmin, xmax = -1000, 1000
    arh, xm = split_equal_areas(ddd, xmin, xmax)
    arh1, x0 = split_equal_areas(ddd, xmin, xm)
    arh2, x1 = split_equal_areas(ddd, xm, xmax)

    print x0, xm, x1
    print arh, arh1, arh2, arh1 - arh2, arh - (arh1 + arh2)

    return x0, xm, x1, arh1, arh2

def split_equal_areas(data, x0, x1):
    """
    Split histogram-like `data` into two parts with equal areas between `x0`
    and `x1`.
    """
    x, y = data[:, 0].copy(), data[:, 1]
    n_data = data.shape[0]

    dx = x[1] - x[0]

    xs = x - 0.5 * dx

    i0 = np.searchsorted(xs, x0)
    if i0 == 0:
        sub0 = 0.0
        i0 = 1

    else:
        sub0 = (x0 - xs[i0 - 1]) * y[i0 - 1]

    i1 = np.searchsorted(xs, x1)
    if i1 == n_data:
        sub1 = 0.0

    else:
        sub1 = (xs[i1] - x1) * y[i1 - 1]

    yy = y[i0 - 1:i1] * dx
    area = np.sum(yy) - sub0 - sub1

    ca = np.cumsum(yy) - sub0
    ih = np.searchsorted(ca, 0.5 * area)

    da = ca[ih] - 0.5 * area
    dxh = da / y[i0 - 1 + ih]

    xh = xs[i0 + ih] - dxh

    return 0.5 * area, xh

def plot_data(data, fdata, bins, neg_shift):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

    td0 = transform_pi_deg(data[:, 0], neg_shift=neg_shift)
    ip = np.argsort(td0)
    xmin, xmax = (0, 180) if neg_shift else (-90, 90)

    ax1.plot(td0[ip], data[ip, 1])
    ax1.set_title('raw data')
    ax1.set_xlim([xmin, xmax])

    ax2.hist(transform_pi_deg(fdata, neg_shift=neg_shift),
             bins=bins, alpha=0.5)
    ax2.set_title('raw data histogram (counts)')
    ax2.set_xlim([xmin, xmax])

    return fig

def plot_rvs_comparison(fdata, rvs, sizes, bins, neg_shift):
    fig = plt.figure(3)
    plt.clf()
    plt.title('original (blue, %d) vs. simulated (green, %s)'
              % (fdata.shape[0], ', '.join('%d' % ii for ii in sizes)))
    plt.hist(transform_pi_deg(fdata, neg_shift=neg_shift),
             bins=bins, alpha=0.5)
    plt.hist(transform_pi_deg(rvs, neg_shift=neg_shift),
             bins=bins, alpha=0.5)

    xmin, xmax = (0, 180) if neg_shift else (-90, 90)
    plt.axis(xmin=xmin, xmax=xmax)

    return fig

def draw_areas(ax, x0, xm, x1, arh1, arh2):
    from matplotlib.patches import Rectangle

    w = xm - x0
    h0 = arh1 / w
    rect = Rectangle((x0, 0), w, h0, color='gray', alpha=0.3)
    ax.add_patch(rect)
    xh = 0.5 * (x0 + xm)
    ax.vlines(xh, 0, h0)
    ax.text(xh, 0.25 * h0, '%+.2f' % (xh - xm))

    w = x1 - xm
    h1 = arh2 / w
    rect = Rectangle((xm, 0), w, h1, color='gray', alpha=0.3)
    ax.add_patch(rect)
    xh = 0.5 * (xm + x1)
    ax.vlines(xh, 0, h1)
    ax.text(xh, 0.75 * h1, '%+.2f' % (xh - xm))

    ax.text(xm, 0.25 * (h0 + h1), '%.2f' % xm)

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
        start_params[1:n2:2] = transform_2pi(aux[0::2]) # mu.

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
        bins = all_bins[::4]

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
        fig = res.model.plot_dist(res.params, xtransform=xtr)
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

if __name__ == '__main__':
    main()
