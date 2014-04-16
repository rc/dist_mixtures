import os

import numpy as np
import matplotlib.pyplot as plt

from analyses.transforms import transform_pi_deg, transform_2pi, fix_range

def plot_data(data, fdata, bins, neg_shift, plot_histogram=False):
    '''create figure with plot of raw data and histogram in subplots
    '''
    nrows = 2 if plot_histogram else 1
    fig, axs = plt.subplots(nrows=nrows, ncols=1, squeeze=False)

    td0 = transform_pi_deg(data[:, 0], neg_shift=neg_shift)
    ip = np.argsort(td0)
    xmin, xmax = (0, 180) if neg_shift else (-90, 90)

    ax = axs[0, 0]
    ax.plot(td0[ip], data[ip, 1], lw=2)
    ax.set_title('raw data')
    ax.set_xlim([xmin, xmax])
    ax.set_xlabel('angle', fontsize='large')
    ax.set_ylabel('coherency', fontsize='large')

    if plot_histogram:
        ax = axs[1, 0]
        ax.hist(transform_pi_deg(fdata, neg_shift=neg_shift),
             bins=bins, alpha=0.5)
        ax.set_title('raw data histogram')
        ax.set_xlim([xmin, xmax])
        ax.set_xlabel('angle')
        ax.set_ylabel('count')

    return fig

def plot_rvs_comparison(fdata, rvs, sizes, bins, neg_shift):
    '''plot 2 histograms given by fdata and rvs

    Parameters
    ----------
    fdata : ndarray
        original data
    rvs : ndarray
        simulated data
    sizes : list, iterable
        list of the numbers of observations of mixture components
    bins :
        directly used by matplotlib ``hist``
    negshift : bool
        If False, keep range in (-90, 90).
        If True, shift range to (0, 180).
    '''

    fig = plt.figure(3)
    plt.clf()
    plt.title('orig. (blue, %d) vs. sim. (green, %s)'
              % (fdata.shape[0], ', '.join('%d' % ii for ii in sizes)),
              fontsize='medium')
    plt.hist(transform_pi_deg(fdata, neg_shift=neg_shift),
             bins=bins, alpha=0.5)
    plt.hist(transform_pi_deg(rvs, neg_shift=neg_shift),
             bins=bins, alpha=0.5)

    xmin, xmax = (0, 180) if neg_shift else (-90, 90)
    plt.axis(xmin=xmin, xmax=xmax)

    plt.xlabel('angle', fontsize='large')
    plt.ylabel('count', fontsize='large')

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

def plot_raw_data(output_dir, source, area_angles=None):
    data, fdata, bins = source.get_source_data()

    figname = os.path.join(output_dir, source.current.dir_base + '-data.png')
    fig = plot_data(data, fdata, bins, neg_shift=source.neg_shift)

    if area_angles is not None:
        draw_areas(fig.axes[0], *area_angles)

    plt.tight_layout(pad=0.5)
    fig.savefig(figname)
    plt.close(fig)

def plot_estimated_dist(output_dir, result, source, pset_id=None):
    data, fdata, bins = source.get_source_data()

    xtr = lambda x: transform_pi_deg(x, neg_shift=source.neg_shift)
    rbins = transform_2pi(bins) - np.pi * (source.neg_shift == True)
    fig = result.model.plot_dist(result.full_params, xtransform=xtr, bins=rbins,
                                 data=fdata)
    fig.axes[0].set_title('estimated distribution')
    fig.axes[0].set_xlabel('angle', fontsize='large')
    fig.axes[0].set_ylabel('probability density function', fontsize='large')

    if pset_id is None:
        name = source.current.dir_base + '-fit.png'

    else:
        name = source.current.dir_base + '-fit-%d.png' % pset_id

    figname = os.path.join(output_dir, name)

    plt.tight_layout(pad=0.5)
    fig.savefig(figname)
    plt.close(fig)

def plot_histogram_comparison(output_dir, result, source, pset_id=None):
    data, fdata, bins = source.get_source_data()

    rvs, sizes = result.model.rvs_mix(result.full_params, size=fdata.shape[0],
                                      ret_sizes=True)
    rvs = fix_range(rvs)

    fig = plot_rvs_comparison(fdata, rvs, sizes, bins,
                              neg_shift=source.neg_shift)

    if pset_id is None:
        name = source.current.dir_base + '-cmp.png'

    else:
        name = source.current.dir_base + '-cmp-%d.png' % pset_id

    figname = os.path.join(output_dir, name)

    plt.tight_layout(pad=0.5)
    fig.savefig(figname)
    plt.close(fig)
