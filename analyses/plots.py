import numpy as np
import matplotlib.pyplot as plt

from analyses.transforms import transform_pi_deg

def plot_data(data, fdata, bins, neg_shift):
    '''create figure with plot of raw data and histogram in subplots


    '''
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
