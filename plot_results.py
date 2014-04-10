#!/usr/bin/env python
import sys
import os.path as op

import numpy as np
import matplotlib.pyplot as plt

from analyses.logs import read_logs
from groups import read_group_info, map_group_names, get_datasets_of_group

def get_fig_size(nx):
    fig_size = (np.clip(int(8 * nx / 20.), 8, 32), 6)

    de = 0.1 / (fig_size[0] / 8.0)
    return fig_size, de

def make_colors(num):
    """
    Make `num` continuously changing rainbow-like RGB colors.
    """
    def g(n):
        """
        Map sine [-1.0 .. 1.0] => color byte [0 .. 255].
        """
        return 255 * (n + 1) / 2.0

    def f(start, stop, num):

        interval = (stop - start) / num

        for n in range(num):
            coefficient = start + interval * n
            yield g(np.sin(coefficient * np.pi))

    red = f(0.5, 1.5, num)
    green = f(1.5, 3.5, num)
    blue = f(1.5, 2.5, num)

    rgbs = [('#%02x%02x%02x' % rgb) for rgb in zip(blue, green, red)]
    return rgbs

def get_colors(num):
    if num <= 5:
        colors = ['b', 'g', 'r', 'c', 'm']

    else:
        colors = make_colors(num)

    return colors

class Cycler(list):
    def __init__(self, sequence):
        list.__init__(self, sequence)
        self.n_max = len(sequence)

    def __getitem__(self, ii):
        return list.__getitem__(self, ii % self.n_max)

markers = Cycler(['o', 'v', 's','^', 'D', '<', 'p', '>', 'h'])

def plot_fit_info(fig_num, logs, key, transform, ylabel=None):
    fig_size, de = get_fig_size(len(logs[0].items))
    fig = plt.figure(fig_num, fig_size)
    plt.clf()
    plt.subplots_adjust(bottom=0.12, left=2*de, right=1.0-de)
    ax = plt.gca()

    if ylabel is None:
        ylabel = r'$log_{10}(%s - min_{N_c}(%s) + 0.1)$' % (key, key)

    dys = []
    labels = []
    for log in logs:
        dy = log.get_value(key)
        dys.append(dy)
        labels.append('%d' % log.n_components)
    dys = np.asarray(dys)

    dx = np.arange(dys.shape[1])
    dymin = dys.min(axis=0)

    num = len(dys)
    colors = get_colors(num)
    for ic, dy in enumerate(dys):
        plt.plot(dx, transform(dy, dymin), color=colors[ic],
                 marker=markers[ic], alpha=0.5)

    ylim = ax.get_ylim()
    yshift = 0.2 * (ylim[1] - ylim[0])

    for ii, _dx in enumerate(dx):
        plt.text(_dx, ylim[0] - 0.3 * yshift, '%.2e' % dymin[ii],
                 size=10, rotation=-50)

    ax.set_ylim([ylim[0] - yshift, ylim[1]])

    plt.xticks(rotation=70)
    ax.set_xticks(dx)
    ax.set_xticklabels([ii.dir_base for ii in logs[0].items])
    ax.set_xlim((-1, dx[-1] + 1))
    ax.set_ylabel(ylabel)
    ax.grid(axis='x')
    ax.legend(labels)

    return fig

def plot_params(fig_num, logs, n_components, gmap, dir_bases=None,
                cut_prob=0.1, sort_x=False, select_x=None):

    for log in logs:
        if log.n_components == n_components:
            break

    else:
        raise ValueError('no log with %d components!' % n_components)

    params = np.array(log.get_value('params'))

    aux = params.view('f8,f8,f8')
    sparams = np.sort(aux, order=['f2'], axis=1).view(np.float64)
    sparams = sparams[:, ::-1, :]

    used_dir_bases = [ii.dir_base for ii in logs[0].items]
    # Sort to have alphabetical order for equal probabilities.
    if dir_bases is None:
        dir_bases = sorted(used_dir_bases, reverse=sort_x)

    else:
        dir_bases = sorted(list(set(dir_bases).intersection(used_dir_bases)),
                           reverse=sort_x)

    ix = np.array([ii for ii in range(sparams.shape[0])
                   if logs[0].items[ii].dir_base in dir_bases])

    fig_size, de = get_fig_size(ix.shape[0])
    fig = plt.figure(fig_num, fig_size)
    plt.clf()

    if sort_x:
        ii = sparams[ix, 0, -1].argsort(kind='mergesort')[::-1]
        ix = ix[ii]
        dir_bases = [dir_bases[ic] for ic in ii]

    fig, axs = plt.subplots(3, 1, sharex=True, num=fig_num)

    dx = np.arange(ix.shape[0])

    num = sparams.shape[1]
    colors = get_colors(num)
    for ic in range(num):
        ms = 200 * np.sqrt(sparams[ix, ic, 2])
        axs[0].scatter(dx, sparams[ix, ic, 0], ms, c=colors[ic],
                       marker='o', alpha=0.5)
        axs[1].scatter(dx, sparams[ix, ic, 1], ms, c=colors[ic],
                       marker='o', alpha=0.5)
        axs[2].scatter(dx, sparams[ix, ic, 2], ms, c=colors[ic],
                       marker='o', alpha=0.5)

    axs[0].grid(axis='x')
    axs[1].grid(axis='x')
    axs[2].grid(axis='x')

    axs[0].set_ylabel(r'$\mu$')
    axs[1].set_ylabel(r'$\kappa$')
    axs[2].set_ylabel('prob.')

    axs[0].set_ylim((0, 180))

    for ii, dir_base in enumerate(dir_bases):
        axs[0].text(dx[ii] - 0.5, 185, '%s%d' % gmap[dir_base])

    axs[1].set_yscale('log')

    plt.xticks(rotation=70)
    axs[2].set_xticks(dx)
    axs[2].set_xticklabels(dir_bases)
    axs[2].set_xlim((-1, ix.shape[0]))
    axs[2].set_ylim((-0.05, 1.05))
    axs[2].hlines([cut_prob], -1, sparams.shape[0])

    plt.subplots_adjust(bottom=0.12, top=0.95, left=de, right=1.0-de)

    return fig

def save_fig(fig, filename, suffixes):
    if isinstance(suffixes, type('')):
        suffixes = [suffixes]

    for suffix in suffixes:
        fig.savefig(filename + suffix, dpi=300)

def tr_log10(dy, dymin):
    return np.log10(dy - dymin + 1e-1)

def tr_none(dy, dymin):
    return dy

suffix = ['.png', '.pdf']

args = sys.argv[1:]

dirname = args[0]
logs = read_logs(dirname, 'log_?.csv')

group_info = read_group_info()
gmap = map_group_names(group_info)

if len(args) == 2:
    group = args[1]
    try:
        group = int(args[1])

    except ValueError:
        pass

    dir_bases = get_datasets_of_group(group_info, group)

else:
    dir_bases = None
    group = None

plt.close('all')
for log in logs:
    nc = log.n_components
    fig = plot_params(10 + nc, logs, nc, gmap, dir_bases=dir_bases, sort_x=True)
    esuffix = '' if group is None else '_%s' % group
    save_fig(fig, op.join(dirname, 'params_%d' % nc + esuffix), suffix)

fig = plot_fit_info(1, logs, 'nllf', tr_log10)
save_fig(fig, op.join(dirname, 'nllf'), suffix)
fig = plot_fit_info(2, logs, 'aic', tr_log10)
save_fig(fig, op.join(dirname, 'aic'), suffix)
fig = plot_fit_info(3, logs, 'bic', tr_log10)
save_fig(fig, op.join(dirname, 'bic'), suffix)
fig = plot_fit_info(4, logs, 'chisquare', tr_log10)
save_fig(fig, op.join(dirname, 'chisquare'), suffix)
fig = plot_fit_info(5, logs, 'chisquare p-value', tr_none, 'chisquare p-value')
save_fig(fig, op.join(dirname, 'chisquare p-value'), suffix)

fig = plot_fit_info(6, logs, 'chisquare(e)', tr_log10)
save_fig(fig, op.join(dirname, 'chisquare(e)'), suffix)
fig = plot_fit_info(7, logs, 'chisquare(e) p-value', tr_none,
                    'chisquare(e) p-value')
save_fig(fig, op.join(dirname, 'chisquare(e) p-value'), suffix)
fig = plot_fit_info(8, logs, 'chisquare(e) power', tr_none,
                    'chisquare(e) power')
save_fig(fig, op.join(dirname, 'chisquare(e) power'), suffix)

plt.show()
