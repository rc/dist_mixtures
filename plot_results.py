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

def plot_fit_info(fig_num, logs, key):
    fig_size, de = get_fig_size(len(logs[0].items))
    fig = plt.figure(fig_num, fig_size)
    plt.clf()
    plt.subplots_adjust(bottom=0.12, left=2*de, right=1.0-de)
    ax = plt.gca()

    dys = []
    labels = []
    for log in logs:
        dy = log.get_value(key)
        dys.append(dy)
        labels.append('%d' % log.n_components)
    dys = np.asarray(dys)

    dx = np.arange(dys.shape[1])
    dymin = dys.min(axis=0)
    for dy in dys:
        plt.plot(dx, np.log10(dy - dymin + 1e-1), marker='o')

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
    ax.set_ylabel(r'$log_{10}(%s - min_{N_c}(%s) + 0.1)$' % (key, key))
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
    if dir_bases is None:
        dir_bases = used_dir_bases

    else:
        dir_bases = list(set(dir_bases).intersection(used_dir_bases))

    ix = np.array([ii for ii in range(sparams.shape[0])
                   if logs[0].items[ii].dir_base in dir_bases])

    fig_size, de = get_fig_size(ix.shape[0])
    fig = plt.figure(fig_num, fig_size)
    plt.clf()

    if sort_x:
        ii = sparams[ix, 0, -1].argsort()[::-1]
        ix = ix[ii]

    fig, axs = plt.subplots(3, 1, sharex=True, num=fig_num)

    dx = np.arange(ix.shape[0])

    colors = ['b', 'g', 'r', 'c', 'm']
    for ic in range(sparams.shape[1]):
        ms = 200 * np.sqrt(sparams[ix, ic, 2])
        axs[0].scatter(dx, sparams[ix, ic, 0], ms, c=colors[ic],
                       marker='o', alpha=0.8)
        axs[1].scatter(dx, sparams[ix, ic, 1], ms, c=colors[ic],
                       marker='o', alpha=0.8)
        axs[2].scatter(dx, sparams[ix, ic, 2], ms, c=colors[ic],
                       marker='o', alpha=0.8)

    axs[0].grid(axis='x')
    axs[1].grid(axis='x')
    axs[2].grid(axis='x')

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

suffix = '.pdf'

args = sys.argv[1:]

dirname = args[0]
logs = read_logs(dirname, 'log_*.csv')

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
    esuffix = suffix if group is None else '_%s' % group + suffix
    fig.savefig(op.join(dirname, 'params_%d' % nc + esuffix))

fig = plot_fit_info(1, logs, 'nllf')
fig.savefig(op.join(dirname, 'nllf' + suffix))
fig = plot_fit_info(2, logs, 'aic')
fig.savefig(op.join(dirname, 'aic' + suffix))
fig = plot_fit_info(3, logs, 'bic')
fig.savefig(op.join(dirname, 'bic' + suffix))
fig = plot_fit_info(4, logs, 'chisquare')
fig.savefig(op.join(dirname, 'chisquare' + suffix))
fig = plot_fit_info(5, logs, 'chisquare p-value')
fig.savefig(op.join(dirname, 'chisquare p-value' + suffix))

plt.show()
