#!/usr/bin/env python
import sys
import os.path as op
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from analyses.logs import CSVLog

plt.close('all')

dirname = sys.argv[1]
filenames = sorted(glob(op.join(dirname, 'log_*.csv')))

datas = []
for filename in filenames:
    print filename
    data = CSVLog.from_file(filename)
    datas.append(data)

def get_fig_size(nx):
    fig_size = (np.clip(int(8 * nx / 20.), 8, 32), 6)

    de = 0.1 / (fig_size[0] / 8.0)
    return fig_size, de

def plot_fit_info(fig_num, datas, key):
    fig_size, de = get_fig_size(len(datas[0].items))
    fig = plt.figure(fig_num, fig_size)
    plt.clf()
    plt.subplots_adjust(bottom=0.12, left=2*de, right=1.0-de)
    ax = plt.gca()

    dys = []
    labels = []
    for data in datas:
        dy = data.get_value(key)
        dys.append(dy)
        labels.append('%d' % data.n_components)
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
    ax.set_xticklabels([ii.dir_base for ii in datas[0].items])
    ax.set_xlim((-1, dx[-1] + 1))
    ax.set_ylabel(r'$log_{10}(%s - min_{N_c}(%s) + 0.1)$' % (key, key))
    ax.grid(axis='x')
    ax.legend(labels)

    return fig

def plot_params(fig_num, datas, n_components, cut_prob=0.1):

    for data in datas:
        if data.n_components == n_components:
            break

    else:
        raise ValueError('no log with %d components!' % n_components)

    params = np.array(data.get_value('params'))

    fig_size, de = get_fig_size(params.shape[0])
    fig = plt.figure(fig_num, fig_size)
    plt.clf()

    aux = params.view('f8,f8,f8')
    sparams = np.sort(aux, order=['f2'], axis=1).view(np.float64)
    sparams = sparams[:, ::-1, :]

    fig, axs = plt.subplots(3, 1, sharex=True, num=fig_num)

    dx = np.arange(sparams.shape[0])

    colors = ['b', 'g', 'r', 'c', 'm']
    for ic in range(sparams.shape[1]):
        ms = 200 * np.sqrt(sparams[:, ic, 2])
        axs[0].scatter(dx, sparams[:, ic, 0], ms, c=colors[ic],
                       marker='o', alpha=0.8)
        axs[1].scatter(dx, sparams[:, ic, 1], ms, c=colors[ic],
                       marker='o', alpha=0.8)
        axs[2].scatter(dx, sparams[:, ic, 2], ms, c=colors[ic],
                       marker='o', alpha=0.8)

    axs[0].grid(axis='x')
    axs[1].grid(axis='x')
    axs[2].grid(axis='x')

    axs[0].set_ylim((0, 180))

    axs[1].set_yscale('log')

    plt.xticks(rotation=70)
    axs[2].set_xticks(dx)
    axs[2].set_xticklabels([ii.dir_base for ii in datas[0].items])
    axs[2].set_xlim((-1, sparams.shape[0]))
    axs[2].set_ylim((-0.05, 1.05))
    axs[2].hlines([cut_prob], -1, sparams.shape[0])

    plt.subplots_adjust(bottom=0.12, left=de, right=1.0-de)

    return fig

suffix = '.pdf'

for data in datas:
    nc = data.n_components
    fig = plot_params(10 + nc, datas, nc)
    fig.savefig(op.join(dirname, 'params_%d' % nc + suffix))

fig = plot_fit_info(1, datas, 'nllf')
fig.savefig(op.join(dirname, 'nllf' + suffix))
fig = plot_fit_info(2, datas, 'aic')
fig.savefig(op.join(dirname, 'aic' + suffix))
fig = plot_fit_info(3, datas, 'bic')
fig.savefig(op.join(dirname, 'bic' + suffix))
fig = plot_fit_info(4, datas, 'chisquare')
fig.savefig(op.join(dirname, 'chisquare' + suffix))
fig = plot_fit_info(5, datas, 'chisquare p-value')
fig.savefig(op.join(dirname, 'chisquare p-value' + suffix))

plt.show()
