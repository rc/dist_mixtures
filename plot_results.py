#!/usr/bin/env python
"""
Plot overall/group results from logs with mixtures of von Mises distributions.
"""
import os.path as op
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt

from analyses.logs import read_logs
from groups import read_group_info, map_group_names, get_datasets_of_group
from fit_von_mises import parse_rc

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

def format_group(val):
    return ('%s%s%s' % (val[0], val[2][0], val[3][0])).upper()

def plot_fit_info(fig_num, logs, key, transform, ylabel=None):
    fig_size, de = get_fig_size(len(logs[0].items))
    fig = plt.figure(fig_num, fig_size)
    plt.clf()
    plt.subplots_adjust(bottom=0.12, left=2*de, right=1.0-de)
    ax = plt.gca()

    if ylabel is None:
        ylabel = r'$log_{10}(%s - min_{i}(%s) + 0.1)$' % (key, key)

    dys = []
    labels = []
    for ilog, log in enumerate(logs):
        dy = log.get_value(key)
        dys.append(dy)
        labels.append('%d' % ilog)
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

    if transform != tr_none:
        for ii, _dx in enumerate(dx):
            plt.text(_dx, ylim[0] - 0.3 * yshift, '%.2e' % dymin[ii],
                     size='small', rotation=-50)

    ax.set_ylim([ylim[0] - yshift, ylim[1]])

    plt.xticks(rotation=70)
    ax.set_xticks(dx)
    ax.set_xticklabels([ii.dir_base for ii in logs[0].items])
    ax.set_xlim((-1, dx[-1] + 1))
    ax.set_ylabel(ylabel, fontsize='large')
    ax.grid(axis='x')
    ax.legend(labels)

    return fig

def plot_params(fig_num, log, gmap, dir_bases=None,
                cut_prob=0.1, sort_x=False, select_x=None, equal_probs=False):
    params = np.array(log.get_value('params'))

    if dir_bases is None:
        dir_bases = [ii.dir_base for ii in log.items]

    # Select parameters corresponding to dir_bases.
    ix = np.array([ii for ii in range(params.shape[0])
                   if log.items[ii].dir_base in dir_bases])
    params = params[ix]

    # Sort components by probabilities (descending).
    aux = params.view('f8,f8,f8')
    sparams = np.sort(aux, order=['f2'], axis=1).view(np.float64)
    sparams = sparams[:, ::-1, :]

    # Sort to have alphabetical order for equal probabilities.
    ii = np.argsort(dir_bases, kind='mergesort')[::-1]
    dir_bases = np.array(dir_bases)[ii]
    sparams = sparams[ii]

    fig_size, de = get_fig_size(ix.shape[0])
    fig = plt.figure(fig_num, fig_size)
    plt.clf()

    if sort_x:
        # Sort according to the highest component probability (descending).
        ii = sparams[:, 0, -1].argsort(kind='mergesort')[::-1]
        sparams = sparams[ii]
        dir_bases = dir_bases[ii]

    n_ax = 2 + ((not equal_probs)
                or (sparams[:, :, 2] != sparams[0, 0, 2]).any())
    fig, axs = plt.subplots(n_ax, 1, sharex=True, num=fig_num)

    dx = np.arange(ix.shape[0])

    num = sparams.shape[1]
    colors = get_colors(num)
    for ic in range(num):
        ms = 200 * np.sqrt(sparams[:, ic, 2])
        axs[0].scatter(dx, sparams[:, ic, 0], ms, c=colors[ic],
                       marker='o', alpha=0.5)
        axs[1].scatter(dx, sparams[:, ic, 1], ms, c=colors[ic],
                       marker='o', alpha=0.5)
        if n_ax == 3:
            axs[2].scatter(dx, sparams[:, ic, 2], ms, c=colors[ic],
                           marker='o', alpha=0.5)

    axs[0].grid(axis='x')
    axs[1].grid(axis='x')

    axs[0].set_ylabel(r'$\mu$', fontsize='large')
    axs[1].set_ylabel(r'$\kappa$', fontsize='large')

    if n_ax == 3:
        axs[2].grid(axis='x')
        axs[2].set_ylabel('prob.', fontsize='large')

    axs[0].set_ylim((0, 180))
    axs[0].set_yticks(np.linspace(0, 180, 7))

    for ii, dir_base in enumerate(dir_bases):
        axs[0].text(dx[ii] - 0.5, 185, '%s' % format_group(gmap[dir_base]),
                    fontsize='small', family='monospace')

    axs[1].set_yscale('log')

    plt.xticks(rotation=70)
    axs[-1].set_xticks(dx)
    axs[-1].set_xticklabels(dir_bases)
    axs[-1].set_xlim((-1, ix.shape[0]))
    if n_ax == 3:
        axs[-1].set_ylim((-0.05, 1.05))
        axs[-1].hlines([cut_prob], -1, sparams.shape[0])

    plt.subplots_adjust(bottom=0.12, top=0.95, left=de, right=1.0-de)

    return fig

def save_fig(fig, filename, suffixes):
    if isinstance(suffixes, type('')):
        suffixes = [suffixes]

    plt.tight_layout()
    for suffix in suffixes:
        fig.savefig(filename + suffix, dpi=300)

def tr_log10(dy, dymin):
    return np.log10(dy - dymin + 1e-1)

def tr_none(dy, dymin):
    return dy

def try_int(arg):
    try:
        return int(arg)

    except:
        return arg

usage = '%prog [options] log_pattern\n' + __doc__.rstrip()

helps = {
    'group' : 'group name and group value',
    'plot_equal_probs' : 'plot probabilities even if all are equal to'
    ' a single value',
    'suffixes' : 'figure suffixes for saving [default: %default]',
    'rc' : 'matplotlib resources',
    'show' :
    'show the figures',
}

def get_options_parser():
    parser = OptionParser(usage=usage, version='%prog')
    parser.add_option('-g', '--group', metavar='group_name,group_value',
                      action='store', dest='group',
                      default=None, help=helps['group'])
    parser.add_option('-e', '--equal-probs',
                      action='store_true', dest='plot_equal_probs',
                      default=False, help=helps['plot_equal_probs'])
    parser.add_option('', '--suffixes', metavar='suffix1,...',
                      action='store', dest='suffixes',
                      default='png,pdf', help=helps['suffixes'])
    parser.add_option('--rc', type='str', metavar='key=val,...',
                      action='callback', dest='rc',
                      callback=parse_rc, default={}, help=helps['rc'])
    parser.add_option('-s', '--show',
                      action='store_true', dest='show',
                      default=False, help=helps['show'])

    return parser

def main():
    parser = get_options_parser()
    options, args = parser.parse_args()

    if len(args) != 1:
        parser.print_help()
        return

    dirname, pattern = op.split(args[0])
    logs = read_logs(dirname, pattern)

    group_info = read_group_info()
    gmap = map_group_names(group_info)

    if options.group is not None:
        aux = options.group.split(',')
        group_name = try_int(aux[0])
        val = try_int(aux[1])

        dir_bases = get_datasets_of_group(group_info, group_name, val)

    else:
        dir_bases = None
        group_name = None

    equal_probs = not options.plot_equal_probs
    suffixes = ['.' + ii for ii in options.suffixes.split(',')]

    plt.rcParams.update(options.rc)

    plt.close('all')
    for ilog, log in enumerate(logs):
        fig = plot_params(20 + ilog, log, gmap, dir_bases=dir_bases,
                          sort_x=True, equal_probs=equal_probs)
        esuffix = '' if group_name is None else '_%s_%s' % (group_name, val)
        save_fig(fig, op.join(dirname, 'params_%d' % ilog + esuffix), suffixes)

    fig = plot_fit_info(1, logs, 'nllf', tr_log10)
    save_fig(fig, op.join(dirname, 'nllf'), suffixes)
    fig = plot_fit_info(2, logs, 'aic', tr_log10)
    save_fig(fig, op.join(dirname, 'aic'), suffixes)
    fig = plot_fit_info(3, logs, 'bic', tr_log10)
    save_fig(fig, op.join(dirname, 'bic'), suffixes)
    fig = plot_fit_info(4, logs, 'chisquare', tr_log10)
    save_fig(fig, op.join(dirname, 'chisquare'), suffixes)
    fig = plot_fit_info(5, logs, 'chisquare p-value', tr_none,
                        'chisquare p-value')
    save_fig(fig, op.join(dirname, 'chisquare p-value'), suffixes)

    fig = plot_fit_info(6, logs, 'chisquare(e)', tr_log10)
    save_fig(fig, op.join(dirname, 'chisquare(e)'), suffixes)
    fig = plot_fit_info(7, logs, 'chisquare(e) p-value', tr_none,
                        'chisquare(e) p-value')
    save_fig(fig, op.join(dirname, 'chisquare(e) p-value'), suffixes)
    fig = plot_fit_info(8, logs, 'chisquare(e) power', tr_none,
                        'chisquare(e) power')
    save_fig(fig, op.join(dirname, 'chisquare(e) power'), suffixes)

    fig = plot_fit_info(11, logs, 'nllf', tr_none, '-llf')
    save_fig(fig, op.join(dirname, 'nllf-raw'), suffixes)
    fig = plot_fit_info(12, logs, 'aic', tr_none, 'aic')
    save_fig(fig, op.join(dirname, 'aic-raw'), suffixes)
    fig = plot_fit_info(13, logs, 'bic', tr_none, 'bic')
    save_fig(fig, op.join(dirname, 'bic-raw'), suffixes)
    fig = plot_fit_info(14, logs, 'chisquare', tr_none, 'chisquare')
    save_fig(fig, op.join(dirname, 'chisquare-raw'), suffixes)
    fig = plot_fit_info(15, logs, 'chisquare(e)', tr_none, 'chisquare(e)')
    save_fig(fig, op.join(dirname, 'chisquare(e)-raw'), suffixes)

    if options.show:
        plt.show()

if __name__ == '__main__':
    main()
