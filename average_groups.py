#!/usr/bin/env python
import sys
import os.path as op
import csv

import numpy as np
import matplotlib.pyplot as plt

from analyses.logs import read_logs
from groups import read_group_info, map_group_names, get_datasets_of_group

def save_fig(fig, filename, suffixes):
    if isinstance(suffixes, type('')):
        suffixes = [suffixes]

    for suffix in suffixes:
        fig.savefig(filename + suffix, dpi=300)

args = sys.argv[1:]

same_shapes = len(args) == 2

if same_shapes:
    loc_range = [50, 150]

else:
    loc_range = [0, 180]

dirname, logname = op.split(args[0])
log = read_logs(dirname, logname)[0]

prefix = op.splitext(logname)[0]

all_params = np.array(log.get_value('params'))
all_dir_bases = np.array([ii.dir_base for ii in log.items])

all_chi_p = np.array(log.get_value('chisquare(e) p-value'))
all_aic = np.array(log.get_value('aic'))
all_bic = np.array(log.get_value('bic'))

group_info = read_group_info()
gmap = map_group_names(group_info)

group_names = ['age_group', 'pig_group', 'segment']

suffix = ['.png', '.pdf']

rows = []
for group_name in group_names:
    print group_name
    for val in np.unique(group_info[group_name]):
        dir_bases = get_datasets_of_group(group_info, group_name, val)
        ix = np.array([ii for ii in range(all_params.shape[0])
                       if log.items[ii].dir_base in dir_bases])
        print '  ', val, len(dir_bases), len(ix)

        if not len(ix): continue

        params = all_params[ix]
        dir_bases = all_dir_bases[ix]

        chi_p = all_chi_p[ix]
        aic = all_aic[ix]
        bic = all_bic[ix]

        locs = params[:, :2, 0]
        shapes = params[:, :2, 1]
        probs = params[:, :2, 2]

        if same_shapes:
            assert((shapes[:, 0] == shapes[:, 1]).all())
            assert((probs[:, 0] == probs[:, 1]).all())

        # Sort by locations.
        ic = locs.argsort(axis=1)
        ir = np.arange(len(ix))[:, None]
        locs = locs[ir, ic]

        if not same_shapes:
            shapes = shapes[ir, ic]
            probs = probs[ir, ic]

        for ir, dir_base in enumerate(dir_bases):
            rows.append([group_name, val, dir_base,
                         chi_p[ir], aic[ir], bic[ir]] + locs[ir].tolist()
                        + [shapes[ir, 0], probs[ir, 0]])

        fig = plt.figure(1)
        ax0 = plt.subplot2grid((1, 4 + 2 * (not same_shapes)),
                               (0, 0), colspan=2)
        ax0.boxplot([locs[:, 0], locs[:, 1]])

        if same_shapes:
            ax1 = plt.subplot2grid((1, 4), (0, 2), colspan=1)
            ax2 = plt.subplot2grid((1, 4), (0, 3), colspan=1)

            ax1.boxplot([shapes[:, 0]])
            ax2.boxplot([probs[:, 0]])

        else:
            ax1 = plt.subplot2grid((1, 6), (0, 2), colspan=2)
            ax2 = plt.subplot2grid((1, 6), (0, 4), colspan=2)

            ax1.boxplot([shapes[:, 0], shapes[:, 1]])
            ax2.boxplot([probs[:, 0], probs[:, 1]])

        ax0.set_ylim(*loc_range)
        ax1.set_ylim(0, 10)
        ax2.set_ylim(0, 1)

        ax0.set_ylabel(r'$\mu$')
        ax1.set_ylabel(r'$\kappa$')
        ax2.set_ylabel('prob.')

        plt.tight_layout(pad=0.5)

        esuffix = '%s_%s' % (group_name, val)
        save_fig(fig, op.join(dirname, prefix + '_group_' + esuffix), suffix)
        plt.close(fig)

with open(op.join(dirname, 'groups.csv'), 'w') as fd:
    fd.write('group name, group value, dataset, chisquare(e) p-value,'
             ' aic, bic, mu0, mu1, kappa, prob\n')

    writer = csv.writer(fd)
    for row in rows:
        writer.writerow(row)
