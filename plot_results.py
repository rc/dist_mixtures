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
print datas

def plot_fit_info(fig_num, datas, key):
    plt.figure(fig_num)
    plt.clf()
    plt.subplots_adjust(bottom=0.12)
    ax = plt.gca()

    dys = []
    labels = []
    for data in datas:
        dy = data.get_value(key)
        dys.append(dy)
        labels.append('%d' % data.items[0].n_components)
    dys = np.asarray(dys)

    dx = np.arange(dys.shape[1])
    dymin = dys.min(axis=0)
    for dy in dys:
        plt.plot(dx, dy - dymin, marker='o')

    ylim = ax.get_ylim()
    yshift = 0.1 * (ylim[1] - ylim[0])

    for ii, _dx in enumerate(dx):
        plt.text(_dx, ylim[0] - 0.5 * yshift, '%.2e' % dymin[ii],
                 size=10, rotation=-20)

    ax.set_ylim([ylim[0] - yshift, ylim[1]])

    plt.xticks(rotation=70)
    ax.set_xticks(dx)
    ax.set_xticklabels([ii.dir_base for ii in datas[0].items])
    ax.set_ylabel(key)
    ax.legend(labels)

plot_fit_info(1, datas, 'nllf')
plot_fit_info(2, datas, 'aic')
plot_fit_info(3, datas, 'bic')
plot_fit_info(4, datas, 'chisquare')
plot_fit_info(5, datas, 'chisquare p-value')
plt.show()
