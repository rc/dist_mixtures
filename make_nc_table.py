#!/usr/bin/env python
import sys
import csv

import numpy as np

from analyses.logs import read_logs

outname = sys.argv[1]
dirname = sys.argv[2]

logs = read_logs(dirname, 'log_*.csv')
nlogs = len(logs)

from analyses.fit_mixture import make_summary
summary = make_summary(logs)

id2nc = np.array([log.n_components for log in logs])

dir_bases = [item.dir_base for item in logs[0].items]

crits = {0 : 'nllf', 1 : 'aic', 2 : 'bic'}

max_ncs = range(2, id2nc.max() + 1)

header1 = ', '.join(['N_c(%d), , ' % ii for ii in max_ncs])
header2 = ', '.join(['nllf, aic, bic' for ii in max_ncs])
print header1
print header2

columns = []

weak_prob = 0.1
background_kappa = 0.75

filtered_ncs = []
for log in logs:
    ncs, iws = log.filter_weak_components(weak_prob=weak_prob)
    print ncs
    ncs, ibs = log.filter_background(background_kappa=background_kappa,
                                     iws=iws)
    print ncs
    filtered_ncs.append(ncs)

filtered_ncs =  np.array(filtered_ncs)

# 1. order by criteria
# 2. remove weak/background components

# Loop over max. allowed nc.
for max_nc in max_ncs:
    print 'max. nc', max_nc

    column3 = np.zeros((len(dir_bases), 3), dtype=np.int32)

    for idir, dir_base in enumerate(dir_bases):
        print dir_base

        ss = summary[dir_base]

        # Criterion loop.
        for ik, crit in crits.iteritems():
            print ik, crit

            ids = ss.pset_ids[:, ik]
            ncs = id2nc[ids]
            print ncs

            inc = np.where(ncs <= max_nc)[0]
            chosen_id = ids[inc[0]]
            chosen_nc = ncs[inc[0]]

            print 'chosen id, nc', chosen_id, chosen_nc

            log = logs[ids[inc[0]]]

            column3[idir, ik] = filtered_ncs[chosen_id, idir]

    columns.append(column3)

columns = np.concatenate(columns, axis=1)

with open('output/n_c_rc.csv', 'rb') as fd:
    reader = csv.reader(fd)
    header = reader.next()
    rows = [row for row in reader]

with open(outname, 'w') as fd:
    fd.write(','.join(header) + ', ' + header1 + '\n')
    fd.write(' , , ' + header2 + '\n')

    writer = csv.writer(fd)
    for row in rows:
        ii = dir_bases.index(row[0])
        writer.writerow(row + columns[ii].tolist())

# weak_prob in percentage of single component prob assuming equal probs?

