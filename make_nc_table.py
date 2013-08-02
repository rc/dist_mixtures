#!/usr/bin/env python
import sys
import csv

import numpy as np

from analyses.logs import read_logs

outname = sys.argv[1]
dirname = sys.argv[2]

# The reduced table does not contain the subjective N_c column and nllf
# criterion.
reduced = True

logs = read_logs(dirname, 'log_?.csv')
nlogs = len(logs)

from analyses.fit_mixture import make_summary
summary = make_summary(logs)

id2nc = np.array([log.n_components for log in logs])

dir_bases = [item.dir_base for item in logs[0].items]

min_nc = 1 if reduced else 2
max_ncs = range(min_nc, id2nc.max() + 1)

if reduced:
    crits = {1 : 'aic', 2 : 'bic'}
    header1 = ', '.join(['N_c, %d' % ii for ii in max_ncs])
    header2 = ', '.join(['aic, bic' for ii in max_ncs])

else:
    crits = {0 : 'nllf', 1 : 'aic', 2 : 'bic'}
    header1 = ', '.join(['N_c, %d, ' % ii for ii in max_ncs])
    header2 = ', '.join(['nllf, aic, bic' for ii in max_ncs])

print header1
print header2

columns = []

weak_prob = 0.1
background_kappa = 0.75
#background_kappa = 1.0

filtered_ncs = []
pvals = []
for id, log in enumerate(logs):
    # weak_prob in percentage of single component prob assuming equal probs?
    # wp = (1.0 / id2nc[id]) * weak_prob
    wp = weak_prob
    ncs, iws = log.filter_weak_components(weak_prob=wp)
    print ncs
    ncs, ibs = log.filter_background(background_kappa=background_kappa,
                                     iws=iws)
    print ncs
    filtered_ncs.append(ncs)

    pvals.append(log.get_value('chisquare(e) p-value'))

filtered_ncs =  np.array(filtered_ncs)

filtered_ncs =  np.maximum(filtered_ncs, 1)

# 1. order by criteria
# 2. remove weak/background components

# Loop over max. allowed nc.
ncol = 2 if reduced else 3
for max_nc in max_ncs:
    print 'max. nc', max_nc

    column3 = np.zeros((len(dir_bases), ncol), dtype='a7')

    for idir, dir_base in enumerate(dir_bases):
        print dir_base

        ss = summary[dir_base]

        # Criterion loop.
        ic = 0
        for ik, crit in crits.iteritems():
            print ik, crit

            ids = ss.pset_ids[:, ik]
            ncs = id2nc[ids]
            print ncs

            inc = np.where(ncs <= max_nc)[0]
            chosen_id = ids[inc[0]]
            chosen_nc = ncs[inc[0]]

            print 'chosen id, nc', chosen_id, chosen_nc

            fnc = filtered_ncs[chosen_id, idir]
            pval = pvals[ids[inc[0]]][idir]
            if reduced:
                if pval >= 0.99999:
                    mark = '***'
                elif pval >= 0.99:
                    mark = '** '
                elif pval >= 0.95:
                    mark = '*  '
                else:
                    mark = '   '

            else:
                if pval >= 0.95:
                    mark = '*  '
                else:
                    mark = '   '

            column3[idir, ic] = '%d(%d)%s' % (fnc, chosen_nc, mark)
            ic += 1

    columns.append(column3)

columns = np.concatenate(columns, axis=1)

with open('output/n_c_rc.csv', 'rb') as fd:
    reader = csv.reader(fd)
    header = reader.next()
    rows = [row for row in reader]

with open(outname, 'w') as fd:
    if reduced:
        fd.write(header[0] + ', ' + header1 + '\n')
        fd.write(' , ' + header2 + '\n')

    else:
        fd.write(','.join(header) + ', ' + header1 + '\n')
        fd.write(' , , ' + header2 + '\n')

    writer = csv.writer(fd)
    for row in rows:
        ii = dir_bases.index(row[0])
        rr = row[:1] if reduced else row
        writer.writerow(rr + columns[ii].tolist())

