#!/usr/bin/env python
import csv

with open('output/area-angles/log_angles.csv', 'rb') as fd:
    reader = csv.reader(fd)
    header = reader.next()
    rows = [row for row in reader]

with open('output/area_angles.csv', 'w') as fd:
    fd.write('dataset, -delta angle, centre angle, +delta angle,'
             ' -area, +area\n')

    writer = csv.writer(fd)
    for row in rows:
        xs = [float(ii) for ii in row[1:8]]
        xm = xs[2]
        rr = [row[0], '%+.2f' % (xs[1] - xm), '%.2f' % xm,
              '%+.2f' % (xs[3] - xm), '%.2e' % xs[5], '%.2e' % xs[6]]
        writer.writerow(rr)
