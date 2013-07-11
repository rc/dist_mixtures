"""
Logging.
"""
import csv
import os

import numpy as np
from dist_mixtures.base import Struct

def create_logs(psets):
    """
    Create CSV logs for given parameter sets.
    """
    import analyses.ioutils as io

    logs = []
    for ii, pset in enumerate(psets):
        io.ensure_path(pset.output_dir)

        log_name = os.path.join(pset.output_dir, 'log_%d.csv' % ii)
        log = CSVLog(log_name, pset.n_components)
        log.write_header()
        logs.append(log)

    return logs

def read_logs(dirname, pattern):
    """
    Read all csv logs in directory `dirname` with file names given by
    `pattern`.
    """
    from glob import glob

    filenames = sorted(glob(os.path.join(dirname, pattern)))

    logs = []
    for filename in filenames:
        print filename
        log = CSVLog.from_file(filename)
        logs.append(log)

    return logs

class CSVLog(Struct):
    """
    Log von Mises mixture fitting results.
    """
    @classmethod
    def from_file(cls, filename):
        obj = cls(filename, -1)

        with open(filename, 'rb') as fd:
            reader = csv.reader(fd)
            obj.header = reader.next()
            obj.items = [obj.parse_row(row) for row in reader]

        obj.setup_items_dict()
        obj.n_components = obj.items[0].n_components

        return obj

    def __init__(self, log_name, n_components):
        self.log_name = log_name
        self.n_components = n_components
        self.reset()

    def reset(self):
        self.items = []

    def write_header(self):
        fd = open(self.log_name, 'w')

        header = ['directory', 'converged', 'nllf', 'aic', 'bic',
                  'chisquare', 'chisquare p-value', 'n_components']
        for ii in range(self.n_components):
            header.extend(['mu%d' % ii, 'kappa%d' % ii, 'prob%d' % ii])
        header.extend(['number of files', 'filenames'])

        writer = csv.writer(fd)
        writer.writerow(header)

        fd.close()

    def write_row(self, dir_base, base_names, chisquare, params, converged,
                  fit_criteria):
        n_components = params.shape[0]
        item = Struct(dir_base=dir_base, base_names=base_names,
                      chisquare=chisquare, params=params, converged=converged,
                      n_components=n_components, fit_criteria=fit_criteria)
        self.items.append(item)

        fd = open(self.log_name, 'a')

        writer = csv.writer(fd)
        writer.writerow([dir_base, int(converged)] + fit_criteria +
                        [chisquare[0], chisquare[1], n_components]
                        + params.ravel().tolist()
                        + [len(base_names) , ', '.join(base_names)])
        fd.close()

    def parse_row(self, row):
        """
        Parse single hitogram row.
        """
        n_components = int(row[7])
        off = 8 + 3 * n_components
        params = np.array(map(float, row[8:off])).reshape((n_components, 3))
        item = Struct(dir_base=row[0], converged=int(row[1]),
                      fit_criteria=map(float, row[2:5]),
                      chisquare=map(float, row[5:7]),
                      n_components=n_components,
                      params=params,
                      base_names=[ii.strip()
                                  for ii in row[off + 1].split(',')])
        assert(len(item.base_names) == int(row[off]))

        return item

    def setup_items_dict(self):
        for item in self.items:
            item.vals = {'nllf' : item.fit_criteria[0],
                         'aic' : item.fit_criteria[1],
                         'bic' : item.fit_criteria[2],
                         'chisquare' : item.chisquare[0],
                         'chisquare p-value' : item.chisquare[1],
                         'n_components' : item.n_components,
                         'params' : item.params}

    def get_value(self, key):
        return [item.vals.get(key) for item in self.items]

    def filter_weak_components(self, weak_prob=0.1):
        """
        For each item return number of components corrected by ignoring
        components with probability less then `weak_prob`.
        """
        ncs = []
        for item in self.items:
            iw = np.where(item.params[:, 2] < weak_prob)[0]
            ncs.append(item.n_components - iw.shape[0])

        return np.array(ncs)

class AnglesCSVLog(Struct):
    """
    Log area angles.
    """

    def __init__(self, log_name):
        self.log_name = log_name
        self.reset()

    def reset(self):
        self.items = []

    def write_header(self):
        fd = open(self.log_name, 'w')

        header = ['directory', 'x0', 'x0m', 'xm', 'x1m', 'x1', 'area1', 'area2',
                  'number of files', 'filenames']

        writer = csv.writer(fd)
        writer.writerow(header)

        fd.close()

    def write_row(self, dir_base, base_names, area_angles):
        x0, xm, x1, area1, area2 = area_angles
        item = Struct(dir_base=dir_base, base_names=base_names,
                      x0=x0, xm=xm, x1=x1, area1=area1, area2=area2)
        self.items.append(item)

        fd = open(self.log_name, 'a')

        writer = csv.writer(fd)
        writer.writerow([dir_base] + [x0, 0.5 * (x0 + xm), xm,
                                      0.5 * (xm + x1), x1, area1, area2]
                        + [len(base_names), ', '.join(base_names)])
        fd.close()
