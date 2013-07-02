"""
Logging.
"""
import csv
import os

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

class CSVLog(Struct):
    """
    Log von Mises mixture fitting results.
    """

    def __init__(self, log_name, n_components):
        self.log_name = log_name
        self.n_components = n_components
        self.reset()

    def reset(self):
        self.items = []

    def write_header(self):
        fd = open(self.log_name, 'w')

        header = ['directory', 'converged', 'chisquare', 'chisquare p-value']
        for ii in range(self.n_components):
            header.extend(['mu%d' % ii, 'kappa%d' % ii, 'prob%d' % ii])
        header.extend(['nllf', 'aic', 'bic', 'number of files', 'filenames'])

        writer = csv.writer(fd)
        writer.writerow(header)

        fd.close()

    def write_row(self, dir_base, base_names, chisquare, params, converged,
                  fit_criteria):
        item = Struct(dir_base=dir_base, base_names=base_names,
                      chisquare=chisquare, params=params, converged=converged,
                      fit_criteria=fit_criteria)
        self.items.append(item)

        fd = open(self.log_name, 'a')

        writer = csv.writer(fd)
        writer.writerow([dir_base, int(converged), chisquare[0], chisquare[1]]
                        + params.ravel().tolist() + fit_criteria
                        + [len(base_names), ', '.join(base_names)])
        fd.close()

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
