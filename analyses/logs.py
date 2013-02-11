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

    def write_header(self):
        fd = open(self.log_name, 'w')

        header = ['directory', 'fit not converged flag']
        for ii in range(self.n_components):
            header.extend(['mu%d' % ii, 'kappa%d' % ii, 'prob%d' % ii])
        header.extend(['nllf', 'aic', 'bic', 'number of files', 'filenames'])

        writer = csv.writer(fd)
        writer.writerow(header)

        fd.close()

    def write_row(self, dir_base, base_names, params, flags, fit_criteria):
        fd = open(self.log_name, 'a')

        writer = csv.writer(fd)
        writer.writerow([dir_base, flags[0]]
                        + params.ravel().tolist() + fit_criteria
                        + [len(base_names), ', '.join(base_names)])
        fd.close()
