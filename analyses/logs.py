"""
Logging.
"""
import csv

class CSVLog(object):
    """
    Log von Mises mixture fitting results.
    """

    def __init__(self, log_name, n_components):
        self.log_name = log_name
        self.n_components = n_components

    def write_header(self):
        fd = open(self.log_name, 'w')

        header = ['directory', 'negative kappa flag', 'fit not converged flag']
        for ii in range(self.n_components):
            header.extend(['mu%d' % ii, 'kappa%d' % ii, 'prob%d' % ii])
        header.extend(['nllf', 'aic', 'bic', 'number of files', 'filenames'])

        writer = csv.writer(fd)
        writer.writerow(header)

        fd.close()

    def write_row(self, dir_base, base_names, params, flags, fit_criteria):
        fd = open(self.log_name, 'a')

        writer = csv.writer(fd)
        writer.writerow([dir_base, flags[0], flags[1]]
                        + params.ravel().tolist() + fit_criteria
                        + [len(base_names), ', '.join(base_names)])
        fd.close()
