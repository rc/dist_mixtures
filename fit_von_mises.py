#!/usr/bin/env python
"""
Fit data files with names matching a given pattern by a mixture of von Mises
distributions.
"""
import os
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt

import analyses.ioutils as io
import analyses.transforms as tr
import analyses.plots as pl
from analyses.logs import CSVLog
from analyses.fit_mixture import get_start_params, fit, log_results, DataSource

usage = '%prog [options] pattern data_dir\n' + __doc__.rstrip()

defaults = {
    'n_components' : 2,
    'output_dir' : 'output',
}

help = {
    'output_dir' :
    'output directory [default: %s]' % defaults['output_dir'],
    'conf' :
    'use configuration file with parameter sets.'
    ' Ignored, if n_components option is given.',
    'n_components' :
    'number of components of the mixture [default: %s]' \
    % defaults['n_components'],
    'parameters' :
    'initial guess of von Mises parameters for each component as a comma'
    ' separated list, e.g., for two components: "1,0,1,0" corresponding'
    ' to kappa0, mu0, kappa1, mu1 respectively. The location parameter'
    ' mu should be given in degrees in [-90, 90[.',
    'dir_pattern' :
    'pattern that subdrectories should match [default: %default]',
    'spread_data' :
    'spread raw data using their counts instead of just repeating them',
    'area_angles' :
    'compute and draw angles of two systems of fibres determined by'
    ' equal histogram areas',
    'neg_shift' :
    'do not add 180 degrees to negative angles',
    'show' :
    'show the figures',
}

def get_options_parser():
    parser = OptionParser(usage=usage, version='%prog')
    parser.add_option('-o', '--output-dir', metavar='dirname',
                      action='store', dest='output_dir',
                      default=None, help=help['output_dir'])
    parser.add_option('-c', '--conf', metavar='filename',
                      action='store', dest='conf',
                      default=None, help=help['conf'])
    parser.add_option('-n', '--n-components', type=int, metavar='positive_int',
                      action='store', dest='n_components',
                      default=None, help=help['n_components'])
    parser.add_option('-p', '--parameters', metavar='kappa0,mu0,kappa1,mu1,...',
                      action='store', dest='parameters',
                      default=None, help=help['parameters'])
    parser.add_option('-d', '--dir-pattern', metavar='pattern',
                      action='store', dest='dir_pattern',
                      default='*', help=help['dir_pattern'])
    parser.add_option('', '--spread-data',
                      action='store_true', dest='spread_data',
                      default=False, help=help['spread_data'])
    parser.add_option('-a', '--area-angles',
                      action='store_true', dest='area_angles',
                      default=False, help=help['area_angles'])
    parser.add_option('', '--no-neg-shift',
                      action='store_false', dest='neg_shift',
                      default=True, help=help['neg_shift'])
    parser.add_option('-s', '--show',
                      action='store_true', dest='show',
                      default=False, help=help['show'])

    return parser

def main():
    parser = get_options_parser()
    options, args = parser.parse_args()

    if len(args) == 2:
        pattern, data_dir = args
    else:
        parser.print_help()
        return

    default_conf = [
        {
            'n_components' : 2,
            'parameters' : [2.0, 0.0], # Starting value.
            'solver' : ('bfgs', {'gtol' : 1e-8}),
        }
    ]

    if (options.conf is not None) and (options.n_components is None):
        import imp
        from analyses.parameter_sets import create_parameter_sets
        name = os.path.splitext(options.conf)[0]
        aux = imp.find_module(name)
        cc = imp.load_module('pars', *aux)
        psets = create_parameter_sets(cc.parameter_sets)

    else:
        psets = create_parameter_sets(default_conf)

    logs = []
    for ii, pset in enumerate(psets):
        previous = psets[ii - 1] if ii > 0 else None
        pset.override(options, default_conf[0], previous)

        io.ensure_path(pset.output_dir)

        log_name = os.path.join(pset.output_dir, 'log_%d.csv' % ii)
        log = CSVLog(log_name, pset.n_components)
        log.write_header()
        logs.append(log)

    get_data = io.locate_files(pattern, data_dir,
                               dir_pattern=options.dir_pattern,
                               group_last_level=True)
    source = DataSource(get_data, options.spread_data, options.neg_shift)
    for data, fdata, bins in source():
        pl.plot_raw_data(psets[0].output_dir, source,
                         area_angles=options.area_angles)

        # Loop over parameter sets. Each has its own CSVLog.
        res = None
        for ii, pset in enumerate(psets):
            if (ii > 0) and pset.parameters == 'previous':
                start_parameters = get_start_params(pset.n_components,
                                                    res.params)

            else:
                start_parameters = get_start_params(pset.n_components,
                                                    pset.parameters)
            print 'starting parameters:', start_parameters

            res = fit(fdata, start_parameters)
            res.model.summary_params(res.params,
                                     name='%d components' % pset.n_components)

            pl.plot_estimated_dist(pset.output_dir, res, source)
            pl.plot_histogram_comparison(pset.output_dir, res, source)

            log = logs[ii]
            log_results(log, res, source)

            if options.show:
                plt.show()

if __name__ == '__main__':
    main()
