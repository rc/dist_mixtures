#!/usr/bin/env python
"""
Fit data files with names matching a given pattern by a mixture of von Mises
distributions.
"""
import os
import sys
import shutil
from optparse import OptionParser

import dist_mixtures.mixture_von_mises as mvm
from dist_mixtures.base import ordered_iteritems, LogOutput
import analyses.ioutils as io
from analyses.parameter_sets import ParameterSets
from analyses.fit_mixture import (analyze, print_results, make_summary,
                                  print_summary, print_angles, DataSource)

def parse_rc(option, opt, value, parser):
    pars = {}
    for pair in value.split(','):
        key, val = pair.split('=')
        pars[key] = eval(val)

    setattr(parser.values, option.dest, pars)

usage = '%prog [options] pattern data_dir\n' + __doc__.rstrip()

defaults = {
    'n_components' : 2,
    'n_merge_bins' : None,
    'plot_bins_step' : 4,
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
    'pattern that subdirectories should match [default: %default]',
    'n_merge_bins' :
    'number of consecutive bins in data to merge [default: %s]' \
    % defaults['n_merge_bins'],
    'plot_bins_step' :
    'step to choose bins from all bins for histogram plots [default: %s]' \
    % defaults['plot_bins_step'],
    'spread_data' :
    'spread raw data using their counts instead of just repeating them',
    'area_angles' :
    'compute and draw angles of two systems of fibres determined by'
    ' equal histogram areas',
    'neg_shift' :
    'do not add 180 degrees to negative angles',
    'rc' : 'matplotlib resources',
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
    parser.add_option('-m', '--merge-bins', type=int, metavar='positive_int',
                      action='store', dest='n_merge_bins',
                      default=None, help=help['n_merge_bins'])
    parser.add_option('', '--plot-bins-step', type=int, metavar='positive_int',
                      action='store', dest='plot_bins_step',
                      default=None, help=help['plot_bins_step'])
    parser.add_option('', '--spread-data',
                      action='store_true', dest='spread_data',
                      default=False, help=help['spread_data'])
    parser.add_option('-a', '--area-angles',
                      action='store_true', dest='area_angles',
                      default=False, help=help['area_angles'])
    parser.add_option('', '--no-neg-shift',
                      action='store_false', dest='neg_shift',
                      default=True, help=help['neg_shift'])
    parser.add_option('--rc', type='str', metavar='key=val,...',
                      action='callback', dest='rc',
                      callback=parse_rc, default={}, help=help['rc'])
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
            'model_class' : mvm.VonMisesMixture,
            'n_components' : 2,
            'parameters' : [2.0, 0.0], # Starting value.
            'solver' : ('bfgs', {'gtol' : 1e-8, 'disp' : True}),
            'output_dir' : defaults['output_dir']
        }
    ]

    if (options.conf is not None) and (options.n_components is None):
        import imp
        name = os.path.splitext(options.conf)[0]
        aux = imp.find_module(name)
        cc = imp.load_module('pars', *aux)
        psets = ParameterSets.from_conf(cc.parameter_sets)

    else:
        psets = ParameterSets.from_conf(default_conf)

    psets.setup_options(options, default_conf=default_conf)

    # Log input parameters.
    output_dir = psets[0].output_dir
    io.ensure_path(output_dir)

    stdout = sys.stdout
    output = LogOutput(sys.stdout, os.path.join(output_dir,'output_log.txt'))
    sys.stdout = output

    pars_filename = os.path.join(output_dir, 'pars.txt')
    with open(pars_filename, 'w') as fd:
        fd.write('command line\n')
        fd.write('------------\n\n')
        fd.write(' '.join(sys.argv) + '\n')

        fd.write('\noptions\n')
        fd.write('-------\n\n')
        for key, val in ordered_iteritems(vars(options)):
            fd.write('%s: %s\n' % (key, val))

        fd.write('\ndefaults\n')
        fd.write('--------\n\n')
        for key, val in ordered_iteritems(default_conf[0]):
            fd.write('%s: %s\n' % (key, val))

    if (options.conf is not None) and (options.n_components is None):
        # Copy configuration file.
        shutil.copy2(options.conf, output_dir)

    get_data = io.locate_files(pattern, data_dir,
                               dir_pattern=options.dir_pattern,
                               group_last_level=True)
    source = DataSource(get_data, options.n_merge_bins, options.plot_bins_step,
                        options.spread_data, options.neg_shift)

    logs, alog = analyze(source, psets, options)
    print_results(psets, logs)

    summary = make_summary(logs)
    print_summary(summary, logs)

    if alog is not None:
        print_angles(alog)

    sys.stdout = stdout

if __name__ == '__main__':
    main()
