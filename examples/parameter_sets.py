"""
Fitting parameter sets configuration.

By default, unspecified options are taken from a previous set. In case of the
first set, defaults are provided in the code. Command-line options can be used
to override the settings below.
"""

parameter_sets = [
    {
        'n_components' : 1,
        'parameters' : [2.0, 0.0], # Starting values.
        'solver' : ('bfgs', {'gtol' : 1e-8}),
        'output_dir' : 'output/1',
    },
    {
        'n_components' : 2,
        'parameters' : 'previous', # Starting value from the previous run.
        'output_dir' : 'output/2',
    },
    {
        'n_components' : 3,
        'output_dir' : 'output/3',
    },
]
