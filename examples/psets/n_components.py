"""
Parameter sets with a changing number of von Mises mixture components.

By default, unspecified options are taken from a previous set. In case of the
first set, defaults are provided in the code. Command-line options can be used
to override the settings below.
"""
import dist_mixtures.mixture_von_mises as mvm

parameter_sets = [
    {
        'model_class' : mvm.VonMisesMixtureBinned,
        'n_components' : 1,
        'parameters' : [2.0, 0.0], # Starting values.
        'solver' : ('bfgs', {'gtol' : 1e-8}),
        'output_dir' : 'output/n_components',
    },
    {
        'n_components' : 2,
        'parameters' : 'previous', # Starting value from the previous run.
    },
    {
        'n_components' : 3,
    },
]
