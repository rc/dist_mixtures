"""
Parameter sets with 2 von Mises mixture components. Starting parameters of the
fitting procedure are changing.

By default, unspecified options are taken from a previous set. In case of the
first set, defaults are provided in the code. Command-line options can be used
to override the settings below.
"""
import dist_mixtures.mixture_von_mises as mvm

parameter_sets = [
    {
        'model_class' : mvm.VonMisesMixtureBinned,
        'n_components' : 2,
        'parameters' : [2.0, 0.0, 2.0, 0.0], # Starting values.
        'solver' : ('bfgs', {'gtol' : 1e-8}),
        'output_dir' : 'output/start_parameters',
    },
    {
        'parameters' : [2.0, 0.0, 4.0, 0.0],
    },
    {
        'parameters' : [4.0, 0.0, 2.0, 4.0],
    },
    {
        'parameters' : [1.0, -10.0, 8.0, 0.0],
    },
]
