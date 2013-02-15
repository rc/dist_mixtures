"""
Parameter sets with 2 von Mises mixture components. The fitting model class is
changing.

By default, unspecified options are taken from a previous set. In case of the
first set, defaults are provided in the code. Command-line options can be used
to override the settings below.
"""
import dist_mixtures.mixture_von_mises as mvm

parameter_sets = [
    {
        'model_class' : mvm.VonMisesMixtureBinned,
        'n_components' : 2,
        'parameters' : [2.0, 0.0, 3.0, 0.0], # Starting values.
        'solver' : ('bfgs', {'gtol' : 1e-8}),
        'output_dir' : 'output/models',
    },
    {
        'model_class' : mvm.VonMisesMixture,
    },
]
