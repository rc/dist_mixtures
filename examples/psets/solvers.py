"""
Parameter sets with changing the optimization solver.

By default, unspecified options are taken from a previous set. In case of the
first set, defaults are provided in the code. Command-line options can be used
to override the settings below.

Local minimizer methods:

  - 'Nelder-Mead'
  - 'Powell'
  - 'CG'
  - 'BFGS'
  - 'Newton-CG'
  - 'Anneal'
  - 'L-BFGS-B'
  - 'TNC'
  - 'COBYLA'
  - 'SLSQP'
  - 'dogleg'
  - 'trust-ncg'
"""
import dist_mixtures.mixture_von_mises as mvm

parameter_sets = [
    {
        'model_class' : mvm.VonMisesMixture,
        'n_components' : 2,
        'parameters' : [2.0, 0.0], # Starting values.
        'solver' : ('bfgs', {'gtol' : 1e-8}),
        'output_dir' : 'output/solvers',
    },
    {
        'solver' : ('basinhopping', {
            'T' : 0.00001, 'stepsize' : 0.1, 'niter' : 20,
            'minimizer' : {
                'method' : 'L-BFGS-B',
                'tol' : 1e-9,
            }
        }),
    },
    {
        'n_components' : 3,
        'solver' : ('bfgs', {'gtol' : 1e-8}),
    },
    {
        'solver' : ('basinhopping', {
            'T' : 0.00001, 'stepsize' : 0.1, 'niter' : 20,
            'minimizer' : {
                'method' : 'L-BFGS-B',
                'tol' : 1e-9,
            }
        }),
    },
]
