"""
Parameter sets to demonstrate user-defined parameter constraints.

Notes
-----
Needs to be run with the --area-angles option.
"""
import numpy as np

import dist_mixtures.mixture_von_mises as mvm
import analyses.transforms as tr

def get_pars(pset, area_angles):
    """
    Get starting parameters given the area angles of two systems.
    """
    x0, xm, x1, area1, area2 = area_angles

    mu0 = 0.5 * (x0 + xm)
    mu1 = 0.5 * (xm + x1)

    print mu0, mu1

    mu0, mu1 = tr.fix_range(tr.transform_2pi([mu0, mu1]))

    pars = np.r_[[2.0, mu0, 2.0, mu1]
                 + [0.1, 0.0] * (pset.n_components - 2)]

    return pars

def constrain_sym(pset, pars0):
    """
    Single shape, same probability for two systems, other systems free.

    Returns
    -------
    use : array
        True for parameters that are to be fitted.
    pars : array
        Initial values of parameters that are to be fitted.
    """
    nc = pset.n_components

    use = np.ones(pars0.shape, dtype=np.bool)

    if nc < 2: return use

    use[0] = False # Kappa_0
    use[-1] = False # Gamma_nc-1

    return use, pars0.copy()

def expand_sym(self, pars):
    """
    Use with `constrain_sym()`.
    """
    parsfull = self.fixed_params.copy()
    parsfull[self.fixed_paramsmask] = pars
    parsfull[0] = parsfull[2]

    nc = (len(parsfull) - 2) / 3 + 1
    if nc == 3:
        parsfull[-2] = parsfull[-1] = pars[-1]

    return parsfull

def constrain_sym_delta(pset, pars0):
    """
    As `constrain_sym()` and fixed difference between locations of the two
    symmetrical systems.

    Returns
    -------
    use : array
        True for parameters that are to be fitted.
    pars : array
        Initial values of parameters that are to be fitted.
    """
    nc = pset.n_components

    use, pars = constrain_sym(pset, pars0)
    if nc < 2: return use

    use[3] = False # Delta.
    pars[3] = pars[3] - pars[1]

    return use, pars

def expand_sym_delta(self, pars):
    """
    Use with `constrain_sym_delta()`.
    """
    parsfull = expand_sym(self, pars)
    parsfull[3] = parsfull[1] + parsfull[3]

    return parsfull

def constrain_sym_loc(pset, pars0):
    """
    As `constrain_sym()` and fixed locations of the two symmetrical systems.

    Returns
    -------
    use : array
        True for parameters that are to be fitted.
    pars : array
        Initial values of parameters that are to be fitted.
    """
    nc = pset.n_components

    use, pars = constrain_sym(pset, pars0)
    if nc < 2: return use

    use[1] = False # Mu_0.
    use[3] = False # Mu_1.

    return use, pars

parameter_sets = [
    {
        'model_class' : mvm.VonMisesMixtureBinned,
        'n_components' : 3,
        'parameters' : (get_pars, 'area_angles'),
        'constraints' : (constrain_sym_loc, expand_sym),
        'solver' : ('bfgs', {'gtol' : 1e-8}),
        'output_dir' : 'output/constraints',
    },
    {
        'constraints' : (constrain_sym_delta, expand_sym_delta),
    },
    {
        'constraints' : (constrain_sym, expand_sym),
    },
    {
        'constraints' : (),
    },
]
