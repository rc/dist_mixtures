from dist_mixtures.base import Struct

def create_parameter_sets(confs):
    """
    Create a list of parameter set objects from a list of paramter set
    configurations.
    """
    return [ParameterSet.from_conf(conf) for conf in confs]

def _override(attr, obj, options, defaults, previous=None):
    value = getattr(options, attr, None)
    if value is not None:
        setattr(obj, attr, value)

    value = getattr(obj, attr)
    if value is None:
        if previous is None:
            setattr(obj, attr, getattr(defaults, attr))

        else:
            setattr(obj, attr, getattr(previous, attr))

    value = getattr(obj, attr)
    assert value is not None

class ParameterSet(Struct):

    @staticmethod
    def from_conf(conf):
        return ParameterSet(conf.get('n_components'),
                            conf.get('parameters'),
                            conf.get('solver'), conf.get('output_dir'))

    def __init__(self, n_components, parameters, solver_conf, output_dir):
        Struct.__init__(self, n_components=n_components, parameters=parameters,
                        solver_conf=solver_conf, output_dir=output_dir)

    def override(self, options, defaults, previous=None):
        """
        Override parameter from `options`, provide default values of unspecified
        attributes from `previous` or `defaults`.
        """
        for attr in ['n_components', 'parameters', 'solver_conf',
                     'output_dir']:
            _override(attr, self, options, defaults, previous=previous)
