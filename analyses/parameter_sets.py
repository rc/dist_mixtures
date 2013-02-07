from dist_mixtures.base import Struct

def create_parameter_sets(confs):
    """
    Create a list of parameter set objects from a list of paramter set
    configurations.
    """
    return [ParameterSet.from_conf(conf) for conf in confs]

class ParameterSet(Struct):

    @staticmethod
    def from_conf(conf):
        return ParameterSet(conf.get('n_components'),
                            conf.get('parameters'),
                            conf.get('solver'), conf.get('output_dir'))

    def __init__(self, n_components, parameters, solver_conf, output_dir):
        Struct.__init__(self, n_components=n_components, parameters=parameters,
                        solver_conf=solver_conf, output_dir=output_dir)
