from dist_mixtures.base import Struct

class ParameterSets(Struct):

    @staticmethod
    def from_conf(confs):
        """
        Create a parameter sets from a list of paramter set configurations.
        """
        psets = [ParameterSet.from_conf(conf) for conf in confs]
        obj = ParameterSets(psets)

        return obj

    def __init__(self, psets):
        Struct.__init__(self, psets=psets)

    def __getitem__(self, ii):
        return self.psets[ii]

    def __iter__(self):
        return iter(self.psets)

    def setup_options(self, options, default_conf=None):
        """
        Setup parameter sets using options and default configuration.
        """
        for ii, pset in enumerate(self.psets):
            previous = self.psets[ii - 1] if ii > 0 else None
            pset.override(options, default_conf[0], previous)

def _override(attr, obj, options, defaults, previous=None):
    value = getattr(options, attr, None)
    if value is not None:
        setattr(obj, attr, value)

    value = getattr(obj, attr)
    if value is None:
        if previous is None:
            if isinstance(defaults, dict):
                setattr(obj, attr, defaults.get(attr))

            else:
                setattr(obj, attr, getattr(defaults, attr))

        else:
            setattr(obj, attr, getattr(previous, attr))

    value = getattr(obj, attr)
    assert value is not None

class ParameterSet(Struct):

    @staticmethod
    def from_conf(conf):
        return ParameterSet(conf.get('model_class'), conf.get('n_components'),
                            conf.get('parameters'),
                            conf.get('solver'), conf.get('output_dir'))

    def __init__(self, model_class, n_components, parameters, solver_conf,
                 output_dir):
        Struct.__init__(self, model_class=model_class,
                        n_components=n_components, parameters=parameters,
                        solver_conf=solver_conf, output_dir=output_dir)

    def override(self, options, defaults, previous=None):
        """
        Override parameter from `options`, provide default values of unspecified
        attributes from `previous` or `defaults`.
        """
        for attr in ['model_class', 'n_components', 'parameters',
                     'solver_conf', 'output_dir']:
            _override(attr, self, options, defaults, previous=previous)
