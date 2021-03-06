import numpy as np
import scipy.sparse as sp

def get_debug():
    """
    Utility function providing ``debug()`` function.
    """
    import sys
    try:
        import IPython

    except ImportError:
        debug = None

    else:
        old_excepthook = sys.excepthook

        def debug(frame=None):
            if IPython.__version__ >= '0.11':
                from IPython.core.debugger import Pdb

                try:
                    ip = get_ipython()

                except NameError:
                    from IPython.frontend.terminal.embed \
                         import InteractiveShellEmbed
                    ip = InteractiveShellEmbed()

                colors = ip.colors

            else:
                from IPython.Debugger import Pdb
                from IPython.Shell import IPShell
                from IPython import ipapi

                ip = ipapi.get()
                if ip is None:
                    IPShell(argv=[''])
                    ip = ipapi.get()

                colors = ip.options.colors

            sys.excepthook = old_excepthook

            if frame is None:
                frame = sys._getframe().f_back

            Pdb(colors).set_trace(frame)

    if debug is None:
        import pdb
        debug = pdb.set_trace

    debug.__doc__ = """
    Start debugger on line where it is called, roughly equivalent to::

        import pdb; pdb.set_trace()

    First, this function tries to start an `IPython`-enabled
    debugger using the `IPython` API.

    When this fails, the plain old `pdb` is used instead.
    """

    return debug

debug = get_debug()

class Struct(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def _format_sequence(self, seq, threshold):
        threshold_half = threshold / 2

        if len(seq) > threshold:
            out = ', '.join(str(ii) for ii in seq[:threshold_half]) \
                  + ', ..., ' \
                  + ', '.join(str(ii) for ii in seq[-threshold_half:])

        else:
            out = str(seq)

        return out

    def __str__(self):
        """
        Print instance class, name and items in alphabetical order.

        If the class instance has '_str_attrs' attribute, only the attributes
        listed there are taken into account. Other attributes are provided only
        as a list of attribute names (no values).

        For attributes that are Struct instances, if
        the listed attribute name ends with '.', the attribute is printed fully
        by calling str(). Otherwise only its class name/name are printed.

        Attributes that are NumPy arrays or SciPy sparse matrices are
        printed in a brief form.

        Only keys of dict attributes are printed. For the dict keys as
        well as list or tuple attributes only several edge items are
        printed if their length is greater than the threshold value 20.
        """
        return self._str()

    def _str(self, keys=None, threshold=20):
        ss = '%s' % self.__class__.__name__
        if hasattr(self, 'name'):
            ss += ':%s' % self.name
        ss += '\n'

        if keys is None:
            keys = self.__dict__.keys()

        str_attrs = sorted(self.get('_str_attrs', keys))
        printed_keys = []
        for key in str_attrs:
            if key[-1] == '.':
                key = key[:-1]
                full_print = True
            else:
                full_print = False

            printed_keys.append(key)

            try:
                val = getattr(self, key)

            except AttributeError:
                continue

            if isinstance(val, Struct):
                if not full_print:
                    ss += '  %s:\n    %s' % (key, val.__class__.__name__)
                    if hasattr(val, 'name'):
                        ss += ':%s' % val.name
                    ss += '\n'

                else:
                    aux = '\n' + str(val)
                    aux = aux.replace('\n', '\n    ');
                    ss += '  %s:\n%s\n' % (key, aux[1:])

            elif isinstance(val, dict):
                sval = self._format_sequence(val.keys(), threshold)
                sval = sval.replace('\n', '\n    ')
                ss += '  %s:\n    dict with keys: %s\n' % (key, sval)

            elif isinstance(val, list):
                sval = self._format_sequence(val, threshold)
                sval = sval.replace('\n', '\n    ')
                ss += '  %s:\n    list: %s\n' % (key, sval)

            elif isinstance(val, tuple):
                sval = self._format_sequence(val, threshold)
                sval = sval.replace('\n', '\n    ')
                ss += '  %s:\n    tuple: %s\n' % (key, sval)

            elif isinstance(val, np.ndarray):
                ss += '  %s:\n    %s array of %s\n' \
                      % (key, val.shape, val.dtype)

            elif isinstance(val, sp.spmatrix):
                ss += '  %s:\n    %s spmatrix of %s, %d nonzeros\n' \
                      % (key, val.shape, val.dtype, val.nnz)

            else:
                aux = '\n' + str(val)
                aux = aux.replace('\n', '\n    ');
                ss += '  %s:\n%s\n' % (key, aux[1:])

        other_keys = sorted(set(keys).difference(set(printed_keys)))
        if len(other_keys):
            ss += '  other attributes:\n    %s\n' \
                  % '\n    '.join(key for key in other_keys)

        return ss.rstrip()

    def __repr__( self ):
        ss = '%s' % self.__class__.__name__
        if hasattr( self, 'name' ):
            ss += ':%s' % self.name
        return ss

    def str_class(self):
        """
        As __str__(), but for class attributes.
        """
        return self._str(self.__class__.__dict__.keys())

    def get(self, key, default=None, msg_if_none=None):
        """
        A dict-like get() for Struct attributes.
        """
        out = getattr(self, key, default)

        if (out is None) and (msg_if_none is not None):
            raise ValueError(msg_if_none)

        return out

    def update(self, other, **kwargs):
        """
        A dict-like update for Struct attributes.
        """
        if other is None: return

        if not isinstance(other, dict):
            other = other.to_dict()
        self.__dict__.update(other, **kwargs)

    def set_default(self, key, default=None):
        """
        Behaves like dict.setdefault().
        """
        return self.__dict__.setdefault(key, default)

class LogOutput(Struct):
    """
    Log print statements to a given file.
    """

    def __init__(self, stdout, filename):
        self.stdout = stdout
        self.filename = filename
        with open(filename, 'w'):
            pass

    def write(self, text):
        self.stdout.write(text)
        with open(self.filename, 'a') as fd:
            fd.write(text)

    def close(self):
        pass

def ordered_iteritems(adict):
    keys = adict.keys()
    order = np.argsort(keys)
    for ii in order:
        key = keys[ii]
        yield key, adict[key]
