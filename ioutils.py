"""
IO utility functions.
"""
import os
import fnmatch

def ensure_path(filename):
    """
    Check if path to `filename` exists and if not, create the necessary
    intermediate directories.
    """
    dirname = os.path.dirname(filename + os.path.sep)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

def locate_files(pattern, root_dir=os.curdir, dir_pattern=None,
                 group_last_level=False):
    """
    Locate all files matching fiven filename pattern in and below
    supplied root directory.
    """
    root_dir = os.path.abspath(root_dir)
    dir_pattern = dir_pattern if dir_pattern is not None else '*'

    if group_last_level:
        for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
            if not fnmatch.fnmatch(dirpath, dir_pattern): continue
            dirnames.sort()
            aux = fnmatch.filter(filenames, pattern)
            if not aux: continue
            yield [os.path.join(dirpath, ii) for ii in aux]

    else:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in fnmatch.filter(filenames, pattern):
                yield os.path.join(dirpath, filename)
