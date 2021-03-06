"""
IO utility functions.
"""
import os
import fnmatch

import numpy as np

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

def split_dir_base(filenames):
    """
    Split filenames to the base directory name (the directory containing the
    files) and the actual filenames..

    Returns
    -------
    dir_base : str
        The base directory name.
    base_names : list of str
        The file names.
    """
    aux = filenames[0].split(os.path.sep)
    dir_base = aux[-2] if len(aux) > 1 else ''
    base_names = [os.path.basename(ii) for ii in filenames]

    return dir_base, base_names

def load_data(filenames, transform=None):
    if transform is None:
        transform = lambda x: x

    datas = []
    for filename in filenames:
        data = np.genfromtxt(filename)
        if not np.isfinite(data[0, 0]): # First line is comment.
            data = data[1:].copy()
        data[:, 0] = transform(data[:, 0])
        datas.append(data)

    datas = np.array(datas)

    merged_data = datas[0]
    for dd in datas[1:]:
        if not (dd[:, 0] == merged_data[:, 0]).all():
            raise ValueError('x axes do not match in %s' % filenames)
        merged_data[:, 1] += dd[:, 1]

    return merged_data
