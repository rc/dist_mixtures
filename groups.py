"""
f = female
m = male
u = unknown
c = castrated male
"""
import csv
import numpy as np

age_groups = np.array([[5, 10], [11, 20], [21, 30], [31, 100], [101, 1000]])

def get_age_group(age):
    return 1 + (np.searchsorted(age_groups.ravel(), [age])[0] // 2)

def parse_row(row):
    return (row[0], np.int32(row[1]), np.int32(row[2]), row[3], row[4])

def read_group_info():
    group_info = np.empty(83,
                          dtype=[('dataset', 'a6'),
                                 ('number', 'i4'),
                                 ('age', 'i4'),
                                 ('sex', 'a1'),
                                 ('segment', 'a1')])

    with open('data-aorta/smc_orientation_tangential.csv', 'rb') as fd:
        reader = csv.reader(fd)
        reader.next()
        group_info[:] = [parse_row(row) for row in reader]

    return group_info

def map_group_names(group_info):
    gmap = {}
    for row in group_info:
        gmap[row['dataset']] = (row['segment'], get_age_group(row['age']))

    return gmap

def get_datasets_of_group(group_info, group):
    if isinstance(group, int):
        age = group_info['age']
        col = np.array([get_age_group(ii) for ii in age])

    else:
        col = group_info['segment']

    ii = np.where(col == group)[0]
    return group_info['dataset'][ii]
