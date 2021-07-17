import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import h5py

# from pprint import pprint

from plasmapy.diagnostics.langmuir import Characteristic, swept_probe_analysis

# Make sure to add code comments!


def open_hdf5(filename):
    return h5py.File(filename, 'r')


def group_at_path(file, subpath):
    return file[subpath]


def structures_at_path(file, subpath):
    curr = file[subpath]
    # print(list(curr.keys()))
    groups = []
    datasets = []
    other = []
    for item in curr.keys():
        # categs[h5py.Group.get(item, getclass=True)]
        # print(curr.get(item, getclass=True))
        # itemType = ""
        if curr.get(item, getclass=True) == h5py._hl.group.Group:
            groups.append(subpath+'/'+item)
            # itemType = "Groups"
        elif curr.get(item, getclass=True) == h5py._hl.dataset.Dataset:
            datasets.append(subpath + '/' + item)
            # itemType = "Datasets"
        else:
            other.append(subpath + '/' + item)
            # itemType = "Other"
    categories = {"Name": subpath, "Groups": groups, "Datasets": datasets, "Other": other}
    return categories