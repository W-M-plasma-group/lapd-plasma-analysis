import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import h5py

# from pprint import pprint

from plasmapy.diagnostics.langmuir import Characteristic, swept_probe_analysis

# import plasmapy.plasma.sources as src
# import plasmapy.plasma.sources.openpmd_hdf5 as pmd

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


# test = open_hdf5('HDF5/8-3500A.hdf5')
# print(list(test.keys()))
# msi = test['MSI']
# print(list(msi.keys()))
# print(structuresAtPath(test, '/Raw data + config/6K Compumotor'))

# msi_accessor = groupAtPath(test, 'MSI')
# print(list(msi_accessor.keys()))

""" 
path = os.path.join(os.curdir, "HDF5", "8-3500A.hdf5")
contents = src.HDF5Reader(path)
"""

"""

function path_info = h5infopath(filename,hd5_group,disperror)
%Outputs the information structure at hd5_path in hdf5 file filename
%Faster than h5infotree since substructures of the group are ignored.

if (nargin==2)
    disperror=0;
end
% * if disperror argument (display error) omitted, hide errors by default

try
    fid = H5F.open(filename, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
catch therror
    if (disperror)
        disp(therror.message);
    end
    path_info=0;
    % File could not be opened
    return;
end

try
    gid = H5G.open(fid,hd5_group);
catch therror
    if (disperror)
        disp(therror.message);
    end
    path_info=-1;
    % File opened, but error reading path
    H5F.close(fid);
    return;
end

idx(1:5)=0; % creates array of five 0s
groups=[];
datasets=[];
datatypes=[];
links=[];
others=[];
idx_type = 'H5_INDEX_NAME';
order = 'H5_ITER_NATIVE';
lapl_id = 'H5P_DEFAULT';

info = H5G.get_info(gid);
for i=1:info.nlinks
    name = H5L.get_name_by_idx(fid,hd5_group,idx_type,order,i-1,lapl_id);
    statbuf = H5G.get_objinfo(gid, name, 0);
    switch (statbuf.type)
        case H5ML.get_constant_value('H5G_GROUP')
            idx(1)=idx(1)+1;
            groups{idx(1)}=name;
        case H5ML.get_constant_value('H5G_DATASET')
            idx(2)=idx(2)+1;
            datasets{idx(2)}=name;
        case H5ML.get_constant_value('H5G_TYPE')
            idx(3)=idx(3)+1;
            datatypes{idx(3)}=name;
        case H5ML.get_constant_value('H5G_LINK')
            idx(4)=idx(4)+1;
            links{idx(4)}=name;
        otherwise
            others{idx(5)}=name;
    end
end
H5G.close(gid);
H5F.close(fid);

path_info=struct('Filename', filename, 'Name', hd5_group, ...
    'Groups', char(groups), 'Datasets', char(datasets), ...
    'Datatypes', char(datatypes), 'Links', char(links), ...
    'Others', char(others));

"""
