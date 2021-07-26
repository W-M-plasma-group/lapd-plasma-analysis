import h5py

# Make sure to add code comments!


def open_hdf5(filename):
    return h5py.File(filename, 'r')


def item_at_path(file, subpath):
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
