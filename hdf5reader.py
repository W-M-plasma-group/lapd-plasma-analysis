import h5py


def open_hdf5(filename):  # Create an h5py file object
    return h5py.File(filename, 'r')


def item_at_path(file, subpath):  # Return the data structure at the given path
    return file[subpath]


def structures_at_path(file, subpath):  # Return a dictionary of all structures at given path
    curr = file[subpath]
    groups = []
    datasets = []
    other = []
    for item in curr.keys():
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
