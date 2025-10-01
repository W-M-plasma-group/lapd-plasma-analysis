"""
Provide messages to add to each saved dataset (WIP) # todo
"""

from lapd_plasma_analysis.langmuir.plots import get_title

# General metadata

general = """
General information

This object is a dataset containing data from one experiment at the Large Plasma Device (LAPD) at UCLA.
This dataset is in the NetCDF format and contains Mach probe data.
This data can be accessed through Python, for example, by using the xarray library's open_dataset function.
Diagnostic data was calculated from experiments using the online lapd-plasma-analysis library.
For more information, see the 'contents', 'structure', 'use', and 'source' elements 
of the 'attrs' dictionary attribute of this dataset.
"""


def get_supplemental_metadata(dataset):
    return {"general": general,
            "contents": get_contents_metadata(dataset),
            "structure": get_structure_metadata(dataset),
            "use": get_use_metadata(dataset),
            "source": get_source_metadata(dataset)}


# Contents metadata

contents = """
Contents information

This dataset is an instance of `xarray.Dataset`. It contains the following variable(s), referred to by
their Dataset variable (short) name followed by their descriptive (long) name:
    {variables_list_string} 
Visit https://docs.xarray.dev/en/stable/index.html for documentation and information on xarray Datasets and DataArrays.
(WIP)
"""


def get_contents_metadata(dataset):
    """

    Parameters
    ----------
    dataset : `xarray.Dataset`
        WIP

    Returns
    -------
    `str`

    """

    variable_short_names = [str(key) for key in dataset.keys()]
    variable_long_names = [get_title(variable) for variable in variable_short_names]
    variable_combined_names = [variable_short_names[i].ljust(20) + "  --  " + variable_long_names[i]
                               for i in range(len(variable_short_names))]
    variables_list_string = "\n\t".join(variable_combined_names)
    return contents.format(variables_list_string=variables_list_string,
                           )


# Structure metadata

structure = """
Structure information

Each individual variable in this dataset is stored as an individual `xarray.DataArray` object.
This dataset, and by extension each of its individual DataArrays, are {num_dimensions}-dimensional, 
with the following dimensions and sizes, ordered from outermost to innermost dimension: 
    {dimensions_dict_string}
Information on `xarray` Datasets and DataArrays can be found at the following links.
Quick overview:  https://docs.xarray.dev/en/stable/getting-started-guide/quick-overview.html
Terminology:     https://docs.xarray.dev/en/stable/user-guide/terminology.html
Data structures: https://docs.xarray.dev/en/stable/user-guide/data-structures.html
Indexing data:   https://docs.xarray.dev/en/stable/user-guide/indexing.html
"""


def get_structure_metadata(dataset):

    variables_list = list(dataset.keys())
    dimensions_dict = dict(dataset[variables_list[0]].sizes.items())
    dimensions_dict_string = "\n\t".join([f"{dim:9} --  {dimensions_dict[dim]}" for dim in dimensions_dict])
    return structure.format(num_dimensions=len(dimensions_dict),
                            dimensions_dict=dimensions_dict,
                            dimensions_dict_string=dimensions_dict_string
                            )


# Use metadata

use = """
Usage information

Below are some examples of Python code to access and display data from this dataset.

# Open the dataset
import xarray as xr
dset = xr.open_dataset("path_to_file/dataset_file_name.nc")  # change the path name for your device

# Access the DataArray with '{variables_list[0]}' data
da = dset['{variables_list[0]}']   # note the difference between 'dset' and 'da' 

# Access and print various metadata
print(dset)                 # Try these yourself
print(dset.general)    
print(dset.use)
print(dset.attrs.keys())

# Two ways to access the '{variables_list[0]}' DataArray entry that has index 0 in every dimension
da[{dataset_index_string}].values 
da.loc[{{{dataset_loc_index_string}}}].values

# Access the units of '{variables_list[0]}' data; convertible to string or AstroPy quantity
da.attrs['units']

# Plot some example data
import matplotlib.pyplot as plt
da_squeezed = da.squeeze()   # remove length-one dimensions
da_squeezed_2D = da_squeezed[{dataset_index_string_truncated}]  # 2D array at 0th pos of first (num_dims - 2) dimensions
da_squeezed_2D.plot.contourf()
plt.show()

# Access the first coordinate in the list of coordinates
dset.coords[list(dset.coords)[-1]]

"""


def get_use_metadata(dataset):

    # todo Add full name support for printing e.g. variables_list[0], such as "Electron temperature" instead of "T_e" ?
    dimensions_list = list(dataset.dims)
    num_squeezed_dimensions = len([dimension for dimension in dataset.dims if dataset.sizes[dimension] <= 1])
    dataset_index_string = ', '.join(['0' for _ in dimensions_list])
    dataset_loc_index_string = ', \n        '.join(
        [repr(dimension) + ': ' + repr(dataset.coords[dimension][0].item())
         for dimension in dimensions_list])

    return use.format(variables_list=list(dataset.keys()),
                      dimensions_list=dimensions_list,
                      dataset_index_string=dataset_index_string,
                      dataset_loc_index_string=dataset_loc_index_string,
                      dataset_index_string_truncated=dataset_index_string[:-3 * (2 + num_squeezed_dimensions)],
                      )


# Source metadata

source = """
Source information

Mach number values were calculated using lapd-plasma-analysis.
Documentation for the lapd-plasma-analysis library is available online at this link: 
    https://lapd-plasma-analysis.readthedocs.io/en/index.html
Further information on Mach number calculations is not yet complete. (WIP)
"""

# TODO most important part - how was this data calculated?


def get_source_metadata(dataset):
    return source.format()
