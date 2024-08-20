"""
Provide messages to add to each saved dataset (WIP) # todo
"""

# Info metadata

info = """
This is one NetCDF dataset containing Langmuir diagnostic plasma data from the Large Plasma Device (LAPD) at UCLA.
Diagnostic data was calculated from experiments using the online lapd-plasma-analysis library.
This data can be accessed through Python, for example, by using the xarray library's open_dataset function.
For more information, find the 'contents', 'structure', 'use', and 'source' elements 
of the 'attrs' dictionary attribute of this dataset. 
"""


def get_supplemental_metadata(dataset):
    return {"info": info,
            "contents": get_contents_metadata(dataset),
            "structure": get_structure_metadata(dataset),
            "use": get_use_metadata(dataset),
            "source": get_source_metadata(dataset)}


# Contents metadata

contents = """
This dataset is an instance of `xarray.Dataset`. It contains the following variables: 
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

    # TODO link to the xarray website (in multiple metadatas?)

    variables_list = list(dataset.keys())
    variables_list_string = "\n\t".join(variables_list)
    return contents.format(variables_list_string=variables_list_string,
                           )


# Structure metadata

structure = """
Each individual variable in this dataset is stored as an individual `xarray.DataArray` object.
This dataset, and by extension each of its individual DataArrays, are {num_dimensions}-dimensional, 
with the following dimensions, from outermost to innermost: 
    {dimensions_list}
Information on `xarray` Datasets and DataArrays can be found at the following links.
Quick overview:  https://docs.xarray.dev/en/stable/getting-started-guide/quick-overview.html
Terminology:     https://docs.xarray.dev/en/stable/user-guide/terminology.html
Data structures: https://docs.xarray.dev/en/stable/user-guide/data-structures.html
Indexing data:   https://docs.xarray.dev/en/stable/user-guide/indexing.html
"""


def get_structure_metadata(dataset):

    dimensions_list = list(dataset.dims)
    dimensions_list_string = "\n\t".join(dimensions_list)
    return structure.format(num_dimensions=len(dimensions_list),
                            dimensions_list=dimensions_list,
                            dimensions_list_string=dimensions_list_string
                            )


# Use metadata

use = """
Below are some examples of Python code to access and display data from this dataset.

# Open the dataset
import xarray as xr
ds = xr.open_dataset("path_to_file/dataset_file_name.nc")  # change the path name for your device

# Access the DataArray with '{variables_list[0]}' data
da = ds['{variables_list[0]}']   # note the difference between 'ds' and 'da' 

# Access and print various metadata
ds.attrs['info']  # try printing these
ds.attrs['use']
ds.attrs.keys()   # to learn what is there     

# Two ways to access the '{variables_list[0]}' DataArray entry that has index 0 in every dimension
da[{dataset_index_string}] 
da.loc[{{{dataset_loc_index_string}}}]

# Access the units of '{variables_list[0]}' data; convertible to string or AstroPy quantity
da.attrs['units']

# Plot some example data
import matplotlib.pyplot as plt
da_squeezed = da.squeeze()   # remove length-one dimensions
da_squeezed_2D = da_squeezed[{dataset_index_string_truncated}]  # 2D array at 0th pos of first (num_dims - 2) dimensions
da_squeezed_2D.plot.contourf()
plt.show()

# Access the first coordinate in the list of coordinates
ds.coords[list(ds.coords)[-1]]

"""


def get_use_metadata(dataset):

    # todo Add full name support for printing e.g. variables_list[0], such as "Electron temperature" instead of "T_e" ?
    dimensions_list = list(dataset.dims)
    dataset_index_string = ', '.join(['0' for _ in dimensions_list])
    dataset_loc_index_string = ', \n        '.join(
        [repr(dimension) + ': ' + str(dataset.coords[dimension][0].data)
         for dimension in dimensions_list])

    return use.format(variables_list=list(dataset.keys()),
                      dimensions_list=dimensions_list,
                      dataset_index_string=dataset_index_string,
                      dataset_loc_index_string=dataset_loc_index_string,
                      dataset_index_string_truncated=dataset_index_string[:-6],
                      )


# Source metadata

source = """
Documentation for the lapd-plasma-analysis library is available online at this link: 
https://lapd-plasma-analysis.readthedocs.io/en/index.html

Further information on lapd-plasma-analysis is not yet complete.
"""


def get_source_metadata(dataset):
    return source.format()
