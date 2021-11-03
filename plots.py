# Add comments
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np


def radial_diagnostic_plot(diagnostics_dataset, diagnostic='T_e', plot='contour'):
    # TODO is this actually a linear diagnostic plot?
    # Return a plot of the specified type for the specified diagnostic in the dataset.

    if diagnostics_dataset.sizes['x'] > 1 and diagnostics_dataset.sizes['y'] > 1:
        raise ValueError("x and y dimensions have lengths", diagnostics_dataset.shape[:2],
                         "both greater than 1. A radial plot cannot be made. Areal plots are not yet supported.")
    radial_diagnostics_dataset = diagnostics_dataset.squeeze()

    if diagnostic not in diagnostics_dataset:
        raise ValueError("The input choice of diagnostic", diagnostic,
                         "is not a valid diagnostic. Valid diagnostics are:", diagnostics_dataset.keys())
    pos_time_diagnostic_xarray = radial_diagnostics_dataset[diagnostic]
    units = pos_time_diagnostic_xarray.attrs['units']

    allowed_plot_types = ('contour', 'surface')
    if plot not in allowed_plot_types:
        warn("The type of plot '" + str(plot) + "' is not in the supported plot type list " + str(allowed_plot_types) +
             ". Defaulting to contour plot.")
        plot = 'contour'

    # Determine if chosen diagnostic has one or multiple values in the quantity
    dimensions = len(pos_time_diagnostic_xarray.shape)
    if dimensions == 2:  # a diagnostic has a single value; for example, a non-bimaxwellian electron temperature
        # add a dummy dimension to allow iteration over the one variable for the diagnostic
        pos_time_diagnostic_xarray = pos_time_diagnostic_xarray.expand_dims(dim={'last': 1}, axis=-1)
    elif dimensions == 3:  # a diagnostic has multiple values; for example, a bimaxwellian electron temperature
        pass
    else:
        raise ValueError("The xarray of the chosen diagnostic has a number of dimensions", dimensions,
                         "that is invalid for surface plotting (must be two or three-dimensional)")

    # Plot the variable, creating separate plots for multi-variable diagnostics
    number_of_vars = pos_time_diagnostic_xarray.shape[-1]
    for var in range(number_of_vars):
        radial_crop_plot(pos_time_diagnostic_xarray[..., var], plot_type=plot)
    if number_of_vars == 2:  # two-variable diagnostic, for example bimaxwellian electron temperature
        # Note: this option is hardcoded specifically to plot T_e_hot - T_e_cold for bimaxwellian temperature data
        difference_data = pos_time_diagnostic_xarray[..., 1] - pos_time_diagnostic_xarray[..., 0]
        radial_crop_plot(difference_data.assign_attrs({"standard_name": diagnostic + " difference",
                                                       "units": units}), plot_type=plot)
    # TODO difference in bimaxwellian temperature not working correctly


def radial_crop_plot(pos_time_var_xarray, plot_type):
    if plot_type == "contour":
        pos_time_var_xarray.plot.contourf(x='time', y='x', robust=True)
        plt.show()
    elif plot_type == "surface":
        # TODO raise issue on xarray about surface plotting not handling np.nan properly in choosing colormap
        # color_min, color_max = np.nanpercentile(pos_time_var_xarray, 2), np.nanpercentile(pos_time_var_xarray, 98)
        # cropped_var_xarray.plot.surface(x='time', y='x', cmap='viridis', vmin=color_min, vmax=color_max)

        # crop outlier diagnostic values; make sure that this is acceptable data handling
        # note: only crops high values; cropping low and high would require xarray.ufuncs.logical_and, aka "both"
        q1, q3 = np.nanpercentile(pos_time_var_xarray, [25, 75])
        cropped_var_xarray = pos_time_var_xarray.where(pos_time_var_xarray <= q3 + 1.5 * (q3 - q1))
        color_min, color_max = np.nanmin(cropped_var_xarray), np.nanmax(cropped_var_xarray)

        cropped_var_xarray.plot.surface(x='time', y='x', cmap='viridis', vmin=color_min, vmax=color_max)
        plt.show()
