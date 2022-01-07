# Add comments
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np


def linear_diagnostic_plot(diagnostics_dataset, diagnostic='T_e', plot='contour'):
    # Plots the given diagnostic(s) from the dataset in the given style

    diagnostic_list = np.atleast_1d(diagnostic)

    if diagnostics_dataset.sizes['x'] > 1 and diagnostics_dataset.sizes['y'] > 1:
        raise ValueError("x and y dimensions have lengths " + str(diagnostics_dataset.shape[:2]) +
                         " both greater than 1. A linear plot cannot be made. Areal plots are not yet supported.")
    linear_diagnostics_dataset = diagnostics_dataset.squeeze()

    if np.any([choice not in diagnostics_dataset for choice in diagnostic_list]):
        # print([choice not in diagnostics_dataset.keys() for choice in diagnostic_list])
        raise ValueError("The input choice(s) of diagnostic " + repr(diagnostic_list) +
                         " is not a valid diagnostic. Valid diagnostics are:" + repr(diagnostics_dataset.keys()))

    allowed_plot_types = {'contour', 'surface'}
    if plot not in allowed_plot_types:
        warn("The type of plot " + repr(plot) + " is not in the supported plot type list " + repr(allowed_plot_types) +
             ". Defaulting to contour plot.")
        plot = 'contour'

    for choice in diagnostic_list:
        diagnostic_xarray = linear_diagnostics_dataset[choice]
        # units = diagnostic_xarray.attrs['units']
        linear_crop_plot(diagnostic_xarray, plot_type=plot, diagnostic_name=choice)

        # check number of dimensions for validity?


def linear_crop_plot(pos_time_var_xarray, plot_type, diagnostic_name):

    full_names = {'V_P': "Plasma potential",
                  'V_F': "Floating potential",
                  'I_es': "Electron saturation current",
                  'I_is': "Ion saturation current",
                  'n_e': "Electron density",
                  'n_i': "Ion density",
                  'T_e': "Electron temperature",
                  'n_i_OML': "Ion density (OML)",
                  'hot_fraction': "Hot electron fraction",
                  'T_e_cold': "Cold electron temperature",
                  'T_e_hot': "Hot electron temperature",
                  'T_e_avg': "Average bimaxwellian electron temperature"}

    if plot_type == "contour":
        pos_time_var_xarray.plot.contourf(x='time', y='x', robust=True)
    elif plot_type == "surface":
        # TODO raise issue on xarray about surface plotting not handling np.nan properly in choosing colormap
        # crop outlier diagnostic values; make sure that this is acceptable data handling
        # note: only crops high values; cropping low and high would require numpy.logical_and, aka "both"
        q1, q3 = np.nanpercentile(pos_time_var_xarray, [25, 75])
        cropped_var_xarray = pos_time_var_xarray.where(pos_time_var_xarray <= q3 + 1.5 * (q3 - q1))
        color_min, color_max = np.nanmin(cropped_var_xarray), np.nanmax(cropped_var_xarray)
        cropped_var_xarray.plot.surface(x='time', y='x', cmap='viridis', vmin=color_min, vmax=color_max)

    if diagnostic_name in full_names:
        plt.title("Linear plot: " + full_names[diagnostic_name])
    plt.show()
