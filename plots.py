# Add comments
import warnings
from astropy import visualization


def radial_plot(diagnostics_xarray, diagnostic='T_e', plot='contour'):
    # Return a plot of the specified type for the specified diagnostic.

    if diagnostics_xarray.sizes['x'] > 1 and diagnostics_xarray.sizes['y'] > 1:
        raise ValueError("x and y dimensions have lengths", diagnostics_xarray.shape[:2],
                         "both greater than 1. A radial plot cannot be made. Areal plots are not yet supported.")
    radial_diagnostics_xarray = diagnostics_xarray.squeeze()

    if diagnostic not in diagnostics_xarray:
        raise ValueError("The input choice of diagnostic", diagnostic,
                         "is not a valid diagnostic. Valid diagnostics are:", diagnostics_xarray.keys())
    pos_time_diagnostic_xarray = radial_diagnostics_xarray[diagnostic]

    allowed_plot_types = ('contour', 'surface')
    if plot not in allowed_plot_types:
        warnings.warn("The type of plot '" + str(plot) + "' is not supported. Defaulting to contour plot.")
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
    for var in range(pos_time_diagnostic_xarray.shape[-1]):
        pos_time_var_xarray = pos_time_diagnostic_xarray[..., var]
        # print(pos_time_var_xarray)
        with visualization.quantity_support():
            if plot == "contour":
                pos_time_var_xarray.plot.contourf(x='time', y='x', robust=True)
            elif plot == "surface":
                pos_time_var_xarray.plot.surface(x='time', y='x', robust=True)
