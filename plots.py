# Add comments
import warnings
from astropy import visualization
import matplotlib.pyplot as plt


def radial_plot(diagnostics_xarray, diagnostic='T_e', plot='contour'):
    # Return plots of ne and te; make plot and diagnostic customizable

    if diagnostics_xarray.shape[0] > 1 and diagnostics_xarray.shape[1] > 1:
        raise ValueError("x and y dimensions have lengths", diagnostics_xarray.shape[:2],
                         "both greater than 1. A radial plot cannot be made. Areal plots are not yet developed.")
    radial_diagnostics_xarray = diagnostics_xarray.squeeze()

    if diagnostic not in diagnostics_xarray.coords['diagnostic']:
        raise ValueError("The input choice of diagnostic", diagnostic,
                         "is not a valid diagnostic. Valid diagnostics are:", diagnostics_xarray.coords['diagnostic'])
    pos_time_var_xarray = radial_diagnostics_xarray.sel(diagnostic=diagnostic)

    with visualization.quantity_support():
        # x_time_var_xarray = diagnostics_xarray.squeeze("y")
        # x_time_var_xarray.sel(diagnostic='T_e').plot(robust=True)
        if plot == "contour":
            pos_time_var_xarray.plot.contourf(robust=True)
        elif plot == "surface":
            pos_time_var_xarray.plot.surface(robust=True)
        else:
            warnings.warn("The type of plot '" + str(plot) + "' is not supported. Defaulting to contour plot.")
            pos_time_var_xarray.plot.contourf(robust=True)
        plt.show()
