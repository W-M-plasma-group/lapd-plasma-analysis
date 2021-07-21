# Add comments
import warnings
from astropy import visualization
import matplotlib.pyplot as plt


def radial_plot(diagnostics_xarray, diagnostic='T_e', plot='contour'):
    # Return plots of ne and te; make plot and diagnostic customizable

    if diagnostics_xarray.sizes['x'] > 1 and diagnostics_xarray.sizes['y'] > 1:
        raise ValueError("x and y dimensions have lengths", diagnostics_xarray.shape[:2],
                         "both greater than 1. A radial plot cannot be made. Areal plots are not yet supported.")
    radial_diagnostics_xarray = diagnostics_xarray.squeeze()

    if diagnostic not in diagnostics_xarray.coords['diagnostic']:
        raise ValueError("The input choice of diagnostic", diagnostic,
                         "is not a valid diagnostic. Valid diagnostics are:", diagnostics_xarray.coords['diagnostic'])
    pos_time_diagnostic_xarray = radial_diagnostics_xarray.sel(diagnostic=diagnostic)

    allowed_plot_types = ('contour', 'surface')
    if plot not in allowed_plot_types:
        warnings.warn("The type of plot '" + str(plot) + "' is not supported. Defaulting to contour plot.")
        plot = 'contour'

    """
    # Needed to detect bimaxwellian T_e?
    is_scalar_xarray = xr.apply_ufunc(lambda q: q.isscalar, pos_time_diagnostic_xarray, vectorize=True).any()
    if is_scalar_xarray:
        list_of_vars = [0]
    else:
        # list_of_vars = range(np.nanmin(np.vectorize(len)(pos_time_diagnostic_xarray)))
        list_of_vars = range(xr.apply_ufunc(len, pos_time_diagnostic_xarray, vectorize=True).min(skipna=True))
    """
    dimensions = len(pos_time_diagnostic_xarray.shape)
    if dimensions < 2:
        raise ValueError("Too few dimensions in xarray of chosen diagnostic")
    elif dimensions == 2:  # a diagnostic has a single value; for example, a non-bimaxwellian electron temperature
        slices = [...]
    else:                  # a diagnostic has multiple values; for example, a bimaxwellian electron temperature
        diagnostic_length = pos_time_diagnostic_xarray.shape[-1]
        slices = [(..., i) for i in range(diagnostic_length)]

    for cut in slices:
        pos_time_var_xarray = pos_time_diagnostic_xarray[cut]
        print(pos_time_var_xarray)
        with visualization.quantity_support():
            if plot == "contour":
                pos_time_var_xarray.plot.contourf(robust=True)
            elif plot == "surface":
                pos_time_var_xarray.plot.surface(robust=True)
            plt.show()
