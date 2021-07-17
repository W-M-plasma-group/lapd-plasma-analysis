# Add comments
from astropy import visualization
import matplotlib.pyplot as plt


def plot_ne_te(diagnostics_xarray):
    # Return plots of ne and te; should make customizable what plots are desired?

    with visualization.quantity_support():
        # print(diagnostics_xarray)
        x_time_var_xarray = diagnostics_xarray.squeeze("y")
        # print(x_time_var_xarray)
        # x_time_var_xarray.sel(diagnostic='T_e').plot(robust=True)
        x_time_var_xarray.sel(diagnostic='T_e').plot.contourf(robust=True)
        plt.show()
