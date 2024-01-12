import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from plasmapy.diagnostics.langmuir import swept_probe_analysis, reduce_bimaxwellian_temperature, Characteristic


plt.rcParams["figure.dpi"] = 160
core_radius = 26. * u.cm


def value_safe(quantity_or_scalar):  # Get value of quantity or scalar, depending on type

    try:
        val = quantity_or_scalar.value  # input is a quantity with dimension and value
    except AttributeError:
        val = quantity_or_scalar  # input is a dimensionless scalar with no value
    return val


def unit_safe(quantity_or_scalar):  # Get unit of quantity or scalar, if possible

    try:
        unit = quantity_or_scalar.unit
    except AttributeError:
        unit = None  # The input data is dimensionless
    return unit


def unpack_bimaxwellian(diagnostics):
    t_e_cold, t_e_hot = diagnostics['T_e']
    hot_frac = diagnostics['hot_fraction']
    t_e_avg = reduce_bimaxwellian_temperature([t_e_cold, t_e_hot], hot_frac)
    return diagnostics.update({'T_e_cold': t_e_cold,
                               'T_e_hot': t_e_hot,
                               'T_e_avg': t_e_avg}
                              ).pop('T_e')


def get_diagnostic_keys_units(probe_area=1.*u.mm**2, ion_type="He-4+", bimaxwellian=False):
    # Perform diagnostic on some sample data to get all diagnostic names and units as dictionary of strings

    bias = np.arange(-20, 20, 1) * u.V
    current = ((bias.value / 100 + 0.2) ** 2 - 0.01) * u.A
    chara = Characteristic(bias, current)
    diagnostics = swept_probe_analysis(chara, probe_area, ion_type, bimaxwellian)
    if bimaxwellian:
        diagnostics = unpack_bimaxwellian(diagnostics)
    keys_units = {key: str(unit_safe(value)) for key, value in diagnostics.items()}
    keys_units.update({"n_e_cal": str(u.m ** -3)})
    keys_units.update({"P_e": str(u.Pa)})
    return keys_units


def port_selector(ds, vector):  # TODO should separate diagnostics_main and plot_main anyway!
    r"""
    Select a port or linear combination of ports from a diagnostic dataset.
    For example, on a dataset with two probes at different ports,
    [1,  0] would return the data at the first (lowest-port-number) probe
    [1, -1] would return the parallel difference (low port number minus high port number)
    # [[1, 0], [1, -1]] would return a list containing both of the above [NOT YET IMPLEMENTED]
    :param ds: The dataset to select from
    :param vector: The linear combination of ports to compute
    :return: Dataset containing data from the selected port or combination of ports
    """

    # use "port_list = dataset.port" if switch to dataset.sel
    manual_attrs = ds.attrs  # TODO raise xarray issue about losing attrs even with xr.set_options(keep_attrs=True):
    manual_sub_attrs = {key: ds[key].attrs for key in ds}
    ds_port_selected = 0 * ds.isel(port=0)
    for i in range(ds.sizes['port']):
        ds_port_selected += vector[i] * ds.isel(port=i)
    for key in ds:
        ds_port_selected[key] = ds_port_selected[key].assign_attrs(manual_sub_attrs[key])
    return ds_port_selected.assign_attrs(manual_attrs)
    # ask user for a linear transformation/matrix?
    # Add a string attribute to the dataset to describe which port(s) comes from
