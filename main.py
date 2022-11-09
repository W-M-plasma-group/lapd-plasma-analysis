"""
The lapd-plasma-analysis repository was written by Leo Murphy based on code
written in MATLAB by Conor Perks (MIT) and using the PlasmaPy library.
Comments are added inline. A separate documentation page is not yet complete.
"""

from getIVsweep import *
from characterization import *
from diagnostics import *
from plots import *
from netCDFaccess import *
from interferometry import *
from neutrals import *
from experimental import *

hdf5_folder = "/Users/leo/lapd-data/March_2022/"
langmuir_nc_folder = hdf5_folder + "lang_nc/"

# User global parameters
probe_area = 2. * u.mm ** 2                                      # From MATLAB code
core_radius = 26. * u.cm                                         # From MATLAB code
ion_type = 'He-4+'
bimaxwellian = False
smoothing_margin = 20                                            # Optimal values in range 0-25
# steady_state_plateaus = (6, 13) if True else (5, 11)          # TODO automatically detect!!

# User file options
save_diagnostics = True  # Set save_diagnostics to True to save calculated diagnostic data to NetCDF files
vsweep_board_channel = (1, 1)  # (1, 3)
isweep_boards_channels = [(1, 2), (1, 3)]  # tuple or list of tuples; board/channel for interferometer isweep data first  (1, 2)
port_resistances = {29: 2.20, 35: 2.20}  # TODO hardcoded; change to 11 if 2018, get from metadata! {25: 11}
isweep_receptacles = {2: 1, 3: 2}  # TODO hardcoded  {2: 1}
# QUESTION: can we calibrate both Langmuir probes using an interferometry ratio depending only on one of them?
# NOTE: Port 27 is near middle and near interferometer


def port_selector(ds):  # TODO allow multiple modified datasets to be returned
    # port_list = dataset.port  # use if switch to dataset.sel
    manual_attrs = ds.attrs  # TODO raise xarray issue about losing attrs even with xr.set_options(keep_attrs=True):
    ds_port_selected = ds.isel(port=0)  # - ds.isel(port=1)  # TODO user change for ex. delta-P-parallel
    return ds_port_selected.assign_attrs(manual_attrs)
    # use the "exec" function to prompt user to input desired transformation? Or ask for a linear transformation
    # Add a string attribute to the dataset to describe which port(s) comes from


if __name__ == "__main__":

    netcdf_folder = ensure_directory(langmuir_nc_folder)  # Create folder to save NetCDF files if not yet existing

    diagnostic_name_dict = {key: get_title(key)
                            for key in get_diagnostic_keys_units(probe_area, ion_type, bimaxwellian).keys()}
    diagnostic_name_list = list(diagnostic_name_dict.values())

    print("The following diagnostics are available to plot: ")
    diagnostic_chosen_ints = choose_multiple_list(diagnostic_name_list, "diagnostic")
    diagnostic_chosen_list = [list(diagnostic_name_dict.keys())[choice] for choice in diagnostic_chosen_ints]
    print("Diagnostics selected:", diagnostic_chosen_list)

    print("The following NetCDF files were found in the NetCDF folder (specified in main.py): ")
    nc_paths = sorted(search_folder(netcdf_folder, 'nc', limit=26))
    nc_chosen_ints = choose_multiple_list(nc_paths, "NetCDF file", null_action="perform diagnostics on HDF5 files")
    if len(nc_chosen_ints) > 0:
        datasets = [xr.open_dataset(nc_paths[choice]) for choice in nc_chosen_ints]
    else:
        print("The following HDF5 files were found in the HDF5 folder (specified in main.py): ")
        hdf5_paths = sorted(search_folder(hdf5_folder, "hdf5", limit=26))
        hdf5_chosen_ints = choose_multiple_list(hdf5_paths, "HDF5 file")
        hdf5_chosen_list = [hdf5_paths[choice] for choice in hdf5_chosen_ints]

        datasets = []
        for hdf5_path in hdf5_chosen_list:  # TODO improve loading bar for many datasets

            print("\nOpening file", repr(hdf5_path), "...")

            exp_params_dict = get_exp_params(hdf5_path)
            bias, currents, positions, sample_sec, ports = get_isweep_vsweep(
                hdf5_path, vsweep_board_channel, isweep_boards_channels, isweep_receptacles, port_resistances)

            characteristics, ramp_times = characterize_sweep_array(bias, currents, smoothing_margin, sample_sec)
            diagnostics_dataset = langmuir_diagnostics(characteristics, positions, ramp_times, ports, probe_area,
                                                       ion_type, bimaxwellian=bimaxwellian)

            # Detect beginning and end of steady state period
            steady_state_plateaus = detect_steady_state_ramps(diagnostics_dataset['n_e'], core_radius)

            # Perform interferometry calibration for electron density
            calibrated_electron_density = interferometry_calibration(diagnostics_dataset['n_e'], hdf5_path,
                                                                     steady_state_plateaus, core_radius)
            diagnostics_dataset = diagnostics_dataset.assign({"n_e_cal": calibrated_electron_density})

            # Find electron pressure
            electron_temperature = diagnostics_dataset['T_e_avg'] if bimaxwellian else diagnostics_dataset['T_e']
            pressure = (3 / 2) * electron_temperature * calibrated_electron_density * (1. * u.eV * u.m ** -3).to(u.Pa)
            diagnostics_dataset = diagnostics_dataset.assign({'P_e': pressure})

            # Assign experimental parameters to diagnostic data attributes
            diagnostics_dataset = diagnostics_dataset.assign_attrs(exp_params_dict)

            datasets.append(diagnostics_dataset)

            save_diagnostic_path = make_path(netcdf_folder, exp_params_dict['Run name'], "nc")
            if save_diagnostics:
                write_netcdf(diagnostics_dataset, save_diagnostic_path)

    # Plot chosen diagnostics for each individual dataset
    # """
    for plot_diagnostic in diagnostic_chosen_list:
        for dataset in datasets:
            try:
                steady_state_plateaus = detect_steady_state_ramps(dataset['n_e'], core_radius)
                line_time_diagnostic_plot(port_selector(dataset), plot_diagnostic, 'contour', steady_state_plateaus)
            except Exception as e:
                print(e)
    # """

    # PLOT radial profiles of diagnostic (steady state time average), in color corresponding to first attribute,
    #    and in plot position on multiplot corresponding to second attribute
    """
    for plot_diagnostic in diagnostic_chosen_list:
        plot_line_diagnostic_by(datasets, plot_diagnostic, port_selector,
                                attribute=["Nominal discharge",
                                           "Nominal gas puff"], steady_state=steady_state_plateaus)
        # TODO user select attribute(s) from menu
    """

# Note: Not all MATLAB code has been transferred (e.g. neutrals, ExB)
