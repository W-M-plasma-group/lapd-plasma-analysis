"""
The lapd-plasma-analysis repository was written by Leo Murphy based on code
written in MATLAB by Conor Perks (MIT) and using the PlasmaPy and bapsflib libraries.
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
from preconfiguration import *

# hdf5_folder = "/Users/leomurphy/lapd-data/April_2018/"                      # end this with slash
# hdf5_folder = "/Users/leomurphy/lapd-data/November_2022/"                 # end this with slash
hdf5_folder = "/Users/leomurphy/lapd-data/March_2022/"                 # end this with slash
langmuir_nc_folder = hdf5_folder + "lang_nc/"

# User file options
save_diagnostics = True  # Set save_diagnostics to True to save calculated diagnostic data to NetCDF files
interferometry_calibrate = False  # TODO make automatic

# User global parameters                                         # From MATLAB code  # TODO move to preconfig?
ion_type = 'He-4+'
bimaxwellian = False

smoothing_margin = 20                                            # Optimal values in range 0-25
plot_tolerance = np.nan  # TODO user adjust plot_tolerance; np.nan = keep all data points; ~0.2-0.5 works well

# QUESTION: can we calibrate both Langmuir probes using an interferometry ratio depending only on one of them?
core_radius = 26. * u.cm                                         # From MATLAB code
# TODO Insert diagram of LAPD


def port_selector(ds):  # TODO allow multiple modified datasets to be returned
    # port_list = dataset.port  # use if switch to dataset.sel

    manual_attrs = ds.attrs  # TODO raise xarray issue about losing attrs even with xr.set_options(keep_attrs=True):
    ds_port_selected = ds.isel(port=0)  # - ds.isel(port=1)  # TODO user change for ex. delta-P-parallel
    return ds_port_selected.assign_attrs(manual_attrs)
    # ask for a linear transformation/matrix?
    # Add a string attribute to the dataset to describe which port(s) comes from


if __name__ == "__main__":

    if not interferometry_calibrate:
        print("Interferometry calibration is OFF. "
              "Interferometry-calibrated electron density ('n_e_cal') is not available.")

    netcdf_folder = ensure_directory(langmuir_nc_folder)  # Create folder to save NetCDF files if not yet existing

    # list possible diagnostics and their full names, e.g. "n_e" and "Electron density"
    diagnostic_name_dict = {key: get_title(key)
                            for key in get_diagnostic_keys_units(bimaxwellian=bimaxwellian).keys()}
    diagnostic_name_list = list(diagnostic_name_dict.values())

    # TODO move until after files loaded!
    print("The following diagnostics are available to plot: ")
    diagnostics_to_plot_ints = choose_multiple_list(diagnostic_name_list, "diagnostic")
    diagnostic_to_plot_list = [list(diagnostic_name_dict.keys())[choice] for choice in diagnostics_to_plot_ints]
    print("Diagnostics selected:", diagnostic_to_plot_list)

    print("The following NetCDF files were found in the NetCDF folder (specified in main.py): ")
    nc_paths = sorted(search_folder(netcdf_folder, 'nc', limit=52))
    nc_paths_to_open_ints = choose_multiple_list(nc_paths, "NetCDF file", null_action="perform diagnostics on HDF5 files")

    if len(nc_paths_to_open_ints) > 0:
        datasets = [xr.open_dataset(nc_paths[choice]) for choice in nc_paths_to_open_ints]
    else:
        print("The following HDF5 files were found in the HDF5 folder (specified in main.py): ")
        hdf5_paths = sorted(search_folder(hdf5_folder, "hdf5", limit=52))
        hdf5_chosen_ints = choose_multiple_list(hdf5_paths, "HDF5 file")
        hdf5_chosen_list = [hdf5_paths[choice] for choice in hdf5_chosen_ints]

        chara_view_mode = ""
        if len(hdf5_chosen_list) == 1:
            while chara_view_mode not in ["y", "n"]:
                chara_view_mode = input("Use characteristic plotting mode? (y/n) ").lower()
            chara_view_mode = (chara_view_mode == "y")

        datasets = []
        for hdf5_path in hdf5_chosen_list:  # TODO improve loading bar for many datasets

            print("\nOpening file", repr(hdf5_path), "...")

            exp_params_dict = get_exp_params(hdf5_path)  # list of experimental parameters
            config_id = get_config_id(exp_params_dict['Exp name'])
            vsweep_board_channel = get_vsweep_bc(config_id)
            langmuir_probes = get_probe_config(hdf5_path, config_id)

            bias, currents, positions, sample_sec, ports = get_isweep_vsweep(
                hdf5_path, vsweep_board_channel, langmuir_probes)  # get current and bias data from Langmuir probe
            characteristics, ramp_times = characterize_sweep_array(bias, currents, smoothing_margin, sample_sec)

            # TODO move to separate file
            if chara_view_mode:
                x = np.unique(positions[:, 0])
                y = np.unique(positions[:, 1])
                print(f"\nDimensions of plateaus array: {characteristics.shape[0]} probes, {len(x)} x-positions, "
                      f"{len(y)} y-positions, {characteristics.shape[2]} shots, and {characteristics.shape[-1]} ramps.")
                print(f"  * probe ports are {ports}",
                      f"  * x positions range from {min(x)} to {max(x)}",
                      f"  * y positions range from {min(y)} to {max(y)}",
                      f"  * shot indices range from 0 to {characteristics.shape[2]}",
                      f"  * ramp times range from {min(ramp_times):.2f} to {max(ramp_times):.2f}.", sep="\n")
                probe_x_y_ramp_to_plot = [0, 0, 0, 0, 0]
                variables_to_enter = ["probe", "x position", "y position", "shot", "ramp"]
                print("\nNote: Enter a non-integer below to terminate characteristics plotting mode.")
                while chara_view_mode:
                    for i in range(len(probe_x_y_ramp_to_plot)):
                        try:
                            index_given = int(input(f"Enter a zero-based index for {variables_to_enter[i]}: "))
                        except ValueError:
                            chara_view_mode = False
                            break
                        probe_x_y_ramp_to_plot[i] = index_given
                    if not chara_view_mode:
                        break
                    print()
                    loc_x, loc_y = x[probe_x_y_ramp_to_plot[1]], y[probe_x_y_ramp_to_plot[2]]
                    loc = (positions == [loc_x, loc_y]).all(axis=1).nonzero()[0][0]
                    indices_to_plot = (probe_x_y_ramp_to_plot[0], loc) + tuple(probe_x_y_ramp_to_plot[3:])
                    characteristics[indices_to_plot].plot()
                    """ while chara_view_mode not in ["s", "a"]:
                        chara_view_mode = input("(S)how current plot or (a)dd another Characteristic?").lower()
                    if chara_view_mode == "s": """
                    plt.title(f"Run: {exp_params_dict['Run name']}\n"
                              f"Port: {ports[probe_x_y_ramp_to_plot[0]]}, x: {loc_x}, y: {loc_y}, "
                              f"shot: {probe_x_y_ramp_to_plot[3]}, time: {ramp_times[probe_x_y_ramp_to_plot[4]]:.2f}")
                    plt.show()

            # characteristic = characteristic_arrays[p, l, r]
            # diagnostics_ds[key].loc[port, positions[l, 0], positions[l, 1], ramp_times[r]] = val

            diagnostics_dataset = langmuir_diagnostics(characteristics, positions, ramp_times, ports,
                                                       langmuir_probes['area'], ion_type, bimaxwellian=bimaxwellian)

            # Perform interferometry calibration for electron density
            if interferometry_calibrate:
                steady_state_plateaus = detect_steady_state_ramps(diagnostics_dataset['n_e'], core_radius)
                calibrated_electron_density = interferometry_calibration(diagnostics_dataset['n_e'], hdf5_path,
                                                                         steady_state_plateaus, core_radius)
                diagnostics_dataset = diagnostics_dataset.assign({"n_e_cal": calibrated_electron_density})
            else:
                calibrated_electron_density = diagnostics_dataset['n_e']

            # Find electron pressure
            pressure_unit = u.Pa
            electron_temperature = diagnostics_dataset['T_e_avg'] if bimaxwellian else diagnostics_dataset['T_e']
            pressure = (3 / 2) * electron_temperature * calibrated_electron_density * (1. * u.eV * u.m ** -3
                                                                                       ).to(pressure_unit)
            diagnostics_dataset = diagnostics_dataset.assign({'P_e': pressure}
                                                             ).assign_attrs({'units': str(pressure_unit)})

            # Assign experimental parameters to diagnostic data attributes
            diagnostics_dataset = diagnostics_dataset.assign_attrs(exp_params_dict)
            diagnostics_dataset = diagnostics_dataset.assign_attrs({"Interferometry calibrated":
                                                                    interferometry_calibrate})

            datasets.append(diagnostics_dataset)

            save_diagnostic_path = make_path(netcdf_folder, exp_params_dict['Run name'], "nc")
            if save_diagnostics:
                write_netcdf(diagnostics_dataset, save_diagnostic_path)

    steady_state_plateaus_runs = [detect_steady_state_ramps(dataset['n_e'], core_radius) for dataset in datasets]

    """Plot chosen diagnostics for each individual dataset"""
    # """
    for plot_diagnostic in diagnostic_to_plot_list:
        for i in range(len(datasets)):
            plot_line_diagnostic(port_selector(datasets[i]), plot_diagnostic, 'contour', steady_state_plateaus_runs[i],
                                 tolerance=plot_tolerance)
    # """

    """
    Plot radial profiles of diagnostic (steady-state time average), with color corresponding to first attribute
        and plot position on multiplot corresponding to second attribute
    """
    # """
    for plot_diagnostic in diagnostic_to_plot_list:
        multiplot_line_diagnostic(datasets, plot_diagnostic, port_selector,
                                  steady_state_plateaus_runs, tolerance=plot_tolerance)
    # """

# Note: Not all MATLAB code has been transferred (e.g. neutrals, ExB)
