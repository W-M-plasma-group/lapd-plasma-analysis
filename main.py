"""
The lapd-plasma-analysis repository was written by Leo Murphy based on code
written in MATLAB by Conor Perks (MIT) and using the PlasmaPy library.
Comments are added inline. A separate documentation page is not yet complete.
"""

# ONE PROBE PER PORT

from getIVsweep import *
from characterization import *
from diagnostics import *
from plots import *
from netCDFaccess import *
from interferometry import *
from neutrals import *
from experimental import *

hdf5_folder = "/Users/leo/lapd-data/November_2022/"                 # end this with a slash
langmuir_nc_folder = hdf5_folder + "lang_nc/"

# User global parameters                                         # From MATLAB code
core_radius = 26. * u.cm                                         # From MATLAB code
ion_type = 'He-4+'
bimaxwellian = False
smoothing_margin = 20                                            # Optimal values in range 0-25

# User file options
save_diagnostics = True  # Set save_diagnostics to True to save calculated diagnostic data to NetCDF files
interferometry_calibrate = False  # TODO make automatic

# TODO for user: change these to match your run!
# November 2022 configuration
vsweep_board_channel = (1, 1)
# """
langmuir_probes = np.array([(1, 2, 29, 2.20, 2. * u.mm ** 2),  # receptacle 1
                            # (1, 3, 35, 2.20, 2. * u.mm ** 2)   # receptacle 4
                            ], dtype=[('board', int),
                                      ('channel', int),
                                      ('port', int),
                                      ('resistance', float),
                                      ('area', u.Quantity)])
# """
# March 2022 configuration
"""
vsweep_board_channel = (1, 1)  # (1, 3)
langmuir_probes = np.array([(1, 2, 27, 1.25, 1. * u.mm ** 2),   # receptacle 1
                            (1, 3, 43, 2.10, 1. * u.mm ** 2)],  # receptacle 2
                           dtype=[('board', int), 
                                  ('channel', int), 
                                  ('port', int),
                                  ('resistance', float), 
                                  ('area', u.Quantity)])
# """
# April 2018 configuration
"""
vsweep_board_channel = (1, 3)
langmuir_probes = np.array([(1, 2, 25, 11., 1. * u.mm ** 2)],  # receptacle 1
                           dtype=[('board', int), 
                                  ('channel', int), 
                                  ('port', int),
                                  ('resistance', float), 
                                  ('area', u.Quantity)])
# """
# QUESTION: can we calibrate both Langmuir probes using an interferometry ratio depending only on one of them?
# Insert diagram of LAPD


def port_selector(ds):  # TODO allow multiple modified datasets to be returned
    # port_list = dataset.port  # use if switch to dataset.sel
    manual_attrs = ds.attrs  # TODO raise xarray issue about losing attrs even with xr.set_options(keep_attrs=True):
    ds_port_selected = ds.isel(port=0)  # - ds.isel(port=1)  # TODO user change for ex. delta-P-parallel
    return ds_port_selected.assign_attrs(manual_attrs)
    # use the "exec" function to prompt user to input desired transformation? Or ask for a linear transformation
    # Add a string attribute to the dataset to describe which port(s) comes from


if __name__ == "__main__":

    if not interferometry_calibrate:
        print("Interferometry calibration is OFF. "
              "Interferometry-calibrated electron density ('n_e_cal') is not available.")

    netcdf_folder = ensure_directory(langmuir_nc_folder)  # Create folder to save NetCDF files if not yet existing

    diagnostic_name_dict = {key: get_title(key)
                            for key in get_diagnostic_keys_units(langmuir_probes['area'][0], ion_type, bimaxwellian).keys()}
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
        # show_receptacles = True  # TODO elaborate on this. This prints out a list of probes and their receptacles
        for hdf5_path in hdf5_chosen_list:  # TODO improve loading bar for many datasets
            # TODO ALLOW INNER PEEKING into what going on in diagnostics

            print("\nOpening file", repr(hdf5_path), "...")

            """
            if show_receptacles:
                print("List of detected Compumotor receptacles and their respective ports and probes "
                      "(check this in main.py):")
                with lapd.File(hdf5_path) as f:
                    for probe in f.controls['6K Compumotor'].configs:
                        print(f"\t{f.controls['6K Compumotor'].configs[probe]['receptacle']}: "
                              f"Port {f.controls['6K Compumotor'].configs[probe]['probe']['port']}, "
                              f"{f.controls['6K Compumotor'].configs[probe]['probe']['probe name']}")
                show_receptacles = False
            """

            exp_params_dict = get_exp_params(hdf5_path)

            print("before ivsweep")
            bias, currents, positions, sample_sec, ports = get_isweep_vsweep(
                hdf5_path, vsweep_board_channel, langmuir_probes)
            print("got ivsweep")

            # PLOT STACKED CURRENTS
            """Sanity-check sample of sweep data"""
            # Recall dimensions of currents are p; l, s, r
            # """
            alpha = np.prod(bias.shape[:2]) ** -0.5
            num_frames = currents.shape[-1]
            color_map_loc = plt.cm.get_cmap("plasma")(np.linspace(0, 1, currents.shape[1]))
            for current_probe in currents[..., int(0.46 * num_frames):int(0.54 * num_frames):100]:
                for l in range(len(current_probe)):
                    for current_shot in current_probe[l]:
                        plt.plot(current_shot, color=color_map_loc[l], alpha=alpha)
                plt.show()
            # """
            # END PLOT STACKED CURRENTS

            # TODO loc_bounds is a debug option; otherwise, set to None
            loc_bounds = (30, 34)
            if loc_bounds is not None:
                positions = positions[min(loc_bounds):max(loc_bounds)]
            characteristics, ramp_times = characterize_sweep_array(bias, currents, smoothing_margin, sample_sec, loc_bounds)
            diagnostics_dataset = langmuir_diagnostics(characteristics, positions, ramp_times, ports, langmuir_probes['area'],
                                                       ion_type, bimaxwellian=bimaxwellian)

            # Detect beginning and end of steady state period
            steady_state_plateaus = detect_steady_state_ramps(diagnostics_dataset['n_e'], core_radius)

            # Perform interferometry calibration for electron density
            if interferometry_calibrate:
                calibrated_electron_density = interferometry_calibration(diagnostics_dataset['n_e'], hdf5_path,
                                                                         steady_state_plateaus, core_radius)
                diagnostics_dataset = diagnostics_dataset.assign({"n_e_cal": calibrated_electron_density})
            else:
                calibrated_electron_density = diagnostics_dataset['n_e']

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
    steady_state_plateaus_runs = [detect_steady_state_ramps(dataset['n_e'], core_radius) for dataset in datasets]
    print("Steady state period for each run:", steady_state_plateaus_runs)
    # """
    for plot_diagnostic in diagnostic_chosen_list:
        for i in range(len(datasets)):  # dataset in datasets:
            # try:
                # steady_state_plateaus = detect_steady_state_ramps(dataset['n_e'], core_radius)
            line_time_diagnostic_plot(port_selector(datasets[i]), plot_diagnostic, 'contour', steady_state_plateaus_runs[i])
            # except Exception as e:
            #     print(e)
    # """

    # PLOT radial profiles of diagnostic (steady state time average), in color corresponding to first attribute,
    #    and in plot position on multiplot corresponding to second attribute
    # """
    for plot_diagnostic in diagnostic_chosen_list:
        plot_line_diagnostic_by(datasets, plot_diagnostic, port_selector,
                                attribute=["Nominal discharge",
                                           "Nominal gas puff"], steady_state_by_runs=steady_state_plateaus_runs,
                                tolerance=1)  # TODO redefine tolerance to take advantage of multiple shots per plateau
        # TODO user select attribute(s) from menu
    # """

# Note: Not all MATLAB code has been transferred (e.g. neutrals, ExB)
