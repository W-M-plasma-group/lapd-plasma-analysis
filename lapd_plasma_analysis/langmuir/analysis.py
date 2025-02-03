from lapd_plasma_analysis.file_access import *
from lapd_plasma_analysis.experimental import get_exp_params

from lapd_plasma_analysis.langmuir.helper import *
from lapd_plasma_analysis.langmuir.configurations import *
from lapd_plasma_analysis.langmuir.getIVsweep import get_sweep_voltage, get_sweep_current, get_shot_positions
from lapd_plasma_analysis.langmuir.characterization import make_characteristic_array, isolate_ramps
from lapd_plasma_analysis.langmuir.preview import preview_raw_sweep, preview_characteristics
from lapd_plasma_analysis.langmuir.diagnostics import (langmuir_diagnostics, detect_steady_state_times, get_pressure,
                                                       get_electron_ion_collision_frequencies)
from lapd_plasma_analysis.langmuir.neutrals import get_neutral_density
from lapd_plasma_analysis.langmuir.interferometry import interferometry_calibration
from lapd_plasma_analysis.langmuir.plots import get_title
from lapd_plasma_analysis.langmuir.metadata_for_dataset import get_supplemental_metadata


def get_langmuir_datasets(langmuir_nc_folder, hdf5_folder, interferometry_folder, interferometry_mode,
                          core_radius, bimaxwellian, plot_save_directory):
    """
    Retrieve datasets of Langmuir probe diagnostic data from LAPD experiments.
    Also returns the steady-state time period of each dataset
    and, if applicable, a list of the HDF5 file paths chosen by the user.

    Parameters
    ----------
    langmuir_nc_folder : `str`
        Path to folder storing Langmuir probe diagnostic data NetCDF files.

    hdf5_folder : `str`
        Path to folder storing LAPD experiment HDF5 files.

    interferometry_folder : `str`
        Path to folder storing interferometry data. If interferometry data is stored in the HDF5 files,
        then this should be equal to `hdf5_folder`.

    interferometry_mode : {'skip', 'append', 'overwrite'}
        Mode for handling interferometry data.
            - 'skip' skips all interferometry calibration calculations.
            - 'append' performs interferometry calibration only on datasets that do not already have calibrated densities.
            - 'overwrite' recalculates calibrated densities for all datasets.

    core_radius : `astropy.units.Quantity`
        Radius of core region, or region with high, stable densities. Used to calculate steady-state period
        and in interferometry calibration.

    bimaxwellian : `bool`
        Specifies if plasmapy Langmuir diagnostics functions should assume a bimaxwellian plasma
        (i.e. mixed hot and cold electrons).

    plot_save_directory : `str`
        Path to directory for saving plots, including sweep voltage, current, and curve preview plots.

    Returns
    -------
    datasets : list of `xarray.Dataset`
        List of Langmuir diagnostic datasets.

    steady_state_times_runs : list of `astropy.unit.Quantity`
        List of ordered pairs (tuples?) of `astropy.unit.Quantity` objects indicating the beginning and end time
        of the steady-state period.

    hdf5_paths
        List of paths to HDF5 files opened. Can be `None` if NetCDF files used.

    See Also
    --------
    load_datasets : creates `xarray.Dataset` datasets used in this function

    """

    # Create folder to save NetCDF files if not yet existing
    netcdf_folder = ensure_directory(langmuir_nc_folder)

    # Ask user to choose either NetCDF files or HDF5 files, then create datasets from them
    datasets, hdf5_paths = load_datasets(hdf5_folder, netcdf_folder, bimaxwellian, plot_save_directory)
    if datasets is None:
        return None, None, None

    steady_state_times_runs = [detect_steady_state_times(dataset, core_radius) for dataset in datasets]

    # Calibrate electron densities using interferometry data, depending on interferometry mode
    for density_diagnostic in ('n_e', 'n_i', 'n_i_OML'):
        datasets = interferometry_calibrate_datasets(datasets, density_diagnostic, interferometry_folder,
                                                     interferometry_mode, core_radius, steady_state_times_runs)

    # Calculate pressures
    for i in range(len(datasets)):
        # TODO check behavior with advisor: use avg or cold T_e for bimaxwellian plasmas?
        electron_temperature = datasets[i]['T_e_avg'] if 'T_e_avg' in datasets[i] else datasets[i]['T_e']
        temperature = electron_temperature + ion_temperature.to(u.eV).value
        datasets[i] = datasets[i].assign({'P_e': get_pressure(datasets[i]['n_e'], electron_temperature),
                                          'P_ei': get_pressure(datasets[i]['n_e'], temperature),
                                          'P_e_from_n_i_OML': get_pressure(datasets[i]['n_i_OML'], electron_temperature),
                                          'P_ei_from_n_i_OML': get_pressure(datasets[i]['n_i_OML'], temperature)})
        if not np.isnan(datasets[i]['n_e_cal']).all():
            datasets[i] = datasets[i].assign({"P_e_cal": get_pressure(datasets[i]['n_e_cal'], electron_temperature),
                                              "P_ei_cal": get_pressure(datasets[i]['n_e_cal'], temperature)})

    # Calculate collision frequency
    for i in range(len(datasets)):
        electron_ion_collision_frequencies_da = datasets[i]['n_e'].copy().rename("nu_ei")   # DataArray
        electron_ion_collision_frequencies = get_electron_ion_collision_frequencies(        # Quantity array
            datasets[i],
            ion_type=get_ion(datasets[i].attrs['Run name']))
        electron_ion_collision_frequencies_da[...] = electron_ion_collision_frequencies
        electron_ion_collision_frequencies_da = electron_ion_collision_frequencies_da.assign_attrs(
            {"units": str(electron_ion_collision_frequencies.unit)})
        datasets[i] = datasets[i].assign({'nu_ei': electron_ion_collision_frequencies_da})

    # Calculate neutral density
    for i in range(len(datasets)):
        # change units of saved fill pressure to Pa?
        neutral_density = get_neutral_density(value_safe(datasets[i].attrs['Fill pressure']) * u.Torr)
        datasets[i] = datasets[i].assign_attrs({"Neutral density": neutral_density})

    # Final save diagnostics datasets to folder (after earlier save point in load_datasets function)
    save_datasets_nc(datasets, netcdf_folder, "lang_", bimaxwellian)

    return datasets, steady_state_times_runs, hdf5_paths


def print_user_file_choices(hdf5_folder, lang_nc_folder, interferometry_folder, interferometry_mode, isweep_choices):
    r"""Displays file directory, interferometry mode, and I-sweep choices.

    Prompts the user to press any key to proceed.

    Parameters
    ----------
    hdf5_folder : `str`
        The directory in which the HDF5 files are stored.

    lang_nc_folder : `str`
        A directory at which to deposit generated NetCDF files and/or a directory
        already containing NetCDF files.

    interferometry_folder : `str`
        Directory at which to store interferometry data.

    interferometry_mode : `str`
        Options are `'append'`, `'skip'`, and `'overwrite'`. Controls how new interferometry
        data will be created and stored.
        `'append'`: Collects and stores interferometry data only for HDF5/NetCDF files
        which have not already been analyzed.
        `'skip'`: Skips collecting and storing interferometry data from all selected
        HDF5/NetCDF files.
        `'overwrite'`: Collects and stores interferometry data for all selected files
        regardless of prior analyzation status.

    isweep_choices : `list`
        # wip TODO
        A vector or list of vectors (e.g. `[1, 0]` or `[[1, 0], [1, -1]]` which
        specifies how the current data from different probe faces are combined.

    """
    interferometry_mode_actions = ({"skip": "skipped",
                                    "append": "added to uncalibrated datasets only",
                                    "overwrite": "recalculated for all datasets"})
    print(f"Interferometry mode is {repr(interferometry_mode)}. "
          f"Calibrated density data will be {interferometry_mode_actions[interferometry_mode]}.")

    print("Current HDF5 directory path:           \t",   repr(hdf5_folder),
          "\nCurrent NetCDF directory path:         \t", repr(lang_nc_folder),
          "\nCurrent interferometry directory path: \t", repr(interferometry_folder),
          "\nLinear combinations of isweep sources: \t", repr(isweep_choices),
          "\nThese can be changed in main.py.")
    input("Enter any key to continue: ")


def load_datasets(hdf5_folder, lang_nc_folder, bimaxwellian, plot_save_directory):
    r"""

    Parameters
    ----------
    hdf5_folder : `str`
        The directory in which the HDF5 files are stored.

    lang_nc_folder : `str`
        A directory at which to deposit generated NetCDF files and/or a directory
        already containing NetCDF files.

    bimaxwellian : `bool`
        Specifies whether the plasma should be considered to consist of both hot and
        cold ions (`True`) or not (`False`).

    plot_save_directory : `str`
        Directory in which to save any plots. If `' '`, then no plots are saved.

    Returns
    -------
    `tuple`
        (`datasets`, `hdf5_chosen_list`) OR `None`, if user skips to Mach calculation
        `datasets`: List of `xarray.Dataset` objects containing Langmuir diagnostic data.
        `hdf5_chosen_list`: List of `str`, the paths to the selected HDF5 files.
        Can also be `None` if no HDF5 files were chosen.

    """

    print("\nThe following Langmuir NetCDF files were found in the NetCDF folder (specified in main.py): ")
    nc_paths = sorted(search_folder(lang_nc_folder, 'nc', limit=52))
    nc_paths_chosen_ints = choose_multiple_from_list(nc_paths, "Langmuir data NetCDF file",
                                                     null_action="perform diagnostics on HDF5 files")

    if len(nc_paths_chosen_ints) > 0:
        hdf5_chosen_list = None
        datasets = [xr.open_dataset(nc_paths[choice]) for choice in nc_paths_chosen_ints]
    else:
        print("\nThe following HDF5 files were found in the HDF5 folder (specified in main.py): ")
        hdf5_paths = sorted(search_folder(hdf5_folder, "hdf5", limit=52))
        hdf5_chosen_ints = choose_multiple_from_list(hdf5_paths, "HDF5 file",
                                                     null_action="skip to Mach number calculations only")

        if len(hdf5_chosen_ints) == 0:
            return None

        hdf5_chosen_list = [hdf5_paths[choice] for choice in hdf5_chosen_ints]

        sweep_view_mode = (len(hdf5_chosen_list) == 1) and ask_yes_or_no("Use raw sweep preview mode? (y/n) ")
        chara_view_mode = (len(hdf5_chosen_list) == 1) and ask_yes_or_no("Use characteristic preview mode? (y/n) ")

        datasets = []
        print("(Note: plasmapy.langmuir.diagnostics pending deprecation FutureWarnings are suppressed)")
        for hdf5_path in hdf5_chosen_list:

            print(f"\nOpening file {repr(hdf5_path)} ...")

            exp_params_dict = get_exp_params(hdf5_path)  # list of experimental parameters

            ion_type = get_ion(exp_params_dict['Run name'])
            exp_params_dict = exp_params_dict | {"Ion type": ion_type}
            config_id = get_config_id(exp_params_dict['Exp name'])
            vsweep_board_channel = get_vsweep_bc(config_id)

            langmuir_configs = get_langmuir_config(hdf5_path, config_id)
            voltage_gain = get_voltage_gain(config_id)
            orientation = get_orientation(config_id)

            # todo revise get current and bias data from Langmuir probe and store in... [???]
            bias, dt = get_sweep_voltage(hdf5_path, vsweep_board_channel, voltage_gain)
            ramp_bounds = isolate_ramps(bias)
            ramp_times = ramp_bounds[:, 1] * dt.to(u.ms)
            # todo NOTE: MATLAB code stores peak voltage time (end of plateaus), then only uses plateau times for very first position
            #  This uses the time of the peak voltage for the average of all shots ("top of the average ramp")

            # This for loop extracts sweep data and creates Characteristic objects
            characteristic_arrays = []
            position_arrays = []
            print(f"Creating characteristics ...")
            for i in range(len(langmuir_configs)):
                current, motor_data = get_sweep_current(hdf5_path, langmuir_configs[i], orientation)

                # ensure "hardcoded" ports listed in configurations.py match those listed in HDF5 file
                assert motor_data.info['controls']['6K Compumotor']['probe']['port'] == langmuir_configs[i]['port']

                position_array, num_positions, shots_per_position, selected_shots = get_shot_positions(motor_data)
                position_arrays += [position_array]

                # Drop some shots from the data because they don't fit into a 3D structure
                if len(bias.shape) == 2:  # already selected certain shots in bias data
                    bias = bias[selected_shots, ...]
                current = current[selected_shots, ...]

                # Make bias and current 3D (position, shot_at_a_certain_position, frame) arrays
                #    as opposed to 2D (shot number, frame) arrays
                bias = bias.reshape(num_positions,       shots_per_position, -1)
                current = current.reshape(num_positions, shots_per_position, -1)
                # Dimensions of bias and current arrays:   position, shot, frame   (e.g. (71, 15, 55296))

                if sweep_view_mode:
                    preview_raw_sweep(bias, current, position_array, langmuir_configs[i], exp_params_dict, dt,
                                      plot_save_directory=plot_save_directory)

                # organize sweep bias and current into a 3D array of Characteristic objects
                characteristic_array = make_characteristic_array(bias, current, ramp_bounds)
                characteristic_arrays += [characteristic_array]

                if chara_view_mode:
                    preview_characteristics(characteristic_array, position_array, ramp_times,
                                            langmuir_configs[i], exp_params_dict,
                                            diagnostics=True, ion=ion_type,
                                            bimaxwellian=bimaxwellian,
                                            plot_save_directory=plot_save_directory)

                # cleanup 1
                del current

            # cleanup 2
            del bias

            # Check that all probes access the same positions at all times. Independent probe positions not implemented
            assert np.all([position == position_arrays[0] for position in position_arrays])
            positions = position_arrays[0]

            characteristics = np.stack(characteristic_arrays, axis=0)
            # Above: characteristics has one extra dimension "in front",
            #  to represent characteristics (sweep curves) from different probes or probe faces.
            #  Probe/face combinations are ordered by the order of elements in langmuir_configs, from configurations.py.

            # Perform langmuir diagnostics on each dataset
            print(f"Calculating Langmuir diagnostics...")
            diagnostics_dataset = langmuir_diagnostics(characteristics, positions, ramp_times,
                                                       langmuir_configs, ion_type, bimaxwellian=bimaxwellian)

            # cleanup 3
            del characteristics, positions

            diagnostics_dataset = diagnostics_dataset.assign_attrs(exp_params_dict)

            # Intermediate save point in case diagnostics are interrupted later
            save_datasets_nc([diagnostics_dataset], lang_nc_folder, "lang_", bimaxwellian)
            datasets.append(diagnostics_dataset)

    for i in range(len(datasets)):
        datasets[i] = datasets[i].assign_attrs(get_supplemental_metadata(datasets[i]))
    return datasets, hdf5_chosen_list


def interferometry_calibrate_datasets(datasets, density_diagnostic, interferometry_folder, interferometry_mode,
                                      core_radius, steady_state_times):
    """ Perform interferometry calibration on Langmuir diagnostic datasets. """
    # datasets[i].attrs = exp params
    calibrated_density_diagnostic = density_diagnostic + '_cal'
    print(f"{get_title(density_diagnostic)} mean steady-state interferometry calibration factors")

    for i in range(len(datasets)):
        if interferometry_mode == "overwrite" or (interferometry_mode == "append"
                                                  and (calibrated_density_diagnostic not in datasets[i]
                                                       or np.isnan(datasets[i][calibrated_density_diagnostic]).all())):
            print("\t", end="")  # indent
            calibrated_electron_density = interferometry_calibration(datasets[i][density_diagnostic].copy(),
                                                                     datasets[i].attrs,
                                                                     interferometry_folder,
                                                                     steady_state_times[i],
                                                                     core_radius)
            datasets[i] = datasets[i].assign({calibrated_density_diagnostic: calibrated_electron_density})

            """
            except (IndexError, ValueError, TypeError, AttributeError, KeyError) as e:
                print(f"Error in calibrating electron density: \n{str(e)}")
                calibrated_electron_density = datasets[i]['n_e'].copy()
            """

    return datasets


def get_diagnostics_to_plot(diagnostic_name_dict):
    """ Ask users for a list of diagnostics to plot. """
    print("The following diagnostics are available to plot: ")
    diagnostics_sort_indices = np.argsort(list(diagnostic_name_dict.values()))
    diagnostics_to_plot_ints = choose_multiple_from_list(
        np.array(list(diagnostic_name_dict.values()))[diagnostics_sort_indices], "diagnostic", null_action="skip")
    return [np.array(list(diagnostic_name_dict.keys()))[diagnostics_sort_indices][choice]
            for choice in diagnostics_to_plot_ints]


def save_datasets_nc(datasets: list[xr.Dataset], nc_folder, prefix="lang", bimaxwellian=False):
    """ Save a list of Langmuir diagnostic datasets to NetCDF files. """
    for i in range(len(datasets)):
        ds_save_name = get_dataset_save_name(datasets[i].attrs, prefix, bimaxwellian)
        ds_save_path = make_path(nc_folder, ds_save_name, "nc")
        write_netcdf(datasets[i], ds_save_path)


def get_dataset_save_name(attrs, prefix, bimaxwellian=False):
    """ Create a descriptive file name to save Langmuir diagnostic datasets as NetCDF (.nc) files. """
    return (prefix + attrs['Exp name'][:3] + attrs['Exp name'][-2:] + "_"
            + attrs['Run name'] + ("_bimax" if bimaxwellian else ""))
