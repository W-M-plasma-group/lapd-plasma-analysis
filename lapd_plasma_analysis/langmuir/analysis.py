from lapd_plasma_analysis.file_access import *
from lapd_plasma_analysis.experimental import get_exp_params

from lapd_plasma_analysis.langmuir.helper import *
from lapd_plasma_analysis.langmuir.configurations import *
from lapd_plasma_analysis.langmuir.getIVsweep import get_isweep_vsweep
from lapd_plasma_analysis.langmuir.characterization import characterize_sweep_array
from lapd_plasma_analysis.langmuir.preview import preview_raw_sweep, preview_characteristics
from lapd_plasma_analysis.langmuir.diagnostics import (langmuir_diagnostics, detect_steady_state_times, get_pressure,
                                                       get_electron_ion_collision_frequencies)
from lapd_plasma_analysis.langmuir.neutrals import get_neutral_density
from lapd_plasma_analysis.langmuir.interferometry import interferometry_calibration
from lapd_plasma_analysis.langmuir.plots import get_title


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
        List of paths to HDF5 files opened. Can be None if NetCDF files used.

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
    interferometry_mode_actions = ({"skip": "skipped",
                                    "append": "added to uncalibrated datasets only",
                                    "overwrite": "recalculated for all datasets"})
    print(f"Interferometry mode is {repr(interferometry_mode)}. "
          f"Calibrated density data will be {interferometry_mode_actions[interferometry_mode]}.")

    print("Current HDF5 directory path:           \t", repr(hdf5_folder),
          "\nCurrent NetCDF directory path:         \t", repr(lang_nc_folder),
          "\nCurrent interferometry directory path: \t", repr(interferometry_folder),
          "\nLinear combinations of isweep sources:   \t", repr(isweep_choices),
          "\nThese can be changed in main.py.")
    input("Enter any key to continue: ")


def load_datasets(hdf5_folder, lang_nc_folder, bimaxwellian, plot_save_directory):
    """
    Load Langmuir datasets from NetCDF files or generate them from HDF5 files.
    """

    print("\nThe following Langmuir NetCDF files were found in the NetCDF folder (specified in main.py): ")
    nc_paths = sorted(search_folder(lang_nc_folder, 'nc', limit=52))
    nc_paths_chosen_ints = choose_multiple_list(nc_paths, "Langmuir data NetCDF file",
                                                null_action="perform diagnostics on HDF5 files")

    if len(nc_paths_chosen_ints) > 0:
        hdf5_chosen_list = None
        datasets = [xr.open_dataset(nc_paths[choice]) for choice in nc_paths_chosen_ints]
    else:
        print("\nThe following HDF5 files were found in the HDF5 folder (specified in main.py): ")
        hdf5_paths = sorted(search_folder(hdf5_folder, "hdf5", limit=52))
        hdf5_chosen_ints = choose_multiple_list(hdf5_paths, "HDF5 file",
                                                null_action="skip to Mach number calculations only")

        if len(hdf5_chosen_ints) == 0:
            return None

        hdf5_chosen_list = [hdf5_paths[choice] for choice in hdf5_chosen_ints]

        sweep_view_mode = (len(hdf5_chosen_list) == 1) and ask_yes_or_no("Use raw sweep preview mode? (y/n) ")
        chara_view_mode = (len(hdf5_chosen_list) == 1) and ask_yes_or_no("Use characteristic preview mode? (y/n) ")

        datasets = []
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

            # get current and bias data from Langmuir probe
            bias, currents, positions, dt = get_isweep_vsweep(hdf5_path, vsweep_board_channel,
                                                              langmuir_configs, voltage_gain, orientation)

            if sweep_view_mode:
                preview_raw_sweep(bias, currents, positions, langmuir_configs[['port', 'face']], exp_params_dict, dt,
                                  plot_save_directory=plot_save_directory)

            # organize sweep bias and current into an array of Characteristic objects
            characteristics, ramp_times = characterize_sweep_array(bias, currents, dt)

            # cleanup 1
            del bias, currents

            if chara_view_mode:
                preview_characteristics(characteristics, positions, ramp_times,
                                        langmuir_configs, exp_params_dict,
                                        diagnostics=True, ion=ion_type,
                                        bimaxwellian=bimaxwellian,
                                        plot_save_directory=plot_save_directory)

            # perform langmuir diagnostics on each dataset
            diagnostics_dataset = langmuir_diagnostics(characteristics, positions, ramp_times,
                                                       langmuir_configs, ion_type, bimaxwellian=bimaxwellian)

            # cleanup 2
            del characteristics, positions

            diagnostics_dataset = diagnostics_dataset.assign_attrs(exp_params_dict)

            # Intermediate save point in case diagnostics are interrupted later
            save_datasets_nc([diagnostics_dataset], lang_nc_folder, "lang_", bimaxwellian)
            datasets.append(diagnostics_dataset)

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
    diagnostics_to_plot_ints = choose_multiple_list(
        np.array(list(diagnostic_name_dict.values()))[diagnostics_sort_indices],
        "diagnostic", null_action="skip")
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
