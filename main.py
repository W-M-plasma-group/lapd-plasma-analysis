"""
The lapd-plasma-analysis repository was written by Leo Murphy based on code
written in MATLAB by Conor Perks (MIT) and using the PlasmaPy and bapsflib libraries.
Comments are added inline. A separate documentation page is not yet complete.
"""

from file_access import ask_yes_or_no

from langmuir.configurations import get_config_id, get_ion
from langmuir.analysis import get_langmuir_datasets, get_diagnostics_to_plot, save_datasets_nc
from langmuir.plots import *

""" HDF5 file directory; end path with a slash """              # TODO user adjust
# hdf5_folder = "/Users/leomurphy/lapd-data/April_2018/"
# hdf5_folder = "/Users/leomurphy/lapd-data/March_2022/"
# hdf5_folder = "/Users/leomurphy/lapd-data/November_2022/"
# hdf5_folder = "/Users/leomurphy/lapd-data/January_2024/January_2024_all_working/"
hdf5_folder = "/Users/leomurphy/lapd-data/combined_lang_nc/"

""" Langmuir NetCDF directory; end path with a slash """        # TODO user adjust
assert hdf5_folder.endswith("/")
langmuir_nc_folder = hdf5_folder + "lang_nc/"

""" Interferometry file directory; end path with a slash """    # TODO user adjust
interferometry_mode = "append"                                  # "skip", "append", "overwrite"; recommended: "append"
interferometry_folder = ("/Users/leomurphy/lapd-data/November_2022/uwave_288_GHz_waveforms/"
                         if "November_2022" in hdf5_folder else hdf5_folder)

""" User parameters """                                         # TODO user adjust
# isweep_choice is user choice for probe or linear combination to plot; see isweep_selector in helper.py for more
# e.g. coefficients are for [[p1f1, p1f2], [p2f1, p2f2]]
isweep_choices = [[[1, 0], [0, 0]],     # first combination to plot: 1 * (first face on first probe)
                  [[0, 0], [1, 0]]]     # second combination to plot: 1 * (first face on second probe)
# isweep_choices = [[[0, 0], [1, 0]]]
# isweep_choices = [[[1, 0], [-1, 0]]]   # combination to plot: 1 * (face 1 on probe 1) - 1 * (face 1 on probe 2)
# isweep_choices = [[1, 0, 0, 0], [0, 0, 1, 0]]
# isweep_choices = [[1, 0]]  # , [0, 1]]
# isweep_choices = [0]
bimaxwellian = False                                        # note to self: perform both and store in same NetCDF file?
core_radius = 26. * u.cm                                    # From MATLAB code

""" Optional directory to save plots; end path with a slash"""          # TODO user adjust
plot_save_folder = ("/Users/leomurphy/Desktop/wm/Plasma research/Research images/Research images spring 2024/"
                    "new research plots mar-apr 2024/saved plots/")


# Diagram of LAPD
"""
       <- ~18m plasma length -> 
  ____________________________________       
  |    |                      '      |       A       
  |    |                      '      |       |   ~75 cm plasma diameter
  |____|______________________'______|       V
      (a)                    (b)    (c)
        +z direction (+ports) ==>
                  plasma flow ==> ???
            magnetic field B0 <==

a) LaB6 electron beam cathode
b) downstream mesh anode
c) downstream cathode
"""

if __name__ == "__main__":

    # TODO list of hardcoded parameters
    #    (16 ms, 24 ms) for January_2024 steady state period (detect_steady_state_period in diagnostics.py)

    print("\n===== Langmuir probe analysis =====")
    datasets, steady_state_times_runs, hdf5_paths = get_langmuir_datasets(
        langmuir_nc_folder, hdf5_folder, interferometry_folder, interferometry_mode,
        isweep_choices, core_radius, bimaxwellian, plot_save_folder)
    print("Diagnostics selected:", diagnostics_to_plot_list)

    # Plot chosen diagnostics for each individual dataset
    if ask_yes_or_no("Generate contour plot of selected diagnostics over time and radial position? (y/n) "):
        for plot_diagnostic in diagnostics_to_plot_list:
            for i in range(len(datasets)):
                plot_line_diagnostic(datasets[i], isweep_choices, plot_diagnostic, 'contour',
                                     steady_state_times_runs[i])

    # Plot radial profiles of diagnostic (steady-state time average), with color corresponding to first attribute
    #    and plot position on multiplot corresponding to second attribute
    if ask_yes_or_no("Generate line plot of selected diagnostics over radial position? (y/n) "):
        for plot_diagnostic in diagnostics_to_plot_list:
            multiplot_line_diagnostic(datasets, plot_diagnostic, isweep_choices, 'x',
                                      steady_state_by_runs=steady_state_times_runs,
                                      core_rad=core_radius, save_directory=plot_save_folder, tolerance=plot_tolerance)

    if ask_yes_or_no("Generate line plot of selected diagnostics over time? (y/n) "):
        for plot_diagnostic in diagnostics_to_plot_list:
            multiplot_line_diagnostic(datasets, plot_diagnostic, isweep_choices, 'time',
                                      steady_state_by_runs=steady_state_times_runs,
                                      core_rad=core_radius, save_directory=plot_save_folder)

    # Create indices for preserving order in which datasets were entered
    unsort_run_indices = np.argsort(np.array(sorted(np.arange(len(datasets)), key=lambda i: datasets[i].attrs['Run name'])))
    datasets = sorted(datasets, key=lambda ds: ds.attrs['Run name'])
    unsort_exp_indices = np.argsort(np.array(sorted(np.arange(len(datasets)), key=lambda i: datasets[i].attrs['Exp name'])))
    datasets = sorted(datasets, key=lambda ds: ds.attrs['Exp name'])
    unsort_indices = unsort_exp_indices[unsort_run_indices]

    # Split two steady state periods for jan_2024 data: (16, 24) and (27, 33) and plot with dotted
    datasets_split = datasets.copy()
    steady_state_times_runs_split = steady_state_times_runs.copy()
    # linestyles_split = ["solid"] * len(datasets)
    available_marker_styles = ('D', 'o', '^', 's')  # markers for Apr_18, Mar_22, Nov_22, Jan_24
    marker_styles_split = [available_marker_styles[get_config_id(dataset.attrs['Exp name'])] for dataset in datasets]
    for i in range(len(datasets)):
        if datasets[i].attrs['Exp name'] == "January_2024":
            datasets_split += [datasets[i]]
            steady_state_times_runs_split += [(27, 33) * u.ms]
            # linestyles_split += ["dotted"]
            marker_styles_split += ['x']
    # marker_styles_split = ['o' if style == 'solid' else 'x' for style in linestyles_split]

    # List that identifies probes and faces for 1) midplane and 2) upstream/downstream
    probes_faces_midplane_split = [(1, 0) if dataset.attrs['Exp name'] == "January_2024" else (0, 0)
                                   for dataset in datasets_split]
    probes_faces_parallel_split = [((0, 0), (1, 0)) for dataset in datasets_split]
    # above: first face on first probe & first face on second probe

    if ask_yes_or_no(f"Generate parallel plot of selected diagnostics? (y/n) "):  # plot of {parallel_diagnostics[key]}
        for plot_diagnostic in diagnostics_to_plot_list:  # for key in parallel_diagnostics:
            plot_parallel_diagnostic(datasets_split, steady_state_times_runs_split,
                                     probes_faces_midplane_split, probes_faces_parallel_split,
                                     marker_styles_split, diagnostic=plot_diagnostic, operation="median",
                                     core_radius=core_radius, save_directory=plot_save_folder)

    if ask_yes_or_no("Generate scatter plot of first two selected diagnostics? (y/n) "):
        scatter_plot_diagnostics(datasets_split, diagnostics_to_plot_list, steady_state_times_runs_split,
                                 probes_faces_midplane_split, marker_styles_split, operation="mean",
                                 core_radius=core_radius, save_directory=plot_save_folder)

    if ask_yes_or_no("Generate plot of inverse gradient scale length by position for selected diagnostics? (y/n) "):
        for plot_diagnostic in diagnostics_to_plot_list:
            plot_parallel_inverse_scale_length(datasets_split, steady_state_times_runs_split, plot_diagnostic,
                                               probes_faces_midplane_split, probes_faces_parallel_split,
                                               marker_styles_split, "median", core_radius, plot_save_folder)

    if ask_yes_or_no("Generate vertically stacked line plots for selected diagnostics? (y/n) "):
        datasets_unsorted = [datasets[unsort_indices[i]] for i in range(len(datasets))]
        steady_state_plateaus_runs_unsorted = np.array(steady_state_plateaus_runs)[unsort_indices]
        probes_faces_midplane_unsorted = np.array(probes_faces_midplane_split)[unsort_indices]
        probes_faces_parallel_unsorted = np.array(probes_faces_parallel_split)[unsort_indices]
        for x_dim in ("x", "time"):
            plot_grid(datasets_unsorted, diagnostics_to_plot_list, steady_state_times_runs_unsorted,
                      probes_faces_midplane_unsorted, probes_faces_parallel_unsorted, "mean", core_radius, x_dim,
                      num_rows=1, plot_save_folder=plot_save_folder)

    # (UNFINISHED) Shot plot: multiplot line diagnostics at specific time, with x-axis = x pos and curve color = shot #
    """if ask_yes_or_no("Generate line shot plot of selected diagnostics over radial position? (y/n) "):
        for plot_diagnostic in diagnostic_to_plot_list:
            multiplot_line_diagnostic(datasets, plot_diagnostic, isweep_choice, steady_state_plateaus_runs,
                                      tolerance=plot_tolerance)
    """




    #
    #

    if ask_yes_or_no("Print run parameters and diagnostics? (y/n) "):
        """
        z_distances = [((anode_z - dataset['z'].isel(probe=1).item() * u.cm)
                        - (anode_z - dataset['z'].isel(probe=0).item() * u.cm)).to(u.m)
                       for dataset in datasets]
        """

        print(f"Run ID \t\t\t T_e \t\t n_i_OML \t\t\t P_from_n_i_OML \t\t nu_ei \t\t\t v_para ")
        for i in range(len(datasets_split)):
            probes_faces = probes_faces_parallel_split[i]
            dataset = datasets_split[i]  # [{"probe": probes_faces_midplane_split[i][0], "face": probes_faces_midplane_split[i][1]}]

            core_steady_state_args = core_radius, steady_state_times_runs_split[i], "mean"
            density = []
            pressure = []
            temperature = []
            collision_frequency = []
            parallel_velocity = []
            for j in range(len(probes_faces)):
                ds = dataset[{"probe": probes_faces[j][0],
                              "face": probes_faces[j][1]}]
                temperature += [core_steady_state(ds['T_e'],                *core_steady_state_args).item()]
                pressure += [core_steady_state(ds['P_ei_from_n_i_OML'],     *core_steady_state_args).item()]
                density += [core_steady_state(ds['n_i_OML'],                *core_steady_state_args).item()]
                collision_frequency += [core_steady_state(ds['nu_ei'],      *core_steady_state_args).item()]
                parallel_velocity += [core_steady_state(ds['v_para'],       *core_steady_state_args).item()]
                # gas_pressure_torr = dataset.attrs['Fill pressure']
                # gas_density = dataset.attrs['Neutral density']

            print(f"{get_exp_run_string(datasets_split[i].attrs)} \t "
                  f"{temperature[0]:5.2f} {temperature[1]:5.2f} \t"
                  f"{density[0]:.2e} {density[1]:.2e} \t "
                  f"{pressure[0]:5.2f} {pressure[1]:5.2f} \t "
                  f"{collision_frequency[0]:.2e} {collision_frequency[1]:.2e} \t"
                  f"{parallel_velocity[0]:.2e} {parallel_velocity[1]:.2e} \t")

# TODO Not all MATLAB code has been transferred (e.g. neutrals, ExB)
# QUESTION: can we calibrate both Langmuir probes using an interferometry ratio depending only on one of them? (NO)
