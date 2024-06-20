"""
The lapd-plasma-analysis repository was written by Leo Murphy based on code
written in MATLAB by Conor Perks (MIT) and using the PlasmaPy and bapsflib libraries.
Comments are added inline. Separate documentation is not yet complete.
"""

from file_access import ask_yes_or_no

from langmuir.configurations import get_config_id, get_ion
from langmuir.analysis import get_langmuir_datasets, get_diagnostics_to_plot, save_datasets_nc
from langmuir.plots import *

from mach.analysis import get_mach_datasets, get_velocity_datasets


""" HDF5 file directory; end path with a slash """                      # TODO user adjust
# hdf5_folder = "/Users/leomurphy/lapd-data/April_2018/"
# hdf5_folder = "/Users/leomurphy/lapd-data/March_2022/"
# hdf5_folder = "/Users/leomurphy/lapd-data/November_2022/"
# hdf5_folder = "/Users/leomurphy/lapd-data/January_2024/January_2024_all_working/"
hdf5_folder = "/Users/leomurphy/lapd-data/combined/"

assert hdf5_folder.endswith("/")
""" Langmuir & Mach NetCDF directories; end path with a slash """       # TODO user adjust
langmuir_nc_folder = hdf5_folder + "lang_nc/"
mach_nc_folder = hdf5_folder + "mach_nc/"

""" Interferometry file directory; end path with a slash """            # TODO user adjust
interferometry_folder = ("/Users/leomurphy/lapd-data/November_2022/uwave_288_GHz_waveforms/"
                         if "November_2022" in hdf5_folder else hdf5_folder)

""" Interferometry & Mach access modes """  # Options are "skip", "append", "overwrite"; recommended is "append"
interferometry_mode = "skip"                                            # TODO user adjust
mach_velocity_mode = "append"                                           # not fully implemented

""" Other user parameters """                                           # TODO user adjust
# isweep_choice is user choice for probe or linear combination to plot; see isweep_selector in helper.py for more
# e.g. coefficients are for [[p1f1, p1f2], [p2f1, p2f2]]
isweep_choices = [[[1, 0], [0, 0]],     # first combination to plot: 1 * (first face on first probe)
                  [[0, 0], [1, 0]]]     # second combination to plot: 1 * (first face on second probe)
# isweep_choices = [[[0, 0], [1, 0]]]
# isweep_choices = [[[1, 0], [-1, 0]]]   # combination to plot: 1 * (face 1 on probe 1) - 1 * (face 1 on probe 2)
bimaxwellian = False                                        # note to self: perform both and store in same NetCDF file?
core_radius = 21 * u.cm  # 26. * u.cm                                   # From MATLAB code TODO !
plot_tolerance = np.nan  # 0.25                                         # TODO
velocity_plot_unit = u.km / u.s         # TODO not yet working          # TODO adjust !

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
                  plasma flow ==> 
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

    print("\n===== Mach probe analysis =====")
    mach_datasets = get_mach_datasets(mach_nc_folder, hdf5_folder, datasets, hdf5_paths, mach_velocity_mode)
    if len(mach_datasets) > 0:
        datasets = get_velocity_datasets(datasets, mach_datasets, mach_velocity_mode)
        save_datasets_nc(datasets, langmuir_nc_folder, "lang_", bimaxwellian)
        print(f"Velocities will be plotted in {str(velocity_plot_unit)}. Ensure that plots are labeled correctly.")

    # Get possible diagnostics and their full names, e.g. "n_e" and "Electron density", and ask user to select some
    diagnostic_name_dict = {key: get_title(key) for key in set.intersection(*[set(dataset) for dataset in datasets])}
    diagnostics_to_plot_list = get_diagnostics_to_plot(diagnostic_name_dict)
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
    """
    unsort_run_indices = np.argsort(np.array(sorted(np.arange(len(datasets)), key=lambda i: datasets[i].attrs['Run name'])))
    datasets = sorted(datasets, key=lambda ds: ds.attrs['Run name'])
    unsort_exp_indices = np.argsort(np.array(sorted(np.arange(len(datasets)), key=lambda i: datasets[i].attrs['Exp name'])))
    datasets = sorted(datasets, key=lambda ds: ds.attrs['Exp name'])
    unsort_indices = unsort_exp_indices[unsort_run_indices]
    """
    # original order of datasets as entered by user
    datasets_unsorted = datasets.copy()
    steady_state_times_runs_unsorted = steady_state_times_runs.copy()

    # Calculate list of indices that sort datasets array
    sort_run_indices = np.array(sorted(np.arange(len(datasets)), key=lambda i: datasets[i].attrs['Run name']))
    sort_exp_indices = np.array(sorted(np.arange(len(datasets)), key=lambda i: datasets[sort_run_indices[i]
                                                                                        ].attrs['Exp name']))
    sort_indices = sort_run_indices[sort_exp_indices]
    unsort_indices = np.argsort(sort_indices)
    del sort_run_indices, sort_exp_indices

    # sort by lexicographical order of experiment series name and run name
    datasets = [datasets[i] for i in sort_indices]
    steady_state_times_runs = [steady_state_times_runs[i] for i in sort_indices]

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

    # List that identifies probes and faces for 1) midplane and 2) low-z / high-z
    probes_faces_midplane_split = [(1, 0) if dataset.attrs['Exp name'] == "January_2024" else (0, 0)
                                   for dataset in datasets_split]
    probes_faces_parallel_split = [((0, 0), (1, 0)) for dataset in datasets_split]
    # format: ((low-z probe, low-z face), (high-z probe, high-z face)) for ds in ...
    # above: selects first (0th) probe's first (0th) face & second (1th) probe's first (0th) face

    if ask_yes_or_no(f"Generate parallel plot of selected diagnostics? (y/n) "):  # plot of {parallel_diagnostics[key]}
        for plot_diagnostic in diagnostics_to_plot_list:  # for key in parallel_diagnostics:
            plot_parallel_diagnostic(datasets_split, steady_state_times_runs_split,
                                     probes_faces_midplane_split, probes_faces_parallel_split,
                                     marker_styles_split, diagnostic=plot_diagnostic, operation="mean",
                                     core_radius=core_radius, save_directory=plot_save_folder)

    if ask_yes_or_no("Generate scatter plot of first two selected diagnostics? (y/n) "):
        scatter_plot_diagnostics(datasets_split, diagnostics_to_plot_list, steady_state_times_runs_split,
                                 probes_faces_midplane_split, marker_styles_split, operation="mean",
                                 core_radius=core_radius, save_directory=plot_save_folder)

    if ask_yes_or_no("Generate plot of gradient scale length by position for selected diagnostics? (y/n) "):
        for plot_diagnostic in diagnostics_to_plot_list:
            plot_parallel_inverse_scale_length(datasets_split, steady_state_times_runs_split, plot_diagnostic,
                                               probes_faces_midplane_split, probes_faces_parallel_split,
                                               marker_styles_split, "mean", core_radius, plot_save_folder,
                                               scale_length_mode="exponential")  # 'linear' or 'exponential'

    if ask_yes_or_no("Generate grid line plots for selected diagnostics? (y/n) "):
        time_unit = unit_safe(steady_state_times_runs_split[0])
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
    #
    #
    #
    #
    #

    print("\nBeginning Mach plots")
    if ask_yes_or_no("Plot contour plots for Mach number? (y/n) "):
        for i in range(len(mach_datasets)):
            mach_ds = mach_datasets[i]
            for variable in mach_ds:
                for probe in range(len(mach_ds.probe)):
                    da = mach_ds.isel(probe=probe)[variable].mean(dim='shot', keep_attrs=True).squeeze()
                    da.plot.contourf(vmax=np.nanquantile(da, 0.9), x='time')  # robust=True
                    title = f"{get_title(variable)} (Probe {mach_ds.probe[probe].item()})"
                    plt.title(get_exp_run_string(mach_ds.attrs) + "\n" + title)
                    plt.show()

    if ask_yes_or_no("Plot multiline plots for Mach number? (y/n) "):
        for i in range(len(mach_datasets)):
            mach_ds = mach_datasets[i]
            for variable in mach_ds:
                for probe in range(len(mach_ds.probe)):
                    z = (anode_z - mach_ds.coords['z'][probe].item() * u.cm).to(u.m).value
                    da = mach_ds.isel(probe=probe)[variable].mean(dim='shot', keep_attrs=True)
                    core_steady_state_profile = core_steady_state(da_input=da,
                                                                  steady_state_times=steady_state_times_runs_split[i],
                                                                  operation="mean",
                                                                  dims_to_keep=['x']  # , 'time']
                                                                  ).squeeze().plot(x='x', label=f"z = {z:.2f} m")
                plt.title(get_exp_run_string(mach_ds.attrs) + "\n" + get_title(variable))
                plt.legend()
                plt.tight_layout()
                plt.show()

    #
    #

    z_distances = [((anode_z - dataset['z'].isel(probe=1).item() * u.cm)
                    - (anode_z - dataset['z'].isel(probe=0).item() * u.cm)).to(u.m)
                   for dataset in datasets]

    parallel_velocity_average = [(dataset['v_para'].isel(probe=1) + dataset['v_para'].isel(probe=0)) / 2
                                 for dataset in datasets]
    parallel_velocity_gradient = [(datasets[i]['v_para'].isel(probe=1) - datasets[i]['v_para'].isel(probe=0)
                                   ) / z_distances[i] for i in range(len(datasets))]    # z is in m; v is in m/s
    parallel_acceleration = [parallel_velocity_average[i] * parallel_velocity_gradient[i]
                             for i in range(len(datasets))]

    parallel_pressure_gradient = [(datasets[i]['P_ei_from_n_i_OML'].isel(probe=1)
                                   - datasets[i]['P_ei_from_n_i_OML'].isel(probe=0)
                                   ) / z_distances[i] for i in range(len(datasets))]
    from plasmapy.particles import particle_mass
    parallel_density_average = [particle_mass(get_ion(datasets[i].attrs['Run name']))
                                * (datasets[i]['n_i_OML'].isel(probe=1)
                                   + datasets[i]['n_i_OML'].isel(probe=0)) / 2
                                for i in range(len(datasets))]
    parallel_pressure_gradient_over_density = [-parallel_pressure_gradient[i] / parallel_density_average[i]
                                               for i in range(len(datasets))]

    #
    #

    if ask_yes_or_no("Plot parallel plasma acceleration term versus parallel pressure gradient? (y/n) "):
        plt.rcParams['figure.figsize'] = (5.5, 3.5)
        collision_frequencies = extract_collision_frequencies(datasets, core_radius, steady_state_times_runs,
                                                              probes_faces_midplane_split, "mean")
        collision_frequencies_log_normalized = normalize(np.log(collision_frequencies), 0, 0.9)
        color_map = matplotlib.colormaps["plasma"](collision_frequencies_log_normalized)

        max_pressure_gradient = 0
        for i in range(len(datasets)):
            pressure_gradient = core_steady_state(parallel_pressure_gradient_over_density[i], core_radius,
                                                  steady_state_times_runs[i], "mean")
            acceleration = core_steady_state(parallel_acceleration[i], core_radius,
                                             steady_state_times_runs[i], "mean")
            plt.scatter(pressure_gradient, acceleration, color=color_map[i], marker=marker_styles_split[i])
            if max_pressure_gradient < pressure_gradient:
                max_pressure_gradient = pressure_gradient

        normalizer = matplotlib.colors.LogNorm(vmin=np.min(collision_frequencies),
                                               vmax=np.max(collision_frequencies))
        color_bar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalizer, cmap='plasma'), ax=plt.gca())
        color_bar.set_label(r"$\nu_{ei}$" " [Hz]\n(midplane)", rotation=0, labelpad=30)

        # plt.title("LHS vs RHS of equation 2.2 in thesis")
        # plt.ylabel(r"$v_z \ \frac{\partial v_z}{\partial_z}$ [m / s2]")
        plt.title(r"$v_z \ \frac{\partial v_z}{\partial z}$ [m / s$^2$]  ", y=0.85, loc='right')
        plt.xlabel(r"$- \frac{1}{\rho} \frac{\partial p}{\partial z}$ [m / s$^2$]")
        plt.tight_layout()
        if plot_save_folder:
            plt.savefig(f"{plot_save_folder}parallel_acceleration_vs_parallel_pressure_gradient.pdf")
        plt.show()

    #
    #

    if ask_yes_or_no("Plot parallel plasma acceleration term versus parallel pressure gradient "
                     "with predicted trend lines? (y/n) "):
        plt.rcParams['figure.figsize'] = (5.5, 3.5)
        collision_frequencies = extract_collision_frequencies(datasets, core_radius, steady_state_times_runs,
                                                              probes_faces_midplane_split, "mean")
        collision_frequencies_log_normalized = normalize(np.log(collision_frequencies), 0, 0.9)
        color_map = matplotlib.colormaps["plasma"](collision_frequencies_log_normalized)

        scaled_parallel_pressure_gradients = []
        for i in range(len(datasets)):
            scaled_parallel_pressure_gradient = core_steady_state(
                parallel_pressure_gradient_over_density[i], core_radius, steady_state_times_runs[i], "mean")
            scaled_parallel_pressure_gradients += [value_safe(scaled_parallel_pressure_gradient)]
            acceleration = core_steady_state(parallel_acceleration[i], core_radius,
                                             steady_state_times_runs[i], "mean")
            plt.scatter(scaled_parallel_pressure_gradient, acceleration, color=color_map[i],
                        marker=marker_styles_split[i])

        max_scaled_parallel_pressure_gradient = np.max(scaled_parallel_pressure_gradients)
        x = np.linspace(0,
                        1.4 * max_scaled_parallel_pressure_gradient, 10)
        x2 = np.linspace(0.8 * np.min(scaled_parallel_pressure_gradients),
                         0.39 * max_scaled_parallel_pressure_gradient, 10)
        plt.plot(x, 1 * x,  color='black',      linestyle='--',     # expected trend line
                 # label=r"$v_z \ \frac{\partial v_z}{\partial z} = - \frac{1}{\rho} \frac{\partial p}{\partial z}$")
                 label=r"$y = x$")  # r"$y = x$ (model)"
        plt.plot(x, 0 * x,  color='gray',       linestyle=':',      # zero line
                 # label=r"$v_z \ \frac{\partial v_z}{\partial z} = 0$")
                 label=r"$y = 0$")  # r"$y = 0$"
        plt.plot(x2, x2 - 2.7e7, color='silver',     linestyle='--',  # (0, (5, 10)),
                 label=r"$y = x$, offset")   # r"$y = x - 2.7e7$"

        normalizer = matplotlib.colors.LogNorm(vmin=np.min(collision_frequencies),
                                               vmax=np.max(collision_frequencies))
        color_bar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalizer, cmap='plasma'), ax=plt.gca())
        color_bar.set_label(r"$\nu_{ei}$" " [Hz]\n(midplane)", rotation=0, labelpad=30)

        # plt.title("LHS vs RHS of equation 2.2 in thesis")
        # plt.ylabel(r"$v_z \ \frac{\partial v_z}{\partial_z}$ [m / s2]")

        plt.xlim(left=0, right=1.05 * max_scaled_parallel_pressure_gradient)
        plt.ylim(top=max_scaled_parallel_pressure_gradient / 3)
        plt.title(r"  $v_z \ \frac{\partial v_z}{\partial z}$ [m/s$^2$]  ",
                  y=0.85, loc='right')  # loc='left'  # [m/s$^2$]
        plt.xlabel(r"  $- \frac{1}{\rho} \frac{\partial p}{\partial z}$ [m/s$^2$]  ")
        plt.legend(loc='upper right', bbox_to_anchor=(1.0, 0.84))  # loc="center right"   # \ \left[ \frac{\text{m}}{\text{s}^2} \right]
        plt.tight_layout()
        if plot_save_folder:
            plt.savefig(f"{plot_save_folder}parallel_acceleration_vs_parallel_pressure_gradient_with_expectation.pdf")
        plt.show()

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
