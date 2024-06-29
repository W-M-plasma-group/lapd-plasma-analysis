"""
The lapd-plasma-analysis repository was written by Leo Murphy based on code
written in MATLAB by Conor Perks (MIT) and using the PlasmaPy and bapsflib libraries.
Comments are added inline. Separate documentation is not yet complete.
"""

from file_access import ask_yes_or_no

from langmuir.configurations import get_config_id
from langmuir.analysis import get_langmuir_datasets, get_diagnostics_to_plot, save_datasets_nc
from langmuir.plots import *

from mach.analysis import get_mach_datasets, get_velocity_datasets


""" HDF5 file directory; end path with a slash """                      # TODO user adjust
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
isweep_choices = [[[1, 0], [0, 0]],     # . 1st combination to plot: 1 * (first face on first probe)
                  [[0, 0], [1, 0]]]     # . 2nd combination to plot: 1 * (first face on second probe)
# isweep_choices = [[[1, 0], [-1, 0]]]  # .     combination to plot: 1 * (face 1 on probe 1) - 1 * (face 1 on probe 2)
bimaxwellian = False                                                    # TODO do both and store in same NetCDF file?
core_radius = 21. * u.cm                                                # TODO user can adjust (26 cm in MATLAB code)
plot_tolerance = np.nan  # 0.25                                         # TODO
velocity_plot_unit = u.km / u.s         # TODO not yet working          # TODO adjust !

""" Optional directory to save plots; end path with a slash"""          # TODO user adjust
plot_save_folder = ("/Users/leomurphy/Desktop/wm/Plasma research/Research images/Research images spring 2024/"
                    "new research plots mar-apr 2024/saved plots/")


# Diagram of LAPD
"""
           <- ~18m plasma length -> 
  __________________________________       
  |||     '                        |       A       
  |||     '                        |       |  ~60 cm plasma diameter
  |||_____'________________________|       V
  (a)    (b)                      (c)
        
        +z direction (+ports) -->>
                  plasma flow -->> 
            magnetic field B0 <<--
 
 a) heated LaB6 cathode
 b) grid anode
 c) end anode
"""

if __name__ == "__main__":

    # TODO list of hardcoded parameters
    #    (16 ms, 24 ms) for January_2024 steady state period (detect_steady_state_period in diagnostics.py)
    #    (27 ms, 33 ms) for January_2024 second steady-state period (main.py, below)

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
                plot_linear_diagnostic(datasets[i], isweep_choices, plot_diagnostic, 'contour',
                                       steady_state_times_runs[i])

    # Plot radial profiles of diagnostic (steady-state time average), with color corresponding to first attribute
    #    and plot position on multiplot corresponding to second attribute
    if ask_yes_or_no("Generate line plot of selected diagnostics over radial position? (y/n) "):
        for plot_diagnostic in diagnostics_to_plot_list:
            multiplot_linear_diagnostic(datasets, plot_diagnostic, isweep_choices, 'x',
                                        steady_state_by_runs=steady_state_times_runs, core_rad=core_radius,
                                        tolerance=plot_tolerance, save_directory=plot_save_folder)

    # Plot time profiles
    if ask_yes_or_no("Generate line plot of selected diagnostics over time? (y/n) "):
        for plot_diagnostic in diagnostics_to_plot_list:
            multiplot_linear_diagnostic(datasets, plot_diagnostic, isweep_choices, 'time',
                                        steady_state_by_runs=steady_state_times_runs, core_rad=core_radius,
                                        save_directory=plot_save_folder)

    # Split two steady state periods for jan_2024 data: (16, 24) and (27, 33) and plot with dotted
    datasets_split = datasets.copy()
    steady_state_times_runs_split = steady_state_times_runs.copy()
    # linestyles_split = ["solid"] * len(datasets)
    available_marker_styles = ('D', 'o', '^', 's')  # markers for Apr_18, Mar_22, Nov_22, Jan_24
    marker_styles_split = [available_marker_styles[get_config_id(dataset.attrs['Exp name'])] for dataset in datasets]
    for i in range(len(datasets)):
        if datasets[i].attrs['Exp name'] == "January_2024":  # Add copies of Jan24 experiments at end for 2nd steady st.
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
        time_unit = unit_safe(steady_state_times_runs[0])
        for x_dim in ("x", "time"):
            plot_grid(datasets, diagnostics_to_plot_list, steady_state_times_runs,
                      probes_faces_midplane, probes_faces_parallel, "mean", core_radius, x_dim,
                      num_rows=1, plot_save_folder=plot_save_folder)

    # (UNFINISHED) Shot plot: multiplot line diagnostics at specific time, with x-axis = x pos and curve color = shot #
    """if ask_yes_or_no("Generate line shot plot of selected diagnostics over radial position? (y/n) "):
        for plot_diagnostic in diagnostic_to_plot_list:
            multiplot_line_diagnostic(datasets, plot_diagnostic, isweep_choice, steady_state_plateaus_runs,
                                      tolerance=plot_tolerance)
    """

    #
    #
    #       Mach
    #       plots
    #       below
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
    #   Mixed plots and functions below
    #   (incorporating velocity and other diagnostics)
    #

    if ask_yes_or_no("Plot parallel plasma acceleration term versus parallel pressure gradient? (y/n) "):
        plot_acceleration_vs_pressure_gradient(datasets, steady_state_times_runs, core_radius,
                                               probes_faces_midplane_split, marker_styles_split, "mean",
                                               plot_save_folder, with_expectation=False)

    if ask_yes_or_no("Plot parallel plasma acceleration term versus parallel pressure gradient "
                     "with predicted trend lines? (y/n) "):
        plot_acceleration_vs_pressure_gradient(datasets, steady_state_times_runs, core_radius,
                                               probes_faces_midplane_split, marker_styles_split, "mean",
                                               plot_save_folder, with_expectation=True, offset=-2.7e7)

    if ask_yes_or_no("Print run parameters and diagnostics? (y/n) "):
        print(f"Run ID \t\t\t T_e \t\t n_i_OML \t\t\t P_from_n_i_OML \t\t nu_ei \t\t\t v_para ")
        for i in range(len(datasets_split)):
            probes_faces = probes_faces_parallel_split[i]
            dataset = datasets_split[i]

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

# TODO Not all MATLAB code has been transferred (e.g. ExB)
# QUESTION: can we calibrate both Langmuir probes using an interferometry ratio depending only on one of them? (NO)
