from lapd_plasma_analysis.langmuir.helper import *


def preview_raw_sweep(bias, current, positions, langmuir_config, exp_params_dict, dt, plot_save_directory=""):
    """
    WIP

    Parameters
    ----------
    bias
    current : `astropy.units.Quantity`
        2D array (dimensions: x-y position, shot at x-y position) of Langmuir sweep (collected) current
    positions
    langmuir_config
    exp_params_dict
    dt
    plot_save_directory

    Returns
    -------

    """
    plt.rcParams['figure.figsize'] = (8, 3)

    x = np.unique(positions[:, 0])
    y = np.unique(positions[:, 1])
    print(f"\nPort {langmuir_config['port']}, face {langmuir_config['face']}")
    print(f"Dimensions of sweep bias and current array: "
          f"{len(x)} x-positions, "
          f"{len(y)} y-positions, and "
          f"{current.shape[1]} shots")
    print(f"  * x positions range from {min(x)} to {max(x)}",
          f"  * y positions range from {min(y)} to {max(y)}",
          f"  * shot indices range from 0 to {current.shape[1] - 1}.", sep="\n")
    x_y_shot_to_plot = [0, 0, 0]
    variables_to_enter = ["x position", "y position", "shot"]
    print("\nNotes: \tIndices are zero-based; choose an integer between 0 and n - 1, inclusive."
          "\n \t \tEnter a non-integer below (such as the empty string) to skip raw sweep preview mode "
          "for this isweep source and continue to diagnostics.")
    sweep_view_mode = True
    while sweep_view_mode:
        for i in range(len(x_y_shot_to_plot)):
            try:
                index_given = int(input(f"Enter a zero-based index for {variables_to_enter[i]}: "))
            except ValueError:
                sweep_view_mode = False
                break
            x_y_shot_to_plot[i] = index_given
        if not sweep_view_mode:
            break
        print()

        try:
            loc_x, loc_y = x[x_y_shot_to_plot[0]], y[x_y_shot_to_plot[1]]
            loc = (positions == [loc_x, loc_y]).all(axis=1).nonzero()[0][0]

            # Abbreviate "current" as "curr"
            bias_to_plot = bias[(loc,       *x_y_shot_to_plot[2:])]
            current_to_plot = current[(loc, *x_y_shot_to_plot[2:])]

            # String indicating port and face of sweep current source (e.g. "20L" or "27")
            port_face_string = f"{langmuir_config['port']}{langmuir_config['face'] if langmuir_config['face'] else ''}"
            plt.plot(np.arange(len(bias_to_plot)) * dt.to(u.ms).value, bias_to_plot)  # , label="Bias (V)")
            plt.title(f"Run: {exp_params_dict['Exp name']}, {exp_params_dict['Run name']}\n"
                      f"Probe port and face: {port_face_string}, "
                      f"x: {loc_x}, y: {loc_y}, shot: {x_y_shot_to_plot[2]}\n\n"
                      f"Voltage [V]")
            plt.xlabel("Time [ms]")
            # plt.legend()
            plt.tight_layout()
            if plot_save_directory:
                plt.savefig(f"{plot_save_directory}vsweep_time.pdf")
            plt.show()

            plt.plot(np.arange(len(bias_to_plot)) * dt.to(u.ms).value, current_to_plot)  # , label="Current (A)")
            plt.title(f"Run: {exp_params_dict['Exp name']}, {exp_params_dict['Run name']}\n"
                      f"Probe port and face: {port_face_string}, "
                      f"x: {loc_x}, y: {loc_y}, shot: {x_y_shot_to_plot[2]}\n\n"
                      f"Current [A]")
            plt.xlabel("Time [ms]")
            # plt.legend()
            plt.tight_layout()
            if plot_save_directory:
                plt.savefig(f"{plot_save_directory}isweep_time.pdf")
            plt.show()
        except IndexError as e:
            print(e)
            continue


def preview_characteristics(characteristics_array, positions, ramp_times, langmuir_config, exp_params_dict,
                            diagnostics=False, ion=None, bimaxwellian=False, plot_save_directory=""):
    """
    WIP

    Parameters
    ----------
    characteristics_array
    positions
    ramp_times
    langmuir_config
    exp_params_dict
    diagnostics
    ion
    bimaxwellian
    plot_save_directory

    Returns
    -------

    """

    area = langmuir_config['area']
    plt.rcParams['figure.figsize'] = (6, 4)

    x = np.unique(positions[:, 0])
    y = np.unique(positions[:, 1])
    print(f"\nPort {langmuir_config['port']}, face {langmuir_config['face']}")
    print(f"Dimensions of plateaus array: "
          f"{len(x)} x-positions, "
          f"{len(y)} y-positions, "
          f"{characteristics_array.shape[2]} shots, and "
          f"{characteristics_array.shape[-1]} ramps.")
    print(f"  * x positions range from {min(x)} to {max(x)}",
          f"  * y positions range from {min(y)} to {max(y)}",
          f"  * shot indices range from 0 to {characteristics_array.shape[2]}",
          f"  * ramp times range from {min(ramp_times):.2f} to {max(ramp_times):.2f}.", sep="\n")
    x_y_shot_ramp_to_plot = [0, 0, 0, 0]
    variables_to_enter = ["x position", "y position", "shot", "ramp"]
    print("\nNotes: \tIndices are zero-based; choose an integer between 0 and n - 1, inclusive."
          "\n \t \tEnter a non-integer below (such as the empty string) to quit characteristic preview mode "
          "and continue to diagnostics.")
    chara_view_mode = True
    while chara_view_mode:
        for i in range(len(x_y_shot_ramp_to_plot)):
            try:
                index_given = int(input(f"Enter a zero-based index for {variables_to_enter[i]}: "))
            except ValueError:
                chara_view_mode = False
                break
            x_y_shot_ramp_to_plot[i] = index_given
        if not chara_view_mode:
            break
        print()

        try:
            loc_x, loc_y = x[x_y_shot_ramp_to_plot[0]], y[x_y_shot_ramp_to_plot[1]]
            loc = (positions == [loc_x, loc_y]).all(axis=1).nonzero()[0][0]
            chara_to_plot = characteristics_array[(loc, *x_y_shot_ramp_to_plot[2:])]
            """ while chara_view_mode not in ["s", "a"]:
                chara_view_mode = input("(S)how current plot or (a)dd another Characteristic?").lower()
            if chara_view_mode == "s": """
            port_face_string = (str(langmuir_config['port'])
                                + (f" {langmuir_config['face']}" if langmuir_config['face'] else ""))
            plot_title = (f"Run: {exp_params_dict['Exp name']}, {exp_params_dict['Run name']}\n"
                          f"x: {loc_x}, y: {loc_y}, shot: {x_y_shot_ramp_to_plot[2]}, "
                          f"time: {ramp_times[x_y_shot_ramp_to_plot[3]]:.2f}")
            if diagnostics:
                if area is None or ion is None:
                    raise ValueError("Area or ion not specified in characteristic preview mode, but 'diagnostics' is True.")
                try:
                    diagnostics = swept_probe_analysis(chara_to_plot, area, ion, bimaxwellian)
                    electron_temperature = diagnostics['T_e'] if not bimaxwellian \
                        else unpack_bimaxwellian(diagnostics)['T_e_avg']
                    ion_density = diagnostics['n_i_OML']
                    plot_title += f"\nTemperature: {electron_temperature:.3f}; ion density: {ion_density:.2e}"
                    chara_to_plot.plot()
                    plt.plot(diagnostics['V_F'],
                             chara_to_plot.current[array_lookup(chara_to_plot.bias, diagnostics['V_F'])], 'go',
                             label=r"$V_F$")
                    plt.plot(diagnostics['V_P'],
                             chara_to_plot.current[array_lookup(chara_to_plot.bias, diagnostics['V_P'])], 'ro',
                             label=r"$V_P$")
                    plt.legend()

                except (ValueError, RuntimeError, TypeError) as e:
                    plot_title += f"\n(Error calculating plasma diagnostics: \n{str(e)[:35]})"
                    chara_to_plot.plot()

                plt.title(plot_title)
                plt.tight_layout()

                if plot_save_directory:
                    plt.savefig(f"{plot_save_directory}langmuir_sweep.pdf")
                plt.show()
        except IndexError as e:
            print(e)
            continue
