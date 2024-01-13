from helper import *


def display_characteristics(characteristics_array, positions, ports, ramp_times, exp_params_dict, diagnostics=False,
                            areas=None, ion=None, bimaxwellian=False):

    x = np.unique(positions[:, 0])
    y = np.unique(positions[:, 1])
    print(f"\nDimensions of plateaus array: {characteristics_array.shape[0]} probes, {len(x)} x-positions, "
          f"{len(y)} y-positions, {characteristics_array.shape[2]} shots, and {characteristics_array.shape[-1]} ramps.")
    print(f"  * probe ports are {ports}",
          f"  * x positions range from {min(x)} to {max(x)}",
          f"  * y positions range from {min(y)} to {max(y)}",
          f"  * shot indices range from 0 to {characteristics_array.shape[2]}",
          f"  * ramp times range from {min(ramp_times):.2f} to {max(ramp_times):.2f}.", sep="\n")
    probe_x_y_ramp_to_plot = [0, 0, 0, 0, 0]
    variables_to_enter = ["probe", "x position", "y position", "shot", "ramp"]
    print("\nNote: Enter a non-integer below to terminate characteristics plotting mode.")
    chara_view_mode = True
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
        chara_to_plot = characteristics_array[(probe_x_y_ramp_to_plot[0], loc, *probe_x_y_ramp_to_plot[3:])]
        """ while chara_view_mode not in ["s", "a"]:
            chara_view_mode = input("(S)how current plot or (a)dd another Characteristic?").lower()
        if chara_view_mode == "s": """
        plot_title = (f"Run: {exp_params_dict['Exp name']}, {exp_params_dict['Run name']}\n"
                      f"Port: {ports[probe_x_y_ramp_to_plot[0]]}, x: {loc_x}, y: {loc_y}, "
                      f"shot: {probe_x_y_ramp_to_plot[3]}, time: {ramp_times[probe_x_y_ramp_to_plot[4]]:.2f}")
        if diagnostics:
            if areas is None or ion is None:
                raise ValueError("Area or ion not specified in characteristic display mode, but 'diagnostics' is True.")
            try:
                diagnostics = swept_probe_analysis(chara_to_plot, areas[probe_x_y_ramp_to_plot[0]], ion, bimaxwellian)
                electron_temperature = diagnostics['T_e'] if not bimaxwellian \
                    else unpack_bimaxwellian(diagnostics)['T_e_avg']
                plot_title += f"\nTemperature: {electron_temperature:.3f}"
            except (ValueError, RuntimeError, TypeError) as e:
                plot_title += f"\n(Error calculating plasma diagnostics: \n{str(e)[:35]})"
        chara_to_plot.plot()
        plt.title(plot_title)
        plt.tight_layout()
        plt.show()