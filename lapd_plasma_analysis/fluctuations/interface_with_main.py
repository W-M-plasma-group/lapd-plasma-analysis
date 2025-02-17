
from lapd_plasma_analysis.fluctuations.analysis import *
from lapd_plasma_analysis.file_access import ask_yes_or_no, choose_multiple_list
import ast



def ask_about_plots(data_list):
    quantities = ['density', 'isat', 'vf', 'dvf']
    choice_indices = choose_multiple_list(quantities, "Quantities to plot")

    if ask_yes_or_no("Plot profiles (y/n)?"):
        for i in choice_indices:
            x = get_plotting_params("x")
            time = get_plotting_params("time")
            shot = get_plotting_params("shot")
            for data in data_list:
                get_profile(data[quantities[i]], x=x, time=time, shot=shot, plot=True)

    if ask_yes_or_no("Plot time series (y/n)?"):
        for i in choice_indices:
            x = get_plotting_params("x")
            time = get_plotting_params("time")
            shot = get_plotting_params("shot")
            for data in data_list:
                get_time_series(data[quantities[i]], x=x, time=time, shot=shot, plot=True)

    if ask_yes_or_no("Plot PSD (y/n)?"):
        for i in choice_indices:
            x = get_plotting_params("x")
            bin = get_plotting_params("bin")
            shot = get_plotting_params("shot")
            for data in data_list:
                get_psd_from_data(data[quantities[i]], x=x, bin=bin, shot=shot, plot=True)

    if ask_yes_or_no("Make contour plot(s) (y/n)?"):
        for i in choice_indices:
            x = get_plotting_params("x")
            time = get_plotting_params("time")
            shot = get_plotting_params("shot")
            for data in data_list:
                get_contour(data[quantities[i]], x=x, time=time, shot=shot, plot=True)

    if ask_yes_or_no("Make plot of integrated PSD vs density gradient scale length (y/n)?"):
        integrated_psds = []
        lns = []
        x = get_plotting_params("x")
        bin = get_plotting_params("bin")
        shot = get_plotting_params("shot")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for data in data_list:
            ln, psd = plot_total_flux_vs_Ln(data['density'], x=x, shot=shot, bin=bin, plot=False)
            lns.append(ln)
            integrated_psds.append(psd)
            ax.plot(ln, psd, marker='o')
        ax.set_xlabel('$L_n$ (m)')
        ax.set_ylabel('integrated PSD (cm^-6/Hz I think)')
        #ax.plot(lns, integrated_psds, marker='o', linestyle='', color='black')
        fig.show()

def get_plotting_params(parameter_as_string):
    print(f"      Input "+parameter_as_string+" value (ex. '12.0') or range (ex. '(10.0, 13.5)')\n"
           "      to plot, or press enter to plot over all " + parameter_as_string + ".")

    inp = input("      "+parameter_as_string+":")

    if inp == "":
        return None

    try:
        inp = float(inp)
        return inp

    except ValueError:
        try:
            tup = ast.literal_eval(inp)
            return tup
        except ValueError:
            print('Unable to decipher provided string, plotting over all ' + parameter_as_string + '.')
            return None
