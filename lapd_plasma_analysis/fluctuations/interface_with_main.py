
from lapd_plasma_analysis.fluctuations.analysis import *
from lapd_plasma_analysis.file_access import ask_yes_or_no, choose_multiple_from_list
import ast



def ask_about_plots(data_list, plot_save_folder=None, langmuir_folder=None, filenames = None):
    """
    Lets the user interface with the fluctuation data when `main.py` is run.
    Asks the user to ask which data they would like visualized and over which coordinates.

    Parameters
    ----------
    data_list : `list` of `xarray.Dataset`
        A list of datasets, each of which corresponds to a single .nc file.

    """
    quantities = ['density', 'isat', 'vf', 'dvf']
    choice_indices = choose_multiple_from_list(quantities, "Quantities to plot")

    # Checked and good for main_luke so long as you input a time to average over
    if ask_yes_or_no("Plot profiles (y/n)?"):
        x = get_plotting_params("x")
        time = get_plotting_params("time")
        shot = get_plotting_params("shot")
        for i in choice_indices:
            for data in data_list:
                for z in data.coords["z"].values:
                    get_profile(data[quantities[i]].sel(z=z), x=x, time=time, shot=shot, z=z,
                                plot=True, plot_save_folder=plot_save_folder)

    # Checked and good for main_luke so long as you input a time to average over
    if ask_yes_or_no("Plot time series (y/n)?"):
        x = get_plotting_params("x")
        time = get_plotting_params("time")
        shot = get_plotting_params("shot")
        for i in choice_indices:
            for data in data_list:
                for z in data.coords["z"].values:
                    get_time_series(data[quantities[i]].sel(z=z), x=x, time=time, shot=shot, z=z,
                                    plot=True, plot_save_folder=plot_save_folder)

    if ask_yes_or_no("Plot PSD (y/n)?"):
        x = get_plotting_params("x")
        bin = get_plotting_params("bin")
        shot = get_plotting_params("shot")
        for i in choice_indices:
            for data in data_list:
                for z in data.coords["z"].values:
                    get_spectrum_from_data(data[quantities[i]].sel(z=z), x=x, bin=bin, shot=shot, z=z, plot=True,
                                 scaling="psd", plot_save_folder=plot_save_folder)

    if ask_yes_or_no("Plot power spectrum (y/n)?"):
        x = get_plotting_params("x")
        bin = get_plotting_params("bin")
        shot = get_plotting_params("shot")
        for i in choice_indices:
            for data in data_list:
                for z in data.coords["z"].values:
                    get_spectrum_from_data(data[quantities[i]].sel(z=z), x=x, bin=bin, shot=shot, z=z, plot=True,
                                 scaling="power spectrum", plot_save_folder=plot_save_folder)

    if ask_yes_or_no("Plot amplitude spectrum (y/n)?"):
        x = get_plotting_params("x")
        bin = get_plotting_params("bin")
        shot = get_plotting_params("shot")
        for i in choice_indices:
            for data in data_list:
                for z in data.coords["z"].values:
                    get_spectrum_from_data(data[quantities[i]].sel(z=z), x=x, bin=bin, shot=shot, z=z, plot=True,
                                 scaling="amplitude", plot_save_folder=plot_save_folder)

    if ask_yes_or_no("Make contour plot(s) (y/n)?"):
        x = get_plotting_params("x")
        time = get_plotting_params("time")
        shot = get_plotting_params("shot")
        for i in choice_indices:
            for data in data_list:
                for z in data.coords["z"].values:
                    get_contour(data[quantities[i]].sel(z=z), x=x, time=time, shot=shot, z=z,
                                plot=True, plot_save_folder=plot_save_folder)

    if ask_yes_or_no("Make plot of integrated PSD vs density gradient scale length\n"
                     "with z position as the color (y/n)?"):

        x1 = get_plotting_params("x range for left (- x) linear fit")
        x2 = get_plotting_params("x range for right (+ x) linear fit")
        Ln_range1 = get_plotting_params("Ln range for left (- x)")
        Ln_range2 = get_plotting_params("Ln range for right (+ x)")
        bin = get_plotting_params("bin")
        shot = get_plotting_params("shot")
        print(f"Available z values: {[data.coords['z'].values for data in data_list]}")
        z_fix = get_plotting_params("z (pick one, or press enter for all)")
        assert x1 is not None or x2 is not None
        for i in choice_indices:
            integrated_psds = []
            lns = []
            fig = plt.figure()
            ax = fig.add_subplot(111)
            norm = plt.Normalize(vmin=np.min(600), vmax=np.max(900))
            for data in tqdm(data_list, desc="Processing..."):
                for z in data.coords["z"].values:
                    if ((z_fix is not None) and (z == z_fix)) or z_fix is None:
                        if x1 is not None:
                            ln1, psd1, ln1_err, psd1_err = plot_total_flux_vs_Ln(data.sel(z=z), quantities[i], x=x1, Ln_range=Ln_range1, shot=shot, time=bin, bin=bin, plot=False)
                            lns.append(ln1)
                            integrated_psds.append(psd1)
                            #ax.plot(ln1, psd1, marker='o')
                            ax.errorbar(ln1, psd1, psd1_err, ln1_err, marker='<', markerfacecolor=cmap(norm(z)),
                                        markeredgecolor="black", ecolor="black", capsize=1.5, elinewidth=0.5, capthick=0.5)
                        if x2 is not None:
                            ln2, psd2, ln2_err, psd2_err = plot_total_flux_vs_Ln(data.sel(z=z), quantities[i], x=x2, Ln_range=Ln_range2, shot=shot, time=bin, bin=bin, plot=False)
                            lns.append(ln2)
                            integrated_psds.append(psd2)
                            #ax.plot(ln2, psd2, marker='o')
                            ax.errorbar(ln2, psd2, psd2_err, ln2_err, marker='>', markerfacecolor=cmap(norm(z)),
                                        markeredgecolor="black", ecolor="black", capsize=1.5, elinewidth=0.5, capthick=0.5)

            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('$z$ [cm]')
            ax.set_xlabel('$L_n$ [cm]')
            ax.set_ylabel(f'normalized {quantities[i]} fluctuations') #[$\text{cm}^{-3}$]')
            #ax.plot(lns, integrated_psds, marker='o', linestyle='', color='black')
            fig.tight_layout()
            fig.show()
            fig.savefig(plot_save_folder + data[quantities[i]].name + '_' + "fluctuations_vs_Ln_"+ get_time() + '.png',
                        dpi=150)

    if ask_yes_or_no("Make plot of integrated PSD vs density gradient scale length\n"
                     "with collision freq as the color (y/n)?"):

        x1 = get_plotting_params("x range for left (- x) linear fit")
        x2 = get_plotting_params("x range for right (+ x) linear fit")
        Ln_range1 = get_plotting_params("Ln range for left (- x)")
        Ln_range2 = get_plotting_params("Ln range for right (+ x)")
        bin = get_plotting_params("bin")
        shot = get_plotting_params("shot")
        print(f"Available z values: {[data.coords['z'].values for data in data_list]}")
        z_fix = get_plotting_params("z (pick one, or press enter for all)")

        param = "T_e"
        min_vals = []
        max_vals = []
        color_vals = []
        for data in tqdm(data_list, desc="Parsing data..."):
            for z in data.coords["z"].values:
                langmuir_dataset = get_langmuir_dataset(data)
                da1, _ =  get_langmuir_profiles(langmuir_dataset, param, z,
                                                                    x=Ln_range1, time=bin, shot=shot,
                                                                    plot=False)
                da2, _ = get_langmuir_profiles(langmuir_dataset, param, z,
                                                                    x=Ln_range2, time=bin, shot=shot,
                                                                    plot=False)
                # da = xr.concat([da1, da2], dim='x')
                # da_min, da_max = float(da.min(skipna=True)), float(da.max(skipna=True))
                # min_vals.append(da_min)
                # max_vals.append(da_max)
                qx1, qx2 = Ln_range1
                color_val = abs((da1.values[-1] - da1.values[0]) / (qx2 - qx1))
                color_vals.append(color_val)
                qx1, qx2 = Ln_range2
                color_val = abs((da2.values[-1] - da2.values[0]) / (qx2 - qx1))
                color_vals.append(color_val)

        norm = plt.Normalize(vmin=np.nanmin(color_vals), vmax=np.nanmax(color_vals))

        assert x1 is not None or x2 is not None
        for i in choice_indices:
            integrated_psds = []
            lns = []
            fig = plt.figure()
            ax = fig.add_subplot(111)

            for data in tqdm(data_list, desc="Processing..."):
                langmuir_dataset = get_langmuir_dataset(data)
                for z in data.coords["z"].values:
                    if ((z_fix is not None) and (z == z_fix)) or z_fix is None:
                        if x1 is not None:
                            try:
                                ln1, psd1, ln1_err, psd1_err = plot_total_flux_vs_Ln(data.sel(z=z), quantities[i], x=x1, Ln_range=Ln_range1, shot=shot, time=bin, bin=bin, plot=False)
                                lns.append(ln1)
                                integrated_psds.append(psd1)
                                #ax.plot(ln1, psd1, marker='o')
                                mean_langmuir, std_langmuir = get_langmuir_profiles(langmuir_dataset, param, z, x=Ln_range1, time=bin, shot=shot, plot=False)
                                # color_val =  float(mean_langmuir.mean(dim="x"))
                                xxx, xxxx = Ln_range1
                                color_val = abs((mean_langmuir.values[-1] - mean_langmuir.values[0])/(xxxx-xxx))
                                if ln1 < 25 and ln1_err < ln1:
                                    ax.errorbar(ln1, psd1, psd1_err, ln1_err, marker='<', markerfacecolor=cmap(norm(color_val)),
                                            markeredgecolor="black", ecolor="black", capsize=1.5, elinewidth=0.5, capthick=0.5)
                            except:
                                pass
                        if x2 is not None:
                            try:
                                ln2, psd2, ln2_err, psd2_err = plot_total_flux_vs_Ln(data.sel(z=z), quantities[i], x=x2, Ln_range=Ln_range2, shot=shot, time=bin, bin=bin, plot=False)
                                lns.append(ln2)
                                integrated_psds.append(psd2)
                                #ax.plot(ln2, psd2, marker='o')
                                mean_langmuir, std_langmuir = get_langmuir_profiles(langmuir_dataset, param, z,
                                                                                    x=Ln_range2, time=bin, shot=shot,
                                                                                    plot=False)
                                # color_val = float(mean_langmuir.mean(dim="x"))
                                xxx, xxxx = Ln_range2
                                color_val = abs((mean_langmuir.values[-1] - mean_langmuir.values[0]) / (xxxx - xxx))
                                if ln2 < 25 and ln2_err < ln2:
                                    ax.errorbar(ln2, psd2, psd2_err, ln2_err, marker='>', markerfacecolor=cmap(norm(color_val)),
                                            markeredgecolor="black", ecolor="black", capsize=1.5, elinewidth=0.5, capthick=0.5)
                            except:
                                pass
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(r'$|\nabla T_e|$ [eV/cm]')
            ax.set_xlabel('$L_n$ [cm]')
            ax.set_ylabel(f'normalized {quantities[i]} fluctuations') #[$\text{cm}^{-3}$]')
            #ax.plot(lns, integrated_psds, marker='o', linestyle='', color='black')
            fig.tight_layout()
            fig.show()
            fig.savefig(plot_save_folder + data[quantities[i]].name + '_' + "fluctuations_vs_Ln_"+ get_time() + '.png',
                        dpi=150)

    if ask_yes_or_no("Plot radial amplitude spectrogram (y/n)?"):
        x = get_plotting_params("x")
        bin = get_plotting_params("bin")
        shot = get_plotting_params("shot")
        for i in choice_indices:
            j = 0
            for data in data_list:
                for z in data.coords["z"].values:
                    get_radial_spectrogram(data[quantities[i]].sel(z=z), x=x, bin=bin, shot=shot, z=z, plot=True,
                                       plot_save_folder=plot_save_folder,
                                           filename = filenames[j])
                j += 1

def get_plotting_params(parameter_as_string):
    """
    Auxiliary function to `ask_about_plots`. Facilitates the user's selection of
    coordinates against which to plot the data.

    Parameters
    ----------
    parameter_as_string : `string`
        The axis along which the coordinates are to be selected.

    """
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
