import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
import astropy.constants as const
from scipy.interpolate import UnivariateSpline
from lapd_plasma_analysis.PINQUED_Functions.Curve_fitting import *
from lapd_plasma_analysis.PINQUED_Functions.PINQUED_Auxillary_functions import *
from lapd_plasma_analysis.file_access import ensure_directory

full_folder = "/Users/lukec/Downloads/Pinqued_csv_Jan_26/"
csv_folder = "/Users/lukec/Downloads/Pinqued_csv_Jan_26/data-lprobe-2025-10-21/"
output_file = 'luke_calculations'

# Choose whether to include or not include the ion current
w_ion_current = False
if w_ion_current:
    output_file = output_file + "_w_ion_current"
if not w_ion_current:
    output_file = output_file + "_wo_ion_current"
assert csv_folder.endswith("/")

probe_tip_diameter = 0.0008 * u.m # mm
probe_tip_length = 0.0144 * u.m # mm

probe_circumference = np.pi * probe_tip_diameter
probe_area = probe_circumference * probe_tip_length

data = []
for file in os.listdir(csv_folder):
    if file.endswith(".csv"): # and file == os.listdir(csv_folder)[0]:
        # Look for and read .csv files in the csv folder
        filename = "-".join(file.split("-")[2:])
        filename = filename.split(".")[0]

        df = pd.read_csv(csv_folder + file, usecols=['Voltage (V)','Current (A)'])
        bias = df['Voltage (V)'].to_numpy() * u.V
        current = df['Current (A)'].to_numpy() * u.A

        # Make sure current and bias are sorted in order of ascending bias values
        sorted_b_indices = np.argsort(bias.value)
        sorted_bias = bias[sorted_b_indices]
        sorted_current = current[sorted_b_indices]

        # Get the floating potential at the bias where the current crosses 0
        v_f_bias, _, v_f_index = p_get_floating_potential(sorted_bias, sorted_current)
        plt.plot(sorted_bias, sorted_current, color = 'steelblue', linestyle = 'None', marker = '.')
        plt.axvline(v_f_bias.value, color = 'k', linestyle = '--')
        plt.axhline(0, color = 'k', linestyle = '--')
        plt.xlabel('Bias (V)')
        plt.ylabel('Current (A)')
        plt.title(filename)

        plt.tight_layout()
        directory_name = 'individual_sweeps/'
        ensure_directory(full_folder + directory_name)
        plt.savefig(full_folder + directory_name + filename + ".svg")
        print('plot saved to: ', full_folder + directory_name + filename + ".svg")
        plt.show()


        # Find the ion current by looking at the first 50% of data points before the floating potential and fitting a
        # line
        ion_current = find_ion_current(sorted_bias, sorted_current, v_f_index)

        # Option to include or not include ion current in calculations

        if not w_ion_current:
            sorted_current = sorted_current - ion_current

        # Remove all values (and corresponding current values) where there is a repeated value for bias
        _, unique_b_mask = np.unique(sorted_bias,return_index=True)
        unique_bias = sorted_bias[unique_b_mask]
        unique_current = sorted_current[unique_b_mask]


        # Create a spline fit for the data to plot on top of the raw data
        spline = UnivariateSpline(unique_bias, unique_current, s = 1e-12, k=3)
        current_fit = spline(unique_bias)

        # Take the first derivative of the spline fit
        spline_deriv = spline.derivative(n=1)
        dIdV = spline_deriv(unique_bias)


        # Take the second derivative of the spline fit
        spline_2_deriv = spline.derivative(n=2)
        dI2dV2 = spline_2_deriv(unique_bias)

        # Find Plasma Potential from the maximum of the derivative curve - Chen method (Pace says look at E-sat region
        # and draw a line through that and the region where we are calculating the T_e and where they intersect is the
        # plasma potential
        max_idxs = []
        for i in range(len(dI2dV2) - 1):
            if (np.sign(dI2dV2[i]) == 1 and np.sign(dI2dV2[i + 1]) == -1) or np.sign(dI2dV2[i]) == 0:
                max_idxs.append(i)
        max_idxs = np.array(max_idxs)
        valid_guesses_mask = dIdV[max_idxs] > (np.max(dIdV[max_idxs]) * .7)
        v_p_idx = max_idxs[valid_guesses_mask][0]
        v_p = unique_bias[v_p_idx]
        s_b_v_p_idx = np.where(v_p == sorted_bias)[0][0]
        u_b_v_f_idx = np.where(v_f_bias <= unique_bias)[0][0]

        # Strip the units off of the current array and take the logarithm
        val_sorted_current = sorted_current.to(u.A).value.astype(np.float64)
        t_e_offset = abs(min(val_sorted_current)) + 1e-9
        current_to_adj = val_sorted_current + t_e_offset
        adjusted_current = np.log(current_to_adj)

        # Make the adj current compatible with the unique bias
        unique_adj_current = adjusted_current[unique_b_mask]

        # Create a spline for the log data
        log_spline = UnivariateSpline(unique_bias, unique_adj_current, s=1e-3, k=3)
        log_current_fit = log_spline(unique_bias)

        # First derivative of the spline of the log data
        log_spline_deriv = log_spline.derivative(n=1)
        dlnIdV = log_spline_deriv(unique_bias)

        # Second derivative of the spline of the log data
        log_spline_2_deriv = log_spline.derivative(n=2)
        dlnI2dV2 = log_spline_2_deriv(unique_bias)

        # Because everything of interest is going to be within a narrow range, cut down the viewing window to make the
        # data easier to see
        log_mask = ((unique_bias.value >= -10) & (unique_bias.value <= 10))
        log_b_plot = unique_bias[log_mask]
        dlnIdV_plot = dlnIdV[log_mask]

        # Calculate the electron temperature
        slope, t_e_intercept = electron_temperature_max(unique_bias, unique_adj_current, dlnIdV, dlnI2dV2,
                                               u_b_v_f_idx, log_mask, filename)
        t_e = 1 / slope * u.eV

        if not w_ion_current:
            sorted_current = sorted_current + ion_current

        # Obtain the ion saturation current an average of the first few bias values
        ion_isat, _ = p_get_ion_isat_min(sorted_current, sorted_bias)

        # Get ion density - Electron Density is the same as Ion density because of quasineutrality
        n_i = get_ion_density("Ar+", ion_isat, probe_area, t_e)

        # Debye Length
        lamba_D = ((const.eps0.value * t_e.value)/(n_i.value * const.e.si.value)) ** 0.5 * u.m

        # Chi
        chi = probe_tip_diameter/2 * 1/lamba_D

        # # Create Plots
        # fig, ax = plt.subplots(1,2,figsize = (8,4))
        # ax = ax.flatten()
        #
        # ax[0].plot(unique_bias,current_fit,'-',color='m')
        # ax[0].scatter(sorted_bias, sorted_current)
        # ax[0].plot(unique_bias[v_p_idx], unique_current[v_p_idx], label='Plasma Potential',marker = 'o',
        #            color = 'red')
        # ax[0].plot(sorted_bias[v_f_index], sorted_current[v_f_index], label = 'Floating Potential', marker = 'o',
        #            color = 'yellow')
        # ax[0].set_xlabel('Voltage (V)')
        # ax[0].set_ylabel('Current (A)')
        # ax[0].legend(loc = 'upper left')
        #
        # ax[1].plot(unique_bias, dIdV, '-', color='m')
        # ax[1].plot(unique_bias[v_p_idx], dIdV[v_p_idx], label='Plasma Potential', marker='o',
        #            color='red')
        # ax[1].set_xlabel('Voltage (V)')
        # ax[1].set_ylabel(r'$\frac{\text{d}I}{\text{d}V}$ ($\frac{\text{A}}{\text{V}}$)')
        # ax[1].legend(loc='lower right')

        # # Show Electron Temperature Calculation
        # ax[2].plot(sorted_bias[v_f_index:s_b_v_p_idx + 1], adjusted_current[v_f_index:s_b_v_p_idx + 1], '.', color='b')
        # ax[2].plot(sorted_bias[s_b_v_p_idx], adjusted_current[s_b_v_p_idx], label='Plasma Potential', marker='o',
        #            color='red')
        # ax[2].plot(sorted_bias[v_f_index], adjusted_current[v_f_index], label='Floating Potential', marker='o',
        #            color='yellow')
        # ax[2].plot(sorted_bias[v_f_index:s_b_v_p_idx + 1],
        #            1/t_e.value * sorted_bias[v_f_index:s_b_v_p_idx + 1].value + t_e_intercept + t_e_offset,
        #            '-', color='m', label = r"$\frac{{1}}{{T_e}} + intecept$")
        # ax[2].set_xlabel('Voltage (V)')
        # ax[2].set_ylabel(f'ln(Current + {t_e_offset})')
        # ax[2].legend(loc = 'upper left')
        #
        # ax[3].plot(sorted_bias, adjusted_current,'.', color='blue')
        # ax[3].plot(sorted_bias[s_b_v_p_idx], adjusted_current[s_b_v_p_idx], label='Plasma Potential', marker='o',
        #            color='red')
        # ax[3].plot(sorted_bias[v_f_index], adjusted_current[v_f_index], label='Floating Potential', marker='o',
        #            color='yellow')
        # ax[3].plot(sorted_bias,
        #            1 / t_e.value * sorted_bias.value + t_e_intercept + t_e_offset,
        #            '-', color='m', label=r"$\frac{{1}}{{T_e}} + intecept$")
        # ax[3].set_xlabel('Voltage (V)')
        # ax[3].set_ylabel(f'ln(Current + {t_e_offset})')
        # ax[3].legend(loc='upper left')
        #
        # fig.suptitle(filename)
        # fig.tight_layout()
        #
        # plt.show()
        #
        # fig2, ax2 = plt.subplots(1, 2, figsize=(8, 4))
        # ax = ax.flatten()
        #
        # ax2[0].plot(sorted_bias, adjusted_current,'.', color='blue')
        # ax2[0].plot(unique_bias, log_current_fit, '-', color='m')
        # ax2[0].plot(sorted_bias[s_b_v_p_idx], adjusted_current[s_b_v_p_idx], label='Plasma Potential', marker='o',
        #            color='red')
        # ax2[0].plot(sorted_bias[v_f_index], adjusted_current[v_f_index], label='Floating Potential', marker='o',
        #            color='yellow')
        #
        #
        # ax2[1].plot(log_b_plot, dlnIdV_plot,'-',color='m')
        # # ax[4].plot(unique_bias[v_p_idx], dlnIdV[v_p_idx], label='Plasma Potential', marker='o',
        # #            color='red')
        # # ax[4].plot(unique_bias[u_b_v_f_idx], dlnIdV[u_b_v_f_idx], label='Floating Potential', marker='o',
        # #            color='yellow')
        # ax[1].set_xlabel('Voltage (V)')
        # ax[1].set_ylabel(r'$\frac{\text{d ln}(I)}{\text{d}V}$ ($\frac{\text{A}}{\text{V}}$)')
        # ax[1].legend(loc='best')




        data.append({
            'filename' : filename,
            'Electron Density (1/m^3)': n_i.value,
            'Electron Temperature (eV)': t_e.value,
            'V_p (V)' : v_p.value,
            'V_f (V)' : sorted_bias[v_f_index].value,
            'Debye length (mm)' : lamba_D.value * 1000,
            'chi (dimensionless)': chi
        })

summary_df = pd.DataFrame(data)
summary_df.to_csv(os.path.join(full_folder, output_file),index = False)
print('CSV saved to: ', os.path.join(full_folder, output_file))
plt.close('all')

