import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

def find_ion_current(sorted_bias,sorted_current, v_f_idx):
    '''
    Parameters
    ----------
    sorted_bias - Quantity (V) Langmuir probe obtained bias sorted from lowest to highest
    sorted_current - Quantity (A) Langmuir probe obtained current sorted from lowest to highest corresponding bias voltage
    v_f_idx - Index within sorted bias and sorted current of the floating potential

    Returns
    -------
    ion_current - Quantity (A) Array of the ion current piecewise from a fitted section. Once the fitted section
    crosses 0 the ion current is assumed to be 0
    '''

    # Index the part of the sweep that should be mostly ions
    end_idx = int(v_f_idx * 0.5)

    # Index the bias and current that the data will be fit to
    idxed_bias = sorted_bias[0 : end_idx + 1].value
    idxed_current = sorted_current[0 : end_idx + 1].value

    # Fit a line to the indexed bias and current
    slope, intercept = np.polyfit(idxed_bias, idxed_current, 1)

    # Create an array of ion current values
    full_slope_current = slope * sorted_bias.value + intercept

    # Generate the full ion current array -- Only take values less than 0, values greater than 0 are taken to be 0
    ion_current = np.zeros(len(full_slope_current)) * u.A
    for i in range(len(full_slope_current)):
        if full_slope_current[i] <= 0:
            ion_current[i] = full_slope_current[i] * u.A

    # fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    # ax = ax.flatten()
    #
    # ax[0].plot(sorted_bias.value, ion_current.value, label='ion current')
    # ax[1].plot(sorted_bias.value, sorted_current.value - ion_current.value,marker = '.', color = 'r', label='original - ion')
    # ax[1].plot(sorted_bias.value, sorted_current.value, marker= '.', color = 'b', label='original')
    # plt.show()
    # plt.close()

    return ion_current

def electron_temperature_max(unique_bias,unique_adj_current, dlnIdV, dlnI2dV2,v_f_index, log_mask, filename):
    '''

    Parameters
    ----------
    unique_bias - Unique Bias values from the langmuir probe sweep
    unique_adj_current - Unique ln current values from the langmuir probe sweep
    dlnIdV - derivative of the natural log of current curve
    dlnI2dV -  2nd derivative of the natural log of current curve
    log_mask - logical mask applied to the derivative curve

    Returns
    -------
    slope - Slope of the line of best fit of the linear region of the logarithmic curve -> Corresponds to 1/t_e in eV
    intercept - Intercept of the line of best fit of the linear region of the logarithmic curve
    '''

    # Ensure that all parameters are compatible for the analysis
    max_idxs =[]
    unique_bias = unique_bias[log_mask]
    unique_adj_current = unique_adj_current[log_mask]
    dlnIdV = dlnIdV[log_mask]
    dlnI2dV2 = dlnI2dV2[log_mask]

    # We only care about what is going on in the region past the floating potential
    dlnI2dV2_search = dlnI2dV2[v_f_index:]

    # Search for all the maxima in the range to be searched
    for i in range(len(dlnI2dV2_search) - 1):
        if ((np.sign(dlnI2dV2_search[i]) == 1
             and np.sign(dlnI2dV2_search[i + 1]) == -1)
                or np.sign(dlnI2dV2_search[i]) == 0):
            max_idxs.append(i + v_f_index)

    max_idxs = np.array(max_idxs)

    # Set a minimum value the maximum must have to be considered a valid index
    valid_guesses_mask = dlnIdV[max_idxs] > (np.max(dlnIdV[max_idxs]) * .7)

    # Choose the very last max index to be chosen
    v_max_idx = max_idxs[valid_guesses_mask][-1]
    v_max = unique_bias[v_max_idx]

    # Search for where the minimum is to the left of the maximum that we are looking at is by the opposite method of how
    # we found the maxima
    try:
        min_idxs = []
        for j in range(len(dlnI2dV2[:v_max_idx]) - 1):
            if ((np.sign(dlnI2dV2[j]) == -1
                 and np.sign(dlnI2dV2[j + 1]) == 1)
                    or np.sign(dlnI2dV2[j]) == 0):
                min_idxs.append(j)

        min_idx = min_idxs[-1]

        # Look for where the points where the curve is above tol % of the maximum I found 70% to be about right.
        # Too much past this and the data starts to curve
        tol = 0.7
        left_edge = np.where(dlnIdV[min_idx:] >= tol * dlnIdV[v_max_idx])[0][0] + min_idx
        right_edge = np.where(dlnIdV[min_idx:] >= tol * dlnIdV[v_max_idx])[0][-1] + min_idx

        # Sometimes there is only 4-5 data points within the original tolerance range so this incrementally increases
        # the tolerance to try and obtain a better fit
        while right_edge - left_edge <= 8:
            tol = tol - 0.01
            left_edge = np.where(dlnIdV[min_idx:] >= tol * dlnIdV[v_max_idx])[0][0] + min_idx
            right_edge = np.where(dlnIdV[min_idx:] >= tol * dlnIdV[v_max_idx])[0][-1] + min_idx

    # If there is no minimum detected just start with the tolerance approximations of the linear region
    except IndexError:
        tol = 0.7
        left_edge = np.where(dlnIdV[:v_max_idx] <= (tol * dlnIdV[v_max_idx]))[0][-1]
        right_edge = np.where(dlnIdV[v_max_idx:] <= tol * dlnIdV[v_max_idx])[0][0] + v_max_idx

        while right_edge - left_edge <= 8:
            tol = tol - 0.01
            left_edge = np.where(dlnIdV[:v_max_idx] >= tol * dlnIdV[v_max_idx])[0][0]
            right_edge = np.where(dlnIdV[v_max_idx:] >= tol * dlnIdV[v_max_idx])[0][-1] + v_max_idx

    # Fit the region with a line and return the slope and intercept
    slope, intercept = np.polyfit(unique_bias[left_edge:right_edge],
                                   unique_adj_current[left_edge:right_edge], 1)


    #
    # Create plots
    fig, ax = plt.subplots(2,2, figsize = (8,8))
    ax = ax.flatten()
    ax[0].plot(unique_bias, unique_adj_current, marker=".", color='b', linestyle='None', label= 'Original data')
    ax[0].plot(unique_bias, slope * unique_bias.value + intercept, linestyle="--", color='r', label = 'Temperature Fit')
    ax[0].plot(unique_bias[left_edge], unique_adj_current[left_edge], marker=".", color='y', label = 'Left edge of fit')
    ax[0].plot(unique_bias[right_edge], unique_adj_current[right_edge], marker=".", color='y', label = 'Right edge of fit')
    ax[0].plot(unique_bias[v_max_idx], unique_adj_current[v_max_idx], marker=".", color='g', label = 'Maximum index')
    ax[0].set_title('log plot ' + filename)
    ax[0].set_xlabel(r'Voltage (V)')
    ax[0].set_ylabel(r'$\text{ln}(I)$')
    ax[0].legend(loc='lower right')

    ax[1].plot(unique_bias, dlnIdV, marker=".", color='b', label = 'Derivative of spline fit')
    ax[1].plot(unique_bias[v_max_idx], dlnIdV[v_max_idx], marker=".", color='m', label = 'Maximum index')
    ax[1].axvline(x=unique_bias[left_edge].value, color='g', label = 'Left edge of fit')
    ax[1].axvline(x=unique_bias[right_edge].value, color='y', label = 'Right edge of fit')
    ax[1].set_xlabel('Voltage (V)')
    ax[1].set_ylabel(r'$\frac{\text{dln}(I)}{\text{d}V}$')
    ax[1].set_title('log plot derivative ' + filename)
    ax[1].legend(loc='lower right')

    ax[2].plot(unique_bias[left_edge:right_edge], unique_adj_current[left_edge:right_edge], marker=".", color='b',
               linestyle = 'None',label = 'Original Data')
    ax[2].plot(unique_bias[left_edge:right_edge], slope * unique_bias[left_edge:right_edge].value + intercept,
               linestyle="--", color='r', label = 'Temperature fit')
    ax[2].plot(unique_bias[v_max_idx], unique_adj_current[v_max_idx], marker=".", color='g', label = 'Maximum index')
    ax[2].legend(loc='lower right')
    ax[2].set_xlabel('Voltage (V)')
    ax[2].set_ylabel(r'$\text{ln}(I)$')
    ax[2].set_title('Zoomed log plot ' + filename)

    plt.tight_layout()
    plt.show()


    return slope, intercept











