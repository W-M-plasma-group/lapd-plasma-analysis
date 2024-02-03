from helper import *

import warnings
import bottleneck as bn
from scipy.signal import find_peaks
from tqdm import tqdm
import sys


def characterize_sweep_array(bias, currents, dt):
    r"""
    Function that processes bias and current data into a DataArray of distinct Characteristics.
    Takes in bias and current arrays, smooths them, divides them into separate ramp sections, 
    and creates a Characteristic object for each ramp at each unique x,y position.

    Parameters
    ----------
    :param bias: array, units of voltage
    :param currents: array, units of current
    :param dt: float, units of time
    :return: 2D array of Characteristic objects by shot number and plateau number
    """

    bias, currents = ensure_sweep_units(bias, currents)

    # trim bad, distorted averaged ends in isolated plateaus
    ramp_bounds = isolate_plateaus(bias)

    ramp_times = ramp_bounds[:, 1] * dt.to(u.ms)
    # NOTE: MATLAB code stores peak voltage time (end of plateaus), then only uses plateau times for very first position
    # This uses the time of the peak voltage for the average of all shots ("top of the average ramp")

    return characteristic_array(bias, currents, ramp_bounds), ramp_times


def smooth_array(raw_array, margin: int, method: str = "median") -> np.ndarray:
    r"""
    Smooth an array using a moving mean or median applied over a window.
    :param raw_array:
    :param margin:
    :param method:
    :return:
    """

    array = raw_array.copy()
    if margin > 0:
        if method == "mean":
            smooth_func = bn.move_mean
        elif method == "median":
            smooth_func = bn.move_median
        else:
            raise ValueError(f"Invalid smoothing method {repr(method)}; 'mean' or 'median' expected")
        array = smooth_func(array, window=margin)
    return array


def isolate_plateaus(bias, margin=0):
    r"""
    Find indices corresponding to the beginning and end of every bias ramp.

    Parameters
    ----------
    :param bias:
    :param margin:
    :return: num_plateaus-by-2 array; start indices in first column, end indices in second
    """

    # Assume strictly that all plateaus start and end at the same time after the start of the shot as in any other shot
    bias_axes_to_average = tuple(np.arange(bias.ndim)[:-1])
    bias_avg = np.mean(bias, axis=bias_axes_to_average)  # mean of bias across all positions and shots, preserving time

    # Report on how dissimilar the vsweep biases are and if they can be averaged together safely?

    # Initial fit to guess number of peaks
    min_plateau_width = 500  # change as necessary
    guess_num_plateaus = len(find_peaks(bias_avg, height=0, distance=min_plateau_width)[0])
    guess_plateau_spacing = bias.shape[-1] // guess_num_plateaus

    # Second fit to find maximum bias frames
    peak_frames, peak_properties = find_peaks(bias_avg, height=0, distance=guess_plateau_spacing // 2,
                                              width=min_plateau_width, rel_height=0.97)  # TODO 0.97 may be hardcoded

    return np.stack((peak_properties['left_ips'].astype(int) + margin // 2, peak_frames - margin // 2), axis=-1)


def ensure_sweep_units(bias, current):
    try:
        if bias.unit.is_equivalent(u.V):
            new_bias = bias.to(u.V)
        else:
            raise ValueError(f"Probe bias has units of {bias.unit} when units convertible to Volts were expected.")
    except AttributeError:
        warnings.warn("Input bias array is missing explicit units. Assuming units of Volts.")
        new_bias = bias * u.V
    try:
        if current.unit.is_equivalent(u.A):
            new_current = current.to(u.A)
        else:
            raise ValueError(f"Probe current has units of {current.unit} when units convertible to Amps were expected.")
    except AttributeError:
        warnings.warn("Input current array is missing explicit units. Assuming units of Amps.")
        new_current = current * u.A
    return new_bias, new_current


def characteristic_array(bias, currents, ramp_bounds):
    # 4D: num_isweep * unique_position * shot * plateau_num
    # "currents" has "isweep" dimension in front; may have size 1

    ramp_slices = np.array([slice(ramp[0], ramp[1]) for ramp in ramp_bounds])

    num_isweep = currents.shape[0]
    num_loc = currents.shape[1]
    num_shot = currents.shape[2]
    num_ramp = len(ramp_slices)

    print(f"Creating characteristics ...")
    warnings.simplefilter(action='ignore', category=FutureWarning)  # Suppress FutureWarnings to not break loading bar
    print("\t(plasmapy.langmuir.diagnostics pending deprecation FutureWarning suppressed)")

    num_characteristics = num_isweep * num_loc * num_shot * num_ramp
    chara_array = np.empty((len(currents), num_loc, num_shot, len(ramp_slices)), dtype=Characteristic)
    with tqdm(total=num_characteristics, unit="characteristic", file=sys.stdout) as pbar:
        for swp in range(num_isweep):
            for loc in range(num_loc):
                for shot in range(num_shot):
                    for ramp in range(num_ramp):
                        chara_array[swp,  loc, shot, ramp] = Characteristic(
                            bias[loc,          shot, ramp_slices[ramp]],
                            currents[swp, loc, shot, ramp_slices[ramp]])
                        pbar.update(1)

    return chara_array
