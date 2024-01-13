from helper import *

import warnings
import bottleneck as bn
from scipy.signal import find_peaks


def characterize_sweep_array(unsmooth_bias, unsmooth_currents, margin, sample_sec):
    r"""
    Function that processes bias and current data into a DataArray of distinct Characteristics.
    Takes in bias and current arrays, smooths them, divides them into separate ramp sections, 
    and creates a Characteristic object for each ramp at each unique x,y position.

    Parameters
    ----------
    :param unsmooth_bias: array, units of voltage
    :param unsmooth_currents: array, units of current
    :param margin: int, positive
    :param sample_sec: float, units of time
    :return: 2D array of Characteristic objects by shot number and plateau number
    """

    ensure_sweep_units(unsmooth_bias, unsmooth_currents)

    bias = smooth_array(unsmooth_bias,         margin, "median") * u.V
    currents = smooth_array(unsmooth_currents, margin, "median") * u.A

    # trim bad, distorted averaged ends in isolated plateaus
    ramp_bounds = isolate_plateaus(bias, margin=margin)

    ramp_times = ramp_bounds[:, 1] * sample_sec.to(u.ms)
    # NOTE: MATLAB code stores peak voltage time (end of plateaus), then only uses plateau times for very first position
    # This uses the time of the peak voltage for the average of all shots ("top of the average ramp")

    return characteristic_array(bias, currents, ramp_bounds), ramp_times


def smooth_array(raw_array, margin: int, method: str = "mean") -> np.ndarray:
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
                                              width=min_plateau_width, rel_height=0.97)  # 0.97 may be hardcoded

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


def characteristic_array(bias, current, plateau_ranges):
    # 4D: probe * unique_position * shot * plateau_num

    currents = current  # "currents" has "probe" dimension in front; may have size 1
    num_pos = bias.shape[0]
    num_shot = bias.shape[1]

    plateau_slices = np.array([slice(plateau[0], plateau[1]) for plateau in plateau_ranges])

    # Mixed arbitrary/indexed list comprehension
    print(f"Creating characteristics for {currents.shape[0]} probe(s) ...")
    warnings.simplefilter(action='ignore', category=FutureWarning)  # Suppress FutureWarnings to not break loading bar
    print("\t(plasmapy.langmuir.diagnostics pending deprecation FutureWarning suppressed)")
    return np.concatenate([np.array([[[Characteristic(bias[pos, shot, plateau_slice], current[pos, shot, plateau_slice])
                                       for plateau_slice in plateau_slices]
                                      for shot in range(num_shot)]
                                     for pos in range(num_pos)])[np.newaxis, ...]
                           for current in currents])
    # for pos in trange(num_pos, unit="position", file=sys.stdout)])[np.newaxis, ...]
