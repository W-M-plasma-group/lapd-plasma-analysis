import warnings
import bottleneck as bn
from scipy.signal import find_peaks
from tqdm import tqdm
import sys

from lapd_plasma_analysis.langmuir.helper import *


def make_characteristic_array(bias, current, ramp_bounds):
    """
    Function that processes bias and current data into a DataArray of distinct Characteristics.
    Takes in bias and current arrays, smooths them, divides them into separate ramp sections, 
    and creates a Characteristic object for each ramp at each unique x,y position.

    Parameters
    ----------
    bias : `astropy.units.Quantity`
        Array with units of voltage, representing Langmuir probe voltage over time, with dimensions of
        position, shot, and frame (within a sweep).
    currents : `astropy.units.Quantity`
        Array with units of currents, representing Langmuir probe voltage over time, with dimensions of
        isweep (probe/face combination), position, shot, and frame (within a sweep).
    dt : `astropy.units.Quantity`
        Timestep in units of time between individual "frames" of sweep voltage/current measurement within a single shot.

    Returns
    -------
    `numpy.ndarray`
        4D array of Characteristic objects with isweep, location, shot, and ramp number dimensions.
    """

    bias, currents = ensure_sweep_units(bias, currents)

    # trim bad, distorted averaged ends in isolated plateaus
    ramp_bounds = isolate_plateaus(bias)

    ramp_times = ramp_bounds[:, 1] * dt.to(u.ms)
    # NOTE: MATLAB code stores peak voltage time (end of plateaus), then only uses plateau times for very first position
    # This uses the time of the peak voltage for the average of all shots ("top of the average ramp")

    return characteristic_array(bias, currents, ramp_bounds), ramp_times


def smooth_array(raw_array, margin: int, method: str = "median") -> np.ndarray:
    """
    Smooth an array by a rolling window mean or median (function not currently used).
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
    """
    Find indices corresponding to the beginning and end of every bias ramp.

    Parameters
    ----------
    bias : `astropy.units.Quantity`
        Quantity array containing Langmuir probe applied voltages, with final dimension representing frames (time).
    margin : `int`, default=0
        Rolling window margin width, in number of frames, used to smooth bias and current.

    Returns
    -------
    `numpy.ndarray`
        num_plateaus-by-2 array; start indices in first column, end indices in second
    """

    # Assume strictly that all plateaus start and end at the same time after the start of the shot as in any other shot
    bias_axes_to_average = tuple(np.arange(bias.ndim)[:-1])
    bias_avg = np.mean(bias, axis=bias_axes_to_average)  # mean of bias across all positions and shots, preserving time

    # todo Report on how dissimilar the vsweep biases are and if they can be averaged together safely?

    # Initial fit to guess number of peaks
    min_plateau_width = 500  # change as necessary
    guess_num_plateaus = len(find_peaks(bias_avg, height=0, distance=min_plateau_width)[0])
    guess_plateau_spacing = bias.shape[-1] // guess_num_plateaus

    # Second fit to find maximum bias frames
    peak_frames, peak_properties = find_peaks(bias_avg, height=0, distance=guess_plateau_spacing // 2,
                                              width=min_plateau_width, rel_height=0.97)  # TODO 0.97 is hardcoded

    return np.stack((peak_properties['left_ips'].astype(int) + margin // 2, peak_frames - margin // 2), axis=-1)
