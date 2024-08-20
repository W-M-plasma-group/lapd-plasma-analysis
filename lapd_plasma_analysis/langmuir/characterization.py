"""
(WIP) # TODO
"""
import warnings
import bottleneck as bn
from scipy.signal import find_peaks
from tqdm import tqdm
import sys

from lapd_plasma_analysis.langmuir.helper import *


def make_characteristic_array(bias, current, ramp_bounds):
    """
    Process bias and current data into an array of Characteristic objects, each representing one sweep curve.
    The resulting array is 3D, with dimensions: unique (x, y) position, shot, ramp.
    Takes in a bias and a current array, selects each ramp, and creates a Characteristic object
    for each ramp at each unique x,y position.
    Later concatenated with 3D arrays from other isweeps (other sources of sweep current, e.g.
    faces on a Langmuir probe) to make a larger 4D array to pass to diagnostics.py.
        - Note: a "ramp" is a series of frames in which a Langmuir probe voltage sweep is performed.
          It is called a "ramp" because the applied voltage, instead of remaining at the typical large negative value,
          increases steadily to a large positive value (ramp), then rapidly drops back to negative (quench).
          (WIP explain better # todo)

    Parameters
    ----------
    bias : `astropy.units.Quantity`
        3D array with units of voltage, representing Langmuir probe voltage over time, with dimensions of
        position, shot, and frame (within a sweep).
    current : `astropy.units.Quantity`
        3D array with units of current, representing Langmuir probe voltage over time, with dimensions of
         position, shot, and frame (within a sweep).
    ramp_bounds : `numpy.ndarray` of `int`
        2D array of shape (number of ramps, 2). Ramp start indices are in the first column; end indices in the second.
        Calculated by the `isolate_ramps` function.

    Returns
    -------
    `numpy.ndarray` of `plasmapy.diagnostics.langmuir.Characteristic`
        3D array of Characteristic objects with location, shot, and ramp number dimensions.

    See Also
    --------
    isolate_ramps

    """

    ramp_slices = np.array([slice(ramp[0], ramp[1]) for ramp in ramp_bounds])

    num_loc = current.shape[0]
    num_shot = current.shape[1]
    num_ramp = len(ramp_slices)

    print(f"Creating characteristics ...")
    warnings.simplefilter(action='ignore', category=FutureWarning)  # Suppress FutureWarnings to not break loading bar
    print("\t(plasmapy.langmuir.diagnostics pending deprecation FutureWarning suppressed)")

    num_characteristics = num_loc * num_shot * num_ramp
    chara_array = np.empty((num_loc, num_shot, len(ramp_slices)), dtype=Characteristic)
    with tqdm(total=num_characteristics, unit="characteristic", file=sys.stdout) as pbar:
        for loc in range(num_loc):
            for shot in range(num_shot):
                for ramp in range(num_ramp):
                    chara_array[loc, shot, ramp] = Characteristic(
                        bias[loc,    shot, ramp_slices[ramp]],
                        current[loc, shot, ramp_slices[ramp]])
                    pbar.update(1)

    return chara_array


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


def isolate_ramps(bias, margin=0):
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
    `numpy.ndarray` of `int`
        2D array of shape (number of ramps, 2). Ramp start indices are in the first column; end indices in the second.
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
