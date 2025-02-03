import sys
import warnings
from tqdm import tqdm

from plasmapy.formulary.collisions import Coulomb_logarithm
from plasmapy.formulary.collisions.frequencies import MaxwellianCollisionFrequencies

from lapd_plasma_analysis.langmuir.helper import *


def langmuir_diagnostics(characteristic_arrays, positions, ramp_times, langmuir_configs, ion_type, bimaxwellian=False):
    """
    Performs plasma diagnostics on a DataArray of Characteristic objects and returns the diagnostics as a Dataset.

    Parameters
    ----------
    characteristic_arrays : `numpy.ndarray` of `plasmapy.diagnostics.langmuir.Characteristic`
        4D NumPy array of Characteristics (dims: isweep, position #, shot, plateau) (WIP)
    positions : `list` of coordinates for position of each shot
        (WIP)
    ramp_times : `astropy.units.Quantity`
        Time-based Quantities corresponding to time of each shot (peak vsweep) (WIP)
    langmuir_configs
        (WIP)
    ion_type : `string`
        String corresponding to a PlasmaPy Particle name.
    bimaxwellian : `boolean`
        Specifies whether to assume a bimaxwellian plasma during plasmapy Langmuir analysis.

    Returns
    -------
    `xarray.Dataset`
        Dataset containing diagnostic values at each position
    """

    x = np.unique(positions[:, 0])
    y = np.unique(positions[:, 1])
    ports = np.unique(langmuir_configs['port'])
    faces = np.unique(langmuir_configs['face'])
    port_zs = np.array([portnum_to_z(port).to(u.cm).value for port in ports])

    num_probe = len(ports)
    num_face = len(faces)
    num_x = len(x)
    num_y = len(y)
    num_plateaus = characteristic_arrays.shape[-1]
    num_shots = characteristic_arrays.shape[2]

    probe_areas = langmuir_configs['area']
    keys_units = get_diagnostic_keys_units(probe_areas[0], ion_type, bimaxwellian=bimaxwellian)

    # num_probe * num_face * num_x * num_y * num_shots * num_plateaus template numpy_array
    templates = {key: np.full(shape=(num_probe, num_face, num_x, num_y, num_shots, num_plateaus),
                              fill_value=np.nan, dtype=float)
                 for key in keys_units.keys()}
    diagnostics_ds = xr.Dataset({key: xr.DataArray(data=templates[key],
                                                   dims=['probe', 'face', 'x', 'y', 'shot', 'time'],
                                                   coords=(('probe', np.arange(num_probe)),
                                                           ('face', faces),
                                                           ('x', x, {"units": str(u.cm)}),
                                                           ('y', y, {"units": str(u.cm)}),
                                                           ('shot', np.arange(num_shots)),
                                                           ('time', ramp_times.to(u.ms).value, {"units": str(u.ms)}))
                                                   ).assign_coords({'plateau': ('time', np.arange(num_plateaus) + 1),
                                                                    'port': ('probe', ports),
                                                                    'z':    ('probe', port_zs, {"units": str(u.cm)})}
                                                                   ).assign_attrs({"units": keys_units[key]})
                                 for key in keys_units.keys()})

    num_characteristics = (diagnostics_ds.sizes['probe'] * diagnostics_ds.sizes['face'] * diagnostics_ds.sizes['x']
                           * diagnostics_ds.sizes['y'] * diagnostics_ds.sizes['shot'] * diagnostics_ds.sizes['time'])

    """
    error_types = []
    error_chart = np.zeros(shape=(num_probe, num_face, num_x, num_y, num_shots, num_plateaus))
    # """

    warnings.simplefilter(action='ignore')  # Suppress warnings to not break progress bar
    with tqdm(total=num_characteristics, unit="characteristic", file=sys.stdout) as pbar:
        for i in range(characteristic_arrays.shape[0]):  # isweep
            for l in range(characteristic_arrays.shape[1]):  # location  # noqa
                for s in range(characteristic_arrays.shape[2]):  # shot
                    for r in range(characteristic_arrays.shape[3]):  # ramp
                        characteristic = characteristic_arrays[i, l, s, r]
                        diagnostics = diagnose_char(characteristic, probe_areas[i], ion_type,
                                                    bimaxwellian=bimaxwellian)
                        pbar.update(1)
                        if isinstance(diagnostics, str):  # error with diagnostics
                            """
                            if diagnostics not in error_types:
                                error_types.append(diagnostics)
                            error_chart[i,
                                        np.where(x == positions[l, 0])[0][0],
                                        np.where(y == positions[l, 1])[0][0], s, r] = error_types.index(diagnostics) + 1
                            """
                            continue
                        if bimaxwellian:
                            diagnostics = unpack_bimaxwellian(diagnostics)
                        for key in diagnostics.keys():
                            val = value_safe(diagnostics[key])
                            diagnostics_ds[key].loc[{"probe": np.where(ports == langmuir_configs['port'][i])[0][0],
                                                     "face": langmuir_configs['face'][i],
                                                     "x": positions[l, 0],  "y": positions[l, 1],
                                                     "shot": s,             "time": ramp_times[r]}] = val

    warnings.simplefilter(action='default')  # Restore warnings to default handling

    """
    # Leo debug below: display plots showing types of errors
    show_error_plot = False
    if show_error_plot:
        for s in range(error_chart.shape[3]):
            ax = plt.subplot()
            im = ax.imshow(error_chart[0, :, 0, s, :],
                           extent=(positions[:, 0].min(), positions[:, 0].max(),
                                   ramp_times.value.min(), ramp_times.value.max()),
                           origin="lower")
            cbar = plt.colorbar(im)
            cbar.set_ticks(list(np.arange(len(error_types) + 1)))
            cbar.set_ticklabels(["No error"] + [error_types[t] for t in np.arange(len(error_types))])
            plt.title(f"Error types (s = {s})")
            plt.tight_layout()
            plt.show()
    # """

    return diagnostics_ds


def filter_characteristic(characteristic) -> bool:
    bias = characteristic.bias
    current = characteristic.current

    # only accept characteristic if floating potential is below 0 V
    return np.max(bias[current < 0]) < 0 * u.V


def get_pressure(density, temperature):
    """
    Calculate pressure from temperature and density values of choice.

    Parameters
    ----------
    density : `astropy.units.Quantity`
        Quantity representing species number density.
    temperature : `astropy.units.Quantity`
        Quantity representing species temperature.

    Returns
    -------
    `astropy.units.Quantity`
        Quantity representing thermal/fluid pressure.
    """
    pressure_unit = u.Pa
    pressure = 1 * temperature * density * (1. * u.eV * u.m ** -3).to(pressure_unit)  # 3 / 2 replaced by 1
    return pressure.assign_attrs({'units': str(pressure_unit)})


def get_electron_ion_collision_frequencies(lang_ds: xr.Dataset, ion_type="He-4+"):
    T_e = (lang_ds['T_e_avg'] if 'T_e_avg' in lang_ds else lang_ds['T_e']).data * u.eV  # noqa
    n_e = (lang_ds['n_e_cal'] if not np.isnan(lang_ds['n_e_cal']).all() else lang_ds['n_e']).data * u.m ** -3
    coulomb_logarithm = Coulomb_logarithm(T_e, n_e, ('e-', ion_type), z_mean=0.5, method="hls_full_interp")
    electron_ion_collision_frequencies = MaxwellianCollisionFrequencies(
        "e-",
        ion_type,
        v_drift=lang_ds['n_i_OML'].data * 0 * u.m / u.s,
        n_a=n_e,
        T_a=T_e,
        n_b=lang_ds['n_i_OML'].data * u.m ** -3,  # check this too
        T_b=0 * u.eV,
        Coulomb_log=coulomb_logarithm * u.dimensionless_unscaled
    ).Maxwellian_avg_ei_collision_freq
    return electron_ion_collision_frequencies


def detect_steady_state_times(langmuir_dataset: xr.Dataset, core_rad):
    """
    Return start and end times for the steady-state period (where density is ~constant in time) (WIP).

    Parameters
    ----------
    langmuir_dataset : `xarray.Dataset`
        (WIP)
    core_rad : `astropy.units.Quantity`
        (WIP)

    Returns
    -------
    `astropy.units.Quantity`
        Two-element Quantity representing estimate beginning and end times of steady-state period, inclusive.
    """

    # TODO very hardcoded!
    exp_name = langmuir_dataset.attrs['Exp name']
    if "january" in exp_name.lower():
        return [16, 24] * u.ms
    else:
        density = langmuir_dataset['n_i_OML']
        core_density_vs_time = core_steady_state(density.isel(probe=0, face=0), core_rad=core_rad,
                                                 operation="median", dims_to_keep=["time"])
        threshold = 0.9 * core_density_vs_time.max()
        start_time = core_density_vs_time.time[core_density_vs_time > threshold].idxmin().item()
        end_time = core_density_vs_time.time[core_density_vs_time > threshold].idxmax().item()

        time_unit = u.Unit(langmuir_dataset.coords['time'].attrs['units'])
        return (start_time, end_time) * time_unit


def diagnose_char(characteristic, probe_area, ion_type, bimaxwellian, indices=None):
    """ Use swept_probe_analysis function to extract diagnostic data from Characteristic, or return error string. """
    if characteristic is None:
        return "characteristic is None"
    try:
        diagnostics = swept_probe_analysis(characteristic, probe_area, ion_type, bimaxwellian=bimaxwellian)
    except (ValueError, TypeError, RuntimeError) as e:
        diagnostics = str(e)
    return diagnostics


def debug_char(characteristic, error_str, *pos):
    """ A debug function to plot plateaus that cause errors. """
    # TODO update the below core-checker
    # if 5 < pos[-1] < 13 and 0.3 < pos[-2] / 71 < 0.7:  # max(characteristic.current) > 20 * u.mA
    #      pass
    #      tqdm.write("Plateau at position " + str(pos) + " is unusable")
    characteristic.plot()
    plt.title(f"{pos}\n{error_str}")
    plt.show()


def crop_value(diagnostic, minimum, maximum):
    """ Discard diagnostic values (e.g. T_e) outside a specified range. """
    value = value_safe(diagnostic)
    return value if minimum <= value <= maximum else np.nan
