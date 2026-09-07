from astropy.units import Quantity
from tqdm import tqdm
import sys
import warnings
from plasmapy.utils.exceptions import RelativityWarning
from lapd_plasma_analysis.obtain_plots.Auxillary_functions import *
from lapd_plasma_analysis.plasma_py_comparison import *
from lapd_plasma_analysis.langmuir.find_temperature_and_v_p import get_t_e_breakpoint

def build_xarrays(bias,current,positions,ramp_times,dt,langmuir_configs, ion_type, params_dict):
    """
    Parameters
    ----------
    bias
    current
    positions
    ramp_times
    dt
    langmuir_configs
    ion_type
    params_dict

    Returns
    -------

    """

    # Suppress printing of RelativityWarning globally
    warnings.filterwarnings("ignore", category=RelativityWarning)

    # Suppress poorly conditioned polyfit warnings
    warnings.filterwarnings("ignore", category=np.RankWarning)

    x_vals = np.unique(positions[:, 0])
    y_vals = np.unique(positions[:, 1])
    num_shots = current.shape[2]
    num_sweeps = len(ramp_times) - 1
    # print('Number of shots: ', num_shots)
    # print('Number of sweeps: ', num_sweeps)
    # print('Number of x_vals: ', len(x_vals))
    # print('Number of y_vals: ', len(y_vals))
    # Only the left faces are sweeping - the other two are just collecting data
    sweeping_configs = [cfg for cfg in langmuir_configs if cfg['face'] == 'L']

    # Accounts for non Jan 2024 data we want to get both probes
    if len(sweeping_configs) == 0:
        sweeping_configs = [langmuir_configs[0], langmuir_configs[1]]

    # Only half of the faces actually are conducting sweeps
    num_probes = len(sweeping_configs)
    # print('Number of probes: ', num_probes)

    t_i = 1 * u.eV  # LAPD is held at an ion temperature of 1 eV as was described in Perks and in Leo's thesis -
    # We can't directly measure it yet.

    # How many values are we going to have in each dimension
    shape = (num_probes, len(x_vals), len(y_vals), num_shots, num_sweeps)

    # Fill all of our variable arrays with nans
    n_e = np.full(shape, np.nan)
    n_i = np.full(shape, np.nan)

    t_e = np.full(shape, np.nan)

    v_f = np.full(shape, np.nan)
    v_p = np.full(shape, np.nan)
    nu_ei = np.full(shape, np.nan)

    ion_isat = np.full(shape, np.nan)
    electron_isat = np.full(shape, np.nan)

    p_e = np.full(shape, np.nan)
    p_ei = np.full(shape, np.nan)

    # Define a dictionary of positions with the key as the x,y position and the value as the index in the positions array
    pos_idx = {(x,y): i for i,(x,y) in enumerate(positions)}

    # Initialize values so we don't have to repeatedly call the value function repeatedly in the loop
    dt_ms = dt.to(u.ms).value
    ramp_times_ms = ramp_times.to(u.ms).value
    # print('ramp_times_ms', len(ramp_times_ms)-1)

    # Here we are assuming the sweeps are taken at approximately the same time for all x,y,shot combinations (there are
    # the same number of sweeps for each combination and say sweep 5 always occurs between 4.9 and 5.1 ms for example
    sweep_windows = []
    sweep_times = []

    mid_x = x_vals[len(x_vals) // 2]
    mid_y = y_vals[len(y_vals) // 2]
    mid_shot = num_shots // 2

    key = (mid_x, mid_y)
    if key not in pos_idx:
        raise ValueError("Middle (x,y) position not found")
    loc = pos_idx[key]

    probe_index = 0
    test_bias = bias[probe_index,loc,mid_shot,:]
    test_current = current[probe_index, loc, mid_shot,:]
    time_array = np.arange(len(test_bias)) * dt_ms



    for sweep in range(num_sweeps):
        search_times = ((time_array >= ramp_times_ms[sweep]) &
                        (time_array <= ramp_times_ms[sweep+1]))
        try:
            first_index, last_index = find_sweep_indices(time_array, ramp_times_ms[sweep + 1], search_times, test_bias,
                                                         dt_ms)
            sweep_windows.append((first_index, last_index))
            sweep_times.append(first_index * dt_ms)

        except IndexError:
            sweep_windows.append((None,None))
            sweep_times.append(np.nan)

    # print('Sweep windows: ', len(sweep_windows))

    with tqdm(total = num_probes * len(x_vals) * len(y_vals) * num_shots, desc = 'Position-Shot combinations') as pbar_ls:
        # We want both the index and the actual value for x and y so we use enumerate. (Allows us to index the xarray)
        for probe_index, config in enumerate(sweeping_configs):
            # print('Probe: ', probe_index)
            A_p = config['area']
            port = config['port']
            face = config['face']

            for ix, x in enumerate(x_vals):
                for iy, y in enumerate(y_vals):

                    # Determine the location in the position array where we can find this xy combination
                    key = (x,y)
                    if key not in pos_idx:
                        continue
                    loc = pos_idx[key]

                    for shot in range(num_shots):
                        full_bias = bias[probe_index, loc, shot, :]
                        full_current = current[probe_index, loc, shot, :]

                        for sweep, (first_index, last_index) in enumerate(sweep_windows):

                            # If we can't find the sweep index originally, we skip the sweep. This should be noted as
                            # an Index error

                            if first_index is None:
                                continue

                            start_time = (first_index * dt_ms)
                            end_time = (last_index * dt_ms)
                            mask = ((time_array >= start_time) & (time_array <= end_time))
                            sort_index = np.argsort(full_bias[mask])
                            sorted_bias = full_bias[mask][sort_index]
                            sorted_current = full_current[mask][sort_index]

                            try:
                                v_f_value, _, v_f_index = get_floating_potential(sorted_bias, sorted_current)
                            except Exception as e:
                                v_f_value = np.nan

                            # print('Plasma potential value', v_p_value)
                            try:
                                i_ion_sat_value, _ = get_ion_isat_min(sorted_current, sorted_bias)
                                if i_ion_sat_value > 0:
                                    i_ion_sat_value = np.nan
                            except Exception as e:
                                i_ion_sat_value = np.nan

                            try:
                                if v_f_value is not None:
                                    ion_current = get_ion_current(sorted_bias, sorted_current, v_f_value)
                                else:
                                    ion_current = np.nan
                                breakpoint_dict = get_t_e_breakpoint(sorted_bias, sorted_current, v_f_value, ion_current,
                                                                     batch_mode=True)

                                t_e_value = breakpoint_dict['Te']
                                if t_e_value is not None:
                                    v_p_value = breakpoint_dict['V_p']
                                    i_electron_sat_value = np.nan
                                else:
                                    t_e_value = np.nan
                                    v_p_value = np.nan
                                    i_electron_sat_value = np.nan

                                if np.isnan(i_ion_sat_value):
                                    t_e_value = np.nan
                                    v_p_value = np.nan
                                    i_electron_sat_value = np.nan

                            except Exception as e:
                                t_e_value = np.nan
                                v_p_value = np.nan
                                i_electron_sat_value = np.nan

                            if t_e_value is np.nan:

                                n_e_value = np.nan
                                n_i_value = np.nan

                                nu_ei_value = np.nan

                                p_e_value = np.nan
                                p_ei_value = np.nan
                            else:
                                # if not isinstance(i_electron_sat_value, Quantity):
                                #     n_e_value = np.nan
                                # else:
                                #     try:
                                #         n_e_value = get_electron_density(i_electron_sat_value, A_p, t_e_value)
                                #     except Exception as e:
                                #         n_e_value = np.nan

                                try:
                                    n_i_value = get_ion_density(ion_type, i_ion_sat_value, A_p, t_e_value)
                                    n_e_value = n_i_value
                                except Exception as e:
                                    n_i_value = np.nan
                                    n_e_value = np.nan

                                try:
                                    nu_ei_value = get_electron_ion_collision_frequency(n_e_value, ion_type,
                                                                                           t_e_value)
                                except Exception as e:
                                    nu_ei_value = np.nan

                                try:
                                    p_e_value = l_get_pressure(t_e_value,n_e_value)
                                except Exception as e:
                                    p_e_value = np.nan

                                try:
                                    p_ei_value = l_get_pressure(t_e_value + t_i, n_e_value)
                                except Exception as e:
                                    p_ei_value = np.nan

                            # Assign each of these values to an array index in the xarray
                            t_e[probe_index, ix, iy, shot, sweep] = safe_value(t_e_value)

                            n_e[probe_index, ix, iy, shot, sweep] = safe_value(n_e_value)
                            n_i[probe_index, ix, iy, shot, sweep] = safe_value(n_i_value)

                            v_f[probe_index, ix, iy, shot, sweep] = safe_value(v_f_value)
                            v_p[probe_index, ix, iy, shot, sweep] = safe_value(v_p_value)

                            nu_ei[probe_index, ix, iy, shot, sweep] = safe_value(nu_ei_value)

                            ion_isat[probe_index, ix, iy, shot, sweep] = safe_value(i_ion_sat_value)
                            electron_isat[probe_index, ix, iy, shot, sweep] = safe_value(i_electron_sat_value)
                            # print('ion_isat', safe_value(i_ion_sat_value))
                            # print('electron_isat', safe_value(i_electron_sat_value))

                            p_e[probe_index, ix, iy, shot, sweep] = safe_value(p_e_value)
                            p_ei[probe_index, ix, iy, shot, sweep] = safe_value(p_ei_value)
                            # print(f"probe={probe_index}, x={ix}, y={iy}, shot={shot}, sweep={sweep}")

                        pbar_ls.update(1)

        # Now we create our xarray with these individual variable arrays
        ports = [cfg['port'] for cfg in sweeping_configs]
        faces = [cfg['face'] for cfg in sweeping_configs]
        port_zs = [portnum_to_z(p).value for p in ports]
        gp_voltage_values = []
        for k, v in params_dict.items():
            if k.startswith("GPV"):
                try:
                    gp_voltage_values.append(float(v))
                except (TypeError, ValueError):
                    pass  # skip non-convertible values

        gp_voltage = sum(gp_voltage_values) / len(gp_voltage_values) if gp_voltage_values else None

        # gas = params_dict['Gas type']
        # if 'h' in gas.lower():
        #     ion_type = 'H+'
        # elif 'he' in gas.lower():
        #     ion_type = 'He+'


        ds = xr.Dataset({
            "t_e": (["probe", "x", "y", "shot", "sweep"], t_e, {"units": "eV", "long_name": "Electron Temperature"}),
            "n_e": (["probe", "x", "y", "shot", "sweep"], n_e, {"units": "1/m^3", "long_name": "Electron Density"}),
            "n_i": (["probe", "x", "y", "shot", "sweep"], n_i, {"units": "1/m^3", "long_name": "Ion Density"}),
            "v_f": (["probe", "x", "y", "shot", "sweep"], v_f, {"units": "V", "long_name": "Floating Potential"}),
            "v_p": (["probe", "x", "y", "shot", "sweep"], v_p, {"units": "V", "long_name": "Plasma Potential"}),
            "nu_ei": (["probe", "x", "y", "shot", "sweep"], nu_ei,
                      {"units": "1/s", "long_name": "Electron-Ion Collision Frequency"}),
            "ion_isat": (["probe", "x", "y", "shot", "sweep"], ion_isat,
                         {"units": "A", "long_name": "Ion Saturation Current"}),
            "electron_isat": (["probe", "x", "y", "shot", "sweep"], electron_isat,
                              {"units": "A", "long_name": "Electron Saturation Current"}),
            "p_e": (["probe", "x", "y", "shot", "sweep"], p_e, {"units": "Pa", "long_name": "Electron Pressure"}),
            "p_ei": (["probe", "x", "y", "shot", "sweep"], p_ei, {"units": "Pa", "long_name": "Plasma Pressure"})
        },
            coords={
                "probe": np.arange(num_probes),
                "port": ("probe", ports),
                "face": ("probe", faces),
                "z": ("probe", port_zs),
                "x": x_vals,
                "y": y_vals,
                "shot": np.arange(num_shots),
                "sweep": np.arange(num_sweeps),
                "time": ("sweep", sweep_times)
            },
            attrs={
                "ion_type": str(ion_type),
                "ion_temperature": str(t_i.value),
                "ion_temperature_units": str(t_i.unit),
                "time_units": "ms",
                "x_units": "cm",
                "y_units": "cm",
                "Exp name": params_dict['Exp name'],
                "GP Voltage": str(gp_voltage) + ' V',
                "B-field": str(params_dict['Magenta']) + ' kG',
                "Black South": str(params_dict['Black South']) + ' kG',
                "Yellow" : str(params_dict['Yellow']) + ' kG',
                "Cathode Current": str(params_dict['Cathode Current']) + ' A',
                "Run number": params_dict['Run number']
            }
        )
        return ds

def build_xarrays_tanh(bias,current,positions,ramp_times,dt,langmuir_configs, ion_type, params_dict):
    """
    Parameters
    ----------
    bias
    current
    positions
    ramp_times
    dt
    langmuir_configs
    ion_type
    params_dict

    Returns
    -------

    """

    # Suppress printing of RelativityWarning globally
    warnings.filterwarnings("ignore", category=RelativityWarning)

    # Suppress poorly conditioned polyfit warnings
    warnings.filterwarnings("ignore", category=np.RankWarning)

    x_vals = np.unique(positions[:, 0])
    y_vals = np.unique(positions[:, 1])
    num_shots = current.shape[2]
    num_sweeps = len(ramp_times) - 1
    # print('Number of shots: ', num_shots)
    # print('Number of sweeps: ', num_sweeps)
    # print('Number of x_vals: ', len(x_vals))
    # print('Number of y_vals: ', len(y_vals))
    # Only the left faces are sweeping - the other two are just collecting data
    sweeping_configs = [cfg for cfg in langmuir_configs if cfg['face'] == 'L']

    # Accounts for non Jan 2024 data we want to get both probes
    if len(sweeping_configs) == 0:
        sweeping_configs = [langmuir_configs[0], langmuir_configs[1]]

    # Only half of the faces actually are conducting sweeps
    num_probes = len(sweeping_configs)
    # print('Number of probes: ', num_probes)

    t_i = 1 * u.eV  # LAPD is held at an ion temperature of 1 eV as was described in Perks and in Leo's thesis -
    # We can't directly measure it yet.

    # How many values are we going to have in each dimension
    shape = (num_probes, len(x_vals), len(y_vals), num_shots, num_sweeps)

    # Fill all of our variable arrays with nans
    n_e = np.full(shape, np.nan)
    n_i = np.full(shape, np.nan)

    t_e = np.full(shape, np.nan)

    v_f = np.full(shape, np.nan)
    v_p = np.full(shape, np.nan)
    nu_ei = np.full(shape, np.nan)

    ion_isat = np.full(shape, np.nan)
    electron_isat = np.full(shape, np.nan)

    p_e = np.full(shape, np.nan)
    p_ei = np.full(shape, np.nan)

    # Define a dictionary of positions with the key as the x,y position and the value as the index in the positions array
    pos_idx = {(x,y): i for i,(x,y) in enumerate(positions)}

    # Initialize values so we don't have to repeatedly call the value function repeatedly in the loop
    dt_ms = dt.to(u.ms).value
    ramp_times_ms = ramp_times.to(u.ms).value
    # print('ramp_times_ms', len(ramp_times_ms)-1)

    # Here we are assuming the sweeps are taken at approximately the same time for all x,y,shot combinations (there are
    # the same number of sweeps for each combination and say sweep 5 always occurs between 4.9 and 5.1 ms for example
    sweep_windows = []
    sweep_times = []

    mid_x = x_vals[len(x_vals) // 2]
    mid_y = y_vals[len(y_vals) // 2]
    mid_shot = num_shots // 2

    key = (mid_x, mid_y)
    if key not in pos_idx:
        raise ValueError("Middle (x,y) position not found")
    loc = pos_idx[key]

    probe_index = 0
    test_bias = bias[probe_index,loc,mid_shot,:]
    test_current = current[probe_index, loc, mid_shot,:]
    time_array = np.arange(len(test_bias)) * dt_ms



    for sweep in range(num_sweeps):
        search_times = ((time_array >= ramp_times_ms[sweep]) &
                        (time_array <= ramp_times_ms[sweep+1]))
        try:
            first_index, last_index = find_sweep_indices(time_array, ramp_times_ms[sweep + 1], search_times, test_bias,
                                                         dt_ms)
            sweep_windows.append((first_index, last_index))
            sweep_times.append(first_index * dt_ms)

        except IndexError:
            sweep_windows.append((None,None))
            sweep_times.append(np.nan)

    # print('Sweep windows: ', len(sweep_windows))

    with tqdm(total = num_probes * len(x_vals) * len(y_vals) * num_shots, desc = 'Position-Shot combinations') as pbar_ls:
        # We want both the index and the actual value for x and y so we use enumerate. (Allows us to index the xarray)
        for probe_index, config in enumerate(sweeping_configs):
            # print('Probe: ', probe_index)
            A_p = config['area']
            port = config['port']
            face = config['face']

            for ix, x in enumerate(x_vals):
                for iy, y in enumerate(y_vals):

                    # Determine the location in the position array where we can find this xy combination
                    key = (x,y)
                    if key not in pos_idx:
                        continue
                    loc = pos_idx[key]

                    for shot in range(num_shots):
                        full_bias = bias[probe_index, loc, shot, :]
                        full_current = current[probe_index, loc, shot, :]

                        for sweep, (first_index, last_index) in enumerate(sweep_windows):

                            # If we can't find the sweep index originally, we skip the sweep. This should be noted as
                            # an Index error

                            if first_index is None:
                                continue

                            start_time = (first_index * dt_ms)
                            end_time = (last_index * dt_ms)
                            mask = ((time_array >= start_time) & (time_array <= end_time))
                            sort_index = np.argsort(full_bias[mask])
                            sorted_bias = full_bias[mask][sort_index]
                            sorted_current = full_current[mask][sort_index]

                            try:
                                v_f_value, _, v_f_index = get_floating_potential(sorted_bias, sorted_current)
                            except Exception as e:
                                v_f_value = np.nan

                            # print('Plasma potential value', v_p_value)
                            try:
                                i_ion_sat_value, _ = get_ion_isat_min(sorted_current, sorted_bias)
                                if i_ion_sat_value > 0:
                                    i_ion_sat_value = np.nan
                            except Exception as e:
                                i_ion_sat_value = np.nan

                            try:
                                t_e_value,_,v_p_value,i_electron_sat_value,_,_ = get_t_e_spline(
                                    sorted_bias, sorted_current, v_f_value)
                                if np.isnan(i_ion_sat_value):
                                    t_e_value = np.nan
                                    v_p_value = np.nan
                                    i_electron_sat_value = np.nan
                                if t_e_value is None:
                                    t_e_value = np.nan
                                    v_p_value = np.nan
                                    i_electron_sat_value = np.nan
                            except Exception as e:
                                t_e_value = np.nan
                                v_p_value = np.nan
                                i_electron_sat_value = np.nan

                            if t_e_value is np.nan:

                                n_e_value = np.nan
                                n_i_value = np.nan

                                nu_ei_value = np.nan

                                p_e_value = np.nan
                                p_ei_value = np.nan
                            else:
                                # if not isinstance(i_electron_sat_value, Quantity):
                                #     n_e_value = np.nan
                                # else:
                                #     try:
                                #         n_e_value = get_electron_density(i_electron_sat_value, A_p, t_e_value)
                                #     except Exception as e:
                                #         n_e_value = np.nan

                                try:
                                    n_i_value = get_ion_density(ion_type, i_ion_sat_value, A_p, t_e_value)
                                    n_e_value = n_i_value
                                except Exception as e:
                                    n_i_value = np.nan
                                    n_e_value = np.nan

                                try:
                                    nu_ei_value = get_electron_ion_collision_frequency(n_e_value, ion_type,
                                                                                           t_e_value)
                                except Exception as e:
                                    nu_ei_value = np.nan

                                try:
                                    p_e_value = l_get_pressure(t_e_value,n_e_value)
                                except Exception as e:
                                    p_e_value = np.nan

                                try:
                                    p_ei_value = l_get_pressure(t_e_value + t_i, n_e_value)
                                except Exception as e:
                                    p_ei_value = np.nan

                            # Assign each of these values to an array index in the xarray
                            t_e[probe_index, ix, iy, shot, sweep] = safe_value(t_e_value)

                            n_e[probe_index, ix, iy, shot, sweep] = safe_value(n_e_value)
                            n_i[probe_index, ix, iy, shot, sweep] = safe_value(n_i_value)

                            v_f[probe_index, ix, iy, shot, sweep] = safe_value(v_f_value)
                            v_p[probe_index, ix, iy, shot, sweep] = safe_value(v_p_value)

                            nu_ei[probe_index, ix, iy, shot, sweep] = safe_value(nu_ei_value)

                            ion_isat[probe_index, ix, iy, shot, sweep] = safe_value(i_ion_sat_value)
                            electron_isat[probe_index, ix, iy, shot, sweep] = safe_value(i_electron_sat_value)
                            # print('ion_isat', safe_value(i_ion_sat_value))
                            # print('electron_isat', safe_value(i_electron_sat_value))

                            p_e[probe_index, ix, iy, shot, sweep] = safe_value(p_e_value)
                            p_ei[probe_index, ix, iy, shot, sweep] = safe_value(p_ei_value)
                            # print(f"probe={probe_index}, x={ix}, y={iy}, shot={shot}, sweep={sweep}")

                        pbar_ls.update(1)

        # Now we create our xarray with these individual variable arrays
        ports = [cfg['port'] for cfg in sweeping_configs]
        faces = [cfg['face'] for cfg in sweeping_configs]
        port_zs = [portnum_to_z(p).value for p in ports]
        gp_voltage_values = []
        for k, v in params_dict.items():
            if k.startswith("GPV"):
                try:
                    gp_voltage_values.append(float(v))
                except (TypeError, ValueError):
                    pass  # skip non-convertible values

        gp_voltage = sum(gp_voltage_values) / len(gp_voltage_values) if gp_voltage_values else None

        # gas = params_dict['Gas type']
        # if 'h' in gas.lower():
        #     ion_type = 'H+'
        # elif 'he' in gas.lower():
        #     ion_type = 'He+'


        ds = xr.Dataset({
            "t_e": (["probe", "x", "y", "shot", "sweep"], t_e, {"units": "eV", "long_name": "Electron Temperature"}),
            "n_e": (["probe", "x", "y", "shot", "sweep"], n_e, {"units": "1/m^3", "long_name": "Electron Density"}),
            "n_i": (["probe", "x", "y", "shot", "sweep"], n_i, {"units": "1/m^3", "long_name": "Ion Density"}),
            "v_f": (["probe", "x", "y", "shot", "sweep"], v_f, {"units": "V", "long_name": "Floating Potential"}),
            "v_p": (["probe", "x", "y", "shot", "sweep"], v_p, {"units": "V", "long_name": "Plasma Potential"}),
            "nu_ei": (["probe", "x", "y", "shot", "sweep"], nu_ei,
                      {"units": "1/s", "long_name": "Electron-Ion Collision Frequency"}),
            "ion_isat": (["probe", "x", "y", "shot", "sweep"], ion_isat,
                         {"units": "A", "long_name": "Ion Saturation Current"}),
            "electron_isat": (["probe", "x", "y", "shot", "sweep"], electron_isat,
                              {"units": "A", "long_name": "Electron Saturation Current"}),
            "p_e": (["probe", "x", "y", "shot", "sweep"], p_e, {"units": "Pa", "long_name": "Electron Pressure"}),
            "p_ei": (["probe", "x", "y", "shot", "sweep"], p_ei, {"units": "Pa", "long_name": "Plasma Pressure"})
        },
            coords={
                "probe": np.arange(num_probes),
                "port": ("probe", ports),
                "face": ("probe", faces),
                "z": ("probe", port_zs),
                "x": x_vals,
                "y": y_vals,
                "shot": np.arange(num_shots),
                "sweep": np.arange(num_sweeps),
                "time": ("sweep", sweep_times)
            },
            attrs={
                "ion_type": str(ion_type),
                "ion_temperature": str(t_i.value),
                "ion_temperature_units": str(t_i.unit),
                "time_units": "ms",
                "x_units": "cm",
                "y_units": "cm",
                "Exp name": params_dict['Exp name'],
                "GP Voltage": str(gp_voltage) + ' V',
                "B-field": str(params_dict['Magenta']) + ' kG',
                "Black South": str(params_dict['Black South']) + ' kG',
                "Yellow" : str(params_dict['Yellow']) + ' kG',
                "Cathode Current": str(params_dict['Cathode Current']) + ' A',
                "Run number": params_dict['Run number']
            }
        )
        return ds

def safe_value(quantity):
    """

    Parameters
    ----------
    quantity - An astropy quantity that we want to return the value of, but if it is a NaN we want to return NaN

    Returns
    -------
    value - The quantity with stripped units or a NaN depending on what was inputted
    """
    if quantity is None:
        return np.nan
    try:
        return quantity.value
    except AttributeError:
        return np.nan

def build_pp_xarray(bias,current,positions,ramp_times,dt,langmuir_configs,ion_type):
    """
        Parameters
        ----------
        bias
        current
        positions
        ramp_times
        dt
        langmuir_configs
        ion_type

        Returns
        -------

        """
    x_vals = np.unique(positions[:, 0])
    y_vals = np.unique(positions[:, 1])
    num_shots = current.shape[2]
    num_sweeps = len(ramp_times) - 1

    # Only the left faces are sweeping - the other two are just collecting data
    sweeping_configs = [cfg for cfg in langmuir_configs if cfg['face'] == 'L']

    # Only half of the faces actually are conducting sweeps
    num_probes = len(sweeping_configs)

    t_i = 1 * u.eV  # LAPD is held at an ion temperature of 1 eV as was described in Perks and in Leo's thesis -
    # We can't directly measure it yet.

    # How many values are we going to have in each dimension
    shape = (num_probes, len(x_vals), len(y_vals), num_shots, num_sweeps)

    # Fill all of our variable arrays with nans
    n_e = np.full(shape, np.nan)
    n_i = np.full(shape, np.nan)

    t_e = np.full(shape, np.nan)

    v_f = np.full(shape, np.nan)
    v_p = np.full(shape, np.nan)
    nu_ei = np.full(shape, np.nan)

    ion_isat = np.full(shape, np.nan)
    electron_isat = np.full(shape, np.nan)

    p_e = np.full(shape, np.nan)
    p_ei = np.full(shape, np.nan)

    # Define a dictionary of positions with the key as the x,y position and the value as the index in the positions array
    pos_idx = {(x, y): i for i, (x, y) in enumerate(positions)}

    # Initialize values so we don't have to repeatedly call the value function repeatedly in the loop
    dt_ms = dt.to(u.ms).value
    ramp_times_ms = ramp_times.to(u.ms).value
    print('ramp_times_ms', len(ramp_times_ms) - 1)

    # Here we are assuming the sweeps are taken at approximately the same time for all x,y,shot combinations (there are
    # the same number of sweeps for each combination and say sweep 5 always occurs between 4.9 and 5.1 ms for example
    sweep_windows = []
    sweep_times = []

    mid_x = x_vals[len(x_vals) // 2]
    mid_y = y_vals[len(y_vals) // 2]
    mid_shot = num_shots // 2

    key = (mid_x, mid_y)
    if key not in pos_idx:
        raise ValueError("Middle (x,y) position not found")
    loc = pos_idx[key]

    probe_index = 0
    test_bias = bias[probe_index, loc, mid_shot, :]
    test_current = current[probe_index, loc, mid_shot, :]
    time_array = np.arange(len(test_bias)) * dt_ms

    for sweep in range(num_sweeps):
        search_times = ((time_array >= ramp_times_ms[sweep]) &
                        (time_array <= ramp_times_ms[sweep + 1]))
        try:
            first_index, last_index = find_sweep_indices(time_array, ramp_times_ms[sweep + 1], search_times, test_bias,
                                                         dt_ms)
            sweep_windows.append((first_index, last_index))
            sweep_times.append(first_index * dt_ms)

        except IndexError:
            sweep_windows.append((None, None))
            sweep_times.append(np.nan)

    print('Sweep windows: ', len(sweep_windows))

    with tqdm(total=num_probes * len(x_vals) * len(y_vals) * num_shots, desc='Position-Shot combinations') as pbar_ls:
        # We want both the index and the actual value for x and y so we use enumerate. (Allows us to index the xarray)
        for probe_index, config in enumerate(sweeping_configs):
            A_p = config['area']
            port = config['port']
            face = config['face']

            for ix, x in enumerate(x_vals):
                for iy, y in enumerate(y_vals):

                    # Determine the location in the position array where we can find this xy combination
                    key = (x, y)
                    if key not in pos_idx:
                        continue
                    loc = pos_idx[key]

                    for shot in range(num_shots):
                        full_bias = bias[probe_index, loc, shot, :]
                        full_current = current[probe_index, loc, shot, :]

                        for sweep, (first_index, last_index) in enumerate(sweep_windows):

                            # If we can't find the sweep index originally, we skip the sweep this should be noted as an
                            # an Index error

                            if first_index is None:
                                continue

                            start_time = (first_index * dt_ms)
                            end_time = (last_index * dt_ms)
                            mask = ((time_array >= start_time) & (time_array <= end_time))
                            sort_index = np.argsort(full_bias[mask])
                            sorted_bias = full_bias[mask][sort_index]
                            sorted_current = full_current[mask][sort_index]

                            try:
                                v_f_value = pp_get_floating_potential(sorted_bias, sorted_current)
                            except Exception as e:
                                v_f_value = np.nan

                            try:
                                i_electron_sat_value = pp_get_electron_isat(sorted_bias, sorted_current)
                            except Exception as e:
                                i_electron_sat_value = np.nan

                            try:
                                v_p_value, v_p_index = pp_get_plasma_potential(sorted_bias, sorted_current,return_arg=True)

                            except Exception as e:
                                v_p_value = np.nan

                            # print('Plasma potential value', v_p_value)
                            try:
                                i_ion_sat_value, _ = get_ion_isat_min(sorted_current, sorted_bias)
                            except Exception as e:
                                i_ion_sat_value = np.nan

                            # if not isinstance(v_p_value, Quantity):
                            #     t_e_value = np.nan
                            #
                            #     n_e_value = np.nan
                            #     n_i_value = np.nan
                            #
                            #     nu_ei_value = np.nan
                            #
                            #     p_e_value = np.nan
                            #     p_ei_value = np.nan
                            # else:
                            try:
                                exponential_section_b, exponential_section_c = pp_extract_exponential_section(
                                    sorted_bias, sorted_current)
                                l_T_e = pp_get_electron_temperature(exponential_section_b, exponential_section_c)
                                t_e_value = l_T_e[0][0]
                            except Exception as e:
                                t_e_value = np.nan

                            try:
                                n_e_value = get_electron_density(i_electron_sat_value, A_p, t_e_value)
                            except Exception as e:
                                n_e_value = np.nan

                            try:
                                n_i_value = get_ion_density(ion_type, i_ion_sat_value, A_p, t_e_value)
                            except Exception as e:
                                n_i_value = np.nan

                            try:
                                nu_ei_value = get_electron_ion_collision_frequency(n_e_value, ion_type, t_e_value)
                            except Exception as e:
                                nu_ei_value = np.nan

                            try:
                                p_e_value = l_get_pressure(t_e_value, n_e_value)
                            except Exception as e:
                                p_e_value = np.nan

                            try:
                                p_ei_value = l_get_pressure(t_e_value + t_i, n_e_value)
                            except Exception as e:
                                p_ei_value = np.nan

                            # Assign each of these values to an array index in the xarray
                            t_e[probe_index, ix, iy, shot, sweep] = safe_value(t_e_value)

                            n_e[probe_index, ix, iy, shot, sweep] = safe_value(n_e_value)
                            n_i[probe_index, ix, iy, shot, sweep] = safe_value(n_i_value)

                            v_f[probe_index, ix, iy, shot, sweep] = safe_value(v_f_value)
                            v_p[probe_index, ix, iy, shot, sweep] = safe_value(v_p_value)

                            nu_ei[probe_index, ix, iy, shot, sweep] = safe_value(nu_ei_value)

                            ion_isat[probe_index, ix, iy, shot, sweep] = safe_value(i_ion_sat_value)
                            electron_isat[probe_index, ix, iy, shot, sweep] = safe_value(i_electron_sat_value)
                            # print('ion_isat', safe_value(i_ion_sat_value))
                            # print('electron_isat', safe_value(i_electron_sat_value))

                            p_e[probe_index, ix, iy, shot, sweep] = safe_value(p_e_value)
                            p_ei[probe_index, ix, iy, shot, sweep] = safe_value(p_ei_value)
                            # print(f"probe={probe_index}, x={ix}, y={iy}, shot={shot}, sweep={sweep}")

                        pbar_ls.update(1)

        # Now we create our xarray with these individual variable arrays
        ports = [cfg['port'] for cfg in sweeping_configs]
        faces = [cfg['face'] for cfg in sweeping_configs]
        port_zs = [portnum_to_z(p).value for p in ports]

        ds = xr.Dataset({
            "t_e": (["probe", "x", "y", "shot", "sweep"], t_e, {"units": "eV", "long_name": "Electron Temperature"}),
            "n_e": (["probe", "x", "y", "shot", "sweep"], n_e, {"units": "1/m^3", "long_name": "Electron Density"}),
            "n_i": (["probe", "x", "y", "shot", "sweep"], n_i, {"units": "1/m^3", "long_name": "Ion Density"}),
            "v_f": (["probe", "x", "y", "shot", "sweep"], v_f, {"units": "V", "long_name": "Floating Potential"}),
            "v_p": (["probe", "x", "y", "shot", "sweep"], v_p, {"units": "V", "long_name": "Plasma Potential"}),
            "nu_ei": (["probe", "x", "y", "shot", "sweep"], nu_ei,
                      {"units": "1/s", "long_name": "Electron-Ion Collision Frequency"}),
            "ion_isat": (["probe", "x", "y", "shot", "sweep"], ion_isat,
                         {"units": "A", "long_name": "Ion Saturation Current"}),
            "electron_isat": (["probe", "x", "y", "shot", "sweep"], electron_isat,
                              {"units": "A", "long_name": "Electron Saturation Current"}),
            "p_e": (["probe", "x", "y", "shot", "sweep"], p_e, {"units": "Pa", "long_name": "Electron Pressure"}),
            "p_ei": (["probe", "x", "y", "shot", "sweep"], p_ei, {"units": "Pa", "long_name": "Plasma Pressure"})
        },
            coords={
                "probe": np.arange(num_probes),
                "port": ("probe", ports),
                "face": ("probe", faces),
                "z": ("probe", port_zs),
                "x": x_vals,
                "y": y_vals,
                "shot": np.arange(num_shots),
                "sweep": np.arange(num_sweeps),
                "time": ("sweep", sweep_times)
            },
            attrs={
                "ion_type": str(ion_type),
                "ion_temperature": str(t_i.value),
                "ion_temperature_units": str(t_i.unit),
                "time_units": "ms",
                "x_units": "cm",
                "y_units": "cm"
            }
        )
        return ds




