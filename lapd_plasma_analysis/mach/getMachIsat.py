import numpy as np
import xarray as xr
import astropy.units as u
from bapsflib import lapd
from bapsflib.lapd.tools import portnum_to_z
import dask.array as da
from dask import delayed
import xarray as xr

from lapd_plasma_analysis.langmuir.getIVsweep import get_shot_positions


@delayed
def lazy_read_signal(fname, bc):
    """Opens the file safely just for this specific read operation."""
    with lapd.File(fname) as f:
        return f.read_data(*bc, silent=True)['signal']

# Note: This code is based on getIVsweep.py in the lapd-plasma-analysis repository.
# TODO merge into getIVsweep.py ?

# Todo see what langmuir functions are referenced the most in Mach analysis and consider moving them to a tools package?


def get_mach_isat(filename, mach_configs):
    """
    Get the ion saturation current signal from the Mach probe and restore it to meaningful units. (WIP)

    Parameters
    ----------
    filename : `str`
        Path to an HDF5 file.
    mach_configs : `numpy.ndarray`
        Structured array (WIP)

    Returns
    -------

    """

    with lapd.File(filename) as lapd_file:
        run_name = lapd_file.info['run name']
        mach_motor_datas = [lapd_file.read_controls([('6K Compumotor', mach_configs['receptacle'][0])])]

        sample_bc = (mach_configs['board'][0], mach_configs['channel'][0])
        sample_data = lapd_file.read_data(*sample_bc, silent=True)
        dt = sample_data.dt

        signal_shape = sample_data['signal'].shape
        num_frames = signal_shape[-1]  # <--- ADD THIS LINE

    mach_bcs = np.atleast_1d(mach_configs[['board', 'channel']])
    num_isweep = len(mach_configs)

    lazy_signals = [lazy_read_signal(filename, bc) for bc in mach_bcs]

    # Pass that exact shape to Dask so it knows how to slice later
    dask_arrays = [da.from_delayed(sig, shape=signal_shape, dtype=float) for sig in lazy_signals]

    isat = da.stack(dask_arrays, axis=0)
    # Motor positions (Keep your existing logic here)
    positions, num_positions, num_shots_per_position, selected_shots = get_shot_positions(mach_motor_datas[0])

    # Apply the slice to select only the valid shots
    isat = isat[:, selected_shots, ...]

    # Force Dask to recount the chunk sizes so it knows the new dimensions
    isat = isat.compute_chunk_sizes()

    # Apply the reshape
    isat = isat.reshape((num_isweep, num_positions, num_shots_per_position, num_frames))

    # Apply calibration factors lazily
    for i in range(len(isat)):
        isat[i] = isat[i] / mach_configs['resistance'][i] / mach_configs['gain'][i] / mach_configs['area'][i]

    # Hand off to the xarray builder
    isat_da = to_mach_isat_da(isat, positions, num_shots_per_position, mach_configs, dt).rename(run_name)

    # Lazy offset calculation
    isat_offsets = isat_da[..., -2000:].mean(dim="time")
    isat_da -= isat_offsets
    isat_da = isat_da.where(isat_da > 0)

    return isat_da


def to_mach_isat_da(isat, positions, shots_per_position, mach_configs, dt):
    """

    Parameters
    ----------
    isat
    positions
    shots_per_position
    mach_configs
    dt

    Returns
    -------

    """

    ports_unique = np.unique([port for port in mach_configs['port']])
    x_pos = np.unique(positions[:, 0])
    y_pos = np.unique(positions[:, 1])

    num_frames = isat.shape[-1]
    isat_signals_shape = (len(x_pos), len(y_pos), shots_per_position, num_frames)

    # Reshape all signals lazily
    isat_signals = [probe_face_isat.reshape(isat_signals_shape) for probe_face_isat in isat]

    da_list = []
    for i in range(len(isat)):
        # Calculate coordinate metadata
        port_val = mach_configs['port'][i]
        face_val = mach_configs['face'][i]
        probe_idx = np.where(ports_unique == port_val)[0][0]
        z_val = portnum_to_z(port_val).to(u.cm).value

        # Build a single DataArray for this specific probe face
        single_da = xr.DataArray(
            name="isat",  # <--- 1. ADD THIS NAME
            data=isat_signals[i],
            dims=['x', 'y', 'shot', 'time'],
            coords={
                'x': x_pos,
                'y': y_pos,
                'shot': np.arange(shots_per_position),
                'time': np.arange(num_frames) * dt.to(u.ms).value
            }
        )

        # Tag it with its specific probe and face coordinates
        single_da = single_da.assign_coords(
            probe=probe_idx,
            face=face_val
        )

        # Expand dimensions so xarray can stitch them together
        single_da = single_da.expand_dims(['probe', 'face'])

        # Explicitly link port and z to the 'probe' dimension
        single_da = single_da.assign_coords(
            port=('probe', [port_val]),
            z=('probe', [z_val])
        )

        da_list.append(single_da)

        # Let xarray stitch the virtual grid together.
        # 2. Add compat='override' to clear the new warning
    combined_ds = xr.combine_by_coords(da_list, join='outer', compat='no_conflicts')

    # 3. Extract the DataArray from the Dataset!
    # This prevents the .rename() ValueError in the parent function.
    isat_da = combined_ds['isat']

    # Ensure attributes carry over units
    isat_da['x'].attrs['units'] = 'cm'
    isat_da['y'].attrs['units'] = 'cm'
    isat_da['time'].attrs['units'] = 'ms'

    return isat_da
