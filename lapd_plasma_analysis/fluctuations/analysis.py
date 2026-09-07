import os
from lapd_plasma_analysis.fluctuations.fourier import *
from scipy.ndimage import uniform_filter1d
from lapd_plasma_analysis.Read_hdf5.read_metadata import *
from  lapd_plasma_analysis.obtain_plots.xarray_plots import shortened_exp_name
import pandas as pd
from lapd_plasma_analysis.Read_hdf5.read_metadata import metadata_dict
from plasmapy.particles import Particle
from lapd_plasma_analysis.file_access import ask_yes_or_no


def remove_outliers(time_series, window, threshold):
    arr = time_series.values.astype(float)
    n = len(arr)

    if 2 * window + 1 > n:
        raise ValueError("window too large for data length")

    w_full = 2 * window + 1

    mean_all = uniform_filter1d(arr, size=w_full, mode="reflect")
    mean_sq_all = uniform_filter1d(arr**2, size=w_full, mode="reflect")

    mean_excl = (mean_all * w_full - arr) / (w_full - 1)
    var_excl = (mean_sq_all * w_full - arr**2) / (w_full - 1) - mean_excl**2
    var_excl[var_excl < 0] = 0.0
    std_excl = np.sqrt(var_excl)

    dev = np.abs(arr - mean_excl)

    outlier_mask = dev > threshold * std_excl

    outlier_mask[:window] = False
    outlier_mask[-window:] = False

    removed_count = np.count_nonzero(outlier_mask)
    arr[outlier_mask] = np.nan

    print(f"Removed outliers: {removed_count}")

    return xr.DataArray(arr, coords=time_series.coords, dims=time_series.dims), outlier_mask

def get_isat_vf(filename, hdf5_path, flux_nc_folder, main_luke = False):
    """
    Obtains saturation current, floating potential, floating potential difference, sound speed,
    and density data from the HDF5 file specified by `filename`. Deposits a NetCDF file containing this data (as
    an `xarray.Dataset` in `flux_nc_folder`. The NetCDF file will have the same name as the HDF5 file.

    Parameters
    ----------
    filename : `str`
        String containing path to specific HDF5 file

    hdf5_path : `str`
        String containing path to all HDF5 files

    flux_nc_folder : `str`
        String giving path to the folder containing the NetCDF files for fluctuation data

    Returns
    -------
    dataset : `xarray.Dataset`
        Dataset containing floating voltage, saturation current, sound speed and density data

    """
    assert os.path.exists(filename)
    assert get_config_id(lapd.File(filename).info['exp name']) in [1, 2, 3]

    use_updated_ds = ask_yes_or_no('\n If it exists, use the updated Langmuir Dataset for temperatures? (y/n) ')
    file = lapd.File(filename)


    # Get Params to create a consistent file naming convention with Langmuir .nc files
    params_dict = metadata_dict(file.info['run description'],
                                file.info['exp name'],
                                file.info['file'])

    gp_voltage_values = []
    for k, v in params_dict.items():
        if k.startswith("GPV"):
            try:
                gp_voltage_values.append(float(v))
            except (TypeError, ValueError):
                pass  # skip non-convertible values
    gp_voltage = sum(gp_voltage_values) / len(gp_voltage_values) if gp_voltage_values else None
    gp_voltage = str(gp_voltage) + 'V'

    exp_name = shortened_exp_name(params_dict['Exp name'])
    run_number = params_dict['Run number']
    b_field = str(params_dict['Magenta']) + 'kG'
    cath_curr = str(params_dict['Cathode Current']) + 'A'
    ion_type = get_ion(filename)
    nc_filename = (exp_name + '_' + run_number + '_' + gp_voltage + '_' + b_field + '_' + cath_curr + '_' +
                   ion_type)

    config_id = get_config_id(file.info['exp name'])

    # scaling = (resistance * gain)**(-1)

    # config_id == 0 -> Nov 2018
    # config_id == 1 -> Mar 2022
    # config_id == 2 -> Nov 2022
    # config_id == 3 -> Jan 2024
    if config_id == 1:
        # board, channel, receptacle, port, scaling, area
        isat_probes = [
            (2, 1, 1, 27, 1/15, 8) # don't actually know probe area ! might be 2mm^2, not 8
            # ,(3, 1, 3, 29), # mach | don't know about
            # (3, 4, 4, 45)  # mach | using these two
        ]

        vf_receptacle = 2
        vf_board = 2
        vftop_channel = 2
        vfbot_channel = 3

    if config_id == 2:
        # board, channel, receptacle, port, scaling, area
        isat_probes = [
            (2, 1, 3, 27, 2/16.1, 8),
            (2, 3, 2, 33, 2/16.1, 8) # Cathode facing probes
        ]


    if config_id == 3:
        # board, channel, receptacle, port, scaling, area
        isat_probes = [
             (2, 1, 4, 18, 1/(7.36*0.96), 8),
             #(2, 2, 4, 18, 1/(7.32*0.95), 8),
             (3, 1, 1, 29, 1/(5.26*1.03), 8), # kept only one per z-position for now
             #(3, 3, 1, 29, 1/(5.21*1.014), 8) # Cathode facing probes
        ]

    isat_data_arrays = []
    density_data_arrays = []
    full_updated_ds = False
    # This for loop allows us to look at one probe at a time (one z position)
    for isat_board, isat_channel, isat_receptacle, isat_port, isat_scaling, probe_area in isat_probes:
        isat_data = file.read_data(isat_board, isat_channel, silent=True, add_controls=[('6K Compumotor', isat_receptacle)])

        # -----Aside about file.read_data-----

        # isat data returns a 3-D structured array where each element takes the form of the following columns
        # [('shotnum', '<u4'), ('signal', '<f4', (#,)), ('xyz', '<f4', (3,)), ('ptip_rot_theta', '<f8'),
        # ('ptip_rot_phi', '<f8')]. We are scanning across a fixed number of shots at each x-position (radial). Each
        # unique shot-position combination gets its own shotnum and thus its own dimension. Most of the
        # columns have just one value in it. However, 'signal' and 'xyz' have > 0 values in them. Each value in 'signal'
        # corresponds to a different voltage measurement from the probe. Each value in 'xyz' corresponds to the probe's
        # spatial location within LAPD see more in which spatial dimension is which below.

        # shotnum: Integer number of the certain plasma discharge within the experiment (in sequential order of when taken
        # This does not necessarily start at 0 if there were shots before the data was measured)
        # signal: Probe signal - the number gives the total number of measurements taken
        # xyz: The x, y, z postition of the probe x = 0, y = 1, z = 2 if you call that dimension
        # ptip_rot_theta: Probe tip rotation angle theta -- we're not focused on anything involving this so ignore
        # ptip_rot_phi: Probe tip rotation angle phi -- we're not focused on anything involving this so ignore

        # You can call a specific set of structured elements by doing something like isat_data['shotnum'].

        # Unique attributes are:
        # dt - Temporal step size
        # dv - Voltage step size
        # info - Metadata dictionary

        # For this function, we only care about the z-position of the probe. We never move the probe in z, so let's just
        # take the value for the first dimension (The 0 in the expression below).

        # -----End Aside-----

        dt = isat_data.dt.value
        times = np.linspace(0, len(isat_data[0][1])*dt, len(isat_data[0][1]))
        z = isat_data[0]['xyz'][2]
        # z = [isat_data[0][2][2]]

        isat_vs_x = []
        isat_vs_x.append([])
        x_index = 0
        x_array = []
        # Get the first x position to start building up an array of all the x-indices
        x_array.append(isat_data[0]['xyz'][0])

        # x_array.append(isat_data[0][2][0])

        # For each dimension in isat_data check to see if we have a unique x-position. Build a set of nested lists. The
        # first index in the nested list describes the x-position of the obtained data. The second index in the nested
        # list describes the shot associated with the x-value. Thus, we are making an array of the measured voltage
        # values sorted first by x-position and then within each x-position by shot

        # isat_data is essentially a list of all the discharges concatanated together. This loop finds the density, and
        # Isat for each discharge (x,z, shot combination)
        for discharge in isat_data:
            if abs(discharge[2][0] - x_array[x_index]) <= 0.01:
                isat_vs_x[x_index].append(discharge[1]*isat_scaling)

            else:
                x_index += 1
                isat_vs_x.append([])
                x_array.append(discharge[2][0])
                isat_vs_x[x_index].append(discharge[1]*isat_scaling)


        isat_vs_x = np.array(isat_vs_x)
        _, numshot, __ = isat_vs_x.shape
        shot_array = range(numshot)
        isat_vs_x = np.expand_dims(isat_vs_x, 0)
        isat_data_array = xr.DataArray(isat_vs_x * u.ampere, dims = ['z', 'x', 'shot', 'time'],
                                                           coords = {'x': np.round(x_array)*u.cm,
                                                                     'shot': shot_array,
                                                                     'time': times*u.s.to(u.ms),
                                                                     'z': np.round(z)*u.cm},
                                                           name = 'isat',
                                                           attrs={'units': 'A'})

        isat_data_array.coords['x'].attrs['units'] = 'cm'
        isat_data_array.coords['time'].attrs['units'] = 'ms'
        # Appends the isat data for the probe at a specific z value
        isat_data_arrays.append(isat_data_array)

        sound_speed_data_array, density_data_array, updated_ds_bool = get_density_data(file, hdf5_path, isat_data_array,
                                                                                       shot_array,
                                                                                      probe_area, z, main_luke,
                                                                                      use_updated_ds = use_updated_ds)
        if updated_ds_bool:
            full_updated_ds = True
        # Appends the density data for a specifc z value
        density_data_arrays.append(density_data_array)

    if not main_luke:
        isat_data_array = xr.concat(isat_data_arrays, dim="z").sortby("z")
        density_data_array = xr.concat(density_data_arrays, dim="z").sortby("z")
    else:
        # Extract the physical z-values first
        isat_z_values = [float(arr.coords['z'].values) for arr in isat_data_arrays]

        # Manually rebuild each DataArray, copying everything EXCEPT 'z'
        isat_clean_list = [
            xr.DataArray(
                arr.data,
                dims=arr.dims,
                coords={k: v for k, v in arr.coords.items() if k != 'z'},
                attrs=arr.attrs,
                name=arr.name
            )
            for arr in isat_data_arrays
        ]

        # Create a proper dimension index and concatenate
        isat_z_index = pd.Index(isat_z_values, name="z")
        isat_data_array = xr.concat(isat_clean_list, dim=isat_z_index).sortby("z")

        # ==========================================
        # 2. FIX AND CONCATENATE DENSITY DATA
        # ==========================================

        # Apply the exact same manual reconstruction for density
        density_z_values = [float(arr.coords['z'].values) for arr in density_data_arrays]

        density_clean_list = [
            xr.DataArray(
                arr.data,
                dims=arr.dims,
                coords={k: v for k, v in arr.coords.items() if k != 'z'},
                attrs=arr.attrs,
                name=arr.name
            )
            for arr in density_data_arrays
        ]

        density_z_index = pd.Index(density_z_values, name="z")
        density_data_array = xr.concat(density_clean_list, dim=density_z_index).sortby("z")

    if config_id == 1:
        vf_top_data = file.read_data(vf_board, vftop_channel, silent=True, add_controls=[('6K Compumotor', vf_receptacle)])
        vf_bottom_data = file.read_data(vf_board, vfbot_channel, silent=True, add_controls=[('6K Compumotor', vf_receptacle)])
        dt = vf_bottom_data.dt.value
        times = np.linspace(0, len(vf_bottom_data[0][1])*dt, len(vf_bottom_data[0][1]))
        z = vf_bottom_data[0][2][2]

        dvf_vs_x = []
        vf_vs_x = []
        dvf_vs_x.append([])
        vf_vs_x.append([])
        x_index = 0
        x_array = []
        x_array.append(vf_bottom_data[0][2][0])
        for index in range(len(vf_bottom_data)):
            if abs(vf_bottom_data[index][2][0] - x_array[x_index]) <= 0.01:
                dvf_vs_x[x_index].append((vf_top_data[index][1] - vf_bottom_data[index][1])*10)  # /10 in exp
                vf_vs_x[x_index].append(vf_bottom_data[index][1]*10)

            else:
                x_index += 1
                dvf_vs_x.append([])
                vf_vs_x.append([])
                x_array.append(vf_bottom_data[index][2][0])
                dvf_vs_x[x_index].append((vf_top_data[index][1] - vf_bottom_data[index][1]) * 10)
                vf_vs_x[x_index].append(vf_bottom_data[index][1]*10)

        dvf_vs_x = np.array(dvf_vs_x)
        vf_vs_x = np.array(vf_vs_x)
        dvf_data_array = xr.DataArray(dvf_vs_x*u.volt, dims=['x', 'shot', 'time'],
                                                       coords = {'x': np.round(x_array)*u.cm,
                                                                 'shot': shot_array,
                                                                 'time': times*u.s.to(u.ms),
                                                                 'z': np.round(z)*u.cm},
                                                       name = 'dvf',
                                                       attrs = {'units': 'V'})
        dvf_data_array.coords['x'].attrs['units'] = 'cm'
        dvf_data_array.coords['time'].attrs['units'] = 'ms'

        vf_data_array = xr.DataArray(vf_vs_x*u.volt, dims = ['x', 'shot', 'time'],
                                                     coords = {'x': np.round(x_array)*u.cm,
                                                               'shot': shot_array,
                                                               'time': times*u.s.to(u.ms),
                                                               'z': np.round(z)*u.cm},
                                                     name = 'vf',
                                                     attrs={'units': 'V'})
        vf_data_array.coords['x'].attrs['units'] = 'cm'
        vf_data_array.coords['time'].attrs['units'] = 'ms'

    # Find any coordinates that are not in the master isat array
    extra_coords_ss = [c for c in sound_speed_data_array.coords if c not in isat_data_array.coords]
    extra_coords_den = [c for c in density_data_array.coords if c not in isat_data_array.coords]

    # Drop the hitchhikers safely
    sound_speed_data_array = sound_speed_data_array.drop_vars(extra_coords_ss, errors='ignore')
    density_data_array = density_data_array.drop_vars(extra_coords_den, errors='ignore')

    if config_id == 1:
        isat_vf_dataset = xr.merge([isat_data_array, dvf_data_array,
                                    vf_data_array, sound_speed_data_array, density_data_array])

    else:
        isat_vf_dataset = xr.merge([isat_data_array, sound_speed_data_array, density_data_array])

    isat_vf_dataset.attrs["HDF5 file name"] = "filename"
    isat_vf_dataset.attrs["HDF5 file path"] = hdf5_path
    isat_vf_dataset.attrs["Run name"] = file.info["run name"]
    isat_vf_dataset.attrs["Experiment series"] = file.info["exp name"]
    isat_vf_dataset.attrs["Config ID"] = config_id

    # isat_vf_dataset.to_netcdf(flux_nc_folder +
    #                           filename.replace(hdf5_path, '').replace('.hdf5', '') + '.nc')
    if full_updated_ds:
        isat_vf_dataset.to_netcdf(flux_nc_folder + nc_filename + '_updated.nc')
    else:
        isat_vf_dataset.to_netcdf(flux_nc_folder + nc_filename + '.nc')
    return isat_vf_dataset

def extend_dim_repeat(d1, d2, dim: str):
    """
    Extend d1 along dimension `dim` to match d2's coordinates by repeating values.

    Works for both xarray.DataArray and xarray.Dataset.

    Example:
        d1.t = [0, 0.5, 1]
        d1.data = [0, 1, 2]
        d2.t = [0, 0.25, 0.5, 0.75, 1, 1.25]
        -> returns [0, 0, 1, 1, 2, 2]
        d1 = T_e_data
        d2 = isat_data_array
        dim = 'time'/'sweep'
    """
    d1_coords = d1[dim].values
    d2_coords = d2[dim].values
    # print('d1 coords:', d1_coords)
    # print('d2 coords:', d2_coords)

    idxs = np.searchsorted(d1_coords, d2_coords, side="right") - 1
    idxs = np.clip(idxs, 0, len(d1_coords) - 1)
    # print('idxs:', idxs)
    # print('0: ', len(np.where(idxs < 1)[0]))
    # print('1: ', len(np.where(idxs == 1)[0]))
    # print('2: ', len(np.where(idxs == 2)[0]))
    # print('3: ', len(np.where(idxs == 3)[0]))
    # print('4: ', len(np.where(idxs == 4)[0]))
    # print('5: ', len(np.where(idxs == 5)[0]))
    # print('6: ', len(np.where(idxs == 6)[0]))
    #
    # print('41', len(np.where(idxs == 41)[0]))
    # print('len idxs:', len(idxs))
    # print('da: ', {di: d1[di].shape for di in d1.dims})
    # Get the axis number in the original array of that we want to duplicate
    dim_axis = d1.get_axis_num(dim)
    # print('d1 axis num time: ', dim_axis)
    # Create a new array where
    extended_data = np.take(d1.values, idxs, axis=dim_axis)
    # print('length extended_data:', len(extended_data))
    new_coords = {}

    # Loop through every coordinate currently attached to d1
    for coord_name in d1.coords:

        # Check if the dimension we just stretched (e.g., 'time') is used by this coordinate
        if dim not in d1[coord_name].dims:
            # If it DOES NOT depend on 'time' (like 'x' or 'shot'), copy it over safely
            new_coords[coord_name] = d1[coord_name]
    new_coords[dim] = d2_coords
    new_xarray = xr.DataArray(extended_data, dims=d1.dims, coords=new_coords, attrs=d1.attrs, name = d1.name)
    # print('new coords: ', new_xarray.coords)

    return new_xarray



    # 2. Use .isel() to apply the indices.
    # This automatically extends the data AND any dependent coordinates (like 'sweep')
    # If d1 is a Dataset, it intelligently skips variables that don't have the 'dim' dimension.
    # d1_extended = d1.isel({dim: idxs})

    # 3. Swap out the old repeated coordinate values for the exact d2 coordinates
    d1_extended = d1.reindex({dim: d2_coords}, method='nearest')

    # print('extended d1', d1_extended.dropna(dim = 'time'))
    return d1_extended

    # def _extend_array(da: xr.DataArray) -> xr.DataArray:
    #     print('da: ', {di: da[di].shape for di in da.dims})
    #     """Extend a single DataArray by repeating along dim."""
    #     axis = da.get_axis_num(dim)
    #     print('axis: ', axis)
    #     extended_data = np.take(da.values, idxs, axis=axis)
    #     print('extended_data: ', extended_data)
    #     new_coords = dict(da.coords)
    #     print('new_coords: ', new_coords)
    #     new_coords[dim] = d2_coords
    #     print('new_coords[dim]: ', new_coords[dim])
    #
    #     return xr.DataArray(
    #         extended_data,
    #         dims=da.dims,
    #         coords=new_coords,
    #         attrs=da.attrs,
    #         name=da.name,
    #     )
    #
    # if isinstance(d1, xr.DataArray):
    #     return _extend_array(d1)
    #
    # elif isinstance(d1, xr.Dataset):
    #     extended_vars = {}
    #     for var in d1.data_vars:
    #         da = d1[var]
    #         if dim in da.dims:
    #             extended_vars[var] = _extend_array(da)
    #         else:
    #             extended_vars[var] = da
    #     return xr.Dataset(extended_vars, attrs=d1.attrs)
    #
    # else:
    #     raise TypeError("Input d1 must be an xarray.DataArray or xarray.Dataset")


def get_density_data(fileobj, hdf5_folder, isat_data_array, shot_array, probe_area, z, main_luke = False,
                     use_updated_ds = False):
    """
    Auxiliary function to `get_isat_vf`. Retrieves electron temperature data from langmuir
    analysis, uses it to compute the sound speed and density data using the saturation current data.
    Upsampling via linear interpolation of the electron temperature data is done to align the coordinates
    of the sound speed dataset with those of the saturation current data array.

    Right now, it requires the user to go through the prompting as if they are about to plot the Langmuir
    analysis plots.

    Parameters
    ----------
    filename : str
        Currently unused.

    isat_data_array : xarray.DataArray or xarray.Dataset
        The saturation current data array calculated in `get_isat_vf`.

    shot_array : numpy.array
        Currently unused.

    Returns
    -------
    `tuple`
        A tuple of `xarray.DataArray` or `xarray.Dataset`,`(sound_speed_data_array, density_data_array)`.

    """
    #todo hardcoded

    # note this only supports the use of 1 hdf5 file at a time, so as to be compatible
    # with the rest of this module
    updated_ds = False
    if main_luke:
        #TODO what happens if it doesn't exist?
        params_dict = metadata_dict(fileobj.info['run description'],
                                        fileobj.info["exp name"],
                                        fileobj.info['file'])

        exp_name = shortened_exp_name(params_dict['Exp name'])

        run_num = params_dict['Run number']
        langmuir_nc_folder = hdf5_folder + "lang_nc/"
        updated_langmuir_nc_folder = langmuir_nc_folder + "updated/"
        search_term = exp_name + '_' + run_num
        nc_in_lang_list = sorted([f for f in os.listdir(langmuir_nc_folder) if f.endswith(".nc")])
        dataset = None

        for file in nc_in_lang_list:
            if search_term in file:
                if use_updated_ds:
                    try:
                        updated_ds = True
                        pathname = os.path.join(updated_langmuir_nc_folder, file)
                        dataset = xr.open_dataset(pathname, engine="netcdf4")
                    except FileNotFoundError:
                        pathname = os.path.join(langmuir_nc_folder, file)
                        dataset = xr.open_dataset(pathname, engine="netcdf4")
                else:
                    pathname = os.path.join(langmuir_nc_folder, file)
                    dataset = xr.open_dataset(pathname, engine="netcdf4")

                ion_type = dataset.attrs['ion_type']


    else:
        month, year = fileobj.info["exp name"].split("_")
        prefix = "lang_" + month[:3] + year[-2:] + "_"

        if use_updated_ds:
            try:
                lang_nc_file_path = hdf5_folder + "lang_nc/updated/" + prefix + fileobj.info["run name"] + ".nc"
                updated_ds = True
            except FileNotFoundError:
                lang_nc_file_path = hdf5_folder + "lang_nc/" + prefix + fileobj.info["run name"] + ".nc"

        else:
            lang_nc_file_path = hdf5_folder + "lang_nc/" + prefix + fileobj.info["run name"] + ".nc"

        dataset = get_langmuir_dataset(lang_nc_file_path)
        ion_type = dataset.attrs['ion_type']


    target_mass_unit = u.eV * u.s**2 /u.m**2
    M = Particle(ion_type).mass.to(target_mass_unit)
    print('mass: ', M.to(u.kg))
    c = 299792458*u.m/u.s

    # Find the temperature data associated with the Langmuir probe closest to the fluctuations probe
    if main_luke:
        # Luke nc files are not indexable by z but rather probe index associated with langmuir configs so must adjust
        # Likewise time is not a dimension but rather sweep number so this fixes both of these issues
        lang_z_values = dataset.z.values
        lang_z = dataset.z.values[np.argmin(np.abs(lang_z_values - z))]
        probe_num = np.where(lang_z_values == lang_z)[0][0]
        T_e_data = dataset['t_e'].sel(probe = probe_num, method = 'nearest')
        T_e_data = T_e_data.swap_dims({'sweep':'time'})

    else:
        T_e_data = dataset['T_e'].sel(z=z, method='nearest')


    # T_e_data = T_e_data.assign_coords(z=isat_data_array.z)
    # print(T_e_data.dims)
    # print({dim: T_e_data[dim].shape for dim in T_e_data.dims})
    # Make the T_e arra the same size in the time dimension as isat_data_array

    # 1. HEAL THE RAW DATA FIRST (Before any stretching)
    # This averages the 8 shots, allowing good sweeps to cover for the missing 38%.
    T_e_mean_raw = T_e_data.mean(dim="shot", keep_attrs=True, skipna=True)

    # 2. Stretch the healed, solid timeline to match Isat
    # (This creates the microsecond gaps, but now every gap has a solid starting anchor)
    T_e_interp = extend_dim_repeat(T_e_mean_raw, isat_data_array, "time")

    # 3. Calculate the leash dynamically
    isat_dt = float(isat_data_array.time.values[1] - isat_data_array.time.values[0])
    langmuir_dt = float(T_e_data.time.diff(dim='time').median())
    max_steps = int(langmuir_dt / isat_dt) + 5

    # 4. Forward fill the gaps
    # (This will finally work because the anchor points are guaranteed to exist)
    T_e_interp = T_e_interp.ffill(dim="time", limit=max_steps)

    # print("\n--- TIME STEP DIAGNOSTICS ---")
    # print(f"Isat dt: {isat_dt}")
    # print(f"Langmuir dt: {langmuir_dt}")
    # print(f"Calculated Max Steps (Leash): {max_steps}")
    # print("-----------------------------\n")
    #
    # print("\n--- NaN DIAGNOSTICS ---")

    # # 1. Total NaN counts across the major arrays
    # print(f"Total NaNs in isat_data_array: {int(isat_data_array.isnull().sum())}")
    # print(f"Total NaNs in T_e_data (raw):  {int(T_e_data.isnull().sum())}")
    # print(f"Total NaNs in T_e_mean:        {int(T_e_mean_raw.isnull().sum())}")
    # print(f"Total NaNs in T_e_interp:      {int(T_e_interp.isnull().sum())}")


    # T_e_interp = T_e_interp.broadcast_like(isat_data_array)
    sound_speed_data_array = np.sqrt(T_e_interp/M).rename('sound_speed')

    sound_speed_data_array.attrs['units'] = 'm/s'
    sound_speed_data_array.coords['time'].attrs['units'] = 'ms'
    sound_speed_data_array.coords['x'].attrs['units'] = 'cm'

    area = probe_area * u.mm ** 2  # Phil suggested off by x5
    area = area.to(u.m ** 2)
    e_plus = 1.60217663e-19*u.C

    area_val = area.value
    e_plus_val = e_plus.value
    e = np.exp(1)
    isat_clean = isat_data_array.squeeze()
    sound_speed_clean = sound_speed_data_array.squeeze()

    density_data_array = (1e-6*isat_clean*np.sqrt(e)/(sound_speed_clean*area_val*e_plus_val)).rename('density')
    # density_data_array = (isat_data_array / (sound_speed_data_array * area * e_plus)).rename('density')
    density_data_array.attrs['units'] = '$cm^{-3}$'
    if 'z' in isat_data_array.coords:
        sound_speed_data_array = sound_speed_data_array.assign_coords(z=isat_data_array.z)
        density_data_array = density_data_array.assign_coords(z=isat_data_array.z)

    # # --- 2. FINAL NAN CHECK ---
    # print("\n--- FINAL DENSITY DIAGNOSTICS ---")
    # print(f"Total NaNs in density_data_array: {int(density_data_array.isnull().sum())}")
    # print("---------------------------------\n")
    # print("density data array")
    # print(density_data_array)

    return sound_speed_data_array, density_data_array, updated_ds

def get_langmuir_dataset(data):
    """ where data is a flux nc dataset """

    if isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray):
        month, year = data.attrs["Experiment series"].split("_")
        filename = data.attrs["HDF5 file path"] + "lang_nc/lang_" + month[:3] + year[-2:] + "_" + data.attrs["Run name"] + ".nc"
    else:
        filename = data
    interferometry_folder = " "
    langmuir_nc_folder = " "
    interferometry_mode = 'skip'
    bimaxwellian = False
    core_radius = 21. * u.cm
    plot_save_folder = " "
    datasets, _, _ = get_langmuir_datasets(
        langmuir_nc_folder, filename, interferometry_folder, interferometry_mode,
        core_radius, bimaxwellian, plot_save_folder, silent=True)

    datasets = [
        ds.assign_coords(
            probe=("probe", ds["z"].values),
            time=ds["time"] - ds["time"].values[0]
        )
        .drop_vars(["port", "z", "plateau"])
        .rename(probe="z")
        .mean(dim=["y", "face"])
        for ds in datasets
    ]

    return datasets[0]


if __name__ == '__main__':
    november_file = ("/home/michael/Documents/school/Plasma/LAPD Plasma Analysis/HDF5 Files/November_2022_HDF5 and "
                     "NetCDF/18_line_valves95V_7500A.hdf5")
    march_file = ("/home/michael/Documents/school/Plasma/LAPD Plasma Analysis/HDF5 Files/March_2022_HDF5 "
                  "and NetCDF/18_line_valves105V_7000A.hdf5")
    march_file_nc = ("/home/michael/Documents/school/Plasma/LAPD Plasma Analysis/HDF5 Files/March_2022_HDF5 "
                  "and NetCDF/lang_nc/lang_Mar22_18_line_valves105V_7000A.nc")
    march_file_nc_flux = ("/home/michael/Documents/school/Plasma/LAPD Plasma Analysis/HDF5 Files/March_2022_HDF5 "
                     "and NetCDF/flux_nc/18_line_valves105V_7000A.nc")
    january_file = ("/home/michael/Documents/school/Plasma/LAPD Plasma Analysis/HDF5 Files/January_2024_HDF5 and NetCDF/"
                    "26_line_valves100V_5600A_1kG_He 2024-02-06 14.29.08-012.hdf5")

    # march is done

    hdf5_folder = ("/home/michael/Documents/school/Plasma/LAPD Plasma Analysis/HDF5 Files/March_2022_HDF5 "
                  "and NetCDF/")

    flux_nc_folder = hdf5_folder + 'flux_nc/'
    isat_data = get_isat_vf(march_file, hdf5_folder, flux_nc_folder)
    print(isat_data)
    get_time_series(isat_data["density"], time=None, x=(-10, 10), shot=None, z=860, plot=True)
    # print(isat_data.coords)
    # print(isat_data.data_vars)
    #
    #
    # print("success")
    #
    #
    # file = lapd.File(march_file)
    # file.run_description()
    # isat_board = 2
    # isat_channel = 1
    # isat_receptacle = 1
    # isat_data = file.read_data(isat_board, isat_channel, silent=True, add_controls=[('6K Compumotor', isat_receptacle)])
    # print("binted")
    # print(isat_data[0][2][2])
    # val = []
    # for discharge in isat_data:
    #     val.append(np.mean(discharge[1][39000:39500]))
    # plt.plot(val)
    # plt.show()

    # dataset = get_langmuir_dataset(march_file_nc)
    # ds = dataset
    # print(dataset)
    # print(dataset.coords["z"].values)
    # for name, da in ds.data_vars.items():
    #     print(f"{name}: {list(da.dims)} {da.shape}")
    # param = "nu_ei"
    # dim = "x"
    # z_target = 800
    # x = (10, 20)
    # time = (7, 15)
    # shot = None
    # mean_param, std_param = get_langmuir_profiles(ds, param, z_target, x=x, time=time, shot=shot, plot=True)
    # print(float(mean_param.mean(dim="x")))
    # dataset = dataset[param].sel(z=z_target, method="nearest")
    # dataset.mean(dim=[d for d in dataset.dims if d != dim]).plot()
    # plt.show()

    def basic_plot(data, dim):
        data.mean(dim=[d for d in data.dims if d != dim]).plot()

    flux_ds = xr.open_dataset(march_file_nc_flux)
    lang_ds = get_langmuir_dataset(march_file_nc)
    isat = flux_ds["isat"]
    T_e = lang_ds["T_e"]
    print("Isat")
    print(isat)
    print("T_e")
    print(T_e)
    # basic_plot(isat, "time")
    # basic_plot(T_e, "time")

    T_e_data = T_e.sel(z=860.0, method='nearest')
    # basic_plot(T_e_data, "time")
    T_e_data = T_e_data.expand_dims(z=isat.z)

    print(T_e_data)
    print(T_e_data.dims, T_e_data.sizes["time"])

    T_e_interp = extend_dim_repeat(T_e_data, isat, "time")

    T_e_mean = T_e_interp.mean(dim="shot", keep_attrs=True)
    T_e_interp = T_e_mean.broadcast_like(T_e_interp)

    print(T_e_interp)

    basic_plot(T_e_interp, dim="shot")

    get_langmuir_profiles(lang_ds, "T_e", z=860.0, x=None, time=8, shot=None, plot=True)
    basic_plot(T_e_interp.sel(time=8, method="nearest"), "x")
    plt.ylim([.8, 2.2])
    plt.show()