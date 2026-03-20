import os
from lapd_plasma_analysis.fluctuations.fourier import *
from scipy.ndimage import uniform_filter1d

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

def get_isat_vf(filename, hdf5_path, flux_nc_folder):
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

    file = lapd.File(filename)

    config_id = get_config_id(file.info['exp name'])

    # scaling = (resistance * gain)**(-1)
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
    for isat_board, isat_channel, isat_receptacle, isat_port, isat_scaling, probe_area in isat_probes:
        isat_data = file.read_data(isat_board, isat_channel, silent=True, add_controls=[('6K Compumotor', isat_receptacle)])
        dt = isat_data.dt.value
        times = np.linspace(0, len(isat_data[0][1])*dt, len(isat_data[0][1]))
        z = [isat_data[0][2][2]]

        isat_vs_x = []
        isat_vs_x.append([])
        x_index = 0
        x_array = []
        x_array.append(isat_data[0][2][0])
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
        isat_data_array = xr.DataArray(isat_vs_x*u.ampere, dims = ['z', 'x', 'shot', 'time'],
                                                           coords = {'x': np.round(x_array)*u.cm,
                                                                     'shot': shot_array,
                                                                     'time': times*u.s.to(u.ms),
                                                                     'z': np.round(z)*u.cm},
                                                           name = 'isat',
                                                           attrs={'units': 'A'})
        isat_data_array.coords['x'].attrs['units'] = 'cm'
        isat_data_array.coords['time'].attrs['units'] = 'ms'
        isat_data_arrays.append(isat_data_array)

        sound_speed_data_array, density_data_array = get_density_data(file, hdf5_path, isat_data_array, shot_array,
                                                                      probe_area, z)
        density_data_arrays.append(density_data_array)

    isat_data_array = xr.concat(isat_data_arrays, dim="z").sortby("z")
    density_data_array = xr.concat(density_data_arrays, dim="z").sortby("z")

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

    isat_vf_dataset.to_netcdf(flux_nc_folder +
                              filename.replace(hdf5_path, '').replace('.hdf5', '') + '.nc')
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
    """
    d1_coords = d1[dim].values
    d2_coords = d2[dim].values

    idxs = np.searchsorted(d1_coords, d2_coords, side="right") - 1
    idxs = np.clip(idxs, 0, len(d1_coords) - 1)

    def _extend_array(da: xr.DataArray) -> xr.DataArray:
        """Extend a single DataArray by repeating along dim."""
        axis = da.get_axis_num(dim)
        extended_data = np.take(da.values, idxs, axis=axis)
        new_coords = dict(da.coords)
        new_coords[dim] = d2_coords

        return xr.DataArray(
            extended_data,
            dims=da.dims,
            coords=new_coords,
            attrs=da.attrs,
            name=da.name,
        )

    if isinstance(d1, xr.DataArray):
        return _extend_array(d1)

    elif isinstance(d1, xr.Dataset):
        extended_vars = {}
        for var in d1.data_vars:
            da = d1[var]
            if dim in da.dims:
                extended_vars[var] = _extend_array(da)
            else:
                extended_vars[var] = da
        return xr.Dataset(extended_vars, attrs=d1.attrs)

    else:
        raise TypeError("Input d1 must be an xarray.DataArray or xarray.Dataset")


def get_density_data(fileobj, hdf5_folder, isat_data_array, shot_array, probe_area, z):
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

    month, year = fileobj.info["exp name"].split("_")
    prefix = "lang_" + month[:3] + year[-2:] + "_"

    lang_nc_file_path = hdf5_folder + "lang_nc/" + prefix + fileobj.info["run name"] + ".nc"

    dataset = get_langmuir_dataset(lang_nc_file_path)

    c = 299792458*u.m/u.s
    M = 4.002603254 * (931.49410372*u.MeV/(c**2)).to(u.eV*u.s*u.s/u.m/u.m) #helium
    # print("mass: ", M.to(u.kg))
    T_e_data = dataset['T_e'].sel(z=z, method='nearest')
    T_e_data = T_e_data.assign_coords(z=isat_data_array.z)

    # print("T_e_data:", T_e_data)
    T_e_interp = extend_dim_repeat(T_e_data, isat_data_array, "time")
    T_e_mean = T_e_interp.mean(dim="shot", keep_attrs=True)
    T_e_interp = T_e_mean.broadcast_like(T_e_interp)
    sound_speed_data_array = np.sqrt(T_e_interp/M).rename('sound_speed')

    sound_speed_data_array.attrs['units'] = 'm/s'
    sound_speed_data_array.coords['time'].attrs['units'] = 'ms'
    sound_speed_data_array.coords['x'].attrs['units'] = 'cm'

    area = probe_area * u.mm ** 2  # Phil suggested off by x5
    area = area.to(u.m ** 2)
    e_plus = 1.60217663e-19*u.C
    e = np.exp(1)

    density_data_array = (1e-6*isat_data_array*np.sqrt(e)/(sound_speed_data_array*area*e_plus)).rename('density')
    # density_data_array = (isat_data_array / (sound_speed_data_array * area * e_plus)).rename('density')
    density_data_array.attrs['units'] = '$cm^{-3}$'
    # print("density data array")
    # print(density_data_array)

    return sound_speed_data_array, density_data_array

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

