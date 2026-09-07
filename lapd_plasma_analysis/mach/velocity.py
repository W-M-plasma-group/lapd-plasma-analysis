import numpy as np
import xarray as xr
import astropy.units as u
from plasmapy.particles import particle_mass
import os
import matplotlib.pyplot as plt
from lapd_plasma_analysis.obtain_plots.xarray_plots import build_subplots
import matplotlib.ticker as ticker
from lapd_plasma_analysis.file_access import ensure_directory

from lapd_plasma_analysis.langmuir.helper import crunch_data, ion_temperature
from lapd_plasma_analysis.calculation_helpers import sound_speed_calculation
from matplotlib.colors import CenteredNorm
def get_mach_numbers(mach_isat_da: xr.DataArray):
    """
    Returns Dataset of Mach numbers at each position and time increment.
    Dimensions are (probe, face, x, y, shot, time).
    Note that Mach probes have much higher time resolution than the Langmuir measurement frequency.

    Parameters
    ----------
    mach_isat_da : `xr.DataArray`
        DataArray containing saturation current data

    Returns
    -------
    `xr.DataArray`
        Mach numbers for

    Notes
    -----
    By C. Perks and by "Mach probes" (Chung 2012), the parallel Mach number :math:`M_z` is given by

    .. math::

        M_z = M_c \ln(R_1)

    where :math:`M_c` is a magnetization factor :math:`= 1/K` in Chung 2012,
    and :math:`R_1` is the ratio of upstream ion saturation current to downstream ion saturation current.

    By the same sources, using a Gundestrup probe that measures the ion saturation current in multiple directions,
    the perpendicular Mach number :math:`M_\perp` may be calculated as

    .. math::

        M_\perp = M_c \ln(R_1 / R_2) / \cot(\alpha) \\

        \ \ \ \ \ \ = (M_c \ln(R_1)  - M_c \ln(R_2)) \cdot \tan(\alpha)

        \ \ \ \ \ \ = (M_z - M_c \ln(R_2)) \cdot \tan(\alpha)

    where :math:`R_2` is the ratio of more-upstream ion saturation current to more-downstream ion saturation current
    for probe faces lying along an axis at an angle :math:`\alpha` from perpendicular to the flow;
    for example, :math:`\alpha = \pi/2` for a perfect upstream-downstream probe face alignment.
    This model is valid when :math:`\pi/6 < \alpha < 5 \pi/6`.

    In this function, the perpendicular Mach number is found as the average of two estimates for :math:`M_\perp`
    based on probe face axis alignment offsets of :math:`\pi/4` and :math:`3\pi/4` from horizontal.

    """

    #     Model of Mach probe faces (perfect octagon)
    #                         ___________
    #              |         /           \
    #     fore     |    3  /               \  4
    #              |      |                 |
    # (<- Cathode) |   2  |                 |  5            <----  B-field
    #              |      |                 |
    #     aft      |    1  \               /  6
    #              |         \___________/

    """ CONSTANTS AND DESCRIPTIONS ARE TAKEN FROM MATLAB CODE WRITTEN BY CONOR PERKS
        Additional information can be found in, among other sources, 'Mach probes' (Chung 2012) """
    magnetization_factor = 0.5          # Mag. factor value from Hutchinson's derivation incorporating diamagnetic drift
    angle_fore = np.pi / 4 * u.rad      # [rad] Angle the face plane in fore direction makes with B-field (pi/2 head-on)
    angle_aft = np.pi / 4 * u.rad       # [rad] Angle the face plane in aft direction makes with B-field (pi/2 head-on)

    print("Calculating Mach numbers...")

    """Parallel Mach number"""
    parallel_mach = magnetization_factor * np.log(
        mach_isat_da.sel(face=2) / mach_isat_da.sel(face=5))  # .sortby("probe")
    mach_ds = xr.Dataset({"M_para": parallel_mach})                             # Parallel Mach number

    """Perpendicular Mach number"""
    if np.isin(np.array([1, 3, 4, 6]), mach_isat_da.face).all():
        mach_correction_fore = magnetization_factor * np.log(mach_isat_da.sel(face=3) / mach_isat_da.sel(face=6))
        mach_correction_aft = magnetization_factor * np.log(mach_isat_da.sel(face=1) / mach_isat_da.sel(face=4))

        perpendicular_mach_fore = (parallel_mach - mach_correction_fore) * np.tan(angle_fore)
        perpendicular_mach_aft = (parallel_mach - mach_correction_aft) * np.tan(angle_aft)
        perpendicular_mach = (perpendicular_mach_fore + perpendicular_mach_aft) / 2

        mach_ds = mach_ds.assign({"M_perp":      perpendicular_mach,            # Perpendicular Mach number
                                  "M_perp_fore": perpendicular_mach_fore,       # Perpendicular fore Mach number
                                  "M_perp_aft":  perpendicular_mach_aft})       # Perpendicular aft Mach number

    return mach_ds


def get_velocity(mach_ds: xr.Dataset, electron_temperature_da: xr.DataArray, ion_type, ion_temperature):
    """
    Returns Dataset of flow velocity at each position and time.
    Dimensions are (probe, face, x, y, shot, time (matching Langmuir plateaus))

    Notes
    -----
    From MATLAB code by C. Perks:
    "Note that :math:`M=v/C_s` where :math:`C_s = \sqrt{(T_e+T_i)/M_i}`, but we will assume that :math:`T_i \sim 1` eV".
    A supporting 1 eV estimate for LAPD ion temperature was found on the LAPD BAPSF website.

    Parameters
    ----------
    mach_ds : `xr.Dataset`
        WIP
    electron_temperature_da : `xr.DataArray`
        param
    ion_type : str
        param

    Returns
    -------
    `xr.Dataset`
        Dataset containing parallel and, if applicable, perpendicular velocity data
    """

    # Electron temperature DataArray will have dimensions
    #     probe,     (additional coordinates: port, z)
    #     face,
    #     x,
    #     y,
    #     shot,
    #     time      (additional coordinates: plateau (1-based))

    ion_mass = particle_mass(ion_type)
    ion_adiabatic_index = 0
    velocity_unit = u.m / u.s



    # sound_speed = np.sqrt((electron_temperature_da + ion_adiabatic_index * ion_temperature.to(u.eV).value) / ion_mass)  # .sortby("probe")
    # sound_speed *= np.sqrt(1 * u.eV / u.kg).to(velocity_unit).value  # convert speed from sqrt(eV/kg) to [velocity unit]

    sound_speed = sound_speed_calculation(electron_temperature_da, t_i = ion_temperature, ion_type = ion_type,
                                             gamma_i = ion_adiabatic_index)


    crunched_mach_ds = crunch_data(mach_ds, "time", sound_speed.coords['time'])
    crunched_mach_ds.coords['time'] = sound_speed.coords['time']  # Ensure time has units in new dataset'

    # Below: reindex Mach probe data according to nearest Langmuir probe in electron density dataset
    crunched_mach_ds = crunched_mach_ds.swap_dims({"probe": "port"}).reindex_like(
        sound_speed.swap_dims({"probe": "port"}), method="nearest", tolerance=3
    ).swap_dims({"port": "probe"})

    # Parallel velocity
    parallel_velocity = crunched_mach_ds['M_para'] * sound_speed
    parallel_velocity.attrs['units'] = str(velocity_unit)

    velocity = xr.Dataset({"v_para": parallel_velocity})

    print(f"\n--- Diagnostic Evidence for x = {0} cm ---")

    # Isolate a single scalar point in time (taking the first available index of the steady-state window)
    sample_te = electron_temperature_da.sel(x=0, shot = 4,probe = 0, method='nearest').isel(sweep=13)
    sample_cs = sound_speed.sel(x=0, shot = 4, probe = 0, method='nearest').isel(sweep=13)

    # Extract raw scalar values using .item() to avoid array wrapper formatting
    print("Raw Te value from array:      ", sample_te.item())
    print("Calculated sound_speed (Cs):  ", sample_cs.item())
    print("Ion mass variable used:       ", ion_mass)
    print("-------------------------------------------\n")

    # Perpendicular velocity
    if "M_perp" in crunched_mach_ds:
        perpendicular_velocity = crunched_mach_ds['M_perp'] * sound_speed
        perpendicular_velocity.attrs['units'] = str(velocity_unit)
        velocity = velocity.assign({"v_perp": perpendicular_velocity})

    return velocity


def check_isats_contour(mach_ds: xr.Dataset, figure_folder, title_identifier=None, face_up=2, face_down=5,
                        save_fig=True):
    """
    Generate 2D contour plots for Upstream Isat, Downstream Isat, and their Ratio
    using build_subplots with identical colorbars formatted to 2 decimal places.
    """
    if title_identifier is None:
        title_identifier = mach_ds.attrs.get("description", "Mach Probe Isat")

    # 1. Average across shot dimension
    upstream = mach_ds['isat'].sel(face=face_up)
    downstream = mach_ds['isat'].sel(face=face_down)

    up_mean = upstream.mean(dim='shot', skipna=True).squeeze()
    down_mean = downstream.mean(dim='shot', skipna=True).squeeze()
    ratio = up_mean / down_mean

    # Extract unit metadata
    time_unit = mach_ds.attrs.get("time_units", "s")
    x_unit = mach_ds.attrs.get("x_units", "cm")
    isat_unit = mach_ds['isat'].attrs.get('units', 'A')

    # Iterate through probes
    for probe in mach_ds['probe'].values:
        run_folder = ensure_directory(figure_folder + f'{title_identifier}/')
        probe_folder = ensure_directory(run_folder + f'probe_{probe}/')
        up_data = up_mean.sel(probe=probe)
        down_data = down_mean.sel(probe=probe)
        ratio_data = ratio.sel(probe=probe)


        # Unified dynamic colorbar bounds across BOTH upstream & downstream
        combined_isat = np.concatenate([up_data.values.ravel(), down_data.values.ravel()])
        valid_isat = combined_isat[~np.isnan(combined_isat)]

        if len(valid_isat) > 0:
            v_min, v_max = np.nanpercentile(valid_isat, [2, 98])
        else:
            v_min, v_max = 0.0, 1.0
        v_min = 0.0  # Force lower bound to 0

        # Uniform ticks across both Isat plots
        isat_ticks = np.linspace(v_min, v_max, 6)

        # Ratio dynamic bounds & ticks
        valid_ratio = ratio_data.values.ravel()[~np.isnan(ratio_data.values.ravel())]
        if len(valid_ratio) > 0:
            r_min, r_max = np.nanpercentile(valid_ratio, [2, 98])
        else:
            r_min, r_max = 0.5, 2.0
        ratio_ticks = np.linspace(r_min, r_max, 6)

        # Build subplots: 2 on top row, 1 wide on bottom row
        fig, ax, letters = build_subplots(layout=[[2], [1]], fig_width=6.0, fig_height=4.5)
        ax_up, ax_down, ax_ratio = ax[letters[0]], ax[letters[1]], ax[letters[2]]

        # Upstream Plot
        up_data.plot(
            ax=ax_up,
            x='time',
            y='x',
            vmin=v_min,
            vmax=v_max,
            cmap='turbo',
            add_colorbar=True,
            cbar_kwargs={
                'label': '',
                'ticks': isat_ticks,
                'format': '%.2f'  # <--- Forces 2 decimal places
            }
        )
        ax_up.set_title(f"Upstream $|I_{{sat}}|$ ({isat_unit}) - Face {face_up}", fontsize=16)

        # Downstream Plot
        down_data.plot(
            ax=ax_down,
            x='time',
            y='x',
            vmin=v_min,
            vmax=v_max,
            cmap='turbo',
            add_colorbar=True,
            cbar_kwargs={
                'label': '',
                'ticks': isat_ticks,
                'format': '%.2f'  # <--- Forces 2 decimal places
            }
        )
        ax_down.set_title(f"Downstream $|I_{{sat}}|$ ({isat_unit}) - Face {face_down}", fontsize=16)

        # Ratio Plot
        ratio_data.plot(
            ax=ax_ratio,
            x='time',
            y='x',
            vmin=r_min,
            vmax=r_max,
            cmap='coolwarm',
            add_colorbar=True,
            cbar_kwargs={
                'label': '',
                'ticks': ratio_ticks,
                'format': '%.2f'  # <--- Forces 2 decimal places
            }
        )
        ax_ratio.set_title(f"$I_{{sat}}$ Ratio ($I_{{upstream}} / I_{{downstream}}$)", fontsize=16)



        # Axis labeling across subplots
        for ltr in letters:
            curr_ax = ax[ltr]
            curr_ax.set_xlabel(f'Time ({time_unit})', fontsize=14)
            curr_ax.set_ylabel(f'x ({x_unit})', fontsize=14)

        # Suptitle & Layout formatting
        probe_z = mach_ds['z'].sel(probe=probe).item() if 'z' in mach_ds else "Unknown"
        fig.suptitle(f"{title_identifier}\nProbe {probe} $I_{{sat}}$ Contours (z: {probe_z} m)", fontsize=20)
        plt.tight_layout()


        # Save figure
        if figure_folder and save_fig:
            os.makedirs(probe_folder, exist_ok=True)
            clean_identifier = str(title_identifier).replace(" ", "_").replace("/", "_")
            filename = f"isat_contour_{clean_identifier}_probe_{probe}_full_avg.png"
            save_path = os.path.join(probe_folder, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to: {save_path}")
        plt.close()

        # Check shot specific stuff
        for shot in upstream['shot'].values:
            up_data = upstream.sel(shot = shot, probe=probe)
            down_data = downstream.sel(shot = shot, probe=probe)
            ratio_data = up_data/down_data
            mach_data = 0.5 * np.log(ratio_data)
            # Unified dynamic colorbar bounds across BOTH upstream & downstream
            combined_isat = np.concatenate([up_data.values.ravel(), down_data.values.ravel()])
            valid_isat = combined_isat[~np.isnan(combined_isat)]

            if len(valid_isat) > 0:
                v_min, v_max = np.nanpercentile(valid_isat, [2, 98])
            else:
                v_min, v_max = 0.0, 1.0
            v_min = 0.0  # Force lower bound to 0

            # Uniform ticks across both Isat plots
            isat_ticks = np.linspace(v_min, v_max, 6)

            # Ratio dynamic bounds & ticks
            valid_ratio = ratio_data.values.ravel()[~np.isnan(ratio_data.values.ravel())]
            if len(valid_ratio) > 0:
                r_min, r_max = np.nanpercentile(valid_ratio, [2, 98])
            else:
                r_min, r_max = 0.5, 2.0

            ratio_ticks = np.linspace(r_min, r_max, 6)

            mach_raveled = mach_data.values.ravel()
            valid_mach = mach_raveled[~np.isnan(mach_raveled)]



            # Build subplots: 2 on top row, 2 wide on bottom row
            fig, ax, letters = build_subplots(layout=[[2], [2]], fig_width=6.0, fig_height=4.5)
            ax_up, ax_down, ax_ratio, ax_mach = ax[letters[0]], ax[letters[1]], ax[letters[2]], ax[letters[3]]

            # Upstream Plot
            up_data.plot(
                ax=ax_up,
                x='time',
                y='x',
                vmin=v_min,
                vmax=v_max,
                cmap='turbo',
                add_colorbar=True,
                cbar_kwargs={
                    'label': '',
                    'ticks': isat_ticks,
                    'format': '%.2f'  # <--- Forces 2 decimal places
                }
            )
            ax_up.set_title(f"Upstream $|I_{{sat}}|$ ({isat_unit}) - Face {face_up}", fontsize=16)

            # Downstream Plot
            down_data.plot(
                ax=ax_down,
                x='time',
                y='x',
                vmin=v_min,
                vmax=v_max,
                cmap='turbo',
                add_colorbar=True,
                cbar_kwargs={
                    'label': '',
                    'ticks': isat_ticks,
                    'format': '%.2f'  # <--- Forces 2 decimal places
                }
            )
            ax_down.set_title(f"Downstream $|I_{{sat}}|$ ({isat_unit}) - Face {face_down}", fontsize=16)

            # Ratio Plot
            ratio_data.plot(
                ax=ax_ratio,
                x='time',
                y='x',
                vmin=r_min,
                vmax=r_max,
                cmap="RdBu_r",
                add_colorbar=True,
                cbar_kwargs={
                    'label': '',
                    'ticks': ratio_ticks,
                    'format': '%.2f'  # <--- Forces 2 decimal places
                }
            )
            ax_ratio.set_title(f"$I_{{sat}}$ Ratio ($I_{{upstream}} / I_{{downstream}}$)", fontsize=16)

            # Mach Plot
            if len(valid_mach) > 0:
                m_max = np.nanpercentile(np.abs(valid_mach), 98)
            else:
                m_max = 1.0
            mach_ticks = np.linspace(-m_max, m_max, 7)

            mach_data.plot(
                ax=ax_mach,
                x='time',
                y='x',
                norm=CenteredNorm(
                    vcenter=0.0, halfrange=m_max
                ),  # <--- Forces 0.0 strictly to neutral white
                # vmin=m_min,
                # vmax=m_max,
                cmap="RdBu_r",
                add_colorbar=True,
                cbar_kwargs={
                    'label': '',
                    'ticks': mach_ticks,
                    'format': '%.2f'  # <--- Forces 2 decimal places
                }
            )

            ax_mach.set_title(f"Mach number", fontsize=16)
            # Axis labeling across subplots
            for ltr in letters:
                curr_ax = ax[ltr]
                curr_ax.set_xlabel(f'Time ({time_unit})', fontsize=14)
                curr_ax.set_ylabel(f'x ({x_unit})', fontsize=14)

            # Suptitle & Layout formatting
            probe_z = mach_ds['z'].sel(probe=probe).item() if 'z' in mach_ds else "Unknown"
            fig.suptitle(f"{title_identifier}\n$I_{{sat}}$ Contours (z: {probe_z} m), shot = {shot}", fontsize=20)
            plt.tight_layout()

            # Save figure
            if figure_folder and save_fig:
                os.makedirs(probe_folder, exist_ok=True)
                clean_identifier = str(title_identifier).replace(" ", "_").replace("/", "_")
                filename = f"isat_contour_{clean_identifier}_probe_{probe}_shot_{shot}.png"
                save_path = os.path.join(probe_folder, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved figure to: {save_path}")
            plt.close()