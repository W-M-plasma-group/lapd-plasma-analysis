import matplotlib.pyplot as plt

from getIVsweep import *
from astropy import visualization
from pprint import pprint

print("Imported helper files")
# plateau = get_isweep_vsweep('HDF5/8-3500A.hdf5')
sample_sec = (100 / 16 * 10 ** 6) ** (-1) * u.s  # Note that this is a small number. 10^6 is in the denominator
filename = 'HDF5/8-3500A.hdf5'
plateau = create_ranged_characteristic(filename, 25000, 25620)
print("Finished creating I-V plasma Characteristic object")

smooth_plateau = smooth_characteristic(plateau, 7)

"""
# debug
with visualization.quantity_support():
    # plt.plot(plateau.bias, plateau.current)
    # plt.plot(smooth_plateau.bias, smooth_plateau.current)
    plateau.plot()
    smooth_plateau.plot()
    plt.show()
"""

# Calculate the cylindrical probe surface area
# probe_area = 1/(1000)**2 (From MATLAB code)
probe_area = (1.*u.mm)**2


# pprint(swept_probe_analysis(characteristic, probe_area, 'He-4+', visualize=True, plot_EEDF=True))
# Remember to make ion type customizable by user
pprint(swept_probe_analysis(smooth_plateau, probe_area, 'He-4+', visualize=True, plot_EEDF=True))
plt.show()

# bias, current = get_isweep_vsweep(filename)
# print(split_plateaus(bias, current, isolate_plateaus(bias, current)))
