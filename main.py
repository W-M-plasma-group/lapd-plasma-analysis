import matplotlib.pyplot as plt

from hdf5reader import *
from getIVsweep import *
from astropy import visualization
from pprint import pprint

print("Finished importing helper files")
plateau = getIsweepVsweep('HDF5/8-3500A.hdf5')
print("Finished calculating I-V plasma characteristic")

plateau.plot()
plt.show()

# characteristic_plateau = Characteristic(characteristic.bias[25000:25620], characteristic.current[25000:25620])

smooth_plateau = smooth_characteristic(plateau, 12)

# Create the Characteristic object, taking into account the correct units
# characteristic = Characteristic(u.Quantity(bias,  u.V), u.Quantity(current, u.A))

# """
with visualization.quantity_support():
    plt.plot(plateau.bias, plateau.current)
    plt.plot(smooth_plateau.bias, smooth_plateau.current)
    plt.show()
# """

"""
# Calculate the cylindrical probe surface area
probe_length = 1.145 * u.mm
probe_diameter = 1.57 * u.mm
probe_area = (probe_length * np.pi * probe_diameter + np.pi * 0.25 * probe_diameter**2)
"""

# probe_area = 1/(1000)**2 (From MATLAB code)
probe_area = (1.*u.mm)**2


# pprint(swept_probe_analysis(characteristic, probe_area, 'He-4+', visualize=True, plot_EEDF=True))
pprint(swept_probe_analysis(smooth_plateau, probe_area, 'He-4+', visualize=True, plot_EEDF=True))
plt.show()
