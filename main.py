from hdf5reader import *
from getIVsweep import *
from astropy import visualization
from pprint import pprint

print("Finished importing helper files")
characteristic = getIsweepVsweep('HDF5/8-3500A.hdf5')
print("Finished calculating I-V plasma characteristic")

characteristic2 = smooth_characteristic(characteristic, 12)

# bias, current = np.load(path)
# Create the Characteristic object, taking into account the correct units
# characteristic = Characteristic(u.Quantity(bias,  u.V), u.Quantity(current, u.A))

"""
with visualization.quantity_support():
    plt.plot(characteristic.bias, characteristic.current)
    plt.plot(characteristic2.bias, characteristic2.current)
    #plt.plot(characteristic.bias[13000:13400], characteristic.current[13000:13400])
    plt.show()
"""

# Calculate the cylindrical probe surface area
probe_length = 1.145 * u.mm
probe_diameter = 1.57 * u.mm
probe_area = (probe_length * np.pi * probe_diameter + np.pi * 0.25 * probe_diameter**2)

#probe_area = 1/(1000)**2


# pprint(swept_probe_analysis(characteristic, probe_area, 'He-4+', visualize=True, plot_EEDF=True))
pprint(swept_probe_analysis(characteristic2, probe_area, 'He-4+', visualize=True, plot_EEDF=True))
plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
