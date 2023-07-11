import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 160


def value_safe(quantity_or_scalar):  # Get value of quantity or scalar, depending on type

    try:
        val = quantity_or_scalar.value  # input is a quantity with dimension and value
    except AttributeError:
        val = quantity_or_scalar  # input is a dimensionless scalar with no value
    return val


def unit_safe(quantity_or_scalar):  # Get unit of quantity or scalar, if possible

    try:
        unit = quantity_or_scalar.unit
    except AttributeError:
        unit = None  # The input data is dimensionless
    return unit
