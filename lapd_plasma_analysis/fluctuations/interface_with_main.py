
from lapd_plasma_analysis.fluctuations.analysis import *
from lapd_plasma_analysis.file_access import ask_yes_or_no

def ask_about_plots(data):
    if ask_yes_or_no("Plot profiles (y/n)?"):
        for quantity in ['density', 'isat', 'vf']:
            if ask_yes_or_no(f"Plot {quantity} profile (y/n)?"):
                get_profile(data[quantity], plot=True)
