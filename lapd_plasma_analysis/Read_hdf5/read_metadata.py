import re


def get_gas_puff_voltage(desc):
    '''

        Parameters
        ----------
        desc - String containing experimental parameters from lapd.File(...).info

        Returns
        -------
        gp_voltages - Dictionary of east and west gas puff voltage values in V
    '''
    gp_voltages = {}

    lines = desc.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if 'east' in line.lower():
            gp_east = re.findall(r'(\d+(?:\.\d+)?)\s*V\s*from east',
                                 line,
                                 re.IGNORECASE)
            if gp_east:
                gp_voltages['GPV east'] = float(gp_east[0])
        if 'west' in line.lower():
            gp_west = re.findall(r'(\d+(?:\.\d+)?)\s*V\s*from west',
                                 line,
                                 re.IGNORECASE)
            if gp_west:
                gp_voltages['GPV west'] = float(gp_west[0])

        # For the Mar 2022 and Nov 2022 datasets which list the gas puff voltage as both
        if 'both' in line.lower() and 'v' in line.lower():
            gp_both = re.findall(r'Both\s+at\s*(\d+(?:\.\d*)?)\s*V', line, re.IGNORECASE)
            if gp_both:
                gp_voltages['GPV west'] = float(gp_both[0])
                gp_voltages['GPV east'] = float(gp_both[0])

    return gp_voltages

def get_cathode_current(desc):
    '''

    Parameters
    ----------
    desc - String containing experimental parameters from lapd.File(...).info

    Returns
    -------
    cathode_currents - Dictionary of cathode current values in A
    '''
    cathode_currents = {}

    lines = desc.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Search for everything succeeded by 'A cathode current'
        curr_matches = re.findall(r'(\d+(?:\.\d+)?)\s*A\s*discharge current',
                                  line,
                                  re.IGNORECASE)
        if not curr_matches:
            curr_matches = re.findall(r'Idis\s*=\s*(\d+(?:\.\d+)?)\s*A',
                                      line,
                                      re.IGNORECASE)

        # Only works if discharge current is mentioned once otherwise it will pick the last one
        if curr_matches:
            cathode_currents['Cathode Current'] = float(curr_matches[0])

    return cathode_currents


def get_bfield_dict(desc):
    '''

    Parameters
    ----------
    desc - String containing experimental parameters from lapd.File(...).info

    Returns
    -------
    magnet_fields - Dictionary of magnetic field values for each set of magnets in kG
    '''

    # The Mar 2022 and Nov 2022 datasets label the magnets by south Source and main chamber this allows for the
    # interpretation to be consistent across all datasets
    location_map = {
        "south source": "Black South",
        "main chamber": "Magenta"
    }

    magnet_fields = {}

    lines = desc.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # if magnet_fields:
        #     continue

        # ---- Black magnets ----
        # Looks for a line that starts with the word Black. It then separates the line into 2 groups:
        # 1) The magnet descriptor (North or South) up to the colon 2) the numerical value in kG.
        # Optionally the run may contain something in parentheses after the colon. The second set of
        # parentheses here tells the code to ignore that
        black_match = re.match(r'(Black.*?):(?:\s*\(.*?\))?\s*([\d.]+)\s*kG', line, re.IGNORECASE)
        if black_match:
            # black_match.group(0) is the entire matched text
            name = black_match.group(1).strip()
            value = float(black_match.group(2))
            # Normalize north/south
            lname = name.lower()
            if "south" in lname:
                magnet_fields["Black South"] = value
            elif "north" in lname:
                magnet_fields["Black North"] = value
            else:
                magnet_fields[name] = value
            continue

        # ---- Magenta/Yellow magnets ----
        # Find all kG values and optional parentheses colors

        # Searches for all numerical data that comes before kG in the line. If there is no matches it
        # returns an empty list
        kg_matches = re.findall(r'([\d.]+)\s*kG', line, re.IGNORECASE)

        # Find all values within the line that are within parentheses. \w captures the letters, numbers
        # and underscores. \s captures the spaces. If there is no matches it returns an empty list
        paren_colors = re.findall(r'\(([\w\s]+)\)', line, re.IGNORECASE)

        # If there is at least one thing in parentheses
        if paren_colors:
            # Assign values to colors in parentheses
            # Zip pairs a color with a parentheses. If any list is empty then zip stops at the shorter
            # one. If it is empty then it won't return anything
            for value, color in zip(kg_matches, paren_colors):
                # For each color in the zip file assign a key value pair in the dictionary
                magnet_fields[color.capitalize()] = float(value)
        else:
            # No parentheses: check if line mentions magenta/yellow
            line_lower = line.lower()
            if "magenta" in line_lower:
                magnet_fields["Magenta"] = float(kg_matches[0])
            if "yellow" in line_lower:
                # If only one value, assign that; if two, take second
                if len(kg_matches) > 1:
                    magnet_fields["Yellow"] = float(kg_matches[1])
                else:
                    magnet_fields["Yellow"] = float(kg_matches[0])

        # For the Mar 22 and Nov 22 data
        if not magnet_fields:
            matches = re.findall(r'([\d.]+)\s*kG\s*([\w\s]+?)(?=\s*(?:to|$))', line, re.IGNORECASE)
            magnet_fields = {location_map.get(loc.strip().lower(), loc.strip()): float(val)
                             for val, loc in matches}
            if magnet_fields:
                magnet_fields["Yellow"] = magnet_fields["Magenta"]
                magnet_fields["Black North"] = float(0)

    return magnet_fields

def get_gas_type(desc):
    '''

    Parameters
    ----------
    desc - String containing experimental parameters from lapd.File(...).info

    Returns
    -------
    gas_params - Dictionary containing gas type and the pressure injected at
    '''
    gas_params = {}

    lines = desc.splitlines()
    for line in lines:
        line = line.strip()
        # Match gas type and PSI anywhere in the line requires the gas type to be Hydrogen Helium, or Argon
        m = re.search(
            r'^(?:Using\s+)?(H2|He|Ar|Ne|O2)\b(?:\s+only)?\s*(?:at)?\s*~?([\d.]+)\s*PSI',
            line,
            flags=re.IGNORECASE
        )
        if m:
            gas_params['Gas type'] = m.group(1)
            gas_params['Gas pressure'] = float(m.group(2))

    if not gas_params:
        gas_params['Gas type'] = 'He'

    return gas_params

def metadata_dict(desc, exp_name, filename):
    '''

    Parameters
    ----------
    desc - String containing experimental parameters from lapd.File(...).info

    Returns
    -------
    params_dict - Dictionary containing all metadata obtained from lapd.File(...).info

    '''

    # Gets the gas puff voltages from the east and the west in V
    gp_voltages = get_gas_puff_voltage(desc)

    # Gets the cathode current in A
    cathode_currents = get_cathode_current(desc)

    # Gets the gas type and the gas pressure in PSI
    gas_params = get_gas_type(desc)

    # Gets magnetic fields of the different coils in kG
    b_fields = get_bfield_dict(desc)

    params_dict = gp_voltages | cathode_currents | gas_params | b_fields

    params_dict['Exp name'] = exp_name

    # Get the run number within the experiment
    filename = filename.split('_')

    if ('mar' in exp_name.lower()
        and '22' in exp_name.lower()):
        run_num = filename[1]

    elif ('nov' in exp_name.lower()
          and '22' in exp_name.lower()):
        run_num = filename[0]

    elif ('jan' in exp_name.lower()
          and '24' in exp_name.lower()):
        run_num = filename[0]

    else:
        run_num = filename[1]

    params_dict['Run number'] = run_num

    return params_dict

