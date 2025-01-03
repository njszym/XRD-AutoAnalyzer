from pymatgen.core import Structure, Composition, PeriodicSite
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from scipy.signal import find_peaks, filtfilt, resample
from itertools import combinations_with_replacement
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.analysis import structure_matcher
from scipy.ndimage import gaussian_filter1d
from itertools import product
from scipy import interpolate
from functools import reduce
from shutil import copytree
from pyts import metrics
import pymatgen as mg
import numpy as np
import itertools
import shutil
import math
import time
import os
import re
import copy


common_oxi = {
    'H': [1, -1],  # Hydrogen
    'He': [0],  # Helium
    'Li': [1],  # Lithium
    'Be': [2],  # Beryllium
    'B': [3],  # Boron
    'C': [-4, 4],  # Carbon
    'N': [-3],  # Nitrogen
    'O': [-2],  # Oxygen
    'F': [-1],  # Fluorine
    'Ne': [0],  # Neon
    'Na': [1],  # Sodium
    'Mg': [2],  # Magnesium
    'Al': [3],  # Aluminum
    'Si': [-4, 4],  # Silicon
    'P': [-3, 3, 5],  # Phosphorus
    'S': [-2, 4, 6],  # Sulfur
    'Cl': [-1],  # Chlorine
    'Ar': [0],  # Argon
    'K': [1],  # Potassium
    'Ca': [2],  # Calcium
    'Sc': [3],  # Scandium
    'Ti': [2, 3, 4],  # Titanium
    'V': [2, 3, 4, 5],  # Vanadium
    'Cr': [2, 3, 6],  # Chromium
    'Mn': [2, 3, 4, 7],  # Manganese
    'Fe': [2, 3],  # Iron
    'Co': [2, 3],  # Cobalt
    'Ni': [2],  # Nickel
    'Cu': [1, 2],  # Copper
    'Zn': [2],  # Zinc
    'Ga': [3],  # Gallium
    'Ge': [2, 4],  # Germanium
    'As': [-3, 3, 5],  # Arsenic
    'Se': [-2, 4, 6],  # Selenium
    'Br': [-1],  # Bromine
    'Kr': [0],  # Krypton
    'Rb': [1],  # Rubidium
    'Sr': [2],  # Strontium
    'Y': [3],  # Yttrium
    'Zr': [4],  # Zirconium
    'Nb': [3, 5],  # Niobium
    'Mo': [2, 3, 4, 5, 6],  # Molybdenum
    'Tc': [7],  # Technetium
    'Ru': [2, 3, 4, 6, 8],  # Ruthenium
    'Rh': [2, 3, 4],  # Rhodium
    'Pd': [2, 4],  # Palladium
    'Ag': [1], # Silver
    'Cd': [2],  # Cadmium
    'In': [3],  # Indium
    'Sn': [2, 4],  # Tin
    'Sb': [-3, 3, 5],  # Antimony
    'Te': [-2, 4, 6],  # Tellurium
    'I': [-1],  # Iodine
    'Xe': [0],  # Xenon
    'Cs': [1],  # Cesium
    'Ba': [2],  # Barium
    'La': [3],  # Lanthanum
    'Ce': [3, 4],  # Cerium
    'Pr': [3],  # Praseodymium
    'Nd': [3],  # Neodymium
    'Pm': [3],  # Promethium
    'Sm': [2, 3],  # Samarium
    'Eu': [2, 3],  # Europium
    'Gd': [3],  # Gadolinium
    'Tb': [3, 4],  # Terbium
    'Dy': [3],  # Dysprosium
    'Ho': [3],  # Holmium
    'Er': [3],  # Erbium
    'Tm': [2, 3],  # Thulium
    'Yb': [2, 3],  # Ytterbium
    'Lu': [3],  # Lutetium
    'Hf': [4],  # Hafnium
    'Ta': [5],  # Tantalum
    'W': [2, 3, 4, 5, 6],  # Tungsten
    'Re': [2, 3, 4, 6, 7],  # Rhenium
    'Os': [2, 3, 4, 6, 8],  # Osmium
    'Ir': [2, 3, 4, 6],  # Iridium
    'Pt': [2, 4],  # Platinum
    'Au': [1, 3],  # Gold
    'Hg': [1, 2],  # Mercury
    'Tl': [1, 3],  # Thallium
    'Pb': [2, 4],  # Lead
    'Bi': [3, 5],  # Bismuth
    'Th': [4],  # Thorium
    'Pa': [5],  # Protactinium
    'U': [3, 4, 5, 6],  # Uranium
    'Np': [3, 4, 5, 6, 7],  # Neptunium
    'Pu': [3, 4, 5, 6, 7, 8],  # Plutonium
    'Am': [2, 3, 4, 5, 6],  # Americium
    'Cm': [3],  # Curium
    'Bk': [3, 4],  # Berkelium
    'Cf': [2, 3, 4],  # Californium
    'Es': [3],  # Einsteinium
    'Fm': [3],  # Fermium
    'Md': [2, 3],  # Mendelevium
    'No': [2, 3],  # Nobelium
    'Lr': [3],  # Lawrencium
    'Rf': [4],  # Rutherfordium
    'Db': [5],  # Dubnium
    'Sg': [6],  # Seaborgium
    'Bh': [7],  # Bohrium
    'Hs': [8],  # Hassium
}

def calc_std_dev(two_theta, tau):
    """
    calculate standard deviation based on angle (two theta) and domain size (tau)
    Args:
        two_theta: angle in two theta space
        tau: domain size in nm
    Returns:
        standard deviation for gaussian kernel
    """
    ## Calculate FWHM based on the Scherrer equation
    K = 0.9 ## shape factor
    wavelength = XRDCalculator().wavelength * 0.1 ## angstrom to nm
    theta = np.radians(two_theta/2.) ## Bragg angle in radians
    beta = (K * wavelength) / (np.cos(theta) * tau) # in radians

    ## Convert FWHM to std deviation of gaussian
    sigma = np.sqrt(1/(2*np.log(2)))*0.5*np.degrees(beta)
    return sigma**2

def remap_pattern(angles, intensities):

    steps = np.linspace(10, 100, 4501)
    signals = np.zeros([len(angles), steps.shape[0]])

    for i, ang in enumerate(angles):
        # Map angle to closest datapoint step
        idx = np.argmin(np.abs(ang-steps))
        signals[i,idx] = intensities[i]
    domain_size = 25.0
    step_size = (100-10)/4501
    for i in range(signals.shape[0]):
        row = signals[i,:]
        ang = steps[np.argmax(row)]
        std_dev = calc_std_dev(ang, domain_size)
        # Gaussian kernel expects step size 1 -> adapt std_dev
        signals[i,:] = gaussian_filter1d(row, np.sqrt(std_dev)*1/step_size, mode='constant')

    # Combine signals
    signal = np.sum(signals, axis=0)

    # Normalize signal
    norm_signal = 100 * signal / max(signal)

    # Combine signals
    signal = np.sum(signals, axis=0)

    # Normalize signal
    norm_signal = 100 * signal / max(signal)

    return norm_signal

def smooth_spectrum(spectrum, n=20):
    """
    Process and remove noise from the spectrum.

    Args:
        spectrum: list of intensities as a function of 2-theta
        n: parameters used to control smooth. Larger n means greater smoothing.
            20 is typically a good number such that noise is reduced while
            still retaining minor diffraction peaks.
    Returns:
        smoothed_ys: processed spectrum after noise removal
    """

    # Smoothing parameters defined by n
    b = [1.0 / n] * n
    a = 1

    # Filter noise
    smoothed_ys = filtfilt(b, a, spectrum)

    return smoothed_ys

def strip_spectrum(warped_spectrum, orig_y):
    """
    Subtract one spectrum from another. Note that when subtraction produces
    negative intensities, those values are re-normalized to zero. This way,
    the CNN can handle the spectrum reliably.

    Args:
        warped_spectrum: spectrum associated with the identified phase
        orig_y: original (measured) spectrum
    Returns:
        fixed_y: resulting spectrum from the subtraction of warped_spectrum
            from orig_y
    """

    # Subtract predicted spectrum from measured spectrum
    stripped_y = orig_y - warped_spectrum

    # Normalize all negative values to 0.0
    fixed_y = []
    for val in stripped_y:
        if val < 0:
            fixed_y.append(0.0)
        else:
            fixed_y.append(val)

    return fixed_y

def scale_spectrum(pred_y, obs_y):
    """
    Scale the magnitude of a calculated spectrum associated with an identified
    phase so that its peaks match with those of the measured spectrum being classified.

    Args:
        pred_y: spectrum calculated from the identified phase after fitting
            has been performed along the x-axis using DTW
        obs_y: observed (experimental) spectrum containing all peaks
    Returns:
        scaled_spectrum: spectrum associated with the reference phase after scaling
            has been performed to match the peaks in the measured pattern.
    """

    # Ensure inputs are numpy arrays
    pred_y = np.array(pred_y)
    obs_y = np.array(obs_y)

    # Find scaling constant that minimizes MSE between pred_y and obs_y
    all_mse = []
    for scale_spectrum in np.linspace(1.1, 0.05, 101):
        ydiff = obs_y - (scale_spectrum*pred_y)
        mse = np.mean(ydiff**2)
        all_mse.append(mse)
    best_scale = np.linspace(1.0, 0.05, 101)[np.argmin(all_mse)]
    scaled_spectrum = best_scale*np.array(pred_y)

    return scaled_spectrum, best_scale

def get_reduced_pattern(y1, y2, last_normalization=1.0):
    """
    Subtract y1 from y2 using dynamic time warping (DTW) and return the new spectrum.
    Returns:
        stripped_y: new spectrum obtained by subtrating the peaks of the identified phase
    """

    # Convert to numpy arrays
    pred_y = np.array(y1)
    orig_y = np.array(y2)

    # Downsample spectra (helps reduce time for DTW)
    downsampled_res = 0.1 # new resolution: 0.1 degrees
    num_pts = int((100-10) / downsampled_res)
    orig_y = resample(orig_y, num_pts)
    pred_y = resample(pred_y, num_pts)

    # Calculate window size for DTW
    allow_shifts = 0.75 # Allow shifts up to 0.75 degrees
    window_size = int(allow_shifts * num_pts / (100-10))

    # Get warped spectrum (DTW)
    distance, path = metrics.dtw(pred_y, orig_y, method='sakoechiba', options={'window_size': window_size}, return_path=True)
    index_pairs = path.transpose()
    warped_spectrum = orig_y.copy()
    for ind1, ind2 in index_pairs:
        distance = abs(ind1 - ind2)
        if distance <= window_size:
            warped_spectrum[ind2] = pred_y[ind1]
        else:
            warped_spectrum[ind2] = 0.0

    # Now, upsample spectra back to their original size (4501)
    warped_spectrum = resample(warped_spectrum, 4501)
    orig_y = resample(orig_y, 4501)

    # Scale warped spectrum so y-values match measured spectrum
    scaled_spectrum, scaling_constant = scale_spectrum(warped_spectrum, orig_y)

    # Subtract scaled spectrum from measured spectrum
    stripped_y = strip_spectrum(scaled_spectrum, orig_y)
    stripped_y = smooth_spectrum(stripped_y)
    stripped_y = np.array(stripped_y) - min(stripped_y)

    return stripped_y

def round_dict_values(data):
    """
    Used to round off coefficients
    of highly complex formulae.
    """

    for key, value in data.items():
        if value > 1e5:
            data[key] = round(value, -3)
        elif value > 1e4:
            data[key] = round(value, -2)
        elif value > 1e3:
            data[key] = round(value, -1)

    # Reduce coefficients by gcd
    gcd = reduce(math.gcd, list(data.values()))
    for key in data:
        data[key] = int(data[key]/gcd)

    return data

def parse_formula(formula):

    # Convert to alphabetical (no parentheses)
    c = Composition(formula)
    formula = c.alphabetical_formula.replace(' ', '')

    element_pattern = r'([A-Z][a-z]*)(\d*)'
    compound_pattern = r'\(([A-Z][a-z]*\d*)\)(\d*)'

    # Expand compounds in parentheses
    while '(' in formula:
        match = re.search(compound_pattern, formula)
        compound, multiplier = match.groups()
        expanded = ''.join(f"{element}{int(count)*int(multiplier)}" for element, count in re.findall(element_pattern, compound))
        formula = formula.replace(match.group(), expanded)

    # Parse elements and their counts
    parsed = re.findall(element_pattern, formula)
    counts = {element: int(count) if count else 1 for element, count in parsed}

    multi_oxi = False
    for elem in counts.keys():
        if len(common_oxi[elem]) > 1:
            multi_oxi = True

    """
    If coefficients of the chemical formula are unreasonably large, reduce them by rounding.
    However, only do this in the case of multiple oxidation states per element.
    Without rounding, these can lead to combinatorial explosion.
    """
    if multi_oxi:
        counts = round_dict_values(counts)

    return counts

def balance_oxidation_states(formula, oxidation_states, max_time=10):
    """
    Note: this is *not* an exhaustive oxidation state solver.
    Rather, it will find if there exists at least one solution
    that satisfies charge balance given the possible oxidation
    states. This method is fast and suitable for the current
    application; however, caution should be used if one
    implements it outside of XRD-AutoAnalyzer.
    """
    element_counts = parse_formula(formula)
    elements = list(element_counts.keys())

    balanced_combinations = []

    for el in elements:
        if len(oxidation_states[el]) > 1:
            multi_valent_element = el
            multi_valent_count = element_counts[el]
            possible_states = oxidation_states[el]
            start_time=time.time()
            for combination in combinations_with_replacement(possible_states, multi_valent_count):
                current_time=time.time()
                sum_states = sum([oxidation_states[el][0]*element_counts[el] for el in elements if el != multi_valent_element])
                sum_states += sum(combination)
                if current_time - start_time > max_time:
                    break
                if sum_states == 0:
                    unique_combination = tuple(set(combination))
                    if len(unique_combination) == 1:
                        unique_combination = unique_combination[0]
                    balanced_combinations.append(
                        {**{el: oxidation_states[el][0] for el in elements if el != multi_valent_element},
                         multi_valent_element: unique_combination})
                    break

    if not balanced_combinations:  # no multivalent elements or no solution found yet
        all_state_combinations = product(*[oxidation_states[el] for el in elements])

        for state_combination in all_state_combinations:
            if sum(element_counts[el] * state for el, state in zip(elements, state_combination)) == 0:
                balanced_combination = dict(zip(elements, state_combination))
                if balanced_combination not in balanced_combinations:
                    balanced_combinations.append(balanced_combination)

    return balanced_combinations

class StructureFilter(object):
    """
    Class used to parse a list of CIFs and choose unique,
    stoichiometric reference phases that were measured
    under (or nearest to) ambient conditions.
    """

    def __init__(self, cif_directory, enforce_order):
        """
        Args:
            cif_directory: path to directory containing
                the CIF files to be considered as
                possible reference phases
        """

        self.cif_dir = cif_directory
        self.enforce_order = enforce_order

    @property
    def stoichiometric_info(self):
        """
        Filter strucures to include only those which do not have
        fraction occupancies and are ordered. For those phases, tabulate
        the measurement conditions of the associated CIFs.

        Returns:
            stoich_strucs: a list of ordered pymatgen Structure objects
            temps: temperatures that each were measured at
            dates: dates the measurements were reported
        """

        strucs, temps, dates = [], [], []
        for cmpd in os.listdir(self.cif_dir):
            # Allowing some tolerance in site occupancies
            parser = CifParser('%s/%s' % (self.cif_dir, cmpd), occupancy_tolerance=1.25)
            struc = parser.parse_structures()[0]
            if self.enforce_order:
                if struc.is_ordered:
                    strucs.append(struc)
                    t, d = self.parse_measurement_conditions(cmpd)
                    temps.append(t)
                    dates.append(d)
            else:
                strucs.append(struc)
                t, d = self.parse_measurement_conditions(cmpd)
                temps.append(t)
                dates.append(d)

        return strucs, temps, dates

    def parse_measurement_conditions(self, filename):
        """
        Parse the temperature and date from a CIF file

        Args:
            filename: filename of CIF to be parsed
        Returns:
            temp: temperature at which measurement was conducted
            date: date which measurement was reported
        """

        temp, date = 0.0, None
        with open('%s/%s' % (self.cif_dir, filename)) as entry:
            for line in entry.readlines():
                if '_audit_creation_date' in line:
                    date = line.split()[-1]
                if '_cell_measurement_temperature' in line:
                    temp = float(line.split()[-1])
        return temp, date

    @property
    def unique_struc_info(self):
        """
        Create distinct lists of Structure objects where each
        list is associated with a unique strucural prototype

        Returns:
            grouped_strucs: a list of sub-lists containing pymatgen
                Structure objects organize by the strucural prototype
            grouped_temps and grouped_dates: similarly grouped temperatures and dates
                associated with the corresponding measurements
        """

        stoich_strucs, temps, dates = self.stoichiometric_info

        matcher = structure_matcher.StructureMatcher(scale=True, attempt_supercell=True, primitive_cell=False)

        XRD_calculator = XRDCalculator(wavelength='CuKa', symprec = 0.0)

        unique_frameworks = []
        for struc_1 in stoich_strucs:

            unique = True
            for struc_2 in unique_frameworks:

                # Check if structures are identical. If so, exclude.
                if matcher.fit(struc_1, struc_2):
                    unique = False

                # Check if compositions are similar If so, check structural framework.
                temp_struc_1 = struc_1.copy()
                reduced_comp_1_dict = temp_struc_1.composition.remove_charges().reduced_composition.to_reduced_dict
                divider_1 = 1

                for key in reduced_comp_1_dict:
                    divider_1 = max(divider_1,reduced_comp_1_dict[key])

                reduced_comp_1 = temp_struc_1.composition.remove_charges().reduced_composition/divider_1
                temp_struc_2 = struc_2.copy()
                reduced_comp_2_dict = temp_struc_2.composition.remove_charges().reduced_composition.to_reduced_dict
                divider_2 = 1

                for key in reduced_comp_2_dict:
                    divider_2 = max(divider_2,reduced_comp_2_dict[key])
                reduced_comp_2 = temp_struc_2.composition.remove_charges().reduced_composition/divider_2

                if reduced_comp_1.almost_equals(reduced_comp_2, atol=0.5):

                    # Replace with dummy species (H) for structural framework check.
                    temp_struc_1 = struc_1.copy()
                    for index,site in enumerate(temp_struc_1.sites):
                        site_dict = site.as_dict()
                        site_dict['species'] = []
                        site_dict['species'].append({'element': 'H', 'oxidation_state': 0.0, 'occu': 1.0}) # dummy species
                        temp_struc_1[index]=PeriodicSite.from_dict(site_dict)
                    temp_struc_2 = struc_2.copy()
                    for index,site in enumerate(temp_struc_2.sites):
                        site_dict = site.as_dict()
                        site_dict['species'] = []
                        site_dict['species'].append({'element': 'H', 'oxidation_state': 0.0, 'occu': 1.0}) # dummy species
                        temp_struc_2[index]=PeriodicSite.from_dict(site_dict)

                    # Checking structural framework.
                    if matcher.fit(temp_struc_1, temp_struc_2):

                        # Before excluding, check if their XRD patterns differ.
                        """
                        This check is necessary as sometimes materials with identical compositions can adopt the
                        same structural framework but still differ in their XRD, e.g., when individual site
                        occupancies differ between them. For example, site inversion in spinels.

                        Accordingly, we still include identical structures/compositions in the cases
                        where their XRD patterns differ by some predefined amount.
                        """
                        y_1 = remap_pattern(XRD_calculator.get_pattern(struc_1, scaled=True, two_theta_range=(10,100)).x, XRD_calculator.get_pattern(struc_1, scaled=True, two_theta_range=(10,100)).y)
                        y_2 = remap_pattern(XRD_calculator.get_pattern(struc_2, scaled=True, two_theta_range=(10,100)).x, XRD_calculator.get_pattern(struc_2, scaled=True, two_theta_range=(10,100)).y)
                        reduced_pattern=np.array(get_reduced_pattern(y_1,y_2))

                        # If 20% peak intensity remains after subtracting one pattern from the other.
                        diff_threshold = 20.0
                        if (reduced_pattern < diff_threshold).all():
                            unique = False

            if unique:
                unique_frameworks.append(struc_1)

        grouped_strucs, grouped_temps, grouped_dates = [], [], []
        for framework in unique_frameworks:
            struc_class, temp_class, date_class = [], [], []
            for (struc, t, d) in zip(stoich_strucs, temps, dates):
                if matcher.fit(framework, struc):
                    struc_class.append(struc)
                    temp_class.append(t)
                    date_class.append(d)

            grouped_strucs.append(struc_class)
            grouped_temps.append(temp_class)
            grouped_dates.append(date_class)

        return grouped_strucs, grouped_temps, grouped_dates

    @property
    def filtered_refs(self):
        """
        For each list of strucures associated with a strucural prototype,
        choose that which was measured under (or nearest to) ambient conditions
        and which was reported most recently. Priority is given to the former.

        Returns:
            filtered_cmpds: a list of unique pymatgen Structure objects
        """

        grouped_strucs, grouped_temps, grouped_dates = self.unique_struc_info

        filtered_cmpds = []
        for (struc_class, temp_class, date_class) in zip(grouped_strucs, grouped_temps, grouped_dates):
            normalized_temps = abs(np.array(temp_class) - 293.0) # Difference from RT
            zipped_info = list(zip(struc_class, normalized_temps, date_class))
            sorted_info = sorted(zipped_info, key=lambda x: x[1]) # Sort by temperature
            best_entry = sorted_info[0] # Take the entry measured at the temperature closest to RT
            candidate_strucs, candidate_dates = [], []
            for entry in sorted_info:
                if entry[1] == best_entry[1]: # If temperature matches best entry
                    candidate_strucs.append(entry[0])
                    candidate_dates.append(entry[2])
            zipped_info = list(zip(candidate_strucs, candidate_dates))
            try:
                sorted_info = sorted(zipped_info, key=lambda x: x[1]) ## Sort by date
                final_struc = sorted_info[-1][0] # Take the entry that was measured most recently
            # If no dates available
            except TypeError:
                final_struc = zipped_info[-1][0]
            filtered_cmpds.append(final_struc)

        return filtered_cmpds


def oxi_filter(cif_dir):
    """
    Removes any reference compounds that have
    unusual oxidation states.

    Args:
        cif_dir: directory containing CIFs
    """

    for filename in os.listdir(cif_dir):

        oxi_okay = False

        struc = Structure.from_file('%s/%s' % (cif_dir, filename))
        formula = struc.composition.get_integer_formula_and_factor()[0]

        oxi_guesses = balance_oxidation_states(formula, common_oxi)

        if len(oxi_guesses) > 0:

            check_list = []
            for oxi_dict in oxi_guesses:
                plausible = True
                for elem in oxi_dict.keys():
                    if type(oxi_dict[elem]) is tuple:
                        for oxi_state in oxi_dict[elem]:
                            if int(oxi_state) not in common_oxi[elem]:
                                plausible = False
                    else:
                        if int(oxi_dict[elem]) not in common_oxi[elem]:
                            plausible = False
                check_list.append(plausible)

            if True in check_list:
                oxi_okay = True

        if not oxi_okay:
            os.remove('%s/%s' % (cif_dir, filename))


def write_cifs(unique_strucs, dir, include_elems):
    """
    Write structures to CIF files

    Args:
        strucs: list of pymatgen Structure objects
        dir: path to directory where CIF files will be written
    """

    if not os.path.isdir(dir):
        os.mkdir(dir)

    for struc in unique_strucs:
        num_elems = struc.composition.elements
        if num_elems == 1:
            if not include_elems:
                continue
        formula = struc.composition.reduced_formula
        f = struc.composition.reduced_formula
        try:
            sg = struc.get_space_group_info()[1]
            filepath = '%s/%s_%s.cif' % (dir, f, sg)
            cif_writer = CifWriter(structure)
            cif_writer.write_file(filename)
        except:
            try:
                print('%s Space group cannot be determined, lowering tolerance' % str(f))
                sg = struc.get_space_group_info(symprec=0.1, angle_tolerance=5.0)[1]
                filepath = '%s/%s_%s.cif' % (dir, f, sg)
                cif_writer = CifWriter(structure)
                cif_writer.write_file(filename)
            except:
                print('%s Space group cannot be determined even after lowering tolerance, Setting to None' % str(f))

    assert len(os.listdir(dir)) > 0, 'Something went wrong. No reference phases were found.'

def main(cif_directory, ref_directory, filter_oxi=False, include_elems=True, enforce_order=False):

    if filter_oxi:
        copytree(cif_directory, 'Filtered_CIFs')
        oxi_filter('Filtered_CIFs')
        cif_directory = 'Filtered_CIFs'

    # Get unique structures
    struc_filter = StructureFilter(cif_directory, enforce_order)
    final_refs = struc_filter.filtered_refs

    # Write unique structures (as CIFs) to reference directory
    write_cifs(final_refs, ref_directory, include_elems)

