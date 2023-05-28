from pymatgen.core import Structure, Composition
from pymatgen.analysis import structure_matcher
from shutil import copytree
import pymatgen as mg
import numpy as np
import shutil
import os


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
            struc = Structure.from_file('%s/%s' % (self.cif_dir, cmpd))
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

        unique_frameworks = []
        for struc_1 in stoich_strucs:
            unique = True
            for struc_2 in unique_frameworks:
                if matcher.fit(struc_1, struc_2):
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

        oxi_guesses = Composition(formula).oxi_state_guesses()

        if len(oxi_guesses) > 0:

            check_list = []
            for oxi_dict in oxi_guesses:
                plausible = True
                for elem in oxi_dict.keys():
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
            struc.to(filename=filepath, fmt='cif')
        except:
            try:
                print('%s Space group cant be determined, lowering tolerance' % str(f))
                sg = struc.get_space_group_info(symprec=0.1, angle_tolerance=5.0)[1]
                filepath = '%s/%s_%s.cif' % (dir, f, sg)
                struc.to(filename=filepath, fmt='cif')
            except:
                print('%s Space group cant be determined even after lowering tolerance, Setting to None' % str(f))

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

