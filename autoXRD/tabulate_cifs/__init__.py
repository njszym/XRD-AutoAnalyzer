import numpy as np
import shutil
import os
import pymatgen as mg
from pymatgen.core import Structure
from pymatgen.analysis import structure_matcher


class StructureFilter(object):
    """
    Class used to parse a list of CIFs and choose unique,
    stoichiometric reference phases that were measured
    under (or nearest to) ambient conditions.
    """

    def __init__(self, cif_directory):
        """
        Args:
            cif_directory: path to directory containing
                the CIF files to be considered as
                possible reference phases
        """

        self.cif_dir = cif_directory

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

        stoich_strucs, temps, dates = [], [], []
        for cmpd in os.listdir(self.cif_dir):
            struc = Structure.from_file('%s/%s' % (self.cif_dir, cmpd))
            if struc.is_ordered:
                stoich_strucs.append(struc)
                t, d = self.parse_measurement_conditions(cmpd)
                temps.append(t)
                dates.append(d)
        return stoich_strucs, temps, dates

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
            sorted_info = sorted(zipped_info, key=lambda x: x[1]) ## Sort by date
            final_struc = sorted_info[-1][0] # Take the entry that was measured most recently
            filtered_cmpds.append(final_struc)

        return filtered_cmpds


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
        sg = struc.get_space_group_info()[1]
        filepath = '%s/%s_%s.cif' % (dir, f, sg)
        struc.to(filename=filepath, fmt='cif')

    assert len(os.listdir(dir)) > 0, 'Something went wrong. No reference phases were found.'



def main(cif_directory, ref_directory, include_elems=True):

    # Get unique structures
    struc_filter = StructureFilter(cif_directory)
    final_refs = struc_filter.filtered_refs

    # Write unique structures (as CIFs) to reference directory
    write_cifs(final_refs, ref_directory, include_elems)

