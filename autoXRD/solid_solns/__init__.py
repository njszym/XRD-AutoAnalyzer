from itertools import combinations as comb
from pymatgen.core import PeriodicSite
from pymatgen.analysis import structure_matcher
import warnings
from pymatgen.core import periodic_table as pt
import numpy as np
import os
import pymatgen as mg
from pymatgen.core import Composition
from pymatgen import analysis
import multiprocessing
from multiprocessing import Pool, Manager
from pymatgen.core import Structure


class SolidSolnsGen(object):
    """
    Class used to generate hypothetical solid solutions by interpolating
    from a list of stoichiometric reference phases
    """

    def __init__(self, reference_directory):
        """
        Args:
            reference_directory: path to directory containing
                the stoichiometric reference phases (CIFs)
        """

        self.ref_dir = reference_directory
        self.matcher = structure_matcher.StructureMatcher(
            scale=True, attempt_supercell=True, primitive_cell=False,
            comparator=mg.analysis.structure_matcher.FrameworkComparator())
        self.num_cpu = multiprocessing.cpu_count()

    @property
    def soluble_pairs(self):
        """
        Returns:
            matching_pairs: list of tuples containing pairs of soluble compounds
                (denoted by their filenames)
        """

        # Only included ordered structures
        ordered_refs = []
        for fname in os.listdir(self.ref_dir):
            struc = Structure.from_file('%s/%s' % (self.ref_dir, fname))
            if struc.is_ordered:
                ordered_refs.append(fname)
        all_pairs = list(comb(ordered_refs, 2))

        # Get soluble pairs
        with Manager() as manager:
            pool = Pool(self.num_cpu)
            matching_pairs = pool.map(self.are_soluble, all_pairs)
            matching_pairs = [pair for pair in matching_pairs if pair != None]
            return matching_pairs


    def are_soluble(self, pair):
        """
        Predict whether a pair of compounds are soluble with one another.

        Args:
            pair_info: a tuple containing containing two compounds
                denoted by their filenames
        Returns:
            The pair of compounds, if they are soluble.
            Otherwise, Nonetype is returned.
        """

        reference_directory = self.ref_dir

        cmpd_A, cmpd_B = pair[0], pair[1]
        struc_A = Structure.from_file('%s/%s' % (reference_directory, cmpd_A))
        formula_A = struc_A.composition.reduced_formula
        struc_B = Structure.from_file('%s/%s' % (reference_directory, cmpd_B))
        formula_B = struc_B.composition.reduced_formula

        if formula_A != formula_B:
            if self.matcher.fit(struc_A, struc_B):
                solubility = True
                comp_A = Composition(formula_A)
                comp_B = Composition(formula_B)
                probable_oxis_A = comp_A.oxi_state_guesses()
                probable_oxis_B = comp_B.oxi_state_guesses()
                if len(probable_oxis_A) == 0: ## This occurs for metals, in which case we'll just use atomic radii
                    oxi_dict_A = {}
                    for elem in [str(f) for f in comp_A.elements]:
                        oxi_dict_A[elem] = 0.0
                else: ## Take most probable list of oxidation states
                    oxi_dict_A = probable_oxis_A[0]
                if len(probable_oxis_B) == 0:
                    oxi_dict_B = {}
                    for elem in [str(f) for f in comp_B.elements]:
                        oxi_dict_B[elem] = 0.0
                else:
                    oxi_dict_B = probable_oxis_B[0]
                try:
                    struc_B = self.matcher.get_s2_like_s1(struc_A, struc_B)
                except ValueError: ## Sometimes order matters
                    struc_A = self.matcher.get_s2_like_s1(struc_B, struc_A)
                index = 0
                for (site_A, site_B) in zip(struc_A, struc_B):
                    elem_A = ''.join([char for char in str(site_A.species_string) if char.isalpha()])
                    elem_B = ''.join([char for char in str(site_B.species_string) if char.isalpha()])
                    if solubility == True:
                        site_A_oxi = oxi_dict_A[elem_A]
                        site_B_oxi = oxi_dict_B[elem_B]
                        if site_A_oxi.is_integer():
                            if site_A_oxi == 0:
                                possible_rA = [pt.Element(elem_A).atomic_radius]
                            else:
                                possible_rA = [pt.Element(elem_A).ionic_radii[site_A_oxi]]
                        else:
                            possible_rA = []
                            possible_rA.append(pt.Element(elem_A).ionic_radii[int(np.floor(site_A_oxi))])
                            possible_rA.append(pt.Element(elem_A).ionic_radii[int(np.ceil(site_A_oxi))])
                        if site_B_oxi.is_integer():
                            if site_B_oxi == 0:
                                possible_rB = [pt.Element(elem_B).atomic_radius]
                            else:
                                possible_rB = [pt.Element(elem_B).ionic_radii[site_B_oxi]]
                        else:
                            possible_rB = []
                            possible_rB.append(pt.Element(elem_B).ionic_radii[int(np.floor(site_B_oxi))])
                            possible_rB.append(pt.Element(elem_B).ionic_radii[int(np.ceil(site_B_oxi))])
                        possible_diffs = []
                        for rA in possible_rA:
                            for rB in possible_rB:
                                possible_diffs.append(abs(float(rA) - float(rB))/max([float(rA), float(rB)]))
                        if min(possible_diffs) > 0.15:
                            solubility = False

                if solubility == True:
                    return [cmpd_A, cmpd_B]

    def generate_solid_solns(self, pair, num_solns=3):
        """
        From a given pair of soluble compounds, interpolate a list
        of solid solutions.

        Args:
            pair: a list containing two compounds denoated by
                their filenames
            num_solns: number of solid solutions to interpolate
                between each pair of compounds
        Returns:
            interp_structs: a list of interpolated solids solutions
                (pymatgen Structure objects)
        """

        struc_A = Structure.from_file('%s/%s' % (self.ref_dir, pair[0]))
        struc_B = Structure.from_file('%s/%s' % (self.ref_dir, pair[1]))

        try:
            struc_B = self.matcher.get_s2_like_s1(struc_A, struc_B)
        except ValueError:
            struc_A = self.matcher.get_s2_like_s1(struc_B, struc_A)

        # If a good match cannot be made, ignore this pair
        if (struc_A is None) or (struc_B is None):
            return None, None

        else:

            # Create dummy structures but save original species and occupancies (needed for interpolation)
            index = 0
            A_species = []
            for site in struc_A:
                site_dict = site.as_dict()
                A_species.append(site_dict['species'][0]['element'])
                site_dict['species'] = []
                site_dict['species'].append({'element': 'Li', 'oxidation_state': 0.0, 'occu': 1.0})
                struc_A[index] = PeriodicSite.from_dict(site_dict)
                index += 1

            index = 0
            B_species = []
            for site in struc_B:
                site_dict = site.as_dict()
                B_species.append(site_dict['species'][0]['element'])
                site_dict['species'] = []
                site_dict['species'].append({'element': 'Li', 'oxidation_state': 0.0, 'occu': 1.0})
                struc_B[index] = PeriodicSite.from_dict(site_dict)
                index += 1

            # Interpolate structure but ignore the first returned entry (this is just an end-member)
            interp_structs = struc_A.interpolate(struc_B, nimages=num_solns, interpolate_lattices=True)[1:]

            soln_interval = 1.0/(num_solns + 1)
            soln_fractions = [0.0, soln_interval]
            while not np.isclose(soln_fractions[-1], 1.0, atol=0.01):
                next_fraction = round(soln_fractions[-1] + soln_interval, 2)
                soln_fractions.append(next_fraction)
            soln_fractions = soln_fractions[1:-1] # Remove 0.0 and 1.0

            index = 0
            for (A, B) in zip(A_species, B_species):
                if A == B:
                    for i in range(num_solns):
                        site_dict = interp_structs[i][index].as_dict()
                        site_dict['species'] = []
                        site_dict['species'].append({'element': A, 'oxidation_state': 0.0, 'occu': 1.0})
                        interp_structs[i][index] = PeriodicSite.from_dict(site_dict)

                else:
                    for i in range(num_solns):
                        site_dict = interp_structs[i][index].as_dict()
                        site_dict['species'] = []
                        c1 = 1 - soln_fractions[i]
                        c2 = soln_fractions[i]
                        site_dict['species'].append({'element': A, 'oxidation_state': 0.0, 'occu': c1})
                        site_dict['species'].append({'element': B, 'oxidation_state': 0.0, 'occu': c2})
                        interp_structs[i][index] = PeriodicSite.from_dict(site_dict)

                index += 1

            return interp_structs

    @property
    def all_solid_solns(self):
        """
        Returns:
            all_solid_solns: a list of interpolated pymatgen Structure
                objects associated with hypothetical solid solutions
        """

        soluble_pairs = self.soluble_pairs

        all_solid_solns = []
        for pair in soluble_pairs:
            solid_solutions = self.generate_solid_solns(pair)
            if solid_solutions != None:
                for struc in solid_solutions:
                    all_solid_solns.append(struc)

        return all_solid_solns



def main(reference_directory):

    ns_generator = SolidSolnsGen(reference_directory)

    solid_solns = ns_generator.all_solid_solns

    for struc in solid_solns:

        # Name file according to its composition and space group
        filepath = '%s/%s_%s.cif' % (reference_directory, struc.composition.reduced_formula,
            struc.get_space_group_info()[1])

        # Do not write if a known stoichiometric reference phase already exists
        if filepath.split('/')[1] not in os.listdir(reference_directory):
            struc.to(filename=filepath, fmt='cif')


