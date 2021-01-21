import warnings
from pymatgen.core import periodic_table as pt
import numpy as np
import os
import pymatgen as mg
from pymatgen import Composition
from pymatgen.analysis import structure_matcher as sm


matcher = sm.StructureMatcher(scale=True, attempt_supercell=True, primitive_cell=False, comparator=mg.analysis.structure_matcher.FrameworkComparator())
nn_analyzer = mg.analysis.local_env.CrystalNN()
matching_pairs = []
for cmpd_A in os.listdir('References/'):
    struct_A = mg.Structure.from_file('References/%s' % cmpd_A)
    formula_A = struct_A.composition.reduced_formula
    for cmpd_B in os.listdir('References/'):
        struct_B = mg.Structure.from_file('References/%s' % cmpd_B)
        formula_B = struct_B.composition.reduced_formula
        if formula_A != formula_B:
            if matcher.fit(struct_A, struct_B):
                print(formula_A, formula_B)
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
                    struct_B = matcher.get_s2_like_s1(struct_A, struct_B)
                except ValueError: ## Sometimes order matters
                    struct_A = matcher.get_s2_like_s1(struct_B, struct_A)
                index = 0
                for (site_A, site_B) in zip(struct_A, struct_B):
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
                    matching_pairs.append([cmpd_A, cmpd_B])

unique_pairs = []
for pair in matching_pairs:
    if set(pair) not in unique_pairs:
        unique_pairs.append(set(pair))
os.mkdir('Solid_Solns')
for pair in unique_pairs:
    print(pair)
    pair = list(pair)
    struct_A = mg.Structure.from_file('References/%s' % pair[0])
    struct_B = mg.Structure.from_file('References/%s' % pair[1])
    try:
        struct_B = matcher.get_s2_like_s1(struct_A, struct_B)
    except ValueError:
        struct_A = matcher.get_s2_like_s1(struct_B, struct_A)
    if (struct_A is None) or (struct_B is None): ## If a good match could not be made, ignore this pair
        continue
    else:
        index = 0
        A_species = []
        for site in struct_A: ## Create dummy structure but save original species and occupancies (needed for interpolation)
            site_dict = site.as_dict()
            A_species.append(site_dict['species'][0]['element'])
            site_dict['species'] = []
            site_dict['species'].append({'element': 'Li', 'oxidation_state': 0.0, 'occu': 1.0})
            struct_A[index] = mg.PeriodicSite.from_dict(site_dict)
            index += 1
        index = 0
        B_species = []
        for site in struct_B: ## Create dummy structure but save original species and occupancies (needed for interpolation)
            site_dict = site.as_dict()
            B_species.append(site_dict['species'][0]['element'])
            site_dict['species'] = []
            site_dict['species'].append({'element': 'Li', 'oxidation_state': 0.0, 'occu': 1.0})
            struct_B[index] = mg.PeriodicSite.from_dict(site_dict)
            index += 1
        interp_structs = struct_A.interpolate(struct_B, nimages=3, interpolate_lattices=True)
        index = 0
        for (A, B) in zip(A_species, B_species):
            if A == B:
                site_dict = interp_structs[1][index].as_dict()
                site_dict['species'] = []
                site_dict['species'].append({'element': A, 'oxidation_state': 0.0, 'occu': 1.0})
                interp_structs[1][index] = mg.PeriodicSite.from_dict(site_dict)
                site_dict = interp_structs[2][index].as_dict()
                site_dict['species'] = []
                site_dict['species'].append({'element': A, 'oxidation_state': 0.0, 'occu': 1.0})
                interp_structs[2][index] = mg.PeriodicSite.from_dict(site_dict)
                site_dict = interp_structs[3][index].as_dict()
                site_dict['species'] = []
                site_dict['species'].append({'element': A, 'oxidation_state': 0.0, 'occu': 1.0})
                interp_structs[3][index] = mg.PeriodicSite.from_dict(site_dict)
            else:
                site_dict = interp_structs[1][index].as_dict()
                site_dict['species'] = []
                site_dict['species'].append({'element': A, 'oxidation_state': 0.0, 'occu': 0.75})
                site_dict['species'].append({'element': B, 'oxidation_state': 0.0, 'occu': 0.25})
                interp_structs[1][index] = mg.PeriodicSite.from_dict(site_dict)
                site_dict = interp_structs[2][index].as_dict()
                site_dict['species'] = []
                site_dict['species'].append({'element': A, 'oxidation_state': 0.0, 'occu': 0.5})
                site_dict['species'].append({'element': B, 'oxidation_state': 0.0, 'occu': 0.5})
                interp_structs[2][index] = mg.PeriodicSite.from_dict(site_dict)
                site_dict = interp_structs[3][index].as_dict()
                site_dict['species'] = []
                site_dict['species'].append({'element': A, 'oxidation_state': 0.0, 'occu': 0.25})
                site_dict['species'].append({'element': B, 'oxidation_state': 0.0, 'occu': 0.75})
                interp_structs[3][index] = mg.PeriodicSite.from_dict(site_dict)
            index += 1
        os.chdir('Solid_Solns/')
        fname = '%s_%s.cif' % (interp_structs[1].composition.reduced_formula, interp_structs[1].get_space_group_info()[1])
        interp_structs[1].to(filename=fname, fmt='cif')
        fname = '%s_%s.cif' % (interp_structs[2].composition.reduced_formula, interp_structs[2].get_space_group_info()[1])
        interp_structs[2].to(filename=fname, fmt='cif')
        fname = '%s_%s.cif' % (interp_structs[3].composition.reduced_formula, interp_structs[3].get_space_group_info()[1])
        interp_structs[3].to(filename=fname, fmt='cif')
        os.chdir('../')
