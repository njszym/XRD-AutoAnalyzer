import os
import sys
import pymatgen as mg
from autoXRD import tabulate_cifs as tc


check = True
if 'References' in os.listdir('.'):
    if input('References directory exists. Do you wish to overwrite? (y/n)') != 'y':
        check = False

if check == True:

    stoich_refs, temps, dates = tc.get_stoichiometric_info(sys.argv[-1])
    grouped_structs, grouped_temps, grouped_dates = tc.get_unique_struct_info(stoich_refs, temps, dates)
    final_cmpds = tc.get_recent_RT_entry(grouped_structs, grouped_temps, grouped_dates)

    os.mkdir('References')
    for struct in final_cmpds:
        formula = struct.composition.reduced_formula
        f = struct.composition.reduced_formula
        sg = struct.get_space_group_info()[1]
        filepath = 'References/%s_%s.cif' % (f, sg)
        struct.to(filename=filepath, fmt='cif')

