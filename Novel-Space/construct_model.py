import os
import sys
import pymatgen as mg
from autoXRD import *


check = True
if 'References' in os.listdir('.'):
    if input('References directory already exists. Do you wish to overwrite? (y/n)') != 'y':
        check = False

if 'XRD.npy' in os.listdir('.'):
    if input('XRD.npy already exists. Do you wish to overwrite? (y/n)') != 'y':
        check = False

if check == True:

    stoich_refs, temps, dates = tabulate_cifs.get_stoichiometric_info(sys.argv[-1])
    grouped_structs, grouped_temps, grouped_dates = tabulate_cifs.get_unique_struct_info(stoich_refs, temps, dates)
    final_cmpds = tabulate_cifs.get_recent_RT_entry(grouped_structs, grouped_temps, grouped_dates)

    os.mkdir('References')
    for struct in final_cmpds:
        formula = struct.composition.reduced_formula
        f = struct.composition.reduced_formula
        sg = struct.get_space_group_info()[1]
        filepath = 'References/%s_%s.cif' % (f, sg)
        struct.to(filename=filepath, fmt='cif')

    assert len(os.listdir('References')) > 0:, 'Something went wrong. No references phases were found.'

    xrd = generate_spectra.get_augmented_patterns('References')
    np.save('XRD', xrd)

    cnn.train_model(xrd)
