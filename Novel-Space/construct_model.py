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

    ## Filter unique reference phases from CIF list
    if len(sys.argv) != 1:
        stoich_refs, temps, dates = tabulate_cifs.get_stoichiometric_info(sys.argv[-1])
    else:
        stoich_refs, temps, dates = tabulate_cifs.get_stoichiometric_info('All_CIFs')
    grouped_structs, grouped_temps, grouped_dates = tabulate_cifs.get_unique_struct_info(stoich_refs, temps, dates)
    final_cmpds = tabulate_cifs.get_recent_RT_entry(grouped_structs, grouped_temps, grouped_dates)

    ## Write structure files to reference folder
    os.mkdir('References')
    for struct in final_cmpds:
        formula = struct.composition.reduced_formula
        f = struct.composition.reduced_formula
        sg = struct.get_space_group_info()[1]
        filepath = 'References/%s_%s.cif' % (f, sg)
        struct.to(filename=filepath, fmt='cif')
    assert len(os.listdir('References')) > 0:, 'Something went wrong. No reference phases were found.'

    ## Generate hypothetical solid solutions
    soluble_phases = tabulate_soluble_pairs('References')
    for pair in soluble_phases:
        solid_solutions = generate_solid_solns(pair)
        if solid_solutions != None:
            for struct in solid_solutions:
                filepath = 'References/%s_%s.cif' % (struct.composition.reduced_formula, struct.get_space_group_info()[1])
                if filepath.split('/')[1] not in os.listdir('References'): ## Give preference to known references
                    struct.to(filename=filepath, fmt='cif')

    ## Simulate augmented XRD spectra from all reference phases
    xrd = generate_spectra.get_augmented_patterns('References')
    np.save('XRD', xrd)

    ## Train and save the CNN using the simulated XRD spectra
    cnn.train_model(xrd)
