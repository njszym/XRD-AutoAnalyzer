from autoXRD import cnn, spectrum_generation, solid_solns, tabulate_cifs
import numpy as np
import os
import sys
import pymatgen as mg


check = True
if 'References' in os.listdir('.'):
    if input('References directory already exists. Do you wish to overwrite? (y/n)') != 'y':
        check = False

if 'XRD.npy' in os.listdir('.'):
    if input('XRD.npy already exists. Do you wish to overwrite? (y/n)') != 'y':
        check = False

if check == True:

    # Filter CIF files to create unique reference phases
    tabulate_cifs.main('All_CIFs', 'References')

    # Generate hypothetical solid solutions
    if '--include_ns' in sys.argv:
        solid_solns.main('References')

    # Simulate and save augmented XRD spectra
    xrd_obj = spectrum_generation.SpectraGenerator('References')
    xrd_specs = xrd_obj.augmented_spectra
    np.save('XRD', xrd_specs)

    # Train, test, and save the CNN
    cnn.main(xrd_specs, testing_fraction=0.2)
