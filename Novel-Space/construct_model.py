from autoXRD import cnn, spectrum_generation, solid_solns, tabulate_cifs
import numpy as np
import os
import sys
import pymatgen as mg


if __name__ == '__main__':

    max_texture = 0.5 # default: texture associated with up to +/- changes in peak intensities
    min_domain_size, max_domain_size = 1.0, 100.0 # default: domain sizes ranging from 1 to 100 nm
    max_strain = 0.04 # default: up to +/- 4% strain
    num_spectra = 50 # Number of spectra to simulate per phase
    for arg in sys.argv:
        if '--max_texture' in arg:
            max_texture = float(arg.split('=')[1])
        if '--min_domain_size' in arg:
            min_domain_size = float(arg.split('=')[1])
        if '--max_domain_size' in arg:
            max_domain_size = float(arg.split('=')[1])
        if '--max_strain' in arg:
            max_strain = float(arg.split('=')[1])
        if '--num_spectra' in arg:
            num_spectra = float(arg.split('=')[1])

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
        xrd_obj = spectrum_generation.SpectraGenerator('References', num_spectra, max_texture, min_domain_size, max_domain_size, max_strain)
        xrd_specs = xrd_obj.augmented_spectra
        np.save('XRD', xrd_specs)

        # Train, test, and save the CNN
        cnn.main(xrd_specs, testing_fraction=0.2)
