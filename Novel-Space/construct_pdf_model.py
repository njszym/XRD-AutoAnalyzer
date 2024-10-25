from autoXRD import cnn, spectrum_generation, solid_solns, tabulate_cifs
import numpy as np
import os
import sys
import pymatgen as mg


if __name__ == '__main__':

    max_texture = 0.5 # default: texture associated with up to +/- 50% changes in peak intensities
    min_domain_size, max_domain_size = 5.0, 30.0 # default: domain sizes ranging from 5 to 30 nm
    max_strain = 0.03 # default: up to +/- 3% strain
    max_shift = 0.5 # default: up to +/- 0.5 degrees shift in two-theta
    impur_amt = 70.0 # Max amount of impurity phases to include (%)
    num_spectra = 50 # Number of spectra to simulate per phase
    separate = False # If False: apply all artifacts simultaneously
    min_angle, max_angle = 10.0, 80.0
    num_epochs = 50
    for arg in sys.argv:
        if '--max_texture' in arg:
            max_texture = float(arg.split('=')[1])
        if '--min_domain_size' in arg:
            min_domain_size = float(arg.split('=')[1])
        if '--max_domain_size' in arg:
            max_domain_size = float(arg.split('=')[1])
        if '--max_strain' in arg:
            max_strain = float(arg.split('=')[1])
        if '--max_shift' in arg:
            max_shift = float(arg.split('=')[1])
        if '--impur_amt' in arg:
            impur_amt = float(arg.split('=')[1])
        if '--num_spectra' in arg:
            num_spectra = int(arg.split('=')[1])
        if '--min_angle' in arg:
            min_angle = float(arg.split('=')[1])
        if '--max_angle' in arg:
            max_angle = float(arg.split('=')[1])
        if '--num_epochs' in arg:
            num_epochs = int(arg.split('=')[1])
        if '--separate_artifacts' in arg:
            separate = True

    # Ensure an XRD model has already been trained, but not yet a PDF model
    assert 'Models' not in os.listdir('.'), 'Models folder already exists. Please remove it or use existing models.'
    assert 'Model.h5' in os.listdir('.'), 'Cannot find a trained Model.h5 file in current directory. Please train an XRD model first.'
    assert 'References' in os.listdir('.'), 'Cannot find a References folder in your current directory.'

    # Move trained XRD model to new directory
    os.mkdir('Models')
    os.rename('Model.h5', 'Models/XRD_Model.h5')

    # Simualted vrtual PDFs
    pdf_obj = spectrum_generation.SpectraGenerator('References', num_spectra, max_texture, min_domain_size,
        max_domain_size, max_strain, max_shift, impur_amt, min_angle, max_angle, separate, is_pdf=True)
    pdf_specs = pdf_obj.augmented_spectra

    # Save PDFs if flag is specified
    if '--save' in sys.argv:
        np.save('PDF', np.array(pdf_specs))

    # Train, test, and save the CNN
    test_fraction = 0.2
    cnn.main(pdf_specs, num_epochs, test_fraction, is_pdf=True)
    os.rename('Model.h5', 'Models/PDF_Model.h5')

