from autoXRD import spectrum_analysis, visualizer, quantifier
import sys
import numpy as np
import pymatgen as mg
import matplotlib.pyplot as plt


if __name__ == '__main__':

    wavelength = 'CuKa'
    min_angle, max_angle = 10.0, 80.0
    show_reduced = False
    inc_pdf = False
    save = False
    phases = []
    for arg in sys.argv:
        if '--spectrum' in arg:
            spectrum_fname = arg.split('=')[1]
        if '--ph' in arg:
            ph_fname = '%s.cif' % arg.split('=')[1]
            phases.append(ph_fname)
        if '--wavelength' in arg:
            wavelength = float(arg.split('=')[1])
        if '--min_angle' in arg:
            min_angle = float(arg.split('=')[1])
        if '--max_angle' in arg:
            max_angle = float(arg.split('=')[1])
        if '--show_reduced' in arg:
            show_reduced = True
        if '--inc_pdf' in arg:
            inc_pdf = True
        if '--save' in arg:
            save = True

    scale_factors = None # Not known yet
    final_spectrum = None # Not known yet
    visualizer.main('Spectra', spectrum_fname, phases, scale_factors, final_spectrum,
        min_angle, max_angle, wavelength, save, show_reduced, inc_pdf)

    if '--weights' in sys.argv:

        # Get weight fractions
        weights = quantifier.main('Spectra', spectrum_fname, phases, scale_factors, min_angle, max_angle, wavelength)
        weights = [round(val, 2) for val in weights]
        print('Phases: %s' % phases)
        print('Weight fractions: %s' % weights)

