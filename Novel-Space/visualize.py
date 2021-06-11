from autoXRD import spectrum_analysis, visualizer
import sys
import numpy as np
import pymatgen as mg
import matplotlib.pyplot as plt


if __name__ == '__main__':

    wavelength = 'CuKa'
    min_angle, max_angle = 10.0, 80.0
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

    visualizer.main('Spectra', spectrum_fname, phases, min_angle, max_angle)
