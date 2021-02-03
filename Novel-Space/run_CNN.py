from autoXRD import spectrum_analysis, visualizer
import sys
import numpy as np
import pymatgen as mg
import matplotlib.pyplot as plt


spectrum_names, predicted_phases, confidences = spectrum_analysis.main('Spectra', 'References')

for (spectrum_fname, phase_set, confidence) in zip(spectrum_names, predicted_phases, confidences):

    print('Filename: %s' % spectrum_fname)
    print('Predicted phases: %s' % phase_set)
    print('Confidence: %s' % confidence)

    if '--plot' in sys.argv:

        # Format predicted phases into a list of their CIF filenames
        predicted_phases = phase_set.split(' + ')
        predicted_phases = ['%s.cif' % phase for phase in predicted_phases]

        # Plot measured spectrum with line profiles of predicted phases
        visualizer.main('Spectra', spectrum_fname, predicted_phases)
