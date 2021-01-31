from autoXRD import spectrum_analysis
import numpy as np
import pymatgen as mg
import matplotlib.pyplot as plt


spectrum_names, predicted_phases, confidences = spectrum_analysis.main('Spectra', 'References')

for (spectrum, phase_set, confidence) in zip(spectrum_names, predicted_phases, confidences):

    print('Filename: %s' % spectrum)
    print('Predicted phases: %s' % phase_set)
    print('Confidence: %s' % confidence)

