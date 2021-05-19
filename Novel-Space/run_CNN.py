from autoXRD import spectrum_analysis, visualizer
import sys
import numpy as np
import pymatgen as mg
import matplotlib.pyplot as plt


if __name__ == '__main__':

    max_phases = 3 # default: a maximum 3 phases in each mixture
    cutoff_intensity = 10 # default: ID all peaks with I >= 10% maximum spectrum intensity
    wavelength = 'CuKa' # default: spectra was measured using Cu K_alpha radiation
    for arg in sys.argv:
        if '--max_phases' in arg:
            max_phases = int(arg.split('=')[1])
        if '--cutoff_intensity' in arg:
            cutoff_intensity = int(arg.split('=')[1])
        if '--wavelength' in arg:
            wavelength = float(arg.split('=')[1])

    spectrum_names, predicted_phases, confidences = spectrum_analysis.main('Spectra', 'References', max_phases, cutoff_intensity, wavelength)

    for (spectrum_fname, phase_set, confidence) in zip(spectrum_names, predicted_phases, confidences):

        if '--all' not in sys.argv: # By default: only include phases with a confidence > 25%
            all_phases = phase_set.split(' + ')
            all_probs = [float(val[:-1]) for val in confidence]
            final_phases, final_confidence = [], []
            for (ph, cf) in zip(all_phases, all_probs):
                if cf >= 25.0:
                    final_phases.append(ph)
                    final_confidence.append(cf)

            print('Filename: %s' % spectrum_fname)
            print('Predicted phases: %s' % final_phases)
            print('Confidence: %s' % final_confidence)

        else: # If --all is specified, print *all* suspected phases
            print('Filename: %s' % spectrum_fname)
            print('Predicted phases: %s' % phase_set)
            print('Confidence: %s' % confidence)

        if '--plot' in sys.argv:

            # Format predicted phases into a list of their CIF filenames
            predicted_phases = phase_set.split(' + ')
            predicted_phases = ['%s.cif' % phase for phase in predicted_phases]

            # Plot measured spectrum with line profiles of predicted phases
            visualizer.main('Spectra', spectrum_fname, predicted_phases)
