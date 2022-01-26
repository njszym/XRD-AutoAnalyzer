from autoXRD import spectrum_analysis, visualizer, quantifier
import sys
import numpy as np
import pymatgen as mg
import matplotlib.pyplot as plt
import time


if __name__ == '__main__':

    start = time.time()

    max_phases = 4 # default: a maximum 4 phases in each mixture
    cutoff_intensity = 5 # default: ID all peaks with I >= 5% maximum spectrum intensity
    wavelength = 'CuKa' # default: spectra was measured using Cu K_alpha radiation
    min_angle, max_angle = 10.0, 80.0
    for arg in sys.argv:
        if '--max_phases' in arg:
            max_phases = int(arg.split('=')[1])
        if '--cutoff_intensity' in arg:
            cutoff_intensity = int(arg.split('=')[1])
        if '--wavelength' in arg:
            wavelength = float(arg.split('=')[1])
        if '--min_angle' in arg:
            min_angle = float(arg.split('=')[1])
        if '--max_angle' in arg:
            max_angle = float(arg.split('=')[1])

    spectrum_names, predicted_phases, confidences = spectrum_analysis.main('Spectra', 'References', max_phases, cutoff_intensity, wavelength, min_angle, max_angle)

    for (spectrum_fname, phase_set, confidence) in zip(spectrum_names, predicted_phases, confidences):

        if '--all' not in sys.argv: # By default: only include phases with a confidence > 25%
            final_phases, final_confidence = [], []
            for (ph, cf) in zip(phase_set, confidence):
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

        if ('--plot' in sys.argv) and (phase_set != 'None'):

            # Format predicted phases into a list of their CIF filenames
            final_phasenames = ['%s.cif' % phase for phase in final_phases]

            # Plot measured spectrum with line profiles of predicted phases
            visualizer.main('Spectra', spectrum_fname, final_phasenames, min_angle, max_angle, wavelength)

        if ('--weights' in sys.argv) and (phase_set != 'None'):

            # Format predicted phases into a list of their CIF filenames
            final_phasenames = ['%s.cif' % phase for phase in final_phases]

            # Get weight fractions
            weights = quantifier.main('Spectra', spectrum_fname, final_phasenames, min_angle, max_angle, wavelength)
            weights = [round(val, 2) for val in weights]
            print('Weight fractions: %s' % weights)

    end = time.time()

    elapsed_time = round(end - start, 1)
    print('Total time: %s sec' % elapsed_time)
