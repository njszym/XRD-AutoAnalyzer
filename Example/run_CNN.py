from autoXRD import spectrum_analysis, visualizer, quantifier
import matplotlib.pyplot as plt
import pymatgen as mg
import numpy as np
import shutil
import time
import sys
import os


if __name__ == '__main__':

    start = time.time()

    max_phases = 3 # default: a maximum 4 phases in each mixture
    cutoff_intensity = 5 # default: ID all peaks with I >= 5% maximum spectrum intensity
    min_conf = 40.0 # Minimum confidence included in predictions
    wavelength = 'CuKa' # default: spectra was measured using Cu K_alpha radiation
    unknown_threshold = 25.0 # default: raise warning when peaks with >= 25% intensity are unknown
    show_reduced = False # Whether to plot reduced spectrum (after subtraction of known phases)
    inc_pdf = False # Whether to include PDF analysis (requires trained model first)
    parallel = True # Run phase analysis in parallel across available CPUs
    raw = True # Whether to show the raw spectrum or its denoised product
    show_indiv = False # Whether to show individual predictions from XRD and PDF
    refined_phases_dir = None
    min_angle, max_angle = 10.0, 80.0
    for arg in sys.argv:
        if '--max_phases' in arg:
            max_phases = int(arg.split('=')[1])
        if '--cutoff_intensity' in arg:
            cutoff_intensity = int(arg.split('=')[1])
        if '--min_conf' in arg:
            min_conf = int(arg.split('=')[1])
        if '--wavelength' in arg:
            wavelength = float(arg.split('=')[1])
        if '--min_angle' in arg:
            min_angle = float(arg.split('=')[1])
        if '--max_angle' in arg:
            max_angle = float(arg.split('=')[1])
        if '--unknown_thresh' in arg:
            unknown_threshold = float(arg.split('=')[1])
        if '--show_reduced' in arg:
            show_reduced = True
        if '--inc_pdf' in arg:
            inc_pdf = True
        if '--show_indiv' in arg:
            show_indiv = True
        if '--refined_phases_dir' in arg:
            refined_phases_dir = str(arg.split('=')[1])

    # Make sure at least one spectrum is provided
    assert len(os.listdir('Spectra')) > 0, 'Please provide at least one pattern in the Spectra directory.'

    # Keep results separate
    results = {'XRD': {}, 'PDF': {}}

    # XRD/PDF ensemble requires all predictions
    if inc_pdf:
        final_conf = min_conf # For final (merged) predictions
        min_conf = 10.0 # For individual (XRD/PDF) predictions

    # Path to trained CNN model
    if inc_pdf:
        model_path = 'Models/XRD_Model.h5'
    else:
        model_path = 'Model.h5'

    # Used for writing temporary files
    if not os.path.exists('temp'):
        os.mkdir('temp')

    # Get predictions from XRD analysis
    results['XRD']['filenames'], results['XRD']['phases'], results['XRD']['confs'], results['XRD']['backup_phases'], \
        results['XRD']['scale_factors'], results['XRD']['reduced_spectra'] = spectrum_analysis.main('Spectra', 'References',
        max_phases, cutoff_intensity, min_conf, wavelength, min_angle, max_angle, parallel, model_path, is_pdf=False)

    if inc_pdf:
        # If specified, get predictions from PDF analysis
        model_path = 'Models/PDF_Model.h5'
        results['PDF']['filenames'], results['PDF']['phases'], results['PDF']['confs'], results['PDF']['backup_phases'], \
            results['PDF']['scale_factors'], results['PDF']['reduced_spectra'] = spectrum_analysis.main('Spectra', 'References',
            max_phases, cutoff_intensity, min_conf, wavelength, min_angle, max_angle, parallel, model_path, is_pdf=True)

        # Aggregate XRD and PDF predictions
        results['Merged'] = spectrum_analysis.merge_results(results, final_conf, max_phases)

    else:
        # Otherwise, rely only on predictions from XRD
        results['Merged'] = results['XRD']

    # Remove temporary files
    if os.path.exists('temp'):
        shutil.rmtree('temp')

    ph_index = 0
    for (spectrum_fname, phase_set, confidence, backup_set, heights, final_spectrum) in \
        zip(results['Merged']['filenames'], results['Merged']['phases'], results['Merged']['confs'], \
        results['Merged']['backup_phases'], results['Merged']['scale_factors'], results['Merged']['reduced_spectra']):

        # Print phase ID info
        print('Filename: %s' % spectrum_fname)
        print('Predicted phases: %s' % phase_set)
        print('Confidence: %s' % confidence)

        # If there are unknown peaks with intensity > threshold, raise warning
        if (len(phase_set) > 0) and ('None' not in phase_set):
            remaining_I = max(final_spectrum)
            if remaining_I > unknown_threshold:
                print('WARNING: some peaks (I ~ %s%%) were not identified.' % int(remaining_I))
        else:
            print('WARNING: no phases were identified')
            continue

        # If this option is specified, show backup predictions (2nd-most probable phases)
        if '--show_backups' in sys.argv:
            print('Alternative phases: %s' % backup_set)

        if show_indiv:

            print('XRD predicted phases: %s' % results['XRD']['phases'][ph_index])
            print('XRD confidence: %s' % results['XRD']['confs'][ph_index])
            print('PDF predicted phases: %s' % results['PDF']['phases'][ph_index])
            print('PDF confidence: %s' % results['PDF']['confs'][ph_index])
            ph_index += 1

        if ('--plot' in sys.argv) and ('None' not in phase_set):

            # Whether to save as png file
            save = False
            if '--save' in sys.argv:
                save = True

            # Whether to plot XRD and PDF
            plot_both = False
            if '--both' in sys.argv:
                plot_both = True

            # Format predicted phases into a list of their CIF filenames
            phasenames = ['%s.cif' % phase for phase in phase_set]

            # Plot measured spectrum with line profiles of predicted phases
            visualizer.main('Spectra', spectrum_fname, phasenames, heights, final_spectrum,
                min_angle, max_angle, wavelength, save, show_reduced, inc_pdf, plot_both, raw,
                    refined_phases_dir=refined_phases_dir)

        if ('--weights' in sys.argv) and ('None' not in phase_set):

            # Format predicted phases into a list of their CIF filenames
            phasenames = ['%s.cif' % phase for phase in phase_set]

            # Get weight fractions
            weights = quantifier.main('Spectra', spectrum_fname, phasenames, heights, min_angle, max_angle,
                wavelength, refined_phases_dirrefined_phases_dir)
            weights = [round(val, 2) for val in weights]
            print('Weight fractions: %s' % weights)

    end = time.time()

    elapsed_time = round(end - start, 1)
    print('Total time: %s sec' % elapsed_time)
