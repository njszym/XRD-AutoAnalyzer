from autoXRD import phase_ID
from autoXRD import multiphase_tools
from autoXRD import generate_spectra
import numpy as np
import sys
import pymatgen as mg
import matplotlib.pyplot as plt


reference_dir = sys.argv[-2]
spectrum_dir = sys.argv[-1]

spectrum_names, predicted_phases, confidences = phase_ID.analyze(spectrum_dir, reference_dir)

for (spectrum, phase_set, confidence) in zip(spectrum_names, predicted_phases, confidences):

    print('Filename: %s' % spectrum)
    print('Predicted phases: %s' % phase_set)
    print('Confidence: %s' % confidence)

    if '-plot' in sys.argv:
        y = multiphase_tools.prepare_pattern('%s/%s' % (spectrum_dir, spectrum))
        y = multiphase_tools.smooth_spectrum(y)
        x = np.linspace(10, 80, 4501)
        plt.figure()
        plt.plot(x, y, 'b-', label='Measured: %s' % spectrum)
        color_list = ['g', 'r', 'm']
        i = 0
        for phase in phase_set.split(' + '):
            struct = mg.Structure.from_file('%s/%s.cif' % (reference_dir, phase))
            angles, intensities = generate_spectra.calc_XRD(struct, stick_pattern=True)
            for (x, y) in zip(angles, intensities):
                plt.vlines(x, 0, y, color=color_list[i])
            plt.plot([0], [0], color=color_list[i], label='Predicted: %s' % phase)
            i += 1
        plt.xlim(10, 80)
        plt.ylim(0, 105)
        plt.legend(prop={'size': 16})
        plt.xlabel(r'2$\Theta$', fontsize=16, labelpad=12)
        plt.ylabel('Intensity', fontsize=16, labelpad=12)
        plt.show()
