from autoXRD import phase_ID
from autoXRD import generate_spectra
import sys
import matplotlib.pyplot as plt


reference_dir = sys.argv[-2]
spectrum_dir = sys.argv[-1]

spectrum_names, predicted_phases, confidences = phase_ID.analyze(spectrum_dir, reference_dir)

for (spectrum, phase_set, confidence) in zip(spectrum_names, predicted_phases, confidences):

    print('Filename: %s' % spectrum)
    print('Predicted phase(s): %s' % phase_set)
    print('Confidence: %s' % confidence)

    if '-plot' in sys.argv:
        y = prepare_pattern('%s/%s' % (spectrum_dir, spectrum))
        x = np.linspace(10, 80, 4501)
        plt.figure()
        plt.plot(x, y, 'b-', label='Measured: %s' % spectrum)
        color_list = ['g', 'r', 'm']
        i = 0
        for phase in phase_set.split('+'):
            struct = mg.Structure.from_file('%s/%s.cif' % (reference_dir, phase))
            angles, intensities = generate_spectra(struct, stick_pattern=True)
            for (x, y) in zip(angles, intensities):
                plt.vlines(x, 0, y, color=color[i])
            i += 1
        plt.xlabel(r'2$\Theta$')
        plt.ylabel('Intensity')
        plt.show()
