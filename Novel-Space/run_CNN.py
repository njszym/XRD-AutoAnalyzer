from autoXRD import phase_ID
import sys

reference_dir = sys.argv[-2]
spectrum_dir = sys.argv[-1]

spectrum_names, predicted_phases, confidences = phase_ID.analyze(spectrum_dir, reference_dir)

for (spectrum, phase_set, confidence) in zip(spectrum_names, predicted_phases, confidences):
    print('Filename: %s' % spectrum)
    print('Predicted phase(s): %s' % phase_set)
    print('Confidence: %s' % confidence)
