from autoXRD.multiphase_tools import *
import multiprocessing
from multiprocessing import Pool, Manager


num_cpu = multiprocessing.cpu_count()

def identify(fname):
    """
    Identify all phases in a given XRD spectrum

    Args:
        fname: filename string of the spectrum to be classified
    Returns:
        fname: filename, same as in Args
        predicted_set: string of compounds predicted by phase ID algo
        max_conf: confidence associated with the prediction
    """

    total_confidence, all_predictions = [], []
    tabulate_conf, predicted_cmpd_set = [], []
    mixtures, confidence = classify_mixture('%s/%s' % (spec_dir, fname), reference_phases)
    if len(confidence) > 0:
        max_conf_ind = np.argmax(confidence)
        max_conf = 100*confidence[max_conf_ind]
        predicted_cmpds = [fname[:-4] for fname in mixtures[max_conf_ind]]
        predicted_set = ' + '.join(predicted_cmpds)
    else:
        max_conf = 0.0
        predicted_set = 'None'
    return [fname, predicted_set, max_conf]


def analyze(spectrum_dir, reference_dir):
    """
    Enumerate all spectra in a given directory for phase identification

    Args:
        spectrum_dir: path to directory containing spectra to be classified.
            Note these files should be in xy format.
        reference_dir: path to directory containing CIFs of reference phases
    Returns:
        spectrum_names: filenames of spectra being classified
        predicted_phases: a list of the predicted phases in the mixture
        confidences: the associated confidence with the prediction above
    """

    global reference_phases, spec_dir
    spec_dir = spectrum_dir
    reference_phases = sorted(os.listdir(reference_dir))

    with Manager() as manager:

        spectrum_filenames = os.listdir(spectrum_dir)
        pool = Pool(num_cpu)
        all_info = pool.map(identify, spectrum_filenames)
        spectrum_names = [info[0] for info in all_info]
        predicted_phases = [info[1] for info in all_info]
        confidences = [info[2] for info in all_info]

        return spectrum_names, predicted_phases, confidences
