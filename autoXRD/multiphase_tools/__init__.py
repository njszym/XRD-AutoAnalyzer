import matplotlib.pyplot as plt
import keras
import keras.backend as K
from scipy.signal import find_peaks
from dtw import dtw, warp
from scipy.signal import lfilter
import random
import pymatgen as mg
from pymatgen.analysis.diffraction import xrd
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
import tensorflow as tf
from tensorflow.python.keras.backend import eager_learning_phase_scope
from scipy import interpolate as ip
import ast
import numpy as np
import os


def classify_mixture(spectrum, reference_phases):
    """
    Perform multi-phase classification for a given XRD spectrum

    Args:
        spectrum: a numpy array containing the measured spectrum that is to be classified
        kdp: a KerasDropoutPrediction model object
        reference_phases: a list of reference phase strings
    Returns:
        prediction_list: a list of all enumerated mixtures
        confidence_list: a list of probabilities associated with the above mixtures
    """

    model = tf.keras.models.load_model('Model.h5', custom_objects={'sigmoid_cross_entropy_with_logits_v2': tf.nn.sigmoid_cross_entropy_with_logits})
    kdp = KerasDropoutPrediction(model)
    spectrum = prepare_pattern(spectrum)
    prediction_list, confidence_list = enumerate_routes(spectrum, kdp, reference_phases)
    return prediction_list, confidence_list


def enumerate_routes(spectrum, kdp, reference_phases, indiv_conf=[], indiv_pred=[], confidence_list=[], prediction_list = [], max_phases=3):
    """
    A branching algorithm designed to explore all suspected mixtures predicted by the CNN.
    For each mixture, the associated phases and probabilities are tabulated.

    Args:
        spectrum: a numpy array containing the measured spectrum that is to be classified
        kdp: a KerasDropoutPrediction model object
        reference_phases: a list of reference phase strings
        indiv_conf: list of probabilities associated with an individual mixture (one per branch)
        indiv_pred: list of predicted phases in an individual mixture (one per branch)
        confidence_list: a list of averaged probabilities associated with all suspected mixtures
        predictions_list: a list of the phases predicted in all suspected mixtures
        max_phases: the maximum number of phases considered for a single mixture.
            By default, this is set to handle  up tothree-phase patterns. The function is readily
            extended to handle arbitrary many phases. Caution, however, that the computational time
            required will scale exponentially with the number of phases.
    Returns:
        prediction_list: a list of all enumerated mixtures
        confidence_list: a list of probabilities associated with the above mixtures
    """

    global updated_pred, updated_conf ## Global variables are updated recursively
    prediction, num_phases, certanties = kdp.predict(spectrum)

    ## Explore all phases with a non-trival probability
    for i in range(num_phases):

        ## If individual predictions have been updated recursively, use them for this iteration
        if 'updated_pred' in globals():
            if updated_pred != None:
                indiv_pred, indiv_conf = updated_pred, updated_conf
                updated_pred, updated_conf = None, None

        prediction, num_phases, certanties = kdp.predict(spectrum)
        phase_index = np.array(prediction).argsort()[-(i+1)]
        predicted_cmpd = reference_phases[phase_index]

        ## If the predicted phase has already been identified for the mixture, ignore and move on
        if predicted_cmpd in indiv_pred:
            if i == (num_phases - 1):
                confidence_list.append(sum(indiv_conf)/len(indiv_conf))
                prediction_list.append(indiv_pred)
                updated_conf, updated_pred = indiv_conf[:-1], indiv_pred[:-1]
            continue

        indiv_pred.append(predicted_cmpd)
        indiv_conf.append(certanties[i])
        reduced_spectrum, norm = get_reduced_pattern(predicted_cmpd, spectrum)

        ## If all phases have been identified, tabulate mixture and move on to next
        if norm == None:
            confidence_list.append(sum(indiv_conf)/len(indiv_conf))
            prediction_list.append(indiv_pred)
            if i == (num_phases - 1):
                updated_conf, updated_pred = indiv_conf[:-2], indiv_pred[:-2]
            else:
                indiv_conf, indiv_pred = indiv_conf[:-1], indiv_pred[:-1]
            continue

        else:
            ## If the maximum number of phases has been reached, tabulate mixture and move on to next
            if len(indiv_pred) == max_phases:
                confidence_list.append(sum(indiv_conf)/len(indiv_conf))
                prediction_list.append(indiv_pred)
                if i == (num_phases - 1):
                    updated_conf, updated_pred = indiv_conf[:-2], indiv_pred[:-2]
                else:
                    indiv_conf, indiv_pred = indiv_conf[:-1], indiv_pred[:-1]
                continue

            ## Otherwise if more phases are to be explored, recursively enter enumerate_routes with the newly reduced spectrum
            prediction_list, confidence_list = enumerate_routes(reduced_spectrum, kdp, reference_phases, indiv_conf, indiv_pred, confidence_list, prediction_list)

    return prediction_list, confidence_list


def get_reduced_pattern(predicted_cmpd, orig_y, last_normalization=1.0, cutoff=5):
    """
    Subtract a phase that has already been identified from a given XRD spectrum.
    If all phases have already been identified, halt the iteration.

    Args:
        predicted_cmpd: phase that has been identified
        orig_y: measured spectrum including the phase the above phase
        last_normalization: normalization factor used to scale the previously stripped
            spectrum to 100 (required by the CNN). This is necessary to determine the
            magnitudes of intensities relative to the initially measured pattern.
        cutoff: the % cutoff used to halt the phase ID iteration. If all intensities are
            below this value in terms of the originally measured maximum intensity, then
            the code assumes that all phases have been identified.
    Returns:
        stripped_y: new spectrum obtained by subtrating the peaks of the identified phase
        new_normalization: scaling factor used to ensure the maximum intensity is equal to 100
        Or
        If intensities fall below the cutoff, preserve orig_y and return Nonetype
            the for new_normalization constant.
    """

    pred_y = generate_pattern(predicted_cmpd)
    dtw_info = dtw(pred_y, orig_y, window_type="slantedband", window_args={'window_size': 20}) ## corresponds to about 1.5 degree shift
    warp_indices = warp(dtw_info)
    warped_spectrum = list(pred_y[warp_indices])
    warped_spectrum.append(0.0)
    warped_spectrum = scale_spectrum(warped_spectrum, orig_y)
    stripped_y = strip_spectrum(warped_spectrum, orig_y)
    stripped_y = smooth_spectrum(stripped_y)
    stripped_y = np.array(stripped_y) - min(stripped_y)
    if max(stripped_y) >= (cutoff*last_normalization):
        new_normalization = 100/max(stripped_y)
        stripped_y = new_normalization*stripped_y
        return stripped_y, new_normalization
    else:
        return orig_y, None


def prepare_pattern(spectrum_name, smooth=True):
    """
    Cleans up a measured spectrum and conerts into a form that
    is directly readable by the CNN.

    Args:
        spectrum_name: filename of the spectrum that is being considered
    Returns:
        Processed XRD spectrum in 4501x1 form.
    """

    ## Load data
    data = np.loadtxt(spectrum_name)
    x = data[:, 0]
    y = data[:, 1]

    ## Fit to 4,501 values as to be compatible with CNN
    f = ip.CubicSpline(x, y)
    xs = np.linspace(10, 80, 4501)
    ys = f(xs)

    ## Smooth out noise
    ys = smooth_spectrum(ys)

    ## Map to integers in range 0 to 255 so cv2 can handle
    ys = [val - min(ys) for val in ys]
    ys = [255*(val/max(ys)) for val in ys]
    ys = [int(val) for val in ys]

    ## Perform baseline correction with cv2
    pixels = []
    for q in range(10):
        pixels.append(ys)
    pixels = np.array(pixels)
    img, background = subtract_background_rolling_ball(pixels, 800, light_background=False,
                                         use_paraboloid=True, do_presmooth=False)
    yb = np.array(background[0])
    ys = np.array(ys) - yb

    ## Normalize from 0 to 100
    ys = np.array(ys) - min(ys)
    ys = list(100*np.array(ys)/max(ys))

    return ys

def generate_pattern(cmpd, scale_vol=1.0, std_dev=0.15):
    """
    Calculate the XRD spectrum of a given compound.

    Args:
        cmpd: filename of the structure file to calculate the spectrum for
        scale_vol: factor used to scale the unit cell volume if necessary
        std_dev: controlls peaks widths
    Returns:
        XRD spectrum (list) with 4,501 values
    """

    struct = mg.Structure.from_file('References/%s' % cmpd)
    equil_vol = struct.volume
    struct.scale_lattice(scale_vol * equil_vol)
    calculator = xrd.XRDCalculator()
    pattern = calculator.get_pattern(struct, two_theta_range=(10,80))
    angles = pattern.x
    peaks = pattern.y

    x = np.linspace(10, 80, 4501)
    y = []
    for val in x:
        ysum = 0
        for (ang, pk) in zip(angles, peaks):
            if np.isclose(ang, val, atol=0.05):
                ysum += pk
        y.append(ysum)
    conv = []
    for (ang, int) in zip(x, y):
        if int != 0:
            gauss = [int*np.exp((-(val - ang)**2)/std_dev) for val in x]
            conv.append(gauss)
    mixed_data = zip(*conv)
    all_I = []
    for values in mixed_data:
        all_I.append(sum(values))

    all_I = np.array(all_I) - min(all_I)
    all_I = 100*all_I/max(all_I)

    return all_I


class KerasDropoutPrediction(object):
    """
    Ensemble model used to perform phase identification and quantify uncertainty

    Args:
        object: trained CNN model
    """

    def __init__(self, model):
        self.f = tf.keras.backend.function(model.layers[0].input, model.layers[-1].output)

    def predict(self, x, n_iter=1000):
        """
        Args:
            self: keras backend function based on the trained CNN model
            x: XRD spectrum to be classified
        Returns:
            prediction: distribution of probabilities associated with reference phases
            len(certainties): number of phases with probabilities > 10%
            certanties: associated probabilities
        """

        x = [[val] for val in x]
        x = np.array([x])
        result = []
        with eager_learning_phase_scope(value=1):
            for _ in range(n_iter):
                result.append(self.f(x))

        result = np.array([list(np.array(sublist).flatten()) for sublist in result]) ## Individual predictions
        prediction = result.mean(axis=0) ## Average prediction

        all_preds = [np.argmax(pred) for pred in result] ## Individual max indices (associated with phases)

        counts = []
        for index in set(all_preds):
            counts.append(all_preds.count(index)) ## Tabulate how many times each prediction arises

        certanties = []
        for each_count in counts:
            conf = each_count/sum(counts)
            if conf >= 0.1: ## If prediction occurs at least 10% of the time
                certanties.append(conf)
        certanties = sorted(certanties, reverse=True)

        return prediction, len(certanties), certanties


def smooth_spectrum(warped_spectrum, n=20):
    """
    Process and remove noise from the spectrum.

    Args:
        warped_spectrum: measured spectrum (possibly with noise)
        n: parameters used to control smooth. Larger n means greater smoothing.
            20 is typically a good number such that noise is reduced while
            still retaining minor diffraction peaks.
    Returns:
        shifted_ys: processed spectrum after noise removal
    """

    b = [1.0 / n] * n
    a = 1
    yy = lfilter(b, a, warped_spectrum)
    ys = yy
    shifted_ys = []
    for val in ys[11:]:
        shifted_ys.append(val)
    for z in range(11):
        shifted_ys.append(0.0)
    return shifted_ys

def scale_spectrum(warped_spectrum, orig_y):
    """
    Scale the magnitude of a calculated spectrum associated with an identified
    phase so that its peaks match with those of the measured spectrum being classified.

    Args:
        warped_spectrum: spectrum calculated from the identified phase after fitting
            has been performed along the x-axis using DTW
        orig_y: original (measured) spectrum containing all peaks
    Returns:
        scaled_spectrum: spectrum associated with the reference phase after scaling
            has been performed to match the peaks in the measured pattern.
    """

    orig_peaks = find_peaks(orig_y, height=5)[0] ## list of peak indices
    pred_peaks = find_peaks(warped_spectrum, height=5)[0] ## list of peak indices
    matched_orig_peaks = []
    matched_pred_peaks = []
    for a in orig_peaks:
        for b in pred_peaks:
            if np.isclose(a, b, atol=50): ## within 50 indices of one another
                matched_orig_peaks.append(a)
                matched_pred_peaks.append(b)
    num_match = []
    for scale_spectrum in np.linspace(1.2, 0.2, 101):
        check = scale_spectrum*np.array(warped_spectrum)
        good_peaks = 0
        for (a, b) in zip(matched_orig_peaks, matched_pred_peaks):
            A_magnitude = orig_y[a]
            B_magnitude = check[b]
            if abs((A_magnitude - B_magnitude)/A_magnitude) < 0.1: ## If peaks are within 10% of one another
                good_peaks += 1
        num_match.append(good_peaks)
    best_scale = np.linspace(1.2, 0.2, 101)[np.argmax(num_match)] ## Will give highest scaling constant which yields best match
    scaled_spectrum = best_scale*np.array(warped_spectrum) ## Scale
    return scaled_spectrum


def scale_line_profile(orig_y, peaks, angles):
    """
    Similar to the scale_spectrum() fn, except now the goal is take
    find the scaling factor that minimizes the differences between a line
    profile and the peaks in a measured XRD spectrum. This is used during
    plotting for visualization of identified phases.

    Args:
        orig_y: original (measured) spectrum containing all peaks
        peaks: a list of peak intensities (stick profile)
        angles: a list of diffraction angles (stick profile)
    Returns:
        best_scale: a float ranging from 0.05 to 1.0 that has been optimized
            to ensure maximal overlap between the line profile and the peaks
            in the measured spectrum.
    """

    peak_inds = (4501./70.)*(np.array(angles) - 10.)
    peak_inds = [int(i) for i in peak_inds]
    y = []
    qi = 0
    for x in range(4501):
        if x in peak_inds:
            y.append(peaks[qi])
            qi += 1
        else:
            y.append(0.0)

    orig_peaks = find_peaks(orig_y, height=5)[0] ## list of peak indices
    pred_peaks = find_peaks(y, height=5)[0] ## list of peak indices
    matched_orig_peaks = []
    matched_pred_peaks = []
    for a in orig_peaks:
        for b in pred_peaks:
            if np.isclose(a, b, atol=100): ## within 50 indices of one another
                matched_orig_peaks.append(a)
                matched_pred_peaks.append(b)
    num_match = []
    for scale_spectrum in np.linspace(1, 0.05, 101):
        check = scale_spectrum*np.array(y)
        good_peaks = 0
        for (a, b) in zip(matched_orig_peaks, matched_pred_peaks):
            A_magnitude = orig_y[a]
            B_magnitude = check[b]
            if abs((A_magnitude - B_magnitude)/A_magnitude) < 0.1: ## If peaks are within 5% of one another
                good_peaks += 1
        num_match.append(good_peaks)
    best_scale = np.linspace(1.0, 0.05, 101)[np.argmax(num_match)] ## Will give highest scaling constant which yields best match
    return best_scale


def strip_spectrum(warped_spectrum, orig_y):
    """
    Subtract one spectrum from another. Note that when subtraction produces
    negative intensities, those values are re-normalized to zero. This way,
    the CNN can handle the spectrum reliably.

    Args:
        warped_spectrum: spectrum associated with the identified phase
        orig_y: original (measured) spectrum
    Returns:
        stripped_y: resulting spectrum from the subtraction of warped_spectrum
            from orig_y
    """
    stripped_y = orig_y - warped_spectrum
    fixed_y = []
    for val in stripped_y:
        if val < 0:
            fixed_y.append(0.0)
        else:
            fixed_y.append(val)
    stripped_y = fixed_y
    return stripped_y


def plot_spectra(warped_spectrum, stripped_y, orig_y):
    x = np.linspace(10, 80, 4501)
    plt.figure()
    plt.plot(x, orig_y[-1], 'b-')
    plt.plot(x, warped_spectrum, 'r-')
    plt.plot(x, stripped_y, 'g-')
    plt.show()

