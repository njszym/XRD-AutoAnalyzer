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


def explore_mixtures(spectrum, kdp, reference_phases):
    """
    A branching algorithm designed to explore all suspected mixtures predicted by the CNN.
    For each mixture, the associated phases and probabilities are tabulated.
    Currently, mixtures with as many as three phases can be handled. Will extend to arbitary mixtures soon.

    Args:
        spectrum: a numpy array containing the measured spectrum that is to be classified
        kdp: a KerasDropoutPrediction model object
        reference_phases: a list of reference phase strings
    Returns:
        all_predictions: a list of all enumerated mixtures
        total_confidence: a list of probabilities associated with the above mixtures
    """

    total_confidence, all_predictions = [], []
    tabulate_conf, predicted_cmpd_set = [], []
    measured_spectrum = prepare_pattern(spectrum)
    prediction_1, num_phases_1, certanties_1 = kdp.predict(measured_spectrum) ## Return predicted vector, number of probable phases (confidence > 10%), and associated confidence values
    for i1 in range(num_phases_1): ## Consider all probable phases
        tabulate_conf.append(certanties_1[i1])
        phase_index = np.array(prediction_1).argsort()[-(i1+1)] ## Get index of 1st, 2nd, etc. most probable phase depending on i1
        predicted_cmpd = reference_phases[phase_index] ## Get predicted compound associated with probable phase defined above
        predicted_cmpd_set.append(predicted_cmpd)
        stripped_y_1, norm_1 = get_reduced_pattern(predicted_cmpd, measured_spectrum) ## Strips away predicted phase from original spectrum
        if stripped_y_1 == 'All phases identified': ## If intensities fall below 10% original maximum, assume all phases have been identified. May tune this cutoff to be more sensitive.
            total_confidence.append(sum(tabulate_conf)/len(tabulate_conf))
            all_predictions.append(predicted_cmpd_set)
            tabulate_conf, predicted_cmpd_set = [], []
        else: ## If intensities remain, investigate second phase
            prediction_2, num_phases_2, certanties_2 = kdp.predict([[val] for val in stripped_y_1])
            for i2 in range(num_phases_2):
                phase_index = np.array(prediction_2).argsort()[-(i2+1)]
                predicted_cmpd = reference_phases[phase_index]
                if predicted_cmpd in predicted_cmpd_set: ## If we've already predicted this compound, go to next most probable
                    all_predictions.append(predicted_cmpd_set)
                    total_confidence.append(sum(tabulate_conf)/len(tabulate_conf))
                    if i2 == (num_phases_2 - 1):
                        tabulate_conf, predicted_cmpd_set = [], [] ## If we're out of phases, then move on to next possible mixture
                    continue
                else: ## If 2nd phase is new
                    tabulate_conf.append(certanties_2[i2])
                    predicted_cmpd_set.append(predicted_cmpd)
                stripped_y_2, norm_2 = get_reduced_pattern(predicted_cmpd, stripped_y_1, norm_1)
                if stripped_y_2 == 'All phases identified': ## If intensities fall below 10% original maximum, assume all phases have been identified. May tune this cutoff to be more sensitive.
                    total_confidence.append(sum(tabulate_conf)/len(tabulate_conf))
                    all_predictions.append(predicted_cmpd_set)
                    if i2 == (num_phases_2 - 1):
                        tabulate_conf, predicted_cmpd_set = [], [] ## If we're out of phases, then move on to next possible mixture
                    else:
                        tabulate_conf, predicted_cmpd_set = tabulate_conf[:-1], predicted_cmpd_set[:-1]
                else:  ## If intensities remain, investigate third phase
                    prediction_3, num_phases_3, certanties_3 = kdp.predict([[val] for val in stripped_y_2])
                    for i3 in range(num_phases_3):
                        phase_index = np.array(prediction_3).argsort()[-(i3+1)]
                        predicted_cmpd = reference_phases[phase_index]
                        if predicted_cmpd in predicted_cmpd_set: ## If we've already predicted this compound, go to next most probable
                            all_predictions.append(predicted_cmpd_set)
                            total_confidence.append(sum(tabulate_conf)/len(tabulate_conf))
                            continue
                        else: ## If 3rd phase is new
                            tabulate_conf.append(certanties_3[i3])
                            predicted_cmpd_set.append(predicted_cmpd)
                            all_predictions.append(predicted_cmpd_set)
                            total_confidence.append(sum(tabulate_conf)/len(tabulate_conf))
                            tabulate_conf = tabulate_conf[:-1]
                            predicted_cmpd_set = predicted_cmpd_set[:-1]

    return all_predictions, total_confidence


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
        If intensities fall below the cutoff, return Nonetype
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
        return 'All phases identified', None


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

