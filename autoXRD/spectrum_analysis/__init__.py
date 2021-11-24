from scipy.signal import find_peaks, filtfilt
from dtw import dtw, warp
import warnings
import random
from tqdm import tqdm
import pymatgen as mg
from pymatgen.analysis.diffraction import xrd
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.keras.backend import eager_learning_phase_scope
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate as ip
import numpy as np
import multiprocessing
from multiprocessing import Pool, Manager
from pymatgen.core import Structure
import math


class SpectrumAnalyzer(object):
    """
    Class used to process and classify xrd spectra.
    """

    def __init__(self, spectra_dir, spectrum_fname, max_phases, cutoff_intensity, wavelen='CuKa', reference_dir='References', min_angle=10.0, max_angle=80.0):
        """
        Args:
            spectrum_fname: name of file containing the
                xrd spectrum (in xy format)
            reference_dir: path to directory containing the
                reference phases (CIF files)
            wavelen: wavelength used for diffraction (angstroms).
                Defaults to Cu K-alpha radiation (1.5406 angstroms).
        """

        self.spectra_dir = spectra_dir
        self.spectrum_fname = spectrum_fname
        self.ref_dir = reference_dir
        self.calculator = xrd.XRDCalculator()
        self.max_phases = max_phases
        self.cutoff = cutoff_intensity
        self.wavelen = wavelen
        self.min_angle = min_angle
        self.max_angle = max_angle

    @property
    def reference_phases(self):
        return sorted(os.listdir(self.ref_dir))

    @property
    def suspected_mixtures(self):
        """
        Returns:
            prediction_list: a list of all enumerated mixtures
            confidence_list: a list of probabilities associated with the above mixtures
        """

        spectrum = self.formatted_spectrum

        self.model = tf.keras.models.load_model('Model.h5', compile=False)
        self.kdp = KerasDropoutPrediction(self.model)

        prediction_list, confidence_list = self.enumerate_routes(spectrum)

        return prediction_list, confidence_list

    def convert_angle(self, angle):
        """
        Convert two-theta into Cu K-alpha radiation.
        """

        orig_theta = math.radians(angle/2.)

        orig_lambda = self.wavelen
        target_lambda = 1.5406 # Cu k-alpha
        ratio_lambda = target_lambda/orig_lambda

        asin_argument = ratio_lambda*math.sin(orig_theta)

        # Curtail two-theta range if needed to avoid domain errors
        if asin_argument <= 1:
            new_theta = math.degrees(math.asin(ratio_lambda*math.sin(orig_theta)))
            return 2*new_theta

    @property
    def formatted_spectrum(self):
        """
        Cleans up a measured spectrum and format it such that it
        is directly readable by the CNN.

        Args:
            spectrum_name: filename of the spectrum that is being considered
        Returns:
            ys: Processed XRD spectrum in 4501x1 form.
        """

        ## Load data
        data = np.loadtxt('%s/%s' % (self.spectra_dir, self.spectrum_fname))
        x = data[:, 0]
        y = data[:, 1]

        ## Convert to Cu K-alpha radiation if needed
        if str(self.wavelen) != 'CuKa':
            Cu_x, Cu_y = [], []
            for (two_thet, intens) in zip(x, y):
                scaled_x = self.convert_angle(two_thet)
                if scaled_x is not None:
                    Cu_x.append(scaled_x)
                    Cu_y.append(intens)
            x, y = Cu_x, Cu_y

        # Allow some tolerance (0.2 degrees) in the two-theta range
        if (min(x) > self.min_angle) and np.isclose(min(x), self.min_angle, atol=0.2):
            x = np.concatenate([np.array([self.min_angle]), x])
       	    y = np.concatenate([np.array([y[0]]), y])
       	if (max(x) < self.max_angle) and np.isclose(max(x), self.max_angle, atol=0.2):
       	    x = np.concatenate([x, np.array([self.max_angle])])
            y = np.concatenate([y, np.array([y[-1]])])

        # Otherwise, raise an assertion error
        assert (min(x) <= self.min_angle) and (max(x) >= self.max_angle), """
               Measured spectrum does not span the specified two-theta range!
               Either use a broader spectrum or change the two-theta range via
               the --min_angle and --max_angle arguments."""

        ## Fit to 4,501 values as to be compatible with CNN
        f = ip.CubicSpline(x, y)
        xs = np.linspace(self.min_angle, self.max_angle, 4501)
        ys = f(xs)

        ## Smooth out noise
        ys = self.smooth_spectrum(ys)

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

    def smooth_spectrum(self, spectrum, n=20):
        """
        Process and remove noise from the spectrum.

        Args:
            spectrum: list of intensities as a function of 2-theta
            n: parameters used to control smooth. Larger n means greater smoothing.
                20 is typically a good number such that noise is reduced while
                still retaining minor diffraction peaks.
        Returns:
            smoothed_ys: processed spectrum after noise removal
        """

        # Smoothing parameters defined by n
        b = [1.0 / n] * n
        a = 1

        # Filter noise
        smoothed_ys = filtfilt(b, a, spectrum)

        return smoothed_ys

    def enumerate_routes(self, spectrum, indiv_conf=[], indiv_pred=[], confidence_list=[], prediction_list=[], is_first=True):
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
            is_first: determines whether this is the first iteration for a given mixture. If it is,
                all global variables will be reset
        Returns:
            prediction_list: a list of all enumerated mixtures
            confidence_list: a list of probabilities associated with the above mixtures
        """

        # Make prediction and confidence lists global so they can be updated recursively
        # If this is the top-level of a new mixture (is_first), reset all variables
        if is_first:
            global updated_pred, updated_conf
            updated_pred, updated_conf = None, None
            prediction_list, confidence_list = [], []
            indiv_pred, indiv_conf = [], []

        prediction, num_phases, certanties = self.kdp.predict(spectrum)

        # Explore all phases with a non-trival probability
        for i in range(num_phases):

            # If individual predictions have been updated recursively, use them for this iteration
            if 'updated_pred' in globals():
                if updated_pred != None:
                    indiv_pred, indiv_conf = updated_pred, updated_conf
                    updated_pred, updated_conf = None, None

            phase_index = np.array(prediction).argsort()[-(i+1)]
            predicted_cmpd = self.reference_phases[phase_index]

            # If the predicted phase has already been identified for the mixture, ignore and move on
            if predicted_cmpd in indiv_pred:
                if i == (num_phases - 1):
                    confidence_list.append(indiv_conf)
                    prediction_list.append(indiv_pred)
                    updated_conf, updated_pred = indiv_conf[:-1], indiv_pred[:-1]
                continue

            # Otherwise if phase is new, add to the suspected mixture
            indiv_pred.append(predicted_cmpd)

            # Tabulate the probability associated with the predicted phase
            indiv_conf.append(certanties[i])

            # Subtract identified phase from the spectrum
            reduced_spectrum, norm = self.get_reduced_pattern(predicted_cmpd, spectrum)

            # If all phases have been identified, tabulate mixture and move on to next
            if norm == None:
                confidence_list.append(indiv_conf)
                prediction_list.append(indiv_pred)
                if i == (num_phases - 1):
                    updated_conf, updated_pred = indiv_conf[:-2], indiv_pred[:-2]
                else:
                    indiv_conf, indiv_pred = indiv_conf[:-1], indiv_pred[:-1]
                continue

            else:
                # If the maximum number of phases has been reached, tabulate mixture and move on to next
                if len(indiv_pred) == self.max_phases:
                    confidence_list.append(indiv_conf)
                    prediction_list.append(indiv_pred)
                    if i == (num_phases - 1):
                        updated_conf, updated_pred = indiv_conf[:-2], indiv_pred[:-2]
                    else:
                        indiv_conf, indiv_pred = indiv_conf[:-1], indiv_pred[:-1]
                    continue

                # Otherwise if more phases are to be explored, recursively enter enumerate_routes with the newly reduced spectrum
                prediction_list, confidence_list = self.enumerate_routes(reduced_spectrum, indiv_conf, indiv_pred, confidence_list, prediction_list, is_first=False)

        return prediction_list, confidence_list

    def get_reduced_pattern(self, predicted_cmpd, orig_y, last_normalization=1.0):
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

        # Simulate spectrum for predicted compounds
        pred_y = self.generate_pattern(predicted_cmpd)

        # Perform dynamic time warping to fit predicted spectrum to measured spectrum
        dtw_info = dtw(pred_y, orig_y, window_type="slantedband",
            window_args={'window_size': 20}) # a window_size of 20 allows a ~1.5 degree shift
        warp_indices = warp(dtw_info)
        warped_spectrum = list(pred_y[warp_indices])
        warped_spectrum.append(0.0) # Necessary to preserve length

        # Scale warped spectrum so y-values match measured spectrum
        scaled_spectrum = self.scale_spectrum(warped_spectrum, orig_y)

        # Subtract scaled spectrum from measured spectrum
        stripped_y = self.strip_spectrum(scaled_spectrum, orig_y)
        stripped_y = self.smooth_spectrum(stripped_y)
        stripped_y = np.array(stripped_y) - min(stripped_y)

        # If intensities remain above cutoff, return stripped spectrum
        if max(stripped_y) >= (self.cutoff*last_normalization):
            new_normalization = 100/max(stripped_y)
            stripped_y = new_normalization*stripped_y
            return stripped_y, new_normalization

        # Otherwise if intensities are too low, halt the enumaration
        else:
            return orig_y, None

    def calc_std_dev(self, two_theta, tau):
        """
        calculate standard deviation based on angle (two theta) and domain size (tau)
        Args:
            two_theta: angle in two theta space
            tau: domain size in nm
        Returns:
            standard deviation for gaussian kernel
        """
        ## Calculate FWHM based on the Scherrer equation
        K = 0.9 ## shape factor
        wavelength = self.calculator.wavelength * 0.1 ## angstrom to nm
        theta = np.radians(two_theta/2.) ## Bragg angle in radians
        beta = (K * wavelength) / (np.cos(theta) * tau) # in radians

        ## Convert FWHM to std deviation of gaussian
        sigma = np.sqrt(1/(2*np.log(2)))*0.5*np.degrees(beta)
        return sigma**2

    def generate_pattern(self, cmpd):
        """
        Calculate the XRD spectrum of a given compound.

        Args:
            cmpd: filename of the structure file to calculate the spectrum for
        Returns:
            all_I: list of intensities as a function of two-theta
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # don't print occupancy-related warnings
            struct = Structure.from_file('%s/%s' % (self.ref_dir, cmpd))
        equil_vol = struct.volume
        pattern = self.calculator.get_pattern(struct, two_theta_range=(self.min_angle, self.max_angle))
        angles = pattern.x
        intensities = pattern.y

        steps = np.linspace(self.min_angle, self.max_angle, 4501)

        signals = np.zeros([len(angles), steps.shape[0]])

        for i, ang in enumerate(angles):
            # Map angle to closest datapoint step
            idx = np.argmin(np.abs(ang-steps))
            signals[i,idx] = intensities[i]

        # Convolute every row with unique kernel
        # Iterate over rows; not vectorizable, changing kernel for every row
        domain_size = 25.0
        step_size = (self.max_angle - self.min_angle)/4501
        for i in range(signals.shape[0]):
            row = signals[i,:]
            ang = steps[np.argmax(row)]
            std_dev = self.calc_std_dev(ang, domain_size)
            # Gaussian kernel expects step size 1 -> adapt std_dev
            signals[i,:] = gaussian_filter1d(row, np.sqrt(std_dev)*1/step_size,
                                             mode='constant')

        # Combine signals
        signal = np.sum(signals, axis=0)

        # Normalize signal
        norm_signal = 100 * signal / max(signal)

        return norm_signal

    def scale_spectrum(self, warped_spectrum, orig_y):
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

        # Get peak indices
        orig_peaks = find_peaks(orig_y, height=5)[0]
        pred_peaks = find_peaks(warped_spectrum, height=5)[0]

        # Determine which peaks are associated with one another
        matched_orig_peaks = []
        matched_pred_peaks = []
        for a in orig_peaks:
            for b in pred_peaks:
                if np.isclose(a, b, atol=50):
                    matched_orig_peaks.append(a)
                    matched_pred_peaks.append(b)

        # Find scaling factor that gives best match in peak intensities
        num_match = []
        for scaled_spec in np.linspace(1.2, 0.2, 101):
            check = scaled_spec*np.array(warped_spectrum)
            good_peaks = 0
            for (a, b) in zip(matched_orig_peaks, matched_pred_peaks):
                A_magnitude = orig_y[a]
                B_magnitude = check[b]
                if abs((A_magnitude - B_magnitude)/A_magnitude) < 0.1: ## If peaks are within 10% of one another
                    good_peaks += 1
            num_match.append(good_peaks)

        # Scale the predicted spectrum accordingly
        best_scale = np.linspace(1.2, 0.2, 101)[np.argmax(num_match)]
        scaled_spectrum = best_scale*np.array(warped_spectrum)

        return scaled_spectrum

    def strip_spectrum(self, warped_spectrum, orig_y):
        """
        Subtract one spectrum from another. Note that when subtraction produces
        negative intensities, those values are re-normalized to zero. This way,
        the CNN can handle the spectrum reliably.

        Args:
            warped_spectrum: spectrum associated with the identified phase
            orig_y: original (measured) spectrum
        Returns:
            fixed_y: resulting spectrum from the subtraction of warped_spectrum
                from orig_y
        """

        # Subtract predicted spectrum from measured spectrum
        stripped_y = orig_y - warped_spectrum

        # Normalize all negative values to 0.0
        fixed_y = []
        for val in stripped_y:
            if val < 0:
                fixed_y.append(0.0)
            else:
                fixed_y.append(val)

        return fixed_y


class KerasDropoutPrediction(object):
    """
    Ensemble model used to provide a probability distribution associated
    with suspected phases in a given xrd spectrum.
    """

    def __init__(self, model):
        """
        Args:
            model: trained convolutional neural network
                (tensorflow.keras Model object)
        """

        self.f = tf.keras.backend.function(model.layers[0].input, model.layers[-1].output)

    def predict(self, x, n_iter=200):
        """
        Args:
            x: xrd spectrum to be classified
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
            if conf >= 0.10: ## If prediction occurs at least 10% of the time
                certanties.append(conf)
        certanties = sorted(certanties, reverse=True)

        return prediction, len(certanties), certanties


class PhaseIdentifier(object):
    """
    Class used to identify phases from a given set of xrd spectra
    """

    def __init__(self, spectra_directory, reference_directory, max_phases, cutoff_intensity, wavelength, min_angle=10.0, max_angle=80.0, parallel=True):
        """
        Args:
            spectra_dir: path to directory containing the xrd
                spectra to be analyzed
            reference_directory: path to directory containing
                the reference phases
        """

        self.num_cpu = multiprocessing.cpu_count()
        self.spectra_dir = spectra_directory
        self.ref_dir = reference_directory
        self.max_phases = max_phases
        self.cutoff = cutoff_intensity
        self.wavelen = wavelength
        self.parallel = parallel
        self.min_angle = min_angle
        self.max_angle = max_angle

    @property
    def all_predictions(self):
        """
        Returns:
            spectrum_names: filenames of spectra being classified
            predicted_phases: a list of the predicted phases in the mixture
            confidences: the associated confidence with the prediction above
        """

        reference_phases = sorted(os.listdir(self.ref_dir))
        spectrum_filenames = os.listdir(self.spectra_dir)

        if self.parallel:
            with Manager() as manager:
                pool = Pool(self.num_cpu)
                print('Running phase identification')
                all_info = list(tqdm(pool.imap(self.classify_mixture, spectrum_filenames),
                    total=len(spectrum_filenames)))
                spectrum_fnames = [info[0] for info in all_info]
                predicted_phases = [info[1] for info in all_info]
                confidences = [info[2] for info in all_info]

        else:
            all_info = []
            for filename in spectrum_filenames:
                all_info.append(self.classify_mixture(filename))
            spectrum_fnames = [info[0] for info in all_info]
            predicted_phases = [info[1] for info in all_info]
            confidences = [info[2] for info in all_info]

        return spectrum_fnames, predicted_phases, confidences

    def classify_mixture(self, spectrum_fname):
        """
        Args:
            fname: filename string of the spectrum to be classified
        Returns:
            fname: filename, same as in Args
            predicted_set: string of compounds predicted by phase ID algo
            max_conf: confidence associated with the prediction
        """

        total_confidence, all_predictions = [], []
        tabulate_conf, predicted_cmpd_set = [], []

        spec_analysis = SpectrumAnalyzer(self.spectra_dir, spectrum_fname, self.max_phases,
            self.cutoff, wavelen=self.wavelen, min_angle=self.min_angle, max_angle=self.max_angle)

        mixtures, confidences = spec_analysis.suspected_mixtures

        # If classification is non-trival, identify most probable mixture
        if len(confidences) > 0:
            avg_conf = [np.mean(conf) for conf in confidences]
            max_conf_ind = np.argmax(avg_conf)
            confidences = [round(100*val, 2) for val in confidences[max_conf_ind]]
            final_confidences = ['%s%%' % val for val in confidences]
            predicted_cmpds = [fname[:-4] for fname in mixtures[max_conf_ind]]
            predicted_set = ' + '.join(predicted_cmpds)

        # Otherwise, return None
        else:
            final_confidences = ['0%']
            predicted_set = 'None'

        return [spectrum_fname, predicted_set, final_confidences]


def main(spectra_directory, reference_directory, max_phases=3, cutoff_intensity=10, wavelength='CuKa', min_angle=10.0, max_angle=80.0, parallel=True):

    phase_id = PhaseIdentifier(spectra_directory, reference_directory, max_phases,
        cutoff_intensity, wavelength, min_angle, max_angle, parallel)

    spectrum_names, predicted_phases, confidences = phase_id.all_predictions

    return spectrum_names, predicted_phases, confidences
