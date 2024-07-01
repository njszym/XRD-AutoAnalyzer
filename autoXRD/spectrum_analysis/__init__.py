from scipy.signal import find_peaks, filtfilt, resample
from tensorflow.keras.utils import custom_object_scope
from autoXRD.dara import do_refinement_no_saving
from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool, Manager
from scipy import interpolate as ip
from pymatgen.core import Structure
import matplotlib.pyplot as plt
from skimage import restoration
from pathlib import Path
import tensorflow as tf
from scipy import signal
from pyts import metrics
import multiprocessing
from tqdm import tqdm
import pymatgen as mg
import numpy as np
import warnings
import random
import shutil
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(1)
tf.random.set_seed(1)

# Used to apply dropout during training *and* inference
class CustomDropout(tf.keras.layers.Layer):

    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def get_config(self):
        config = super().get_config()
        config.update({
            "rate": self.rate
        })
        return config

    # Always apply dropout
    def call(self, inputs, training=None):
        return tf.nn.dropout(inputs, rate=self.rate)


class SpectrumAnalyzer(object):
    """
    Class used to process and classify xrd spectra.
    """

    def __init__(self, spectra_dir, spectrum_fname, max_phases, cutoff_intensity, min_conf=25.0, wavelen='CuKa',
        reference_dir='References', min_angle=10.0, max_angle=80.0, model_path='Model.h5', is_pdf=False):
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
        self.min_conf = min_conf
        self.wavelen = wavelen
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.model_path = model_path
        self.is_pdf = is_pdf

    @property
    def reference_phases(self):
        # Ignore hidden files that sometimes appear (like .DS_Store on Mac)
        return [fname for fname in sorted(os.listdir(self.ref_dir)) if fname[0] != '.']

    @property
    def suspected_mixtures(self):
        """
        Returns:
            prediction_list: a list of all enumerated mixtures
            confidence_list: a list of probabilities associated with the above mixtures
        """

        spectrum = self.formatted_spectrum

        with custom_object_scope({'CustomDropout': CustomDropout}):
            self.model = tf.keras.models.load_model(self.model_path, compile=False)

        self.kdp = KerasDropoutPrediction(self.model)

        prediction_list, confidence_list, backup_list, scale_list, spec_list = self.enumerate_routes(spectrum)

        return prediction_list, confidence_list, backup_list, scale_list, spec_list

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

        if min(y) < 0:
            print('WARNING: at least one of your patterns contains negative intensity values. These will be shifted to zero for further analysis, but please double-check the quality of your data.')
            y -= min(y)

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

        ## Normalize from 0 to 255
        ys = np.array(ys) - min(ys)
        ys = list(255*np.array(ys)/max(ys))

        # Subtract background
        background = restoration.rolling_ball(ys, radius=800)
        ys = np.array(ys) - np.array(background)

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

    def enumerate_routes(self, xrd_spectrum, indiv_pred=[], indiv_conf=[], indiv_backup=[], prediction_list=[], confidence_list=[],
        backup_list=[], is_first=True, normalization=1.0, indiv_scale=[], scale_list=[], indiv_spec=[], spec_list=[]):
        """
        A branching algorithm designed to explore all suspected mixtures predicted by the CNN.
        For each mixture, the associated phases and probabilities are tabulated.

        Args:
            xrd_spectrum: a numpy array containing the measured spectrum that is to be classified
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
            normalization: keep track of stripped pattern intensity relative to initial maximum.
                For example, a stripped pattern with half the intensity of hte initial maximum
                should be associated with a normalization constant of 2 (I_0/I_new).
        Returns:
            prediction_list: a list of all enumerated mixtures
            confidence_list: a list of probabilities associated with the above mixtures
        """

        # Convert to PDF if specified
        if self.is_pdf:
            pdf_spectrum = self.XRDtoPDF(xrd_spectrum, self.min_angle, self.max_angle)

        # Make prediction and confidence lists global so they can be updated recursively
        # If this is the top-level of a new mixture (is_first), reset all variables
        if is_first:
            global updated_pred, updated_conf, updated_backup, updated_scale, updated_spec
            updated_pred, updated_conf, updated_backup, updated_scale, updated_spec = None, None, None, None, None
            prediction_list, confidence_list, backup_list, scale_list = [], [], [], []
            indiv_pred, indiv_conf, indiv_backup, indiv_scale = [], [], [], []
            indiv_spec, spec_list = [], []

        # Make prediction regarding top phases
        if self.is_pdf:
            prediction, num_phases, certanties, num_outputs = self.kdp.predict(pdf_spectrum, self.min_conf)
        else:
            prediction, num_phases, certanties, num_outputs = self.kdp.predict(xrd_spectrum, self.min_conf)

        assert num_outputs == len(self.reference_phases), \
            """The number of files in the References folder does not
            match the number of outputs in the CNN. Please ensure
            you have not added or removed any of the CIFs in
            your References folder since training the model."""

        # If no phases are suspected
        if num_phases == 0:

            # If individual predictions have been updated recursively, use them for this iteration
            if 'updated_pred' in globals():
                if updated_pred != None:
                    indiv_pred, indiv_conf, indiv_scale, indiv_backup, indiv_spec = updated_pred, updated_conf, updated_scale, updated_backup, updated_spec
                    updated_pred, updated_conf, updated_scale, updated_backup, updated_spec = None, None, None, None, None

            confidence_list.append(indiv_conf)
            prediction_list.append(indiv_pred)
            backup_list.append(indiv_backup)
            scale_list.append(indiv_scale)
            spec_list.append(indiv_spec)

        # Explore all phases with a non-trival probability
        for i in range(num_phases):

            # If individual predictions have been updated recursively, use them for this iteration
            if 'updated_pred' in globals():
                if updated_pred != None:
                    indiv_pred, indiv_conf, indiv_scale, indiv_backup, indiv_spec = updated_pred, updated_conf, updated_scale, updated_backup, updated_spec
                    updated_pred, updated_conf, updated_scale, updated_backup, updated_spec = None, None, None, None, None

            phase_index = np.array(prediction).argsort()[-(i+1)]
            predicted_cmpd = self.reference_phases[phase_index]

            # If there exists two probable phases
            if num_phases > 1:
                # For 1st most probable phase, choose 2nd most probable as backup
                if i == 0:
                    backup_index = np.array(prediction).argsort()[-(i+2)]
                # For 2nd most probable phase, choose 1st most probable as backup
                # For 3rd most probable phase, choose 2nd most probable as backup (and so on)
                elif i >= 1:
                    backup_index = np.array(prediction).argsort()[-i]
                backup_cmpd = self.reference_phases[backup_index]
            # If only one phase is suspected, no backups are needed
            else:
                backup_cmpd = None

            # If the predicted phase has already been identified for the mixture, ignore and move on
            if predicted_cmpd in indiv_pred:
                if i == (num_phases - 1):
                    confidence_list.append(indiv_conf)
                    prediction_list.append(indiv_pred)
                    backup_list.append(indiv_backup)
                    scale_list.append(indiv_scale)
                    spec_list.append(indiv_spec)
                    updated_conf, updated_pred = indiv_conf[:-1], indiv_pred[:-1]
                    updated_backup, updated_scale = indiv_backup[:-1], indiv_scale[:-1]
                    updated_spec = indiv_spec[:-1]

                continue

            # Otherwise if phase is new, add to the suspected mixture
            indiv_pred.append(predicted_cmpd)

            # Tabulate the probability associated with the predicted phase
            indiv_conf.append(certanties[i])

            # Tabulate alternative phases
            indiv_backup.append(backup_cmpd)

            # Subtract identified phase from the spectrum
            reduced_spectrum, norm, scaling_constant, is_done = self.get_reduced_pattern(predicted_cmpd, xrd_spectrum, last_normalization=normalization)

            # Record actual spectrum (non-scaled) after peak substraction of known phases
            actual_spectrum = reduced_spectrum / norm
            indiv_spec.append(actual_spectrum)

            # Record scaling constant for each phase (to be used for weight fraction estimates)
            indiv_scale.append(scaling_constant)

            # If all phases have been identified, tabulate mixture and move on to next
            if is_done:
                reduced_spectrum = xrd_spectrum.copy()
                confidence_list.append(indiv_conf)
                prediction_list.append(indiv_pred)
                scale_list.append(indiv_scale)
                backup_list.append(indiv_backup)
                spec_list.append(indiv_spec)
                if i == (num_phases - 1):
                    updated_conf, updated_pred = indiv_conf[:-2], indiv_pred[:-2]
                    updated_backup, updated_scale = indiv_backup[:-2], indiv_scale[:-2]
                    updated_spec = indiv_spec[:-2]
                else:
                    indiv_conf, indiv_pred = indiv_conf[:-1], indiv_pred[:-1]
                    indiv_backup, indiv_scale = indiv_backup[:-1], indiv_scale[:-1]
                    updated_spec = indiv_spec[:-2]
                continue

            else:
                # If the maximum number of phases has been reached, tabulate mixture and move on to next
                if len(indiv_pred) == self.max_phases:
                    confidence_list.append(indiv_conf)
                    prediction_list.append(indiv_pred)
                    scale_list.append(indiv_scale)
                    backup_list.append(indiv_backup)
                    spec_list.append(indiv_spec)
                    if i == (num_phases - 1):
                        updated_conf, updated_pred = indiv_conf[:-2], indiv_pred[:-2]
                        updated_backup, updated_scale = indiv_backup[:-2], indiv_scale[:-2]
                        updated_spec = indiv_spec[:-2]
                    else:
                        indiv_conf, indiv_pred = indiv_conf[:-1], indiv_pred[:-1]
                        indiv_backup, indiv_scale = indiv_backup[:-1], indiv_scale[:-1]
                        indiv_spec = indiv_spec[:-1]
                    continue

                # Otherwise if more phases are to be explored, recursively enter enumerate_routes with the newly reduced spectrum
                prediction_list, confidence_list, backup_list, scale_list, spec_list = \
                    self.enumerate_routes(reduced_spectrum, indiv_pred, indiv_conf, indiv_backup, prediction_list,
                    confidence_list, backup_list, is_first=False, normalization=norm, indiv_scale=indiv_scale,
                    scale_list=scale_list, indiv_spec=indiv_spec, spec_list=spec_list)

        return prediction_list, confidence_list, backup_list, scale_list, spec_list

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

        x_obs = xs = np.linspace(self.min_angle, self.max_angle, 4501)
        y_obs = np.array(orig_y)

        with open('temp/%s' % self.spectrum_fname, 'w+') as f:
            for xv, yv in zip(x_obs, y_obs):
                f.write('%s %s\n' % (xv, yv + 1e-6))

        result = do_refinement_no_saving(
            pattern_path=Path('temp/%s' % self.spectrum_fname),
            phase_paths=[
                Path('%s/%s' % (self.ref_dir, predicted_cmpd))
            ],
            instrument_name="Rigaku-Miniflex",
            phase_params={
                "gewicht": "SPHAR6",
                "lattice_range": 0.03,  # 3% lattice strain allowed
                "k1": "0_0^0.01",
                "k2": "0_0^0.01",
                "b1": "0_0^0.02",  # the particle size
                "rp": 4,
            }
        )

        cmpd_key = predicted_cmpd[:-4] # Remove .cif
        scaled_spectrum =  result["plot_data"]["structs"][cmpd_key]

        scaling_constant = max(scaled_spectrum) / max(y_obs)
        if scaling_constant > 1:
            scaling_constant = 1.0

        # Over-subtract peaks to avoid large residuals
        scaled_spectrum = 1.25*np.array(scaled_spectrum)

        # Subtract scaled spectrum from measured spectrum
        stripped_y = self.strip_spectrum(scaled_spectrum, orig_y)
        stripped_y = self.smooth_spectrum(stripped_y)
        stripped_y = np.array(stripped_y) - min(stripped_y)

        # Normalization
        new_normalization = 100/max(stripped_y)
        actual_intensity = max(stripped_y)/last_normalization

        # Calculate actual scaling constant
        scaling_constant /= last_normalization

        # If intensities fall above cutoff, halt enumeration
        if actual_intensity < self.cutoff:
            is_done = True
        else:
            is_done = False

        stripped_y = new_normalization*stripped_y

        return stripped_y, last_normalization*new_normalization, scaling_constant, is_done


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

    def scale_spectrum(self, pred_y, obs_y):
        """
        Scale the magnitude of a calculated spectrum associated with an identified
        phase so that its peaks match with those of the measured spectrum being classified.

        Args:
            pred_y: spectrum calculated from the identified phase after fitting
                has been performed along the x-axis using DTW
            obs_y: observed (experimental) spectrum containing all peaks
        Returns:
            scaled_spectrum: spectrum associated with the reference phase after scaling
                has been performed to match the peaks in the measured pattern.
        """

        # Ensure inputs are numpy arrays
        pred_y = np.array(pred_y)
        obs_y = np.array(obs_y)

        # Find scaling constant that minimizes MSE between pred_y and obs_y
        all_mse = []
        for scale_spectrum in np.linspace(1.1, 0.05, 101):
            ydiff = obs_y - (scale_spectrum*pred_y)
            mse = np.mean(ydiff**2)
            all_mse.append(mse)
        best_scale = np.linspace(1.0, 0.05, 101)[np.argmin(all_mse)]
        scaled_spectrum = best_scale*np.array(pred_y)

        return scaled_spectrum, best_scale

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
        stripped_y = np.array(orig_y) - np.array(warped_spectrum)

        # Normalize all negative values to 0.0
        fixed_y = []
        for val in stripped_y:
            if val < 0:
                fixed_y.append(0.0)
            else:
                fixed_y.append(val)

        return fixed_y

    def XRDtoPDF(self, xrd, min_angle, max_angle):

        thetas = np.linspace(min_angle/2.0, max_angle/2.0, 4501)
        Q = np.array([4*math.pi*math.sin(math.radians(theta))/1.5406 for theta in thetas])
        S = np.array(xrd).flatten()

        pdf = []
        R = np.linspace(1, 40, 1000) # Only 1000 used to reduce compute time
        integrand = Q * S * np.sin(Q * R[:, np.newaxis])

        pdf = (2*np.trapz(integrand, Q) / math.pi)
        pdf = list(signal.resample(pdf, 4501))

        return pdf


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

        self.model = model

    def predict(self, x, min_conf=10.0, n_iter=100):
        """
        Args:
            x: xrd spectrum to be classified
        Returns:
            prediction: distribution of probabilities associated with reference phases
            len(certainties): number of phases with probabilities > 10%
            certanties: associated probabilities
        """

        # Convert from % to 0-1 fractional
        if min_conf > 1.0:
            min_conf /= 100.0

        # Format input
        x = np.array([x])

        # Monte Carlo Dropout
        result = []
        for _ in range(n_iter):
            result.append(self.model(x))

        result = np.array([list(np.array(sublist).flatten()) for sublist in result]) ## Individual predictions
        prediction = result.mean(axis=0) ## Average prediction

        num_outputs = len(prediction) # Check how many possible phases there are

        all_preds = [np.argmax(pred) for pred in result] ## Individual max indices (associated with phases)

        counts = []
        for index in set(all_preds):
            counts.append(all_preds.count(index)) ## Tabulate how many times each prediction arises

        certanties = []
        for each_count in counts:
            conf = each_count/sum(counts)
            if conf >= min_conf:
                certanties.append(conf)
        certanties = sorted(certanties, reverse=True)

        return prediction, len(certanties), certanties, num_outputs


class PhaseIdentifier(object):
    """
    Class used to identify phases from a given set of xrd spectra
    """

    def __init__(self, spectra_directory, reference_directory, max_phases, cutoff_intensity, min_conf, wavelength,
        min_angle=10.0, max_angle=80.0, parallel=True, model_path='Model.h5', is_pdf=False):
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
        self.min_conf = min_conf
        self.wavelen = wavelength
        self.parallel = parallel
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.model_path = model_path
        self.is_pdf = is_pdf

    @property
    def all_predictions(self):
        """
        Returns:
            spectrum_names: filenames of spectra being classified
            predicted_phases: a list of the predicted phases in the mixture
            confidences: the associated confidence with the prediction above
        """

        # Ignore hidden files that sometimes appear (like .DS_Store on Mac)
        reference_phases = [fname for fname in sorted(os.listdir(self.ref_dir)) if fname[0] != '.']

        spectrum_filenames = os.listdir(self.spectra_dir)
        spectrum_filenames = [fname for fname in spectrum_filenames if fname[0] != '.']

        if self.parallel:
            with Manager() as manager:
                pool = Pool(self.num_cpu)
                if self.is_pdf:
                    print('Running PDF analysis')
                else:
                    print('Running XRD analysis')
                all_info = list(tqdm(pool.imap(self.classify_mixture, spectrum_filenames),
                    total=len(spectrum_filenames)))

        else:
            all_info = []
            for filename in spectrum_filenames:
                all_info.append(self.classify_mixture(filename))

        spectrum_fnames = [info[0] for info in all_info]
        predicted_phases = [info[1] for info in all_info]
        confidences = [info[2] for info in all_info]
        backup_phases = [info[3] for info in all_info]
        scale_factors = [info[4] for info in all_info]
        spectra = [info[5][-1] for info in all_info]

        return spectrum_fnames, predicted_phases, confidences, backup_phases, scale_factors, spectra

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

        spec_analysis = SpectrumAnalyzer(self.spectra_dir, spectrum_fname, self.max_phases, self.cutoff,
            self.min_conf, wavelen=self.wavelen, min_angle=self.min_angle, max_angle=self.max_angle,
            model_path=self.model_path, is_pdf=self.is_pdf)

        mixtures, confidences, backup_mixtures, scalings, spectra = spec_analysis.suspected_mixtures

        # If classification is non-trival, identify most probable mixture
        if any(confidences):
            avg_conf = [np.mean(conf) for conf in confidences]
            max_conf_ind = np.argmax(avg_conf)
            final_confidences = [round(100*val, 2) for val in confidences[max_conf_ind]]
            scaling_constants = [round(val, 3) for val in scalings[max_conf_ind]]
            specs = spectra[max_conf_ind]
            predicted_set = [fname[:-4] for fname in mixtures[max_conf_ind]]
            backup_set = []
            for ph in backup_mixtures[max_conf_ind]:
                if 'cif' in str(ph):
                    backup_set.append(ph[:-4])
                else:
                    backup_set.append('None')

        # Otherwise, return None
        else:
            final_confidences = [0.0]
            scaling_constants = [0.0]
            specs = [[]]
            predicted_set = ['None']
            backup_set = ['None']

        return [spectrum_fname, predicted_set, final_confidences, backup_set, scaling_constants, specs]

def merge_results(results, cutoff, max_phases):
    """
    Aggregate XRD + PDF predictions through an ensemble approach
    whereby each phase is weighted by its confidence.

    xrd: dictionary containing predictions from XRD analysis
    pdf: dictionary containing predictions from PDF analysis
    cutoff: minimum confidence (%) to include in final predictions
    max_phases: maximum no. of phases to include in final predictions
    """

    # First, sort by filename to ensure consistent ordering
    zipped_xrd = zip(results['XRD']['filenames'], results['XRD']['phases'], results['XRD']['confs'],
            results['XRD']['backup_phases'], results['XRD']['scale_factors'], results['XRD']['reduced_spectra'])
    zipped_pdf = zip(results['PDF']['filenames'], results['PDF']['phases'], results['PDF']['confs'],
            results['PDF']['backup_phases'], results['PDF']['scale_factors'], results['PDF']['reduced_spectra'])
    sorted_xrd = sorted(zipped_xrd, key=lambda x:x[0])
    sorted_pdf = sorted(zipped_pdf, key=lambda x:x[0])
    results['XRD']['filenames'], results['XRD']['phases'], results['XRD']['confs'], results['XRD']['backup_phases'], \
        results['XRD']['scale_factors'], results['XRD']['reduced_spectra'] = \
        list(zip(*sorted_xrd))
    results['PDF']['filenames'], results['PDF']['phases'], results['PDF']['confs'], results['PDF']['backup_phases'], \
        results['PDF']['scale_factors'], results['PDF']['reduced_spectra'] = \
        list(zip(*sorted_pdf))

    # Double-check consistent length and order
    assert len(results['XRD']['filenames']) == len(results['PDF']['filenames']), \
        'XRD and PDF prediction are not the same length. Something went wrong.'
    for i, filename in enumerate(results['XRD']['filenames']):
        assert filename == results['XRD']['filenames'][i], \
        'Mismatch between order of XRD and PDF predictions. Something went wrong'

    # Concatenate phases and confidences
    results['All'] = {}
    results['All']['phases'] = []
    for l1, l2 in zip(results['XRD']['phases'], results['PDF']['phases']):
        results['All']['phases'].append(list(l1) + list(l2))
    results['All']['confs'] = []
    for l1, l2 in zip(results['XRD']['confs'], results['PDF']['confs']):
        results['All']['confs'].append(list(l1) + list(l2))

    # Aggregate XRD and PDF predictions into merged dictionary
    results['Merged'] = {}
    results['Merged']['phases'] = []
    results['Merged']['confs'] = []
    results['Merged']['backup_phases'] = []
    results['Merged']['scale_factors'] = []
    results['Merged']['filenames'] = results['XRD']['filenames']
    results['Merged']['reduced_spectra'] = results['XRD']['reduced_spectra']
    for phases, confs, xrd_phases, xrd_confs, pdf_phases, pdf_confs, xrd_backup_phases, \
        pdf_backup_phases, xrd_scale_factors, pdf_scale_factors in \
        zip(results['All']['phases'], results['All']['confs'], results['XRD']['phases'],\
        results['XRD']['confs'], results['PDF']['phases'], results['PDF']['confs'], \
        results['XRD']['backup_phases'], results['PDF']['backup_phases'],
        results['XRD']['scale_factors'], results['PDF']['scale_factors']):

        # Allocate confidences for each phase
        avg_soln = {}
        for cmpd, cf in zip(phases, confs):
            if cmpd not in avg_soln.keys():
                avg_soln[cmpd] = [cf]
            else:
                avg_soln[cmpd].append(cf)

        # Average over confidences for each phase
        unique_phases, avg_confs = [], []
        for cmpd in avg_soln.keys():
            unique_phases.append(cmpd)
            num_zeros = 2 - len(avg_soln[cmpd])
            avg_soln[cmpd] += [0.0]*num_zeros
            avg_confs.append(np.mean(avg_soln[cmpd]))

        # Sort by confidence
        info = zip(unique_phases, avg_confs)
        info = sorted(info, key=lambda x: x[1])
        info.reverse()

        # Get all unique phases below max no.
        unique_phases, unique_confs = [], []
        for cmpd, cf in info:
            if (len(unique_phases) < max_phases) and (cf > cutoff):
                unique_phases.append(cmpd)
                unique_confs.append(cf)

        # Collect final backups and scale factors
        unique_backups, unique_scales = [], []
        for cmpd in unique_phases:
            if cmpd in xrd_phases:
                xrd_ind = xrd_phases.index(cmpd)
                xrd_conf = xrd_confs[xrd_ind]
            else:
                xrd_conf = 0.0
            if cmpd in pdf_phases:
                pdf_ind = pdf_phases.index(cmpd)
                pdf_conf = pdf_confs[pdf_ind]
            else:
                pdf_conf = 0.0
            if xrd_conf >= pdf_conf:
                unique_backups.append(xrd_backup_phases[xrd_ind])
                unique_scales.append(xrd_scale_factors[xrd_ind])
            else:
                unique_backups.append(pdf_backup_phases[pdf_ind])
                unique_scales.append(pdf_scale_factors[pdf_ind])

        results['Merged']['phases'].append(unique_phases)
        results['Merged']['confs'].append(unique_confs)
        results['Merged']['backup_phases'].append(unique_backups)
        results['Merged']['scale_factors'].append(unique_scales)

    return results['Merged']


def main(spectra_directory, reference_directory, max_phases=3, cutoff_intensity=10, min_conf=10.0,wavelength='CuKa',
    min_angle=10.0, max_angle=80.0, parallel=True, model_path='Model.h5', is_pdf=False):

    phase_id = PhaseIdentifier(spectra_directory, reference_directory, max_phases,
        cutoff_intensity, min_conf, wavelength, min_angle, max_angle, parallel, model_path, is_pdf)

    spectrum_names, predicted_phases, confidences, backup_phases, scale_factors, reduced_spectra = phase_id.all_predictions

    return spectrum_names, predicted_phases, confidences, backup_phases, scale_factors, reduced_spectra
