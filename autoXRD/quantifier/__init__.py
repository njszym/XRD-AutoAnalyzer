from pymatgen.core.periodic_table import Element
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, filtfilt, resample
from autoXRD.dara import do_refinement_no_saving, get_phase_weights, get_structure
from pymatgen.io.cif import CifWriter
from pathlib import Path
import random
import pymatgen as mg
from skimage import restoration
from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate as ip
from pymatgen.core import Structure
from pyts import metrics
import warnings
import numpy as np
import math
import os


class QuantAnalysis(object):
    """
    Class used to plot and compare:
    (i) measured xrd spectra
    (ii) line profiles of identified phases
    """

    def __init__(self, spectra_dir, spectrum_fname, predicted_phases, scale_factors, min_angle=10.0, max_angle=80.0, wavelength='CuKa', reference_dir='References'):
        """
        Args:
            spectrum_fname: name of file containing the
                xrd spectrum (in xy format)
            reference_dir: path to directory containing the
                reference phases (CIF files)
        """

        self.spectra_dir = spectra_dir
        self.spectrum_fname = spectrum_fname
        self.pred_phases = predicted_phases
        self.scale_factors = scale_factors
        self.ref_dir = reference_dir
        self.calculator = xrd.XRDCalculator()
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.wavelen = wavelength

        # If scale factors haven't been calculated yet, do it now
        # For use with the visualize script
        if self.scale_factors == None:
            self.scale_factors = self.calc_scale()

    def calc_scale(self):

        norm = 1.0
        heights = []
        spec = self.formatted_spectrum
        for cmpd in self.pred_phases:
            spec, norm, scale = self.get_reduced_pattern(cmpd, spec, norm)
            heights.append(scale)

        return heights

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

        # Convert to numpy arrays
        pred_y = np.array(pred_y)
        orig_y = np.array(orig_y)

        # Downsample spectra (helps reduce time for DTW)
        downsampled_res = 0.1 # new resolution: 0.1 degrees
        num_pts = int((self.max_angle - self.min_angle) / downsampled_res)
        orig_y = resample(orig_y, num_pts)
        pred_y = resample(pred_y, num_pts)

        # Calculate window size for DTW
        allow_shifts = 0.75 # Allow shifts up to 0.75 degrees
        window_size = int(allow_shifts * num_pts / (self.max_angle - self.min_angle))

        # Get warped spectrum (DTW)
        distance, path = metrics.dtw(pred_y, orig_y, method='sakoechiba', options={'window_size': window_size}, return_path=True)
        index_pairs = path.transpose()
        warped_spectrum = orig_y.copy()
        for ind1, ind2 in index_pairs:
            distance = abs(ind1 - ind2)
            if distance <= window_size:
                warped_spectrum[ind2] = pred_y[ind1]
            else:
                warped_spectrum[ind2] = 0.0

        # Now, upsample spectra back to their original size (4501)
        warped_spectrum = resample(warped_spectrum, 4501)
        orig_y = resample(orig_y, 4501)

        # Scale warped spectrum so y-values match measured spectrum
        scaled_spectrum, scaling_constant = self.scale_spectrum(warped_spectrum, orig_y)

        # Subtract scaled spectrum from measured spectrum
        stripped_y = self.strip_spectrum(scaled_spectrum, orig_y)
        stripped_y = self.smooth_spectrum(stripped_y)
        stripped_y = np.array(stripped_y) - min(stripped_y)

        # Normalization
        new_normalization = 100/max(stripped_y)
        actual_intensity = max(stripped_y)/last_normalization

        # Calculate actual scaling constant
        scaling_constant /= last_normalization

        # Return stripped spectrum
        stripped_y = new_normalization*stripped_y
        return stripped_y, last_normalization*new_normalization, scaling_constant

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
        stripped_y = orig_y - warped_spectrum

        # Normalize all negative values to 0.0
        fixed_y = []
        for val in stripped_y:
            if val < 0:
                fixed_y.append(0.0)
            else:
                fixed_y.append(val)

        return fixed_y

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

    @property
    def scaled_patterns(self):
        """
        Get line profiles of predicted phases that are scaled
        to match with peaks in the measured spectrum
        """

        measured_spectrum = self.formatted_spectrum
        pred_phases = self.pred_phases
        heights = self.scale_factors

        angle_sets, intensity_sets = [], []
        for phase, ht in zip(pred_phases, heights):
            angles, intensities = self.get_stick_pattern(phase)
            scaled_intensities = ht*np.array(intensities)
            angle_sets.append(angles)
            intensity_sets.append(scaled_intensities)

        return angle_sets, intensity_sets

    def get_stick_pattern(self, ref_phase):
        """
        Returns:
            all_I: list of intensities as a function of two-theta
        """

        struct = Structure.from_file('%s/%s' % (self.ref_dir, ref_phase))

        pattern = self.calculator.get_pattern(struct, two_theta_range=(self.min_angle, self.max_angle))
        angles = pattern.x
        intensities = pattern.y

        return angles, intensities

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

    def get_cont_profile(self, angles, intensities):

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

    def scale_line_profile(self, angles, intensities):
        """
        Identify the scaling factor that minimizes the differences between a line
        profile and any associated peaks in a measured XRD spectrum.

        Args:
            angles: a list of diffraction angles
            intensities: a list of peak intensities
        Returns:
            best_scale: a float ranging from 0.05 to 1.0 that has been optimized
                to ensure maximal overlap between the line profile and the peaks
                in the measured spectrum.
        """

        # Get patterns
        obs_y = self.formatted_spectrum
        pred_y = self.get_cont_profile(angles, intensities)

        # Downsample
        downsampled_res = 0.1
        num_pts = int((self.max_angle - self.min_angle) / downsampled_res)
        obs_y = resample(obs_y, num_pts)
        pred_y = resample(pred_y, num_pts)
        x = np.linspace(self.min_angle, self.max_angle, num_pts)

        # Calculate window size for DTW
        allow_shifts = 0.75
        window_size = int(allow_shifts * num_pts / (self.max_angle - self.min_angle))

        # Get warped spectrum
        distance, path = metrics.dtw(pred_y, obs_y, method='sakoechiba', options={'window_size': window_size}, return_path=True)
        index_pairs = path.transpose()
        warped_spectrum = obs_y.copy()
        for ind1, ind2 in index_pairs:
            distance = abs(ind1 - ind2)
            if distance <= window_size:
                warped_spectrum[ind2] = pred_y[ind1]
            else:
                warped_spectrum[ind2] = 0.0
        pred_y = 100*np.array(warped_spectrum)/max(warped_spectrum)

        # Find scaling constant that minimizes MSE between pred_y and obs_y
        all_mse = []
        for scale_spectrum in np.linspace(1.1, 0.01, 101):
            ydiff = obs_y - (scale_spectrum*pred_y)
            mse = np.mean(ydiff**2)
            all_mse.append(mse)
        best_scale = np.linspace(1.1, 0.01, 101)[np.argmin(all_mse)]

        return best_scale


def get_max_intensity(ref_phase, min_angle, max_angle, ref_dir='References'):
    """
    Returns:
        Retrieve maximum intensity for raw (non-scaled) pattern of ref_phase.
    """

    calculator = xrd.XRDCalculator()

    struct = Structure.from_file('%s/%s' % (ref_dir, ref_phase))

    pattern = calculator.get_pattern(struct, two_theta_range=(min_angle, max_angle), scaled=False)
    angles = pattern.x
    intensities = pattern.y

    return max(intensities)

def get_volume(ref_phase, ref_dir='References'):
    """
    Get unit cell volume of ref_phase.
    """

    struct = Structure.from_file('%s/%s' % (ref_dir, ref_phase))
    return struct.volume

def get_density(ref_phase, ref_dir='References'):
    """
    Get mass density of ref_phase.
    """

    struct = Structure.from_file('%s/%s' % (ref_dir, ref_phase))

    mass = 0
    for site in struct:
        elem_dict = site.species.remove_charges().as_dict()
        for elem_key in elem_dict.keys():
            # Take into account occupancies and species
            mass += elem_dict[elem_key]*Element(elem_key).atomic_mass

    return mass/struct.volume

def main(spectra_directory, spectrum_fname, predicted_phases, scale_factors, min_angle=10.0, max_angle=80.0, wavelength='CuKa', rietveld=True, refined_phases_dir=None):

        if rietveld:

            result = do_refinement_no_saving(
                pattern_path=Path('%s/%s' % (spectra_directory, spectrum_fname)),
                phase_paths=[
                    Path('References/%s' % cmpd) for cmpd in predicted_phases
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

            weight_dict = get_phase_weights(result)

            weights = []
            for ph in predicted_phases:
                ph_name = ph[:-4]
                weights.append(weight_dict[ph_name])
                
                if refined_phases_dir:
                    os.makedirs(f"{refined_phases_dir}/{spectrum_fname}", exist_ok=True)
                    structure = get_structure(result["lst_data"]["phases_results"][ph_name])
                    if structure is not None:
                        cif_writer = CifWriter(structure)
                        cif_writer.write_file(f"{refined_phases_dir}/{spectrum_fname}/{ph_name}.cif")

            return weights

        else:

            analyzer = QuantAnalysis(spectra_directory, spectrum_fname, predicted_phases, scale_factors, min_angle, max_angle, wavelength)

            if len(predicted_phases) == 1:
                return [1.0]

            x = np.linspace(min_angle, max_angle, 4501)
            measured_spectrum = analyzer.formatted_spectrum
            angle_sets, intensity_sets = analyzer.scaled_patterns

            I_expec, I_obs, V, dens = [], [], [], []
            for (cmpd, I_set) in zip(predicted_phases, intensity_sets):
                I_obs.append(max(I_set))
                I_expec.append(get_max_intensity(cmpd, min_angle, max_angle))
                V.append(get_volume(cmpd))
                dens.append(get_density(cmpd))

            if len(predicted_phases) == 2:
                c21_ratio = (I_obs[1] / I_obs[0]) * (I_expec[0] / I_expec[1]) * (V[1]**2 / V[0]**2)
                c1 = 1. / (1. + c21_ratio)
                c2 = 1. - c1
                m1 = (dens[0] * c1) / ((dens[0] * c1) + (dens[1] * c2))
                m2 = 1. - m1
                return [m1, m2]

            if len(predicted_phases) == 3:
                c21_ratio = (I_obs[1] / I_obs[0]) * (I_expec[0] / I_expec[1]) * (V[1]**2 / V[0]**2)
                c31_ratio = (I_obs[2] / I_obs[0]) * (I_expec[0] / I_expec[2]) * (V[2]**2 / V[0]**2)
                c1 = 1. / (1. + c21_ratio + c31_ratio)
                c12_ratio = 1./c21_ratio
                c32_ratio = (I_obs[2] / I_obs[1]) * (I_expec[1] / I_expec[2]) * (V[2]**2 / V[1]**2)
                c2 = 1. / (c12_ratio + 1. + c32_ratio)
                c3 = 1. - c1 - c2
                m1 = (dens[0] * c1) / ((dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3))
                m2 = (dens[1] * c2) / ((dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3))
                m3 = 1. - m1 - m2
                return [m1, m2, m3]

            if len(predicted_phases) == 4:
                c21_ratio = (I_obs[1] / I_obs[0]) * (I_expec[0] / I_expec[1]) * (V[1]**2 / V[0]**2)
                c31_ratio = (I_obs[2] / I_obs[0]) * (I_expec[0] / I_expec[2]) * (V[2]**2 / V[0]**2)
                c41_ratio = (I_obs[3] / I_obs[0]) * (I_expec[0] / I_expec[3]) * (V[3]**2 / V[0]**2)
                c1 = 1. / (1. + c21_ratio + c31_ratio + c41_ratio)
                c12_ratio = 1./c21_ratio
                c32_ratio = (I_obs[2] / I_obs[1]) * (I_expec[1] / I_expec[2]) * (V[2]**2 / V[1]**2)
                c42_ratio = (I_obs[3] / I_obs[1]) * (I_expec[1] / I_expec[3]) * (V[3]**2 / V[1]**2)
                c2 = 1. / (c12_ratio + 1. + c32_ratio + c42_ratio)
                c13_ratio = 1./c31_ratio
                c23_ratio = (I_obs[1] / I_obs[2]) * (I_expec[2] / I_expec[1]) * (V[1]**2 / V[2]**2)
                c43_ratio = (I_obs[3] / I_obs[2]) * (I_expec[2] / I_expec[3]) * (V[3]**2 / V[2]**2)
                c3 = 1. / (c13_ratio + c23_ratio + 1. + c43_ratio)
                c4 = 1. - c1 - c2 - c3
                m1 = (dens[0] * c1) / ((dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3) + (dens[3] * c4))
                m2 = (dens[1] * c2) / ((dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3) + (dens[3] * c4))
                m3 = (dens[2] * c3) / ((dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3) + (dens[3] * c4))
                m4 = 1. - m1 - m2 - m3
                return [m1, m2, m3, m4]

            if len(predicted_phases) == 5:
                c21_ratio = (I_obs[1] / I_obs[0]) * (I_expec[0] / I_expec[1]) * (V[1]**2 / V[0]**2)
                c31_ratio = (I_obs[2] / I_obs[0]) * (I_expec[0] / I_expec[2]) * (V[2]**2 / V[0]**2)
                c41_ratio = (I_obs[3] / I_obs[0]) * (I_expec[0] / I_expec[3]) * (V[3]**2 / V[0]**2)
                c51_ratio = (I_obs[4] / I_obs[0]) * (I_expec[0] / I_expec[4]) * (V[4]**2 / V[0]**2)
                c1 = 1. / (1. + c21_ratio + c31_ratio + c41_ratio + c51_ratio)
                c12_ratio = 1./c21_ratio
                c32_ratio = (I_obs[2] / I_obs[1]) * (I_expec[1] / I_expec[2]) * (V[2]**2 / V[1]**2)
                c42_ratio = (I_obs[3] / I_obs[1]) * (I_expec[1] / I_expec[3]) * (V[3]**2 / V[1]**2)
                c52_ratio = (I_obs[4] / I_obs[1]) * (I_expec[1] / I_expec[4]) * (V[4]**2 / V[1]**2)
                c2 = 1. / (c12_ratio + 1. + c32_ratio + c42_ratio + c52_ratio)
                c13_ratio = 1./c31_ratio
                c23_ratio = (I_obs[1] / I_obs[2]) * (I_expec[2] / I_expec[1]) * (V[1]**2 / V[2]**2)
                c43_ratio = (I_obs[3] / I_obs[2]) * (I_expec[2] / I_expec[3]) * (V[3]**2 / V[2]**2)
                c53_ratio = (I_obs[4] / I_obs[2]) * (I_expec[2] / I_expec[4]) * (V[4]**2 / V[2]**2)
                c3 = 1. / (c13_ratio + c23_ratio + 1. + c43_ratio + c53_ratio)
                c14_ratio = 1./c41_ratio
                c24_ratio = (I_obs[1] / I_obs[3]) * (I_expec[3] / I_expec[1]) * (V[1]**2 / V[3]**2)
                c34_ratio = (I_obs[2] / I_obs[3]) * (I_expec[3] / I_expec[2]) * (V[2]**2 / V[3]**2)
                c54_ratio = (I_obs[4] / I_obs[3]) * (I_expec[3] / I_expec[4]) * (V[4]**2 / V[3]**2)
                c4 = 1. / (c14_ratio + c24_ratio + 1. + c34_ratio + c54_ratio)
                c5 = 1. - c1 - c2 - c3 - c4
                m1 = (dens[0] * c1) / ((dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3) + (dens[3] * c4) + (dens[4] * c5))
                m2 = (dens[1] * c2) / ((dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3) + (dens[3] * c4) + (dens[4] * c5))
                m3 = (dens[2] * c3) / ((dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3) + (dens[3] * c4) + (dens[4] * c5))
                m4 = (dens[3] * c4) / ((dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3) + (dens[3] * c4) + (dens[4] * c5))
                m5 = 1. - m1 - m2 - m3 - m4
                return [m1, m2, m3, m4, m5]

            if len(predicted_phases) == 6:
                c21_ratio = (I_obs[1] / I_obs[0]) * (I_expec[0] / I_expec[1]) * (V[1]**2 / V[0]**2)
                c31_ratio = (I_obs[2] / I_obs[0]) * (I_expec[0] / I_expec[2]) * (V[2]**2 / V[0]**2)
                c41_ratio = (I_obs[3] / I_obs[0]) * (I_expec[0] / I_expec[3]) * (V[3]**2 / V[0]**2)
                c51_ratio = (I_obs[4] / I_obs[0]) * (I_expec[0] / I_expec[4]) * (V[4]**2 / V[0]**2)
                c61_ratio = (I_obs[5] / I_obs[0]) * (I_expec[0] / I_expec[5]) * (V[5]**2 / V[0]**2)
                c1 = 1. / (1. + c21_ratio + c31_ratio + c41_ratio + c51_ratio + c61_ratio)
                c12_ratio = 1./c21_ratio
                c32_ratio = (I_obs[2] / I_obs[1]) * (I_expec[1] / I_expec[2]) * (V[2]**2 / V[1]**2)
                c42_ratio = (I_obs[3] / I_obs[1]) * (I_expec[1] / I_expec[3]) * (V[3]**2 / V[1]**2)
                c52_ratio = (I_obs[4] / I_obs[1]) * (I_expec[1] / I_expec[4]) * (V[4]**2 / V[1]**2)
                c62_ratio = (I_obs[5] / I_obs[1]) * (I_expec[1] / I_expec[5]) * (V[5]**2 / V[1]**2)
                c2 = 1. / (c12_ratio + 1. + c32_ratio + c42_ratio + c52_ratio + c62_ratio)
                c13_ratio = 1./c31_ratio
                c23_ratio = (I_obs[1] / I_obs[2]) * (I_expec[2] / I_expec[1]) * (V[1]**2 / V[2]**2)
                c43_ratio = (I_obs[3] / I_obs[2]) * (I_expec[2] / I_expec[3]) * (V[3]**2 / V[2]**2)
                c53_ratio = (I_obs[4] / I_obs[2]) * (I_expec[2] / I_expec[4]) * (V[4]**2 / V[2]**2)
                c63_ratio = (I_obs[5] / I_obs[2]) * (I_expec[2] / I_expec[5]) * (V[5]**2 / V[2]**2)
                c3 = 1. / (c13_ratio + c23_ratio + 1. + c43_ratio + c53_ratio + c63_ratio)
                c14_ratio = 1./c41_ratio
                c24_ratio = (I_obs[1] / I_obs[3]) * (I_expec[3] / I_expec[1]) * (V[1]**2 / V[3]**2)
                c34_ratio = (I_obs[2] / I_obs[3]) * (I_expec[3] / I_expec[2]) * (V[2]**2 / V[3]**2)
                c54_ratio = (I_obs[4] / I_obs[3]) * (I_expec[3] / I_expec[4]) * (V[4]**2 / V[3]**2)
                c64_ratio = (I_obs[5] / I_obs[3]) * (I_expec[3] / I_expec[5]) * (V[5]**2 / V[3]**2)
                c4 = 1. / (c14_ratio + c24_ratio + 1. + c34_ratio + c54_ratio + c64_ratio)
                c15_ratio = 1./c51_ratio
                c25_ratio = (I_obs[1] / I_obs[4]) * (I_expec[4] / I_expec[1]) * (V[1]**2 / V[4]**2)
                c35_ratio = (I_obs[2] / I_obs[4]) * (I_expec[4] / I_expec[2]) * (V[2]**2 / V[4]**2)
                c45_ratio = (I_obs[3] / I_obs[4]) * (I_expec[4] / I_expec[3]) * (V[3]**2 / V[4]**2)
                c65_ratio = (I_obs[5] / I_obs[4]) * (I_expec[4] / I_expec[5]) * (V[5]**2 / V[4]**2)
                c5 = 1. / (c15_ratio + c25_ratio + 1. + c35_ratio + c45_ratio + c65_ratio)
                c6 = 1. - c1 - c2 - c3 - c4 - c5
                m1 = (dens[0] * c1) / ((dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3) + (dens[3] * c4) + (dens[4] * c5) + dens[5] * c6)
                m2 = (dens[1] * c2) / ((dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3) + (dens[3] * c4) + (dens[4] * c5) + dens[5] * c6)
                m3 = (dens[2] * c3) / ((dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3) + (dens[3] * c4) + (dens[4] * c5) + dens[5] * c6)
                m4 = (dens[3] * c4) / ((dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3) + (dens[3] * c4) + (dens[4] * c5) + dens[5] * c6)
                m5 = (dens[4] * c5) / ((dens[0] * c1) + (dens[1] * c2) + (dens[2] * c3) + (dens[3] * c4) + (dens[4] * c5) + dens[5] * c6)
                m6 = 1. - m1 - m2 - m3 - m4 - m5
                return [m1, m2, m3, m4, m5, m6]
