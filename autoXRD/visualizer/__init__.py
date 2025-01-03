import matplotlib.pyplot as plt
from scipy.signal import find_peaks, filtfilt, resample
import matplotlib
import random
import pymatgen as mg
from scipy import signal
from pymatgen.analysis.diffraction import xrd
from autoXRD.dara import do_refinement_no_saving, get_phase_weights, get_structure
from pymatgen.io.cif import CifWriter
from skimage import restoration
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate as ip
from pymatgen.core import Structure
from pyts import metrics
from pathlib import Path
import numpy as np
import warnings
import math
import os


class SpectrumPlotter(object):
    """
    Class used to plot and compare:
    (i) measured xrd spectra
    (ii) line profiles of identified phases
    """

    def __init__(self, spectra_dir, spectrum_fname, predicted_phases, scale_factors,
        min_angle=10.0, max_angle=80.0, wavelength='CuKa', raw=False, reference_dir='References'):
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
        self.raw = raw

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

        ## Fit to 4,501 values as to be compatible with CNN
        f = ip.CubicSpline(x, y)
        xs = np.linspace(self.min_angle, self.max_angle, 4501)
        ys = f(xs)

        if not self.raw:

            ## Smooth out noise
            ys = self.smooth_spectrum(ys)

            ## Normalize from 0 to 255
            ys = np.array(ys) - min(ys)
            ys = list(255*np.array(ys)/max(ys))

            # Subtract background
            background = restoration.rolling_ball(ys, radius=80)
            ys = np.array(ys) - np.array(background)

        ## Normalize from 0 to 100
        ys = np.array(ys) - min(ys)
        ys = list(100*np.array(ys)/max(ys))

        return ys

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

def XRDtoPDF(xrd, min_angle, max_angle):

    thetas = np.linspace(min_angle/2.0, max_angle/2.0, 4501)
    Q = np.array([4*math.pi*math.sin(math.radians(theta))/1.5406 for theta in thetas])
    S = np.array(xrd).flatten()

    pdf = []
    R = np.linspace(1, 40, 1000) # Only 1000 used to reduce compute time
    integrand = Q * S * np.sin(Q * R[:, np.newaxis])

    pdf = (2*np.trapz(integrand, Q) / math.pi)

    return R, pdf


def scale_values(values, new_min, new_max):
    """
    Scale a list of values to a new given range [new_min, new_max].
    """

    # Find the current minimum and maximum values of the list
    old_min = min(values)
    old_max = max(values)

    # Scale each value to the new range
    scaled_values = [
        ((new_max - new_min) * (value - old_min) / (old_max - old_min)) + new_min
        for value in values
    ]

    return scaled_values


def main(spectra_directory, spectrum_fname, predicted_phases, scale_factors, reduced_spectrum, min_angle=10.0, max_angle=80.0,
    wavelength='CuKa', save=False, show_reduced=False, inc_pdf=False, plot_both=False, raw=False, rietveld=True, refined_phases_dir=None):

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

            # Save refined CIFs if directory is provided
            if refined_phases_dir:
                weight_dict = get_phase_weights(result)
                weights = []
                for ph in predicted_phases:
                    ph_name = ph[:-4]
                    os.makedirs(f"{refined_phases_dir}/{spectrum_fname}", exist_ok=True)
                    weights.append(weight_dict[ph_name])
                    structure = get_structure(result["lst_data"]["phases_results"][ph_name])
                    if structure is not None:
                        cif_writer = CifWriter(structure)
                        cif_writer.write_file(f"{refined_phases_dir}/{spectrum_fname}/{ph_name}.cif")

            x_obs, y_obs = result["plot_data"]["x"], result["plot_data"]["y_obs"]

            plt.figure()

            plt.title('Filename: %s' % spectrum_fname, y=1.01, fontsize=16)

            plt.plot(x_obs, y_obs, 'b-', label='Observed')

            phase_names = [fname[:-4] for fname in predicted_phases] # remove .cif
            color_list = ['g', 'r', 'm', 'k', 'c']
            i = 0
            for ph in phase_names:
                ph_zero = np.array(result["plot_data"]["structs"][ph])
                bckgrd = np.array(result["plot_data"]["y_bkg"])
                plt.plot(x_obs, ph_zero + bckgrd, color=color_list[i], label='Predicted: %s' % ph)
                plt.fill_between(x_obs, ph_zero + bckgrd, color=color_list[i], alpha=0.2)
                i += 1

            legend_labels = ['Filename: %s' % spectrum_fname] + list(phase_names)
            longest_label_len = len(max(legend_labels, key=len))
            plt.legend(prop={'size': int(16.0-longest_label_len*0.1)},loc='upper left')

            magnitude = max(y_obs) - min(y_obs)
            lower_lim = max([0, min(y_obs) - 0.025*magnitude])
            plt.ylim(lower_lim, 1.3*max(y_obs))

            plt.xlim(min_angle, max_angle)
            plt.xlabel(r'2$\Theta$', fontsize=16, labelpad=12)
            plt.ylabel('Intensity', fontsize=16, labelpad=12)

            if save:
                savename = '%s.png' % '.'.join(spectrum_fname.split('.')[:-1])
                plt.tight_layout()
                plt.savefig(savename, dpi=400)
                plt.close()

            else:
                plt.tight_layout()
                plt.show()

        else:

            spec_plot = SpectrumPlotter(spectra_directory, spectrum_fname, predicted_phases, scale_factors, min_angle, max_angle, wavelength, raw)

            x = np.linspace(min_angle, max_angle, 4501)
            measured_spectrum = spec_plot.formatted_spectrum
            angle_sets, intensity_sets = spec_plot.scaled_patterns

            plt.figure()

            plt.plot(x, measured_spectrum, 'b-', label='Measured: %s' % spectrum_fname)

            phase_names = [fname[:-4] for fname in predicted_phases] # remove .cif
            color_list = ['g', 'r', 'm', 'k', 'c']
            i = 0
            for (angles, intensities, phase) in zip(angle_sets, intensity_sets, phase_names):
                for (xv, yv) in zip(angles, intensities):
                    plt.vlines(xv, 0, yv, color=color_list[i], linewidth=2.5)
                plt.plot([0], [0], color=color_list[i], label='Predicted: %s' % phase)
                i += 1

            if show_reduced:
                plt.plot(x, reduced_spectrum, color='orange', linestyle='dashed', label='Reduced spectrum')

            # Variable legend and ylim
            legend_labels = ['Measured: %s' % spectrum_fname] + list(phase_names)
            longest_label_len = len(max(legend_labels, key=len))
            plt.legend(prop={'size': int(16.0-longest_label_len*0.1)},loc='upper left')
            plt.ylim(0, 105+(len(phase_names)+1)*10)

            plt.xlim(min_angle, max_angle)
            plt.xlabel(r'2$\Theta$', fontsize=16, labelpad=12)
            plt.ylabel('Intensity', fontsize=16, labelpad=12)

            if save:
                savename = '%s.png' % '.'.join(spectrum_fname.split('.')[:-1])
                plt.tight_layout()
                plt.savefig(savename, dpi=400)
                plt.close()

            else:
                plt.show()

            plt.close()

            if inc_pdf and plot_both:

                r, measured_pdf = XRDtoPDF(measured_spectrum, min_angle, max_angle)

                plt.figure()

                plt.plot(r, measured_pdf, 'b-', label='Measured: %s' % spectrum_fname)

                phase_names = [fname[:-4] for fname in predicted_phases] # remove .cif
                color_list = ['g', 'r', 'm', 'k', 'c']
                i = 0
                for (angles, intensities, phase) in zip(angle_sets, intensity_sets, phase_names):
                    ys = spec_plot.get_cont_profile(angles, intensities)
                    r, ref_pdf = XRDtoPDF(ys, min_angle, max_angle)
                    plt.plot(r, ref_pdf, color=color_list[i], linestyle='dashed', label='Predicted: %s' % phase)
                    i += 1

                plt.xlim(1, 30)
                plt.legend(prop={'size': 16})
                plt.xlabel(r'r (Å)', fontsize=16, labelpad=12)
                plt.ylabel('G(r)', fontsize=16, labelpad=12)

                if save:
                    savename = '%s_PDF.png' % spectrum_fname.split('.')[0]
                    plt.tight_layout()
                    plt.savefig(savename, dpi=400)
                    plt.close()

                else:
                    plt.show()
