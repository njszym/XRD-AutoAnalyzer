import matplotlib.pyplot as plt
from scipy.signal import find_peaks, filtfilt
from dtw import dtw, warp
import random
import pymatgen as mg
from pymatgen.analysis.diffraction import xrd
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
from scipy import interpolate as ip
from pymatgen.core import Structure
import numpy as np
import os


class SpectrumPlotter(object):
    """
    Class used to plot and compare:
    (i) measured xrd spectra
    (ii) line profiles of identified phases
    """

    def __init__(self, spectra_dir, spectrum_fname, predicted_phases, reference_dir='References'):
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
        self.ref_dir = reference_dir
        self.calculator = xrd.XRDCalculator()

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

        ## Fit to 4,501 values as to be compatible with CNN
        f = ip.CubicSpline(x, y)
        xs = np.linspace(10, 80, 4501)
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

    @property
    def scaled_patterns(self):
        """
        Get line profiles of predicted phases that are scaled
        to match with peaks in the measured spectrum
        """

        measured_spectrum = self.formatted_spectrum
        pred_phases = self.pred_phases

        angle_sets, intensity_sets = [], []
        for phase in pred_phases:
            angles, intensities = self.get_stick_pattern(phase)
            scaling_constant = self.scale_line_profile(angles, intensities)
            scaled_intensities = scaling_constant*np.array(intensities)
            angle_sets.append(angles)
            intensity_sets.append(scaled_intensities)

        return angle_sets, intensity_sets

    def get_stick_pattern(self, ref_phase):
        """
        Returns:
            all_I: list of intensities as a function of two-theta
        """

        struct = Structure.from_file('%s/%s' % (self.ref_dir, ref_phase))

        pattern = self.calculator.get_pattern(struct, two_theta_range=(10,80))
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

        ## Convert indices to two-theta
        peak_inds = (4501./70.)*(np.array(angles) - 10.)
        peak_inds = [int(i) for i in peak_inds]
        y = []
        qi = 0
        for x in range(4501):
            if x in peak_inds:
                y.append(intensities[qi])
                qi += 1
            else:
                y.append(0.0)

        # Get peak indices
        orig_y = self.formatted_spectrum
        orig_peaks = find_peaks(orig_y, height=5)[0]
        pred_peaks = find_peaks(y, height=5)[0]

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
        for scale_spectrum in np.linspace(1, 0.05, 101):
            check = scale_spectrum*np.array(y)
            good_peaks = 0
            for (a, b) in zip(matched_orig_peaks, matched_pred_peaks):
                A_magnitude = orig_y[a]
                B_magnitude = check[b]
                if abs((A_magnitude - B_magnitude)/A_magnitude) < 0.1:
                    good_peaks += 1
            num_match.append(good_peaks)

        best_scale = np.linspace(1.0, 0.05, 101)[np.argmax(num_match)]

        return best_scale


def main(spectra_directory, spectrum_fname, predicted_phases):

        spec_plot = SpectrumPlotter(spectra_directory, spectrum_fname, predicted_phases)

        x = np.linspace(10, 80, 4501)
        measured_spectrum = spec_plot.formatted_spectrum
        angle_sets, intensity_sets = spec_plot.scaled_patterns

        plt.figure()

        plt.plot(x, measured_spectrum, 'b-', label='Measured: %s' % spectrum_fname)

        phase_names = [fname[:-4] for fname in predicted_phases] # remove .cif
        color_list = ['g', 'r', 'm', 'k', 'c']
        i = 0
        for (angles, intensities, phase) in zip(angle_sets, intensity_sets, phase_names):
            for (x, y) in zip(angles, intensities):
                plt.vlines(x, 0, y, color=color_list[i], linewidth=2.5)
            plt.plot([0], [0], color=color_list[i], label='Predicted: %s' % phase)
            i += 1

        plt.xlim(10, 80)
        plt.ylim(0, 105)
        plt.legend(prop={'size': 16})
        plt.xlabel(r'2$\Theta$', fontsize=16, labelpad=12)
        plt.ylabel('Intensity', fontsize=16, labelpad=12)
        plt.show()
