from pymatgen.core.periodic_table import Element
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
import math
import os


class QuantAnalysis(object):
    """
    Class used to plot and compare:
    (i) measured xrd spectra
    (ii) line profiles of identified phases
    """

    def __init__(self, spectra_dir, spectrum_fname, predicted_phases, min_angle=10.0, max_angle=80.0, wavelength='CuKa', reference_dir='References'):
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
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.wavelen = wavelength

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

        ## Convert indices to two-theta
        span = self.max_angle - self.min_angle
        peak_inds = (4501./span)*(np.array(angles) - self.min_angle)
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

def main(spectra_directory, spectrum_fname, predicted_phases, min_angle=10.0, max_angle=80.0, wavelength='CuKa'):

        analyzer = QuantAnalysis(spectra_directory, spectrum_fname, predicted_phases, min_angle, max_angle, wavelength)

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

