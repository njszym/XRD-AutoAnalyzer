import pymatgen as mg
from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
import math
import random
import numpy as np


class TextureGen(object):
    """
    Class used to simulate diffraction patterns with
        peak intensities scale according to texture along
        stochastically chosen crystallographic directions.
    """

    def __init__(self, struc, max_texture=0.6, min_angle=10.0, max_angle=80.0):
        """
        Args:
            struc: pymatgen structure object from which xrd
                spectra are simulated
            max_texture: maximum strength of texture applied.
                For example, max_texture=0.6 implies peaks will be
                scaled by as much as +/- 60% of their original
                intensities.
        """
        self.calculator = xrd.XRDCalculator()
        self.struc = struc
        self.max_texture = max_texture
        self.min_angle = min_angle
        self.max_angle = max_angle

    @property
    def pattern(self):
        struc = self.struc
        return self.calculator.get_pattern(struc, two_theta_range=(self.min_angle, self.max_angle))

    @property
    def angles(self):
        return self.pattern.x

    @property
    def intensities(self):
        return self.pattern.y

    @property
    def hkl_list(self):
        return [v[0]['hkl'] for v in self.pattern.hkls]

    def map_interval(self, v):
        """
        Maps a value (v) from the interval [0, 1] to
            a new interval [1 - max_texture, 1]
        """

        bound = 1.0 - self.max_texture
        return bound + ( ( (1.0 - bound) / (1.0 - 0.0) ) * (v - 0.0) )

    @property
    def textured_intensities(self):
        hkls, intensities = self.hkl_list, self.intensities
        scaled_intensities = []

        # Four Miller indicies in hexagonal systems
        if self.struc.lattice.is_hexagonal() == True:
            check = 0.0
            while check == 0.0:
                preferred_direction = [random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1])]
                check = np.dot(np.array(preferred_direction), np.array(preferred_direction)) # Ensure 0-vector is not used

        # Three indicies are used otherwise
        else:
            check = 0.0
            while check == 0.0:
                preferred_direction = [random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1])]
                check = np.dot(np.array(preferred_direction), np.array(preferred_direction)) # Make sure we don't have 0-vector

        for (hkl, peak) in zip(hkls, intensities):
            norm_1 = math.sqrt(np.dot(np.array(hkl), np.array(hkl)))
            norm_2 = math.sqrt(np.dot(np.array(preferred_direction), np.array(preferred_direction)))
            total_norm = norm_1 * norm_2
            texture_factor = abs(np.dot(np.array(hkl), np.array(preferred_direction)) / total_norm)
            texture_factor = self.map_interval(texture_factor)
            scaled_intensities.append(peak*texture_factor)

        return scaled_intensities

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

    @property
    def textured_spectrum(self):

        angles = self.angles
        intensities = self.textured_intensities

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

        noise = np.random.normal(0, 0.25, 4501)
        noisy_signal = norm_signal + noise

        # Formatted for CNN
        form_signal = [[val] for val in noisy_signal]

        return form_signal

def main(struc, num_textured, max_texture=0.6, min_angle=10.0, max_angle=80.0):

    texture_generator = TextureGen(struc, max_texture, min_angle, max_angle)

    textured_patterns = [texture_generator.textured_spectrum for i in range(num_textured)]

    return textured_patterns
