import pymatgen as mg
from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
import random
import math
import numpy as np


class BroadGen(object):
    """
    Class used to simulate xrd spectra with broad peaks
        that are associated with small domain size
    """

    def __init__(self, struc, min_domain_size=1, max_domain_size=100, min_angle=10.0, max_angle=80.0):
        """
        Args:
            struc: structure to simulate augmented xrd spectra from
            min_domain_size: smallest domain size (in nm) to be sampled,
                leading to the broadest peaks
            max_domain_size: largest domain size (in nm) to be sampled,
                leading to the most narrow peaks
        """
        self.calculator = xrd.XRDCalculator()
        self.struc = struc
        self.possible_domains = np.linspace(min_domain_size, max_domain_size, 100)
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.pattern = self.calculator.get_pattern(struc, two_theta_range=(self.min_angle, self.max_angle))

    @property
    def angles(self):
        return self.pattern.x

    @property
    def intensities(self):
        return self.pattern.y

    @property
    def hkl_list(self):
        return [v[0]['hkl'] for v in self.pattern.hkls]

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
    def broadened_spectrum(self):

        angles = self.angles
        intensities = self.intensities

        steps = np.linspace(self.min_angle, self.max_angle, 4501)

        signals = np.zeros([len(angles), steps.shape[0]])

        for i, ang in enumerate(angles):
            # Map angle to closest datapoint step
            idx = np.argmin(np.abs(ang-steps))
            signals[i,idx] = intensities[i]

        # Convolute every row with unique kernel
        # Iterate over rows; not vectorizable, changing kernel for every row
        domain_size = random.choice(self.possible_domains)
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


def main(struc, num_broadened, min_domain_size, max_domain_size, min_angle=10.0, max_angle=80.0):

    broad_generator = BroadGen(struc, min_domain_size, max_domain_size, min_angle, max_angle)

    broadened_patterns = [broad_generator.broadened_spectrum for i in range(num_broadened)]

    return broadened_patterns
