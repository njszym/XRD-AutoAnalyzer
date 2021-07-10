import pymatgen as mg
import numpy as np
import random
import math
import os
from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
from pyxtal import pyxtal

class StrainGen(object):
    """
    Class used to apply stochastic, symmetry-preserving sets of
    strain to a pymatgen structure object.
    """

    def __init__(self, struc, max_strain=0.04, min_angle=10.0, max_angle=80.0):
        """
        Args:
            struc: pymatgen structure object
            max_strain: maximum allowed change in the magnitude
                of the strain tensor components
        """
        self.calculator = xrd.XRDCalculator()
        self.struc = struc
        self.max_strain = max_strain
        self.strain_range = np.linspace(0.0, max_strain, 100)
        self.min_angle = min_angle
        self.max_angle = max_angle

    @property
    def sg(self):
        return self.struc.get_space_group_info()[1]

    @property
    def conv_struc(self):
        sga = mg.symmetry.analyzer.SpacegroupAnalyzer(struc)
        return sga.get_conventional_standard_structure()

    @property
    def lattice(self):
        return self.struc.lattice

    @property
    def matrix(self):
        return self.struc.lattice.matrix

    @property
    def strained_struc(self):
        ref_struc = self.struc.copy()
        xtal_struc = pyxtal()
        xtal_struc.from_seed(ref_struc)
        current_strain = random.choice(self.strain_range)
        xtal_struc.apply_perturbation(d_lat=current_strain, d_coor=0.0)
        pmg_struc = xtal_struc.to_pymatgen()
        return pmg_struc

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
    def strained_spectrum(self):
        struc = self.strained_struc
        pattern = self.calculator.get_pattern(struc, two_theta_range=(self.min_angle, self.max_angle))
        angles, intensities = pattern.x, pattern.y

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


def main(struc, num_strains, max_strain, min_angle=10.0, max_angle=80.0):

    strain_generator = StrainGen(struc, max_strain, min_angle, max_angle)

    strained_patterns = [strain_generator.strained_spectrum for i in range(num_strains)]

    return strained_patterns
