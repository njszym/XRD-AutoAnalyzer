import pymatgen as mg
from pymatgen.analysis.diffraction import xrd
import random
import math
import numpy as np


class BroadGen(object):
    """
    Class used to simulate xrd spectra with broad peaks
        that are associated with small domain size
    """

    def __init__(self, struc, min_domain_size=1, max_domain_size=100):
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
        self.pattern = self.calculator.get_pattern(struc, two_theta_range=(0,80))
        self.possible_domains = np.linspace(min_domain_size, max_domain_size, 100)

    @property
    def angles(self):
        return self.pattern.x

    @property
    def intensities(self):
        return self.pattern.y

    @property
    def hkl_list(self):
        return [v[0]['hkl'] for v in self.pattern.hkls]

    @property
    def broadened_spectrum(self):
        angles = self.angles
        intensities = self.intensities
        tau = random.choice(self.possible_domains)

        x = np.linspace(10, 80, 4501)
        y = []

        step_size = (80. - 10.)/4501.
        half_step = step_size/2.0
        for val in x:
            ysum = 0
            for (ang, pk) in zip(angles, intensities):
                if np.isclose(ang, val, atol=half_step):
                    ysum += pk
            y.append(ysum)

        conv = []
        for (ang, int) in zip(x, y):
            if int != 0:
                ## Calculate FWHM based on the Scherrer eqtn
                K = 0.9 ## shape factor
                wavelength = 0.15406 ## Cu K-alpha in nm
                theta = math.radians(ang/2.) ## Bragg angle in radians
                beta = (K / wavelength) * (math.cos(theta) / tau)

                ## Convert FWHM to std deviation of gaussian
                std_dev = beta/2.35482

                ## Convlution of gaussian
                gauss = [int*np.exp((-(val - ang)**2)/std_dev) for val in x]
                conv.append(gauss)

        mixed_data = zip(*conv)
        all_I = []
        for values in mixed_data:
            noise = random.choice(np.linspace(-0.75, 0.75, 1000))
            all_I.append(sum(values) + noise)

        shifted_vals = np.array(all_I) - min(all_I)
        scaled_vals = 100*np.array(shifted_vals)/max(shifted_vals)
        all_I = [[val] for val in scaled_vals]

        return all_I

def main(struc, num_broadened, min_domain_size, max_domain_size):

    broad_generator = BroadGen(struc, min_domain_size, max_domain_size)

    broadened_patterns = [broad_generator.broadened_spectrum for i in range(num_broadened)]

    return broadened_patterns
