import pymatgen as mg
from pymatgen.analysis.diffraction import xrd
import math
import random
import numpy as np


class TextureGen(object):
    """
    Class used to simulate diffraction patterns with
        peak intensities scale according to texture along
        stochastically chosen crystallographic directions.
    """

    def __init__(self, struc, max_texture=0.6):
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

    @property
    def pattern(self):
        struc = self.struc
        return self.calculator.get_pattern(struc, two_theta_range=(0,80))

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

    @property
    def textured_spectrum(self):
        angles = self.angles
        textured_intensities = self.textured_intensities

        x = np.linspace(10, 80, 4501)
        y = []

        step_size = (80. - 10.)/4501.
        half_step = step_size/2.0
        for val in x:
            ysum = 0
            for (ang, pk) in zip(angles, textured_intensities):
                if np.isclose(ang, val, atol=half_step):
                    ysum += pk
            y.append(ysum)

        conv = []
        for (ang, int) in zip(x, y):
            if int != 0:
                gauss = [int*np.exp((-(val - ang)**2)/0.15) for val in x]
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

def main(struc, num_textured, max_texture=0.6):

    texture_generator = TextureGen(struc, max_texture)

    textured_patterns = [texture_generator.textured_spectrum for i in range(num_textured)]

    return textured_patterns
