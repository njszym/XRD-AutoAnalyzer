import pymatgen as mg
import numpy as np
import random
import math
import os
from pymatgen.analysis.diffraction import xrd

class StrainGen(object):
    """
    Class used to apply stochastic, symmetry-preserving sets of
    strain to a pymatgen structure object.
    """

    def __init__(self, struc, max_strain=0.04):
        """
        Args:
            struc: pymatgen structure object
            max_strain: maximum allowed change in the magnitude
                of the strain tensor components
        """
        self.calculator = xrd.XRDCalculator()
        self.struc = struc
        self.max_strain = max_strain

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
    def diag_range(self):
        max_strain = self.max_strain
        return np.linspace(1-max_strain, 1+max_strain, 1000)

    @property
    def off_diag_range(self):
        max_strain = self.max_strain
        return np.linspace(0-max_strain, 0+max_strain, 1000)

    @property
    def sg_class(self):
        sg = self.sg
        if sg in list(range(195, 231)):
            return 'cubic'
        elif sg in list(range(16, 76)):
            return 'orthorhombic'
        elif sg in list(range(3, 16)):
            return 'monoclinic'
        elif sg in list(range(1, 3)):
            return 'triclinic'
        elif sg in list(range(76, 195)):
            if sg in list(range(75, 83)) + list(range(143, 149)) + list(range(168, 175)):
                return 'low-sym hexagonal/tetragonal'
            else:
                return 'high-sym hexagonal/tetragonal'

    @property
    def strain_tensor(self):
        diag_range = self.diag_range
        off_diag_range = self.off_diag_range
        s11, s22, s33 = [random.choice(diag_range) for v in range(3)]
        s12, s13, s21, s23, s31, s32 = [random.choice(off_diag_range) for v in range(6)]
        sg_class = self.sg_class

        if sg_class in ['cubic', 'orthorhombic', 'monoclinic', 'high-sym hexagonal/tetragonal']:
            v1 = [s11, 0, 0]
        elif sg_class == 'low-sym hexagonal/tetragonal':
            v1 = [s11, s12, 0]
        elif sg_class == 'triclinic':
            v1 = [s11, s12, s13]

        if sg_class in ['cubic', 'high-sym hexagonal/tetragonal']:
            v2 = [0, s11, 0]
        elif sg_class == 'orthorhombic':
            v2 = [0, s22, 0]
        elif sg_class == 'monoclinic':
            v2 = [0, s22, s23]
        elif sg_class == 'low-sym hexagonal/tetragonal':
            v2 = [-s12, s22, 0]
        elif sg_class == 'triclinic':
            v2 = [s21, s22, s23]

        if sg_class == 'cubic':
            v3 = [0, 0, s11]
        elif sg_class == 'high-sym hexagonal/tetragonal':
            v3 = [0, 0, s33]
        elif sg_class == 'orthorhombic':
            v3 = [0, 0, s33]
        elif sg_class == 'monoclinic':
            v3 = [0, s23, s33]
        elif sg_class == 'low-sym hexagonal/tetragonal':
            v3 = [0, 0, s33]
        elif sg_class == 'triclinic':
            v3 = [s31, s32, s33]

        return np.array([v1, v2, v3])

    @property
    def strained_matrix(self):
        return np.matmul(self.matrix, self.strain_tensor)

    @property
    def strained_lattice(self):
        return mg.Lattice(self.strained_matrix)

    @property
    def strained_struc(self):
        new_struc = self.struc.copy()
        new_struc.lattice = self.strained_lattice
        return new_struc

    @property
    def strained_spectrum(self):
        struc = self.strained_struc
        pattern = self.calculator.get_pattern(struc, two_theta_range=(0,80))
        angles, intensities = pattern.x, pattern.y

        x = np.linspace(10, 80, 4501)
        y = []

        for val in x:
            ysum = 0
            for (ang, pk) in zip(angles, intensities):
                if np.isclose(ang, val, atol=0.05):
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


def main(struc, num_strains):

    strain_generator = StrainGen(struc)

    strained_patterns = [strain_generator.strained_spectrum for i in range(num_strains)]

    return strained_patterns
