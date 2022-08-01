from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
from pymatgen.core import Structure
from pymatgen.core import Lattice
from pyxtal import pyxtal
import pymatgen as mg
import numpy as np
import random
import math
import os


class MixedGen(object):
    """
    Class used to apply stochastic, symmetry-preserving sets of
    strain to a pymatgen structure object.
    """

    def __init__(self, struc, max_shift=0.5, max_strain=0.04, min_domain_size=1, max_domain_size=100, max_texture=0.6, impur_amt=70.0, min_angle=10.0, max_angle=80.0, ref_dir='References'):
        """
        Args:
            struc: pymatgen structure object
            max_strain: maximum allowed change in the magnitude
                of the strain tensor components
        """
        self.calculator = xrd.XRDCalculator()
        self.struc = struc
        self.ref_dir = ref_dir
        self.max_shift = max_shift
        self.max_strain = max_strain
        self.possible_domains = np.linspace(min_domain_size, max_domain_size, 100)
        self.max_texture = max_texture
        self.impur_amt = impur_amt
        self.strain_range = np.linspace(0.0, max_strain, 100)
        self.min_angle = min_angle
        self.max_angle = max_angle

        # Generate a single clean spectrum for each reference phase
        self.saved_patterns = self.clean_specs

    @property
    def clean_specs(self):

        # Iterate through all reference structures
        ref_patterns = []
        for struc in self.ref_strucs:

            pattern = self.calculator.get_pattern(struc, two_theta_range=(self.min_angle, self.max_angle))
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
                signals[i,:] = gaussian_filter1d(row, np.sqrt(std_dev)*1/step_size, mode='constant')

            # Combine signals
            signal = np.sum(signals, axis=0)

            # Normalize signal
            norm_signal = 100 * signal / max(signal)

            ref_patterns.append(norm_signal)

        return ref_patterns


    @property
    def impurity_spectrum(self):
        signal = random.choice(self.saved_patterns)
        return signal

    @property
    def ref_strucs(self):
        current_lat = self.struc.lattice.abc
        all_strucs = []
        for fname in os.listdir(self.ref_dir):
            fpath = '%s/%s' % (self.ref_dir, fname)
            struc = Structure.from_file(fpath)
            # Ensure no duplicate structures
            if False in np.isclose(struc.lattice.abc, current_lat, atol=0.01):
                all_strucs.append(struc)
        return all_strucs

    def pattern(self, struc):
        return self.calculator.get_pattern(struc, two_theta_range=(self.min_angle, self.max_angle))

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
        # Pyxtal not compatible with partial occupancies
        if ref_struc.is_ordered:
            xtal_struc = pyxtal()
            xtal_struc.from_seed(ref_struc)
            current_strain = random.choice(self.strain_range)
            xtal_struc.apply_perturbation(d_lat=current_strain, d_coor=0.0)
            pmg_struc = xtal_struc.to_pymatgen()
            return pmg_struc
        else:
            ref_struc.lattice = self.strained_lattice
            return ref_struc

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
        return Lattice(self.strained_matrix)

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
    def textured_pattern(self):

        struc = self.strained_struc
        pattern = self.pattern(struc)
        hkls = [v[0]['hkl'] for v in pattern.hkls]
        angles = pattern.x
        intensities = pattern.y
        scaled_intensities = []

        # Four Miller indicies in hexagonal systems
        if struc.lattice.is_hexagonal() == True:
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

        return angles, scaled_intensities

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
    def mixed_spectrum(self):

        angles, intensities = self.textured_pattern

        shift_range = np.linspace(-self.max_shift, self.max_shift, 1000)
        shift = random.choice(shift_range)
        angles = np.array(angles) + shift

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
        signal = 100 * signal / max(signal)

        # Add impurity signal
        impurity_signal = self.impurity_spectrum
        impurity_magnitude = random.choice(np.linspace(0, self.impur_amt, 100))
        impurity_signal = impurity_magnitude * impurity_signal / max(impurity_signal)
        signal += impurity_signal

        # Renormalize signal
        norm_signal = 100 * signal / max(signal)

        noise = np.random.normal(0, 0.25, 4501)
        noisy_signal = norm_signal + noise

        # Formatted for CNN
        form_signal = [[val] for val in noisy_signal]

        return form_signal


def main(struc, num_specs, max_shift, max_strain, min_domain_size, max_domain_size, max_texture, impur_amt, min_angle=10.0, max_angle=80.0):

    mixed_generator = MixedGen(struc, max_shift, max_strain, min_domain_size, max_domain_size,  max_texture, impur_amt, min_angle, max_angle)

    mixed_patterns = [mixed_generator.mixed_spectrum for i in range(num_specs)]

    return mixed_patterns
