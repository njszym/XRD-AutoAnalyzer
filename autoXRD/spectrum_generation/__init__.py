from autoXRD.spectrum_generation import peak_shifts, intensity_changes, peak_broadening, mixed
import pymatgen as mg
import numpy as np
import os
import multiprocessing
from multiprocessing import Pool, Manager
from pymatgen.core import Structure


class SpectraGenerator(object):
    """
    Class used to generate augmented xrd spectra
    for all reference phases
    """

    def __init__(self, reference_dir, num_spectra=50, max_texture=0.6, min_domain_size=1.0, max_domain_size=100.0, max_strain=0.04, min_angle=10.0, max_angle=80.0, separate=True):
        """
        Args:
            reference_dir: path to directory containing
                CIFs associated with the reference phases
        """
        self.num_cpu = multiprocessing.cpu_count()
        self.ref_dir = reference_dir
        self.num_spectra = num_spectra
        self.max_texture = max_texture
        self.min_domain_size = min_domain_size
        self.max_domain_size = max_domain_size
        self.max_strain = max_strain
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.separate = separate

    def augment(self, phase_info):
        """
        For a given phase, produce a list of augmented XRD spectra.
        By default, 50 spectra are generated per artifact, including
        peak shifts (strain), peak intensity change (texture), and
        peak broadening (small domain size).

        Args:
            phase_info: a list containing the pymatgen structure object
                and filename of that structure respectively.
        Returns:
            patterns: augmented XRD spectra
            filename: filename of the reference phase
        """

        struc, filename = phase_info[0], phase_info[1]
        patterns = []

        if self.separate:
            patterns += peak_shifts.main(struc, self.num_spectra, self.max_strain, self.min_angle, self.max_angle)
            patterns += peak_broadening.main(struc, self.num_spectra, self.min_domain_size, self.max_domain_size, self.min_angle, self.max_angle)
            patterns += intensity_changes.main(struc, self.num_spectra, self.max_texture, self.min_angle, self.max_angle)
        else:
            patterns += mixed.main(struc, self.num_spectra, self.max_strain, self.min_domain_size, self.max_domain_size,  self.max_texture, self.min_angle, self.max_angle)

        return (patterns, filename)

    @property
    def augmented_spectra(self):

        phases = []
        for filename in sorted(os.listdir(self.ref_dir)):
            phases.append([Structure.from_file('%s/%s' % (self.ref_dir, filename)), filename])

        with Manager() as manager:

            pool = Pool(self.num_cpu)
            grouped_xrd = pool.map(self.augment, phases)
            sorted_xrd = sorted(grouped_xrd, key=lambda x: x[1]) ## Sort by filename
            sorted_spectra = [group[0] for group in sorted_xrd]

            return np.array(sorted_spectra)


