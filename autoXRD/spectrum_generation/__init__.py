from autoXRD.spectrum_generation import ideal_pattern, peak_shifts, intensity_changes, peak_broadening
import pymatgen as mg
import numpy as np
import os
import multiprocessing
from multiprocessing import Pool, Manager


class SpectraGenerator(object):
    """
    Class used to generate augmented xrd spectra
    for all reference phases
    """

    def __init__(self, reference_dir):
        """
        Args:
            reference_dir: path to directory containing
                CIFs associated with the reference phases
        """
        self.num_cpu = multiprocessing.cpu_count()
        self.ref_dir = reference_dir

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

        patterns += peak_shifts.main(struc, 50)
        patterns += peak_broadening.main(struc, 50)
        patterns += intensity_changes.main(struc, 50)

        return (patterns, filename)

    @property
    def augmented_spectra(self):

        phases = []
        for filename in sorted(os.listdir(self.ref_dir)):
            phases.append([mg.Structure.from_file('%s/%s' % (self.ref_dir, filename)), filename])

        with Manager() as manager:

            pool = Pool(self.num_cpu)
            grouped_xrd = pool.map(self.augment, phases)
            sorted_xrd = sorted(grouped_xrd, key=lambda x: x[1]) ## Sort by filename
            sorted_spectra = [group[0] for group in sorted_xrd]

            return np.array(sorted_spectra)


