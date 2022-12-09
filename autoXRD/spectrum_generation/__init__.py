from autoXRD.spectrum_generation import strain_shifts, uniform_shifts, intensity_changes, peak_broadening, impurity_peaks, mixed
from multiprocessing import Pool, Manager
from pymatgen.core import Structure
from scipy import signal
import multiprocessing
import pymatgen as mg
import numpy as np
import math
import os


class SpectraGenerator(object):
    """
    Class used to generate augmented xrd spectra
    for all reference phases
    """

    def __init__(self, reference_dir, num_spectra=50, max_texture=0.6, min_domain_size=1.0, max_domain_size=100.0, max_strain=0.04, max_shift=0.25, impur_amt=70.0, min_angle=10.0, max_angle=80.0, separate=True, is_pdf=False):
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
        self.max_shift = max_shift
        self.impur_amt = impur_amt
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.separate = separate
        self.is_pdf = is_pdf

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
        patterns, pdf_specs = [], []

        if self.separate:
            patterns += strain_shifts.main(struc, self.num_spectra, self.max_strain, self.min_angle, self.max_angle)
            patterns += uniform_shifts.main(struc, self.num_spectra, self.max_shift, self.min_angle, self.max_angle)
            patterns += peak_broadening.main(struc, self.num_spectra, self.min_domain_size, self.max_domain_size, self.min_angle, self.max_angle)
            patterns += intensity_changes.main(struc, self.num_spectra, self.max_texture, self.min_angle, self.max_angle)
            patterns += impurity_peaks.main(struc, self.num_spectra, self.impur_amt, self.min_angle, self.max_angle)
        else:
            patterns += mixed.main(struc, 5*self.num_spectra, self.max_shift, self.max_strain, self.min_domain_size, self.max_domain_size,  self.max_texture, self.impur_amt, self.min_angle, self.max_angle)

        if self.is_pdf:
            for xrd in patterns:
                xrd = np.array(xrd).flatten()
                pdf = self.XRDtoPDF(xrd, self.min_angle, self.max_angle)
                pdf = [[v] for v in pdf]
                pdf_specs.append(pdf)
            return (pdf_specs, filename)

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

    def XRDtoPDF(self, xrd, min_angle, max_angle):

        thetas = np.linspace(min_angle/2.0, max_angle/2.0, 4501)
        Q = np.array([4*math.pi*math.sin(math.radians(theta))/1.5406 for theta in thetas])
        S = np.array(xrd).flatten()

        pdf = []
        R = np.linspace(1, 40, 1000) # Only 1000 used to reduce compute time
        integrand = Q * S * np.sin(Q * R[:, np.newaxis])

        pdf = (2*np.trapz(integrand, Q) / math.pi)
        pdf = list(signal.resample(pdf, 4501))

        return pdf

