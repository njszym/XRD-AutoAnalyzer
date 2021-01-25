from autoXRD.pattern_augmentation import *
import pymatgen as mg
import os
import random
from pymatgen.analysis.diffraction import xrd
import numpy as np
import multiprocessing
from multiprocessing import Pool, Manager


calculator = xrd.XRDCalculator()
num_cpu = multiprocessing.cpu_count()

def calc_XRD(struct, stick_pattern=False):
    """
    Calculates the XRD spectrum of a given structure.

    Args:
        struct: pymatgen Structure object
        stick_pattern: if True, return the diffraction angles
        and peaks as discrete lists. Otherwise, a continuous
        spectrum will be calculated with 4,501 values..
    Returns:
        Returns a continuous or discrete XRD spectrum.
    """

    ## Calculate line profile
    pattern = calculator.get_pattern(struct)
    angles = pattern.x
    peaks = pattern.y

    ## If stick_pattern is True, return line profile
    if stick_pattern:
        peaks = 100*np.array(peaks)/max(peaks)
        return angles, peaks

    ## Otherwise, a continuous spectrum will be formed
    x = np.linspace(10, 80, 4501)
    y = []
    for val in x:
        ysum = 0
        for (ang, pk) in zip(angles, peaks):
            if np.isclose(ang, val, atol=0.05):
                ysum += pk
        y.append(ysum)
    conv = []
    for (ang, int) in zip(x, y):
        if int != 0:
            gauss = [int*np.exp((-(val - ang)**2)/0.015) for val in x]
            conv.append(gauss)
    mixed_data = zip(*conv)
    all_I = []
    for values in mixed_data:
        noise = random.choice(np.linspace(-0.75, 0.75, 1000))
        all_I.append(sum(values) + noise)

    ## Normalize intensities from 0 to 100
    shifted_vals = np.array(all_I) - min(all_I)
    scaled_vals = 100*np.array(shifted_vals)/max(shifted_vals)

    ## Re-shape for keras
    all_I = [[val] for val in scaled_vals]
    return all_I

def augment(phase):
    """
    For a given phase, produce a list of augmented XRD spectra

    Args:
        phase: a tuple or list containing the pymatgen structure object
        and filename of that structure respectively.
    Returns:
        patterns: augmented XRD spectra
        filename: filename of the reference phase
    """

    struct, filename = phase[0], phase[1]
    patterns = []
    varied_structs = sample_strains(struct, 50)
    for each_struct in varied_structs: ## Strained structures (peak shifts)
        patterns.append(calc_XRD(each_struct))
    for size in np.linspace(1, 100, 50): ## Small domain size (peak broadening)
        patterns.append(shrink_domain(struct, size))
    for texture_magnitude in np.linspace(0.05, 0.6, 50): ## Texture (peak intensity change)
        patterns.append(apply_texture(struct, texture_magnitude))
    return (patterns, filename)

def get_spectra(reference_folder):
    """
    Get all spectra from a reference phase in a parallel manner

    Args:
        reference_folder: path to CIFs used as references
    Returns:
        Augmented spectra grouped by their reference phase
    """

    ## Tabulate reference phases
    phases = []
    for filename in sorted(os.listdir(reference_folder)):
        phases.append([mg.Structure.from_file('%s/%s' % (reference_folder, filename)), filename])

    with Manager() as manager:

        ## Calculate augmented spectra
        pool = Pool(num_cpu)
        grouped_xrd = pool.map(augment, phases)
        sorted_xrd = sorted(grouped_xrd, key=lambda x: x[1]) ## Sort by filename
        sorted_spectra = [group[0] for group in sorted_xrd]

        return np.array(sorted_spectra)


