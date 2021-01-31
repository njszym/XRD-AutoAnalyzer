import pymatgen as mg
import random
from pymatgen.analysis.diffraction import xrd
import numpy as np


def calculate_xrd(struc, stick_pattern=False):
    """
    Calculates the ideal XRD spectrum of a given structure.

    Args:
        struc: pymatgen Structure object
        stick_pattern: if True, return the diffraction angles
            and peaks as discrete lists. Otherwise, a continuous
            spectrum will be calculated with 4,501 values.
    Returns:
        Returns a continuous or discrete XRD spectrum
    """

    calculator = xrd.XRDCalculator()

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
