from autoXRD.pattern_augmentation import *
from autoXRD.multiphase_tools import *
import pymatgen as mg
import os
import random
from pymatgen.analysis.diffraction import xrd
import numpy as np


def calc_XRD_patterns(struct):

    ## Tabulate diffraction data and space groups
    pattern = calculator.get_pattern(struct)
    angles = pattern.x
    peaks = pattern.y
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

    yvals = all_I
    shifted_vals = np.array(yvals) - min(yvals)
    scaled_vals = 100*np.array(shifted_vals)/max(shifted_vals)

    all_I = [[val] for val in scaled_vals] ## Shape necessary for keras
    return all_I

def generate_spectra(reference_folder):

    calculator = xrd.XRDCalculator()

    structures = []
    os.chdir(reference_folder)
    for phase in sorted(os.listdir('.')):
        structures.append(mg.Structure.from_file(phase))
    os.chdir('../')

    y_vals = []
    for struct in structures:
        grouped_patterns = []
       	varied_structs = sample_strains(struct, 50)
        for each_struct in varied_structs:
            grouped_patterns.append(calc_XRD_patterns(each_struct))
        for size in np.linspace(1, 100, 50):
            grouped_patterns.append(shrink_domain(struct, size))
        for texture_magnitude in np.linspace(0.05, 0.6, 50):
            grouped_patterns.append(apply_texture(struct, texture_magnitude))
        y_vals.append(grouped_patterns)

    return np.array(y_vals)
