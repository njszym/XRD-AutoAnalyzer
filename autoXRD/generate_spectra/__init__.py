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

    ## Tabulate diffraction data and space groups
    pattern = calculator.get_pattern(struct)
    angles = pattern.x
    peaks = pattern.y
    if stick_pattern:
        peaks = 100*np.array(peaks)/max(peaks)
        return angles, peaks
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

def augment(phase):

    struct, filename = phase[0], phase[1]
    patterns = []
    varied_structs = sample_strains(struct, 50)
    for each_struct in varied_structs:
        patterns.append(calc_XRD(each_struct))
    for size in np.linspace(1, 100, 50):
        patterns.append(shrink_domain(struct, size))
    for texture_magnitude in np.linspace(0.05, 0.6, 50):
        patterns.append(apply_texture(struct, texture_magnitude))
    grouped_xrd.append(patterns)
    grouped_filenames.append(filename)


def get_spectra(reference_folder):

    phases = []
    for filename in sorted(os.listdir(reference_folder)): ## Sort this so we can keep a consistent comparison later on
        phases.append([mg.Structure.from_file('%s/%s' % (reference_folder, filename)), filename]) ## Keep track of filename associated with each structure

    with Manager() as manager:

        pool = Pool(num_cpu)
        grouped_xrd = manager.list()
        grouped_filenames = manager.list()

        pool.map(augment, phases)
        zipped_info = list(zip(grouped_xrd, grouped_filenames))
        sorted_info = sorted(zipped_info, key=lambda x: x[1])
        sorted_xrd = [group[0] for group in sorted_info]

        return np.array(sorted_xrd)
