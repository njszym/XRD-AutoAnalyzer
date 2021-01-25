import pymatgen as mg
import numpy as np
import random
import math
import os
from pymatgen.analysis.diffraction import xrd


def sample_strains(struct, num_strains):
    """
    Produce stochastically strained structures that preserve symmetry.

    Args:
        struct: pymatgen structure object that is to be strained
        num_strains: how many strained structures to produce
    Returns:
        varied_structs: a list of strained structures
    """

    ## Get space group (strain must preserve symmetry)
    sg = struct.get_space_group_info()[1]

    ## Strain up to 4% in each component
    space_1 = np.linspace(0.96, 1.04, 1000)
    space_0 = np.linspace(-0.04, 0.04, 1000)

    ## Get conventional structure
    sga = mg.symmetry.analyzer.SpacegroupAnalyzer(struct)
    struct = sga.get_conventional_standard_structure()
    struct_dict = struct.as_dict()
    lat = struct.lattice
    mat = lat.matrix

    varied_structs = []
    if sg in list(range(195, 231)): ## Cubic
        for index in range(num_strains):
            temp_struct = mg.Structure.from_dict(struct_dict)
            orig_vol = temp_struct.volume
            a = random.choice(space_1)
            v1 = [a, 0, 0]
            v2 = [0, a, 0]
            v3 = [0, 0, a]
            strain_tensor = np.array([v1, v2, v3])
            strained_mat = np.matmul(mat, strain_tensor)
            strained_lat = mg.Lattice(strained_mat)
            temp_struct.lattice = strained_lat
            temp_struct.scale_lattice(orig_vol)
            varied_structs.append(temp_struct)

    if sg in list(range(16, 76)): ## Orthorhombic
        for index in range(num_strains):
            temp_struct = mg.Structure.from_dict(struct_dict)
            orig_vol = temp_struct.volume
            a = random.choice(space_1)
            b = random.choice(space_1)
            c = random.choice(space_1)
            v1 = [a, 0, 0]
            v2 = [0, b, 0]
            v3 = [0, 0, c]
            strain_tensor = np.array([v1, v2, v3])
            strained_mat = np.matmul(mat, strain_tensor)
            strained_lat = mg.Lattice(strained_mat)
            temp_struct.lattice = strained_lat
       	    temp_struct.scale_lattice(orig_vol)
            varied_structs.append(temp_struct)

    if sg in list(range(3, 16)): ## Monoclinic
        for index in range(num_strains):
            temp_struct = mg.Structure.from_dict(struct_dict)
            orig_vol = temp_struct.volume
            a = random.choice(space_1)
            b = random.choice(space_1)
            c = random.choice(space_1)
            e = random.choice(space_0)
            f = random.choice(space_0)
            v1 = [a, 0, 0]
            v2 = [0, b, e]
            v3 = [0, f, c]
            strain_tensor = np.array([v1, v2, v3])
            strained_mat = np.matmul(mat, strain_tensor)
            strained_lat = mg.Lattice(strained_mat)
            temp_struct.lattice = strained_lat
       	    temp_struct.scale_lattice(orig_vol)
            varied_structs.append(temp_struct)

    if sg in list(range(1, 3)): ## Triclinic
        for index in range(num_strains):
            temp_struct = mg.Structure.from_dict(struct_dict)
            orig_vol = temp_struct.volume
            a = random.choice(space_1)
            b = random.choice(space_1)
            c = random.choice(space_1)
            v1 = [a, random.choice(space_0), random.choice(space_0)]
            v2 = [random.choice(space_0), b, random.choice(space_0)]
            v3 = [random.choice(space_0), random.choice(space_0), c]
            strain_tensor = np.array([v1, v2, v3])
            strained_mat = np.matmul(mat, strain_tensor)
            strained_lat = mg.Lattice(strained_mat)
            temp_struct.lattice = strained_lat
       	    temp_struct.scale_lattice(orig_vol)
            varied_structs.append(temp_struct)

    ## Separate low- vs. high-symmetry tetragonal/hexagonal groups
    low_sym = list(range(75, 83)) + list(range(143, 149)) + list(range(168, 175))
    high_sym = list(set(list(range(75, 195))) - set(low_sym))

    if sg in low_sym: ## Tetragonal/Hexagonal (lower-sym)
        for index in range(num_strains):
            temp_struct = mg.Structure.from_dict(struct_dict)
            orig_vol = temp_struct.volume
            a = random.choice(space_1)
            b = random.choice(space_1)
            c = random.choice(space_0)
            v1 = [a, c, 0]
            v2 = [-c, a, 0]
            v3 = [0, 0, b]
            strain_tensor = np.array([v1, v2, v3])
            strained_mat = np.matmul(mat, strain_tensor)
            strained_lat = mg.Lattice(strained_mat)
            temp_struct.lattice = strained_lat
       	    temp_struct.scale_lattice(orig_vol)
            varied_structs.append(temp_struct)

    if sg in high_sym: ## Tetragonal/Hexagonal (higher-sym)
        for index in range(num_strains):
            temp_struct = mg.Structure.from_dict(struct_dict)
            orig_vol = temp_struct.volume
            a = random.choice(space_1)
            b = random.choice(space_1)
            v1 = [a, 0, 0]
            v2 = [0, a, 0]
            v3 = [0, 0, b]
            strain_tensor = np.array([v1, v2, v3])
            strained_mat = np.matmul(mat, strain_tensor)
            strained_lat = mg.Lattice(strained_mat)
            temp_struct.lattice = strained_lat
       	    temp_struct.scale_lattice(orig_vol)
            varied_structs.append(temp_struct)

    return varied_structs



def map_interval(x, magnitude):
    """
    Map [0, 1] to [magnitude, 1]

    Args:
        x: value between 0 and 1 that is to be mapped onto new interval
        magnitude: lower bounds on mapping interval
    Returns:
        Newly mapped value
    """

    sc_num = 1.0 - magnitude
    return sc_num + ( ( (1.0 - sc_num) / (1.0 - 0.0) ) * (x - 0.0) )

def apply_texture(struct, magnitude):
    """
    Simulate diffraction patterns with peak intensities scaled as to
    emulate texture in the corresponding sample.

    Args:
        struct: pymatgen structure object
        magnitude: strength of texture (e.g., 0.6 means peaks will be
            scaled by as much as 60% their initial intensities
    Returns:
        A list of XRD spectra with texture applied stochastically
    """

    ## Get pattern and peak indicies
    calculator = xrd.XRDCalculator()
    pattern = calculator.get_pattern(struct, two_theta_range=(0,80))
    angles = pattern.x
    peaks = pattern.y
    hkls = [info[0]['hkl'] for info in pattern.hkls]

    ## Hexagonal systems treated uniquely since they use four Miller indices by convention
    scaled_peaks = []
    if struct.lattice.is_hexagonal() == True:
        check = 0.0
        while check == 0.0:
            preferred_direction = [random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1])]
            check = np.dot(np.array(preferred_direction), np.array(preferred_direction)) ## Make sure we don't have 0-vector

    ## Otherwise, usual 3 indices are employed
    else:
        check = 0.0
        while check == 0.0:
            preferred_direction = [random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1])]
            check = np.dot(np.array(preferred_direction), np.array(preferred_direction)) ## Make sure we don't have 0-vector
    for (hkl, peak) in zip(hkls, peaks):
        norm_1 = math.sqrt(np.dot(np.array(hkl), np.array(hkl)))
        norm_2 = math.sqrt(np.dot(np.array(preferred_direction), np.array(preferred_direction)))
        total_norm = norm_1 * norm_2
        texture_factor = np.dot(np.array(hkl), np.array(preferred_direction)) / total_norm
        texture_factor = map_interval(texture_factor, magnitude)
        print(texture_factor)
        scaled_peaks.append(peak*texture_factor)

    ## Generate spectrum with newly scaled peak intensities
    x = np.linspace(10, 80, 4501)
    y = []
    for val in x:
        ysum = 0
        for (ang, pk) in zip(angles, scaled_peaks):
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

    ## Re-shape for keras
    all_I = [[val] for val in scaled_vals]
    return all_I



def shrink_domain(struct, size):
    """
    Simulate diffraction patterns with peaks broadened according to
    a given domain size using the Scherrer equation.

    Args:
        struct: pymatgen structure object
        size: domain size to be sample
    Returns:
        XRD spectrum with peaks broadened
    """

    calculator = xrd.XRDCalculator()
    pattern = calculator.get_pattern(struct, two_theta_range=(0,80))
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
            ## Calculate FWHM
            tau = float(size) ## particle size in nm
            K = 0.9 ## shape factor
            wavelength = 0.15406 ## Cu K-alpha in nm
            theta = math.radians(ang/2.) ## Bragg angle in radians
            beta = (K / wavelength) * (math.cos(theta) / tau)
            ## Convert FWHM to std deviation of gaussian
            std_dev = beta/2.35482
            ## Convlution of gaussian
            gauss = [int*np.exp((-(val - ang)**2)/std_dev) for val in x]
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
