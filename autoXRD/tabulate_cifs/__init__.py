import numpy as np
import shutil
import os
import pymatgen as mg
from pymatgen.analysis import structure_matcher as sm


def get_stoichiometric_info(cif_directory):

    stoich_structs, temps, dates = [], [], []
    for cmpd in os.listdir(cif_directory):
        struct = mg.Structure.from_file('%s/%s' % (cif_directory, cmpd))
        if struct.is_ordered:
            stoich_structs.append(struct)
            t, d = parse_measurement_conditions(cif_directory, cmpd)
            temps.append(t)
            dates.append(d)
    return stoich_structs, temps, dates

def parse_measurement_conditions(cif_directory, filename):

    temp, date = 0.0, None
    with open('%s/%s' % (cif_directory, filename)) as entry:
        for line in entry.readlines():
            if '_audit_creation_date' in line:
                date = line.split()[-1]
            if '_cell_measurement_temperature' in line:
                temp = float(line.split()[-1])
    return temp, date

def get_unique_struct_info(stoich_refs, temps, dates):

    matcher = sm.StructureMatcher(scale=True, attempt_supercell=True, primitive_cell=False)
    unique_frameworks = []
    for struct_1 in stoich_refs: ## First tabulate all unique structural frameworks
        unique = True
        for struct_2 in unique_frameworks:
            if matcher.fit(struct_1, struct_2):
                unique = False
        if unique:
            unique_frameworks.append(struct_1)
    grouped_structs, grouped_temps, grouped_dates = [], [], []
    for framework in unique_frameworks:
        struct_class, temp_class, date_class = [], [], []
        for (struct, t, d) in zip(stoich_refs, temps, dates):
            if matcher.fit(framework, struct):
                struct_class.append(struct)
                temp_class.append(t)
                date_class.append(d)
        grouped_structs.append(struct_class)
        grouped_temps.append(temp_class)
        grouped_dates.append(date_class)
    return grouped_structs, grouped_temps, grouped_dates

def get_recent_RT_entry(grouped_structs, grouped_temps, grouped_dates):

    filtered_cmpds = []
    for (struct_class, temp_class, date_class) in zip(grouped_structs, grouped_temps, grouped_dates):
        normalized_temps = abs(np.array(temp_class) - 293.0) ## Difference from RT
        zipped_info = list(zip(struct_class, normalized_temps, date_class))
        sorted_info = sorted(zipped_info, key=lambda x: x[1]) ## Sort by temperature
        best_entry = sorted_info[0] ## Take the entry measured at the temperature closest to RT
        candidate_structs, candidate_dates = [], []
        for entry in sorted_info:
            if entry[1] == best_entry[1]: ## If temperature matches best entry
                candidate_structs.append(entry[0])
                candidate_dates.append(entry[2])
        zipped_info = list(zip(candidate_structs, candidate_dates))
        sorted_info = sorted(zipped_info, key=lambda x: x[1]) ## Sort by date
        final_struct = sorted_info[-1][0] ## Take the entry that was measured most recently
        filtered_cmpds.append(final_struct)
    return filtered_cmpds
