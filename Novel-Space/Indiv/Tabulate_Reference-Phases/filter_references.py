import shutil
import os
import pymatgen as mg
from pymatgen.analysis import structure_matcher as sm

os.mkdir('temp')
os.chdir('temp/')
os.mkdir('Stoichiometric')
print('Getting stoichiometric phases')
for cmpd in os.listdir('../All-Possible-CIFs/'):
    s = mg.Structure.from_file('../All-Possible-CIFs/%s' % cmpd)
    for site in s:
        if s.is_ordered:
            shutil.copyfile('../All-Possible-CIFs/%s' % cmpd, 'Stoichiometric/%s' % cmpd)
print('Done')

os.mkdir('Unique')
matcher = sm.StructureMatcher(scale=True, attempt_supercell=True, primitive_cell=False)
print('Getting unique structures')
for cmpd in os.listdir('Stoichiometric/'):
    s1 = mg.Structure.from_file('Stoichiometric/%s' % cmpd)
    f1 = s1.composition.reduced_formula
    sg1 = s1.get_space_group_info()[1]
    matched_formulae = []
    for ref in os.listdir('Unique/'):
        s2 = mg.Structure.from_file('Unique/%s' % ref)
        if matcher.fit(s1, s2):
            is_unique = False
            matched_formulae.append(ref)
    if len(matched_formulae) == 0:
        fname = '%s_%s_1.cif' % (f1, sg1)
        shutil.copyfile('Stoichiometric/%s' % cmpd, 'Unique/%s' % fname)
    else:
        last_fname = sorted(matched_formulae)[-1]
        index = int(last_fname.split('_')[-1].split('.')[0]) + 1
        fname = '%s_%s_%s.cif' % (f1, sg1, index)
        if fname not in os.listdir('Unique/'):
            shutil.copyfile('Stoichiometric/%s' % cmpd, 'Unique/%s' % fname)
print('Done')

os.mkdir('../References/')
base_cmpds = []
print('Filtering structures by conditions and dates')
for cmpd in os.listdir('Unique/'):
    fname = '%s_%s' % (cmpd.split('_')[0], cmpd.split('_')[1])
    base_cmpds.append(fname)
unique_cmpds = list(set(base_cmpds))
all_groups = []
for base in unique_cmpds:
    group = []
    for cmpd in os.listdir('Unique/'):
        if base in cmpd:
            group.append(cmpd)
    all_groups.append(group)
for group in all_groups:
    dates = []
    uncommon_temp = False
    ambient_cmpds = []
    for cmpd in group:
        with open('Unique/%s' % cmpd) as f:
            lines = f.readlines()
        for line in lines:
            if '_audit_creation_date' in line:
                date = line.split()[-1]
            if '_cell_measurement_temperature' in line:
                uncommon_temp = True
        if not uncommon_temp:
            ambient_cmpds.append(cmpd)
            dates.append(date)
    if len(ambient_cmpds) > 0:
        zipped_list = list(zip(ambient_cmpds, dates))
        sorted_list = sorted(zipped_list, key=lambda x: x[1])
        final_cmpd = sorted_list[-1][0] ## Most recently reported under ambient conditions
    else:
        temps = []
        T_cmpds = []
        for cmpd in group:
            with open('Unique/%s' % cmpd) as f:
                lines = f.readlines()
            for line in lines:
                if '_cell_measurement_temperature' in line:
                    T_cmpds.append(cmpd)
                    temps.append(float(line.split()[-1]))
        zipped_list = list(zip(T_cmpds, temps))
        sorted_list = sorted(zipped_list, key=lambda x: x[1])
        final_cmpd = sorted_list[0][0] ## Lowest temperature measurement
    formatted_cmpd_str = '%s_%s.cif' % (cmpd.split('_')[0], cmpd.split('_')[1])
    shutil.copyfile('Unique/%s' % final_cmpd, '../References/%s' % formatted_cmpd_str)
os.chdir('../')
shutil.rmtree('temp')
print('Done')
