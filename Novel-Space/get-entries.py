from pymatgen.core import Structure
import shutil
import os

"""
This script is useful to get entries within a composition space,
assuming you have a folder (here, named ICSD) containing many
CIF files of all compositions.

Simply specify the elements you wish to include (inc_elems).
"""

inc_elems = {'Li', 'Sn', 'In', 'O'}

os.chdir('ICSD/')
all_entries = sorted(os.listdir('.'))

total = len(all_entries)
print(total)
qi = 0
for cmpd in all_entries:

    qi += 1
    if qi%100 == 0:
        print('\n\n')
        print(qi)
        print('\n\n')

    try:
        struc = Structure.from_file(cmpd)
    except:
        print('Oh oh')
        continue

    elems = set([str(e) for e in struc.composition.element_composition.elements])

    if elems.issubset(inc_elems):
        shutil.copyfile(cmpd, '../All_CIFs/%s' % cmpd)
