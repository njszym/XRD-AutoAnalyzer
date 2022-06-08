from pymatgen.core import Structure
import shutil
import os

"""
This script is useful to get entries within a composition space,
assuming you have a folder (here, named ICSD) containing many
CIF files of all compositions.

Simply specify the elements you wish to include (inc_elems).
"""

# Which composition space to work in
inc_elems = {'Li', 'Sn', 'In', 'O'}

# Folder containin all CIF files
os.chdir('ICSD/')
all_entries = sorted(os.listdir('.'))

total = len(all_entries)
for index, cmpd in enumerate(all_entries):

    # Keep track of progress
    if index%100 == 0:
        print('%s/%s' % (index, total))

    # Continue on if structure cannot load
    try:
        struc = Structure.from_file(cmpd)
    except:
        print('Structure failed to load')
        continue

    elems = set([str(e) for e in struc.composition.element_composition.elements])

    # Only include structure if elems are subset of desired composition space
    if elems.issubset(inc_elems):
        shutil.copyfile(cmpd, '../All_CIFs/%s' % cmpd)
