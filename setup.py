from setuptools import setup, find_packages
from shutil import copytree
import os


# Install dependencies
setup(
    name='autoXRD',
    version='0.0.1',
    description='A package designed to automate the process of phase identification from XRD spectra using a probabilistic deep learning trained with physics-informed data augmentation.',
    author='Nathan J. Szymanski',
    author_email='nathan_szymanski@berkeley.edu',
    python_requires='>=3.6.0',
    url='https://github.com/njszym/XRD-AutoAnalyzer',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'pymatgen', 'scipy', 'scikit-image', 'tensorflow>=2.16', 'pyxtal', 'pyts', 'tqdm', 'asteval', 'numexpr>=2.8.3']
)
