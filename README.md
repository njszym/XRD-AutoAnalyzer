# An automatic analysis tool for XRD

A package designed to automate the process of phase identification from XRD spectra using a probabilistic deep learning trained with physics-informed data augmentation.

The corresponding manuscript can be found at [Chemistry of Materials](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.1c01071). Reproducing the published results can be accomplished using the data provided in [figshare](https://figshare.com/s/69030545b8020de35633).

## Installation

First clone the repository:

```
git clone https://github.com/njszym/XRD-AutoAnalyzer.git
```

Then, to install all required modules, navigate to the cloned directory and execute:

```
pip install . --user
```

A word of caution: the classification of multi-phase patterns relies on [fastdtw](https://github.com/slaypni/fastdtw) for peak fitting. This package performs dynamic time warping quickly by using a C++ implementation. In some cases, however, the C++ implementation is not successfully install, forcing the code to fall back to pure Python (without any warning). If one finds that ```run_CNN.py``` takes more > 1 minute per pattern, it is likely indicative of an issue with ```fastdtw```. To resolve this, please see [previous posts](https://stackoverflow.com/questions/44994866/efficient-pairwise-dtw-calculation-using-numpy-or-cython).

## Usage example

A pre-trained model for the Li-Mn-Ti-O-F chemical space is available in the ```Example/``` directory. To classify the experimentally measured patterns tabulated in the ```Spectra/``` sub-folder, run the following:

```
python run_CNN.py
```

The process should take about 10 seconds per spectrum on a single CPU. If multiple spectra are present, parallelization across all available CPU will be executed by default. Once all classifications are made, the predicted phases will be printed along with their associated probabilities.

For a detailed walkthrough of the steps used to perform phase identification, please see https://github.com/njszym/XRD-AutoAnalyzer/blob/main/Tutorial/tutorial-notebook.ipynb.

## Training the model for new compositions

To develop a model that can be used to perform phase identification in a new chemical space, place all relevant CIFs into ```Novel_Space/All_CIFs```. Then navigate to the ```Novel_Space/```directory and execute:

```
python construct_model.py
```

This script will:

1) Filter all unique stoichiometric phases from the provided CIFs. Alternatively, to provide a customized set of reference phases without filtering, place all CIFs into a folder labeled ```References``` and run ```python construct_model.py --skip_filter```. To employ automated filtering but exclude elemental phases, run ```python construct_model.py --ignore_elems```.

2) If ```--include_ns``` is specified: generate hypothetical solid solutions between the stoichiometric phases.

3) Simulate augmented XRD spectra from the phases produced by (1) and (2).

4) Train a convolutional neural network on the augmented spectra.

By default, training spectra will be simulated over 2θ spanning 10-80 degrees in Cu K-alpha radiation. However, this can be customized as follows:

```
python construct_model.py --min_angle=10.0 --max_angle=80.0
```

The model creation process may require a substantial amount of computational resources depending on the size of the composition space considered. For example: performing all necessary steps to create a model in the Li-Mn-Ti-O-F space, which included 255 reference phases, required about 4 hours of computational runtime on a single core. Required computational time should scale linearly with the number of reference phases. Similarily, time is reduced linearly with the number of cores used as all processes executed here are perfectly parallel (i.e., independent of one another).

When the procedure is completed, a trained ```Model.h5``` file will be made available. 

By default, the following bounds are used on artifacts included during data augmentation:

* Peak shifts (non-uniform): up to +/- 3% strain applied to each lattice parameter
* Peak shifts (uniform): up to +/- 0.5 degrees shift in all peak positions due to sample height error
* Peak broadening: domain size ranging from 5-30 nm
* Peak intensity variation: texture causing as much as +/- 50% change in peak height

However, custom bounds can also be specified, e.g., as follows:

```
python construct_model.py --max_strain=0.04 --max_shift=1.0 --min_domain_size=1 --max_domain_size=100 --max_texture=0.5
```

## Characterizing multi-phase spectra

In the directory containing ```Model.h5```, place all spectra to be classified in the ```Spectra/``` folder. These files should be in ```xy``` format.

Once all files are placed in the ```Spectra/``` folder, they can be classified by executing:

```
python run_CNN.py
```

Output will appear as:

```
Filename: (name of the spectrum)
Predicted phases: (phase_1 + phase_2 + ...)
Confidence: (probabilities associated with the phases above)
```

Phase labels are denoted as ```formula_spacegroup```.

By default, only phases with a confidence above 25% will be shown. To also show low-confidence phases, the ```-all``` argument can be used at runtime.

If spectra with a range of 2θ other than 10-80 degrees are considered, then the minimum and maximum diffraction angles (in Cu K-alpha) should be specified manually as shown below. Note: this range must match the range used during model creation (see section above).

```
python run_CNN.py --min_angle=10.0 --max_angle=80.0
```

The model assumes that spectra are measured using Cu K-alpha radiation. However, the user can specify any arbitary wavelength (```lambda```, in angstroms) as follows:

```
python run_CNN.py --wavelength=lambda
```

For each spectrum, the phase identification algorithm runs until either (i) a maximum of four unique compounds have been identified, or (ii) all peaks with intensities greater than or equal to 5% of the spectrum's maximum intensity have been identified. To change these parameters (denoted ```N``` and ```I```), the following arguments can be specified:

```
python run_CNN.py --max_phases=N --cutoff_intensity=I
```

To plot the line profiles of the predicted phases in the measured spectrum for comparison, the ```--plot``` option may also be used:

```
python run_CNN.py --plot
```

Which will yield a plot of the form:

![sample](./Example/sample-image.png)

Based on this plot, weight fractions can also be approximated by adding the ```--weights``` argument:

```
python run_CNN.py --weights
```

Which will provide an additional line in the output:

```
Filename: (name of the spectrum)
Predicted phases: (phase_1 + phase_2 + ...)
Confidence: (probabilities associated with the phases above)
Weight fractions: (% associated with each phase above)
```

We caution that these weight fractions should be treated only as an estimation. They are calculated by fitting over peak heights, whereas weight fraction is instead related to peak areas. For a more reliable quantification of each phase, a separate technique such as Rietveld refinement is required. 

If the user wishes to compare specific reference phases to the measured spectrum, the ```visualize.py``` script can be used as follows:

```
python visualize.py --spectrum='filename' --ph='cmpd1_sg' --ph='cmpd2_sg'
```

Where ```cmpd1_sg``` and ```cmpd2_sg``` refer to the phases that will be fit to the spectrum and plotted. Note that minimum and maximum angles, as well as the wavelength, must also be specified if they differ from default values.

## Stochastic nature of the model

Model performance may vary between different training procedures with the same set of hyperparameters. There are two sources of stochastic variations:

* Simulated spectra are randomly perturbed by experimental artifacts. As long as a sufficient number of spectra are simulated for each reference phase, this effect is usually negligible.
* The CNN is trained using a random seed, which can have a significant influence on convergence.

In cases where a model performs poorly on clean XRD data, one may considering re-training their model from scratch. Alternatively, an ensemble of different models may be used, where the final prediction is an averaged over all individual predictions.

We are currently working on a more generalizable solution to this problem. Stay tuned.


