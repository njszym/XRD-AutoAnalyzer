# AutoAnalyzer for XRD

A package designed to automate the process of phase identification from XRD spectra using a probabilistic deep learning trained with physics-informed data augmentation.

The corresponding manuscript will be made available soon. Reproducing the published results can be accomplished using the data provided in [figshare](https://figshare.com/s/69030545b8020de35633).

## Installation

First clone the repository:

```
git clone https://github.com/njszym/Auto-XRD.git
```

Then, to install all required modules, navigate to the cloned directory and execute:

```
python setup.py install --user
```

## Usage example

A pre-trained model for the Li-Mn-Ti-O-F chemical space is available in the ```Example/``` directory. To classify the experimentally measured patterns tabulated in the ```Spectra/``` sub-folder, run the following:

```
python run_CNN.py
```

The characterization of each spectrum should take around 1-2 minutes on a single processor. Once all classifications are made, the predicted phases will be printed along with their associated confidence.

## Training the model for new compositions

To develop a model that can be used to perform phase identification in a new chemical space, place all relevant CIFs into a reference folder contained in the ```Novel_Space/``` directory (by default, the ```All_CIFs/``` sub-folder will be used). Then execute:

```
python construct_model.py --include_ns
```

This script will:

1) Filter all unique stoichiometric phases from the provided CIFs.

2) Generate hypothetical solid solutions between these materials if ```--include_ns``` is specified.

3) Simulate augmented XRD spectra from the phases produced by (1) and (2).

4) Train a CNN on the simulated spectra.

We caution that this process may require a substantial amount of computational resources depending on the size of the composition space considered. For example: training our model in the Li-Mn-Ti-O-F space, which included 255 reference phases, required about 12 hours of computational runtime on 16 cores. Necessary computational time should scale linearly with the number of reference phases. Similarily, time is reduced linearly with the number of cores used as all processes executed here are perfectly parallel (i.e., independent of one another).

When the procedure is completed, a Model.h5 file will be made available. 

## Characterizing multi-phase spectra

In the directory containing the trained model (Model.h5) and ```References``` folder, place all spectra to be classified in the ```Spectra``` folder. Then, spectra can be classified by executing:

```
python run_CNN.py
```

Output will appear as:

```
Filename: (name of the spectrum)
Predicted phases: (phase_1 + phase_2 + ...)
Confidence: (probability associated with the phases above)
```

Phase labels are denoted as ```formula_spacegroup```.

To plot the line profiles of the predicted phases in the measured spectrum for comparison, the ```--plot``` option may be used:

```
python run_CNN.py --plot
```

Note that the iterative process of phase identification and profile subtraction will proceed until all measured intensities fall below some cutoff value. By default, the cutoff is set to 10% of the maximum diffraction peak intensity. Alternatively, if you are only working with single-phase (or N-phase) patterns, the number of maximum phases considered can be manually set to 1 (or N).
