# Auto-XRD

A package designed to automate the process of phase identification from XRD spectra using a probabilistic deep learning trained with physics-informed data augmentation.

## Installation

## Usage example

A pre-trained model for the Li-Mn-Ti-O-F chemical space is available in the ```Example.``` directory. To classify the experimentally measured patterns tabulated in the ```Spectra/``` sub-folder, run the following:

```
python run_CNN.py
```

The characterization of each spectrum will take around 1-2 minutes on a single processor. As classifications are made, the predicted phases will be printed along with their associated confidence.

## Training the model for new compositions

To develop a model that can be used to perform phase identification in a new chemical space, place all relevant CIFs into a reference folder contained in the ```Novel_Space/``` directory (by default, the ```All_CIFs``` sub-folder will be used). Then execute:

```
python construct_model.py $CIF_FOLDER
```

Where ```$CIF_FOLDER``` denotes the path to the reference folder containing all of the CIF files. This script will:

1) Filter all unique stoichiometric phases from the provided CIFs.

2) Generate hypothetical solid solutions between these materials.

3) Simulate augmented XRD spectra from the phases produced by (1) and (2).

4) Train a CNN on the simulated spectra.

We caution that this process may require a substantial amount of computational resources depending on the size of the composition space considered. For example: training our model in the Li-Mn-Ti-O-F space, which included 255 reference phases, required 20 hours of computational runtime on 16 cores. Necessary computational time should scale linearly with the number of reference phases. Similarily, time is reduced linearly with the number of cores used as all processes executed here are perfectly parallel (i.e., independent of one another).

When the procedure is completed, a Model.h5 file will be made available. Using this model, new spectra can be classified by executing:

```
python run_CNN.py $REFERENCE_FOLDER $SPECTRA_FOLDER
```

Where ```$REFERENCE_FOLDER``` is the path to the folder containing all unique reference phases and ```$SPECTRA_FOLDER``` is the path to the folder containing all spectra that are to be tested. By default, these are set to ```References/``` and ```Spectra``` respectively.


