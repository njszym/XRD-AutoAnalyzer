# Auto-XRD

An autonomous deep learning model trained to perform phase identification from XRD spectra. 

## An example: running the model trained on the Li-Mn-Ti-O-F chemical space

A ready-made model can be found in the Pre-Trained_Example directory. Experimentally measured test spectra are also supplied. The model can be run to classify these spectra by executing the run_CNN.py script.

## Training the model on a new composition space

To perform phase identification in a new composition space, place all relevant CIFs into a reference folder contained by the Novel_Space directory. Then perform:

```
python construct_model.py $CIF_FOLDER
```

Where $CIF_FOLDER denotes the path to the reference folder containing all of the CIF files. This script will:

1) Filter all unique stoichiometric phases from the list of CIFs.

2) Generate hypothetical solid solutions from these phases.

3) Simulate augmented XRD spectra from the phases produced by (1) and (2).

4) Train a CNN on the simulated spectra.

If this procedure is completed successfully, a Model.h5 file will be available in the Novel_Space directory. Using this model, new spectra can be classified by:

```
python run_CNN.py $REFERENCE_FOLDER $SPECTRA_FOLDER
```

Where $REFERENCE_FOLDER is the path to the folder containing all unique reference phases and $SPECTRA_FOLDER is the path to the folder containing all spectra that are to be tested.

The script will give all suspected phases for each spetrum along with their associated probabilities.

