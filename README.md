# Auto-XRD

An autonomous deep learning model trained to perform phase identification from XRD spectra. 

## Training the model on a new composition space

To perform phase identification in a new composition space, carry out the following procedure in the Novel_Space directory:

1) In the Tabulate_Reference_Phases directory, place all possible CIF files of interest in the All-Possible_CIFs subfolder. Then run filter_references.py, which will create a References folder containing all unique reference phases.

2) If non-stoichiometry is to be considered, copy the References folder into the Non-Stoichiometry directory and run interpolate_solid-solns.py. This will create a Solid_Solns folder containing all hypothetical solid solutions that are interpolated from the stoichiometric reference phases. These may then be combined with all phases in the References folder.

3) Copy the References folder into the Generate_Training_Spectra directory and run get_patterns.py to simulate all diffraction spectra that will be used during training. This data will be written as XRD.npy.

4) Copy XRD.npy into the Train_CNN directory and run train.py to train the CNN. The resulting model will be generated as Model.h5.

5) The model is now ready to be used. Place Model.h5 and the References directory into the Run_CNN folder. Any spectra of interest should be written in xy format and placed in the Spectra folder. Then the phase identification algorithm can be run using run_CNN.py, which will anayze any and all spectra contained in the Spectra folder. Predicted phases are given along with their associated confidence.
