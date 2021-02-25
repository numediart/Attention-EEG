# EEG-Attention

Repository for proposed models for attention estimation from Electroencephalogram and Physiological signals.

Due to EULA issues, the raw signals and preprocessed datasets are note provided here but are available on [Zenodo](https://zenodo.org/). 

Different features are given:
* Frequential feature with PSD for 3 different bands (α, β and θ).
* Temporal feaures with [Hjorth parameters](https://en.wikipedia.org/wiki/Hjorth_parameters) (activity, mobility and complexity).
* Handcrafted features from deep AutoEncoder.

Different models are presented:
* [CNN](SimpleCNN.py) for image based temporal or frequential features.
* [RNN](SimpleRNN.py) for temporal or frequential features.
* [Graph](GraphNetwork.py) based model for temporal or frequential features.
* [Deep Autoencoder](AutoEncoder.py) for feature extraction.

A .py file is provided to try each models, you just have to adapt the path to directory where the dataset is located. A notebook explaining how to load signals is also provided, the models implementation into the notebook is coming soon.

## Requirements

In order to run the codes, the following libraries (and their corresponding dependencies) have to been installed:

- Python     3.8
- Pytroch     1.5.1
- Cudatoolkit     10.1.243
- Torch-Geometric 1.6.1 (mandatory only for Graph based model) 

Installation with pip: `pip install -r requirements.txt`
Import of the environment with conda: `conda env create -f environment.yml`
