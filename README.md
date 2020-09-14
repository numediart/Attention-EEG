# EEG-Attention

Repo for the work on attention EEG and saliency map.

Different features are given:
* Frequential feature with PSD for 3 different bands ($\alpha$, $\beta$ and $\theta$).
* Temporal feaures with Hjorth parameters.
* Handcrafted features from deep AutoEncoder.

Different models are presented:
* CNN for image based temporal or frequential features.
* RNN for temporal or frequential features.
* Graph based model.
* Deep Autoencoder for feature extraction.

## Requirements

In order to run the codes, the following libraries (and their corresponding dependencies) have to been installed:

- Python     3.8
- Pytroch     1.5.1
- Cudatoolkit     10.1.243
- Torch-Geometric 1.6.1 (mandatory only for Graph based model) 

Installation with pip: `pip install -r requirements.txt`

Import of the environment with conda: `conda env create -f environment.yml`
