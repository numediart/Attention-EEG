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
- Pytorch     1.5.1
- Cudatoolkit     10.1.243
- Torch-Geometric 1.6.1 (mandatory only for Graph based model) 

Installation with pip: `pip install -r requirements.txt`
Then to install pytorch and pytorch-geometric, follow the installation guide in the corresponding documentation ([pytorch](https://pytorch.org/get-started/locally/) and [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)). 

Import of the environment with conda: `conda env create -f environment.yml`

## Citation

If you use the dataset, please cite the original paper:

	@article{delvigne_phydaa_2021,
		title = {{PhyDAA}: {Physiological} {Dataset} {Assessing} {Attention}},
		issn = {1558-2205},
		shorttitle = {{PhyDAA}},
		doi = {10.1109/TCSVT.2021.3061719},
		journal = {IEEE Transactions on Circuits and Systems for Video Technology},
		author = {Delvigne, V. and Wannous, H. and Dutoit, T. and Ris, L. and Vandeborre, J.-P.},
		year = {2021},
		pages = {1--1}
	}

and for the CNN:

	@inproceedings{delvigne_attention_2020,
		title = {Attention {Estimation} in {Virtual} {Reality} with {EEG} based {Image} {Regression}},
		doi = {10.1109/AIVR50618.2020.00012},
		booktitle = {2020 {IEEE} {International} {Conference} on {Artificial} {Intelligence} and {Virtual} {Reality} ({AIVR})},
		author = {Delvigne, V. and Wannous, H. and Vandeborre, J.-P. and Ris, L. and Dutoit, T.},
		month = dec,
		year = {2020},
		pages = {10--16},
	}
