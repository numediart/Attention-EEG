'''
Created by Victor Delvigne
ISIA Lab, Faculty of Engineering University of Mons, Mons (Belgium)
victor.delvigne@umons.ac.be
Source: Delvigne, et al."PhyDAA: Physiological Dataset Assessing Attention" IEEE Transaction on Circuits and Systems for Video Technology (TCSVT) (2016).
Copyright (C) 2021 - UMons
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

'''

from utils import *

import matplotlib.pyplot as plt

data_path = 'Dataset/'
info_path = 'Information/'

""" General EEG Information """
if not os.path.exists(data_path+'raw_eeg.npy'):
    print('Dataset has not been generated from raw files.\nGenerating Dataset...')
    exec(open("dataset_creation.py").read())
raw_eeg = np.load(data_path + 'raw_eeg.npy', allow_pickle=True)
raw_eeg = mean_filtering(raw_eeg)

""" Frequency related feature extraction """

# Power Spectral Density 
if not os.path.exists(data_path+'psd.npy'):
    print('Spectral analysis has not been performed from EEG.\nGenerating PSD...')
    spectral = psd_dataset(raw_eeg)
    np.save(data_path + 'psd', spectral)
else:
    spectral = np.load(data_path+'psd.npy', allow_pickle=True).all()

# Frequency Band Separation
if not os.path.exists(data_path+'freq_band.npy'):
    print('Frequency bands has not been computed from PSD.\nGenerating Freq bands...')
    BandWith = freq_bands(spectral)
    np.save(data_path+'freq_band', BandWith)
else:
    BandWith = np.load(data_path+'freq_band.npy')

# Frequential Based Images
if not os.path.exists(data_path+'freq_img.npy'):
    print('Frequency images have not been computed.\nGenerating Frequential Images...')
    locs_3d = np.load(info_path+'ChanInfo.npy', allow_pickle=True).all()['position']
    freq_img = band_image(BandWith, locs_3d)
    np.save(data_path+'freq_img', freq_img)
else:
    np.load(data_path+'freq_img.npy')


""" Temporal related feature extraction """

# Hjorth Parameter
if not os.path.exists(data_path+'hjorth.npy'):
    print('Hjorth parameters have not been computed.\nGenerating Hjorth Parameters...')
    hjorth = temporal_dataset(raw_eeg)
    np.save(data_path+'hjorth', hjorth)
else:
    hjorth = np.load(data_path+'hjorth.npy')

# Temporal Based Images
if not os.path.exists(data_path+'temp_img.npy'):
    print('Temporal images have not been computed.\nGenerating Temporal Images...')
    locs_3d = np.load(info_path+'ChanInfo.npy', allow_pickle=True).all()['position']
    temp_img = band_image(hjorth, locs_3d)
    np.save(data_path+'temp_img', temp_img)
else:
    temp_img = np.load(data_path+'temp_img.npy')

""" Image based EEG """
# EEG Down Sampling
down_eeg = down_sampling(raw_eeg)
# EEG Based Images
if not os.path.exists(data_path+'tot_img.npy'):
    print('Temporal images have not been computed.\nGenerating Temporal Images...')
    locs_3d = np.load(info_path+'ChanInfo.npy', allow_pickle=True).all()['position']
    tot_img1 = band_image(down_eeg[0:2000], locs_3d)
    np.save(data_path+'tot_img1', tot_img1)
    tot_img2 = band_image(down_eeg[2000:5000], locs_3d)
    np.save(data_path+'tot_img2', tot_img2)
else:
    # 32gb ram necessary if not need to be adapted
    tot_img =  np.concatenate((np.load(data_path+'tot_img1.npy'),np.load(data_path+'tot_img2.npy')))