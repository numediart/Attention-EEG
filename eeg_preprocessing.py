from utils import *

import matplotlib.pyplot as plt

data_path = 'Dataset/'
info_path = 'Information/'

if not os.path.exists(data_path+'raw_eeg.npy'):
    print('Dataset has not been generated from raw files.\nGenerating Dataset...')
    exec(open("dataset_creation.py").read())
else :
    raw_eeg = np.load(data_path + 'raw_eeg.npy', allow_pickle=True)

if not os.path.exists(data_path+'psd.npy'):
    print('Spectral analysis has not been performed from EEG.\nGenerating PSD...')
    raw_eeg = mean_filtering(raw_eeg)
    spectral = psd_dataset(raw_eeg)
    np.save(data_path + 'psd', spectral)
else:
    spectral = np.load(data_path+'psd.npy', allow_pickle=True).all()

if not os.path.exists(data_path+'freq_band.npy'):
    print('Frequency bands has not been computed from PSD.\nGenerating Freq bands...')
    BandWith = freq_bands(spectral)
    np.save(data_path+'freq_band', BandWith)
else:
    BandWith = np.load(data_path+'freq_band.npy')

if not os.path.exists(data_path+'freq_img.npy'):
    print('Frequency images have not been computed.\nGenerating Images...')
    locs_3d = np.load(info_path+'ChanInfo.npy', allow_pickle=True).all()['position']
    freq_img = band_image(BandWith, locs_3d)
    np.save(data_path+'freq_img', freq_img)
else:
    np.load(data_path+'freq_img.npy')
