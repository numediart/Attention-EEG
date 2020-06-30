from utils import *

data_path = 'Dataset/'
raw_eeg = np.load(data_path + 'raw_eeg.npy', allow_pickle=True)

if not os.path.exists(data_path+'psd.npy'):
    raw_eeg = mean_filtering(raw_eeg)
    spectral = psd_dataset(raw_eeg)
    np.save(data_path + 'psd', spectral)

else:
    spectral = np.load(data_path+'psd.npy', allow_pickle=True).all()

print('ffffffff')

print(spectral)