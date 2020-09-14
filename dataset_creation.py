from utils import *

path_to_save = 'Dataset/'

""" General Directory """

path_raw = 'Raw Data/'
n_sub = len(os.listdir(path_raw))
sub_dir = os.listdir(path_raw)
sub_dir.sort()

""" Eye - Tracking Information """
Eye_track = []

for p in range(n_sub):
    eye_info = raw_eye(path_raw, sub_dir[p])
    Eye_track.append(eye_info)
Eye_preprocess(Eye_track)

score = []
for s in range(len(Eye_track)):
    for k in Eye_track[s].keys():
        score.extend(Eye_track[s][k])
lab = np.asarray(score)//25

""" Electroencephalogram Information """

EEG = []
for p in range(n_sub):
    eeg_info = raw_eeg(path_raw, sub_dir[p])
    EEG.append(eeg_info)

""" Physiological Information """

Phy = []

for p in range(n_sub):
    phy_info = raw_phy(path_raw, sub_dir[p])
    Phy.append(phy_info)

Phy = Phy_preprocess(Phy)

""" Saving the raw files """


np.save(path_to_save + 'Label', lab)
np.save(path_to_save + 'eye_track_score', Eye_track)
np.save(path_to_save + 'raw_eeg', EEG)
np.save(path_to_save + 'phy_sig', Phy)

print('All files save in ' + path_to_save)