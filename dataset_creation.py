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
lab = np.asarray(score)//50 # attention score is between 0 - 100 %

""" Electroencephalogram Information """

EEG = []
for p in tqdm(range(n_sub)):
    eeg_info = raw_eeg(path_raw, sub_dir[p])
    EEG.append(eeg_info)

""" Physiological Information """

Phy = []

for p in range(n_sub):
    phy_info = raw_phy(path_raw, sub_dir[p])
    Phy.append(phy_info)

Phy = Phy_preprocess(Phy)


""" Participant Id """

Participant = []

for p in range(n_sub):
	for k in Eye_track[p].keys():
		Participant.extend([int(p+1)]*Eye_track[p][k].shape[0])

""" Saving the raw files """

np.save(path_to_save + 'Label', lab)
np.save(path_to_save + 'eye_track_score', Eye_track)
np.save(path_to_save + 'raw_eeg', EEG)
np.save(path_to_save + 'phy_sig', Phy)
np.save(path_to_save + 'participant', Participant)

print('All files save in ' + path_to_save)