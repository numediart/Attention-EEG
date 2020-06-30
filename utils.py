import numpy as np

import os
import mne

mne.set_log_level(verbose='CRITICAL')

"""     DATASET CREATION    """


def raw_eye(path_raw, sub_dir):
    eye_info = {}
    path_sub = path_raw + sub_dir
    for r, d, f in os.walk(path_sub):
        if 'TaskRecords' in r:
            for file in f:
                if 'Total' in file:
                    path_file = os.path.join(r, file)
                    if '2' in file:
                        x = np.loadtxt(path_file)
                        eye_info['task2'] = x
                    elif '3' in file:
                        x = np.loadtxt(path_file)
                        eye_info['task3'] = x
    return eye_info


def raw_eeg(path_raw, sub_dir, f_s=500, task_duration=4.25):
    eeg_info = {}
    path_sub = path_raw + sub_dir
    for r, d, f in os.walk(path_sub):
        for file in f:
            if 'vhdr' in file:
                path_file = os.path.join(r, file)
                raw = mne.io.read_raw_brainvision(path_file)  # load file
                event, event_id = mne.events_from_annotations(raw)  # extract event array and event_id dict

                time = event[event[:, -1] == np.unique(event[:, -1])[2], 0]  # keep only the event corresponding to
                # stimuli

                t = []
                for i in range(len(time)):
                    if time[i] - time[i - 1] > 250 or i == 0:  # merge stimuli with ISI < 0.5
                        t.append([time[i], 0, 10003])
                t = np.asarray(t)

                event = t
                event_id = {'Comment/C': 10003}

                picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=True,
                                       include=[], exclude='bads')
                epochs = mne.Epochs(raw, event, event_id, tmin=-1, tmax=3, picks=picks,
                                    baseline=(None, 0), reject=None,
                                    preload=True)

                # f_s = raw.info['sfreq']

                eeg_info['task2'] = epochs.get_data()[np.argwhere(event[:, 0] < event[0, 0] + task_duration * f_s * 60)]
                eeg_info['task3'] = epochs.get_data()[np.argwhere(event[:, 0] > event[0, 0] + task_duration * f_s * 60)]

    return eeg_info


def raw_phy(path_raw, sub_dir):
    phy_info = {}
    path_sub = path_raw + sub_dir
    for r, d, f in os.walk(path_sub):
        if 'RawSignals' in r:
            for file in f:
                if 'PhysiologicalSig' in file:
                    PhySig = np.loadtxt(os.path.join(r, file))
                elif 'Task2' in file:
                    tapp_2 = np.loadtxt(os.path.join(r, file))[:, 0]
                elif 'Task3' in file:
                    tapp_3 = np.loadtxt(os.path.join(r, file))[:, 0]

    Task2 = []
    for t in tapp_2:
        id_t = np.argmin(np.abs(PhySig[:, 0] - t))
        Task2.append(PhySig[id_t - 3 * 5: id_t + 3 * 5, [1, 2, 3, 4, 5, 6, 13]])
    Task2 = np.asarray(Task2)
    phy_info['task2'] = Task2

    Task3 = []
    for t in tapp_3:
        id_t = np.argmin(np.abs(PhySig[:, 0] - t))
        Task3.append(PhySig[id_t - 3 * 5: id_t + 3 * 5, [1, 2, 3, 4, 5, 6, 13]])
    Task3 = np.asarray(Task3)
    phy_info['task3'] = Task3
    return phy_info


def eye_p_t2(x, min_param=-5):
    x = x[::2] + x[1::2]
    x[x <= 0] = min_param
    x = custom_sig(x)
    return x


def eye_p_t3(x, min_param=-2):
    x = 1 / x
    x[x <= 0] = min_param
    x = custom_sig(x)
    return x


def custom_sig(x, scale=4):
    x = 1 / (1 + np.exp(-x / scale))
    return x


def Eye_preprocess(Eye_track):
    for s in range(len(Eye_track)):
        Eye_track[s]['task2'] = eye_p_t2(Eye_track[s]['task2'])
        Eye_track[s]['task3'] = eye_p_t3(Eye_track[s]['task3'])


def Phy_preprocess(Phy):
    for s in range(len(Phy)):
        for k in Phy[s].keys():
            for epoch in range(Phy[s][k].shape[0]):
                for sig in range(Phy[s][k][epoch].shape[1]):
                    Phy[s][k][epoch][:, sig] = comp_acceleration(Phy[s][k][epoch][:, sig])
            Phy[s][k] = Phy[s][k].mean(axis=1)
    return Phy


def comp_acceleration(sig):
    acc = np.gradient(np.gradient(sig))
    acc = np.abs(acc)
    return acc


"""     EEG PREPROCESSING       """


def mean_filtering(raw_eeg):
    for s in range(len(raw_eeg)):
        for k in raw_eeg[s].keys():
            for epoch in range(raw_eeg[s][k].shape[0]):
                raw_eeg[s][k][epoch][0] -= raw_eeg[s][k][epoch][0].mean(axis=0)
    return raw_eeg


def array_to_epoch(array):
    ch_names = ['FP1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'P3', 'P7', 'O1', 'OZ',
                'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'CZ', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'FP2']
    ch_types = len(ch_names) * ['eeg']
    info = mne.create_info(ch_names=ch_names, sfreq=500.0, ch_types=ch_types)
    raw = mne.io.RawArray(array, info)
    return raw


def psd_compute(array, fmin=0.5, fmax=50):
    raw = array_to_epoch(array)
    psds, freqs = mne.time_frequency.psd_multitaper(raw, fmin=fmin, fmax=fmax, n_jobs=10, verbose=50)
    return psds, freqs


def psd_dataset(raw_eeg):
    Freqs = []
    Psd = []
    Subject = []
    Task = []
    for s in range(len(raw_eeg)):
        for k in raw_eeg[s].keys():
            for epoch in range(raw_eeg[s][k].shape[0]):
                psds, freqs = psd_compute(raw_eeg[s][k][epoch][0])
                Freqs.append(freqs)
                Psd.append(psds)
                Subject.append(s)
                Task.append(k)
    Spectral = {}
    Spectral["frequency_sample"] = np.asarray(Freqs)
    Spectral["power_spectral_density"] = np.asarray(Psd)
    Spectral["Participant"] = np.asarray(Subject)
    Spectral["Task"] = np.asarray(Task)
    return Spectral
