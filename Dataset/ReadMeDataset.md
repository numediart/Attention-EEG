 Hello PhD Fellows ! 

####################### READ ME: Dataset for EEG and Saliency Map #######################

As reminder, we have here a dataset with recorded for 30 different particpants. The task during when the signals have been acquired are attention task during which we ask to the participants to look at specific visual stimuli.

The dataset has already been preprocessed, the provided files are:

- Participant.npy [n_trials x 1]: array with the participant id for each EEG signals previously recorded. 

- eye_track_score.npy list of n_subject with dictonnary {'task2'}[n_trial_during_t2 x 1]; {'task3'}[n_trial_during_t3]: list of dictionnary with the attention score (0-100%) for each participants, task, trials. If flatten, it can be reshaped in [n_trials x 1].

- Label.npy [n_trials x 1]: array with the label computed from score for each EEG signals previously recorded.

- Preprocessed_EEG [n_trials x 401 x n_channels]: Array with preprocessed EEG already cut in segment for each trials. The signals have been downsampled with a ratio of 5 (i.e. freq_init/freq_down = 500/100 = 5).

- EEG_img [n_trials x 401 x 32 x 32]: Array with interpolated information from the array to 2D images representation of information (as presented in Bashivan et al. 2016 'Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks'). We take each trial (dim = [1 x 401 x n_channels]) and for each features ([1 x n_channels]), we expressed from the coordinate frame in 3D (x, y, z are the elctrodes position) to a 2D projection (x_proj, y_proj) and then to an interpolated image (x, y to pixel coordinate frame). 

- temp_array.npy [n_trials x 3 x n_channels]: Array with temporal features for each trials. Considered feature are Hjorth parameters (as presented in Lotte et al. 2014 'A Tutorial on EEG Signal-processing Techniques for Mental-state Recognition in Brain–Computer Interfaces').

- freq_array.npy [n_trials x 3 x n_channels]: Array with spectral features for each trials.

- temp_img.npy [n_trials x 3 x 32 x 32]: Array with interpolated information from the array to 2D images representation of information (as presented in Bashivan et al. 2016 'Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks'). Method similar than for EEG_img. 

- freq_img.npy [n_trials x 3 x 32 32]: Array with interpolated information from the array to 2D images representation of information (as presented in Bashivan et al. 2016 'Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks'). Method similar than for EEG_img. 

- Latent.npy [n_trials x 16]: Array with latent representation of the EEG_img after a Deep AutoEncoder network. 
