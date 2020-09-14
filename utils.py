import numpy as np
import math as m

import os
import mne
import torch
import glob
import warnings

from scipy.interpolate import griddata, interp1d
from sklearn.preprocessing import scale
from tqdm import tqdm
from torch_geometric.data import Data, Dataset, DataLoader
from torch.utils.data import random_split, Subset

from torch.utils.tensorboard import SummaryWriter

mne.set_log_level( verbose='CRITICAL' )

"""     DATASET CREATION    """


def raw_eye(path_raw, sub_dir):
    eye_info = {}
    path_sub = path_raw + sub_dir
    for r, d, f in os.walk( path_sub ):
        if 'TaskRecords' in r:
            for file in f:
                if 'Total' in file:
                    path_file = os.path.join( r, file )
                    if '2' in file:
                        x = np.loadtxt( path_file )
                        eye_info['task2'] = x
                    elif '3' in file:
                        x = np.loadtxt( path_file )
                        eye_info['task3'] = x
    return eye_info


def raw_eeg(path_raw, sub_dir, f_s=500, task_duration=4.25):
    eeg_info = {}
    path_sub = path_raw + sub_dir
    for r, d, f in os.walk( path_sub ):
        for file in f:
            if 'vhdr' in file:
                path_file = os.path.join( r, file )
                raw = mne.io.read_raw_brainvision( path_file, preload=True )  # load file
            if 'fif' in file:
                path_file = os.path.join( r, file )
                raw = mne.io.read_raw_fif( path_file, preload=True )  # load file

            if ('fif' in file) or ('vhdr' in file):

                raw.filter( .05, None, fir_design='firwin' )
                raw.filter( None, 50., fir_design='firwin' )

                event, event_id = mne.events_from_annotations( raw )  # extract event array and event_id dict

                # time = event[event[:, -1] == np.unique(event[:, -1])[2], 0]  # keep only the event corresponding to
                time = event
                # stimuli

                t = []
                for i in range( len( time ) ):
                    if time[i, 0] - time[i - 1, 0] > 250 or time[i, 2] != time[
                        i - 1, 2]:  # merge stimuli with ISI < 0.5
                        t.append( [time[i, 0], 0, time[i, 2]] )
                t = np.asarray( t )

                if len( np.argwhere( t[:, -1] == 10001 ).squeeze() ) > 1:
                    id_mid = np.argwhere( t[:, -1] == 10001 ).squeeze().max()

                    id_begin_t2 = np.argwhere( t[:, -1] == 10001 ).squeeze()[-2]

                    # id_begin = np.argwhere(t[:, -1] == 10001).squeeze().min()
                    # id_end = np.argwhere(t[:, -1]==10001).squeeze().min()
                    # t_end =
                    t_stim = 10003
                else:
                    id_mid = np.argwhere( t[:, -1] == 1 ).squeeze().max()
                    id_begin_t2 = np.argwhere( t[:, -1] == 1 ).squeeze()[-2]
                    t_stim = 3

                event_t2 = t[id_begin_t2:id_mid]
                event_t2 = event_t2[event_t2[:, -1] == t_stim]
                event_t3 = t[id_mid:-1]
                event_t3 = event_t3[event_t3[:, -1] == t_stim]

                event_id = {'Comment/C': t_stim}

                picks = mne.pick_types( raw.info, meg=True, eeg=True, stim=False, eog=True,
                                        include=[], exclude='bads' )
                epochs_t2 = mne.Epochs( raw, event_t2, event_id, tmin=-1, tmax=3, picks=picks,
                                        baseline=(None, 0), reject=None,
                                        preload=True )
                epochs_t3 = mne.Epochs( raw, event_t3, event_id, tmin=-1, tmax=3, picks=picks,
                                        baseline=(None, 0), reject=None,
                                        preload=True )

                # f_s = raw.info['sfreq']

                eeg_info['task2'] = epochs_t2.get_data()
                eeg_info['task3'] = epochs_t3.get_data()

                # eeg_info['task2'] = epochs.get_data()[np.argwhere(event[:, 0] < event[t_mid, 0])]
                # eeg_info['task3'] = epochs.get_data()[np.argwhere(event[:, 0] >= event[t_mid, 0])]

    return eeg_info


def raw_phy(path_raw, sub_dir):
    phy_info = {}
    path_sub = path_raw + sub_dir
    for r, d, f in os.walk( path_sub ):
        if 'RawSignals' in r:
            for file in f:
                if 'PhysiologicalSig' in file:
                    PhySig = np.loadtxt( os.path.join( r, file ) )
                elif 'Task2' in file:
                    tapp_2 = np.loadtxt( os.path.join( r, file ) )[:, 0]
                elif 'Task3' in file:
                    tapp_3 = np.loadtxt( os.path.join( r, file ) )[:, 0]

    Task2 = []
    for t in tapp_2:
        id_t = np.argmin( np.abs( PhySig[:, 0] - t ) )
        Task2.append( PhySig[id_t - 3 * 5: id_t + 3 * 5, [1, 2, 3, 4, 5, 6, 13]] )
    Task2 = np.asarray( Task2 )
    phy_info['task2'] = Task2

    Task3 = []
    for t in tapp_3:
        id_t = np.argmin( np.abs( PhySig[:, 0] - t ) )
        Task3.append( PhySig[id_t - 3 * 5: id_t + 3 * 5, [1, 2, 3, 4, 5, 6, 13]] )
    Task3 = np.asarray( Task3 )
    phy_info['task3'] = Task3
    return phy_info


'''
def Eye_preprocess(Eye_track):
    for s in range(len(Eye_track)):
        Eye_track[s]['task2'] = eye_p_t2(Eye_track[s]['task2'])
        Eye_track[s]['task3'] = eye_p_t3(Eye_track[s]['task3'])

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
'''


def custom_sig(x, scale=4):
    x = 1 / (1 + np.exp( -x / scale ))
    return x


def Eye_preprocess(Eye_track):
    task2 = []
    task3 = []
    for s in range( len( Eye_track ) ):
        x = Eye_track[s]['task2']
        x = x[::2] + x[1::2]
        task2.append( x )
        task3.append( Eye_track[s]['task3'] )

    task2 = np.concatenate( task2 )
    task3 = np.concatenate( task3 )

    g_m_s_t2 = np.mean( task2 )
    g_std_s_t2 = np.std( task2 )

    m = 80 / (2 * g_std_s_t2)
    p = 50 - m * g_m_s_t2

    g_m_s_t3 = np.mean( task3 )
    g_std_s_t3 = np.std( task3 )

    x = [0.1, 0.3, 0.725, 1.15]
    y = [90, 50, 35, 7.5]  # values deduced from mean std

    #a, b, c = np.polyfit( x, y, 2 )
    a, b, c, d = np.polyfit(x, y, 3)

    task2 = []
    task3 = []
    for s in range( len( Eye_track ) ):
        # Task2
        x = Eye_track[s]['task2']
        x = x[::2] + x[1::2]
        x = m * x + p
        task2.append( x )
        x[x < 5] = 5
        x[x > 95] = 95

        Eye_track[s]['task2'] = x

        # Task3
        x = Eye_track[s]['task3']
        x[x <= 0] = 2
        #x = a * x ** 2 + b * x + c
        x = a * x ** 3 + b * x ** 2 + c * x + d
        x[x < 5] = 5
        x[x > 95] = 95
        assert isinstance(x, object)
        task3.append( x )

        Eye_track[s]['task3'] = x


def Phy_preprocess(Phy):
    for s in range( len( Phy ) ):
        for k in Phy[s].keys():
            for epoch in range( Phy[s][k].shape[0] ):
                for sig in range( Phy[s][k][epoch].shape[1] ):
                    Phy[s][k][epoch][:, sig] = comp_acceleration( Phy[s][k][epoch][:, sig] )
            Phy[s][k] = Phy[s][k].mean( axis=1 )
    return Phy


def comp_acceleration(sig):
    acc = np.gradient( np.gradient( sig ) )
    acc = np.abs( acc )
    return acc


"""     ELECTRODES POSITION       """


def electrodes_position(raw):
    chan_dict = {}
    pos = []
    name = []
    n_chan = len( raw.info['chs'] )
    for i in range( n_chan ):
        pos.append( raw.info['chs'][i]['loc'][0:3] )
        name.append( raw.info['chs'][i]['ch_name'] )
    chan_dict['position'] = np.asarray( pos )
    chan_dict['chan_name'] = np.asarray( name )
    return chan_dict


def elec_proj(loc_3d):
    locs_2d = []
    for l in loc_3d:
        locs_2d.append( azim_proj( l ) )
    return np.asarray( locs_2d )


"""     EEG PREPROCESSING       """


def mean_filtering(raw_eeg):
    for s in range( len( raw_eeg ) ):
        for k in raw_eeg[s].keys():
            for epoch in range( raw_eeg[s][k].shape[0] ):
                raw_eeg[s][k][epoch][0] -= raw_eeg[s][k][epoch][0].mean( axis=0 )
    return raw_eeg


def array_to_epoch(array):
    ch_names = ['FP1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'P3', 'P7', 'O1', 'OZ',
                'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'CZ', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'FP2']
    ch_types = len( ch_names ) * ['eeg']
    info = mne.create_info( ch_names=ch_names, sfreq=500.0, ch_types=ch_types )
    raw = mne.io.RawArray( array, info )
    return raw


def psd_compute(array, fmin=0.5, fmax=50):
    raw = array_to_epoch( array )
    psds, freqs = mne.time_frequency.psd_multitaper( raw, fmin=fmin, fmax=fmax, n_jobs=10, verbose=50 )
    return psds, freqs


def psd_dataset(raw_eeg):
    Freqs = []
    Psd = []
    Subject = []
    Task = []
    for s in range( len( raw_eeg ) ):
        for k in raw_eeg[s].keys():
            for epoch in range( raw_eeg[s][k].shape[0] ):
                psds, freqs = psd_compute( raw_eeg[s][k][epoch] )
                Freqs.append( freqs )
                Psd.append( psds )
                Subject.append( s )
                Task.append( k )
    spectral = {}
    spectral["frequency_sample"] = np.asarray( Freqs )
    spectral["power_spectral_density"] = np.asarray( Psd )
    spectral["Participant"] = np.asarray( Subject )
    spectral["Task"] = np.asarray( Task )
    return spectral


def freq_bands(spectral_dict, theta_lim=[4, 8], alpha_lim=[8, 13], beta_lim=[13, 30]):
    band_with = []
    for epoch in range( spectral_dict['power_spectral_density'].shape[0] ):
        freqs = spectral_dict['frequency_sample'][epoch]
        psd = spectral_dict['power_spectral_density'][epoch]

        alpha_id = np.logical_and( (freqs > alpha_lim[0]), (freqs < alpha_lim[1]) )
        beta_id = np.logical_and( (freqs > beta_lim[0]), (freqs < beta_lim[1]) )
        theta_id = np.logical_and( (freqs > theta_lim[0]), (freqs < theta_lim[1]) )

        alpha = psd[:, alpha_id].sum( axis=1 )
        beta = psd[:, beta_id].sum( axis=1 )
        theta = psd[:, theta_id].sum( axis=1 )

        band_with.append( np.asarray( [theta, alpha, beta] ) )

    band_with = np.asarray( band_with )
    return band_with


"""     IMAGE GENERATION        """


def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.
    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph( pos[0], pos[1], pos[2] )
    return pol2cart( az, m.pi / 2 - elev )


def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x ** 2 + y ** 2
    r = m.sqrt( x2_y2 + z ** 2 )  # r
    elev = m.atan2( z, m.sqrt( x2_y2 ) )  # Elevation
    az = m.atan2( y, x )  # Azimuth
    return r, elev, az


def pol2cart(theta, rho):
    """
    Transform polar coordinates to Cartesian
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    return rho * m.cos( theta ), rho * m.sin( theta )


def image_generation(feature_matrix, electrodes_loc, n_gridpoints):
    n_electrodes = electrodes_loc.shape[0]  # number of electrodes
    n_bands = feature_matrix.shape[1] // n_electrodes  # number of frequency bands considered in the feature matrix
    n_samples = feature_matrix.shape[0]  # number of samples to consider in the feature matrix.

    # Checking the dimension of the feature matrix
    if feature_matrix.shape[1] % n_electrodes != 0:
        print( 'The combination feature matrix - electrodes locations is not working.' )
    assert feature_matrix.shape[1] % n_electrodes == 0
    new_feat = []

    # Reshape a novel feature matrix with a list of array with shape [n_samples x n_electrodes] for each frequency band
    for bands in range( n_bands ):
        new_feat.append( feature_matrix[:, bands * n_electrodes: (bands + 1) * n_electrodes] )

    # Creation of a meshgrid data interpolation
    #   Creation of an empty grid
    grid_x, grid_y = np.mgrid[
                     np.min( electrodes_loc[:, 0] ): np.max( electrodes_loc[:, 0] ): n_gridpoints * 1j,  # along x_axis
                     np.min( electrodes_loc[:, 1] ): np.max( electrodes_loc[:, 1] ): n_gridpoints * 1j  # along y_axis
                     ]

    interpolation_img = []
    #   Interpolation
    #       Creation of the empty interpolated feature matrix
    for bands in range( n_bands ):
        interpolation_img.append( np.zeros( [n_samples, n_gridpoints, n_gridpoints] ) )
    #   Interpolation between the points
    # print('Signals interpolations.')
    for sample in tqdm( range( n_samples ) ):
        for bands in range( n_bands ):
            interpolation_img[bands][sample, :, :] = griddata( electrodes_loc, new_feat[bands][sample, :],
                                                               (grid_x, grid_y), method='cubic', fill_value=np.nan )
    #   Normalization - replacing the nan values by interpolation
    for bands in range( n_bands ):
        interpolation_img[bands][~np.isnan( interpolation_img[bands] )] = scale(
            interpolation_img[bands][~np.isnan( interpolation_img[bands] )] )
        interpolation_img[bands] = np.nan_to_num( interpolation_img[bands] )
    return np.swapaxes( np.asarray( interpolation_img ), 0, 1 )  # swap axes to have [samples, colors, W, H]


def band_image(frequency_band, electrodes_location, img_size=32):
    locs_2d = elec_proj( electrodes_location )
    frequency_band = frequency_band / np.min( frequency_band )
    frequency_band = frequency_band.reshape( (frequency_band.shape[0], -1) )

    images = image_generation( frequency_band, locs_2d, img_size )
    return images


"""     Graph Network rel.      """


def comp_distance(x1, x2):
    return np.sqrt( np.sum( (x1 - x2) ** 2 ) )

def comp_adjacency_mat(locs):
    Adjancency_mat = []
    for i in range(locs.shape[0]):
        tmp = []
        for j in range(locs.shape[0]):
            tmp.append(comp_distance(locs[i], locs[j]))
        Adjancency_mat.append(tmp)
    Adjancency_mat = np.asarray(Adjancency_mat)
    Adjancency_mat = 5 / (Adjancency_mat ** 2)  # 5 corresponding to delta parameters in RGNN P. Zhong et al. 2020
    Adjancency_mat[np.isinf(Adjancency_mat)] = 1
    Adjancency_mat = Adjancency_mat / np.max(Adjancency_mat)
    return Adjancency_mat

def A2edge_index(adjacency_mat):
    orig = []
    dest = []
    for i in range(adjacency_mat.shape[0]):
        for j in range(adjacency_mat.shape[0]):
            if (adjacency_mat[i, j] > np.mean(adjacency_mat)):
                orig.append(i)
                dest.append(j)
    edge_index = torch.tensor([orig, dest], dtype=torch.long)
    return edge_index

def Grap_Dataset(feat, label, edge_index):
    dataset = []
    for i in range(feat.shape[0]):
        dataset.append(Data(x=torch.tensor(feat[i], dtype=torch.float), edge_index=edge_index, y=label[i]))
    return dataset