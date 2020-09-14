import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch_geometric.nn import SGConv, global_add_pool
from torch_scatter import scatter_add
from torch.utils.data import DataLoader,random_split
from utils import *


### DATASET CLASS ###
class EEGImagesDataset(Dataset):
    """EEGLearn Images Dataset from EEG."""

    def __init__(self, label, image):
        self.label = label
        self.Images = image

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.Images[idx]
        label = self.label[idx]
        sample = (image, label)

        return sample


class ImageAEDataset(Dataset):
    def __init__(self, image):
        self.Images = image

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.Images[idx]
        return image


### MODELS ###
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder
        self.Encoder = nn.Sequential(
            nn.Conv2d(401, 256, kernel_size=3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=1),

            Flatten(),

            nn.Linear(128, 16)
        )
        # decoder
        self.Decoder = nn.Sequential(
            nn.Linear(16, 128),

            UnFlatten(),

            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(256, 401, kernel_size=5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(401, 401, kernel_size=2, stride=2, padding=2)
        )

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)

        return x


class SimpleCNN(nn.Module):

    def __init__(self, in_dim, chan=[8, 16, 32, 64], input_image=torch.zeros(1, 3, 32, 32), kernel=(3, 3), stride=1,
                 padding=1, max_kernel=(2, 2),
                 n_classes=4):
        super(SimpleCNN, self).__init__()

        self.ClassifierCNN = nn.Sequential(
            nn.Conv2d(in_dim, chan[0], kernel_size=3),
            nn.BatchNorm2d(chan[0]),
            nn.ReLU(),

            nn.Conv2d(chan[0], chan[1], kernel_size=3),
            nn.ReLU(),

            nn.Conv2d(chan[1], chan[1], kernel_size=3),
            nn.BatchNorm2d(chan[1]),
            nn.ReLU(),

            nn.Conv2d(chan[1], chan[2], kernel_size=3),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(chan[2], chan[2], kernel_size=3),
            nn.BatchNorm2d(chan[2]),
            nn.ReLU(),

            nn.Conv2d(chan[2], chan[3], kernel_size=3),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(chan[3], chan[3], kernel_size=3),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),

            Flatten(),
            nn.Linear(chan[3], 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

        )

        self.ClassifierFC = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 4),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.ClassifierCNN(x)
        x = self.ClassifierFC(x.view(x.shape[0], -1))
        return x


class RegionRNN( nn.Module ):
    def __init__(self, h_size, n_layer, in_size, b_first=False, bidir=False):
        super( RegionRNN, self ).__init__()

        self.hidden_size = h_size
        self.num_layers = n_layer
        self.input_size = in_size

        self.dict = {'Fr1': np.array( [0, 2, 3] ), 'Fr2': np.array( [30, 28, 29] ),
                     'Tp1': np.array( [4, 8, 9, 13] ),'Tp2': np.array( [25, 24, 19, 18] ),
                     'Cn1': np.array( [5, 6, 7, 10, 11] ), 'Cn2': np.array( [26, 27, 23, 20, 21] ),
                     'Pr1': np.array( [12] ), 'Pr2': np.array( [17] ),
                     'Oc1': np.array( [14] ), 'Oc2': np.array( [16] )}

        self.batch_first = b_first
        self.bidirectional = bidir

        self.RNN_fL = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )
        self.RNN_fR = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )

        self.RNN_f = nn.RNN( self.hidden_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                             bidirectional=self.bidirectional )

        self.RNN_tL = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )
        self.RNN_tR = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )

        self.RNN_t = nn.RNN( self.hidden_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                             bidirectional=self.bidirectional )

        self.RNN_pL = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )
        self.RNN_pR = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )

        self.RNN_p = nn.RNN( self.hidden_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                             bidirectional=self.bidirectional )

        self.RNN_oL = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )
        self.RNN_oR = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )

        self.RNN_o = nn.RNN( self.hidden_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                             bidirectional=self.bidirectional )

        self.fc_f = nn.Sequential(
            nn.Linear( 3 * self.hidden_size, 16 ),
            nn.ReLU(),
        )

        self.fc_t = nn.Sequential(
            nn.Linear( 4 * self.hidden_size, 16 ),
            nn.ReLU(),
        )

        self.fc_p = nn.Sequential(
            nn.Linear( 1 * self.hidden_size, 16 ),
            nn.ReLU(),
        )

        self.fc_o = nn.Sequential(
            nn.Linear( 1 * self.hidden_size, 16 ),
            nn.ReLU(),
        )

        self.fc_final = nn.Sequential(
            nn.Linear(64, 4),
            nn.Softmax()
        )

        self.b_n1 = nn.BatchNorm2d( 3 )
        self.b_n2 = nn.BatchNorm1d( 64 )

    def forward(self, x):
        # Set initial states
        self.batch_size = x.shape[0]

        x = self.b_n1( x.permute( 0, 2, 1 ).reshape( x.shape[0], 3, 1, -1 ) )[:, :, 0].permute( 0, 2, 1 )

        h0 = torch.zeros( self.num_layers, self.batch_size, self.hidden_size ).cuda()
        k = list( self.dict.keys() )

        fr_l = x[:, self.dict[k[0]]].permute( 1, 0, 2 )
        fr_r = x[:, self.dict[k[1]]].permute( 1, 0, 2 )

        tp_l = x[:, self.dict[k[2]]].permute( 1, 0, 2 )
        tp_r = x[:, self.dict[k[2]]].permute( 1, 0, 2 )

        p_l = x[:, self.dict[k[6]]].permute( 1, 0, 2 )
        p_r = x[:, self.dict[k[7]]].permute( 1, 0, 2 )

        o_l = x[:, self.dict[k[8]]].permute( 1, 0, 2 )
        o_r = x[:, self.dict[k[9]]].permute( 1, 0, 2 )

        x_fl, _ = self.RNN_fL( fr_l, h0 )
        x_fr, _ = self.RNN_fR( fr_r, h0 )

        x_tl, _ = self.RNN_tL( tp_l, h0 )
        x_tr, _ = self.RNN_tR( tp_r, h0 )

        x_pl, _ = self.RNN_tL( p_l, h0 )
        x_pr, _ = self.RNN_tR( p_r, h0 )

        x_ol, _ = self.RNN_oL( o_l, h0 )
        x_or, _ = self.RNN_oR( o_r, h0 )

        x_f = x_fr - x_fl
        x_t = x_tr - x_tl
        x_p = x_pr - x_pl
        x_o = x_or - x_ol

        x_f, _ = self.RNN_f( x_f, h0 )
        x_t, _ = self.RNN_f( x_t, h0 )
        x_p, _ = self.RNN_p( x_p, h0 )
        x_o, _ = self.RNN_o( x_o, h0 )

        x_f = x_f.permute( 1, 0, 2 )
        x_t = x_t.permute( 1, 0, 2 )
        x_p = x_p.permute( 1, 0, 2 )
        x_o = x_o.permute( 1, 0, 2 )

        x = torch.cat(
            (self.fc_f( x_f.reshape( self.batch_size, -1 ) ), self.fc_t( x_t.reshape( self.batch_size, -1 ) ),
             self.fc_p( x_p.reshape( self.batch_size, -1 ) ), self.fc_o( x_o.reshape( self.batch_size, -1 ) )), dim=1 )

        x = self.b_n2( x )
        x = x.reshape( self.batch_size, -1 )

        x = self.fc_final(x)

        return x



### MISCELANEOUS ###
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), input.size(1), 1, -1)
