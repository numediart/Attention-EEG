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
import torch

import torch.nn as nn

from models import *
from utils import *
from torch.utils.tensorboard import SummaryWriter

import warnings

device = torch.device('cuda')
warnings.simplefilter("ignore")

""" Load Files """

Images = np.load('Dataset/....npy') #insert the file with EEG Images of the dataset

""" Model Training """

# Introduction: training an autoencoder based on CNN with EEG-based Images.
batch_size = 64
n_epoch = 500
n_rep = 10

Img = ImageAEDataset(Images)

lengths = [int(len(Img) * 0.8), int(len(Img) * 0.2)]
if sum(lengths) != len(Img):
    lengths[0] = lengths[0] + 1
Train, Test = random_split(Img, lengths)
Trainloader = DataLoader(Train, batch_size=batch_size, shuffle=False)
Testloader = DataLoader(Test, batch_size=batch_size, shuffle=False)

net = Autoencoder().cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3,
                             weight_decay=1e-5)
writer = SummaryWriter()
for epoch in range(n_epoch):
    running_loss = 0.0
    for i, data in enumerate(Trainloader, 0):
        img = data
        optimizer.zero_grad()

        output = net(img.to(torch.float32).cuda())
        loss = criterion(output, img.to(torch.float32).cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    running_loss = running_loss/(i+1)

    validation_loss = 0.0
    for i, data in enumerate(Testloader, 0):
        img = data

        output = net(img.to(torch.float32).cuda())
        loss = criterion(output, img.to(torch.float32).cuda())

        validation_loss += loss.item()

    validation_loss = validation_loss/(i+1)
    print("Epoch {} \t---\tLoss {:.4f}, \tVal-Loss {:.4f}" .format(epoch+1, running_loss, validation_loss, ))
        
    writer.add_scalar('Loss/train', running_loss, epoch)
    writer.add_scalar('Loss/test', validation_loss, epoch)