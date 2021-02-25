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
from models import *
import torch
import torch.optim as optim
import time

from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split, Subset

import warnings

device = torch.device('cuda')
warnings.simplefilter("ignore")

""" Load Files """

Images = np.load('Dataset/... .npy')
Label = np.load('Dataset/ ...npy')
participant = np.load('Dataset/ ...npy')

""" Model Training """

# Introduction: training a simple CNN with the mean of the images.
batch_size = 64
n_epoch = 200
n_rep = 10

EEG = EEGImagesDataset(label=Label, image=Images)

t = time.time()

for patient in range(29):
    idx = np.argwhere(participant == patient)[:, 0]
    np.random.shuffle(idx)
    Test = Subset(EEG, idx)
    idx = np.argwhere(participant != patient)[:, 0]
    np.random.shuffle(idx)
    Train = Subset(EEG, idx)

    Trainloader = DataLoader(Train, batch_size=batch_size, shuffle=False)
    Testloader = DataLoader(Test, batch_size=batch_size, shuffle=False)

    net = RegionRNN(64, 4, 3).cuda()
    optimizer = optim.Adam(net.parameters())

    score = []
    for epoch in range(n_epoch):
        running_loss = 0.0
        evaluation = []
        for i, data in enumerate(Trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            del data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(torch.float32).cuda())
            loss = torch.nn.functional.cross_entropy(outputs, labels.to(torch.long).cuda())
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            num_of_true = torch.sum(predicted.detach().cpu()==labels).numpy()
            mean = num_of_true/labels.shape[0]
            running_loss += loss.item()
            evaluation.append(mean)

        running_loss = running_loss / (i + 1)
        running_acc = sum(evaluation) / len(evaluation)

        validation_loss = 0.0
        validation_acc = 0.0
        evaluation = []
        for i, data in enumerate(Testloader, 0):
            input_img, labels = data
            del data
            input_img = input_img.to(torch.float32)
            if True:
                input_img = input_img.cuda()
            outputs = net(input_img)
            loss = torch.nn.functional.cross_entropy(outputs, labels.cuda())
            validation_loss += loss.item()

            _, predicted = torch.max(outputs.cpu().data, 1)
            num_of_true = torch.sum(predicted == labels).numpy()

            evaluation.append(num_of_true/labels.shape[0])

        validation_loss = validation_loss / (i + 1)
        validation_acc = sum(evaluation) / len(evaluation)
        score.append((running_loss, running_acc, validation_loss, validation_acc))
        print("Epoch {} \t---\tLoss {:.4f}, Accuracy {:.4f}\t---\tVal-Loss {:.4f}, Val-Accuracy {:.4f}" .format(epoch+1, running_loss, running_acc, validation_loss, validation_acc))
    

    score = np.asarray(score)
    np.save('res_freq_array_sub_'+str(patient), score)