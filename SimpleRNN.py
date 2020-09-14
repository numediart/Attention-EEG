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

Images = np.load('Dataset/freq_band.npy')
Label = np.load('Dataset/Label.npy')
participant = np.load('Dataset/Participant.npy')

""" Model Training """

# Introduction: training a simple CNN with the mean of the images.
batch_size = 64
n_epoch = 200
n_rep = 10

EEG = EEGImagesDataset(label=Label, image=Images)

t = time.time()

config = [
    [8, 1],
    [8, 2],
    [8, 4],
    [16, 1],
    [16, 2],
    [16, 4],
    [32, 1],
    [32, 2],
    [32, 4],
    [64, 1],
    [64, 2],
    [64, 4],
]


for c in range(len(config)):
    c+=7
    for patient in range(29):
        idx = np.argwhere(participant == patient)[:, 0]
        np.random.shuffle(idx)
        Test = Subset(EEG, idx)
        idx = np.argwhere(participant != patient)[:, 0]
        np.random.shuffle(idx)
        Train = Subset(EEG, idx)

        Trainloader = DataLoader(Train, batch_size=batch_size, shuffle=False)
        Testloader = DataLoader(Test, batch_size=batch_size, shuffle=False)

        net = RegionRNN(config[c][0], config[c][1], 3).cuda()
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

        score = np.asarray(score)
        np.save('res_freq_array/sub_'+str(c)+'_'+str(patient), score)
