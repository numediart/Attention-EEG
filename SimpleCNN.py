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

Images = np.load('Dataset/temp_img.npy')
Label = np.load('Dataset/Label.npy')
participant = np.load('Dataset/Participant.npy')

""" Model Training """

batch_size = 64
n_epoch = 200
n_rep = 10

EEG = EEGImagesDataset(label=Label, image=Images)

t = time.time()

config = [
    [8, 8, 16, 16],
    [8, 16, 16, 16],
    [8, 16, 16, 32],
    [8, 16, 32, 32],
    [8, 16, 32, 64],
    [16, 16, 32, 32],
    [16, 32, 32, 64],
    [16, 32, 64, 64],
    [16, 32, 64, 128],
    [32, 32, 64, 128],
    [32, 64, 64, 128],
    [32, 64, 128, 128],
    [32, 64, 128, 256],
    [64, 64, 128, 128],
    [64, 64, 128, 256],
    [64, 128, 128, 256],
    [64, 128, 256, 256],
]

Res = []
for c in range(len(config)):
    for patient in range(29):
        idx = np.argwhere(participant == patient)[:, 0]
        np.random.shuffle(idx)
        Test = Subset(EEG, idx)
        idx = np.argwhere(participant != patient)[:, 0]
        np.random.shuffle(idx)
        Train = Subset(EEG, idx)

        Trainloader = DataLoader(Train, batch_size=batch_size, shuffle=False)
        Testloader = DataLoader(Test, batch_size=batch_size, shuffle=False)

        net = SimpleCNN(in_dim=3, chan=config[c]).cuda()
        optimizer = optim.Adam(net.parameters())
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
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
        np.save('res_temp_img/sub_'+str(c)+'_'+str(patient), score)
