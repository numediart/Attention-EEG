import torch

import torch.nn as nn

from models import *
from utils import *
from torch.utils.tensorboard import SummaryWriter

import warnings

device = torch.device('cuda')
warnings.simplefilter("ignore")

""" Load Files """

Images = np.load('Dataset/Tot_EEG_Img_tot.npy')

""" Model Training """

# Introduction: training a simple CNN with the mean of the images.
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
    writer.add_scalar('Loss/train', running_loss, epoch)
    writer.add_scalar('Loss/test', validation_loss, epoch)