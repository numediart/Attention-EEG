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
from models import *
from utils import *
from graph_models import *

device = torch.device('cuda')
warnings.simplefilter("ignore")

""" Load Files """

# computation of adjacency matrix 
label = np.load('Dataset/... .npy')
locs_3d = np.load('Information/ChanInfo.npy', allow_pickle=True).all()['position']
A = comp_adjacency_mat(locs_3d)
edge_index = A2edge_index(A)
feat = np.load('Dataset/freq_band.npy').swapaxes(1,2)
participant = np.load('Dataset/Participant.npy')

""" Model Training """

batch_size = 64
n_epoch = 500
n_rep = 10
dataset = Grap_Dataset(feat=feat, label=label, edge_index=edge_index)

Res = []

config = [16, 32, 64, 128] #trying different hidden configuration

for c in range(len(config)):
    for patient in range(29):
        idx = np.argwhere(participant == patient)[:, 0]
        np.random.shuffle(idx)
        Test = Subset(dataset, idx)
        Testloader = DataLoader(Test, batch_size=128, shuffle=False)
        idx = np.argwhere(participant != patient)[:, 0]
        np.random.shuffle(idx)
        Train = Subset(dataset, idx)
        Trainloader = DataLoader(Train, batch_size=128, shuffle=False)
        net = GraphNet(dim=config[c]).cuda()

        optimizer = optim.Adam(net.parameters())
        criterion = nn.CrossEntropyLoss()


        validation_loss = 0.0
        validation_acc = 0.0
        score = []
        for epoch in range(n_epoch):
            running_loss = 0.0
            evaluation = []
            net.train()
            for i, data in enumerate(Trainloader, 0):
                # zero the parameter gradients
                optimizer.zero_grad()
                labels = torch.tensor(data.y, dtype=torch.long)

                # forward + backward + optimize
                outputs = net(data.to(torch.device('cuda')))
                loss = criterion(outputs, labels.to(torch.device('cuda')))
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.cpu().data, 1)
                num_of_true = (predicted==labels).sum().item()
                mean = num_of_true/labels.shape[0]
                evaluation.append(mean)

                running_loss += loss.item()
            running_loss = running_loss / (i + 1)
            running_acc = sum(evaluation) / len(evaluation)
            if True:
                net.eval()
                validation_loss = 0.0
                evaluation = []
                for i, data in enumerate(Testloader, 0):
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    labels = torch.tensor(data.y, dtype=torch.long)


                    # forward + backward + optimize
                    outputs = net(data.to(torch.device('cuda')))
                    validation_loss += criterion(outputs, labels.to(torch.device('cuda'))).item()

                    _, predicted = torch.max(outputs.cpu().data, 1)
                    num_of_true = (predicted == labels).sum().item()
                    mean = num_of_true / labels.shape[0]
                    evaluation.append(mean)

                validation_loss = validation_loss / (i + 1)
                validation_acc = sum(evaluation) / len(evaluation)
            score.append((running_loss, running_acc, validation_loss, validation_acc))
            print("Epoch {} \t---\tLoss {:.4f}, Accuracy {:.4f}\t---\tVal-Loss {:.4f}, Val-Accuracy {:.4f}" .format(epoch+1, running_loss, running_acc, validation_loss, validation_acc))
        score = np.asarray(score)
        np.save('res/res_graph_freq_sub_' + str(patient), score)
