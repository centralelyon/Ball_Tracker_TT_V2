from asyncio import base_tasks
import torch
import os
import csv
import cv2
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
 
        self.Sequence = nn.Sequential(

            # first convolution layer
            nn.Conv2d(3, 32 , (3,3), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout2d(),

            # second convolution layer
            nn.Conv2d(64, 128, (3,3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3,3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout2d(),

            # third convolution layer
            nn.Conv2d(128, 256, (3,3), padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3,3), padding='same'),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((2,2)),

            # fourth convolution layer
            nn.Conv2d(256, 512, (3,3), padding='same'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3,3), padding='same'),
            nn.BatchNorm2d(512),

            # 1D tensor with flatten
            nn.Flatten(),

            # depth layer
            nn.Linear(269824, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,2),
        )
    def forward(self, x):
        logits = self.Sequence(x)
        return logits


class data_set(Dataset):
    def __init__(self,dir_path,transform=None):

        self.landmark = pd.read_csv(f'{dir_path}.csv')
        self.transform = transform
    
    def __len__(self):
        return len(self.landmark)

    def __getitem__(self, idx, device='cuda'):
        if torch.is_tensor(idx):
           idx = idx.tolist()
        img_name = str(self.landmark.iloc[idx,0])
        image = cv2.imread(os.path.join('imager',img_name))
        image = np.array(image)
        sample = {'image': image}
        if self.transform:
            sample = self.transform(sample)
        sample['image'] = sample['image'].to(device)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        transform = transforms.ToTensor()
        image = transform(image)
        return {'image': image}

data = pd.read_csv('resultat_final.csv')
model = NeuralNetwork().cuda()
model.load_state_dict(torch.load(os.path.join(os.getcwd(),'weights_track','model_weights_67.pth')))
model.eval()
batch = DataLoader(data_set('resultat_final',ToTensor()),batch_size = 32, shuffle = False)
liste_x=[]
liste_y=[]
with torch.no_grad():
    for batch, X in enumerate (batch):
        pred = model(X['image'])
        pred = pred.cpu()
        liste_add = pred.tolist()
        liste_add = [[coord*7.5 for coord in elmt] for elmt in liste_add]
        for i in range (len(liste_add)):
            liste_x.append(liste_add[i][0])
            liste_y.append(liste_add[i][1])

data["X"] = liste_x
data["Y"] = liste_y 
data.to_csv("demo_tracker.csv", index=False)



