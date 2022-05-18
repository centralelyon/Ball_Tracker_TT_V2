import torch
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot
import os
import pandas as pd
import glob
from torch import nn
from PIL import Image
import cv2
import csv 

torch.cuda.empty_cache()
data_final = open('resultat_final.csv', 'w')
writer_final = csv.writer(data_final)
data_csv = open('resultats.csv','w')
writer = csv.writer(data_csv)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class data_set(Dataset):
    def __init__(self,dir_path,transform=None):
        
        self.dir_path = dir_path
        self.landmark = pd.read_csv(os.path.join(dir_path,'resultat.csv'))
        self.transform = transform
    
    def __len__(self):
        return len(self.landmark)

    def __getitem__(self, idx, device='cuda'):
        if torch.is_tensor(idx):
           idx = idx.tolist()
        img_name = self.landmark.iloc[idx,0]
        image = cv2.imread(os.path.join(self.dir_path,'imager',img_name))
        image = np.array(image)
        landmark = self.landmark.iloc[idx,1]
        landmark = np.array([landmark])
        sample = {'image':image, 'landmark':landmark, 'nom':img_name}
        if self.transform:
            sample = self.transform(sample)
        sample['image'] = sample['image'].to(device)
        sample['landmark'] = sample['landmark'].to(device)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, landmark, name = sample['image'], sample['landmark'], sample['nom']
        transform = transforms.ToTensor()
        image = transform(image)
        return {'image': image, 'landmark': torch.from_numpy(landmark), 'nom': name}

class Normalize(object):
    def __call__(self, sample):
        image, landmark, name = sample['image'], sample['landmark'], sample['nom']
        normalizer = transforms.Normalize((0.0184,0.0184,0.0184),(0.1145,0.1145,0.1145))       
        image = normalizer(image)
        return {'image':image, 'landmark': landmark, 'nom': name} 

# Normalisation simple on pourra penser a l'affiner plus tard

transform = transforms.Compose([ToTensor()])

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


            # 1D tensor with flatten
            nn.Flatten(),

            # depth layer
            nn.Linear(134912, 1024),
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



if __name__ == "__main__":
    liste_fichier = ['bas_gauche','haut_droit','haut_gauche','haut_haut_gauche']
    dict_class = {}
    dict_resultat = {}
    for i in liste_fichier:
        idx = 0
        model = NeuralNetwork().cuda()
        model.load_state_dict(torch.load(os.path.join(i,'weights','best_model.pth')))
        model.eval()
        batch = DataLoader(data_set(i,transform), batch_size=32, shuffle = False)
        with torch.no_grad():
            for batch, X in enumerate(batch):
                pred = model(X['image'])
                pred = pred.cpu()
                pred = pred.numpy()
                for j in range (len(pred)):
                    if X['nom'][j] not in dict_class.keys():
                        dict_class[X['nom'][j]] = list()
                        dict_resultat[X['nom'][j]] = [i]+pred[j].tolist()
                    dict_class[X['nom'][j]].append([i]+pred[j].tolist())
                    if pred[j].tolist()[1]> dict_resultat[X['nom'][j]][2]:
                         dict_resultat[X['nom'][j]] = [i]+pred[j].tolist()                       
    for i in dict_class:
        writer.writerow([i,dict_class[i][0],dict_class[i][1]])
        writer_final.writerow([i,dict_resultat[i][0])
