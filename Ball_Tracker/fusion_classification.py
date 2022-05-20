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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class data_set(Dataset):
    def __init__(self,dir_path,transform=None):
        
        self.dir_path = dir_path
        self.landmark = pd.read_csv(os.path.join('cropped_test',dir_path,'resultat.csv'))
        self.transform = transform
    
    def __len__(self):
        return len(self.landmark)

    def __getitem__(self, idx, device='cuda'):
        if torch.is_tensor(idx):
           idx = idx.tolist()
        img_name = self.landmark.iloc[idx,0]
        image = cv2.imread(os.path.join('cropped_test',self.dir_path,img_name))
        image = np.array(image)
        sample = {'image':image, 'nom':img_name}
        if self.transform:
            sample = self.transform(sample)
        sample['image'] = sample['image'].to(device)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, name = sample['image'], sample['nom']
        transform = transforms.ToTensor()
        image = transform(image)
        return {'image': image, 'nom': name}

class Normalize(object):
    def __call__(self, sample):
        image, name = sample['image'], sample['nom']
        normalizer = transforms.Normalize((0.0184,0.0184,0.0184),(0.1145,0.1145,0.1145))       
        image = normalizer(image)
        return {'image':image, 'nom': name} 

# Normalisation simple on pourra penser a l'affiner plus tard

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


class data_set_track(Dataset):
    def __init__(self,dir_path,transform=None):

        self.dir_path = dir_path
        self.landmark = pd.read_csv(f"resultats_{dir_path}.csv")
        self.transform = transform
    
    def __len__(self):
        return len(self.landmark)

    def __getitem__(self, idx, device='cuda'):
        if torch.is_tensor(idx):
           idx = idx.tolist()
        img_name = self.landmark.iloc[idx,1]
        image = cv2.imread(os.path.join('cropped_test',self.dir_path,img_name))
        image = np.array(image)
        sample = {'image':image, 'nom':img_name}
        if self.transform:
            sample = self.transform(sample)
        sample['image'] = sample['image'].to(device)
        return sample

class ToTensor_track(object):
    def __call__(self, sample):
        image = sample['image']
        name = sample['nom']
        transform = transforms.ToTensor()
        image = transform(image)
        return {'image': image, 'nom': name}

class Normalize_track(object):
    def __call__(self, sample):
        image, nom = sample['image'], sample['nom']
        normalizer = transforms.Normalize((0.0801,0.0801,0.0801),(0.2228,0.2228,0.2228))       
        image = normalizer(image)
        return {'image':image, 'nom': nom} 

class NeuralNetworkTrack(nn.Module):
    def __init__(self):
        super(NeuralNetworkTrack, self).__init__()
 
        self.Sequence = nn.Sequential(

            # first convolution layer
            nn.Conv2d(3, 32 , (3,3), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # second convolution layer
            nn.Conv2d(64, 128, (3,3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout2d(),
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

if __name__ == "__main__":
    liste_fichier = ['bas_gauche','haut_droit','haut_gauche','haut_haut_gauche']
    list_tableau= [open('resultats_bas_gauche.csv','w'),open('resultats_haut_droit.csv','w'),open('resultats_haut_gauche.csv','w'),open('resultats_haut_haut_gauche.csv','w')]
    dict_writer ={}
    dict_inter = {}
    resultat_final = open('resultats_coord.csv','w')
    writer_final = csv.writer(resultat_final)
    writer_final.writerow(['image', 'pixelx', 'pixely'])
    non_classe = open('non_classe.csv','w')
    write_non_classe = csv.writer(non_classe)
    transform = transforms.Compose([ToTensor()])
    transform_track = transforms.Compose([ToTensor_track(), Normalize_track()])
    noms = {'haut_haut_gauche': {'x': {'x1': 480, 'x2': 960}, 'y': {'y1': 0, 'y2': 270}},'bas_gauche': {'x': {'x1': 480, 'x2': 960}, 'y': {'y1': 540, 'y2': 810}},'haut_gauche': {'x': {'x1': 480, 'x2': 960}, 'y': {'y1': 270, 'y2': 540}},'haut_droit': {'x': {'x1': 960, 'x2': 1440}, 'y': {'y1': 270, 'y2': 540}}}
    for i in range (len(liste_fichier)):
        dict_writer[liste_fichier[i]] = csv.writer(list_tableau[i])
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
                    if X['nom'][j] not in dict_resultat.keys():
                        dict_resultat[X['nom'][j]] = [i]+pred[j].tolist()
                    if pred[j].tolist()[1]> dict_resultat[X['nom'][j]][2]:
                        dict_resultat[X['nom'][j]] = [i]+pred[j].tolist()
    for i in dict_resultat:
        if dict_resultat[i][2]>0:
            dict_writer[dict_resultat[i][0]].writerow([dict_resultat[i][0], i, dict_resultat[i][1], dict_resultat[i][2]])
        else:
            write_non_classe.writerow([i])
    print("Classification Effectuee")
    list_tableau[3].close()
    for i in liste_fichier:
        model = NeuralNetworkTrack().cuda()
        model.load_state_dict(torch.load(os.path.join(i,'weights_track','best_model.pth')))
        model.eval()
        print(f"resultats_{i}")
        batch = DataLoader(data_set_track(i,transform), batch_size=10, shuffle = False)
        with torch.no_grad():
            for batch, X in enumerate(batch):
                print(i)
                pred = model(X['image'])
                pred = pred.cpu()
                pred = pred.numpy()
                for j in range (len(pred)):
                    pred[j][0] = (pred[j][0]*1.875) + noms[i]['x']['x1']
                    pred[j][1] = (pred[j][1]*1.875) + noms[i]['y']['y1']
                    writer_final.writerow([X['nom'][j], pred[j][0], pred[j][1]])
    print("Predictions Effectuees")