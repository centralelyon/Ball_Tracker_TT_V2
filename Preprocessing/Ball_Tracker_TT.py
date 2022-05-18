from curses import erasechar
from logging import logMultiprocessing
from mailbox import NotEmptyError
from socket import NI_NOFQDN
from statistics import LinearRegression
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
from torchvision.io import read_image
from torch import nn
from torchsummary import summary
import joblib
from PIL import Image
import cv2
import optuna
from optuna.pruners import BasePruner
from optuna.trial._state import TrialState
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Definition de ma classe Dataset

class data_set(Dataset):
    def __init__(self,dir_path,transform=None):

        self.landmark = pd.read_csv(f"{dir_path}_tracker.csv")
        self.transform = transform
    
    def __len__(self):
        return len(self.landmark)

    def __getitem__(self, idx, device='cuda'):
        if torch.is_tensor(idx):
           idx = idx.tolist()
        img_name = self.landmark.iloc[idx,0]
        image = cv2.imread(os.path.join('imager',img_name))
        image = np.array(image)
        landmark = np.array([float(self.landmark.iloc[idx,1].replace(",",".")),float(self.landmark.iloc[idx,2].replace(",","."))])
        sample = {'image':image, 'coord':landmark}
        if self.transform:
            sample = self.transform(sample)
        sample['image'] = sample['image'].to(device)
        sample['coord'] = sample['coord'].to(device)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        coord = sample['coord']
        transform = transforms.ToTensor()
        image = transform(image)
        return {'image': image, 'coord': torch.from_numpy(coord).float()}

class Normalize(object):
    def __call__(self, sample):
        image, coord = sample['image'], sample['coord']
        normalizer = transforms.Normalize((0.0184,0.0184,0.0184),(0.1145,0.1145,0.1145))       
        image = normalizer(image)
        return {'image':image, 'coord': coord} 

# Normalisation simple on pourra penser a l'affiner plus tard
transform = transforms.Compose([ToTensor()])

data_train = DataLoader(data_set('train',transform), batch_size=10, shuffle=True)
data_test = DataLoader(data_set('test',transform), batch_size=10, shuffle=True)


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



model = NeuralNetwork().cuda()


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, X in enumerate (dataloader):
        target = torch.cuda.FloatTensor(X['coord']).squeeze(1)
        pred = model(X['image'])
        loss = loss_fn(pred,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch%10 == 0:
            loss, current = loss.item(), batch*len(X['image'])
            print(f"Test loss : {loss:>7f}  Avancement : [{current}/{size}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, X in enumerate (dataloader):
# Converting batch labels in a tensor 1D with good type (_int64)
            target = torch.cuda.FloatTensor(X['coord']).squeeze(1)
            pred = model(X['image'])
            test_loss += loss_fn(pred, target).item()
            correct+= loss_fn(pred, target).item()<=0.5
    test_loss /= num_batches
    correct /= size
    print(f'Avg loss on test_set.csv : {test_loss:>0f} Réussite : {correct} \n')
    return correct
lr= 1e-6
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.5)
for t in range(200):
    print(f"Epoch {t+1}\n----------------------")
    train_loop(data_train, model, loss_fn, optimizer)
    test_loop(data_train, model, loss_fn)
    torch.save(model.state_dict(), f'weights_track/model_weights_{t}.pth')

"""def train_glob(trial):
    model = NeuralNetwork().cuda()
    lr = trial.suggest_loguniform("lr",1e-5,1e-2)
    momentum = trial.suggest_uniform("momentum",0.4,0.99)
    #optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    epochs = 10
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()
    for t in range(epochs):
        print(f"Epoch {t+1}\n----------------------")
        train_loop(data_train, model, loss_fn, optimizer)
        torch.save(model.state_dict(), f'weights/model_weights_{t}.pth')
        accuracy = test_loop(data_train, model, loss_fn)
        trial.report(accuracy, step=epochs)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return accuracy    

sampler = optuna.samplers.TPESampler()

study = optuna.create_study(sampler=sampler, direction='maximize')
study.optimize(train_glob, n_trials=20, show_progress_bar=True)
joblib.dump(study, os.path.join(os.getcwd(),'test_optuna.pkl'))

study = joblib.load(os.path.join(os.getcwd(),'test_optuna.pkl'))
df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete','system_attrs'], axis=1)
df.head(5)

# Idée : Implémenter Momentum, standardize inputs & put BatchNormalization (eventually see dropout)"""