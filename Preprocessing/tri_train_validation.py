
from xml.sax.handler import DTDHandler
import csv
import glob 
import json
import os
import random
data_json = open('resultat.csv','r')
data_ball = csv.reader(data_json, delimiter=',')
train_set = open('train_set.csv','w')
test_set = open('test_set.csv','w')
writer = csv.writer(train_set)
writer2 = csv.writer(test_set)
nbre_positif = 0
nbre_negatif = 0
for j in data_ball:
    if j != [] and int(j[1]) == 1:
        if random.random()>0.7:
            writer2.writerow([j[0],j[1]])               
        else:
            writer.writerow([j[0],j[1]])
        nbre_positif+=1
data_json.seek(0)
liste = []

for i in data_ball:
    if i != [] and int(i[1]) == 0:
        liste.append([i[0],i[1]])

random.shuffle(liste)
for i in liste:    
    if nbre_negatif<nbre_positif:
        if random.random()>0.7:
            writer2.writerow([i[0],i[1]])               
        else:
            writer.writerow([i[0],i[1]])
        nbre_negatif+=1
    else:
        break