import csv
import random
data_tracker = open('resultat_tracker.csv','r')
reader = csv.reader(data_tracker,delimiter=';')

data_train = open('train_tracker.csv','w')
writer = csv.writer(data_train)

data_test = open('test_tracker.csv','w')
writer2 = csv.writer(data_test)

for i in reader:
    if i[2]!='': # On prend uniquement les images annotées / Par la suite cette detection automatique se fera au moyen du premier réseau de neurones
        a=[i[1]+'.jpg',i[2],i[3]]
        if random.random()>0.7:
            writer2.writerow(a)
        else:
            writer.writerow(a)

            
