import cv2
import glob
import os
import csv 

data_csv = open(os.path.join('fichiers','resultat_tracker.csv'),'r')
reader = csv.reader(data_csv, delimiter=';')

noms = {'haut_haut_gauche': {'x': {'x1': 480, 'x2': 960}, 'y': {'y1': 0, 'y2': 270}},'bas_gauche': {'x': {'x1': 480, 'x2': 960}, 'y': {'y1': 540, 'y2': 810}},'haut_gauche': {'x': {'x1': 480, 'x2': 960}, 'y': {'y1': 270, 'y2': 540}},'haut_droit': {'x': {'x1': 960, 'x2': 1440}, 'y': {'y1': 270, 'y2': 540}}}
for i in glob.glob('image\*.jpg'):
    chr = os.path.basename(i)
    elmt = 'image\.jpg'
    name = ''.join([i for i in chr if i not in elmt])
    img=cv2.imread(i)
    data_csv.seek(0)
    for line in reader:
        if line[1] == name:
            coord = [float(line[2].replace(',','.')),float(line[3].replace(',','.'))]
            break
        else:
            coord = []
    nb=0
    for j in noms:
        cropped_image = img[noms[j]['y']['y1']:noms[j]['y']['y2'],noms[j]['x']['x1']:noms[j]['x']['x2']]
        if os.path.isfile(os.path.join('Cropped',j, name+'.jpg'))==False:
            cv2.imwrite(os.path.join('Cropped',j, name+'.jpg'),cropped_image)
        data = open(os.path.join('Cropped',j,'resultat.csv'),'a')
        if coord !=[] and coord[0]>=noms[j]['x']['x1'] and coord[0]<=noms[j]['x']['x2'] and coord[1]>=noms[j]['y']['y1'] and coord[1]<=noms[j]['y']['y2']:
            balle=1
        else:
            balle=0
        writer = csv.writer(data)
        writer.writerow([name+'.jpg', balle, coord])