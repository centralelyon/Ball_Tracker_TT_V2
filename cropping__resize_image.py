import cv2
import glob
import os
import csv 



noms = {'haut_haut_gauche': {'x': {'x1': 480, 'x2': 960}, 'y': {'y1': 0, 'y2': 270}},'bas_gauche': {'x': {'x1': 480, 'x2': 960}, 'y': {'y1': 540, 'y2': 810}},'haut_gauche': {'x': {'x1': 480, 'x2': 960}, 'y': {'y1': 270, 'y2': 540}},'haut_droit': {'x': {'x1': 960, 'x2': 1440}, 'y': {'y1': 270, 'y2': 540}}}
for i in glob.glob(os.path.join('image','non_annote','*.jpg')):
    chr = os.path.basename(i)
    elmt = 'image\non_annote\.jpg'
    name = ''.join([i for i in chr if i not in elmt])
    img=cv2.imread(i)
    for j in noms:
        cropped_image = img[noms[j]['y']['y1']:noms[j]['y']['y2'],noms[j]['x']['x1']:noms[j]['x']['x2']]
        cropped_image = cv2.resize(cropped_image, (256,144))
        if os.path.isfile(os.path.join('Cropped','cropped_test',j, name+'.jpg'))==False:
            cv2.imwrite(os.path.join('Cropped','cropped_test',j, name+'.jpg'),cropped_image)
        data = open(os.path.join('Cropped','cropped_test',j,'resultat.csv'),'a')
        writer = csv.writer(data)
        writer.writerow([name+'.jpg'])