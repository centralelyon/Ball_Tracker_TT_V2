from re import T
import cv2
import glob
import csv
import os
coord = open('resultats_coord.csv','r')
reader = csv.reader(coord)
for i in glob.glob(os.path.join('D:\Ball_Tracker_TT_V2\image','non_annote\*.jpg')):
    coord.seek(0)
    for j in reader:
        chr = i.split("\\")
        string = chr[len(chr)-1]
        if j!=[] and j[0]!='image' and j[0] == string :
            img = cv2.imread(i)
            img = cv2.circle(img,(round(float(j[1])),round(float(j[2]))),3,(0,0,255),6)
            cv2.imwrite(f'image_result\{j[0]}',img)
