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
            img = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
            x,y = float(j[1]),float(j[2])
            sizes = stats[1:, -1]
            x_rond, y_rond = round(x), round(y)
            if output[y_rond,x_rond] != 0 and sizes[output[y_rond,x_rond]-1]>=200 and sizes[output[y_rond,x_rond]-1]<=1000:
                x,y = centroids[output[y_rond,x_rond]]
            elif output[y_rond,x_rond] != 0 and sizes[output[y_rond,x_rond]-1]>1000:
                x,y = x_rond,y_rond
            else: 
                mark=True
                k=1
                while mark == True:
                    for m in (y_rond-k, y_rond+k):
                        if m <= len(output)-1 and m>=-len(output):    
                            for l in range (x_rond-k, x_rond+k):
                                if l <= len(output[0])-1 and l>=len(output[0]):
                                    if output[m,l] != 0 and sizes[output[m,l]-1]>=200 and sizes[output[m,l]-1]<=1000:
                                        mark=False
                                        x,y = centroids[output[m,l]]
                                        break
                                    elif output[m,l] != 0 and sizes[output[m,l]-1]>1000:
                                        mark=False
                                        x,y = l,m
                                        break
                        if mark == False:
                            break
                    for l in (x_rond-k,x_rond+k):
                        if l <= len(output[0])-1 and l>=-len(output[0]):    
                            for m in range (y_rond-k, y_rond+k):
                                if m <= len(output)-1 and m>=-len(output):
                                    if output[m,l] != 0 and sizes[output[m,l]-1]>=200 and sizes[output[m,l]-1]<=1000:
                                        mark=False
                                        x,y = centroids[output[m,l]]
                                        break
                                    elif output[m,l] != 0 and sizes[output[m,l]-1]>1000:
                                        mark=False
                                        x,y = l,m
                                        break
                        if mark == False:
                            break
                    k+=1
            img_print = cv2.imread(os.path.join('D:\Ball_Tracker_TT_V2\image',f'non_annote\RGB{j[0]}'))
            img_print = cv2.circle(img_print,(round(x),round(y)),3,(0,0,255),6)
            print(j[0])
            cv2.imwrite(f'image_result\{j[0]}',img_print)
