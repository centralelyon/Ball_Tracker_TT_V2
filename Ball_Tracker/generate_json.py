import json
import csv

data = open('resultats_coord.csv','r')
reader = csv.reader(data,delimiter=',')
fich_json = open('ball_marker.json','w')
data_add = {}
for i in reader:
    if i!=[] and i[0]!='image':
        chr = '.jpg'
        str = ''.join(k for k in i[0] if k not in chr)
        data_add[str]={"x" : float(i[1]),"y" : float(i[2])}
json.dump(data_add,fich_json)