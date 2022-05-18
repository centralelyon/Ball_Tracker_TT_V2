import cv2
import glob
import os

for i in glob.glob('*.jpg'):
    img = cv2.imread(i)
    resized_img = cv2.resize(img, (256,144))
    cv2.imwrite(f'imager/{os.path.basename(i)}', resized_img)

