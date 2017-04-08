import os
import cv2
import numpy as np
from image_classification import classification

path = "GTSRB/Final_Training/Images/"

images = {}

def add_image_details(file):
    f = open(file)
    data = f.read().split("\n")[1:-1]
    f.close()
    for i in data:
        name,width,height,X1,Y1,X2,Y2,CID = i.split(";")
        images[name]["WIDTH"] = int(width)
        images[name]["HEIGHT"] = int(height)
        images[name]["X1"] = int(X1)
        images[name]["Y1"] = int(Y1)
        images[name]["X2"] = int(X2)
        images[name]["Y2"] = int(Y2)
        images[name]["CID"] = int(CID)

for folder in os.listdir(path):
    folder_index = int(folder)
    sign_type = classification[folder_index]
    for file in os.listdir(path + folder):
        if file.endswith("ppm"):
            images[file] = {"IMAGE":cv2.imread(path + folder+ "/" +file)}
            images[file]["SIGNTYPE"] = sign_type
        else:
            add_image_details(path + folder+ "/" +file)


import random
image = random.choice(images.keys())

img = images[image]
cv2.rectangle(img["IMAGE"],(img["X1"],img["Y1"]),(img["X2"],img["Y2"]),(255,0,0),3)
cv2.imshow("Test",img["IMAGE"])

while True:
    k=cv2.waitKey(1) & 0xFF
    if k== 27: break
cv2.destroyAllWindows()