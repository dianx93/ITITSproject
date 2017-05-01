aspectratios = []
import cv2, os
path = "GTSRB/Final_Training/Images/"
for folder in os.listdir(path):
	files = os.listdir(path + folder)
	for i in files:
		img = cv2.imread(path + folder + "/" + i)
		if i.endswith("ppm"):
			aspectratios.append(float(img.shape[0])/img.shape[1])
			
print sum(aspectratios)/len(aspectratios)
			
