import os, random, cv2
import numpy as np
from image_classifications import classifications,classifications2

#Extracts the image metadata from the .csv files (Bounding box of the traffic sign)
def addImageDetails(file, images):
    f = open(file)
    data = f.read().split("\n")[1:-1]
    f.close()
    for i in data:
        name,width,height,X1,Y1,X2,Y2,CID = i.split(";")
        if name in images:
            images[name]["WIDTH"] = int(width)
            images[name]["HEIGHT"] = int(height)
            images[name]["X1"] = int(X1)
            images[name]["Y1"] = int(Y1)
            images[name]["X2"] = int(X2)
            images[name]["Y2"] = int(Y2)
            images[name]["CID"] = int(CID)

import HoG
def readTrainingImages(path, percentage,classifiers):
    images = {}
    hog = HoG.HoG()
    for folder in os.listdir(path):
        folder_index = int(folder)
        sign_type = classifiers[folder_index]
        files = os.listdir(path + folder)
        # print("Including %d images from %s" % (int(len(files) * percentage), folder))
        # Read the .ppm file containing the images
        # Warning: 1 less image might be loaded if the .csv file is included in the percentage of files
        for i in range(0, int(len(files) * percentage)):
            file = files[i]
            if file.endswith("ppm"):
                images[file] = {"IMAGE": cv2.imread(path + folder + "/" + file)}
                images[file]["SIGNTYPE"] = sign_type
				#Calculates HoG for rescaled image. 
                images[file]["HoG"] = hog.getHoG(images[file]["IMAGE"],(64,64))

        # Read the .csv file containing the bounding boxes of traffic signs
        for file in files:
            if file.endswith("csv"):
                addImageDetails(path + folder + "/" + file, images)
                break
    return images
from sklearn import svm
import time
import TrafficSignExtractor
import pickle
if __name__ == "__main__":

    #How many images from each training set should be included, in range (0, 1].
    percentage = 1.0

   # path = "train/Images/"
    path = "GTSRB/Final_Training/Images/"
    path2 = "BelgianTS/Training/"
    pathImages = "streets/"

  #  images = readTrainingImages(path, percentage, classifications)
  #  images.update(readTrainingImages(path2,percentage,classifications2))
  #  pickle.dump(images,open("Data.txt","wb"))

    images = pickle.load(open("Data.txt","rb"))

    #print images
    #A dataset is a dictionary-like object that holds all the data and some metadata about the data.
    # This data is stored in the .data member, which is a n_samples, n_features array.
    # In the case of supervised problem, one or more response variables are stored in the .target member.
    dataset = {}

    #dataset.data should be a 2D array where first index is the n'th image and each image maps
    #to the features of the image. I guess this is a 1D array of HoG features? 
    #dataset.target should be a 1D array where n'th element is the expected categorization of the n'th image
	
	#DEMO of learning
    data = []
    target = []
    width = 0
    height = 0

    for i in images:
		data.append(images[i]["HoG"].tolist())
		target.append(images[i]["SIGNTYPE"])

    for i in reversed(range(len(data))):
        if np.any(np.isnan(data[i])):
            del data[i]
            del target[i]
    svm = svm.SVC(gamma=0.001,C=100.)
    svm.fit(np.array(data),np.array(target))
    hog = HoG.HoG()
    for j in range(1,31):
        if j == 12 or j==18: continue
        test = cv2.imread(pathImages + str(j)+".png")
        svp = TrafficSignExtractor.TrafficSignExtractor()
        signs = svp.getTrafficSigns(test,5,50)
        predictions = []
        for i in signs:
            subimage = test[i[1]-10:i[1]+i[3]+10, i[0]-10:i[0]+i[2]+10]
            prediction =  svm.predict(hog.getHoG(subimage,(64,64)))
            predictions.append((i[0]-10, i[1]-10, prediction[0]))

        for i in predictions:
            cv2.putText(test,i[2],(i[0],i[1]),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255))

        cv2.imwrite("result/"+ str(j) + ".png", test)
        cv2.imshow(str(j), test)

    #OLD: Open a random image for viewing
  #  image = random.choice(images.keys())
	
  #  img = images[image]
  #  cv2.rectangle(img["IMAGE"],(img["Y1"],img["X1"]),(img["Y2"],img["X2"]),(255,0,0),1)
  #  cv2.imshow("Test",img["IMAGE"])

    while True:
       k=cv2.waitKey(1) & 0xFF
       if k== 27: break
    cv2.destroyAllWindows()
