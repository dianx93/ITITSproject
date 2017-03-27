import cv2
import numpy as np

class HoG:
    def getHoG(self,image):
        #Why i need to divide by 255.0?
        temp = np.float32(image)

        gx = cv2.Sobel(temp, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(temp, cv2.CV_32F, 0, 1, ksize=1)

        mag, angle = cv2.cartToPolar(gx,gy,angleInDegrees=True)
        return np.concatenate(self.normalize(self.patch(temp,mag,angle/2.0)),axis=0)

    def patch(self, temp, mag, angle):
        bins = []

        for y_patch in range(temp.shape[0]/8):
            bins.append([])
            for x_patch in range(temp.shape[1]/8):
                bins[-1].append([0,0,0,0,0,0,0,0,0])
                for j in range(8*y_patch, 8*y_patch+8):
                    for i in range(8*x_patch, 8*x_patch+8):
                        maxMag = max(mag[j][i])
                        maxAngle = angle[j][i][np.where(mag[j][i] == maxMag)][0]
                        if maxAngle % 20 == 0:
                            bins[-1][-1][int(maxAngle // 20)%9] += mag[j][i]
                        else:
                            ratio = maxAngle / 20 - int(maxAngle / 20)
                            bins[-1][-1][(int(maxAngle / 20))%9] += ratio*maxMag
                            bins[-1][-1][(int(maxAngle / 20)+1)%9] += (1-ratio)*maxMag

        return np.array(bins)

    def normalize(self,patches):
        bins = []
        for y_patch in range(patches.shape[0] / 2-1):
            for x_patch in range(patches.shape[1] / 2-1):
                vec = np.concatenate((patches[y_patch][x_patch], patches[y_patch][x_patch+1],
                                     patches[y_patch+1][x_patch], patches[y_patch+1][x_patch+1]),axis = 0)


                norm = np.linalg.norm(np.linalg.norm(vec))
                vec = vec/ norm
                bins.append(vec)
        return np.array(bins)



hog = HoG()

image = cv2.imread("eagle-1.jpg")
print hog.getHoG(image).shape

