import numpy as np
import cv2

class SinglePixelVoting:
	
	def getSigns(self, image, ar,br,ab):
		redmask = np.zeros((image.shape[0],image.shape[1]))
		bluemask = np.zeros((image.shape[0],image.shape[1]))
		
		for y in range(image.shape[0]):
			for x in range(image.shape[1]):
				condition = image[y][x][2]/255.0 > ar*(image[y][x][1]+image[y][x][0])/255.0
				condition2 = image[y][x][2]/255.0 - max(image[y][x][1],image[y][x][0])/255.0 > br*(max(image[y][x][1],image[y][x][0]) - min(image[y][x][1],image[y][x][0]))/255.0
				redmask[y][x] = 255 if condition and condition2 else 0 
				
				bluecondition = image[y][x][0]/255.0 > ar*(image[y][x][1]+image[y][x][2])/255.0
				bluemask[y][x] = 1 if bluecondition else 0 
		
		return redmask,bluemask
		
		
spv = SinglePixelVoting()

image = cv2.imread("streets/9.png")

red,blue = spv.getSigns(image,0.35,0.65,0.0001)

red = red.astype(np.uint8)
tresh = cv2.cvtColor(red, cv2.COLOR_GRAY2BGR)
im2,contours,_ = cv2.findContours(red,1,2)

for i in contours:
	x,y,w,h = cv2.boundingRect(i)
	if w < 15 or h < 15:
		continue
	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
	
cv2.imshow("Test",image)

while True:
	k=cv2.waitKey(1) & 0xFF
	if k== 27: 
		break
cv2.destroyAllWindows()