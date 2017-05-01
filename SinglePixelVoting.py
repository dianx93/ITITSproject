import numpy as np
import cv2

#For reference, see chapter 2.1 of http://perso.lcpc.fr/tarel.jean-philippe/publis/jpt-icpr10.pdf
class SinglePixelVoting:

	#Extracts the red part of the image with some tresholding.
	def getRedMask(self, image, ar, br):
		mask = np.zeros((image.shape[0],image.shape[1]))
		for y in range(image.shape[0]):
			for x in range(image.shape[1]):
				blue = image[y][x][0]
				green = image[y][x][1]
				red = image[y][x][2]

				#Note: division by 255.0 is important to prevent overflows
				if red/255.0 > ar*(green/255.0+blue/255.0):
					#Red color dominates
					if red/255.0 - max(green,blue)/255.0 > br*(max(green,blue) - min(green,blue))/255.0:
						#It's not too yellow or magenta
							mask[y][x] = 255
		return mask

	#TODO: doesn't work yet
	def getBlueMask(self, image, ar = 0.35):
		bluemask = np.zeros((image.shape[0], image.shape[1]))

		for y in range(image.shape[0]):
			for x in range(image.shape[1]):
				bluecondition = image[y][x][0] / 255.0 > ar * (image[y][x][1] + image[y][x][2]) / 255.0
				bluemask[y][x] = 1 if bluecondition else 0

		return bluemask


#Draws rectangles on the image. Does not draw rectangles with width/height smaller than minSize or rectangles
#that are inside a bigger rectangle.
def drawRectsOnImage(img, contours, minSize, color):
	rects = []

	for i in contours:
		x, y, w, h = cv2.boundingRect(i)
		if w < minSize or h < minSize:
			continue
		rects.append((x, y, w, h))

	rects.sort(key=lambda a: a[2] * a[3], reverse=True)

	for i, e in reversed(list(enumerate(rects))):
		if isContained(e, rects[:i]): del rects[i]

	for i in rects:
		#Include an offset to make it easier to see the inner area of the rect
		offset = 2

		w = i[2]
		h = i[3]

		rectWidth = 2
		if w < 24 or h < 24:
			rectWidth = 1

		cv2.rectangle(img, (i[0] - offset, i[1] - offset), (i[0] + i[2] + offset * 2, i[1] + i[3] + offset * 2), color, rectWidth)

	return img

def isContained(rect, others):
	for i in others:
		if rect[0] > i[0] and rect[1]>i[1] and \
			rect[0]+rect[2] < i[0]+i[2] and \
			rect[1]+rect[3] < i[1]+i[3]:
			return True
	return False


spv = SinglePixelVoting()

images = { 9 }
images = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }
showImages = False
saveToFiles = True

for imageId in images:
	print imageId
	image = cv2.imread("streets/" + str(imageId) + ".png")

	red = spv.getRedMask(image, 0.75, 1)

	red = red.astype(np.uint8)
	tresh = cv2.cvtColor(red, cv2.COLOR_GRAY2BGR)
	im2,redContours,_ = cv2.findContours(red,1,2)

	minRectSize = 10
	redSignsColor = [255, 255, 0]
	drawRectsOnImage(image, redContours, minRectSize, redSignsColor)

	if saveToFiles:
		cv2.imwrite("output/" + str(imageId) + ".png", image)

	if showImages:
		cv2.imshow("Test",image)

if showImages:
	while True:
		k=cv2.waitKey(1) & 0xFF
		if k== 27:
			break
cv2.destroyAllWindows()