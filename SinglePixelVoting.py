import numpy as np
import cv2, math

#For reference, see chapter 2.1 of http://perso.lcpc.fr/tarel.jean-philippe/publis/jpt-icpr10.pdf
class SinglePixelVoting:

	def getTrafficSigns(self,image, minSize):
		minRectSize = minSize
		redSignsColor = [255, 255, 0]
		blueSignsColor = [0, 255, 255]
		
		#image = cv2.imread("streets/" + str(imageId) + ".png")

		redMask = self.getRedMask(image, 0.75, 1).astype(np.uint8)
		#tresh = cv2.cvtColor(redMask, cv2.COLOR_GRAY2BGR)
		im2,redContours,_ = cv2.findContours(redMask,1,2)
		redContours = self.removeBadContours(redContours, minRectSize)
		img, possibleSigns = self.drawRectsOnImage(image, redContours, minRectSize, redSignsColor)
		#image = cv2.drawContours(image, redContours, -1, redSignsColor, 1)
		
		
		for contour in redContours:
			#1. Calculate convex hull of the contour
			hull = cv2.convexHull(contour)
			hullLen = cv2.arcLength(hull, True)

			#TODO: Optimize the step size and bounds, our goal is
			# to find the approximation with length 3 and make sure we don't miss it
			for i in np.arange(0.02, 1, 0.02):
				epsilon = i * hullLen
				approx = cv2.approxPolyDP(hull, epsilon, True)
				l = len(approx)
				if l < 3:
					#No good results :(
					break
				elif l == 3:
					#We have found a triangle approximation for this contour

					#See if all the contour angles are within our limits
					minAngle = 35
					maxAngle = 85
					anglesOk = True

					for i in range(0, l):
						p1 = tuple(approx[i][0])
						p2 = approx[(i + 1) % l][0]
						p3 = approx[(i + 2) % l][0]

						ang = self.angle(p1, p2, p3)

						if ang < minAngle or ang > maxAngle:
							anglesOk = False
							break

					#If all angles were within limits, draw the triangle
					if anglesOk:
						for i in range(0, l):
							p1 = tuple(approx[i][0])
							p2 = approx[(i + 1) % l][0]
							cv2.line(image, tuple(p1), tuple(p2), redSignsColor, 1)

					break
			
		#blueMask = spv.getBlueMask(image, 0.7, 1, 0.075, 0.4).astype(np.uint8)
		#tresh = cv2.cvtColor(blueMask, cv2.COLOR_GRAY2BGR)
		#im2, blueContours, _ = cv2.findContours(blueMask, 1, 2)
		#drawRectsOnImage(image, blueContours, minRectSize, blueSignsColor)
		return possibleSigns
			
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

	def getBlueMask(self, image, ar, br, luminanceMin, luminanceMax):
		mask = np.zeros((image.shape[0], image.shape[1]))

		for y in range(image.shape[0]):
			for x in range(image.shape[1]):
				blue = image[y][x][0]
				green = image[y][x][1]
				red = image[y][x][2]

				if blue / 255.0 > ar * (green / 255.0 + red / 255.0):
					#Blue color dominates
					if blue/255.0 - max(green,red)/255.0 > br*(max(green,red) - min(green,red))/255.0:
						#It's not too cyan or whatever
						luminance = (0.3 * red + 0.59 * green + 0.11 * blue)/255.0
						if luminance >= luminanceMin and luminance <= luminanceMax:
							mask[y][x] = 255

		return mask


	#Draws rectangles on the image. Does not draw rectangles with width/height smaller than minSize or rectangles
	#that are inside a bigger rectangle.
	def drawRectsOnImage(self, img, contours, minSize, color):
		rects = []
		signs = []

		for i in contours:
			x, y, w, h = cv2.boundingRect(i)
			if w < minSize or h < minSize:
				continue
			rects.append((x, y, w, h))

		rects.sort(key=lambda a: a[2] * a[3], reverse=True)

		for i, e in reversed(list(enumerate(rects))):
			if self.isContained(e, rects[:i]): del rects[i]

		for i in rects:
			
			#Include an offset to make it easier to see the inner area of the rect
			offset = 2
			signs.append(img[i[1]-offset:i[1]+i[3]+offset*2,i[0]-offset:i[0]+i[2]+offset*2])
			w = i[2]
			h = i[3]

			rectWidth = 2
			if w < 24 or h < 24:
				rectWidth = 1

			cv2.rectangle(img, (i[0] - offset, i[1] - offset), (i[0] + i[2] + offset * 2, i[1] + i[3] + offset * 2), color, rectWidth)

		return img,signs

	#Normalizes the input array.
	def normalize(self, v):
		norm=np.linalg.norm(v)
		if norm==0:
		   return v
		return v/norm

	#Returns the angle between lines ab and bc where a, b, c are [x, y] numpy arrays
	def angle(self, a, b, c):
		return math.degrees(math.acos(np.dot(
			(self.normalize(np.subtract(a, b))),
			(self.normalize(np.subtract(c, b))))))

	def isContained(self,rect, others):
		for i in others:
			if rect[0] > i[0] and rect[1]>i[1] and \
				rect[0]+rect[2] < i[0]+i[2] and \
				rect[1]+rect[3] < i[1]+i[3]:
				return True
		return False

	#Removes all the contours that are smaller than the given size and is convex.
	def removeBadContours(self, contours, minSize):
		out = []

		for contour in contours:
			x, y, w, h = cv2.boundingRect(contour)
			#Simple size check
			if w < minSize or h < minSize:
				continue

			#TODO: move the angle validation stuff here at some point

			out.append(contour)

		return out

if __name__ == "__main__":
	spv = SinglePixelVoting()

	#images = { 2 }
	images = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }
	showImages = False
	saveToFiles = True

	minRectSize = 10
	redSignsColor = [255, 255, 0]
	blueSignsColor = [0, 255, 255]

	for imageId in images:
		print imageId
		image = cv2.imread("streets/" + str(imageId) + ".png")

		redMask = spv.getRedMask(image, 0.75, 1).astype(np.uint8)
		#tresh = cv2.cvtColor(redMask, cv2.COLOR_GRAY2BGR)
		im2,redContours,_ = cv2.findContours(redMask,1,2)
		redContours = spv.removeBadContours(redContours, minRectSize)
		#drawRectsOnImage(image, redContours, minRectSize, redSignsColor)
		#image = cv2.drawContours(image, redContours, -1, redSignsColor, 1)

		for contour in redContours:
			#1. Calculate convex hull of the contour
			hull = cv2.convexHull(contour)
			hullLen = cv2.arcLength(hull, True)

			#TODO: Optimize the step size and bounds, our goal is
			# to find the approximation with length 3 and make sure we don't miss it
			for i in np.arange(0.02, 1, 0.02):
				epsilon = i * hullLen
				approx = cv2.approxPolyDP(hull, epsilon, True)
				l = len(approx)
				if l < 3:
					#No good results :(
					break
				elif l == 3:
					#We have found a triangle approximation for this contour

					#See if all the contour angles are within our limits
					minAngle = 35
					maxAngle = 85
					anglesOk = True

					for i in range(0, l):
						p1 = tuple(approx[i][0])
						p2 = approx[(i + 1) % l][0]
						p3 = approx[(i + 2) % l][0]

						ang = spv.angle(p1, p2, p3)

						if ang < minAngle or ang > maxAngle:
							anglesOk = False
							break

					#If all angles were within limits, draw the triangle
					if anglesOk:
						for i in range(0, l):
							p1 = tuple(approx[i][0])
							p2 = approx[(i + 1) % l][0]
							cv2.line(image, tuple(p1), tuple(p2), redSignsColor, 1)

					break

		#blueMask = spv.getBlueMask(image, 0.7, 1, 0.075, 0.4).astype(np.uint8)
		#tresh = cv2.cvtColor(blueMask, cv2.COLOR_GRAY2BGR)
		#cv2.imshow("Test", tresh)
		#im2, blueContours, _ = cv2.findContours(blueMask, 1, 2)
		#drawRectsOnImage(image, blueContours, minRectSize, blueSignsColor)

		if saveToFiles:
			cv2.imwrite("output/" + str(imageId) + ".png", image)

		#if showImages:
			#cv2.imshow("Test",image)

	if showImages:
		while True:
			k=cv2.waitKey(1) & 0xFF
			if k== 27:
				break
	cv2.destroyAllWindows()
