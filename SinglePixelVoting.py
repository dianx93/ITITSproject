import numpy as np
import cv2, math

#For reference, see chapter 2.1 of http://perso.lcpc.fr/tarel.jean-philippe/publis/jpt-icpr10.pdf
class SinglePixelVoting:

	#TODO: Finish this method
	def getTrafficSigns(self, image, minSize, maxSize):

		redSignsColor = [255, 255, 0]
		blueSignsColor = [0, 255, 255]
		
		possibleSigns = []

		# Optimal parameters for finding everything: 0.65 and 1.03
		redMask = self.getRedMask(image, 0.65, 1.03).astype(np.uint8)
		im2,redContours,_ = cv2.findContours(redMask,1,2)

		redContours = self.removeBadContours(image, redContours, minSize, maxSize)

		#drawRectsOnImage(image, redContours, minRectSize, redSignsColor)

		#cv2.drawContours(image, redContours, -1, redSignsColor, 1)
		return possibleSigns
		
	#Extracts the red part of the image with some tresholding.
	def getRedMask(self, image, ar, br):
		mask = np.zeros((image.shape[0],image.shape[1]))
		for y in range(image.shape[0]):
			for x in range(image.shape[1]):
				if self.isRedEnough(image[y][x], ar, br):
					mask[y][x] = 255
		return mask

	#Returns if a pixel is red enough based on the arguments. Pixel should be an BGR array.
	def isRedEnough(self, pixel, ar, br):
		blue = pixel[0]
		green = pixel[1]
		red = pixel[2]
		# Note: division by 255.0 is important to prevent overflows
		if red / 255.0 > ar * (green / 255.0 + blue / 255.0):
			# Red color dominates
			if red / 255.0 - max(green, blue) / 255.0 > br * (max(green, blue) - min(green, blue)) / 255.0:
				# It's not too yellow or magenta
				return True
		return False

	#Returns if a pixel is blue enough based on the arguments. Pixel should be an BGR array.
	def isBlueEnough(self, pixel, ar, br):
		blue = pixel[0]
		green = pixel[1]
		red = pixel[2]
		# Note: division by 255.0 is important to prevent overflows
		if blue / 255.0 > ar * (green / 255.0 + red / 255.0):
			# Bue color dominates
			if blue / 255.0 - max(green, red) / 255.0 > br * (max(green, red) - min(green, red)) / 255.0:
				# It's not too much of other colors
				return True
		return False

	#Returns if a pixel is green enough based on the arguments. Pixel should be an BGR array.
	def isGreenEnough(self, pixel, ar, br):
		blue = pixel[0]
		green = pixel[1]
		red = pixel[2]
		# Note: division by 255.0 is important to prevent overflows
		if green / 255.0 > ar * (blue / 255.0 + red / 255.0):
			# Green color dominates
			if green / 255.0 - max(blue, red) / 255.0 > br * (max(blue, red) - min(blue, red)) / 255.0:
				# It's not toomuch of other colors
				return True
		return False

	#Returns if the pixel's r,g,b values are within tolerance of one another
	def isGreyscale(self, pixel, tolerance):
		blue = pixel[0]
		green = pixel[1]
		red = pixel[2]

		maxDiff = max(abs(blue/255.0 - green/255.0), abs(blue/255.0 - red/255.0), abs(green/255.0 - red/255.0))

		return maxDiff <= tolerance


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

	# Removes all the contours that are smaller than the given size.
	def removeBadRedContours(self, image, mask, contours, minSize, maxSize):
		out = []

		for contour in contours:
			x, y, w, h = cv2.boundingRect(contour)
			# Simple size check
			if w < minSize or h < minSize or w > maxSize or h > maxSize:
				continue

			#Calculate the convex hull of the contour
			hull = cv2.convexHull(contour)
			hullLen = cv2.arcLength(hull, True)

			if (self.isTriangularSign(image, contour, hull, hullLen)
				or self.isCircularSign(image, contour, hull, hullLen, mask)):
				#or self.isRectangluarSign(image,contour,hull,hullLen)):
				out.append(contour)

		return out

	def isRectangluarSign(self, image, contour, hull, hullLen):
		for i in np.arange(0.06, 0.2, 0.02):
			epsilon = i * hullLen
			approx = cv2.approxPolyDP(hull, epsilon, True)
			l = len(approx)
			#print(str(i) + " " + str(l))
			if l < 4:
				#No good results :(
				break
			elif l == 4:
				#We have found a triangle approximation for this contour
				#See if all the contour angles are within our limits
				#and one triangle side is roughly parallel to the x axis
				minAngle = 70
				maxAngle = 110
				#how many degs can the side roughly parallel to the x axis be rotated in either way
				rotationError = 12
				anglesOk = True
				oneParallelSide = False

				for i in range(0, l):
					p1 = tuple(approx[i][0])
					p2 = approx[(i + 1) % l][0]
					p3 = approx[(i + 2) % l][0]

					ang = angle(p1, p2, p3)

					if ang < minAngle or ang > maxAngle:
						anglesOk = False
						break

					slopeAng = math.degrees(math.atan2(p2[0] * 1.0 - p1[0], p2[1] * 1.0 - p1[1]))
					#Normalize to [0, 180)
					if slopeAng < 0:
						slopeAng += 180

					if(slopeAng > 90 - rotationError and slopeAng < 90 + rotationError):
						#The line is roughly parallel to the x axis
						if oneParallelSide is False:
							oneParallelSide = True
						else:
							#One other side was already found to be roughly parallel to
							#the x axis, abort
							anglesOk = False
							oneParallelSide = False
							break

				#if all angles were within limits, draw the triangle
				if anglesOk and oneParallelSide:
					for i in range(0, l):
						p1 = tuple(approx[i][0])
						p2 = approx[(i + 1) % l][0]
						cv2.line(image, tuple(p1), tuple(p2),  [0, 255, 255], 1)

				return anglesOk
		return True

	def isTriangularSign(self, image, contour, hull, hullLen):
		#TODO: Optimize the step size and bounds, our goal is to find
		# the approximation with length 3 and make sure we don't miss it
		#Possible solution: When we reach 2, take a half step back, if still nothing found, stop.
		for i in np.arange(0.06, 0.2, 0.02):
			epsilon = i * hullLen
			approx = cv2.approxPolyDP(hull, epsilon, True)
			l = len(approx)
			#print(str(i) + " " + str(l))
			if l < 3:
				#No good results :(
				break
			elif l == 3:
				#We have found a triangle approximation for this contour
				#See if all the contour angles are within our limits
				#and one triangle side is roughly parallel to the x axis
				minAngle = 40
				maxAngle = 80
				#how many degs can the side roughly parallel to the x axis be rotated in either way
				rotationError = 12
				anglesOk = True
				oneParallelSide = False

				for i in range(0, l):
					p1 = tuple(approx[i][0])
					p2 = approx[(i + 1) % l][0]
					p3 = approx[(i + 2) % l][0]

					ang = angle(p1, p2, p3)

					if ang < minAngle or ang > maxAngle:
						anglesOk = False
						break

					slopeAng = math.degrees(math.atan2(p2[0] * 1.0 - p1[0], p2[1] * 1.0 - p1[1]))
					#Normalize to [0, 180)
					if slopeAng < 0:
						slopeAng += 180

					if(slopeAng > 90 - rotationError and slopeAng < 90 + rotationError):
						#The line is roughly parallel to the x axis
						if oneParallelSide is False:
							oneParallelSide = True
						else:
							#One other side was already found to be roughly parallel to
							#the x axis, abort
							anglesOk = False
							oneParallelSide = False
							break

				redEnoughPixels = 0;
				for i in range(0, l):
					p1 = tuple(approx[i][0])

					if self.isRedEnough(image[p1[1]][p1[0]], 0.72, 1.1):
						redEnoughPixels += 1

				#if all angles were within limits, draw the triangle
				if anglesOk and oneParallelSide and redEnoughPixels > 1:

					p1 = tuple(approx[0][0])
					p2 = tuple(approx[1][0])
					p3 = tuple(approx[2][0])

					center = ((p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3)

					greyscalePixels = 0
					width = 1
					for y in range(center[1] - width, center[1] + width + 1):
						for x in range(center[0] - width, center[0] + width + 1):
							if self.isGreyscale(image[y][x], 0.06):
								greyscalePixels += 1

					if greyscalePixels > 5:
						for i in range(0, l):
							p1 = tuple(approx[i][0])
							p2 = approx[(i + 1) % l][0]
							cv2.line(image, tuple(p1), tuple(p2),  [0, 255, 0], 1)

				return anglesOk
		return True
	
	def circleIntersectionArea(self, circle1, circle2):
		r0 = circle1[2]
		r1 = circle2[2]
		rr1 = r1*r1
		rr0 = r0*r0
		d = ((circle1[0]-circle2[0])**2 + (circle1[1]-circle2[1])**2)**0.5
		if d > r1+r0:
			# no overlap
			return 0
			
		if d <=	abs(r0-r1) and r0 >= r1:
			# r1 is completely in r0
			return 1
			
		if 	d <= abs(r0-r1) and r0 < r1:
			# r0 is completely in r1
			return -1
			
		phi = (math.acos((rr0 + (d * d) - rr1) / (2 * r0 * d))) * 2	
		theta = (math.acos((rr1 + (d * d) - rr0) / (2 * r1 * d))) * 2
		area1 = 0.5 * theta * rr1 - 0.5 * rr1 * math.sin(theta)
		area2 = 0.5 * phi * rr0 - 0.5 * rr0 * math.sin(phi)
		
		intersection_area = area1 + area2
		
		if r0 >= r1:
			# r1 is more than 50% in the other circle
			return intersection_area/math.pi*rr1 
			
		if r0 < r1:
			# r0 is more than 50% in the other circle
			return -intersection_area/ math.pi*rr0 >= 0.5
			
	def isCircularSign(self, image, contour, hull, hullLen, mask):
		x, y, w, h = cv2.boundingRect(contour)
		offset = 30

		minX = max(0, x-offset)
		minY = max(0, y-offset)
		maxX = min(x+w+offset, image.shape[1] - 1)
		maxY = min(y+h+offset, image.shape[0] - 1)

		#print(str(minX) + " - " + str(maxX) + " " + str(minY) + " - " + str(maxY))
		#Cut a rectangular region around the contour
		#subimage = mask[minY:maxY, minX:maxX]
		subimage = image[minY:maxY, minX:maxX]
		subimage = cv2.blur(subimage, (3, 3))
		subimage = self.getRedMask(subimage, 0.63, 1.1).astype(np.uint8)

		# param1 - it is the higher threshold of the two passed to the Canny() edge detector (the lower one is twice smaller).
		# param2 - it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more
		# false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
		circles = cv2.HoughCircles(subimage, cv2.HOUGH_GRADIENT, 0.5, 10, param1=20, param2=14, minRadius=5,
								   maxRadius=30) 
		#Remove overlapping circles   
		if circles is not None:
			temp = []
			for i in range(len(circles[0])):
				for j in range(i+1,len(circles[0])):
					intersectionarea = self.circleIntersectionArea(circles[0][i], circles[0][j])
					if intersectionarea <= -0.5:
						temp.append(circles[0][i])
					elif intersectionarea >= 0.5:
						temp.append(circles[0][j])
			#Vaja see filter paika saada siis peaks töötamas
			circles[0] = [i for i in circles[0] if i not in temp]
						
		
		if circles is not None:
			#print "Adding " + str(len(circles[0, :])) + " circles"
			for circle in circles[0, :]:
				center = (int(minX + circle[0]), int(minY + circle[1]))


				goodPixels = 0
				width = max(1, min(8, (int)(circle[2] * 0.8)))
				totalPixels = ((width * 2 + 1) * (width *2 + 1))

				averageBrightness = 0

				for y in range(center[1] - width, center[1] + width + 1):
					for x in range(center[0] - width, center[0] + width + 1):
						brightness = image[y][x][0]/3.0 + image[y][x][1]/3.0 + image[y][x][2]/3.0
						averageBrightness += brightness / float(totalPixels)
						if (self.isGreyscale(image[y][x], 0.12)
							or self.isRedEnough(image[y][x], 0.7, 1.0)
							or self.isBlueEnough(image[y][x], 0.75, 1.2)):
							goodPixels += 1

				goodPercent = goodPixels / float(totalPixels)

				#Convert to [0, 1]
				averageBrightness = averageBrightness / 255.0;

				#print ("Radius:" + str(circle[2]) + " width: " + str(width))
				print "Good: " + str(goodPercent) + " Brightness:" + str(averageBrightness)

				if goodPercent > 0.8 and averageBrightness > 0.15:
					cv2.circle(image, (int(minX + circle[0]), int(minY + circle[1])), circle[2], [0, 0, 255], 1)

			return True
		else:
			return False


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

#Normalizes the input array.
def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
       return v
    return v/norm

#Returns the angle between lines ab and bc where a, b, c are [x, y] numpy arrays
def angle(a, b, c):
	return math.degrees(math.acos(np.dot(
		(normalize(np.subtract(a, b))),
		(normalize(np.subtract(c, b))))))

def isContained(rect, others):
	for i in others:
		if rect[0] > i[0] and rect[1]>i[1] and \
			rect[0]+rect[2] < i[0]+i[2] and \
			rect[1]+rect[3] < i[1]+i[3]:
			return True
	return False

if __name__ == "__main__":
	spv = SinglePixelVoting()

	images = { 10 }
	#images = { 2, 4, 5, 6, 7, 8, 9, 10, 11 }
	showImages = False
	saveToFiles = True

	minRectSize = 5
	maxRectSize = 50
	redSignsColor = [255, 255, 0]
	blueSignsColor = [0, 255, 255]

	for imageId in images:
		print "Image " + str(imageId)
		image = cv2.imread("streets/" + str(imageId) + ".png")
		kernel = np.ones((4,4),np.uint8)
		#image = cv2.erode(image,kernel,iterations=1)
		#image = cv2.dilate(image,kernel,iterations=1)
		redMask = spv.getRedMask(image, 0.65, 1.03).astype(np.uint8)
		#redMask = cv2.dilate(redMask,kernel,iterations=1)
		#redMask = cv2.erode(redMask,kernel,iterations=1)
		tresh = cv2.cvtColor(redMask, cv2.COLOR_GRAY2BGR)
		im2,redContours,_ = cv2.findContours(redMask,1,2)
		redContours = spv.removeBadRedContours(image, redMask, redContours, minRectSize, maxRectSize)
		#drawRectsOnImage(image, redContours, minRectSize, redSignsColor)
		
		#redMask = cv2.cvtColor(redMask, cv2.COLOR_GRAY2BGR)
		#image = cv2.drawContours(image, redContours, -1, redSignsColor, 1)
		
		#blueMask = spv.getBlueMask(image, 0.4, 0.7, 0.1, 0.5).astype(np.uint8)
		#blueMask = spv.getBlueMask(image, 0.5, 1, 0.075, 0.4).astype(np.uint8)
		#tresh = cv2.cvtColor(blueMask, cv2.COLOR_GRAY2BGR)
		#cv2.imshow("Test", tresh)
		#im2, blueContours, _ = cv2.findContours(blueMask, 1, 2)
		#blueContours = spv.removeBadContours(image, blueMask, blueContours, minRectSize, maxRectSize)
		#drawRectsOnImage(image, blueContours, minRectSize, blueSignsColor)
		
		#image = cv2.drawContours(image, blueContours, -1, blueSignsColor, 1)
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