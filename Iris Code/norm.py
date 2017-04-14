###########################################################
## Python Definition to extract Iris
###########################################################

## Libraies used
import cv2
import cv2.cv as cv
import numpy as np

## Definitions

def get_circle(img, minR, maxR, p1, p2):
	
	circles = cv2.HoughCircles(img, cv.CV_HOUGH_GRADIENT,1,20, param1=p1,param2=p2,minRadius=minR,maxRadius=maxR)
	circles = np.uint16(np.around(circles))

	#return the first circle
	return circles[0][0]

def print_img(img,x,y,w,h):
	if w!=0 and h!=0:
		cv2.imshow('detected circles',img[x:x+w,y:y+h])
		cv2.imwrite('app-1/1.jpg',img[x:x+w,y:y+h])
	else:
		cv2.imshow('detected circles',img)
		cv2.imwrite('app-1/1.jpg',img)
	cv2.waitKey(0)

	cv2.destroyAllWindows()	

def reject_out(img, xc, yc, r):
	row = len(img)
	col = len(img[0])

	for x in range(0,row):
		for y in range(0, col):
			res = (x-xc)*(x-xc) + (y-yc)*(y-yc)
			if res > r*r :
				img[x][y] = 0

def reject_in(img, xc, yc, r):
	xs = xc-r
	ys = yc-r

	for x in range(xs,xs+2*r):
		for y in range(ys, ys+2*r):
			res = (x-xc)*(x-xc) + (y-yc)*(y-yc)
			if res < r*r :
				img[x][y] = 0

def extract_iris(img):
	cimg	= cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	cord 	= get_circle(img, 35, 0, 50, 40)
	# draw the outer circle
	cv2.circle(cimg,(cord[0],cord[1]),cord[2],(0,255,0),2)
	# draw the center of the circle
	cv2.circle(cimg,(cord[0],cord[1]),2,(0,0,255),3)
	

	h = 2*cord[2]
	w = 2*cord[2]
	x = cord[1]-cord[2]
	y = cord[0]-cord[2]
	nimg = img[x:x+w,y:y+h]
	reject_out(nimg, h/2, w/2, h/2)
	print_img(cimg,0,0,0,0)

	cord 	= get_circle(nimg, 0, cord[2]-1, 50, 30)
	print nimg[0:w][cord[0]]
	# draw the outer circle
	cv2.circle(nimg,(cord[0],cord[1]),cord[2],(0,255,0),2)
	# draw the center of the circle
	cv2.circle(nimg,(cord[0],cord[1]),2,(0,0,255),3)
	reject_in(nimg, cord[1], cord[0], cord[2])
	print_img(nimg,0,0,0,0)
	print nimg[0:w][cord[0]]

	
	
img = cv2.imread('test7.jpg',0)
img = cv2.medianBlur(img,5)
extract_iris(img)
