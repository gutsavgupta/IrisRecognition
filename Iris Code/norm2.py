###########################################################
## Python Definition to extract Iris
###########################################################

## Libraies used
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt

## Definitions

def get_circle(img, minR, maxR, p1, p2):
	
	#return the first circle
	## return first circle with 
	## center near middle
	xc_l = len(img)/2 - 30
	xc_r = len(img)/2 + 30
	yc_t = len(img[0])/2 - 30
	yc_b = len(img[0])/2 + 30
	nimg = img[xc_l:xc_r, yc_t:yc_b]
	print_img(nimg,0,0,0,0)

	circles = cv2.HoughCircles(nimg, cv.CV_HOUGH_GRADIENT,1,20, param1=p1,param2=p2,minRadius=minR,maxRadius=maxR)
	circles = np.uint16(np.around(circles))
	
	# print xc_l,xc_r
	# print yc_t, yc_b 
	# print len(circles[0])
	
	for cir in circles[0]:
		print cir[0],cir[1]
		cir[1] += xc_l
		cir[0] += yc_t
		return cir

def print_img(img,x,y,w,h):
	if w!=0 and h!=0:
		cv2.imshow('detected circles',img[x:x+w,y:y+h])
	else:
		cv2.imshow('detected circles',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	

def print_circle(img, cord):
	cimg	= cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	
	# draw the outer circle
	# draw the center of the circle
	cv2.circle(cimg,(cord[0],cord[1]),cord[2],(0,255,0),2)
	cv2.circle(cimg,(cord[0],cord[1]),2,(0,0,255),3)

	## find the boundaries
	## using hight, widht
	h,w = 2*cord[2],2*cord[2]
	x,y = cord[1]-cord[2], cord[0]-cord[2]

	## print the image 
	## rel. boundaries
	print_img(cimg,x,y,0,h)


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

def extract_irs(img, cord):
	wid 	= len(img[0])
	dis		= min(wid-cord[0], cord[0])
	dis
	## Setting Threshold
	## pix_col	= avg_col(img, cord)
	thres	= 18
	ycent	= cord[1]
	xcent	= cord[0]
	pixco 	= img[ycent+cord[2]*2][xcent+cord[2]*2]
	print pixco
	print cord[2]+5, dis

	## Iterating horizontally
	for i in range(int(cord[2]*2), dis):
		lval = img[ycent][xcent-i]
		rval = img[ycent][xcent+i]
		#print lval,rval
		if (lval > thres+pixco) or (rval > thres+pixco ):
			mval = (int(lval)+int(rval))/2
			if mval > thres+pixco:
				irad = i
				break

	icord 	 = [cord[0], cord[1], irad]
	print cord
	print icord
	

	print_circle(img, icord)
	return cord,icord


def extract_pup(img):

	## code for getting pupil
	## return param of circle
	param = [10, 0, 50, 20]
	""" minR = 10
		maxR = 35
		p1   = 50
		p2   = 40
	"""
	cord	= get_circle(img, param[0], param[1], param[2], param[3])
	print_circle(img,cord)

	return extract_irs(img, cord)

def rectangular_norm(img,pcord,icord):
	#reject_out(img,icord[1],icord[0],icord[2])
	#reject_in(img,pcord[1],pcord[0],pcord[2])

	#print_img(img,0,0,0,0)

	center = (icord[0], icord[1])
	iris_radius = icord[2]-pcord[2]
	nsamples = 360.0
	samples = np.linspace(0,2.0 * np.pi, nsamples)[:-1]
	polar = np.zeros((iris_radius, int(nsamples)))
	#print polar
	for r in range(iris_radius):
	    for theta in samples:
	        x = int((r+pcord[2]) * np.cos(theta) + center[0])
	        y = int((r+pcord[2]) * np.sin(theta) + center[1])
	        t = int(theta * nsamples / 2.0 / np.pi)
	        polar[r][t] = img[y][x]
	        #print polar[r][t]
	
	cv2.imwrite('normalized.jpg', polar)
	npolar = cv2.imread('normalized.jpg',0)
	print_img(npolar,0,0,0,0)
	# plt.imshow(polar, cmap = 'gray', interpolation = 'bicubic')
	# plt.xticks([]), plt.yticks([])
	# plt.show()


for i in range(1,10):
	name = 'test'+str(i)+'.jpg'
	print name
	img1 = cv2.imread(name,0)
	img2 = cv2.medianBlur(img1,3)
	pcord,icord = extract_pup(img2)
	rectangular_norm(img1,pcord,icord)

