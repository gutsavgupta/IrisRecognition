###########################################################
## Python Definition to extract Iris
###########################################################

## Libraies used
import os
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt

## Definitions

def save_image(img,name,cord=None):
	if cord != None and len(cord) == 4:
		cv2.imshow(name,img[cord[0]:cord[0]+cord[2],cord[1]:cord[1]+cord[3]])
		cv2.imwrite(name,img[cord[0]:cord[0]+cord[2],cord[1]:cord[1]+cord[3]])
	else:
		cv2.imshow(name,img)
		cv2.imwrite(name,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def print_circle(img, icord, pcord, name):
	cimg	= cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	cv2.circle(cimg,(pcord[0],pcord[1]),pcord[2],(0,255,0),2)
	cv2.circle(cimg,(icord[0],icord[1]),icord[2],(0,255,0),2)
	cv2.circle(cimg,(icord[0],icord[1]),2,(0,0,255),3)
	cv2.imshow(name, cimg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# def reject_out(img, xc, yc, r):
# 	row = len(img)
# 	col = len(img[0])
# 	for x in range(0,row):
# 		for y in range(0, col):
# 			res = (x-xc)*(x-xc) + (y-yc)*(y-yc)
# 			if res > r*r :
# 				img[x][y] = 0

# def reject_in(img, xc, yc, r):
# 	xs = xc-r
# 	ys = yc-r
# 	for x in range(xs,xs+2*r):
# 		for y in range(ys, ys+2*r):
# 			res = (x-xc)*(x-xc) + (y-yc)*(y-yc)
# 			if res < r*r :
# 				img[x][y] = 0

def patch_value(img,y,x):
	avg,count = 0,0
	wid = len(img[0])
	for i in range(-1,2):
		for j in range(-1,2):
			if (x+j) >= 0 and (x+j) < wid:
				avg = avg+img[y+i][x+j]
				count += 1
	return (avg/count)

def extract_iris(img, cord):
	wid 	= len(img[0])
	dis		= min(wid-cord[0], cord[0])
	irad  	= 60
	thres	= 20
	ycent	= cord[1]
	xcent	= cord[0]
	pixco 	= min((int(img[ycent][xcent+cord[2]*3])+int(img[ycent][xcent-cord[2]*3]))/2,130)
	###### Iterating horizontally #####
	for i in range(int(cord[2]*2), dis):
		lval = patch_value(img,ycent,xcent-i)
		rval = patch_value(img,ycent,xcent+i)
		if (lval > thres+pixco) or (rval > thres+pixco ):
			mval = (int(lval)+int(rval))/2
			if mval > thres+pixco:
				irad = min(irad,i)
				break
	###################################
	icord = [cord[0], cord[1], irad]
	return cord,icord

def extract_pupil(img, minR, maxR, p1, p2):
	xc_l = len(img)/2 - 30
	xc_r = len(img)/2 + 30
	yc_t = len(img[0])/2 - 30
	yc_b = len(img[0])/2 + 30
	nimg = img[xc_l:xc_r, yc_t:yc_b]
	print_img(nimg,0,0,0,0)

	circles = cv2.HoughCircles(nimg, cv.CV_HOUGH_GRADIENT,1,20, param1=p1,param2=p2,minRadius=minR,maxRadius=maxR)
	circles = np.uint16(np.around(circles))

	pupil_cord = circles[0][0]:
	pupil_cord[1] += xc_l
	pupil_cord[0] += yc_t
	return pupil_cord

def extract_cord(img):
	## code for getting pupil
	## return param of circle
	param = [10, 0, 50, 20]
	""" minR = 10
		maxR = 0
		p1   = 50
		p2   = 20 """
	cord = extract_pupil(img, param[0], param[1], param[2], param[3])
	return extract_iris(img, cord)

# def size_norm(img):
# 	global count
# 	count = count+1
# 	res = cv2.resize(img,(360,60), interpolation = cv2.INTER_CUBIC)
# 	cv2.imshow('detected circles',res)
# 	path = 'normimg/'+str(count)+'.jpg'
# 	cv2.imwrite(path,res)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()

# def rectangular_norm(img,pcord,icord):
# 	reject_out(img,icord[1],icord[0],icord[2])
# 	reject_in(img,pcord[1],pcord[0],pcord[2])

# 	print_img(img,0,0,0,0)

# 	center = (icord[0], icord[1])
# 	iris_radius = icord[2]-pcord[2]
# 	nsamples = 360.0
# 	samples = np.linspace(0,2.0 * np.pi, nsamples)[:-1]
# 	polar = np.zeros((iris_radius, int(nsamples)))
# 	#print polar
# 	for r in range(iris_radius):
# 	    for theta in samples:
# 	        x = int((r+pcord[2]) * np.cos(theta) + center[0])
# 	        y = int((r+pcord[2]) * np.sin(theta) + center[1])
# 	        t = int(theta * nsamples / 2.0 / np.pi)
# 	        polar[r][t] = img[y][x]
# 	        #print polar[r][t]
	
# 	cv2.imwrite('normalized.jpg', polar)
# 	npolar = cv2.imread('normalized.jpg',0)
# 	print_img(npolar,0,0,0,0)
# 	size_norm(npolar)
	
# 	# plt.imshow(polar, cmap = 'gray', interpolation = 'bicubic')
# 	# plt.xticks([]), plt.yticks([])
# 	# plt.show()

def process_each_class(single_class, class_path, target_dir):
	files = os.listdir(class_path)
	target_path = target_dir+"/"+single_class
	if not os.path.exists(target_path):
		os.mkdir(target_path,0755)
	
	for filename in files:
		file_path = class_path+"/"+filename
		image = cv2.imread(file_path,0)
		image = cv2.medianBlur(img1,3)
		pupil_cord, iris_cord = extract_cord(image)
		print_circle(image, iris_cord, pupil_cord, "Iris & Pupil boundaries")



def process_raw_images(current_dir, target_dir):
	classes = os.listdir(current_dir)
	for single_class in classes:
		class_path = current_dir+"/"+single_class
		process_each_class(single_class, class_path, target_dir)


db_dir = os.getcwd() + "/Database";
tr_dir = os.getcwd() + "/Normalized"
process_raw_images(db_dir, tr_dir)

