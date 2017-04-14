###########################################################
## Python Definition to extract Iris
###########################################################

######### Libraies used ###########
import os
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
###################################
######## Definitions ##############

def save_image(img,name,cord=None):
	if cord != None and len(cord) == 4:
		#cv2.imshow(name,img[cord[0]:cord[0]+cord[2],cord[1]:cord[1]+cord[3]])
		cv2.imwrite(name,img[cord[0]:cord[0]+cord[2],cord[1]:cord[1]+cord[3]])
	else:
		#cv2.imshow(name,img)
		cv2.imwrite(name,img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

def print_circle(img, pcord, icord, name):
	cimg	= cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	cv2.circle(cimg,(pcord[0],pcord[1]),pcord[2],(0,255,0),2)
	cv2.circle(cimg,(icord[0],icord[1]),icord[2],(0,255,0),2)
	cv2.circle(cimg,(icord[0],icord[1]),2,(0,0,255),3)
	cv2.imshow(name, cimg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

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
	irad,thres   = 60,20
	ycent, xcent = cord[1], cord[0]
	x1, x2	= min(wid-1, xcent+cord[2]*3), max(0,xcent-cord[2]*3) 
	pixco 	= min((int(img[ycent][x1])+int(img[ycent][x2]))/2,130)
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
	circles = cv2.HoughCircles(nimg, cv.CV_HOUGH_GRADIENT,1,20, param1=p1,param2=p2,minRadius=minR,maxRadius=maxR)
	circles = np.uint16(np.around(circles))
	pupil_cord = circles[0][0]
	pupil_cord[1] += xc_l
	pupil_cord[0] += yc_t
	return pupil_cord

def extract_cord(img):
	param = [10, 0, 50, 20]
	""" minR = 10
		maxR = 0
		p1   = 50
		p2   = 20 """
	cord = extract_pupil(img, param[0], param[1], param[2], param[3])
	return extract_iris(img, cord)

def normalize_width(img):
	fixed_img = cv2.resize(img,(360,60), interpolation = cv2.INTER_CUBIC)
	return fixed_img

def normalize_rectangular(img,pcord,icord):
	nsamples = 360.0
	iris_radius = icord[2]-pcord[2]
	polar = np.zeros((iris_radius, int(nsamples)))
	samples = np.linspace(0,2.0 * np.pi, nsamples)[:-1]
	for r in range(iris_radius):
	    for theta in samples:
	        x = int((r+pcord[2]) * np.cos(theta) + icord[0])
	        y = int((r+pcord[2]) * np.sin(theta) + icord[1])
	        t = int(theta * nsamples / 2.0 / np.pi)
	        polar[r][t] = img[y][x]
	
	cv2.imwrite('temp.jpg', polar)
	polar = cv2.imread('temp.jpg',0)
	return normalize_width(polar)

def process_each_class(single_class, class_path, target_dir):
	files = os.listdir(class_path)
	target_path = target_dir+"/"+single_class
	if not os.path.exists(target_path):
		os.mkdir(target_path,0755)
	
	for filename in files:
		if ".jpg" in filename:
			file_path = class_path+"/"+filename
			save_path = target_path+"/"+filename
			if os.path.exists(save_path):
				print "done "+filename	
				continue
			image = cv2.imread(file_path,0)
			image = cv2.medianBlur(image,3)
			pupil_cord, iris_cord = extract_cord(image)
			#print_circle(image, pupil_cord, iris_cord, filename)
			processed_img = normalize_rectangular(image, pupil_cord, iris_cord)
			save_image(processed_img, save_path)
			print "done "+filename

def process_raw_images(current_dir, target_dir):
	classes = os.listdir(current_dir)
	for single_class in classes:
		class_path = current_dir+"/"+single_class
		process_each_class(single_class, class_path, target_dir)


db_dir = os.getcwd() + "/Database";
tr_dir = os.getcwd() + "/Norm_img"
process_raw_images(db_dir, tr_dir)

