###########################################################
## Python Definition to enhance normalize Iris Image
###########################################################

######### Libraies used ############
import os
import cv2
import cv2.cv as cv
import numpy as np
#####################################
########## Definitions ##############

def save_image(img,name,cord=None):
	if cord != None and len(cord) == 4:
		#cv2.imshow(name,img[cord[0]:cord[0]+cord[2],cord[1]:cord[1]+cord[3]])
		cv2.imwrite(name,img[cord[0]:cord[0]+cord[2],cord[1]:cord[1]+cord[3]])
	else:
		#cv2.imshow(name,img)
		cv2.imwrite(name,img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

def smoothing(img_org):
	img = img_org.copy()
	row = len(img)
	col = len(img[0])
	for i in range(0,row):
		for j in range(0,col):
			avg,count = 0,0;
			for ki in range(0,2):
				for kj in range(0,2):
					if (i+ki<row) and (j+kj<col):
						avg += img[i+ki][j+kj]
						count+=1
			img[i][j] = int(avg/count)
	return img

def sharpening(img_org, img_smt):
	img_shrp = cv2.addWeighted( img_org, 1.5, img_smt, -0.6, 0.0);
	return img_shrp

def CLAHE(img):
	clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6,6))
	return clahe.apply(img)

def Enhance_image(img):
	smooth_img = smoothing(img)
	sharp_img  = sharpening(img, smooth_img)
	return CLAHE(sharp_img)

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
			proccesed_img = Enhance_image(image)
			save_image(proccesed_img, save_path)
			print "done "+filename


def process_norm_images(current_dir, target_dir):
	classes = os.listdir(current_dir)
	for single_class in classes:
		class_path = current_dir+"/"+single_class
		process_each_class(single_class, class_path, target_dir)


db_dir = os.getcwd() + "/Norm_img";
tr_dir = os.getcwd() + "/Enhance_img"
process_norm_images(db_dir, tr_dir)



