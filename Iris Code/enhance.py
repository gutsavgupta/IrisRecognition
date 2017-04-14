###########################################################
## Python Definition to extract Iris
###########################################################

## Libraies used
import cv2
import cv2.cv as cv
import numpy as np

#####################################
############ Smoothing ##############

## 4x4 patch used for smmothing

def print_img(img,x,y,w,h, name):
	if w!=0 and h!=0:
		cv2.imshow(name,img[x:x+w,y:y+h])
		#cv2.imwrite('app-2/1.jpg',img[x:x+w,y:y+h])
	else:
		#cv2.imshow(name,img)
		cv2.imwrite('enhance-final/'+name,img)
	#cv2.waitKey(0)

	#cv2.destroyAllWindows()


def smooth(img_org,name):
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

	#print_img(img,0,0,0,0,name)
	return img


def sharpening(img_org, img_smt, name):
	img_shrp = cv2.addWeighted( img_org, 1.5, img_smt, -0.6, 0.0);
	#print_img(img_shrp,0,0,0,0,name)
	return img_shrp

def hequal(img, name):
	equ = cv2.equalizeHist(img)
	#stacking images side by side
	res = np.hstack((img,equ))
	#cv2.imwrite('res.png',res)
	#res = cv2.imread('res.png',0)
	#print_img(res,0,0,0,0,name)

def CLAHE(img, name):
	clahe1 = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4,4))
	clahe2 = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6,6))
	clahe3 = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
	cl1 = clahe1.apply(img)
	cl2 = clahe2.apply(img)
	cl3 = clahe3.apply(img)
	res = np.hstack((cl1,cl2,cl3))
	#cv2.imwrite('clahe_2.jpg',cl1)
	print_img(cl2,0,0,0,0,name)

for i in range(1,10):
	name = 'normimg/'+str(i)+'.jpg'
	print name
	img1 = cv2.imread(name,0)
	img_smt = smooth(img1, "Smooth"+str(i)+".jpg")
	img_shrp = sharpening(img1,img_smt,"sharp"+str(i)+".jpg")
	#hequal(img_shrp, "Histrogran"+str(i)+".jpg")
	CLAHE(img_shrp, "CLAHE"+str(i)+".jpg")
	#CLAHE(img1, "CLAHE normal"+str(i)+".jpg")




