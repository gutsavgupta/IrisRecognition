# Python Definition to extract Iris
###########################################################

## Libraies used
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py

###########################################################
############ DCT Feature Extraction #######################
count = 0

def print_bar_graph(vector):
	global count
	N = len(vector)
	#b = np.hamming(N)
	#b = np.resize(b,(N,1))
	#vec = vector*b
	x = range(N)
	if count <3:
		fig = plt.figure()
		ax = plt.subplot(111)
		width=0.8
		ax.bar(x, vector, width=width)
		plt.show(ax)
		count+=1

def print_img(img,x,y,w,h, name):
	if w!=0 and h!=0:
		cv2.imshow(name,img[x:x+w,y:y+h])
		#cv2.imwrite('app-2/1.jpg',img[x:x+w,y:y+h])
	else:
		cv2.imshow(name,img)
		#cv2.imwrite('enhance-final/'+name,img)
	cv2.waitKey(0)

	cv2.destroyAllWindows()


def DCT(img):
	imf = np.float32(img)-128  # float conversion/scale
	dst = cv2.dct(imf)           # the dct
	imgcv1 = np.float32(dst)    # convert back
	#print imgcv1[0:9]
	print_bar_graph(imgcv1)
	#print_img(imgcv1,0,0,0,0,"")
	b = np.hamming(10)
	b = np.resize(b,(10,1))
	return imgcv1[:10]*b
	#cv2.imwrite('normalized.jpg', imgcv1)

def average(img_patch):
	col = len(img_patch[0])
	row = len(img_patch)
	single_dim = np.array([0.0]*col)

	for i in range(0,col):
		avg = 0.0
		count = 0
		for j in range(0,row):
			avg += img_patch[j][i]
			count+=1
		single_dim[i] = avg/count
	#print single_dim
	return single_dim

def distance(a,b):
	length = len(a)
	avg = 0.0
	count = 0
	for i in range(0,length):
		avg 	+= (a[i]-b[i])*(a[i]-b[i])
		count	+=1
	print avg/count



def extract_feature(img):
	global count
	count = 0
	col = len(img[0])
	row = len(img)
	overlap = 0.5
	patch_dx = 36
	patch_dy = 6
	final_vector = np.array([])

	for i in range(0,row, int(patch_dy*overlap)):
		for j in range(0,col, int(patch_dx*overlap)):
			if(i+patch_dy<row) and (j+patch_dx<col):
				single_dim = average(img[i:i+patch_dy,j:j+patch_dx])
				vector = DCT(single_dim)
				vector = np.resize(vector,(1,len(vector)))
				final_vector = np.append(final_vector,vector)

	return final_vector

# for i in range(1,3):
# 	name = 'normimg/'+str(i)+'.jpg'
# 	print name
# 	img1 = cv2.imread(name,0)
# 	extract_feature(img1)

img1 = cv2.imread('enhance-final/CLAHE1.jpg',0)
img2 = cv2.imread('enhance-final/CLAHE8.jpg',0)

distance(extract_feature(img1),extract_feature(img2))