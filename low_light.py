import imutils
import cv2
import numpy as np
import cv2
import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy.signal import convolve2d
import skimage.color as sc
import matplotlib.pyplot as plt
#%matplotlib inline


def conv2(x, y, mode='same'):
	return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def gauss2d(shape=(15,15),sigma=0.5):
	
	m,n = [(ss-1.)/2. for ss in shape]
	y,x = np.ogrid[-m:m+1,-n:n+1]
	h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
	h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
	sumh = h.sum()
	if sumh != 0:
		h /= sumh
	return h




def low_image_enhancement(im):

	imhsv = sc.rgb2hsv(im)
	row, col, ch = np.shape(im)
	h1 = imhsv[:,:,0]
	s1 = imhsv[:,:,1]
	v1 = imhsv[:,:,2]


	H1 = gauss2d(sigma=15)
	H2 = gauss2d(sigma=80)
	H3 = gauss2d(sigma=250)

	img1 = conv2(v1, H1)
	img2 = conv2(v1, H2)
	img3 = conv2(v1, H3)


	imgguas = (1/3*img1) + (1/3*img2) + (1/3*img3) 


	alpha5 = .1
	k5 = alpha5 * np.sum(s1.flatten()) / ( row * col)
	vnew5 = (v1 * (1+k5) ) / (np.maximum(v1, imgguas) + k5)


	X1 = v1.flatten()
	X2 = vnew5.flatten()
	C = np.cov(X1, X2)
	D, V = np.linalg.eig(C)
	diagD=D
	V1 = np.zeros(np.size(V))

	if (diagD[0] > diagD[1]):
		V1 = V[:,0]   
	else:
		V1 = V[:,1]

	w1 = V1[0] / (V1[0] + V1[1])
	w2 = V1[1] / (V1[0] + V1[1] )

	F = (w1 * v1) + (w2 * vnew5)
	out1 = np.zeros((row,col, ch))
	out1[:,:,0] = h1
	out1[:,:,1] = s1
	out1[:,:,2] = F

	out1 =  sc.hsv2rgb(out1)

	out1=np.uint8(255*out1/np.max(np.ravel(out1)))
  

	return out1


path=r'E:\MAIN PROJECT\FIGURES\lowlight.jpg'
ite = cv2.imread(path)
out=low_image_enhancement(ite)
cv2.imshow('image',out)
cv2.imwrite(r'E:\MAIN PROJECT\OUTPUT\out1.jpg',out)
cv2.waitKey(0)

