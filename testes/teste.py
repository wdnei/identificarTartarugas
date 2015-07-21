import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
import descritores as d
import glob



out=open("hist1.dmp","rb")
index=pickle.load(out)
out.close()

imageName=".\samples\carettacaretta\carettacaretta_4.jpg"
image1=cv2.imread(imageName)
image1=cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

image3=cv2.imread(".\samples\carettacaretta\carettacaretta_1.jpg")
image3=cv2.cvtColor(image3, cv2.COLOR_BGR2HSV)


hist1=cv2.calcHist([image1], [0,1,2], None, [64,64,64],
		[0, 256,0, 256,0,256]).flatten()
#hist1=cv2.normalize(hist1, alpha=0, beta=255, norm_type=cv2.cv.CV_MINMAX)




d=cv2.compareHist(hist1, index[imageName], cv2.cv.CV_COMP_CORREL)
print d
quit()

OPENCV_METHODS = (
	("Correlation", cv2.cv.CV_COMP_CORREL),
	("Chi-Squared", cv2.cv.CV_COMP_CHISQR),
	("Intersection", cv2.cv.CV_COMP_INTERSECT), 
	("BHATTACHARYYA", cv2.cv.CV_COMP_BHATTACHARYYA))
resultsCHI={}
# loop over the comparison methods
for name in index.keys():
                d = cv2.compareHist(hist1, index[name], cv2.cv.CV_COMP_CORREL)
                resultsCHI[name]=d
                print name, d




                
        
