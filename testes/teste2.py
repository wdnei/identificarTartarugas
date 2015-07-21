import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
import descritores as d
import glob



index={}
images={}

for imagePath in glob.glob(".\samples\carettacaretta\*[0-9].jpg"):
	# extract the image filename (assumed to be unique) and
	# load the image, updating the images dictionary
	#filename = imagePath.split("\\")[-1]
	filename = imagePath
	image = cv2.imread(imagePath)
	images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# extract a 3D RGB color histogram from the image,
	# using 8 bins per channel, normalize, and update
	# the index
	hist = cv2.calcHist([image],[0,1,2],None,[64,64,64],[0,256,0,256,0,256]).flatten()
	#histNorm=cv2.normalize(hist, alpha=0, beta=255, norm_type=cv2.cv.CV_MINMAX)
	index[filename] = hist
	print filename,len(hist)

out=open("hist1.dmp","wb")
pickle.dump(index,out)
out.close()
