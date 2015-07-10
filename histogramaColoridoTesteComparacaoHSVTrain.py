import cv2
import numpy as np
from matplotlib import pyplot as plt
import cPickle
import glob
import argparse


def chi2_distance(histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])
 
		# return the chi-squared distance
		return d


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory of images")
args = vars(ap.parse_args())

# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}

histNumBins=[128,128,128]

hbins = 180
sbins = 255
hrange = [0,180]
srange = [0,256]
ranges = hrange+srange

# loop over the image paths
for imagePath in glob.glob(args["dataset"] + "/*/*[0-9].jpg"):
	# extract the image filename (assumed to be unique) and
	# load the image, updating the images dictionary
	#filename = imagePath.split("\\")[-1]
	filename = imagePath
	image = cv2.imread(imagePath)
	images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	# extract a 3D RGB color histogram from the image,
	# using 8 bins per channel, normalize, and update
	# the index
	hist = cv2.calcHist([image],[0,1],None,[180,256],ranges)
	cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
	index[filename] = hist
	print filename


out = open('trainIndex.dmp', 'wb')
cPickle.dump(index,out)
out.close()
print "Dados de treino criados!"
