import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
import glob
import argparse



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory of images")
ap.add_argument("-i", "--image", required = True,
	help = "PathName of Image to be compared")

args = vars(ap.parse_args())

# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}

histNumBins=[64,64,64]


# loop over the image paths
for imagePath in glob.glob(args["dataset"] + "/*/*[0-9].jpg"):
	# extract the image filename (assumed to be unique) and
	# load the image, updating the images dictionary
	filename = imagePath.split("\\")[-1]
	image = cv2.imread(imagePath)
	images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# extract a 3D RGB color histogram from the image,
	# using 8 bins per channel, normalize, and update
	# the index
	hist = cv2.calcHist([image], [0, 1, 2], None, histNumBins,
		[0, 256, 0, 256, 0, 256])
	hist = cv2.normalize(hist).flatten()
	index[filename] = hist
	print filename


#initialize query image
mainImage= cv2.imread(args["image"])
mainImage=cv2.cvtColor(mainImage, cv2.COLOR_BGR2RGB)
mainHist = cv2.calcHist([mainImage], [0, 1, 2], None, histNumBins,[0, 256, 0, 256, 0, 256])
mainHist = cv2.normalize(hist).flatten()


# initialize OpenCV methods for histogram comparison
OPENCV_METHODS = (
	("Correlation", cv2.cv.CV_COMP_CORREL),
	("Chi-Squared", cv2.cv.CV_COMP_CHISQR),
	("Intersection", cv2.cv.CV_COMP_INTERSECT), 
	("Hellinger", cv2.cv.CV_COMP_BHATTACHARYYA))

# loop over the comparison methods
for (methodName, method) in OPENCV_METHODS:
	# initialize the results dictionary and the sort
	# direction
	results = {}
	reverse = False

	# if we are using the correlation or intersection
	# method, then sort the results in reverse order
	if methodName in ("Correlation", "Intersection"):
		reverse = True

	# loop over the index
	for (name, hist) in index.items():
		# compute the distance between the two histograms
		# using the method and update the results dictionary
		d = cv2.compareHist(mainHist, hist, method)
		results[name] = d

	# sort the results
	results = sorted([(v, k) for (k, v) in results.iteritems()], reverse = reverse)

	# show the query image
	fig = plt.figure("Query")
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(mainImage)
	plt.axis("off")

	# initialize the results figure
	fig = plt.figure("Results: %s" % (methodName))
	fig.suptitle(methodName, fontsize = 20)

	# loop over the results
	for (i, (value, name)) in enumerate(results):
		# show the result
		ax = fig.add_subplot(len(images),1 ,i)
		ax.set_title("%s: %.2f" % (name, value))
		plt.imshow(images[name])
		plt.axis("off")
		print  methodName ,name

# show the OpenCV methods
plt.show()

