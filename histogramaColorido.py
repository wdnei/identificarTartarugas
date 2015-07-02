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

args = vars(ap.parse_args())

# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}


# loop over the image paths
for imagePath in glob.glob(args["dataset"] + "/*[0-9].jpg"):
	# extract the image filename (assumed to be unique) and
	# load the image, updating the images dictionary
	filename = imagePath.split("\\")[-1]
	image = cv2.imread(imagePath)
	images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# extract a 3D RGB color histogram from the image,
	# using 8 bins per channel, normalize, and update
	# the index
	hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
	hist = cv2.normalize(hist).flatten()
	index[filename] = hist
	print filename


# initialize OpenCV methods for histogram comparison
OPENCV_METHODS = (
	("Correlation", cv2.cv.CV_COMP_CORREL),
	("Chi-Squared", cv2.cv.CV_COMP_CHISQR),
	("Intersection", cv2.cv.CV_COMP_INTERSECT), 
	("Hellinger", cv2.cv.CV_COMP_BHATTACHARYYA))


# loop over the comparison methods

methodName="Correlation"
method=cv2.cv.CV_COMP_CORREL
conf_arr=[]

for (nameCurrentImage, histCurrentImage) in index.items():
	# initialize the results dictionary and the sort
	# direction
	results = []
	reverse = False

	# if we are using the correlation or intersection
	# method, then sort the results in reverse order
	if methodName in ("Correlation", "Intersection"):
		reverse = True

	# loop over the index
	for (name, hist) in index.items():
		# compute the distance between the two histograms
		# using the method and update the results dictionary
		d = cv2.compareHist(histCurrentImage, hist, method)
		results.append(float("{0:.2f}".format(d)))

	conf_arr.append(results)




#normalize
norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
#plt.clf()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('auto')
#ax.invert_yaxis()
ax.xaxis.tick_top()
res = ax.imshow(np.array(conf_arr),cmap=plt.cm.Greys)

width = len(conf_arr)
height = len(conf_arr[0])

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')

#cb = fig.colorbar(res)
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

plt.xticks(range(width), alphabet[:width])
plt.yticks(range(height), alphabet[:height])


# show the OpenCV methods
plt.show()


































