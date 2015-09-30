import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
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
ap.add_argument("-i", "--image", required = True,
	help = "PathName of Image to be compared")

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

#Imagem do proprio set
#mainImagePath=index.items()[15][0]
#mainHist =index.items()[15][1]

#nova imagem
#initialize query image
mainImagePath=args["image"]
mainImage= cv2.imread(mainImagePath)
mainImage=cv2.cvtColor(mainImage, cv2.COLOR_BGR2HSV)
mainHist =cv2.calcHist([mainImage],[0,1],None,[180,256],ranges)
cv2.normalize(mainHist,mainHist,0,255,cv2.NORM_MINMAX)
print mainHist


# initialize OpenCV methods for histogram comparison
OPENCV_METHODS = (
	("Correlation", cv2.cv.CV_COMP_CORREL),
	("Chi-Squared", cv2.cv.CV_COMP_CHISQR),
	("Intersection", cv2.cv.CV_COMP_INTERSECT), 
	("BHATTACHARYYA", cv2.cv.CV_COMP_BHATTACHARYYA))

# loop over the comparison methods
for (methodName, method) in OPENCV_METHODS:
        histReport="""<html>
        <head>
        </head>
        
        <body><div style=\"border:1px solid black; height:90px\">
                        <div style=\"float:left; padding-right:5px;\">
                        <img src=\""""+mainImagePath+"""\"/>
                        </div>
                        <div>
                        <p>Query Image:"""+mainImagePath+"""</p>
                        
                </span>
                        </div>
                </div><\br>"""
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
                d = cv2.compareHist(hist, mainHist, method)
                results[name] = d

        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)


        # loop over the results
        for (i, (value, name)) in enumerate(results):
                # show the result
                print  methodName ,name
                histText=""
                histReportLine=""
                histReportLine+= """<div style=\"border:1px solid black; height:90px\">
                        <div style=\"float:left; padding-right:5px;\">
                        <img src=\""""+name+"""\"/>
                        </div>
                        <div>
                        <p>"""+name+"""</p>
                        <span style="float:top;\">
                        """+str(value)+"""
                </span>
                        </div>
                </div>"""
                histReport+=histReportLine

        histReport+="</body></html>"
        text_file = open(".\\relatorios\\"+methodName+".html", "w")
        text_file.write(histReport)
        text_file.close()



