import cv2
import numpy as np
from matplotlib import pyplot as plt
import cPickle
import glob
import argparse
import descritores as desc

#Inicializar Descritor RGB
descRGB =desc.HistogramaColoridoRGB

def chi2_distance(histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])

		# return the chi-squared distance
		return d


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--datatrained", required = True,help = "Path to the directory of images")
ap.add_argument("-i", "--image", required = True,help = "PathName of Image to be compared")

args = vars(ap.parse_args())

# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
arquivo=open( args["datatrained"], "rb" )
index = cPickle.load( arquivo )
arquivo.close()
images = {}

histNumBins=256



#nova imagem
#initialize query image
mainImagePath=args["image"]
mainHist= descRGB.get_histograma(mainImagePath,histNumBins)

print len(mainHist)


# initialize OpenCV methods for histogram comparison
methodName="Chi-Squared"
# loop over the comparison methods

histReport="""<html> <head> </head>

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
		d = descRGB.comparar(mainHist, hist)
		results[name] = d

# sort the results
results = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)
# loop over the results
for (i,(value, name)) in enumerate(results):
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
