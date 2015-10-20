import cv2
import numpy as np
from matplotlib import pyplot as plt
import cPickle
import glob
import argparse
import webbrowser
import sys
import knn
sys.path.insert(0, '../descritores')

import MomentosCromaticidade

#Inicializar Descritor RGB
descMC=MomentosCromaticidade


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--datatrained", required = True,help = "Banco das imagens indexadas")
ap.add_argument("-i", "--image", required = True,help = "Imagem a ser usada como query")
ap.add_argument("-c","--classe",required = True,help = "classe")

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
mainCarac= descMC.descrever(mainImagePath,5,5)

trainingSet=index
mainCarac.append(args["classe"])
testSet=[mainCarac]
split = 0.67
print 'Train set: ' + repr(len(trainingSet))
print 'Test set: ' + repr(len(testSet))
# generate predictions
predictions=[]
k = 3
for x in range(len(testSet)):
	neighbors = knn.getNeighbors(trainingSet, testSet[x], k)
	result = knn.getResponse(neighbors)
	predictions.append(result)
	print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
accuracy = knn.getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')

