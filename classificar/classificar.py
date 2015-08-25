import cv2
import numpy as np
from matplotlib import pyplot as plt
import cPickle
import glob
import argparse
import webbrowser
import sys
import knn


import MomentosCromaticidade
import HistogramaColoridoRGB

#Inicializar Descritor
descritor=None


#Inicializar Descritor RGB
descritor=MomentosCromaticidade


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--method", required = True,help = "Metodo a ser usado 1- Momentos de Cromaticidade , 2-Histograma colorido")
ap.add_argument("-d", "--datatrained", required = True,help = "Banco das imagens indexadas")
ap.add_argument("-i", "--image", required = True,help = "Imagem a ser usada como query")
ap.add_argument("-c","--class",required = True,help = "class")

args = vars(ap.parse_args())

if(args["method"]=='1'):
    descritor=MomentosCromaticidade
else:
    descritor=HistogramaColoridoRGB


# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
arquivo=open( args["datatrained"], "rb" )
index = cPickle.load( arquivo )
arquivo.close()
images = {}



#nova imagem
#initialize query image
mainImagePath=args["image"]
mainCarac= descritor.descrever(mainImagePath)

trainingSet=index
mainCarac.append(args["class"])
testSet=[mainCarac]
split = 0.67
print 'Train set: ' + repr(len(trainingSet))
print 'Test set: ' + repr(len(testSet))
# generate predictions
predictions=[]
k = 3 # valor de k igual a 3
for x in range(len(testSet)):
	neighbors = knn.getNeighbors(trainingSet, testSet[x], k)
	result = knn.getResponse(neighbors)
	predictions.append(result)
	print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
accuracy = knn.getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')

