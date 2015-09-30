import cv2
import numpy as np
from matplotlib import pyplot as plt
import cPickle
import glob
import argparse
from descritores import HistogramaColoridoRGB

#Inicializar Descritor RGB
descRGB=HistogramaColoridoRGB



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

histNumBins=256



# loop over the image paths
for filename in glob.glob(args["dataset"] + "/*/*[0-9].jpg"):
    #Descreve a imagem
    hist = descRGB.descrever_estatistico(filename,histNumBins)
    index[filename] = hist
    print filename



out = open('histRGBtrainIndex.dmp', 'wb')
cPickle.dump(index,out)
out.close()
print "Dados de treino criados!"
