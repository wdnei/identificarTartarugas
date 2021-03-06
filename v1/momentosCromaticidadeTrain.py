import cv2
import numpy as np
from matplotlib import pyplot as plt
import cPickle
import glob
import argparse
from descritores import MomentosCromaticidade

#Inicializar Descritor RGB
descMC=MomentosCromaticidade



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory of images")
args = vars(ap.parse_args())

# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves

index={}
# loop over the image paths
for filename in glob.glob(args["dataset"] + "/*/*[0-9].jpg"):
    #Descreve a imagem
    vecCarac = descMC.descrever(filename,5,5)
    index[filename]=vecCarac
    print filename,vecCarac



out = open('mcTrainIndex.dmp', 'wb')
cPickle.dump(index,out)
out.close()
print "Dados de treino criados!"
