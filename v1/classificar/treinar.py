#Arquivo para trinamento de dados usando momentos de cromaticidade
#e histogramas coloridos

import cv2
import numpy as np
from matplotlib import pyplot as plt
import cPickle
import glob
import argparse
import sys

import MomentosCromaticidade
import HistogramaColoridoRGB

#Inicializar Descritor
descritor=None



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Caminho do diretorio de imagens")
ap.add_argument("-n", "--name", required = True,
	help = "Nome do arquivo a ser salvo")
ap.add_argument("-m", "--method", required = True,
	help = "Metodo a ser usado 1-Momentos de cromaticidade 2-Histogramas Coloridos")

args = vars(ap.parse_args())

if(args["method"]=='1'):
    descritor=MomentosCromaticidade
else:
    descritor=HistogramaColoridoRGB

# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves

index=[]
# loop over the image paths
for filename in glob.glob(args["dataset"] + "/*/*[0-9].jpg"):
    #Descreve a imagem
    classe=filename.split("\\")[-2]
    vecCarac = descritor.descrever(filename)
    vecCarac.append(classe)
    index.append(vecCarac)
    print filename,vecCarac



out = open(args["name"]+'.dmp', 'wb')
cPickle.dump(index,out)
out.close()
print "Dados de treino criados!"
