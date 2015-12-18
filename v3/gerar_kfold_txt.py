from random import shuffle
import cv2
import numpy as np
from matplotlib import pyplot as plt
import cPickle as pickle
import glob
import argparse
import webbrowser


import os

import MomentosCromaticidade
import HistogramaColoridoRGB
import classificador
import desenhagraficos

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-f", "--fold", required = True,help = "K-fold do classificador(numero inteiro)")
ap.add_argument("-n", "--nome", required = True,help = "Nome do Arquivo e local")

args = vars(ap.parse_args())

titulo=args["nome"]
fold=int(args["fold"])
allSet=[];

for filename in glob.glob("samples/*/*_[0-9]_[0-9].jpg"):
    #Descreve a imagem
    classe=filename.split(os.sep)[-2]
    allSet.append(filename)


arquivos=[]
for imagensTreino,imagensTeste in classificador.k_fold_cross_validation(allSet, fold, True):
    arquivos.append((imagensTreino,imagensTeste))

pickle.dump( arquivos, open( titulo+".p", "wb" ) )
print "arquivo '"+titulo+ "' gerado"
