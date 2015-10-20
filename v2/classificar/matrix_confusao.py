
import cv2
import numpy as np
from matplotlib import pyplot as plt
import cPickle
import glob
import argparse
import webbrowser

import knn
import os

import MomentosCromaticidade
import HistogramaColoridoRGB

#Inicializar Descritor
descritor=None



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--method", required = True,help = "Metodo a ser usado 1- Momentos de Cromaticidade , 2-Histograma colorido")
ap.add_argument("-d", "--datatrained", required = True,help = "Banco das imagens indexadas")
ap.add_argument("-i", "--image", required = True,help = "Nome da Imagem a ser salva")
#ap.add_argument("-c","--class",required = True,help = "Classe real da imagem a ser testada")

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

testSet=[];

for filename in glob.glob("../testes/*/*[0-9].jpg"):
    #Descreve a imagem
    classe=filename.split(os.sep)[-2]
    vecCarac = descritor.descrever(filename)
    vecCarac.append(classe)
    print(classe +"\n")
    testSet.append(vecCarac)





#nova imagem
#initialize query image
#mainImagePath=args["image"]
#mainCarac= descritor.descrever(mainImagePath)

trainingSet=index
#mainCarac.append(args["class"])
#testSet=[mainCarac]
split = 0.67
print 'Train set: ' + repr(len(trainingSet))
print 'Test set: ' + repr(len(testSet))
# generate predictions
predictions=[]
k = 3 # valor de k igual a 3

preds=[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
dictClasses={'carettacaretta': 0, 'cheloniamydas': 1,'dermochelyscoriacea':2,'eretmochelysimbricata':3,'lepidochelysolivacea':4}

numeroImagensPorClasse=6.00;



for x in range(len(testSet)):
	neighbors = knn.getNeighbors(trainingSet, testSet[x], k)
	result = knn.getResponse(neighbors)
	predictions.append(result)
	indiceClassePred=dictClasses[result]
	indiceClasseActual=dictClasses[testSet[x][-1]]
	preds[indiceClasseActual][indiceClassePred]=preds[indiceClasseActual][indiceClassePred]+1	
	print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
accuracy = knn.getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')








conf_arr = preds

norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
#ax.tick_params(labelbottom='off',labeltop='on')
res = ax.imshow(np.array(norm_conf), cmap="YlGn", 
                interpolation='nearest')

width = len(conf_arr)
height = len(conf_arr[0])

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str("%.2f" % ((conf_arr[x][y]/numeroImagensPorClasse)*100)), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')

cb = fig.colorbar(res)
#cb.ax.set_ylabel('Accuracy:'+str("%.2f" % accuracy)+'%')

plt.xticks(np.arange(0,5), ['Caretta', 'Chelonia','Dermo','Eretmo','Lepido'])
plt.yticks(np.arange(0,5), ['Caretta', 'Chelonia','Dermo','Eretmo','Lepido'])
plt.ylabel('Classe Verdadeira')
plt.xlabel('Classe Prevista')
plt.title('Matriz de Confusao: Classe x Classe - Accuracy:'+str("%.2f" % accuracy)+'%')

#plt.xticks(range(width), alphabet[:width])
#plt.yticks(range(height), alphabet[:height])
plt.savefig(args["image"]+'.png', format='png')





















