"""python matrix_confusao.py  -m 2 -d HC_treinados.dmp -i HC_matrix_confusao_classe_x_classe"""
"""
    Arquivo para criacao de matriz de confusao classesXclasses, no qual eh calculado tambem a acuracia
ver http://www.analyticsvidhya.com/blog/2015/05/k-fold-cross-validation-simple/
"""
from random import shuffle
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


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--method", required = True,help = "Metodo a ser usado 1- Momentos de Cromaticidade , 2-Histograma colorido")
ap.add_argument("-f", "--fold", required = True,help = "Numero de 'fold'")

args = vars(ap.parse_args())



def load_files(listSet,method):
    loadListSet=[]
    descritor=None
    if(method==1):
        descritor=MomentosCromaticidade
    else:
        descritor=HistogramaColoridoRGB
    for filename in listSet:
        #Descreve a imagem
        classe=filename.split(os.sep)[-2]
        vecCarac = descritor.descrever(filename)
        vecCarac.append(classe) #adiciona o nome da classe ao fim para comparar posteriormente
        #print(classe +"\n")
        loadListSet.append(vecCarac)
    
    return loadListSet
    
    

def k_fold_cross_validation(items, k, randomize=False):

    if randomize:
        items = list(items)
        shuffle(items)

    slices = [items[i::k] for i in xrange(k)]

    for i in xrange(k):
        validation = slices[i]
        training = [item
                    for s in slices if s is not validation
                    for item in s]
        yield training, validation



def getClassificationPrecision(trainingSet,testSet,k=3):
    predictions=[]
    preds=[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    dictClasses={'carettacaretta': 0, 'cheloniamydas': 1,'dermochelyscoriacea':2,'eretmochelysimbricata':3,'lepidochelysolivacea':4}
    
    for x in range(len(testSet)):
        neighbors = knn.getNeighbors(trainingSet, testSet[x], k)
        result = knn.getResponse(neighbors)
        #print result
        predictions.append(result)
        indiceClassePred=dictClasses[result]
        indiceClasseActual=dictClasses[testSet[x][-1]]
        preds[indiceClasseActual][indiceClassePred]=preds[indiceClasseActual][indiceClassePred]+1
    
    #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    #print predictions
    precisao = knn.getPrecision(testSet, predictions)
    #print('Accuracy: ' + repr(accuracy) + '%')
    #quit()
    return precisao
    


allSet=[];

for filename in glob.glob("../samples/*/*_[0-9]_[0-9].jpg"):
    #Descreve a imagem
    classe=filename.split(os.sep)[-2]
    allSet.append(filename)
    #count=count+1
    
#allSet=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
method=int(args["method"]) 
rest=fold=int(args["fold"]) 
acumulador=0
precisao=0
count=0

arquivo = open('k_fold_result/K_fold_'+str(fold)+"_"+( "MoCr" if method==1 else "HiCo")+'.txt', 'w')    

for train,test in k_fold_cross_validation(allSet, fold, True):
    #print "training:",train
    #print "test:",test
    trainingSet=load_files(train,method)
    testSet=load_files(test,method)
    precisao=getClassificationPrecision(trainingSet,testSet)
    acumulador+=precisao
    count+=1
    #print len(testSet)
    arquivo.write("%s\n\n" % train)
    arquivo.write("%s\n\n" % test)
    arquivo.write("Presisao: %s\n\n" % precisao)
    rest=rest-1
    print "resta:",rest




print "Fold:",fold
arquivo.write("Media de Precisao:" + str("%.2f" % (acumulador/float(count))))
print "Media:",acumulador/count
arquivo.close()
#Inicializar Descritor
descritor=None



quit()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--method", required = True,help = "Metodo a ser usado 1- Momentos de Cromaticidade , 2-Histograma colorido")
#ap.add_argument("-d", "--datatrained", required = True,help = "Banco das imagens indexadas")
#ap.add_argument("-i", "--image", required = True,help = "Nome da Imagem a ser salva")
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

testSet=[]
validationSet=[]

for filename in glob.glob("../samples/*/*_[0-9]_[0-9].jpg"):
    #Descreve a imagem
    classe=filename.split(os.sep)[-2]
    vecCarac = descritor.descrever(filename)
    vecCarac.append(classe) #coloca o nome da classe ao fim do vetor para ser usado posteriormente
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

numeroImagensPorClasse=30.00;



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
        tmp_arr.append((float(j)/float(a))*100)
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





















