"""python matrix_confusao.py  -m 2 -d HC_treinados.dmp -i HC_matrix_confusao_classe_x_classe"""
"""
    Arquivo avaliacao do metodos usando validacao cruzada e o classificador knn
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


class StatModel(object):
    '''parent class - starting point to add abstraction'''    
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses):
        #setting algorithm parameters
        params = dict( kernel_type = cv2.SVM_LINEAR, 
                       svm_type = cv2.SVM_C_SVC,
                       C = 1 )
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        return np.int32( [self.model.predict(s) for s in samples])



class KNN(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''
    def __init__(self):
        self.model = cv2.KNearest()

    def train(self, samples, responses):
        #setting algorithm parameters
        self.model.train(samples, responses)

    def predict(self, samples):
        ret, results, neighbours ,dist = self.model.find_nearest(samples, 3)
        return results.reshape(1,len(results))[0]
        #return np.float32( [self.model.find_nearest(s, 3) for s in samples])



def carregar_imagens(listSet,metodo):
    dictClasses={'carettacaretta': 0, 'cheloniamydas': 1,'dermochelyscoriacea':2,'eretmochelysimbricata':3,'lepidochelysolivacea':4}
    loadListSet=[]
    loadListLabel=[]
    descritor=None
    if(metodo==1):
        descritor=MomentosCromaticidade
    else:
        descritor=HistogramaColoridoRGB
    for filename in listSet:
        #Descreve a imagem
        classe=filename.split(os.sep)[-2]
        vecCarac = descritor.descrever(filename)
        #adiciona o nome da classe ao fim para comparar posteriormente
        #print(classe +"\n")
        #print vecCarac
        loadListSet.append(vecCarac)
        loadListLabel.append(dictClasses[classe])
        #print np.array(vecCarac, dtype=np.float32)
        
    
    return (np.array(loadListSet, dtype=np.float32),np.array(loadListLabel, dtype=np.int32))
    
    

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



def getClassificationPrecision(trainingSet,trainingResponse,testSet,testResponse,classificador):
    predictions=[]
    preds=[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    dictClasses={'carettacaretta': 0, 'cheloniamydas': 1,'dermochelyscoriacea':2,'eretmochelysimbricata':3,'lepidochelysolivacea':4}
    clf = None
    if(classificador==1):
        clf = KNN()
    else:
        clf = SVM()    
    
    clf.train(trainingSet,trainingResponse)
    result = clf.predict(testSet)
    print result
    mask = result==testResponse
    correct = np.count_nonzero(mask)
    precisao= correct*100.0/result.size
    print "Precisao:",precisao
    return precisao,result
    


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--metodo", required = True,help = "Metodo a ser usado 1- Momentos de Cromaticidade , 2-Histograma colorido")
ap.add_argument("-c", "--classificador", required = True,help = "Classificador a ser usado 1- KNN , 2-SVM")

ap.add_argument("-f", "--fold", required = True,help = "Numero de 'fold'")

args = vars(ap.parse_args())


allSet=[];

for filename in glob.glob("../samples/*/*_[0-9]_[0-9].jpg"):
    #Descreve a imagem
    classe=filename.split(os.sep)[-2]
    allSet.append(filename)
    #count=count+1
    
#allSet=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
metodo=int(args["metodo"]) 
classificador=int(args["classificador"]) 
rest=fold=int(args["fold"]) 
acumulador=0
precisao=0
count=0

arquivo = open('k_fold_result/K_fold_'+str(fold)+"_"+( "KNN" if classificador==1 else "SVM")+"_"+( "MoCr" if metodo==1 else "HiCo")+'.txt', 'w')    

for train,test in k_fold_cross_validation(allSet, fold, False):
    #print "training:",train
    #print "test:",test
    trainingSet,trainingResponse=carregar_imagens(train,metodo)
    testSet,testResponse=carregar_imagens(test,metodo)
    precisao,result=getClassificationPrecision(trainingSet,trainingResponse,testSet,testResponse,classificador)
    acumulador+=precisao
    count+=1
    #print len(testSet)
    arquivo.write("%s\n\n" % train)
    arquivo.write("%s\n\n" % test)
    arquivo.write("Verdadeiro:%s\n\n" % testResponse)
    arquivo.write("Predicao:%s\n\n" % result)
    arquivo.write("Presisao: %s\n\n" % precisao)
    rest=rest-1
    print "resta:",rest




print "Fold:",fold
arquivo.write("Media de Precisao:" + str("%.2f" % (acumulador/float(count))))
print "Media:",acumulador/count
arquivo.close()
#Inicializar Descritor
descritor=None



















