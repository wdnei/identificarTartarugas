"""
    Leave One Out
    Arquivo avaliacao do metodos usando validacao cruzada  e o classificador knn ou svm
"""
from random import shuffle
import cv2
import numpy as np
from matplotlib import pyplot as plt
import cPickle
import glob
import argparse
import webbrowser


import os

import MomentosCromaticidade
import HistogramaColoridoRGB
import classificador
import desenhagraficos




def calcularPrecisao(trainingSet,trainingResponse,testSet,testResponse,tipoClassificador):
    predictions=[]
    preds=[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    dictClasses={'carettacaretta': 0, 'cheloniamydas': 1,'dermochelyscoriacea':2,'eretmochelysimbricata':3,'lepidochelysolivacea':4}
    clf = None
    if(tipoClassificador==1):
        clf = classificador.KNN()
    else:
        clf = classificador.SVM()

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



args = vars(ap.parse_args())


allSet=[];

for filename in glob.glob("../samples/*/*_[0-9]_[0-9].jpg"):
    #Descreve a imagem
    classe=filename.split(os.sep)[-2]
    allSet.append(filename)
    #count=count+1

#allSet=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
#allSet=allSet[0:2]
metodo=int(args["metodo"])
tipoClassificador=int(args["classificador"])
rest=fold=len(allSet)
acumulador=0
precisao=0
count=0

nomeArquivo='leave_one_out_result/leave_one_out_'+( "KNN" if tipoClassificador==1 else "SVM")+"_"+( "MoCr" if metodo==1 else "HiCo")

arquivo = open(nomeArquivo+'.txt', 'w')

preds=[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]#vetor para gerar matriz con
dictClasses={'carettacaretta': 0, 'cheloniamydas': 1,'dermochelyscoriacea':2,'eretmochelysimbricata':3,'lepidochelysolivacea':4}
dictClassesIndex={0:'carettacaretta', 1:'cheloniamydas', 2:'dermochelyscoriacea',3:'eretmochelysimbricata',4:'lepidochelysolivacea'}


for train,test in classificador.k_fold_cross_validation(allSet, fold, True):
    trainingSet,trainingResponse=classificador.carregar_imagens(train,metodo)
    testSet,testResponse=classificador.carregar_imagens(test,metodo)
    precisao,result=calcularPrecisao(trainingSet,trainingResponse,testSet,testResponse,tipoClassificador)
    for index in range(len(result)):
        indiceClassePredita=result[index]
        indiceClasseReal=testResponse[index]
        preds[indiceClasseReal][indiceClassePredita]+=1

    arquivo.write("Treino:%s\n\n" % train)
    arquivo.write("Teste:%s\n\n" % test)
    arquivo.write("Verdadeiro:%s\n\n" % testResponse)
    arquivo.write("Classificado:%s\n\n" % result)
    rest=rest-1
    print "resta:",rest

dictClassesIndex={0:'carettacaretta', 1:'cheloniamydas', 2:'dermochelyscoriacea',3:'eretmochelysimbricata',4:'lepidochelysolivacea'}
arquivo.write("Indices->Classes:%s\n\n" % dictClassesIndex)


arquivo.write("Matriz Confusao:%s\n\n" % preds)

arquivo.write("Tabela Confusao:[verdadeiro_positivo,falso_positivo,verdadeiro_negativo,falso_negativo]\n")

tabelaConfusao=classificador.gerarTabelaConfusao(preds)
print(tabelaConfusao)
for index in range(5):
    tabelaClasse=tabelaConfusao[index]# tupla=>("indice da classe","vetor de valores:verdadeiro_positivo,falso_positivo,verdadeiro_negativo,falso_negativo")
    classeNome=dictClassesIndex[tabelaClasse[0]]
    verdadeiro_positivo,falso_positivo,verdadeiro_negativo,falso_negativo=tabelaClasse[1]
    #calcular Precision
    precision=verdadeiro_positivo/(verdadeiro_positivo+falso_positivo*1.0)
    #calcular Recall
    recall=verdadeiro_positivo/(verdadeiro_positivo+falso_negativo*1.0)
    #calcular F-measure
    fmeasure=2*((precision*recall)/(precision+recall*1.0))
    #calcular Accuracy
    accuracy=(verdadeiro_positivo+verdadeiro_negativo)/(verdadeiro_positivo+falso_positivo+verdadeiro_negativo+falso_negativo*1.0)
    #salvar valores em arquivo
    arquivo.write("%s:" % classeNome)
    arquivo.write("%s\n" % tabelaClasse[1])
    arquivo.write("Precision:%.2f\n" % precision)
    arquivo.write("Recall:%.2f\n" % recall)
    arquivo.write("F-measure:%.2f\n" % fmeasure)
    arquivo.write("Accuracy:%.2f\n\n" % accuracy)




arquivo.close()

titulo="Leave One Out "+( "Momentos de Cromaticidade" if metodo==1 else "Histograma Colorido")+"-> "+( "KNN" if tipoClassificador==1 else "SVM")
desenhagraficos.gerarMatrizConfusao(preds,titulo,nomeArquivo)
