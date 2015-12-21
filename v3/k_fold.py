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


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--metodo", required = True,help = "Metodo a ser usado 1- Momentos de Cromaticidade , 2-Histograma colorido")
ap.add_argument("-c", "--classificador", required = True,help = "Classificador a ser usado 1- KNN , 2-SVM")
ap.add_argument("-f", "--fold", required = True,help = "K-fold do classificador(numero inteiro)")
ap.add_argument("-n", "--nome", required = False,help = "Nome da Imagem e local da imagem")
ap.add_argument("-t", "--titulo", required = False,help = "Titulo da Matriz de confusao")



args = vars(ap.parse_args())


allSet=[];

for filename in glob.glob("samples/*/*_[0-9]_[0-9].jpg"):
    #Descreve a imagem
    classe=filename.split(os.sep)[-2]
    allSet.append(filename)
    #count=count+1

#allSet=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
#allSet=allSet[0:2]
metodo=int(args["metodo"])
tipoClassificador=int(args["classificador"])
rest=fold=int(args["fold"])
acumulador=0
precisao=0
count=0

nomeArquivo=args["nome"]
if(nomeArquivo==None):
    nomeArquivo='k_fold_resultado/k_fold_'+str(fold)+"_"+( "KNN" if tipoClassificador==1 else "SVM")+"_"+( "MoCr" if metodo==1 else "HiCo")
    
print nomeArquivo

arquivo = open(nomeArquivo+'.txt', 'w')

matrizConfusao=[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]#vetor para gerar matriz confusao
classe2index={'carettacaretta': 0, 'cheloniamydas': 1,'dermochelyscoriacea':2,'eretmochelysimbricata':3,'lepidochelysolivacea':4}
index2classe={0:'carettacaretta', 1:'cheloniamydas', 2:'dermochelyscoriacea',3:'eretmochelysimbricata',4:'lepidochelysolivacea'}

#gerar k-fold e testar imagens no classificador
for imagensTreino,imagensTeste in classificador.k_fold_cross_validation(allSet, fold, True):
    #carregar imagens de treino
    vetorTreinamento,classesTreinamento=classificador.carregar_imagens(imagensTreino,metodo)
    #carregar imagens de teste
    vetorTeste,classesTeste=classificador.carregar_imagens(imagensTeste,metodo)
    #testar classificador
    resultado=classificador.testar(vetorTreinamento,classesTreinamento,vetorTeste,tipoClassificador)
    mascara = resultado==classesTeste
    acertos = np.count_nonzero(mascara)
    porcentagem_acertos= acertos*100.0/resultado.size
    for index in range(len(resultado)):
        indiceClassePredita=resultado[index]
        indiceClasseReal=classesTeste[index]
        matrizConfusao[indiceClasseReal][indiceClassePredita]+=1

    arquivo.write("Treino:%s\n\n" % imagensTreino)
    arquivo.write("Teste:%s\n\n" % imagensTeste)
    arquivo.write("Verdadeiro:%s\n\n" % classesTeste)
    arquivo.write("Classificado:%s\n\n" % resultado)
    arquivo.write("Taxa de Acerto:%.2f\n\n" % porcentagem_acertos)
    rest=rest-1
    print "resta:",rest

index2classe={0:'carettacaretta', 1:'cheloniamydas', 2:'dermochelyscoriacea',3:'eretmochelysimbricata',4:'lepidochelysolivacea'}
arquivo.write("Indices->Classes:%s\n\n" % index2classe)


arquivo.write("Matriz Confusao:%s\n\n" % matrizConfusao)

arquivo.write("Tabela Confusao:[verdadeiro_positivo,falso_positivo,verdadeiro_negativo,falso_negativo]\n")

tabelaConfusao=classificador.gerarTabelaConfusao(matrizConfusao)
print(tabelaConfusao)
for index in range(5):
    tabelaClasse=tabelaConfusao[index]# tupla=>("indice da classe","vetor de valores:verdadeiro_positivo,falso_positivo,verdadeiro_negativo,falso_negativo")
    classeNome=index2classe[tabelaClasse[0]]
    verdadeiro_positivo,falso_positivo,verdadeiro_negativo,falso_negativo=tabelaClasse[1]
    #calcular Precision
    precision=verdadeiro_positivo/(verdadeiro_positivo+falso_positivo*1.0)
    #calcular Recall
    recall=verdadeiro_positivo/(verdadeiro_positivo+falso_negativo*1.0)
    #calcular F-measure
    fmeasure=(float(2*precision*recall)/(precision+recall*0.0000000000000000001))
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

titulo=args["titulo"]
if(titulo==None):
    titulo="K-fold "+str(fold)+ ": "+( "Mo. de Cromaticidade" if metodo==1 else "Histograma Colorido")+"-> "+( "KNN" if tipoClassificador==1 else "SVM")

desenhagraficos.gerarMatrizConfusao(matrizConfusao,titulo,nomeArquivo)
