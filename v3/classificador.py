""" Classificador
    Classe responsavel por montar os classificadores a serem usados usando a a biblioteca OpenCV
"""
from random import shuffle
import cv2
import numpy as np
import os
import cPickle
import glob
import MomentosCromaticidade
import HistogramaColoridoRGB



class StatModel(object):
    '''classe para adicionar abstracao'''
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)



class KNN(StatModel):
    '''Classe para o classificador KNN do OpenCV'''
    def __init__(self):
        self.model = cv2.KNearest()

    def train(self, trainingSet, labels):
        #configurando os classificador - fase de treinamento
        self.model.train(trainingSet, labels)

    def predict(self, testSet):
        ret, results, neighbours ,dist = self.model.find_nearest(testSet, 3) # fase de teste para k=3
        return results.reshape(1,len(results))[0].astype(int) #formatando resultados ex:[0,1,3]

class SVM(StatModel):
    '''Classe para o classificador SVM do OpenCV'''
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, trainingSet, labels):
        #configurando os classificador - fase de treinamento
        params = dict( kernel_type = cv2.SVM_LINEAR,
                       svm_type = cv2.SVM_C_SVC,
                       C = 1 )
        self.model.train(trainingSet, labels, params = params)
        
    def predict(self, samples):
        return np.int32( [self.model.predict(s) for s in samples]) #fase de teste e formatando resultados ex:[0,1,3]



        

def carregar_imagens(caminhoImagens,metodo):
    '''Funcao responsavel por ler as imagens e retornar seus respectivos vetores de caracteristicas e classes'''
    dictClasses={'carettacaretta': 0, 'cheloniamydas': 1,'dermochelyscoriacea':2,'eretmochelysimbricata':3,'lepidochelysolivacea':4}
    vecCaracList=[]#Vetor com os vetores de caracteristicas
    classeList=[]#Vetor contendo os nomes das classes dos respectivos vetores de caracteristicas
    descritor=None
    #escolha do metodo de extracao de caracteristicas
    if(metodo==1):
        descritor=MomentosCromaticidade
    else:
        descritor=HistogramaColoridoRGB
    for filename in caminhoImagens:
        #Descreve a imagem
        classe=filename.split(os.sep)[-2] #recupera a classe de acordo com o nome da pasta onde esta a imagem
        vecCarac = descritor.descrever(filename)
        vecCaracList.append(vecCarac)
        classeList.append(dictClasses[classe])

    #retornar usando o array do Numpy para facilitar o uso com a biblioteca do OpenCV
    return (np.array(vecCaracList, dtype=np.float32),np.array(classeList, dtype=np.int32))





def k_fold_cross_validation(items, k, randomize=False):
    """Funcao responsavel por gerar K-fold(dobramentos) separando em conjunto de treinamento e teste"""
    #Gerar em ordem aleatoria
    if randomize:
        items = list(items)
        shuffle(items)

    slices = [items[i::k] for i in xrange(k)]
    #Gerar os conjunto de treinamento(training) e de teste(validation)
    for i in xrange(k):
        validation = slices[i]
        training = [item
                    for s in slices if s is not validation
                    for item in s]
        yield training, validation


def testar(trainingSet,trainingResponse,testSet,tipoClassificador):
    """Calcula a taxa de acertos e retorna os resultados """
    predictions=[]
    preds=[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    dictClasses={'carettacaretta': 0, 'cheloniamydas': 1,'dermochelyscoriacea':2,'eretmochelysimbricata':3,'lepidochelysolivacea':4}
    clf = None
    if(tipoClassificador==1):
        clf = KNN()
    else:
        clf = SVM()

    #treinar
    clf.train(trainingSet,trainingResponse)
    #testar
    result = clf.predict(testSet)
 
    return result



def gerarTabelaConfusao(preds):
    '''Recupera os dados de 'verdadeiro positivo','falso positivo','verdadeiro negativo','falso negativo'
        de cada uma das 5 classes e a retorna em um vetor, sendo o primeiro indice a classe e o segundo outro vetor com os dados
    '''
    resultados=[]
    for classeIndex in range(5):# 5 classes
        verdadeiro_positivo=preds[classeIndex][classeIndex]
        falso_positivo=0
        verdadeiro_negativo=0
        falso_negativo=0
        for linha in range(5):# 5 linhas
            for coluna in range(5): #5 colunas
                if(linha==classeIndex and coluna!=classeIndex):
                    falso_negativo+=preds[linha][coluna]
                elif(coluna==classeIndex and linha!=classeIndex):
                    falso_positivo+=preds[linha][coluna]
                elif(coluna!=classeIndex and linha!=classeIndex):
                    verdadeiro_negativo+=preds[linha][coluna]

        resultados.append((classeIndex,[verdadeiro_positivo,falso_positivo,verdadeiro_negativo,falso_negativo]))

    return resultados


def validacao_cruzada_k_fold(k,tipoDescritor,tipoClassificador,caminhosImagensBase):
    '''Funcao base para a validacao cruzada K-fold, onde é passado, o numero de folds(k),o tipo de descritor
    (1-Momento de Cromaticidade 2-Histogramas coloridos), o tipo de classificador (1-KNN 2-SVM)
    e uma lista contendo o caminho das imagens, sendo estas imagens dividas em pasta com suas respectivas classes'''
    #gerar k-fold e testar imagens no classificador
    resultado=None
    for imagensTreino,imagensTeste in k_fold_cross_validation(caminhosImagensBase, k, True):
        #carregar imagens de treino
        vetorTreinamento,classesTreinamento=carregar_imagens(imagensTreino,tipoDescritor)
        #carregar imagens de teste
        vetorTeste,classesTeste=carregar_imagens(imagensTeste,tipoDescritor)
        #testar classificador
        resultado=testar(vetorTreinamento,classesTreinamento,vetorTeste,tipoClassificador)
    
    return resultado
