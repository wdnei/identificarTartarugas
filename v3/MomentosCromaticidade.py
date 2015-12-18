import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
import glob
import argparse
import math


"""Esta Classe implementa o algoritmo de
   um descritor de imagem usando diagrama de cromaticidade,
   como proposto por George Paschos
"""

def chi2_distance(histA, histB, eps = 1e-10):
        #calcular a distancia chi-squared - eh usado "eps" para evitar divisao por "zero" 
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
        # retorna a distancia chi-squared
        return d

def manhattan_distance(histA, histB):
        #calcular a distancia de manhatan ver https://en.wikipedia.org/wiki/Taxicab_geometry
        d =np.sum([ math.fabs(a-b) for (a, b) in zip(histA, histB)])
        # return the chi-squared distance
        return d

def ler_imagemXYZ(caminho_imagem):
        """Retorna imagem XYZ usando a biblioteca opencv

        """
        imagem = cv2.imread(caminho_imagem)
        imagem= cv2.cvtColor(imagem, cv2.COLOR_BGR2XYZ)
        return imagem


def gerar_xy_coordenadas(caminho_imagem):
        """Retorna uma matriz da imagem usando as coordenadas x-y de cromaticidade

         """
        image=ler_imagemXYZ(caminho_imagem)#ler a imagem ja convertida para XYZ
        numeroLinhas,numeroColunas,numeroCanais=image.shape
        #criar a matriz que recebera as coordenadas xy, ou seja, 2 dimensoes
        xy_espaco_resultado=[ [ 0 for i in range(numeroLinhas) ] for j in range(numeroColunas) ]
        for i in range(numeroLinhas):
                for j in range(numeroColunas):
                        X,Y,Z = image[i,j]
                        soma_XYZ=int(X)+int(Y)+int(Z)
                        if(soma_XYZ<=0):
                            soma_XYZ=1 #evitar divisao por zero
                        #x do diagrama de cromaticidade
                        x_c=float(X)/soma_XYZ
                        #y do diagrama de cromaticidade
                        y_c=float(Y)/soma_XYZ
                        xy_espaco_resultado[i][j]=[x_c,y_c]


        #transformar para numpy arrays
        xy_espaco_resultado=np.array(xy_espaco_resultado,dtype=float)
        #discretizar os valores entre 0 e 100
        xy_espaco_resultado*=100
        #rescalar para que os valores sejam somente inteiros
        xy_espaco_resultado=xy_espaco_resultado.astype(np.uint8)
        return xy_espaco_resultado


def gerar_T(xy_espaco):
        """Retorna o diagrama de cromaticidade para a imagem em x-y coordenadas """
        linhas,colunas,dimensions=xy_espaco.shape
        T=[ [ 0 for i in range(101) ] for j in range(101) ] # gerar uma matriz 100X100  
        
        #gera matriz T-type
        for i in range(linhas):
                for j in range(colunas):
                        x,y=xy_espaco[i,j]
                        T[x][y]=1
        return np.array(T,dtype=int)


def gerar_D(xy_espaco):
        """Retorna a distribuicao do diagrama de cromaticidade para a imagem em x-y coordenadas(i.e histograma)"""
        D=cv2.calcHist([xy_espaco], [0,1], None,[100,100],[0, 100,0,100]) #gera histograma a partir do espaco xy
        return D.astype(int)
        
def calcular_momentos_t_d(matriz_td,m,l):
        """ O valor do momento da matriz T(x,y) ou D(x,y)  para m e l
            matriz_td - Matriz com os valores de T(x,y) ou D(x,y)
            m - momentos para x^m
            l- momentos para y^l
         """
        resultado=0
        for x in range(100):
                for y in range(100):
                        resultado+= math.pow(x,m)*math.pow(y,l)*matriz_td[x,y]

        return resultado




def gerar_combinacao(qtdMax):
        """ Gera um vetor de um dado maximo de numero, para ser usado no calculo de momentos
         ex: gerar_combinacao(5)->[[0, 0], [1, 0], [0, 1], [2, 0], [0, 2]]
         """
        if(qtdMax<=0):
                return []
        combinacao=[[0,0]]
        for i in range(int(qtdMax/2)+1):
                for j in range(i):
                        if(len(combinacao)<qtdMax):
                                combinacao.append([i,j])
                        if(len(combinacao)<qtdMax):
                                combinacao.append([j,i])
                if(len(combinacao)>=qtdMax):
                        break
        return combinacao

def descrever(caminho_imagem,qtdT=5,qtdD=5):
        """ Descreve uma imagem dependendo da quantidade de Momentos para T-type e para D-type
            caminho_imagem - path da imagem
            qtdT - quantidade de momentos para T-type, 5 por padrao
            qtdD- quantidade de momentos para D-type, 5 por padrao
         """
        vetor_caracteristicas=[]
        xy_espaco=gerar_xy_coordenadas(caminho_imagem)
        T=gerar_T(xy_espaco) #gerar T-type
        D=gerar_D(xy_espaco) #gerar D-type

        combinacaoT=gerar_combinacao(qtdT) #gerar combinacao de momentos para T-type
        #calcular os momentos em T-type
        for (p,q) in combinacaoT:
                resultado=calcular_momentos_t_d(T,p,q)
                vetor_caracteristicas.append(resultado)

        combinacaoD=gerar_combinacao(qtdD)#gerar combinacao de momentos para D-type
        #calcular os momentos em T-type
        for (p,q) in combinacaoD:
                resultado=calcular_momentos_t_d(D,p,q)
                vetor_caracteristicas.append(resultado)

        return vetor_caracteristicas



def comparar_chi2_distance(vecA,vecB):
         """Compara usando a distancia chi-squared 

         """
         return chi2_distance(vecA, vecB)

def comparar_manhattan_distance(vecA,vecB):
         """Compara usando a distancia de manhatan ver https://en.wikipedia.org/wiki/Taxicab_geometry 

         """
         return manhattan_distance(vecA, vecB)
