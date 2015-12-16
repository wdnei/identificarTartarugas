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

def ler_imagemXYZ(filename):
        """Retorna imagem XYZ usando a biblioteca opencv

        """
        imagem = cv2.imread(filename)
        imagem= cv2.cvtColor(imagem, cv2.COLOR_BGR2XYZ)
        return imagem


def get_xy_space(filename):
        """Retorna uma matriz da imagem usando as coordenadas x-y de cromaticidade

         """
        image=ler_imagemXYZ(filename)#ler a imagem ja convertida para XYZ
        numeroLinhas,numeroColunas,numeroCanais=image.shape
        #criar a matriz que recebera as coordenadas xy, ou seja, 2 dimensoes
        xy_space_result=[ [ 0 for i in range(numeroLinhas) ] for j in range(numeroColunas) ]
        for i in range(numeroLinhas):
                for j in range(numeroColunas):
                        X,Y,Z = image[i,j]
                        soma_XYZ=int(X)+int(Y)+int(Z)
                        if(soma_XYZ<=0):
                            soma_XYZ=1
                        #x do diagrama de cromaticidade
                        x_c=float(X)/soma_XYZ
                        #y do diagrama de cromaticidade
                        y_c=float(Y)/soma_XYZ
                        xy_space_result[i][j]=[x_c,y_c]


        #transformar para numpy arrays
        xy_space_result=np.array(xy_space_result,dtype=float)
        #discretizar os valores entre 0 e 100
        xy_space_result*=100
        #rescalar para que os valores sejam somente inteiros
        xy_space_result=xy_space_result.astype(np.uint8)
        return xy_space_result


def get_T(xy_space):
        """Retorna o diagrama de cromaticidade para a imagem em x-y coordenadas

         """
        rows,cols,dimensions=xy_space.shape
        T=[ [ 0 for i in range(101) ] for j in range(101) ]
        
        for i in range(rows):
                for j in range(cols):
                        x,y=xy_space[i,j]
                        #print x,y
                        T[x][y]=1
        return np.array(T,dtype=int)


def get_D(xy_space):
        """Retorna a distribuicao do diagrama de cromaticidade para a imagem em x-y coordenadas(i.e histograma)

         """
        D=cv2.calcHist([xy_space], [0,1], None,[100,100],[0, 100,0,100])
        return D.astype(int)




def calcular_M_t(T,m,l):
        """ O valor do momento da matriz T(x,y) para m e l
            T - Matriz com os valores de T(x,y)
            m - momentos para x^m
            l- momentos para y^l
         """
        result=0
        for x in range(100):
                for y in range(100):
                        result+= math.pow(x,m)*math.pow(y,l)*T[x,y]

        return result


def calcular_M_d(D,m,l):
        """ O valor do momento da matriz D(x,y) para m e l
            D - Matriz com os valores de D(x,y)
            m - momentos para x^m
            l- momentos para y^l
         """
        result=0
        for x in range(100):
                for y in range(100):
                        result+= math.pow(x,m)*math.pow(y,l)*D[x,y]

        return result



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

def descrever(filename,qtdT=5,qtdD=5):
        """ Descreve uma imagem dependendo da quantidade de Momentos para T-type e para D-type
            filename - path da imagem
            qtdT - quantidade de momentos para T-type
            qtdD- quantidade de momentos para D-type
         """
        results=[]
        xy_space=get_xy_space(filename)
        T=get_T(xy_space)
        D=get_D(xy_space)

        combinacaoT=gerar_combinacao(qtdT)
        for (p,q) in combinacaoT:
                result=calcular_M_t(T,p,q)
                results.append(result)

        combinacaoD=gerar_combinacao(qtdD)
        for (p,q) in combinacaoD:
                result=calcular_M_d(T,p,q)
                results.append(result)

        return results



def comparar_chi2_distance(vecA,vecB):
         """Compara usando a distancia chi-squared 

         """
         return chi2_distance(vecA, vecB)

def comparar_manhattan_distance(vecA,vecB):
         """Compara usando a distancia de manhatan ver https://en.wikipedia.org/wiki/Taxicab_geometry 

         """
         return manhattan_distance(vecA, vecB)
