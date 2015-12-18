import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
import glob
import argparse
import math




def chi2_distance(histA, histB, eps = 1e-10):
        #compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
        # return the chi-squared distance
        return d

def ler_imagemRGB(caminho_imagem):
        """Retorna imagem RGB usando a biblioteca opencv
         """
         # Ler uma imagem usando OpenCV 
        imagem = cv2.imread(caminho_imagem)
        # Converter a imagem do padrao BGR para RGB
        imagem= cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        return imagem




def get_histograma3D(caminho_imagem,numBins):
        """Retorna o histogram 3D completo da image RGB

            Keyword arguments:
            imagem -- imagem a ser transformada em hitograma
            return -- histograma 3D
         """
        imagem= HistogramaColoridoRGB.ler_imagemRGB(caminho_imagem)
        return cv2.calcHist([imagem], [0, 1, 2], None,[numBins,numBins,numBins],[0, 256, 0, 256, 0, 256]).flatten()


def get_histograma(caminho_imagem,numBins):
        """Retorna o histogram 1D de cada canal(R,G,B) concatenado em um unico vetor completo da image RGB

            Keyword arguments:
            imagem -- imagem a ser transformada em hitograma
            return -- histograma 1
         """
        imagem=ler_imagemRGB(caminho_imagem)
        result=[]
        #Gerar histograma do canal R
        hist_r=cv2.calcHist([imagem], [0], None,[numBins],[0, 256])
        #Gerar histograma do canal G
        hist_g=cv2.calcHist([imagem], [1], None,[numBins],[0, 256])
        #Gerar histograma do canal B
        hist_b=cv2.calcHist([imagem], [2], None,[numBins],[0, 256])
        result=np.append(result,[hist_r,hist_g,hist_b])
        return result




def normalizar_histograma(hist):
        return cv2.normalize(hist)


def densidade_probabilidade(valor,N,M):
        """ Calcular a densidade de probabilidade 
            valor:h(i)
            N e M: sao as dimensoes da imagem"""
        return valor/(N*M)


def descrever(caminho_imagem,numBins=256):
        """Descreve uma imagem com metodo Histograma colorido usando metodos estatisticos
            
            caminho_imagem:caminho da imagem a ser criada por imread(src) padrao RGB 
            numBins:numero de bins a serem usados no histograma, 256 por padrao
            retorno:vetor de caracteristica [Media,Variancia,Curtose,Energia,Entropia] para cada canal
         """
        imagem= ler_imagemRGB(caminho_imagem)#recupera a imagem em formato RGB
        vetor_caracteristica=[]
        #Gerar histograma do canal R
        hist_r=cv2.calcHist([imagem], [0], None,[numBins],[0, 256])
        #Gerar histograma do canal G
        hist_g=cv2.calcHist([imagem], [1], None,[numBins],[0, 256])
        #Gerar histograma do canal B
        hist_b=cv2.calcHist([imagem], [2], None,[numBins],[0, 256])
        
        #Recuperar dimensoes da Imagem
        N, M = imagem.shape[:2]


        #calcular densidade de probabilidade
        p_r=[] #densidade de probabilidade para canal R(vermelho)
        p_g=[] #densidade de probabilidade para canal G(verde)
        p_b=[] #densidade de probabilidade para canal B(azul)
        for index in range(0,numBins):
                p_r.append(densidade_probabilidade(hist_r[index],N,M))
                p_g.append(densidade_probabilidade(hist_g[index],N,M))
                p_b.append(densidade_probabilidade(hist_b[index],N,M))

        #Calcular Media
        media_r=0
        media_g=0
        media_b=0
        for index in range(0,numBins):
                media_r+=index*p_r[index]
                media_g+=index*p_g[index]
                media_b+=index*p_b[index]

        vetor_caracteristica.extend([media_r,media_g,media_b])

        #Calcular Variancia
        variancia_r=0
        variancia_g=0
        variancia_b=0


        for index in range(0,numBins):
                variancia_r+=math.pow((index-media_r),2)*p_r[index]
                variancia_g+=math.pow((index-media_g),2)*p_g[index]
                variancia_b+=math.pow((index-media_b),2)*p_b[index]

        vetor_caracteristica.extend([variancia_r,variancia_g,variancia_b])
        
        #Calcular Curtose
        curtose_r=0
        curtose_g=0
        curtose_b=0


        for index in range(0,numBins):
                curtose_r+=(math.pow((index-media_r),4)*p_r[index])-3
                curtose_g+=(math.pow((index-media_g),4)*p_g[index])-3
                curtose_b+=(math.pow((index-media_b),4)*p_b[index])-3

        curtose_r=math.pow(variancia_r,-8)*curtose_r
        curtose_g=math.pow(variancia_g,-8)*curtose_g
        curtose_b=math.pow(variancia_b,-8)*curtose_b
        vetor_caracteristica.extend([curtose_r,curtose_g,curtose_b])
        
        #Calcular Energia
        energia_r=0
        energia_g=0
        energia_b=0

        for index in range(0,numBins):
                energia_r+=math.pow(p_r[index],2)
                energia_g+=math.pow(p_g[index],2)
                energia_b+=math.pow(p_b[index],2)

        vetor_caracteristica.extend([energia_r,energia_g,energia_b])
        
        #Calcular Entropia
        entropia_r=0
        entropia_g=0
        entropia_b=0

        for index in range(0,numBins):
                entropia_r+=0 if p_r[index]<=0 else p_r[index]*(math.log(p_r[index],2))
                entropia_g+=0 if p_g[index]<=0 else p_g[index]*(math.log(p_g[index],2))
                entropia_b+=0 if p_b[index]<=0 else p_b[index]*(math.log(p_b[index],2))


        entropia_r=-entropia_r
        entropia_g=-entropia_g
        entropia_b=-entropia_b
        vetor_caracteristica.extend([entropia_r,entropia_g,entropia_b])

        return vetor_caracteristica


def comparar(histA,histB):
        return chi2_distance(histA,histB)
