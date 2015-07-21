import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
import glob
import argparse


class HistogramaColoridoRGB:

        def __init__(self,numBins=[8,8,8]):
                """Inicializar um Hitograma de cor

                    Keyword arguments:
                    numBins -- Quantidade de Celulas (Padrao [8,8,8]) [B,G,R]
                 """
                self.numBins=numBins                

        def get_histograma(self,imagem):
                """Retorna o histogram completo da image RGB
        
                    Keyword arguments:
                    imagem -- imagem a ser transformada em hitograma
                    return -- histograma 3D
                 """
                return cv2.calcHist([imagem], [0, 1, 2], None,self.numBins,[0, 256, 0, 256, 0, 256])
        
        def normalizar_hitograma(self,hist):
                return cv2.normalize(hist)

        def densidade_probabilidade(self,valor,N,M):
                return valor/(N*M)

        def descrever(self,imagem):
                """Descreve o histograma no metodo proposto
        
                    Keyword arguments:
                    imagem -- imagem criada por imread(src) padrao BGR opencv
                    return -- [Media,Variancia,Curtose,Energia,Entropia] para cada canal
                 """
                result=[]
                hist_r=cv2.calcHist([imagem], [2], None,self.numBins[2],[0, 256])
                hist_g=cv2.calcHist([imagem], [1], None,self.numBins[1],[0, 256])
                hist_b=cv2.calcHist([imagem], [0], None,self.numBins[0],[0, 256])
                #Dimensoes da Imagem
                N, M = imagem.shape[:2]
                #Media
                media_r=0
                media_g=0
                media_b=0

                for index, valor in enumerate(hist_r, start=0):   # default is zero
                        media_r+=index*self.densidade_probabilidade(valor,N,M)
                print media_r
                














