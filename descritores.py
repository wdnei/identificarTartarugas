import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
import glob
import argparse
import math


class HistogramaColoridoRGB:

        def __init__(self,numBins=256):
                """Inicializar um Hitograma de cor

                    Keyword arguments:
                    numBins -- Quantidade de Celulas (Padrao 256)
                 """
                self.numBins=numBins                

        def get_histograma(self,imagem):
                """Retorna o histogram completo da image RGB
        
                    Keyword arguments:
                    imagem -- imagem a ser transformada em hitograma
                    return -- histograma 3D
                 """
                return cv2.calcHist([imagem], [0, 1, 2], None,[self.numBins,self.numBins,self.numBins],[0, 256, 0, 256, 0, 256])
        
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
                hist_r=cv2.calcHist([imagem], [2], None,[self.numBins],[0, 256])
                hist_g=cv2.calcHist([imagem], [1], None,[self.numBins],[0, 256])
                hist_b=cv2.calcHist([imagem], [0], None,[self.numBins],[0, 256])
                #Dimensoes da Imagem
                N, M = imagem.shape[:2]
                #Media
                media_r=0
                media_g=0
                media_b=0

                #densidade de probabilidade
                #RED
                p_r=[]
                p_g=[]
                p_b=[]
                for index in range(0,self.numBins):
                        p_r.append(self.densidade_probabilidade(hist_r[index],N,M))
                        p_g.append(self.densidade_probabilidade(hist_g[index],N,M))
                        p_b.append(self.densidade_probabilidade(hist_b[index],N,M))
                

                for index in range(0,self.numBins):   # default is zero
                        media_r+=index*p_r[index]
                        media_g+=index*p_g[index]
                        media_b+=index*p_b[index]

                result.append([media_r,media_g,media_b])

                #Variancia
                variancia_r=0
                variancia_g=0
                variancia_b=0
                

                for index in range(0,self.numBins):   # default is zero
                        variancia_r+=math.pow((index-media_r),2)*p_r[index]
                        variancia_g+=math.pow((index-media_g),2)*p_g[index]
                        variancia_b+=math.pow((index-media_b),2)*p_b[index]

                result.append([variancia_r,variancia_g,variancia_b])
                #Curtose
                curtose_r=0
                curtose_g=0
                curtose_b=0
                

                for index in range(0,self.numBins):   # default is zero
                        curtose_r+=(math.pow((index-media_r),4)*p_r[index])-3
                        curtose_g+=(math.pow((index-media_g),4)*p_g[index])-3
                        curtose_b+=(math.pow((index-media_b),4)*p_b[index])-3

                curtose_r=math.pow(variancia_r,-8)*curtose_r
                curtose_g=math.pow(variancia_g,-8)*curtose_g
                curtose_b=math.pow(variancia_b,-8)*curtose_b
                result.append([curtose_r,curtose_g,curtose_b])
                #Energia
                energia_r=0
                energia_g=0
                energia_b=0

                for index in range(0,self.numBins):   # default is zero
                        energia_r+=math.pow(p_r[index],2)
                        energia_g+=math.pow(p_g[index],2)
                        energia_b+=math.pow(p_b[index],2)

                result.append([energia_r,energia_g,energia_b])
                #Entropia
                entropia_r=0
                entropia_g=0
                entropia_b=0

                for index in range(0,self.numBins):   # default is zero
                        entropia_r+=0 if p_r[index]<=0 else p_r[index]*(math.log(p_r[index],2))
                        entropia_g+=0 if p_g[index]<=0 else p_g[index]*(math.log(p_g[index],2))
                        entropia_b+=0 if p_b[index]<=0 else p_b[index]*(math.log(p_b[index],2))
                        
                                   
                entropia_r=-entropia_r
                entropia_g=-entropia_g
                entropia_b=-entropia_b
                result.append([entropia_r,entropia_g,entropia_b])

                return result













