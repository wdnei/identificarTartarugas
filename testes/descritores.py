import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
import glob
import argparse
import math


class HistogramaColoridoRGB:
        @staticmethod
        def chi2_distance(histA, histB, eps = 1e-10):
                #compute the chi-squared distance
                d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
                # return the chi-squared distance
                return d

        def read_image(self,filename):
                """Retorna imagem RGB usando a biblioteca opencv
        
                    
                 """
                imagem = cv2.imread(filename)
                imagem= cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
                return imagem



        
        def get_histograma3D(self,filename,numBins):
                """Retorna o histogram 3D completo da image RGB
        
                    Keyword arguments:
                    imagem -- imagem a ser transformada em hitograma
                    return -- histograma 3D
                 """
                imagem= HistogramaColoridoRGB.read_image(filename)
                return cv2.calcHist([imagem], [0, 1, 2], None,[numBins,numBins,numBins],[0, 256, 0, 256, 0, 256]).flatten()

        
        def get_histograma(self,filename,numBins):
                """Retorna o histogram 1D de cada canal(R,G,B) concatenado em um unico vetor completo da image RGB
        
                    Keyword arguments:
                    imagem -- imagem a ser transformada em hitograma
                    return -- histograma 1
                 """
                imagem= HistogramaColoridoRGB.read_image(filename)
                result=[]
                hist_r=cv2.calcHist([imagem], [0], None,[numBins],[0, 256])
                hist_g=cv2.calcHist([imagem], [1], None,[numBins],[0, 256])
                hist_b=cv2.calcHist([imagem], [2], None,[numBins],[0, 256])
                result=np.append(result,[hist_r,hist_g,hist_b])
                return result

        

        
        def normalizar_hitograma(self,hist):
                return cv2.normalize(hist)

        
        def densidade_probabilidade(self,valor,N,M):
                return valor/(N*M)

        
        def descrever_estatistico(self,filename,numBins):
                """Descreve o histograma usando metodos estatisticos
        
                    Keyword arguments:
                    imagem -- imagem criada por imread(src) padrao RGB opencv
                    return -- [Media,Variancia,Curtose,Energia,Entropia] para cada canal
                 """
                imagem= HistogramaColoridoRGB.read_image(filename)
                result=[]
                hist_r=cv2.calcHist([imagem], [0], None,[numBins],[0, 256])
                hist_g=cv2.calcHist([imagem], [1], None,[numBins],[0, 256])
                hist_b=cv2.calcHist([imagem], [2], None,[numBins],[0, 256])
                #Dimensoes da Imagem
                N, M = imagem.shape[:2]
               

                #densidade de probabilidade
                #RED
                p_r=[]
                p_g=[]
                p_b=[]
                for index in range(0,numBins):
                        p_r.append(HistogramaColoridoRGB.densidade_probabilidade(hist_r[index],N,M))
                        p_g.append(HistogramaColoridoRGB.densidade_probabilidade(hist_g[index],N,M))
                        p_b.append(HistogramaColoridoRGB.densidade_probabilidade(hist_b[index],N,M))
                
                 #Media
                media_r=0
                media_g=0
                media_b=0
                for index in range(0,numBins):   # default is zero
                        media_r+=index*p_r[index]
                        media_g+=index*p_g[index]
                        media_b+=index*p_b[index]

                result.extend([media_r,media_g,media_b])

                #Variancia
                variancia_r=0
                variancia_g=0
                variancia_b=0
                

                for index in range(0,numBins):   # default is zero
                        variancia_r+=math.pow((index-media_r),2)*p_r[index]
                        variancia_g+=math.pow((index-media_g),2)*p_g[index]
                        variancia_b+=math.pow((index-media_b),2)*p_b[index]

                result.extend([variancia_r,variancia_g,variancia_b])
                #Curtose
                curtose_r=0
                curtose_g=0
                curtose_b=0
                

                for index in range(0,numBins):   # default is zero
                        curtose_r+=(math.pow((index-media_r),4)*p_r[index])-3
                        curtose_g+=(math.pow((index-media_g),4)*p_g[index])-3
                        curtose_b+=(math.pow((index-media_b),4)*p_b[index])-3

                curtose_r=math.pow(variancia_r,-8)*curtose_r
                curtose_g=math.pow(variancia_g,-8)*curtose_g
                curtose_b=math.pow(variancia_b,-8)*curtose_b
                result.extend([curtose_r,curtose_g,curtose_b])
                #Energia
                energia_r=0
                energia_g=0
                energia_b=0

                for index in range(0,numBins):   # default is zero
                        energia_r+=math.pow(p_r[index],2)
                        energia_g+=math.pow(p_g[index],2)
                        energia_b+=math.pow(p_b[index],2)

                result.extend([energia_r,energia_g,energia_b])
                #Entropia
                entropia_r=0
                entropia_g=0
                entropia_b=0

                for index in range(0,numBins):   # default is zero
                        entropia_r+=0 if p_r[index]<=0 else p_r[index]*(math.log(p_r[index],2))
                        entropia_g+=0 if p_g[index]<=0 else p_g[index]*(math.log(p_g[index],2))
                        entropia_b+=0 if p_b[index]<=0 else p_b[index]*(math.log(p_b[index],2))
                        
                                   
                entropia_r=-entropia_r
                entropia_g=-entropia_g
                entropia_b=-entropia_b
                result.extend([entropia_r,entropia_g,entropia_b])

                return result

        @staticmethod
        def comparar(histA,histB):
                return HistogramaColoridoRGB.chi2_distance(histA,histB)
                




class MomentosCromaticidade:
        """Esta Classe implementa o algoritmo de
           um descritor de imagem usando diagrama de cromaticidade,
           como proposto por George Paschos     
        """
        def read_image(self,filename):
                """Retorna imagem XYZ usando a biblioteca opencv     
                    
                """
                imagem = cv2.imread(filename)
                imagem= cv2.cvtColor(imagem, cv2.COLOR_BGR2XYZ)
                return imagem

        
        def get_xy_space(self,filename):
                """Retorna uma matriz da imagem usando as coordenadas x-y de cromaticidade     
                    
                 """
                image=MomentosCromaticidade.read_image(filename)
                rows,cols,dimensions=image.shape
                #criar a matriz que receberah as coordenadas xy, ou seja, 2 dimensoes
                xy_space_result=[ [ 0 for i in range(rows) ] for j in range(cols) ]
                for i in range(rows):
                        for j in range(cols):
                                X,Y,Z = image[i,j]
                                soma_XYZ=int(X)+int(Y)+int(Z)
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

        
        def get_T(self,xy_space):
                """Retorna o diagrama de cromaticidade para a imagem em x-y coordenadas    
                    
                 """                
                rows,cols,dimensions=xy_space.shape
                T=[ [ 0 for i in range(100) ] for j in range(100) ]
                for i in range(rows):
                        for j in range(cols):
                                x,y=xy_space[i,j]
                                T[x][y]=1
                return np.array(T,dtype=int)

        
        def get_D(self,xy_space):
                """Retorna a distribuicao do diagrama de cromaticidade para a imagem em x-y coordenadas(i.e histograma) 
                    
                 """                
                D=cv2.calcHist([xy_space], [0,1], None,[100,100],[0, 100,0,100])
                return D.astype(int)
        

        
        
        def calcular_M_t(self,T,m,l):
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

        
        def calcular_M_d(self,D,m,l):
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


        
        def gerar_combinacao(self,qtdMax):
                """ Gera um vetor de um dado maximo de numero, para ser usado no calculo de momentos
                 ex: gerar_combinacao(5)->[[0, 0], [1, 0], [0, 1], [2, 0], [0, 2]]
                 """
                if(qtdMax<=0):
                        return []
                combinacao=[[0,0]]
                for i in range(int(qtxMax/2)+1):
                        for j in range(i):
                                if(len(combinacao)<qtxMax):
                                        combinacao.append([i,j])
                                if(len(combinacao)<qtxMax):
                                        combinacao.append([j,i])
                        if(len(a)>=qtxMax):
                                break
                return combinacao
                
        def descrever(self,filename,qtdT,qtdD):
                """ Descreve uma imagem dependendo da quantidade de Momentos para T-type e para D-type    
                    filename - path da imagem
                    qtdT - quantidade de momentos para T-type 
                    qtdD- quantidade de momentos para D-type
                 """
                results=[]               
                xy_space=self.get_xy_space(filename)
                T=self.get_T(xy_space)
                D=self.get_D(xy_space)

                combinacaoT=self.gerar_combinacao(qtdT)
                for (p,q) in combinacaoT:
                        result=self.calcular_M_t(T,p,q)
                        results.append(result)

                combinacaoD=self.gerar_combinacao(qtdD)
                for (p,q) in combinacaoD:
                        result=self.calcular_M_d(T,p,q)
                        results.append(result)

                return results
                

                        
                        
                
                return 0
        

        @staticmethod
        def comparar(vecA,vecB):
                return 0










