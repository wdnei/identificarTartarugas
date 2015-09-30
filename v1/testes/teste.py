import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
import glob
import itertools




imageName="..\samples\carettacaretta\carettacaretta_4.jpg"
image=cv2.imread(imageName)
image=cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)



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
        

#xy_space_result=np.reshape(xy_space_result,[rows,cols,2])
X,Y,Z=image[50,50]
print(type(X))
soma_XYZ=int(X)+int(Y)+int(Z)
#x do diagrama de cromaticidade
x_c=float(X)/soma_XYZ
#y do diagrama de cromaticidade
y_c=float(Y)/soma_XYZ
xy_space_result=np.array(xy_space_result,dtype=float)
xy_space_result*=100
xy_space_result=xy_space_result.astype(np.uint8)
print xy_space_result.dtype
hist=cv2.calcHist([xy_space_result], [0,1], None,[100,100],[0, 100,0,100])
hist=hist.astype(int)
hist_custom=[ [ 0 for i in range(100) ] for j in range(100) ]

rows,cols,dimensions=xy_space_result.shape
for i in range(rows):
    for j in range(cols):
        x,y=xy_space_result[i,j]
        hist_custom[x][y]+=1


hist_custom=np.array(hist_custom,dtype=int)
print len(hist)
for i in range(len(hist_custom)):
    print hist_custom[i]

print np.array_equal(hist,hist_custom)
print image[50,50],xy_space_result[50][50],X,Y,Z,soma_XYZ,x_c,y_c
qtxMax=0

a=[[0,0]]
for i in range(int(qtxMax/2)+1):
    for j in range(i):
        if(len(a)<qtxMax):
            a.append([i,j])
        if(len(a)<qtxMax):
            a.append([j,i])
    
    if(len(a)>=qtxMax):
            break

print a
                
        
