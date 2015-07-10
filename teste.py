import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
import descritores as d
import glob



image1=cv2.imread(".\samples\carettacaretta\carettacaretta_1.jpg")
image1=cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

image3=cv2.imread(".\samples\carettacaretta\carettacaretta_1.jpg")
image3=cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

r3,g3,b3=cv2.split(image3)
hist3=cv2.calcHist([r3], [0], None, [256],
		[0, 256]).flatten()

r1,g1,b1=cv2.split(image1)
hist1=cv2.calcHist([r1], [0], None, [256],
		[0, 256]).flatten()



hist2=pickle.load( open( "save.p", "r" ) )

for i in range(len(hist1)):
        if(hist1[i]!=hist3[i]):
                print "hhh",i





