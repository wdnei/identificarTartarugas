import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle


img = cv2.imread('foto3x4.jpg')
color = ('b','g','r')
histr={} 
for i,col in enumerate(color):
    histr[col]=cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr[col],color = col)
    plt.xlim([0,256])
cv2.imshow('image',img)
plt.show()
pickle.dump( histr, open( "save.p", "wb" ) )
cv2.waitKey(0)
cv2.destroyAllWindows()

