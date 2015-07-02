import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
import descritores as d

img=cv2.imread("samples\carettacaretta\carettacaretta_1.jpg")

desc= d.HistogramaColoridoRGB(64)
r=desc.descrever(img)
print r


