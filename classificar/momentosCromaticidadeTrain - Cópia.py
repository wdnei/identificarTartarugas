import cv2
import numpy as np
from matplotlib import pyplot as plt
import cPickle
import glob
import argparse

# loop over the image paths
for filename in glob.glob("../samples/*/*[0-9].jpg"):
    #Descreve a imagem    
    print filename.split("\\")[-2];
