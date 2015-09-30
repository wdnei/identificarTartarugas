import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
import glob
import argparse
import descritores as desc


histRGB =desc.HistogramaColoridoRGB(256)


# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}


# loop over the image paths
histReport="""<html>
	<head>
	</head>
	
	<body>"""

for imagePath in glob.glob("./samples/*/*[0-9].jpg"):
        # extract the image filename (assumed to be unique) and
        # load the image, updating the images dictionary
        nomearquivo = imagePath.split("\\")[-1]
        image = cv2.imread(imagePath)
        images[nomearquivo] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vetorCarac=histRGB.descrever(image)
        index[nomearquivo] = vetorCarac
        histText=""
        histReportLine=""
        for key,values in vetorCarac.iteritems():
                #histText+=key +":"+value[0]+","+value[1]+","+value[2]
                histText+=key+":"+ str(values[0])+","+str(values[1])+","+str(values[2])+"</br>"
        #print histText,"\n"
        histReportLine+= """<div style=\"border:1px solid black; height:90px\">
                <div style=\"float:left; padding-right:5px;\">
                <img src=\""""+imagePath+"""\"/>
                </div>
                <div>
                <span style="float:top;\">
                """+histText+"""
        </span>
                </div>
        </div>"""
        histReport+=histReportLine

histReport+="</body></html>"
text_file = open(".\relatorios\Output.html", "w")
text_file.write(histReport)
text_file.close()
print "Relatorio concluido"























