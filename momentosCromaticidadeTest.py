import cv2
import numpy as np
from matplotlib import pyplot as plt
import cPickle
import glob
import argparse
import webbrowser
from descritores import MomentosCromaticidade

#Inicializar Descritor RGB
descMC=MomentosCromaticidade


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--datatrained", required = True,help = "Banco das imagens indexadas")
ap.add_argument("-i", "--image", required = True,help = "Imagem a ser usada como query")
ap.add_argument("-o","--open", action='store_true',required = False,help = "Abrir resultado no Navegador Web")
ap.add_argument("-c","--calc",required = False,help = "Tipo de comparacao: 1- Distancia ChiSquared(default); 2-Distancia de Manhattan")

args = vars(ap.parse_args())

# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
arquivo=open( args["datatrained"], "rb" )
index = cPickle.load( arquivo )
arquivo.close()
images = {}

histNumBins=256



#nova imagem
#initialize query image
mainImagePath=args["image"]
mainCarac= descMC.descrever(mainImagePath,5,5)

print len(mainCarac)


# inicializa o metodo de comparacao a ser usado
nome_metodo="Chi-Squared"
comparar=descMC.comparar_chi2_distance

if(args["calc"] and args["calc"]=='2' ):
    nome_metodo="Manhattan-Distance"
    comparar=descMC.comparar_manhattan_distance
    print "AAAA"

        



#Inicializa resultado
histReport="""<html> <head> </head>

<body>
<h1>Query Usando Momentos de Cromaticidade e """+nome_metodo+""" como metodo de comparacao</h1>
<div style=\"border:1px solid black; height:90px\">
                                
				<div style=\"float:left; padding-right:5px;\">
				<img src=\""""+mainImagePath+"""\"/>
				</div>
				<div>
				<p>Query Image:"""+mainImagePath+"""</p>
		
				</div>
		</div><\br>"""
# inicializa o dicionario de resultados
results = {}
reverse = False






#Calcula a distancia de todos os vetores de caracteristicas com a imagem query
for (name, vecCarac) in index.items():
		# Calcula a distancia de cada vetor de caracteristica com o vetor da imagem de query
		d = comparar(mainCarac, vecCarac)
		results[name] = d

# organiza os resultados em ordem decrescente
results = sorted([(v, k) for (k, v) in results.items()], reverse = False)
# imprime resultados
for (i,(value, name)) in enumerate(results):
		# show the result
		print  nome_metodo ,name
		histText=""
		histReportLine=""
		histReportLine+= """<div style=\"border:1px solid black; height:90px\">
				<div style=\"float:left; padding-right:5px;\">
				<img src=\""""+name+"""\"/>
				</div>
				<div>
				<p>"""+name+"""</p>
				<span style="float:top;\">Distancia:
				"""+str(value)+"""
		</span>
				</div>
		</div>"""
		histReport+=histReportLine

histReport+="</body></html>"
text_filename=".\\relatorios\\"+nome_metodo+"_MomentosCromaticidade.html"
text_file = open(text_filename, "w")
text_file.write(histReport)
text_file.close()
if(args["open"]):
    webbrowser.open(text_filename, new=0, autoraise=True)
