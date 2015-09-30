=>Montar Ambiente de Densenvolvimento
Python (versão 2.7) é a linguagem de programação usada neste trabalho. 
E a biblioteca OpenCV-Python foi usada para fazer a manipulação de imagens. 

Para instalação da biblioteca OpenCV-Python ver o link:
http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html#installing-opencv-from-prebuilt-binaries

=> O Código deste trabalho esta disponível em: https://github.com/wdnei/identificarTartarugas

============> Comandos para executar as tarefas deste trabalho<============
=> Fazer query de conteudo usando uma imagem
Na pasta ->\identificarTartarugas>
Fazer teste de busca por imagem
----Histograma Colorido-------
python histogramaColoridoRGBTrain.py -d samples

python histogramaColoridoRGBTest.py -d histRGBtrainIndex.dmp -i samples\carettacaretta\carettacaretta_2.jpg -o


----Momentos de Cromaticidade----

python momentosCromaticidadeTrain.py -d samples

python momentosCromaticidadeTest.py -d mcTrainIndex.dmp -i samples\carettacaretta\carettacaretta_2.jpg -o


python histogramaColoridoRGBTest.py -d mcTrainIndex.dmp -i samples\carettacaretta\carettacaretta_2.jpg -o


====>Fazer Classificacao<========
Na pasta ->\identificarTartarugas\classificar>
Treinar:

python momentosCromaticidadeTreinar.py -d ..\samples\

Testar:
python momentosCromaticidadeClassificar.py -d mcTrainIndex.dmp -i ..\samples\carettacaretta\carettacaretta_2.jpg -c 'carettacaretta'

python momentosCromaticidadeClassificar.py -d mcTrainIndex.dmp -i ..\samples\cheloniamydas\cheloniamydas_1.jpg -c 'cheloniamydas'