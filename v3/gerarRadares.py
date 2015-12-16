import radar_chart as rc


"""
rc.gerarRadar(titulo='F-measure',data=[
        ['Momentos de Cromaticidade','Histograma de Cor'],
        ('KNN', [
            [0.56, 0.56, 0.67, 0.35,0.25],
            [0.65, 0.69, 0.41, 0.84, 0.41]]),
        ('SVM', [
            [0.62, 0.40, 0.76, 0.37, 0.36],
            [0.45, 0.48, 0.55, 0.91, 0.06]])
        
    ])"""

rc.gerarRadar(titulo='Precision k-fold-10',data=[
        ['Momentos de Cromaticidade','Histograma de Cor'],
        ('KNN', [
            [0.48, 0.64, 0.62, 0.38,0.29],
            [0.63, 0.79, 0.38, 0.80, 0.44]]),
        ('SVM', [
            [0.52, 0.41, 0.63, 0.56, 0.50],
            [0.41, 0.53, 0.50, 0.94, 0.07]])
        
    ])


