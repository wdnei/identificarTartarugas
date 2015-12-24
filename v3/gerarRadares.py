import radar_chart as rc


"""
rc.gerarRadar(titulo='F-measure',data=[
        ['Momentos de Cromaticidade','Histogramas Coloridos'],
        ('KNN', [
            [0.56, 0.56, 0.67, 0.35,0.25],
            [0.65, 0.69, 0.41, 0.84, 0.41]]),
        ('SVM', [
            [0.62, 0.40, 0.76, 0.37, 0.36],
            [0.45, 0.48, 0.55, 0.91, 0.06]])
        
    ])"""



rc.gerarRadar(titulo='F-measure Leave-One-Out',data=[
        ['Momentos de Cromaticidade','Histogramas Coloridos'],
        ('KNN', [
            [0.71, 0.69, 0.97, 0.71,0.63],
            [0.63, 0.67, 0.49, 0.84, 0.41]]),
        ('SVM', [
            [0.70, 0.23, 0.97, 0.42, 0.10],
            [0.47, 0.48, 0.63, 0.65, 0.19]])
        
    ])

"""
rc.gerarRadar(titulo='F-measure K-fold-10',data=[
        ['Momentos de Cromaticidade','Histogramas Coloridos'],
        ('KNN', [
            [0.67, 0.69, 0.97, 0.69,0.59],
            [0.68, 0.69, 0.38, 0.86, 0.39]]),
        ('SVM', [
            [0.68, 0.17, 0.95, 0.47, 0.50],
            [0.36, 0.64, 0.60, 0.77, 0.23]])
        
    ])
"""
