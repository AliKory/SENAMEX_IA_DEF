import cv2
import os
import numpy as np
import time

def obtenerModelo(method, facesData, labels):
    if method == 'EigenFaces':
        emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
    elif method == 'FisherFaces':
        emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
    elif method == 'LBPH':
        emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()
    else:
        print("Método no soportado")
        return

    # Entrenando el reconocedor de rostros
    print("Entrenando ( " + method + " )...")
    inicio = time.time()
    
    if len(set(labels)) < 2:
        print(f"Error: Se requieren al menos dos clases para el método {method}.")
        return
    
    emotion_recognizer.train(facesData, np.array(labels))
    tiempoEntrenamiento = time.time() - inicio
    print("Tiempo de entrenamiento ( " + method + " ): ", tiempoEntrenamiento)

    # Almacenando el modelo obtenido
    emotion_recognizer.write("modelo" + method + ".xml")

dataPath = 'C:/Users/Lenovo/Desktop/ESTADIAS/PROYECTO/ENTREGA 070824/dataemotions'
emotionsList = os.listdir(dataPath)
print('Lista de personas: ', emotionsList)

labels = []
facesData = []
label = 0

for nameDir in emotionsList:
    emotionsPath = os.path.join(dataPath, nameDir)

    for fileName in os.listdir(emotionsPath):
        labels.append(label)
        facesData.append(cv2.imread(os.path.join(emotionsPath, fileName), 0))
    
    label += 1

# Debugging output
print('Número de clases:', len(set(labels)))

obtenerModelo('EigenFaces', facesData, labels)
obtenerModelo('FisherFaces', facesData, labels)
obtenerModelo('LBPH', facesData, labels)
