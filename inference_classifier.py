import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import os
import random

# Inicializar el motor de texto a voz
engine = pyttsx3.init()

# Cargar el modelo entrenado para gestos
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

# Configuración de MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configurar el modelo de manos
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Diccionario de etiquetas para los gestos
labels_dict = {
    0: ' ', 1: ' ', 2: 'saludos', 3: 'me', 4: 'llamo', 5: 'a', 6: 'b', 7: 'c', 8: 'd',
    9: 'e', 10: 'f', 11: 'g', 12: 'h', 13: 'i', 14: 'j', 15: 'k', 16: 'l', 17: 'm', 18: 'n',
    19: 'ene', 20: 'o', 21: 'p', 22: 'q', 23: 'r', 24: 's', 25: 't', 26: 'u', 27: 'v', 28: 'w',
    29: 'y', 30: 'Yo', 31: 'Tu', 32: 'Nosotros', 33: 'Ustedes', 34: 'Ella', 35: 'Hola'
}

# Cargar el modelo entrenado para reconocimiento de emociones
method = 'LBPH'
if method == 'EigenFaces':
    emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
elif method == 'FisherFaces':
    emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
elif method == 'LBPH':
    emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

emotion_recognizer.read('modelo' + method + '.xml')

# Función para cargar los emojis según la emoción
def emotionImage(emotion):
    if emotion == 'Felicidad': image = cv2.imread('Emojis/felicidad.jpeg')
    if emotion == 'Enojo': image = cv2.imread('Emojis/enojo.jpeg')
    if emotion == 'Sorpresa': image = cv2.imread('Emojis/sorpresa.jpeg')
    if emotion == 'Tristeza': image = cv2.imread('Emojis/tristeza.jpeg')
    return image

# Diccionario de emociones para los nombres de las carpetas
labels_emotions = {
    0: 'Enojo', 1: 'Felicidad', 2: 'Sorpresa', 3: 'Tristeza'
}

# Cargar imagen de palomita
check_image = cv2.imread('./check_image/checkmark.png')
if check_image is None:
    print("Error: No se pudo cargar la imagen de la palomita (checkmark)")

# Lista de imágenes de signos
sign_images = [None,None,'saludos.jpg','me.jpg','llamo.jpg','a.jpeg', 'b.jpeg', 'c.jpeg', 'd.jpeg', 'e.jpeg', 'f.jpeg', 'g.jpeg', 'h.jpeg', 
               'i.jpeg', 'j.jpeg', 'k.jpeg', 'l.jpeg', 'm.jpeg', 'n.jpeg', 'ene.jpeg', 'o.jpeg', 
               'p.jpeg', 'q.jpeg', 'r.jpeg', 's.jpeg', 't.jpeg', 'u.jpeg', 'v.jpeg', 'w.jpeg', 
               'y.jpeg','Yo.jpg']

# Filtrar solo gestos con imágenes disponibles
valid_indices = [i for i in range(len(sign_images)) if sign_images[i] is not None]

# Filtrar solo gestos con imágenes disponibles
valid_sign_images = [img for img in sign_images if img is not None]

# Función para elegir una nueva imagen aleatoria solo de los gestos válidos
def new_random_image():
    random_image = './sign_images/' + random.choice(valid_sign_images)
    random_image_name = random_image.split('/')[-1]  # Extraer solo el nombre del archivo
    target_label = labels_dict[sign_images.index(random_image_name)]  # Obtener etiqueta esperada
    return random_image, target_label

# Inicializar la primera imagen aleatoria
random_image, target_label = new_random_image()

# Detectar rostros
face_classif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret:
        print("No se pudo capturar el frame")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_frame = frame.copy()

    # Mostrar imagen random en la parte superior
    target_img = cv2.imread(random_image)
    
    if target_img is None:
        print(f"Error: No se pudo cargar la imagen {random_image}")
        break
    else:
        target_img = cv2.resize(target_img, (150, 150))  
        frame[0:150, 0:150] = target_img

    # Detección de manos
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

        if len(data_aux) < 100:
            data_aux.extend([0] * (100 - len(data_aux)))

        data_aux = np.asarray(data_aux[:100])

        # Predicción de gestos
        prediction = model.predict([data_aux])
        predicted_character = labels_dict[int(prediction[0])]
        proba = model.predict_proba([data_aux])
        confidence = np.max(proba) * 100
        # Imprimir predicción y confianza
        print(f"Predicción: {predicted_character} con confianza: {confidence:.2f}%")
        print(f"Objetivo: {target_label}")

        # Revisión de la comparación entre la predicción y el objetivo
        if predicted_character.strip().lower() == target_label.strip().lower():
            print(f"Coincidencia encontrada: {predicted_character} == {target_label}")
        else:
            print(f"No coincide: {predicted_character} != {target_label}")

        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
        x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Verificar si el gesto coincide con la imagen objetivo
        if predicted_character == target_label and check_image is not None:
            check_image_resized = cv2.resize(check_image, (150, 150))
            frame[0:150, 150:300] = check_image_resized
            random_image, target_label = new_random_image()

    # Detección de emociones
    faces = face_classif.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        rostro = aux_frame[y:y + h, x:x + w]
        rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = emotion_recognizer.predict(rostro)

        emotion_label = labels_emotions[result[0]]
        if result[1] < 80:  # Umbral de confianza
            image = emotionImage(emotion_label)
            image = cv2.resize(image, (frame.shape[1], frame.shape[0]))
            nFrame = cv2.hconcat([frame, image])
            cv2.imshow('Detección de gestos y emociones', nFrame)
        else:
            cv2.putText(frame, 'No identificado', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Detección de gestos y emociones', frame)

    # Comando para cerrar la ventana
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('v'):
        engine.say(predicted_character)
        engine.runAndWait()
    elif key == ord('s'):
        random_image, target_label = new_random_image() # Cambiar imagen al presionar la tecla 's'

