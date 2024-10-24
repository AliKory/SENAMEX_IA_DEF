import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import os

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
    0: ' ', 1: ' ', 2: 'Saludos', 3: 'Me', 4: 'Llamo', 5: 'a', 6: 'b', 7: 'c', 8: 'd',
    9: 'e', 10: 'f', 11: 'g', 12: 'h', 13: 'i', 14: 'j', 15: 'k', 16: 'l', 17: 'm', 18: 'n',
    19: 'ñ', 20: 'o', 21: 'p', 22: 'q', 23: 'r', 24: 's', 25: 't', 26: 'u', 27: 'v', 28: 'w',
    29: 'z', 30: 'Yo', 31: 'Tu', 32: 'Nosotros', 33: 'Ustedes', 34: 'Ella', 35: 'Hola'
}

# Cargar el modelo entrenado para reconocimiento de emociones
method = 'LBPH'  # Cambia esto según el modelo que quieras usar: EigenFaces, FisherFaces o LBPH
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
        print(f"Predicción: {predicted_character} con confianza: {confidence:.2f}%")

        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
        x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Detección de emociones
    faces = face_classif.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        rostro = aux_frame[y:y + h, x:x + w]
        rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)  # Convertir el rostro a escala de grises
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = emotion_recognizer.predict(rostro)

        emotion_label = labels_emotions[result[0]]
        if result[1] < 80:  # Umbral de confianza, ajusta si es necesario
            image = emotionImage(emotion_label)

            # Redimensiona el emoji para que coincida con el tamaño del frame
            image = cv2.resize(image, (frame.shape[1], frame.shape[0]))
            nFrame = cv2.hconcat([frame, image])

            # Mostrar el frame combinado
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

cap.release()
cv2.destroyAllWindows()