import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar el diccionario de datos
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extraer los datos y etiquetas
data = data_dict['data']
labels = data_dict['labels']

# Asegúrate de que data está correctamente cargado antes de proceder
if data is None or len(data) == 0:
    raise ValueError("Los datos no se cargaron correctamente o están vacíos.")

# Establecer la longitud máxima de los datos
max_length = 100  # Ajusta este valor según tus necesidades

# Normalizar los datos con padding o truncamiento
padded_data = [
    np.pad(d, (0, max_length - len(d)), 'constant') if len(d) < max_length else d[:max_length]
    for d in data
]

# Convertir los datos normalizados a un array de Numpy
data = np.asarray(padded_data)
labels = np.asarray(labels)

# División de los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Entrenamiento del modelo
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predicción y evaluación
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Guardar el modelo entrenado
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

# --- Parte de Inferencia (predicción con nuevos datos) ---

# Ejemplo de datos de entrada para predicción
data_aux = [0, 1, 2]  # Este es un ejemplo, reemplázalo con tus datos reales

# Realizar padding o truncamiento de los datos de entrada para que coincidan con la longitud esperada
if len(data_aux) < max_length:
    # Padding con ceros si hay menos características de las esperadas
    data_aux.extend([0] * (max_length - len(data_aux)))
else:
    # Truncar si hay más características de las esperadas
    data_aux = data_aux[:max_length]

# Convertir a array de Numpy y realizar la predicción
data_aux = np.asarray(data_aux)
prediction = model.predict([data_aux])

# Mostrar el resultado de la predicción
print("Predicción:", prediction)
