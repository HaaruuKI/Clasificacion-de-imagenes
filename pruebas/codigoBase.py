import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Función para cargar imágenes y etiquetas desde una carpeta
def cargar_datos(ruta):
    datos = []
    etiquetas = []
    img_size = 200  # Tamaño de la imagen (ajustar según sea necesario)

    for etiqueta in os.listdir(ruta):
        etiqueta_path = os.path.join(ruta, etiqueta)

        for img_name in os.listdir(etiqueta_path):
            if img_name.endswith(".jpg"):  # Verificar la extensión del archivo
                img_path = os.path.join(etiqueta_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises si es necesario
                img = cv2.resize(img, (img_size, img_size))
                datos.append(img)
                etiquetas.append(etiqueta)  # Agregar el nombre de la carpeta como etiqueta

    return np.array(datos), np.array(etiquetas)

# Directorio de las carpetas de imágenes
ruta_principal = '/home/akizuki/Documents/IoT/img_guardadas' # Reemplazar con tu ruta
carpetas = ['cebolla', 'chile', 'tomate']

for carpeta in carpetas:
  archivos = os.path.join(ruta_principal, carpeta)
  print(f"Contenido de la carpeta {carpeta}:")
  contenido_carpeta = os.listdir(archivos)
  for elemento in contenido_carpeta:
    ruta_elemento = os.path.join(archivos, elemento)
    if os.path.isfile(ruta_elemento):
      size_element = os.path.getsize(ruta_elemento)
     # print(f" - {elemento} - Tamaño: {size_element}")

img_size = 200

data = []  # Lista para almacenar imágenes
labels = []  # Lista para almacenar etiquetas/clasificaciones

# Iterar sobre las carpetas en la ruta principal
for folder in os.listdir(ruta_principal):
    folder_path = os.path.join(ruta_principal, folder)

    # Verificar si es una carpeta
    if os.path.isdir(folder_path):
        # Iterar sobre las imágenes en cada carpeta
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            try:
                img = cv2.imread(img_path)

                if img is not None and img.size != 0:  # Verificar si la imagen no está vacía
                    img = cv2.resize(img, (img_size, img_size))  # Redimensionar la imagen
                    data.append(img)
                    labels.append(folder)  # Agregar la etiqueta/clasificación
                    #print(f"Procesado: {img_path}")
                else:
                    print(f"Saltado: Imagen vacía o inválida - {img_path}")
            except Exception as e:
                print(f"Error al procesar {img_path}: {e}")

# Convertir las listas de datos y etiquetas a arreglos numpy
data = np.array(data)
labels = np.array(labels)

# Verificar las dimensiones y formas de los arreglos
print("Dimensiones del arreglo de datos:", data.shape)
print("Dimensiones del arreglo de etiquetas:", labels.shape)

# Cargar datos de las carpetas
datos_entrenamiento = data
etiquetas_entrenamiento = labels

# Normalizar los datos
datos_entrenamiento = datos_entrenamiento / 255.0

# Convertir las etiquetas de cadena a valores numéricos
label_dict = {'cebolla': 0, 'chile': 1, 'tomate': 2}
etiquetas_entrenamiento = np.array([label_dict[label] for label in etiquetas_entrenamiento])

# Crear el modelo
model = Sequential([
    Flatten(input_shape=(200, 200, 3)),  # Ajustar tamaño según las imágenes
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # Número de clases = número de carpetas
])

model.compile(optimizer=Adam(),loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
train = model.fit(datos_entrenamiento, etiquetas_entrenamiento, epochs=100, batch_size=32)  # Ajustar epochs y batch_size según sea necesario




ruta = '/home/akizuki/Documents/IoT/imgrefe'
pruebas=[]
img_size=200
for img in os.listdir(ruta):
  img = cv2.imread(os.path.join(ruta,img))

  img_resize = cv2.resize(img,(img_size,img_size))
  pruebas.append(img_resize)
 # print(len(pruebas))
 # print(pruebas)

pruebas= np.array(pruebas)
print(pruebas.shape)
#numero de imagenes, tamaño de imagen, tamaño de imagen, numero de colores
#print(circulo[0].shape)


prediccion = model.predict(pruebas)
print('Preddiccion: ',prediccion[1])


# Evaluación del modelo
test_eval = model.evaluate(datos_entrenamiento, etiquetas_entrenamiento, verbose=1)
print('TEST:')
print('Perdida:', test_eval[0])
print('Exactitud:', test_eval[1])


# Visualización de métricas de rendimiento
accuracy = train.history['accuracy']
# val_accuracy = train.history['val_accuracy']
loss = train.history['loss']
# val_loss = train.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Precision de entrenamiento')
# plt.plot(epochs, val_accuracy, 'b', label='Exactitud de la validacion')
plt.title('Precisión de entrenamiento y validación')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Perdida de entrenamiento')
# plt.plot(epochs, val_loss, 'b', label='Perdida de validacion')
plt.title('Perdida de entrenamiento y validacion')
plt.legend()
plt.show()