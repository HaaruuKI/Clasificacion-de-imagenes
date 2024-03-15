import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import pandas as pd
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
ruta_principal = '/home/akizuki/Documents/IoT/Verduras_new' # Reemplazar con tu ruta
carpetas = ['Cebolla','Chile', 'Tomate']

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
label_dict = {'Cebolla': 0, 'Chile': 1, 'Tomate': 2}
etiquetas_entrenamiento = np.array([label_dict[label] for label in etiquetas_entrenamiento])

# Crear el modelo
model = Sequential([
    Flatten(input_shape=(200, 200, 3)),  # Ajustar tamaño según las imágenes
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # Número de clases = número de carpetas
])

model.compile(optimizer=Adam(),loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Función para entrenar el modelo
def entrenar_modelo():
    # Entrenar el modelo
    train = model.fit(datos_entrenamiento, etiquetas_entrenamiento, epochs=100, batch_size=32)  # Ajustar epochs y batch_size según sea necesario


# Función para realizar la predicción y guardar los resultados en un archivo Excel
def hacer_prediccion():

    # Abrir ventana de selección de carpeta
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()

    # Obtener la lista de imágenes en la carpeta seleccionada
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]

    # Inicializar el diccionario para almacenar los resultados de la predicción
    prediction_results = {}

    # Realizar la predicción para cada imagen en la carpeta
    for image_file in image_files:
        # Obtener la ruta completa de la imagen
        image_path = os.path.join(folder_path, image_file)

        # Cargar imagen y redimensionar
        img = cv2.imread(image_path)
        img_resize = cv2.resize(img, (img_size, img_size))

        # Preprocesar imagen
        img_preprocessed = img_resize / 255.0

        # Realizar predicción
        prediction = model.predict(np.array([img_preprocessed]))

        # Obtener el índice de la clase con mayor probabilidad
        predicted_class_index = np.argmax(prediction)

        # Obtener el nombre de la clase correspondiente al índice
        predicted_class = list(label_dict.keys())[list(label_dict.values()).index(predicted_class_index)]

        # Obtener el porcentaje de precisión de la predicción
        prediction_accuracy = round(np.max(prediction) * 100, 2)

        # Agregar el resultado de la predicción al diccionario
        prediction_results[image_file] = (predicted_class, prediction_accuracy)

    # Crear un DataFrame con los resultados de la predicción
    df = pd.DataFrame.from_dict(prediction_results, orient='index', columns=['Clase', 'Precisión'])

    # Obtener la ruta de guardado del archivo Excel
    save_path = filedialog.asksaveasfilename(defaultextension='.xlsx')

    # Guardar el DataFrame como archivo Excel
    df.to_excel(save_path, index_label='Imagen')

    # Mostrar mensaje de éxito
    # messagebox.showinfo('Predicción guardada', 'Los resultados de la predicción se han guardado exitosamente en un archivo Excel.')

    # Crear ventana de resultados de la predicción
    result_window = tk.Tk()
    result_window.title("Resultados de la predicción")
    result_window.geometry("500x500")

    # Crear un frame con un scrollbar
    frame = tk.Frame(result_window)
    frame.pack(fill="both", expand=True)

    # Crear un canvas dentro del frame
    canvas = tk.Canvas(frame)
    canvas.pack(side="left", fill="both", expand=True)

    # Crear un scrollbar para el canvas
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")

    # Configurar el scrollbar para que funcione con el canvas
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    # Crear un frame dentro del canvas para mostrar los resultados
    inner_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=inner_frame, anchor="nw")

    # Mostrar los resultados de la predicción en el frame
    for image_file, (predicted_class, prediction_accuracy) in prediction_results.items():
        result_label = tk.Label(inner_frame, text=f"Imagen: {image_file} - Clase: {predicted_class} - Precisión: {prediction_accuracy}%")
        result_label.pack()

    # Configurar el canvas para que se pueda hacer scrolling
    canvas.configure(scrollregion=canvas.bbox("all"))

    # Mostrar la ventana de resultados
    result_window.mainloop()

# Crear ventana con botones
window = tk.Tk()
window.title("IA de clasificación de imágenes")
window.geometry("300x200")
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
window.geometry(f"{screen_width}x{screen_height}")

btn_entrenar = tk.Button(window, text="Entrenar", command=entrenar_modelo)
btn_entrenar.pack()

btn_prediccion = tk.Button(window, text="Hacer predicción", command=hacer_prediccion)
btn_prediccion.pack()

window.mainloop()
