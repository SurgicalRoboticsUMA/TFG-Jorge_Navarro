import os
import random
from scipy import ndarray
import numpy as np

# Librerias para el procesamiento de imagenes
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io

#Función para rotar la imagen de entrada un 25% a la izquierda o derecha
def random_rotation(image_array: ndarray):
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

#Funcion para añadir ruido a la imagen
def random_noise(image_array: ndarray):
    return sk.util.random_noise(image_array)

#Funcion para realizar un giro en horizontal de la imagen
def horizontal_flip(image_array: ndarray):
    return image_array[:, ::-1] #Modificadno el lugar de los pixels en la imagen se obtiene dicho giro

# Opciones de tranasformaciones posibles a una imagen
available_transformations = {
    'Rotacion': random_rotation,
    'Ruido': random_noise,
    'Giro_Horizaontal': horizontal_flip
}

folder_path = 'Ruta de la carpeta '
Archivos_deseados = 400

# Lectura de todos los archivos de la carptea indicada
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

Archvos_generados = 0
while Archvos_generados <= Archivos_deseados:
    #Seleccion aleatoria de una imagen del dataset
    image_path = random.choice(images)
    # Lectura de la imagen de entrada
    image_to_transform = sk.io.imread(image_path)
    # Seleccion aleatoria del numero de trasnformaciones
    num_transformations_to_apply = random.randint(1, len(available_transformations))

    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # Transformación aleatoria a una imagen
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](image_to_transform)
        num_transformations += 1

    new_file_path = '%s/augmented_image_%s.jpg' % (folder_path, Archvos_generados)

    # Guardado de las nuevas imagenes generadas
    io.imsave(new_file_path, transformed_image)
    Archvos_generados += 1
