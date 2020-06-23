
# Código de programación para la conversión de videos a imagenes
import cv2
import math

count = 0
videoFile = "Nombre_Archivo"
cap = cv2.VideoCapture(videoFile)  # Método para la lectura del video
frameRate = cap.get(5)  # Fotográmas por segundo

while cap.isOpened():
    frameId = cap.get(1)  # Frame actual
    state, frame = cap.read()
    if not state:
        break
    if frameId % math.floor(frameRate) == 0:
        filename = "frame%d.jpg" % count
        count += 1
        cv2.imwrite(filename, frame)
cap.release()



# Código de programación para convertir las dimensiones de una imagen
from PIL import Image

img = Image.open('Nombre_Imagen')
Ancho_Imagen= 'Valor ancho de imagen'
Alto_Imagen= 'Valor alto de imagen'
img = img.resize((Ancho_Imagen, Alto_Imagen), Image.ANTIALIAS)
img.save('Nombre_Imagen')
