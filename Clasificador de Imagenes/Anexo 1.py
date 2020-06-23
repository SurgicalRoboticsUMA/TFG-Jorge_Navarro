import os
import re
import pandas as pd
import keras
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import cv2
import math
matplotlib.use('TkAgg')
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


dirname = os.path.join(os.getcwd(), 'Images')
imgpath = dirname + os.sep


images = []
directories = []
dircount = []
prevRoot = ''
cant = 0

print("leyendo imagenes de ", imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant = cant + 1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            images.append(image)
            b = "Leyendo..." + str(cant)
            print(b, end="\r")
            if prevRoot != root:
                print(root, cant)
                prevRoot = root
                directories.append(root)
                dircount.append(cant)
                cant = 0
dircount.append(cant)

dircount = dircount[1:]
dircount[0] = dircount[0] + 1
print('Directorios leidos:', len(directories))
print("Imagenes en cada directorio", dircount)
print('Suma total de imagenes en subdirs:', sum(dircount))

labels = []
indice = 0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice = indice + 1
print("Cantidad etiquetas creadas: ", len(labels))

tareas = []
indice = 0
for directorio in directories:
    name = directorio.split(os.sep)
    print(indice, name[len(name) - 1])
    tareas.append(name[len(name) - 1])
    indice = indice + 1

y = np.array(labels)
X = np.array(images, dtype=np.uint8)  # convierto de lista a numpy

# Find the unique numbers from the train labels
classes = np.unique(y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# Mezclar todo y crear los grupos de entrenamiento y testing
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2)
print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

#Configuracion de los datos de entarada a la red convolucional
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

#Configuracion de los datos de entarada al Perceptron Multicapa
train_X = train_X.reshape(-1, 'Tamaño de los datos de entrada').astype('float32') #Indicar el tamaño de los datos de entrada
test_X = test_X.reshape(-1, 'Tamaño de los datos de entrada').astype('float32') #Indicar el tamaño de los datos de entrada
train_X = train_X / 255.
test_X = test_X / 255.

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

print(train_X.shape, valid_X.shape, train_label.shape, valid_label.shape)

# RED CONVOLUCIONAL
epochs = 15
batch_size = 128
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(35, 35, 3)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nClasses, activation='softmax'))


# RED PERCEPTRON MULTICAPA
epochs = 15
batch_size = 128
model = Sequential([
    Dense(output_dim=500, input_dim=2352, activation='relu'),
    Dropout(0.25),
    Dense(output_dim=500, input_dim=500, activation='relu'),
    Dropout(0.25),
    Dense(output_dim=500, input_dim=500, activation='relu'),
    Dropout(0.25),
    Dense(output_dim=500, input_dim=500, activation='relu'),
    Dropout(0.25),
    Dense(output_dim=500, input_dim=500, activation='relu'),
    Dropout(0.25),
    Dense(output_dim=3, input_dim=500, activation='softmax'),
])

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

history = model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_label))

# Representacion del error cometido
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ERRORES')
plt.ylabel('ERROR (Cross-Entropy)')
plt.xlabel('EPOCAS')
plt.legend(['Train', 'Validation'], loc='lower right')

# Representacion de la precison obtenida
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('PRECISION    ')
plt.ylabel('PRECISION')
plt.xlabel('EPOCAS')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#Evaluacion del modelo
evaluation = model.evaluate(test_X, test_Y_one_hot, verbose=1)

print('Test loss:', evaluation[0])
print('Test accuracy:', evaluation[1])

predicted_classes2 = model.predict(test_X)
predicted_classes = []
for predicted_tareas in predicted_classes2:
    predicted_classes.append(predicted_tareas.tolist().index(max(predicted_tareas)))
predicted_classes = np.array(predicted_classes)

print(predicted_classes.shape, test_Y.shape)

#Prediccion del numero de imagenes correctamente etiquetdas
correct = np.where(predicted_classes == test_Y)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[0:25]):
    plt.subplot(5, 5, i + 1)
    plt.imshow(test_X[correct].reshape(35, 35, 3), cmap='gray', interpolation='none')
    plt.title("{}, {}".format(tareas[predicted_classes[correct]], tareas[test_Y[correct]]))
plt.show()

#Prediccion del numero de imagenes incorrectamente etiquetdas
incorrect = np.where(predicted_classes != test_Y)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[0:25]):
    plt.subplot(5, 5, i + 1)
    plt.imshow(test_X[incorrect].reshape(35, 35, 3), cmap='gray', interpolation='none')
    plt.title("{}, {}".format(tareas[predicted_classes[incorrect]], tareas[test_Y[incorrect]]))
plt.show()

#Representacion de la matriz de confusion
predicted_classes2 = np.argmax(predicted_classes2, axis=1)
cm_mlp = confusion_matrix(test_Y, predicted_classes2)
plt.figure(figsize=(7, 7))
df_cm = pd.DataFrame(cm_mlp, index=[i for i in ["Nombre del todas las clases exixtentes"]], columns=[i for i in ["Nombre del todas las clases exixtentes"]])
sn.heatmap(df_cm, annot=True)
plt.title('Modelo de red neuronal')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

