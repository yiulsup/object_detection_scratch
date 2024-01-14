import tensorflow as tf 
import numpy as np 
import cv2
from tensorflow.keras import Input, layers
import os
from keras.utils import to_categorical

input_shape = (240, 320, 1)
# Data loading

image_size_gray = (240, 320)
num_images = 80

x_train = np.zeros((num_images, *image_size_gray), dtype=np.uint8)
y_train = np.zeros(80, dtype=np.uint8)

y_train = to_categorical(y_train, 1)

for i in range(40):
    y_train[i] = 1

face_filelist = os.listdir("./dataset/face/")
i = 0
for face_file in face_filelist:
    x_train[i] = cv2.imread("./dataset/face/{}".format(face_file), cv2.IMREAD_GRAYSCALE)
    i = i + 1
face_filelist = os.listdir("./dataset/wo_face/")
for wo_face_file in face_filelist:
    x_train[i] = cv2.imread("./dataset/wo_face/{}".format(wo_face_file), cv2.IMREAD_GRAYSCALE)
    i = i + 1

for i in range(80):
    x_train[i] = x_train[i] / 255.

x_train = np.expand_dims(x_train, -1)
y_train = np.expand_dims(y_train, -1)

model = tf.keras.models.Sequential([
    Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=10)
model.save("face.h5")
