# Referred
# https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/
# https://pysource.com/2019/02/15/detecting-colors-hsv-color-space-opencv-with-python/

import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Normally  x--> features, y--> target (train / test)
# Capital big letters have feature set and target both
X = pickle.load(open("Xleaf.pickle", "rb"))
Y = pickle.load(open("yleaf.pickle", "rb"))

# Normalize or scale the data
X = X / 255.0

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(60, input_dim=32))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Saving the model
model_json = model.to_json()
with open("model.json", "w") as json_file :
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

model.save('CNN.model')