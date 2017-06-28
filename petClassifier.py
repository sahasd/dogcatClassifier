from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.pyplot as plt
from keras.utils import np_utils
import numpy as np

data_config = ImageDataGenerator(
	rescale=1./255, 
	horizontal_flip=True,
	vertical_flip=True,
	rotation_range=40)

train_data = data_config.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=5000,
        class_mode='binary')
(X_train, y_train) = train_data.next()
print(len(X_train))
y_train = np_utils.to_categorical(y_train)
print(y_train[0])
print(X_train.shape, y_train.shape)


def cnn_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(64, activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation="sigmoid"))
	model.add(Dense(2, activation="sigmoid"))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

model = cnn_model()
model.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=100) 
scores = model.evaluate(X_train, y_train, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
model.save_weights('petClassifier.h5')
