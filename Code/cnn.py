import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

#KERAS
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
from keras.layers import recurrent, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, BatchNormalization
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import to_categorical
from tensorflow import keras

#PLOTTING
import matplotlib.pyplot as plt


#DATA
from sklearn.model_selection import train_test_split
import data_handler as dh


X = dh.gather_data_img("./imgs")
Y = dh.get_classes_img()

train_data, test_data, train_target, test_target = train_test_split(X,Y, test_size= 0.1, random_state=31 )
print(X.shape, Y.shape)

Wolfgang_Amadeus_Mozart = Sequential()
Wolfgang_Amadeus_Mozart.add(Conv2D(8,kernel_size=(3,3),strides=(1,1), activation = 'relu', input_shape = train_data.shape[1:] ))
Wolfgang_Amadeus_Mozart.add(BatchNormalization(axis=3))
Wolfgang_Amadeus_Mozart.add(MaxPooling2D( pool_size= (2,2)))

Wolfgang_Amadeus_Mozart.add(Conv2D(16,kernel_size=(3,3),strides=(1,1), activation = 'relu', input_shape = train_data.shape[1:] ))
Wolfgang_Amadeus_Mozart.add(BatchNormalization(axis=3))
Wolfgang_Amadeus_Mozart.add(MaxPooling2D( pool_size= (2,2)))
'''
Wolfgang_Amadeus_Mozart.add(Conv2D(32,kernel_size=(3,3),strides=(1,1), activation = 'relu', input_shape = train_data.shape[1:] ))
Wolfgang_Amadeus_Mozart.add(BatchNormalization(axis=-1))
Wolfgang_Amadeus_Mozart.add(MaxPooling2D( pool_size= (2,2)))

Wolfgang_Amadeus_Mozart.add(Conv2D(64,kernel_size=(3,3),strides=(1,1), activation = 'relu', input_shape = train_data.shape[1:] ))
Wolfgang_Amadeus_Mozart.add(BatchNormalization(axis=-1))
Wolfgang_Amadeus_Mozart.add(MaxPooling2D( pool_size= (2,2)))
'''

Wolfgang_Amadeus_Mozart.add(Flatten())
Wolfgang_Amadeus_Mozart.add(Dropout(0.2))
Wolfgang_Amadeus_Mozart.add(Dense(10, activation= "softmax"))

opt = keras.optimizers.Adam(learning_rate=0.001)
Wolfgang_Amadeus_Mozart.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])


history = Wolfgang_Amadeus_Mozart.fit(train_data, train_target, batch_size= 100, epochs= 5, validation_data=(test_data,test_target))