import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

#KERAS
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
from keras.layers import recurrent, Conv1D, MaxPooling1D
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

data = dh.gather_data("./genres")
X, Y = dh.format_data(data)
train_data, test_data, train_target, test_target = train_test_split(X,Y, test_size= 0.1, random_state=31 )
print(X.shape, Y.shape)



Johann_Sebastian_Bach = Sequential()

Johann_Sebastian_Bach.add(Conv1D(100, 5, activation = 'relu', input_shape = X.shape[1:] ))
Johann_Sebastian_Bach.add(MaxPooling1D( pool_size= 2))
Johann_Sebastian_Bach.add(Dropout(0.2))

Johann_Sebastian_Bach.add(Conv1D(300,3, activation = 'relu' ))
Johann_Sebastian_Bach.add(MaxPooling1D( pool_size= 2))
Johann_Sebastian_Bach.add(Dropout(0.2))
Johann_Sebastian_Bach.add(Conv1D(180,2, activation = 'relu' ))
Johann_Sebastian_Bach.add(MaxPooling1D( pool_size= 2))
Johann_Sebastian_Bach.add(Dropout(0.2))

Johann_Sebastian_Bach.add(Flatten())
Johann_Sebastian_Bach.add(Dropout(0.3))

Johann_Sebastian_Bach.add(Dense(10, activation='softmax'))
opt = keras.optimizers.Adam(learning_rate=0.001)
Johann_Sebastian_Bach.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

history = Johann_Sebastian_Bach.fit(train_data, train_target, batch_size= 200, epochs= 200, validation_data=(test_data,test_target))

#Johann_Sebastian_Bach.save("Bach.h5")

plt.title("Close for AIG index", fontsize=14)
plt.plot(history.history["accuracy"],'r-')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()

plt.title("close for AIG index", fontsize=14)
plt.plot(history.history["val_accuracy"],'r-')
plt.xlabel("Epoch")
plt.ylabel("VAL_MSE")
plt.show()

Johann_Sebastian_Bach_train_error = Johann_Sebastian_Bach.evaluate(train_data, train_target, batch_size=batch_size)
Johann_Sebastian_Bach_test_error = Johann_Sebastian_Bach.evaluate(test_data, test_target, batch_size=batch_size)