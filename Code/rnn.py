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

Johann_Sebastian_Bach.save("Bach.h5")
'''
Johann_Sebastian_Bach.add(Conv2D(30, kernel_size=(5,5), activation = 'relu', input_shape = train_data.shape[1:] ))
Johann_Sebastian_Bach.add(MaxPooling2D( pool_size= (2,2)))
Johann_Sebastian_Bach.add(Conv2D(90,kernel_size=(3,3), activation = 'relu' ))
Johann_Sebastian_Bach.add(MaxPooling2D( pool_size= (2,2)))
Johann_Sebastian_Bach.add(Conv2D(540,kernel_size=(3,3), activation = 'relu' ))
Johann_Sebastian_Bach.add(Flatten())
Johann_Sebastian_Bach.add(Dense(100, activation='relu'))
Johann_Sebastian_Bach.add(Dense(25, activation='relu'))
Johann_Sebastian_Bach.add(Dense(10, activation='softmax'))

'''

'''
#PARAMS
number_cells = 100
look_back = 30
batch_size = 100
epochs = 1000





#MODEL LSTM

Ludwig_van_Beethoven = Sequential ()
Ludwig_van_Beethoven.add(LSTM(number_cells, input_shape =(look_back, 2200) ))
Ludwig_van_Beethoven.add(Dropout(0.2))
Ludwig_van_Beethoven.add(Dense(50, input_shape=(look_back, 2200)  ))
Ludwig_van_Beethoven.add(Dropout(0.2))
Ludwig_van_Beethoven.add (Dense (activation = 'linear',units=1))
opt = keras.optimizers.Adam(learning_rate=0.000001)
Ludwig_van_Beethoven.compile (loss =keras.losses.MeanAbsoluteError(), metrics = ["mean_absolute_error"] , optimizer = opt)   
#print(Ludwig_van_Beethoven.summary())




history = Ludwig_van_Beethoven.fit(train_data, train_target, validation_data= (test_data, test_target), batch_size = batch_size, epochs = epochs, shuffle = True )
Ludwig_van_Beethoven.save("model3.h5")


#Ludwig_van_Beethoven = keras.models.load_model('model2.h5')

test = np.array(test_data[:10])
#test = np.expand_dims(test,0)
a = Ludwig_van_Beethoven.predict(test)
b = test_target[:10]
print(b)
a = np.round(np.abs(a))
a = np.reshape(a ,(10,))
print(b-a)


plt.title("Close for AIG index", fontsize=14)
plt.plot(history.history["mean_absolute_error"],'r-')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()

plt.title("close for AIG index", fontsize=14)
plt.plot(history.history["val_mean_absolute_error"],'r-')
plt.xlabel("Epoch")
plt.ylabel("VAL_MSE")
plt.show()

#Ludwig_van_Beethoven_train_error = Ludwig_van_Beethoven.evaluate(train_data, train_target, batch_size=batch_size)
#Ludwig_van_Beethoven_test_error = Ludwig_van_Beethoven.evaluate(test_data, test_target, batch_size=batch_size)

#print(Ludwig_van_Beethoven_train_error, Ludwig_van_Beethoven_test_error)
'''
