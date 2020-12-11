import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

#KERAS
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
from keras.layers import recurrent
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
train_data, test_data, train_target, test_target = train_test_split(X,Y, test_size= 0.3, random_state=30 )
print(X.shape, Y.shape)



#PARAMS
number_cells = 256
look_back = 30
batch_size = 100
epochs = 5

#MODEL LSTM

Ludwig_van_Beethoven = Sequential ()
Ludwig_van_Beethoven.add(LSTM(number_cells, input_shape =(look_back, 22000) ))
Ludwig_van_Beethoven.add(Dropout(0.2))
Ludwig_van_Beethoven.add(Dense(256, input_shape=(look_back, 22000)  ))
Ludwig_van_Beethoven.add(Dropout(0.2))
Ludwig_van_Beethoven.add (Dense (activation = 'softmax',units=10))
opt = keras.optimizers.Adam(learning_rate=0.0001)
Ludwig_van_Beethoven.compile (loss =tf.keras.losses.BinaryCrossentropy(), metrics = ["Accuracy"] , optimizer = opt)   
#print(Ludwig_van_Beethoven.summary())



'''
print(train_data.shape)
print(test_data.shape)
print(train_target.shape)
print(test_target.shape)
'''
history = Ludwig_van_Beethoven.fit(train_data, train_target, validation_data= (test_data, test_target), batch_size = batch_size, epochs = epochs, shuffle = True )
Ludwig_van_Beethoven.save("model.h5")


#Ludwig_van_Beethoven = keras.models.load_model('model.h5')

test = np.array(test_data[0])
test = np.expand_dims(test,0)
a = Ludwig_van_Beethoven.predict(test)

print(test_target[0])
print(a)

'''
Ludwig_van_Beethoven_train_error = Ludwig_van_Beethoven.evaluate(train_data, train_target, batch_size=batch_size)
Ludwig_van_Beethoven_test_error = Ludwig_van_Beethoven.evaluate(test_data, test_target, batch_size=batch_size)

print(Ludwig_van_Beethoven_train_error, Ludwig_van_Beethoven_test_error)
'''
