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

import numpy as np
#PLOTTING
import matplotlib.pyplot as plt


#DATA
from sklearn.model_selection import train_test_split
import data_handler as dh

data = dh.gather_data("./genres")
X, Y = dh.format_data(data)
train_data, test_data, train_target, test_target = train_test_split(X,Y, test_size= 0.1, random_state=31 )
print(X.shape, Y.shape)



#PARAMS
number_cells = 100
look_back = 60
batch_size = 100
epochs = 200





#MODEL LSTM

Ludwig_van_Beethoven = Sequential ()
Ludwig_van_Beethoven.add(LSTM(number_cells,return_sequences=True, input_shape =(look_back, 110) ))
Ludwig_van_Beethoven.add(Dropout(0.3))
Ludwig_van_Beethoven.add(LSTM(number_cells,return_sequences=True))
Ludwig_van_Beethoven.add(Dropout(0.2))
Ludwig_van_Beethoven.add(LSTM(number_cells))
Ludwig_van_Beethoven.add(Dropout(0.3))
Ludwig_van_Beethoven.add (Dense (10,activation = 'softmax'))

opt = keras.optimizers.Adam(learning_rate=0.0001)
Ludwig_van_Beethoven.compile (loss ="categorical_crossentropy", metrics = ["categorical_crossentropy"] , optimizer = opt)   




history = Ludwig_van_Beethoven.fit(train_data, train_target, batch_size= 200, epochs= epochs, validation_data=(test_data,test_target))
Ludwig_van_Beethoven.save("Beethoven.h5")

Ludwig_van_Beethoven = keras.models.load_model('Beethoven.h5')
'''
num = 100
test = test_data
k = test_target
a = Ludwig_van_Beethoven.predict(test)
cont = 0
for i in range(num):
    b = np.where(k[i] == k[i].max())
    c = np.where(a[i] == a[i].max())
    if b == c:
        cont += 1
print(cont)
'''
'''
plt.title("Close for AIG index", fontsize=14)
plt.plot(history.history["categorical_crossentropy"],'r-')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()

plt.title("close for AIG index", fontsize=14)
plt.plot(history.history["val_categorical_crossentropy"],'r-')
plt.xlabel("Epoch")
plt.ylabel("VAL_MSE")
plt.show()
'''
Ludwig_van_Beethoven_train_error = Ludwig_van_Beethoven.evaluate(train_data, train_target, batch_size=batch_size)
Ludwig_van_Beethoven_test_error = Ludwig_van_Beethoven.evaluate(test_data, test_target, batch_size=batch_size)

print(Ludwig_van_Beethoven_train_error, Ludwig_van_Beethoven_test_error)

