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
train_data, valid_data, train_target, valid_target = train_test_split(X,Y, test_size= 0.1, random_state=31 )
print(X.shape, Y.shape)


print(train_data.shape)
#PARAMS
number_cells = 100
look_back = 60
batch_size = 100
epochs = 175





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

Ludwig_van_Beethoven.summary()


#history = Ludwig_van_Beethoven.fit(train_data, train_target, batch_size= 100, epochs= epochs, validation_data=(valid_data,valid_target))
#Ludwig_van_Beethoven.save("Beethoven.h5")

Ludwig_van_Beethoven = keras.models.load_model('Beethoven.h5')

test = dh.gather_data_test("./test")
test_data, test_target = dh.format_data_test(test)

print(test_data.shape, test_target.shape)


num = 100
test = test_data
k = test_target
a = Ludwig_van_Beethoven.predict(test)
cont = 0
for i in range(test_data.shape[0]):
    b = np.where(k[i] == k[i].max())
    c = np.where(a[i] == a[i].max())
    if b == c:
        cont += 1
print("The RNN accuracy is:",cont/test_data.shape[0])

'''
plt.title("Categorical_crossentropy loss", fontsize=14)
plt.plot(history.history["categorical_crossentropy"],'r-')
plt.xlabel("Epoch")
plt.ylabel("CCE")
plt.show()

plt.title("Categorical_crossentropy loss", fontsize=14)
plt.plot(history.history["val_categorical_crossentropy"],'r-')
plt.xlabel("Epoch")
plt.ylabel("VAL_CCE")
plt.show()
'''
Ludwig_van_Beethoven_train_error = Ludwig_van_Beethoven.evaluate(train_data, train_target, batch_size=batch_size)
Ludwig_van_Beethoven_valid_error = Ludwig_van_Beethoven.evaluate(valid_data, valid_target, batch_size=batch_size)
Ludwig_van_Beethoven_test_error = Ludwig_van_Beethoven.evaluate(test_data, test_target, batch_size=batch_size)

print(Ludwig_van_Beethoven_train_error[0])
print(Ludwig_van_Beethoven_valid_error[0])
print(Ludwig_van_Beethoven_test_error[0])


