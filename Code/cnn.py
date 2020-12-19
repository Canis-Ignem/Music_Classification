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
train_data, valid_data, train_target, valid_target = train_test_split(X,Y, test_size= 0.1, random_state=31 )
print(X.shape, Y.shape)



Johann_Sebastian_Bach = Sequential()

Johann_Sebastian_Bach.add(Conv1D(100, 5, activation = 'relu', input_shape = X.shape[1:] ))
#Johann_Sebastian_Bach.add(MaxPooling1D( pool_size= 2))
Johann_Sebastian_Bach.add(Dropout(0.2))

Johann_Sebastian_Bach.add(Conv1D(300,3, activation = 'relu' ))
#Johann_Sebastian_Bach.add(MaxPooling1D( pool_size= 2))
Johann_Sebastian_Bach.add(Dropout(0.2))

Johann_Sebastian_Bach.add(Conv1D(180,2, activation = 'relu' ))
#Johann_Sebastian_Bach.add(MaxPooling1D( pool_size= 2))
Johann_Sebastian_Bach.add(Dropout(0.2))

Johann_Sebastian_Bach.add(Flatten())
Johann_Sebastian_Bach.add(Dropout(0.3))

Johann_Sebastian_Bach.add(Dense(10, activation='softmax'))
opt = keras.optimizers.Adam(learning_rate=0.0001)
Johann_Sebastian_Bach.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['categorical_crossentropy','accuracy'])

Johann_Sebastian_Bach.summary()

#history = Johann_Sebastian_Bach.fit(train_data, train_target, batch_size= 200, epochs= 506, validation_data=(valid_data,valid_target))

#Johann_Sebastian_Bach.save("Bach_new.h5")
'''
plt.title("Categorical_crossentropy loss", fontsize=14)
plt.plot(history.history["categorical_crossentropy"],'r-')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()

plt.title("Categorical_crossentropy loss", fontsize=14)
plt.plot(history.history["val_categorical_crossentropy"],'r-')
plt.xlabel("Epoch")
plt.ylabel("VAL_CCE")
plt.show()
'''
test = dh.gather_data_test("./test")
test_data, test_target = dh.format_data_test(test)

print(test_data.shape, test_target.shape)

Johann_Sebastian_Bach = keras.models.load_model("Bach_no_pooling_out_files.h5")

Johann_Sebastian_Bach_train_error = Johann_Sebastian_Bach.evaluate(train_data, train_target, batch_size=200)
Johann_Sebastian_Bach_valid_error = Johann_Sebastian_Bach.evaluate(valid_data, valid_target, batch_size=200)
Johann_Sebastian_Bach_test_error = Johann_Sebastian_Bach.evaluate(test_data, test_target, batch_size=200)

print(Johann_Sebastian_Bach_train_error,Johann_Sebastian_Bach_valid_error, Johann_Sebastian_Bach_test_error)


##Predict a file

one_song = test_data[:100,:,:] #blues song
print(one_song.shape)
res = Johann_Sebastian_Bach.predict(one_song)
cont = np.zeros((10))
for i in range(res.shape[0]):
    idx = np.where(res[i] == res[i].max())
    cont[idx] += 1
print(cont)


#Predict all files and showcase the matrix

contL =cont = np.zeros((50,10))
idx = 0
i = 0

while i < test_data.shape[0]:
    #print(i)
    song = test_data[i:(i+100),:,:]
    res = Johann_Sebastian_Bach.predict(song)
    for j in range(res.shape[0]):
        max_idx = np.where(res[j] == res[j].max())
        #print(max_idx[0][0])
        contL[idx][max_idx] += 1
    i += 100
    idx += 1
print(contL)

#Showcase accuracy

acc = 0
#print(test_target[1])
new_test = np.zeros((50))
j = 0
idx = 0
while j < test_target.shape[0]:
    corr_idx = np.where(test_target[j] == test_target[j].max())
    new_test[idx] = corr_idx[0][0]
    j+= 100
    idx += 1
print(new_test.shape)

wrong = []

for i in range(contL.shape[0]):
    max_idx = np.where(contL[i] == contL[i].max())
    if max_idx[0][0] == new_test[i]:
        acc += 1
    else:
        wrong.append(i)
print(acc/contL.shape[0])
print("this are the one we got wrong", wrong)