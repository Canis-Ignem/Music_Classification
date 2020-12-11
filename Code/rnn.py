import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from keras.datasets import imdb, mnist, fashion_mnist
# Importing keras libraries
from keras.callbacks import EarlyStopping
from keras.layers import SimpleRNN, LSTM, Conv1D, MaxPooling1D
from keras.layers import recurrent
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.layers.embeddings import Embedding 
from keras.preprocessing import sequence
from keras.utils import to_categorical
from tensorflow import keras