import numpy as np
import numpy.random as rnd
import itertools 
import os
import sys

# We used some utilities from sklearn
from sklearn.preprocessing import StandardScaler


# Tensorflow library used for implementation of the DNNs
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from keras.datasets import mnist, fashion_mnist
from keras.layers import Input, Dense , merge, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras import models
from keras import layers


# Used for plotting and display of figures
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

from IPython.display import display
from IPython.display import Image
from IPython.display import SVG

import data_handler as dh

data = dh.gather_data("./genres")
sr = 22050


data = dh.format_data(data[:100])
data = np.reshape(data,(300,220000))
print(data.shape)

n_inputs = 220000

n_G_noise = 100
n_D_hidden_outputs = 1024
n_G_hidden_outputs = 1024

def xavier_init(size):  # Normal version
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape = [None, n_inputs])

D_W1 = tf.Variable(xavier_init([n_inputs, n_D_hidden_outputs]))
D_b1 = tf.Variable(xavier_init([n_G_hidden_outputs]))

D_W2 = tf.Variable(xavier_init([n_D_hidden_outputs, 1]))
D_b2 = tf.Variable(xavier_init([1]))

theta_D = [D_W1,D_W2,D_b1, D_b2]

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x,D_W1) + D_b1)
    D_logit = tf.matmul(D_h1,D_W2) + D_b2
    d_prob = tf.nn.sigmoid(D_logit)
    
    return d_prob, D_logit
    
    
Z = tf.placeholder(tf.float32, shape=[None, n_G_noise])


G_W1 = tf.Variable(xavier_init([n_G_noise, n_G_hidden_outputs]))
G_b1 = tf.Variable(tf.zeros(shape=[n_G_hidden_outputs]))

G_W2 = tf.Variable(xavier_init([n_G_hidden_outputs, n_inputs]))
G_b2 = tf.Variable(tf.zeros(shape=[n_inputs]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob

G_sample = generator(Z)


D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, 
                                                                     labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                     labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake


G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, 
                                                                labels=tf.ones_like(D_logit_fake)))


D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

# Batch size
mb_size = 250
Z_dim = 100
number_iterations = 41
batch_size = 150
n_batches = data.shape[0]//batch_size
print_cycle = 8


sess = tf.Session() 
sess.run(tf.global_variables_initializer())

if not os.path.exists('gan_out/'):
    os.makedirs('gan_out/')

for it in range(number_iterations):

        
    # 25 random samples are taken from the generator for visualization purposes
    samples = sess.run(G_sample, feed_dict={Z: sample_Z(25, Z_dim)})

        
    for i in range(n_batches):

        # A batch is picked from MNIST    
        X_mb = data[i*batch_size:(i+1)*batch_size]

        # The training data is used to train the GAN
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

    if it % print_cycle == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()