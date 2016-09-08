import numpy as np
import random
import os
import time
import csv
import math
import sys

from scipy.ndimage import imread
import scipy.io as sio
import scipy

import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d

from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Reshape, Flatten, Permute, Lambda, Layer, RepeatVector, merge
from keras.layers import Convolution2D, Convolution3D, MaxPooling2D, AveragePooling2D
from keras.layers import LSTM, GRU, SimpleRNN, TimeDistributed, Recurrent
from keras.layers import BatchNormalization
from keras.activations import relu
from keras import activations, initializations, regularizers
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam

from keras import backend as K
from keras.engine import Layer, InputSpec

import matplotlib.pyplot as plt

from policy import epsilon_greedy

batch_size = 32
glimpse_num = 8
epochs = 100
img_size = 32
glimpse_size = 8
epsilon = 0.1


print 'Building models'

def build_model(train = False):
    #input_layer_whole = Input(batch_shape = (batch_size, 3, 32, 32))
    #if train == False:
    input_layer_glimpse = Input(batch_shape = (batch_size, None, 3, glimpse_size, glimpse_size))
    input_layer_location = Input(batch_shape = (batch_size, None, 2, ))
    #else:
        #input_layer_glimpse = Input(shape = (None, 3, glimpse_size, glimpse_size))
        #input_layer_location = Input(shape = (None, 2, ))

    conv = TimeDistributed(Convolution2D(16, 5, 5, border_mode = 'same', activation = 'linear'))(input_layer_glimpse)
    conv = LeakyReLU()(conv)
    conv = TimeDistributed(Convolution2D(16, 5, 5, border_mode = 'same', activation = 'linear'))(conv)
    conv = LeakyReLU()(conv)
    conv = TimeDistributed(Convolution2D(16, 5, 5, border_mode = 'same', activation = 'linear'))(conv)
    conv = LeakyReLU()(conv)

    fc = TimeDistributed(Flatten())(conv)
    fc = TimeDistributed(Dense(200))(fc)
    fc = LeakyReLU()(fc)
    fc = TimeDistributed(Dense(200))(fc)
    fc = LeakyReLU()(fc)

    fc_l = TimeDistributed(Dense(200))(input_layer_location)
    fc_l = LeakyReLU()(fc_l)
    fc_l = TimeDistributed(Dense(200))(fc_l)
    fc_l = LeakyReLU()(fc_l)

    fc = merge([fc, fc_l], mode = 'mul')

    if train == False:
        rnn1 = LSTM(200, return_sequences = True, stateful = True, consume_less = 'gpu')(fc)
        rnn2 = LSTM(200, return_sequences = True, stateful = True, consume_less = 'gpu')(rnn1)
    else:
        rnn1 = LSTM(200, return_sequences = True, stateful = False, consume_less = 'gpu')(fc)
        rnn2 = LSTM(200, return_sequences = True, stateful = False, consume_less = 'gpu')(rnn1)

    fc_out_location = TimeDistributed(Dense(200))(rnn2)
    fc_out_location = LeakyReLU()(fc_out_location)
    fc_out_location = TimeDistributed(Dense(32 ** 2, activation = 'softmax'))(fc_out_location)
    fc_out_location = TimeDistributed(Reshape((32, 32)))(fc_out_location)

    fc_out_label = TimeDistributed(Dense(200))(rnn1)
    fc_out_label = LeakyReLU()(fc_out_label)
    fc_out_label = TimeDistributed(Dense(10, activation = 'softmax'))(fc_out_label)

    inputs = [input_layer_glimpse, input_layer_location]

    if train == True:
        selection_location = Input(batch_shape = (batch_size, None, 32, 32))
        selection_label = Input(batch_shape = (batch_size, None, 10))
        fc_out_location = merge([fc_out_location, selection_location], mode = 'mul')
        fc_out_label = merge([fc_out_label, selection_label], mode = 'mul')
        inputs += [selection_location, selection_label]

    return Model(input = inputs, output = [fc_out_location, fc_out_label])

def policy_search(reward, actions): #(y_true, y_pred)
    # Sum over feature axes. Note we use a negative sign to turn maximisation into minimisation for Keras
    axis = [i for i in xrange(2, len(actions._keras_shape))]
    print axis
    selected_actions = K.sum(actions, axis = axis)

    selected_actions = K.clip(selected_actions, 1e-8, 1 - 1e-8) # Clip for numerical stability
    sum_logs = -K.sum(K.log(selected_actions), axis = 1)

    expectation = sum_logs * K.sum(reward, axis = 1) / reward.shape[0] # Keras will sum over batch axis for us
    return expectation

model_run = build_model(train = False) # We will use this one to run and generate a series of actions
model_train = build_model(train = True) # We will use this one to train once we know the actions
model_run.set_weights(model_train.get_weights())

print 'Compiling models'

model_run.compile(loss = 'mse', optimizer = 'adam') # Loss and optimizer does not matter as we will not use this model to train
model_train.compile(loss = policy_search, optimizer = 'adam')

print 'Loading data'
data_train = sio.loadmat('train_32x32.mat')
x_train = np.transpose(data_train['X'], (3, 2, 0 ,1))
y_train = data_train['y']
#y_train = np.transpose(data_train['y'], (1, 0))
print x_train.shape, y_train.shape


print 'Training model'
for j in xrange(epochs):
    reward_total = 0
    for k in xrange(0, x_train.shape[0], batch_size):
        if x_train.shape[0] - k < batch_size:
            break

        # Only need to store these for each batch
        x_batch_history = []
        location_history = []
        selected_label_history = []
        selected_location_history = []

        x_train_cur = x_train[k:k + batch_size]
        y_train_cur = y_train[k:k + batch_size]
        for t in xrange(glimpse_num):
            locations = np.random.random_integers(low = 0, high = img_size - glimpse_size - 1, size = (batch_size, 1, 2)) if t == 0 else locations # Initialise locations if necessary
            location_history += [locations] # Add it to the history
            # Get glimpses
            x_batch = np.zeros((batch_size, 1, x_train.shape[1], glimpse_size, glimpse_size))
            for i in xrange(locations.shape[0]):
                x_batch[i, 0] = x_train_cur[i, :, locations[i, 0, 0]:locations[i, 0, 0] + glimpse_size, locations[i, 0, 1]:locations[i, 0, 1] + glimpse_size]
            x_batch_history += [x_batch]

            # Run the model to get actions
            actions = model_run.predict_on_batch([x_batch, locations])
            # Apply greedy epsilon policy to introduce stochasticness
            location_actions = epsilon_greedy(epsilon, actions[0])
            label_actions = epsilon_greedy(epsilon, actions[1])

            # Get the final actions
            for i in xrange(locations.shape[0]):
                locations[i, 0] = np.unravel_index(np.argmax(location_actions[i, 0]), location_actions[i, 0].shape)
            locations = np.clip(locations, 0, img_size - glimpse_size)

            labels = np.argmax(label_actions, axis = 2)

            # Create reward arrays for the locations. This keeps track of actions chosen
            #print location_actions.shape, locations.shape
            selected_locations = np.zeros_like(location_actions)
            selected_labels = np.zeros_like(label_actions)
            for i in xrange(locations.shape[0]):
                selected_locations[i, 0, locations[i, 0, 0], locations[i, 0, 1]] = 1
                selected_labels[i, 0, labels[i]] = 1
            selected_location_history += [selected_locations]
            selected_label_history += [selected_labels]

        # Allocate rewards for each action
        reward = (labels == y_train_cur).astype(np.float32)

        selected_locations = np.zeros_like(location_actions)
        selected_labels = np.zeros_like(label_actions)
        for i in xrange(locations.shape[0]):
            selected_locations[i, 0, locations[i, 0, 0], locations[i, 0, 1]] = 1
            selected_labels[i, 0, labels[i]] = 1
        reward_location += [selected_locations]
        reward_labe += [selected_labels]

        # We have the history of locations, and the reward so we can train
        x_batch_history = np.concatenate(x_batch_history, axis = 1)
        location_history = np.concatenate(location_history, axis = 1)
        selected_label_history = np.concatenate(selected_label_history, axis = 1)
        selected_location_history = np.concatenate(selected_location_history, axis = 1)
        #print x_batch_history.shape, location_history.shape, selected_label_history.shape, selected_location_history.shape

        # Train on the training model
        model_train.train_on_batch([x_batch_history, location_history, selected_location_history, selected_label_history], [reward, reward])
        #print model_train.evaluate([x_batch_history, location_history], [selected_location_history, selected_label_history]), np.sum(selected_label_history)
        reward_total += np.sum(selected_label_history)

        # Update the weights of the run model
        model_run.set_weights(model_train.get_weights())
        #input()
        if k == 0:
            print "Training begun!"

    print reward_total / (x_train.shape[0] - x_train.shape[0] % batch_size)
