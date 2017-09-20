#To print model

import os, random
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
import cPickle, random, sys, keras
from keras.models import Model
from keras.utils import np_utils
from keras.utils import plot_model
from keras.layers.merge import Concatenate
from keras.models import model_from_json
import pydot
import graphviz

dropout_rate = 0.25
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-3)

num_rand_inputs = 100

g_input = Input(shape=[num_rand_inputs])
G = Dense(100*4*5, kernel_initializer='glorot_normal')(g_input)
G = BatchNormalization()(G)
G = Activation('relu')(G)
G = Reshape( [100, 4, 5] )(G)
G = UpSampling2D(size=(2,2), data_format="channels_first")(G)
G = Conv2D(50, (3, 3), padding='same', kernel_initializer='glorot_uniform', data_format="channels_first")(G)
G = BatchNormalization()(G)
G = Activation('relu')(G)
G = Conv2D(25, (3, 3), padding='same', kernel_initializer='glorot_uniform', data_format="channels_first")(G)
G = BatchNormalization()(G)
G = Activation('relu')(G)
G = Conv2D(1, (1, 1), padding='same', kernel_initializer='glorot_uniform', data_format="channels_first")(G)
g_output = Activation('sigmoid')(G)

generator = Model(inputs=g_input, outputs=g_output)
generator.compile(loss='binary_crossentropy', optimizer=opt)
generator.summary()


d_input = Input(shape=[1,8,10])
D = Conv2D(150, (5, 5), strides=(2,2), padding='same', activation='relu', data_format="channels_first")(d_input)
D = LeakyReLU(0.2)(D)
D = Dropout(dropout_rate)(D)
D = Conv2D(200, (5, 5), strides=(2,1), padding='same', activation='relu', data_format="channels_first")(D)
D = LeakyReLU(0.2)(D)
D = Dropout(dropout_rate)(D)
D = Flatten()(D)
D = Dense(150, activation='relu')(D)
D = LeakyReLU(0.2)(D)
D = Dropout(dropout_rate)(D)
d_output = Dense(2, activation='softmax')(D)
#d_output = Dense(1, activation='sigmoid')(D)

discriminator = Model(inputs=d_input, outputs=d_output)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
#discriminator.compile(loss='binary_crossentropy', optimizer=dopt)                                                                             
discriminator.summary()


gan_input = Input(shape=[num_rand_inputs])
H = generator(gan_input)
gan_output = discriminator(H)
GAN = Model(gan_input, gan_output)
#GAN.compile(loss='categorical_crossentropy', optimizer=opt) 
GAN.compile(loss='binary_crossentropy', optimizer=opt)
GAN.summary()


plot_model(GAN, to_file='/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/model_plot/model.png', show_shapes=True)

