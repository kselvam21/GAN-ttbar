#Generative Adversarial Network for ttbar events
#Based on: https://github.com/osh/KerasGAN/blob/master/MNIST_CNN_GAN.ipynb

#########################################################################
#Import statements

import time
start_time = time.time()

import os, random
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
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

from ttbar_normalize import normalize
from ttbar_denormalize import denormalize
from ttbar_plot_loss import plot_loss, plot_gloss, plot_dloss
from ttbar_shape import shape
from ttbar_deshape import deshape

#########################################################################


#########################################################################
#Getting real_dataset
#real_dataset.shape = (x, 38)
#NUM FILES TO BE CHANGED

import h5py
import glob

#myDatasets = ()
##for myFileName in glob.glob('/afs/cern.ch/user/k/kselvam/datasets/TOPGEN/REDUCED_ttbar_lepFilter_13TeV_*.h5'):
#for myFileName in glob.glob('/eos/user/m/mpierini/SYNC/DeepLearning/TOPGEN/DATA/ttbar_lepFilter_13TeV/REDUCED_ttbar_lepFilter_13TeV_*.h5'):
#    myFile = h5py.File(myFileName)
#    myDatasets = myDatasets + (myFile.get('TOPGEN'), )
#myDataset = np.concatenate(myDatasets, axis=0)
#
#real_dataset = myDataset
#
#print "\nLoaded real dataset\n"
#print "\nDataset shape (before filtering): ", real_dataset.shape, "\n"
#########################################################################


########################################################################
#Filtering for good data
#Removing dataset with 0 jets and 0 leptons
#Removing events with JetPt > 180 and LepPt > 180

#HasJet = real_dataset[:,0]>0
#HasLepton = real_dataset[:,32]>0
#Good = HasLepton*HasJet

#real_dataset = real_dataset[Good,:]

#print "\nDataset shape (after filtering): ", real_dataset.shape, "\n"

########################################################################


#########################################################################
#Saving real samples in .h5 file for future use

#f1 = h5py.File("real_dataset.h5", 'w')
#f1.create_dataset("TOPGEN", data=real_dataset)
#f1.close()

#print "\nsaved real dataset\n"
#########################################################################


#########################################################################
#Normalize dataset and save normalized dataset                           
#The zero padding is not included here

#real_dataset_norm = normalize(real_dataset)

#f2 = h5py.File("real_dataset_norm.h5", 'w')
#f2.create_dataset("TOPGEN_NORM", data=real_dataset_norm)
#f2.close()

#print "\nsaved real dataset normalized\n"

#########################################################################


########################################################################
#Getting normalized saved datasets
#Only run this part of the code after running ttbar_getdataset_script.py

filename1 = 'real_dataset_norm.h5'
file1 = h5py.File(filename1, 'r')
real_dataset_norm = np.array(file1.get('TOPGEN_NORM'))

print "\nLoaded normalized dataset \n"

#######################################################################


########################################################################
#Reshaping dataset (also adding zero padding)
#(x, 38) --> (x, 1, 6, 8)
#Extra dimension added for compatibility with keras functions 

real_dataset_norm = shape(real_dataset_norm)

########################################################################


#########################################################################
#Splitting training and testing data
#Train:Test ratio = 70:30

num_events = real_dataset_norm.shape[0]

num_train = int(0.9*num_events)

x_train = real_dataset_norm[0:num_train][:][:][:]
x_test = real_dataset_norm[num_train:][:][:][:]

#######################################################################


########################################################################
#Initializing parameters

img_shape = x_train.shape[1:]
# = (1, 6, 8)
dropout_rate = 0.25
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-3)

########################################################################


########################################################################
#Building Generator Model
#Take x random inputs and map them down to (1, 4, 7) pixel to match dataset

#num_rand_inputs = 100

#g_input = Input(shape=[num_rand_inputs])
#G = Dense(200*1*7, kernel_initializer='glorot_normal')(g_input)
#G = BatchNormalization()(G)
#G = Activation('relu')(G)
#G = Reshape( [200, 1, 7] )(G)
#G = UpSampling2D(size=(2,1), data_format="channels_first")(G)
#G = Conv2D(100, (3, 3), padding='same', kernel_initializer='glorot_uniform', data_format="channels_first")(G)
##G = Conv2D(100, (2, 2), padding='same', kernel_initializer='glorot_uniform', data_format="channels_first")(G)
#G = BatchNormalization()(G)
#G = Activation('relu')(G)
#G = UpSampling2D(size=(2,1), data_format="channels_first")(G)
#G = Conv2D(50, (3, 3), padding='same', kernel_initializer='glorot_uniform', data_format="channels_first")(G)
##G = Conv2D(50, (2, 2), padding='same', kernel_initializer='glorot_uniform', data_format="channels_first")(G)
#G = BatchNormalization()(G)
#G = Activation('relu')(G)
#G = Conv2D(1, (1, 1), padding='same', kernel_initializer='glorot_uniform', data_format="channels_first")(G)
#g_output = Activation('sigmoid')(G)

num_rand_inputs = 10

g_input = Input(shape=[num_rand_inputs])
G = Dense(50*1*7, kernel_initializer='glorot_normal')(g_input)
G = BatchNormalization()(G)
G = Activation('relu')(G)
G = Reshape( [50, 1, 7] )(G)
G = UpSampling2D(size=(2,1), data_format="channels_first")(G)
G = Conv2D(25, (3, 3), padding='same', kernel_initializer='glorot_uniform', data_format="channels_first")(G)
G = BatchNormalization()(G)
G = Activation('relu')(G)
G = UpSampling2D(size=(2,1), data_format="channels_first")(G)
G = Conv2D(10, (3, 3), padding='same', kernel_initializer='glorot_uniform', data_format="channels_first")(G)
G = BatchNormalization()(G)
G = Activation('relu')(G)
G = Conv2D(1, (1, 1), padding='same', kernel_initializer='glorot_uniform', data_format="channels_first")(G)
g_output = Activation('sigmoid')(G)


generator = Model(inputs=g_input, outputs=g_output)
generator.compile(loss='binary_crossentropy', optimizer=opt)
generator.summary()

plot_model(generator, to_file='/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/model_plots/g_model.png', show_shapes=True)

########################################################################


########################################################################
#Build Discriminator Model

d_input = Input(shape=img_shape)
D = Conv2D(256, (5, 5), strides=(2,2), padding='same', data_format="channels_first")(d_input)
#D = Conv2D(50, (5, 5), strides=(2,2), padding='same', data_format="channels_first")(d_input)
D = LeakyReLU(0.2)(D)
D = Dropout(dropout_rate)(D)
D = Conv2D(512, (5, 5), strides=(2,2), padding='same', data_format="channels_first")(D)
#D = Conv2D(100, (5, 5), strides=(2,2), padding='same', data_format="channels_first")(D)
D = LeakyReLU(0.2)(D)
D = Dropout(dropout_rate)(D)
D = Flatten()(D)
D = Dense(256, activation='relu')(D)
#D = Dense(50, activation='relu')(D)
D = LeakyReLU(0.2)(D)
D = Dropout(dropout_rate)(D)
d_output = Dense(2, activation='softmax')(D)

discriminator = Model(inputs=d_input, outputs=d_output)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
discriminator.summary()

plot_model(discriminator, to_file='/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/model_plots/d_model.png', show_shapes=True)

#########################################################################


#########################################################################
#Freeze weights in the discriminator for stacked training

def make_trainable(net, val):
  net.trainable = val
  for i in net.layers:
    i.trainable = val

#make_trainable(discriminator, False)

#########################################################################


#########################################################################
#Label flipping

#def flip_labels(arr):
#  Y = arr
#  num_rows = arr.shape[0]
#  for i in range(num_rows):
#    if i%100 == 0:
#      if arr[i] == 1:
#        Y[i] = 0
#      else:
#        Y[i] = 1

#  return Y

#########################################################################


#########################################################################
#Label smoothing

#def smooth_labels(arr):
#  Y = arr
#  num_rows = arr.shape[0]
#  for i in range(num_rows):
#    if Y[i] == 0:
#      Y[i] = np.random.uniform(0.0, 0.3)
#    else:
#      Y[i] = np.random.uniform(0.7, 1.2)

#  return Y

########################################################################


#########################################################################
#Build stacked GAN Model

gan_input = Input(shape=[num_rand_inputs])
H = generator(gan_input)
gan_output = discriminator(H)
GAN = Model(gan_input, gan_output)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
#GAN.compile(loss='binary_crossentropy', optimizer=opt)
GAN.summary()

plot_model(GAN, to_file='/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/model_plots/gan_model.png', show_shapes=True)

#########################################################################


#########################################################################
#Pre-train the discriminator network

ntrain = 900
trainidx = random.sample(range(0, x_train.shape[0]), ntrain)
XT = x_train[trainidx, :, :, :]

#noise_gen = np.random.normal(0.5, 0.2, size=[XT.shape[0], num_rand_inputs])
noise_gen = np.random.uniform(0, 1, size=[XT.shape[0], num_rand_inputs])
generated_images = generator.predict(noise_gen)

X = np.concatenate((XT, generated_images))

n = XT.shape[0]
Y = np.zeros([2*n, 2])
Y[:n, 1] = 1
Y[n:, 0] = 1
#Y = np.zeros(2*n)
#Y[:n] = 1
#Y[n:] = 0

#Y = flip_labels(Y)
#Y = smooth_labels(Y)

X = X.astype('float32')
Y = Y.astype('float32')

make_trainable(discriminator, True)
discriminator.fit(X, Y, epochs=1, batch_size=100, shuffle=True)
#y_hat = discriminator.predict(X)

###########################################################################


###########################################################################
#Set up loss storage vector

losses = {"d":[], "g":[]}

###########################################################################


############################################################################
#Set up the main training loop

def train_for_n(nb_epoch=5000, plt_frq=25, BATCH_SIZE=32):
  
  for e in range(nb_epoch):
    #Make generative images    
    image_batch = x_train[np.random.randint(0, x_train.shape[0], size=BATCH_SIZE), :, :, :]
    #noise_gen = np.random.normal(0.5, 0.2, size=[BATCH_SIZE, num_rand_inputs])
    noise_gen = np.random.uniform(0, 1, size=[BATCH_SIZE, num_rand_inputs])
    generated_images = generator.predict(noise_gen)

    if e == (nb_epoch-1):
      #noise = np.random.normal(0.5, 0.2, size=[1000000, num_rand_inputs])
      noise = np.random.uniform(0, 1, size=[1000000, num_rand_inputs])
      gen_images = generator.predict(noise)

      fake_dataset_norm = np.array(deshape(gen_images))
      
      #Saving normalized dataset
      f3 = h5py.File("fake_dataset_norm.h5", 'w')
      f3.create_dataset("TOPGEN_FAKE_NORM", data=fake_dataset_norm)
      f3.close()

      #Saving denormalized dataset
      fake_dataset = np.array(denormalize(fake_dataset_norm))
      #WARNING: Values of EleCharge not denormalized
      f4 = h5py.File("fake_dataset.h5", 'w')
      f4.create_dataset("TOPGEN_FAKE", data=fake_dataset)
      f4.close()

      #Saving Generator model and weights
      generator_json = generator.to_json()
      with open ("generator.json", "w") as json_file:
        json_file.write(generator_json)
      generator.save_weights("generator_weights.h5")

      #Saving Discriminator model and weights
      discriminator_json = discriminator.to_json()
      with open ("discriminator.json", "w") as json_file:
        json_file.write(discriminator_json)
      discriminator.save_weights("discriminator_weights.h5")

    
    #Train discriminator on generated images
    X = np.concatenate((image_batch, generated_images))
    Y = np.zeros([2*BATCH_SIZE, 2])
    Y[0:BATCH_SIZE, 1] = 1
    Y[BATCH_SIZE:, 0] = 1
    #Y = np.zeros(2*BATCH_SIZE)
    #Y[0:BATCH_SIZE] = 1
    #Y[BATCH_SIZE:] = 0

    #Added to test
    #if e%100 == 0:
    #  Y[0:BATCH_SIZE] = 0
    #  Y[BATCH_SIZE:] = 1

    #Y = flip_labels(Y)
    #Y = smooth_labels(Y)

    make_trainable(discriminator, True)
    d_loss = discriminator.train_on_batch(X, Y)
    losses["d"].append(d_loss)

    #Train generator-discriminator stack on input noise to non_generated output class
    #noise_train = np.random.normal(0.5, 0.2, size=[BATCH_SIZE, num_rand_inputs])
    noise_train = np.random.uniform(0, 1, size=[BATCH_SIZE, num_rand_inputs])
    Y = np.zeros([BATCH_SIZE, 2])
    Y[:, 1] = 1
    #Y = np.zeros(BATCH_SIZE)
    #Y[:] = 1
    #Y[:] = 0

    #Y = flip_labels(Y)
    #Y = smooth_labels(Y)

    make_trainable(discriminator, False)
    g_loss = GAN.train_on_batch(noise_train, Y)
    losses["g"].append(g_loss)

    #Update plots
    if e%plt_frq == plt_frq - 1:
      plot_loss(losses)
      print "\nplotting...\n" 

###########################################################################


###########################################################################
#Train GAN
#More training examples in the sample code on the webpage (url at the top)

#6000 epochs at original learning rates
train_for_n(nb_epoch=100000, plt_frq=500, BATCH_SIZE=500)
#train_for_n(nb_epoch=10, plt_frq=1, BATCH_SIZE=500)

###########################################################################


###########################################################################
#Plot final loss curve

plot_loss(losses)
plot_dloss(losses)
plot_gloss(losses)
###########################################################################


###########################################################################
#Printing time taken to run the script

print "\n", "--- %s seconds ---" % (time.time() - start_time), "\n"

##########################################################################
