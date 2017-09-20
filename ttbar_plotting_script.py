#Author: Kaviarasan Selvam
#This script calls all the functions necessary to make the required plots of values stored in .h5 files in the directory
#This script should only be run after training the GAN

import numpy as np
import h5py

#from ttbar_plot_values import plot_values
#from ttbar_plot_features import plot_features
from ttbar_plot_values_combined import plot_values_combined
from ttbar_plot_features_combined import plot_features_combined


#Plot features of real dataset
filename1 = 'real_dataset_norm.h5'
file1 = h5py.File(filename1, 'r')
real_dataset_norm = file1.get('TOPGEN_NORM')
#plot_features(real_dataset_norm, True)

#print "\n Plotted features of real_dataset_norm\n"

#Plot values of real dataset
filename2 = 'real_dataset.h5'
file2 = h5py.File(filename2, 'r')
real_dataset = file2.get('TOPGEN')
#plot_values(real_dataset, True)

#print "\n Plotted high-level values of real_dataset\n"

#Plot features of fake dataset
filename3 = 'fake_dataset_norm.h5'
file3 = h5py.File(filename3, 'r')
fake_dataset_norm = np.array(file3.get('TOPGEN_FAKE_NORM'))
#plot_features(fake_dataset_norm, False)

#print "\n Plotted features of fake_dataset_norm\n"

#Plot values of fake dataset
filename4 = 'fake_dataset.h5'
file4 = h5py.File(filename4, 'r')
fake_dataset = np.array(file4.get('TOPGEN_FAKE'))
#plot_values(fake_dataset, False)

#print "\n Plotted high-level values of fake_dataset\n"


print "Plotting combined histograms"

plot_values_combined(real_dataset, fake_dataset)

print "Plotted high-level values (combined)"

plot_features_combined(real_dataset_norm, fake_dataset_norm)

print "Plotted features (combined)"
