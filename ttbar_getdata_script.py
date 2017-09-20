#This script gets all the datasets from the directory and saves them in 1 hdf5 file

import numpy as np
import h5py
import glob

from ttbar_normalize import normalize

####################################################################
#Getting dataset

#myDatasets = ()
##for myFileName in glob.glob('/afs/cern.ch/user/k/kselvam/datasets/TOPGEN/REDUCED_ttbar_lepFilter_13TeV_*.h5'): 
#for myFileName in glob.glob('/eos/user/m/mpierini/SYNC/DeepLearning/TOPGEN/DATA/ttbar_lepFilter_13TeV/REDUCED_ttbar_lepFilter_13TeV_9*.h5'):
#    #print "filename: ", myFileName
#    myFile = h5py.File(myFileName)
#    #print "dataset.shape: ", (myFile.get('TOPGEN')).shape
#    myDatasets = myDatasets + (myFile.get('TOPGEN'), )
#myDataset = np.concatenate(myDatasets, axis=0)

#real_dataset = myDataset

#print "\nLoaded real dataset\n"
#print "\nDataset shape (before filtering): ", real_dataset.shape, "\n"

####################################################################


#########################################################################
##Saving real samples in .h5 file for future use

#f1 = h5py.File("real_dataset9.h5", 'w')
#f1.create_dataset("TOPGEN9", data=real_dataset)
#f1.close()

#print "\nsaved real dataset9\n"

########################################################################


#######################################################################
#Concatenating saved datasets
filename1 = 'real_dataset1.h5'
file1 = h5py.File(filename1, 'r')
real_dataset1 = file1.get('TOPGEN1')

filename2 = 'real_dataset2.h5'
file2 = h5py.File(filename2, 'r')
real_dataset2 = file2.get('TOPGEN2')

filename3 = 'real_dataset3.h5'
file3 = h5py.File(filename3, 'r')
real_dataset3 = file3.get('TOPGEN3')

filename5 = 'real_dataset5.h5'
file5 = h5py.File(filename5, 'r')
real_dataset5 = file5.get('TOPGEN5')

filename6 = 'real_dataset6.h5'
file6 = h5py.File(filename6, 'r')
real_dataset6 = file6.get('TOPGEN6')

filename7 = 'real_dataset7.h5'
file7 = h5py.File(filename7, 'r')
real_dataset7 = file7.get('TOPGEN7')

filename8 = 'real_dataset8.h5'
file8 = h5py.File(filename8, 'r')
real_dataset8 = file8.get('TOPGEN8')

filename9 = 'real_dataset9.h5'
file9 = h5py.File(filename9, 'r')
real_dataset9 = file9.get('TOPGEN9')

real_dataset = np.concatenate((real_dataset1, real_dataset2, real_dataset3, real_dataset5, real_dataset6, real_dataset7, real_dataset8, real_dataset9), axis=0)


#print "real_dataset.shape: ", real_dataset.shape

#######################################################################


####################################################################
#Filtering dataset
#Filter1: Remove events with 0 Jets and 0 Leptons
HasJet = real_dataset[:,0]>0
HasLepton = real_dataset[:,32]>0
Good = HasLepton*HasJet

real_dataset = real_dataset[Good,:]
print "\nDataset shape (after filtering1): ", real_dataset.shape, "\n"


#Filter2: Select events with only 4 jets
HasJet4 = real_dataset[:,15]>0
NoJet5 = real_dataset[:, 20] == 0
good_events = HasJet4*NoJet5

real_dataset = real_dataset[good_events,:]

print "\nDataset shape (after filtering2): ", real_dataset.shape, "\n"

#print "real_dataset[0][:]: ", real_dataset[0][:]
##print "real_dataset[30][:]: ", real_dataset[30][:]


######################################################################## 


#########################################################################
#Saving real samples in .h5 file for future use

f1 = h5py.File("real_dataset.h5", 'w')
f1.create_dataset("TOPGEN", data=real_dataset)
f1.close()

print "\nsaved real dataset\n"

########################################################################


#######################################################################
#Normalize dataset and save normalized dataset

real_dataset_norm = normalize(real_dataset)

f2 = h5py.File("real_dataset_norm.h5", 'w')
f2.create_dataset("TOPGEN_NORM", data=real_dataset_norm)
f2.close()

print "\nsaved real dataset normalized\n"

#######################################################################
