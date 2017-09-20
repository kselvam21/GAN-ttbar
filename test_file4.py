import numpy as np
import h5py
import glob

from test_file5 import plot_features
#from ttbar_filter_dataset import filter_dataset

myDatasets = ()
#for myFileName in glob.glob('/afs/cern.ch/user/k/kselvam/datasets/TOPGEN/REDUCED_ttbar_lepFilter_13TeV_*.h5'):
#for myFileName in glob.glob('/eos/user/m/mpierini/TOPGEN/ttbar_lepFilter_13TeV/REDUCED_ttbar_lepFilter_13TeV_19*.h5'):
for myFileName in glob.glob('/eos/user/m/mpierini/SYNC/DeepLearning/TOPGEN/DATA/ttbar_lepFilter_13TeV/REDUCED_ttbar_lepFilter_13TeV_4092.h5'):
    print "filename: ", myFileName
    myFile = h5py.File(myFileName)
    myDatasets = myDatasets + (myFile.get('TOPGEN'), )
    temp_dataset = myFile.get('TOPGEN')
    #print "filename: ", myFileName
    print "dataset shape: ", temp_dataset.shape, "\n"
myDataset = np.concatenate(myDatasets, axis=0)

real_dataset = myDataset

#print "\nLoaded dataset\n"
print "\nreal_dataset.shape: ", real_dataset.shape, "\n"

#HasJet = real_dataset[:,0]>0
#HasLepton = real_dataset[:,52]>0
#Good = HasLepton*HasJet
#real_dataset = real_dataset[Good,:]

#print "\nreal_dataset.shape (before filtering): ", real_dataset.shape, "\n"

#real_dataset = filter_dataset(real_dataset)

#print "\nreal_dataset.shape (after filtering): ", real_dataset.shape, "\n"

#Plot features from actual dataset
#plot_features(real_dataset)


