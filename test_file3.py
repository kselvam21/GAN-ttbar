#To test ttbar_reshape

import h5py
from ttbar_normalize import normalize
from ttbar_denormalize import denormalize
from ttbar_shape import shape
from ttbar_deshape import deshape

#f = h5py.File("/eos/user/m/mpierini/TOPGEN/ttbar_lepFilter_13TeV/REDUCED_ttbar_lepFilter_13TeV_1.h5")
f = h5py.File("/eos/user/m/mpierini/SYNC/DeepLearning/TOPGEN/DATA/ttbar_lepFilter_13TeV/REDUCED_ttbar_lepFilter_13TeV_1.h5")
dataset = f.get("TOPGEN")

print "dataset.shape: ", dataset.shape

dataset = dataset[0:500, :]

real_dataset_norm = normalize(dataset)

#print "real_dataset_norm.shape: ", real_dataset_norm.shape
#print "real_dataset_norm[30][:]: ", real_dataset_norm[30][:]

real_dataset_norm = deshape(shape(real_dataset_norm))
real_dataset_denorm = denormalize(real_dataset_norm)

print "dataset[31][:]: ", dataset[10][:]
print "real_dataset_denorm[31][:]: ", real_dataset_denorm[10][:]

test = dataset[10][:] - real_dataset_denorm[10][:]
print "test: ", test


#print "dataset[:, 60]: ", dataset[:, 60]
