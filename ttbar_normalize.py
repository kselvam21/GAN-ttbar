#Input dataset shape: (x, 38)

import numpy as np
import h5py
import math

def normalize(dataset):

  dataset_shape = dataset.shape
  num_rows = dataset_shape[0]
  num_cols = dataset_shape[1]

  new_dataset = np.zeros((num_rows, num_cols))

  #Disable warnings for divide by 0
  np.seterr(divide='ignore', invalid='ignore')

  #Normalizing JetPt and lepPt
  #Getting Pt ratios of JetPt after the Jet1
  for i in range(5, 0, -1):
    new_dataset[:, 5*i] = dataset[:, 5*i]/dataset[:, 5*(i-1)]

  #Saving JetPt1 and LepPt as it is
  new_dataset[:, 0] = dataset[:, 0]
  new_dataset[:, 32] = dataset[:, 32]
  
  new_dataset = np.nan_to_num(new_dataset)

  max_val_arr_Pt = new_dataset.max(axis=0)
  np.savetxt("max_val_arr_Pt.txt", max_val_arr_Pt)

  num_jets = 6
  for i in range(num_jets):
    new_dataset[:, 5*i] = new_dataset[:, 5*i]/max_val_arr_Pt[5*i]

  new_dataset[:, 32] = new_dataset[:, 32]/max_val_arr_Pt[32]

  new_dataset = np.nan_to_num(new_dataset)

  #Normalizing JetMass
  max_val_arr = dataset.max(axis=0)
  np.savetxt("max_val_arr.txt", max_val_arr)
  num_jets = 6
  for i in range(num_jets):
    new_dataset[:, 5*i + 3] = dataset[:, 5*i + 3]/max_val_arr[5*i + 3]
  new_dataset = np.nan_to_num(new_dataset)

  #Normalizing the rest
  jet_min_idx = 0
  jet_max_idx = 29
  lep_min_idx = 30
  lep_max_idx = 37

  for i in range(num_rows):

    #analyzing jets (index 0-29)                                                                                                               
    for jet_idx in range(jet_min_idx, jet_max_idx + 1):

      #JetPt (NOT TO BE CHANGED)

      #JetEta                                                                                                            
      if jet_idx%5 == 1:
        if dataset[i][jet_idx] == 0:
          new_dataset[i][jet_idx] = dataset[i][jet_idx]
        else:
          new_dataset[i][jet_idx] = ((dataset[i][jet_idx]) + 4)/8.0

      #JetPhi                                                                                                                                  
      if jet_idx%5 == 2:
        if dataset[i][jet_idx] == 0:
          new_dataset[i][jet_idx] = dataset[i][jet_idx]
        else:
          #new_dataset[i][jet_idx] = ((dataset[i][jet_idx]) + math.pi)/(2*(math.pi))
          new_dataset[i][jet_idx] = (dataset[i][jet_idx])/(2*(math.pi))

      #JetMass (NOT TO BE CHANGED)

      #JetBtag (NO NEED FOR NORMALIZATION)
      if jet_idx%5 == 4:
        new_dataset[i][jet_idx] = dataset[i][jet_idx]


    #analyzing leptons (index 30-37)
    for lep_idx in range(lep_min_idx, lep_max_idx + 1):

      #LepCharge
      if (lep_idx - lep_min_idx)%8 == 0:
        if dataset[i][lep_idx] == 0.0:
          new_dataset[i][lep_idx] = dataset[i][lep_idx]
        elif dataset[i][lep_idx] == -1.0:
          new_dataset[i][lep_idx] = 0.5
        elif dataset[i][lep_idx] == 1.0:
          new_dataset[i][lep_idx] = 1.0
        else:
          print "There was a weird value for LepCharge in Event ", i, " in column ", lep_idx
          print "The value was ", dataset[i][lep_idx]
          new_dataset[i][lep_idx] = 0

      #LepIsEle (NO NEED FOR NORMALIZATION)
      if (lep_idx - lep_min_idx)%8 == 1:
        new_dataset[i][lep_idx] = dataset[i][lep_idx]

      #LepPt (NOT TO BE CHANGED)

      #LepEta                                                                                                                                  
      if (lep_idx - lep_min_idx)%8 == 3:
        if dataset[i][lep_idx] == 0:
          new_dataset[i][lep_idx] = dataset[i][lep_idx]
        else:
          new_dataset[i][lep_idx] = ((dataset[i][lep_idx]) + 3)/6.0

      #LepPhi                                                                                                                                  
      if (lep_idx - lep_min_idx)%8 == 4:
        if dataset[i][lep_idx] == 0:
          new_dataset[i][lep_idx] = dataset[i][lep_idx]
        else:
          #new_dataset[i][lep_idx] = ((dataset[i][lep_idx]) + math.pi)/(2*(math.pi))
          new_dataset[i][lep_idx] = (dataset[i][lep_idx])/(2*(math.pi))

      #LepIsoPhoton 
      if (lep_idx - lep_min_idx)%8 == 5:
        new_dataset[i][lep_idx] = (dataset[i][lep_idx])/30.0

      #LepIsoChHad                                                                                                                             
      if (lep_idx - lep_min_idx)%8 == 6:
        new_dataset[i][lep_idx] = (dataset[i][lep_idx])/30.0

      #LepIsoNeuHad                                                                                                                            
      if (lep_idx - lep_min_idx)%8 == 7:
        new_dataset[i][lep_idx] = (dataset[i][lep_idx])/30.0

  return new_dataset
