#Author: Kaviarasan Selvam
#This function denormalizes input dataset based on predetermined values and outputs denormalized dataset
#Input shape: (x, 38)
#Output shape: (x, 38)

import numpy as np
import math

def denormalize(dataset):
  
  dataset_shape = dataset.shape
  num_rows = dataset_shape[0]
  num_cols = dataset_shape[1]

  new_dataset = np.zeros((num_rows, num_cols))

  #Denormalizing JetPt and LepPt
  max_val_arr_Pt = np.loadtxt("max_val_arr_Pt.txt")
  num_jets = 6
  for i in range(num_jets):
    new_dataset[:, i*5] = dataset[:, i*5]*max_val_arr_Pt[i*5]

  for i in range(1,num_jets):
    new_dataset[:, i*5] = dataset[:, i*5]*new_dataset[:, 5*(i-1)] 

  new_dataset[:, 32] = dataset[:, 32]*max_val_arr_Pt[32]
  

  #Denormalizing JetMass
  max_val_arr = np.loadtxt("max_val_arr.txt")
  num_jets = 6
  for i in range(num_jets):
    new_dataset[:, i*5 + 3] = dataset[:, i*5 + 3]*max_val_arr[i*5 + 3]

  #Denormalizing the rest
  jet_min_idx = 0
  jet_max_idx = 29
  lep_min_idx = 30
  lep_max_idx = 37


  for i in range(num_rows):
  
    #analyzing jets (index 0-49)                                                                                                               
    for jet_idx in range(jet_min_idx, jet_max_idx + 1):

      #JetPt (NO TO BE CHANGED)
      #if jet_idx%5 == 0
      
      #JetEta                                                                                                                               
      if jet_idx%5 == 1:
        if dataset[i][jet_idx] == 0:
          new_dataset[i][jet_idx] = dataset[i][jet_idx]
        else:
          new_dataset[i][jet_idx] = ((dataset[i][jet_idx])*8.0) - 4

      #JetPhi                                                                                                                                 
      if jet_idx%5 == 2:
        if dataset[i][jet_idx] == 0:
          new_dataset[i][jet_idx] = dataset[i][jet_idx]
        else:
          #new_dataset[i][jet_idx] = ((dataset[i][jet_idx])*(2*(math.pi))) - math.pi
          new_dataset[i][jet_idx] = (dataset[i][jet_idx])*(2*(math.pi))

      #JetMass (NOT TO BE CHANGED)                                                                                                  
      #if jet_idx%5 == 3
      
      #JetBtag (NO NEED FOR DENORMALIZATION)                                                                    
      if jet_idx%5 == 4:
        new_dataset[i][jet_idx] = dataset[i][jet_idx]

    #analyzing leptons (index 30-37)
    for lep_idx in range(lep_min_idx, lep_max_idx + 1):
    
      #LepCharge (DIFFICULT TO DENORMALIZE DUE TO DISCREET VALUES)
      if (lep_idx - lep_min_idx)%8 == 0:
        new_dataset[i][lep_idx] = dataset[i][lep_idx]

      #LepIsEle (NO NEED FOR DENORMALIZATION)
      if (lep_idx - lep_min_idx)%8 == 1:
        new_dataset[i][lep_idx] = dataset[i][lep_idx]

      #LepPt (NOT TO BE CHANGED)
      #if (lep_idx - lep_min_idx)%8 == 2:
      
      #LepEta 
      if (lep_idx - lep_min_idx)%8 == 3:
        if dataset[i][lep_idx] == 0:
          new_dataset[i][lep_idx] = dataset[i][lep_idx]
        else:
          new_dataset[i][lep_idx] = ((dataset[i][lep_idx])*6) - 3

      #LepPhi
      if (lep_idx - lep_min_idx)%8 == 4:
        if dataset[i][lep_idx] == 0:
          new_dataset[i][lep_idx] = dataset[i][lep_idx]
        else:
          #new_dataset[i][lep_idx] = ((dataset[i][lep_idx])*(2*(math.pi))) - math.pi
          new_dataset[i][lep_idx] = (dataset[i][lep_idx])*(2*(math.pi))

      #LepIsoPhoton
      if (lep_idx - lep_min_idx)%8 == 5:
        new_dataset[i][lep_idx] = (dataset[i][lep_idx])*30.0

      #LepIsoChHad
      if (lep_idx - lep_min_idx)%8 == 6:
        new_dataset[i][lep_idx] = (dataset[i][lep_idx])*30.0

      #LepIsoNeuHad
      if (lep_idx - lep_min_idx)%8 == 7:
        new_dataset[i][lep_idx] = (dataset[i][lep_idx])*30.0

  return new_dataset
