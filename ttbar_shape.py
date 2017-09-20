#Author: Kaviarasan Selvam
#This function reshapes the input dataset to facilitate the training of the GAN
#Input Shape: (x, 38)
#Output Shape: (x, 1, 4, 7)

# [INPUT]
#  JetPt1     JetEta1     JetPhi1     JetMass1    JetBtag1     LepCharge     LepIsEle
#  JetPt2     JetEta2     JetPhi2     JetMass2    JetBtag2     LepPt         LepEta
#  JetPt3     JetEta3     JetPhi3     JetMass3    JetBtag3     LepPhi        LepIsoPhoton
#  JetPt4     JetEta4     JetPhi4     JetMass4    JetBtag4     LepIsoChHad   LepIsoNeuHad


import numpy as np
import math

def shape(dataset):

  num_events = dataset.shape[0]
  num_features = dataset.shape[1]
  num_rows = 4
  num_cols = 7

  new_dataset = np.zeros((num_events, num_rows, num_cols))

  for event_idx in range(num_events):
 
    #Jets
    for row_idx in range(4):
 
      for col_idx in range(5):

        new_dataset[event_idx][row_idx][col_idx] = dataset[event_idx][row_idx*5 + col_idx]

    #Leptons
    lep_min_idx = 30

    new_dataset[event_idx][0][5] = dataset[event_idx][lep_min_idx+0]
    new_dataset[event_idx][0][6] = dataset[event_idx][lep_min_idx+1]
    new_dataset[event_idx][1][5] = dataset[event_idx][lep_min_idx+2]
    new_dataset[event_idx][1][6] = dataset[event_idx][lep_min_idx+3]
    new_dataset[event_idx][2][5] = dataset[event_idx][lep_min_idx+4]
    new_dataset[event_idx][2][6] = dataset[event_idx][lep_min_idx+5]
    new_dataset[event_idx][3][5] = dataset[event_idx][lep_min_idx+6]
    new_dataset[event_idx][3][6] = dataset[event_idx][lep_min_idx+7]
      
  #Adding extra dimensionality for compatibility with keras networks
  new_dataset = new_dataset.reshape(new_dataset.shape[0], 1, new_dataset.shape[1], new_dataset.shape[2])
  return new_dataset
