#Author: Kaviarasan Selvam
#This function deshapes a dataset that was reshaped to facilitate training of the GAN and returns the deshaped dataset
#Input: (x, 1, 4, 7)
#Output: (x, 38)
#Remark: Values between col 20-29 are zeros (5th and 6th Jet)

# [INPUT]
#  JetPt1     JetEta1     JetPhi1     JetMass1    JetBtag1     LepCharge     LepIsEle
#  JetPt2     JetEta2     JetPhi2     JetMass2    JetBtag2     LepPt         LepEta
#  JetPt3     JetEta3     JetPhi3     JetMass3    JetBtag3     LepPhi        LepIsoPhoton
#  JetPt4     JetEta4     JetPhi4     JetMass4    JetBtag4     LepIsoChHad   LepIsoNeuHad


import numpy as np

def deshape(dataset):

  #reshaping to remove the extra dimension needed for GAN
  #(x, 1, 4, 7) -> (x, 4, 7)
  temp = dataset.reshape(dataset.shape[0], dataset.shape[2], dataset.shape[3])

  num_events = dataset.shape[0]
  num_features = 38
  
  new_dataset = np.zeros((num_events, num_features))

  for event_idx in range(num_events):
    #Jet
    num_rows_jet = 4
    num_cols_jet = 5

    for row_idx in range(num_rows_jet):

      for col_idx in range(num_cols_jet):

        new_dataset[event_idx][row_idx*5 + col_idx] = temp[event_idx][row_idx][col_idx]

    #Lepton
    lep_min_idx = 30

    new_dataset[event_idx][lep_min_idx + 0] = temp[event_idx][0][5]
    new_dataset[event_idx][lep_min_idx + 1] = temp[event_idx][0][6]
    new_dataset[event_idx][lep_min_idx + 2] = temp[event_idx][1][5]
    new_dataset[event_idx][lep_min_idx + 3] = temp[event_idx][1][6]
    new_dataset[event_idx][lep_min_idx + 4] = temp[event_idx][2][5]
    new_dataset[event_idx][lep_min_idx + 5] = temp[event_idx][2][6]
    new_dataset[event_idx][lep_min_idx + 6] = temp[event_idx][3][5]
    new_dataset[event_idx][lep_min_idx + 7] = temp[event_idx][3][6]

  return new_dataset
