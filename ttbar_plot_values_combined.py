#Author: Kaviarasan Selvam
#This function plots high-level physical values using unnormalized low-level features of original and GAN-generated ttbar events 
#Input Shape: (x, 38)
#Output Shape: N/A

import ROOT
import random
import math

def plot_values_combined(real_dataset, fake_dataset):
  
  #Sub-samplong dataset                                                                                                                        
  num_samples = 1000000

  real_dataset = real_dataset[0:num_samples, :]
  fake_dataset = fake_dataset[0:num_samples, :]

  dataset_shape = real_dataset.shape
  num_rows = dataset_shape[0]
  num_cols = dataset_shape[1]

  jet_min_idx = 0
  jet_max_idx_Jet4 = 19
  jet_max_idx = 29
  lep_min_idx = 30
  lep_max_idx = 37

  ROOT.gStyle.SetOptStat(ROOT.kFALSE);

  #Initializing histograms                                                                                                                    
  hHt_real = ROOT.TH1D("hHt_real", "", 200, 0.0, 3000)
  hMHT_pt_real = ROOT.TH1D("hMHT_pt_real", "", 200, 0, 500)
  hMHT_phi_real = ROOT.TH1D("hMHT_phi_real", "", 200, -1, 7)
  hMt_real = ROOT.TH1D("hMt_real", "", 200, 0.0, 500.0)

  hHt_fake = ROOT.TH1D("hHt_fake", "", 200, 0.0, 3000)
  hMHT_pt_fake = ROOT.TH1D("hMHT_pt_fake", "", 200, 0, 500)
  hMHT_phi_fake = ROOT.TH1D("hMHT_phi_fake", "", 200, -1, 7)
  hMt_fake = ROOT.TH1D("hMt_fake", "", 200, 0.0, 500.0)


  #real_dataset
  for i in range(num_rows):
    #initializing necessary variables                                                                                                          
    sum_JetPt_scalar = 0
    sum_LepPt_scalar = 0
    Ht = 0

    JetPt_vec = ROOT.TVector2()
    LepPt_vec = ROOT.TVector2()
    MHT = ROOT.TVector2()

    Mt = 0

    #analyzing jets (index 0-29)                                                                                                               
    for jet_idx in range(jet_min_idx, jet_max_idx_Jet4 + 1):
      #JetPt                                                                                                                                   
      if jet_idx%5 == 0:
        sum_JetPt_scalar += real_dataset[i][jet_idx]
        JetPt_vec += ROOT.TVector2(-real_dataset[i][jet_idx]*math.cos(real_dataset[i][jet_idx+2]), -real_dataset[i][jet_idx]*math.sin(real_dataset[i][jet_idx+2]))

    #analyzing leptons (index 30-37)                                                                                                           
    #LepPt
    lep_idx = 32
    
    sum_LepPt_scalar = real_dataset[i][lep_idx]
    LepPt_vec = ROOT.TVector2(real_dataset[i][lep_idx]*math.cos(real_dataset[i][lep_idx+2]), real_dataset[i][lep_idx]*math.sin(real_dataset[i][lep_idx+2]))


    #Filling in histograms                                                                                                                     
    Ht = sum_JetPt_scalar + sum_LepPt_scalar
    hHt_real.Fill(Ht)

    MHT = JetPt_vec - LepPt_vec
    hMHT_pt_real.Fill(MHT.Mod())
    hMHT_phi_real.Fill(MHT.Phi())

    hMt_real.Fill(MHT.Mod())



  #fake_dataset
  for i in range(num_rows):
    #initializing necessary variables                                                                                                          
    sum_JetPt_scalar = 0
    sum_LepPt_scalar = 0
    Ht = 0

    JetPt_vec = ROOT.TVector2()
    LepPt_vec = ROOT.TVector2()
    MHT = ROOT.TVector2()

    Mt = 0

    #analyzing jets (index 0-29)                                                                                                               
    for jet_idx in range(jet_min_idx, jet_max_idx_Jet4 + 1):
      #JetPt                                                                                                                                   
      if jet_idx%5 == 0:
        sum_JetPt_scalar += fake_dataset[i][jet_idx]
        JetPt_vec += ROOT.TVector2(-fake_dataset[i][jet_idx]*math.cos(fake_dataset[i][jet_idx+2]), -fake_dataset[i][jet_idx]*math.sin(fake_dataset[i][jet_idx+2]))

    #analyzing leptons (index 30-37)
    #LepPt
    lep_idx = 32

    sum_LepPt_scalar = fake_dataset[i][lep_idx]
    LepPt_vec = ROOT.TVector2(fake_dataset[i][lep_idx]*math.cos(fake_dataset[i][lep_idx+2]), fake_dataset[i][lep_idx]*math.sin(fake_dataset[i][lep_idx+2]))

    #Filling in histograms     
    Ht = sum_JetPt_scalar + sum_LepPt_scalar
    hHt_fake.Fill(Ht)

    MHT = JetPt_vec - LepPt_vec
    hMHT_pt_fake.Fill(MHT.Mod())
    hMHT_phi_fake.Fill(MHT.Phi())

    hMt_fake.Fill(MHT.Mod())


  #Making plots                                                                                                                                
  #Ht                                                                                                                                          
  c1 = ROOT.TCanvas("c1", "", 600, 600)
  hHt_real.GetXaxis().SetTitle("Ht")
  hHt_real.GetYaxis().SetTitle("Events")
  hHt_real.SetTitle("Black = Real, Red = Generated")
  hHt_real.SetLineColor(1)
  hHt_real.Draw("pe")
  hHt_fake.SetLineColor(2)
  hHt_fake.Draw("same pe")
  c1.Draw()
  c1.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/value_plots/hHt.png")
  c1.Close()

  #MHT_pt                                                                                                                                      
  c2 = ROOT.TCanvas("c2", "", 600, 600)
  hMHT_pt_real.GetXaxis().SetTitle("MHT_pt")
  hMHT_pt_real.GetYaxis().SetTitle("Events")
  hMHT_pt_real.SetTitle("Black = Real, Red = Generated")
  hMHT_pt_real.SetLineColor(1)
  hMHT_pt_real.Draw("pe")
  hMHT_pt_fake.SetLineColor(2)
  hMHT_pt_fake.Draw("same pe")
  c2.Draw()
  c2.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/value_plots/hMHT_pt.png")
  c2.Close()


  #MHT_phi                                                                                                                                     
  c3 = ROOT.TCanvas("c3", "", 600, 600)
  hMHT_phi_fake.GetXaxis().SetTitle("MHT_phi")
  hMHT_phi_fake.GetYaxis().SetTitle("Events")
  hMHT_phi_fake.SetTitle("Black = Real, Red = Generated")
  hMHT_phi_fake.SetLineColor(2)
  hMHT_phi_fake.Draw("pe")
  hMHT_phi_real.SetLineColor(1)
  hMHT_phi_real.Draw("same pe")
  c3.Draw()
  c3.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/value_plots/hMHT_phi.png")
  c3.Close()


  #Mt                                                                                                                                          
  c4 = ROOT.TCanvas("c4", "", 600, 600)
  hMt_real.GetXaxis().SetTitle("Mt")
  hMt_real.GetYaxis().SetTitle("Events")
  hMt_real.SetTitle("Black = Real, Red = Generated")
  hMt_real.SetLineColor(1)
  hMt_real.Draw("pe")
  hMt_fake.SetLineColor(2)
  hMt_fake.Draw("same pe")
  c4.Draw()
  c4.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/value_plots/hMt.png")
  c4.Close()

