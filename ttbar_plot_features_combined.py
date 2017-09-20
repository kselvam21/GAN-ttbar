#This function plots the features and other necessary values
#This function was specifically written for the ttbar datasets
#Dataset Shape: (x, 66)

import ROOT
import random

def plot_features_combined(real_dataset, fake_dataset):
  #Dataset is normalized

  #Sub-sampling real dataset
  num_samples = 1000000

  real_dataset = real_dataset[0:num_samples, :]
  fake_dataset = fake_dataset[0:num_samples, :]

  dataset_shape = real_dataset.shape
  num_rows = dataset_shape[0]
  num_cols = dataset_shape[1]

  jet_min_idx = 0
  jet_max_idx = 29
  lep_min_idx = 30
  lep_max_idx = 37

  ROOT.gStyle.SetOptStat(ROOT.kFALSE);

  #Initializing histograms
  hJetPt1_real = ROOT.TH1D("hJetPt1_real", "", 100, 0.0, 1.0)
  hJetPt2_real = ROOT.TH1D("hJetPt2_real", "", 100, 0.0, 1.0)
  hJetPt3_real = ROOT.TH1D("hJetPt3_real", "", 100, 0.0, 1.0)
  hJetPt4_real = ROOT.TH1D("hJetPt4_real", "", 100, 0.0, 1.0)
  #hJetPt_real = ROOT.TH1D("hJetPt_real", "", 100, 0.0, 1.0)
  hJetEta1_real = ROOT.TH1D("hJetEta1_real", "", 100, 0.0, 1.0)
  hJetEta2_real = ROOT.TH1D("hJetEta2_real", "", 100, 0.0, 1.0)
  hJetEta3_real = ROOT.TH1D("hJetEta3_real", "", 100, 0.0, 1.0)
  hJetEta4_real = ROOT.TH1D("hJetEta4_real", "", 100, 0.0, 1.0)
  #hJetEta_real = ROOT.TH1D("hJetEta_real", "", 100, 0.0, 1.0)
  hJetPhi1_real = ROOT.TH1D("hJetPhi1_real", "", 100, 0.0, 1.0)
  hJetPhi2_real = ROOT.TH1D("hJetPhi2_real", "", 100, 0.0, 1.0)
  hJetPhi3_real = ROOT.TH1D("hJetPhi3_real", "", 100, 0.0, 1.0)
  hJetPhi4_real = ROOT.TH1D("hJetPhi4_real", "", 100, 0.0, 1.0)
  #hJetPhi_real = ROOT.TH1D("hJetPhi_real", "", 100, 0.0, 1.0)
  hJetMass1_real = ROOT.TH1D("hJetMass1_real", "", 100, 0.0, 1.0)
  hJetMass2_real = ROOT.TH1D("hJetMass2_real", "", 100, 0.0, 1.0)
  hJetMass3_real = ROOT.TH1D("hJetMass3_real", "", 100, 0.0, 1.0)
  hJetMass4_real = ROOT.TH1D("hJetMass4_real", "", 100, 0.0, 1.0)
  #hJetMass_real = ROOT.TH1D("hJetMass_real", "", 100, 0.0, 1.0)
  hLepCharge_real = ROOT.TH1D("hLepCharge_real", "", 100, 0.0, 1.1)
  hLepPt_real = ROOT.TH1D("hLepPt_real", "", 100, 0.0, 1.0)
  hLepEta_real = ROOT.TH1D("hLepEta_real", "", 100, 0.0, 1.0)
  hLepPhi_real = ROOT.TH1D("hLepPhi_real", "", 100, 0.0, 1.0)
  hLepIsoPhoton_real = ROOT.TH1D("hLepIsoPhoton_real", "", 100, 0.0, 0.5)
  hLepIsoChHad_real = ROOT.TH1D("hLepIsoChHad_real", "", 100, 0.0, 0.5)
  hLepIsoNeuHad_real = ROOT.TH1D("hLepIsoNeuHad_real", "", 100, 0.0, 0.5)

  hJetPt1_fake = ROOT.TH1D("hJetPt1_fake", "", 100, 0.0, 1.0)
  hJetPt2_fake = ROOT.TH1D("hJetPt2_fake", "", 100, 0.0, 1.0)
  hJetPt3_fake = ROOT.TH1D("hJetPt3_fake", "", 100, 0.0, 1.0)
  hJetPt4_fake = ROOT.TH1D("hJetPt4_fake", "", 100, 0.0, 1.0)
  #hJetPt_fake = ROOT.TH1D("hJetPt_fake", "", 100, 0.0, 1.0)
  hJetEta1_fake = ROOT.TH1D("hJetEta1_fake", "", 100, 0.0, 1.0)
  hJetEta2_fake = ROOT.TH1D("hJetEta2_fake", "", 100, 0.0, 1.0)
  hJetEta3_fake = ROOT.TH1D("hJetEta3_fake", "", 100, 0.0, 1.0)
  hJetEta4_fake = ROOT.TH1D("hJetEta4_fake", "", 100, 0.0, 1.0)
  #hJetEta_fake = ROOT.TH1D("hJetEta_fake", "", 100, 0.0, 1.0)
  hJetPhi1_fake = ROOT.TH1D("hJetPhi1_fake", "", 100, 0.0, 1.0)
  hJetPhi2_fake = ROOT.TH1D("hJetPhi2_fake", "", 100, 0.0, 1.0)
  hJetPhi3_fake = ROOT.TH1D("hJetPhi3_fake", "", 100, 0.0, 1.0)
  hJetPhi4_fake = ROOT.TH1D("hJetPhi4_fake", "", 100, 0.0, 1.0)
  #hJetPhi_fake = ROOT.TH1D("hJetPhi_fake", "", 100, 0.0, 1.0)
  hJetMass1_fake = ROOT.TH1D("hJetMass1_fake", "", 100, 0.0, 1.0)
  hJetMass2_fake = ROOT.TH1D("hJetMass2_fake", "", 100, 0.0, 1.0)
  hJetMass3_fake = ROOT.TH1D("hJetMass3_fake", "", 100, 0.0, 1.0)
  hJetMass4_fake = ROOT.TH1D("hJetMass4_fake", "", 100, 0.0, 1.0)
  #hJetMass_fake = ROOT.TH1D("hJetMass_fake", "", 100, 0.0, 1.0)
  hLepCharge_fake = ROOT.TH1D("hLepCharge_fake", "", 100, 0.0, 1.1)
  hLepPt_fake = ROOT.TH1D("hLepPt_fake", "", 100, 0.0, 1.0)
  hLepEta_fake = ROOT.TH1D("hLepEta_fake", "", 100, 0.0, 1.0)
  hLepPhi_fake = ROOT.TH1D("hLepPhi_fake", "", 100, 0.0, 1.0)
  hLepIsoPhoton_fake = ROOT.TH1D("hLepIsoPhoton_fake", "", 100, 0.0, 0.5)
  hLepIsoChHad_fake = ROOT.TH1D("hLepIsoChHad_fake", "", 100, 0.0, 0.5)
  hLepIsoNeuHad_fake = ROOT.TH1D("hLepIsoNeuHad_fake", "", 100, 0.0, 0.5)


  #real_dataset
  for i in range(num_rows):
    #analyzing jets (index 0-29)
    for jet_idx in range(jet_min_idx, jet_max_idx + 1):
      #JetPt
      if jet_idx%5 == 0:
        #hJetPt_real.Fill(real_dataset[i][jet_idx])
        if jet_idx == 0:
          hJetPt1_real.Fill(real_dataset[i][jet_idx])
        if jet_idx == 5:
          hJetPt2_real.Fill(real_dataset[i][jet_idx])
        if jet_idx == 10:
          hJetPt3_real.Fill(real_dataset[i][jet_idx])
        if jet_idx == 15:
          hJetPt4_real.Fill(real_dataset[i][jet_idx])

      #JetEta 
      if jet_idx%5 == 1:
        #hJetEta_real.Fill(real_dataset[i][jet_idx])
        if jet_idx == 1:
          hJetEta1_real.Fill(real_dataset[i][jet_idx])
        if jet_idx == 6:
          hJetEta2_real.Fill(real_dataset[i][jet_idx])
        if jet_idx == 11:
          hJetEta3_real.Fill(real_dataset[i][jet_idx])
        if jet_idx == 16:
          hJetEta4_real.Fill(real_dataset[i][jet_idx])

      #JetPhi
      if jet_idx%5 == 2:
        #hJetPhi_real.Fill(real_dataset[i][jet_idx])
        if jet_idx == 2:
          hJetPhi1_real.Fill(real_dataset[i][jet_idx])
        if jet_idx == 7:
          hJetPhi2_real.Fill(real_dataset[i][jet_idx])
        if jet_idx == 12:
          hJetPhi3_real.Fill(real_dataset[i][jet_idx])
        if jet_idx == 17:
          hJetPhi4_real.Fill(real_dataset[i][jet_idx])

      #JetMass
      if jet_idx%5 == 3:
        #hJetMass_real.Fill(real_dataset[i][jet_idx])
        if jet_idx == 3:
          hJetMass1_real.Fill(real_dataset[i][jet_idx])
        if jet_idx == 8:
          hJetMass2_real.Fill(real_dataset[i][jet_idx])
        if jet_idx == 13:
          hJetMass3_real.Fill(real_dataset[i][jet_idx])
        if jet_idx == 18:
          hJetMass4_real.Fill(real_dataset[i][jet_idx])

      #JetBtag
      #jet_idx%5 == 4:
      #NO NEED TO BE PLOTTED

    #analyzing leptons (index 30-37)
    for lep_idx in range(lep_min_idx, lep_max_idx + 1):
      #LepCharge
      if (lep_idx - lep_min_idx)%8 == 0:
        hLepCharge_real.Fill(real_dataset[i][lep_idx])

      #LepIsEle
      #if (lep_idx - lep_min_idx)%8 == 1:
      #NO NEED TO BE PLOTTED

      #LepPt
      if (lep_idx - lep_min_idx)%8 == 2:
        hLepPt_real.Fill(real_dataset[i][lep_idx])

      #LepEta
      if (lep_idx - lep_min_idx)%8 == 3:
        hLepEta_real.Fill(real_dataset[i][lep_idx])

      #LepPhi
      if (lep_idx - lep_min_idx)%8 == 4:
        hLepPhi_real.Fill(real_dataset[i][lep_idx])

      #LepIsoPhoton
      if (lep_idx - lep_min_idx)%8 == 5:
        hLepIsoPhoton_real.Fill(real_dataset[i][lep_idx])

      #LepIsoChHad
      if (lep_idx - lep_min_idx)%8 == 6:
        hLepIsoChHad_real.Fill(real_dataset[i][lep_idx])

      #LepIsoNeuHad
      if (lep_idx - lep_min_idx)%8 == 7:
        hLepIsoNeuHad_real.Fill(real_dataset[i][lep_idx])


  #fake_dataset
  for i in range(num_rows):
    #analyzing jets (index 0-29)
    for jet_idx in range(jet_min_idx, jet_max_idx + 1):
      #JetPt
      if jet_idx%5 == 0:
        #hJetPt_fake.Fill(fake_dataset[i][jet_idx])
        if jet_idx == 0:
          hJetPt1_fake.Fill(fake_dataset[i][jet_idx])
        if jet_idx == 5:
          hJetPt2_fake.Fill(fake_dataset[i][jet_idx])
        if jet_idx == 10:
          hJetPt3_fake.Fill(fake_dataset[i][jet_idx])
        if jet_idx == 15:
          hJetPt4_fake.Fill(fake_dataset[i][jet_idx])

      #JetEta 
      if jet_idx%5 == 1:
        #hJetEta_fake.Fill(fake_dataset[i][jet_idx])
        if jet_idx == 1:
          hJetEta1_fake.Fill(fake_dataset[i][jet_idx])
        if jet_idx == 6:
          hJetEta2_fake.Fill(fake_dataset[i][jet_idx])
        if jet_idx == 11:
          hJetEta3_fake.Fill(fake_dataset[i][jet_idx])
        if jet_idx == 16:
          hJetEta4_fake.Fill(fake_dataset[i][jet_idx])

      #JetPhi
      if jet_idx%5 == 2:
        #hJetPhi_fake.Fill(fake_dataset[i][jet_idx])
        if jet_idx == 2:
          hJetPhi1_fake.Fill(fake_dataset[i][jet_idx])
        if jet_idx == 7:
          hJetPhi2_fake.Fill(fake_dataset[i][jet_idx])
        if jet_idx == 12:
          hJetPhi3_fake.Fill(fake_dataset[i][jet_idx])
        if jet_idx == 17:
          hJetPhi4_fake.Fill(fake_dataset[i][jet_idx])

      #JetMass
      if jet_idx%5 == 3:
        #hJetMass_fake.Fill(fake_dataset[i][jet_idx])
        if jet_idx == 3:
          hJetMass1_fake.Fill(fake_dataset[i][jet_idx])
        if jet_idx == 8:
          hJetMass2_fake.Fill(fake_dataset[i][jet_idx])
        if jet_idx == 13:
          hJetMass3_fake.Fill(fake_dataset[i][jet_idx])
        if jet_idx == 18:
          hJetMass4_fake.Fill(fake_dataset[i][jet_idx])

      #JetBtag
      #jet_idx%5 == 4:
      #NO NEED TO BE PLOTTED

    #analyzing leptons (index 30-37)
    for lep_idx in range(lep_min_idx, lep_max_idx + 1):
      #LepCharge
      if (lep_idx - lep_min_idx)%8 == 0:
        hLepCharge_fake.Fill(fake_dataset[i][lep_idx])

      #LepIsEle
      #if (lep_idx - lep_min_idx)%8 == 1:
      #NO NEED TO BE PLOTTED

      #LepPt
      if (lep_idx - lep_min_idx)%8 == 2:
        hLepPt_fake.Fill(fake_dataset[i][lep_idx])

      #LepEta
      if (lep_idx - lep_min_idx)%8 == 3:
        hLepEta_fake.Fill(fake_dataset[i][lep_idx])

      #LepPhi
      if (lep_idx - lep_min_idx)%8 == 4:
        hLepPhi_fake.Fill(fake_dataset[i][lep_idx])

      #LepIsoPhoton
      if (lep_idx - lep_min_idx)%8 == 5:
        hLepIsoPhoton_fake.Fill(fake_dataset[i][lep_idx])

      #LepIsoChHad
      if (lep_idx - lep_min_idx)%8 == 6:
        hLepIsoChHad_fake.Fill(fake_dataset[i][lep_idx])

      #LepIsoNeuHad
      if (lep_idx - lep_min_idx)%8 == 7:
        hLepIsoNeuHad_fake.Fill(fake_dataset[i][lep_idx])



  #JetPt1
  c1 = ROOT.TCanvas("c1", "", 600, 600)
  c1.SetLogy()
  hJetPt1_real.GetXaxis().SetTitle("JetPt1 (Normalized)")
  hJetPt1_real.GetYaxis().SetTitle("Events")
  hJetPt1_real.SetTitle("Black = Real, Red = Generated")
  hJetPt1_real.SetLineColor(1)
  hJetPt1_real.Draw("pe")
  hJetPt1_fake.SetLineColor(2)
  hJetPt1_fake.Draw("same pe")
  
  c1.Draw()
  c1.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetPt1.png")
  c1.Close()

  #JetPt2
  c1 = ROOT.TCanvas("c1", "", 600, 600)
  c1.SetLogy()
  hJetPt2_real.GetXaxis().SetTitle("JetPt2/JetPt1 (Normalized)")
  hJetPt2_real.GetYaxis().SetTitle("Events")
  hJetPt2_real.SetTitle("Black = Real, Red = Generated")
  hJetPt2_real.SetLineColor(1)
  hJetPt2_real.Draw("pe")
  hJetPt2_fake.SetLineColor(2)
  hJetPt2_fake.Draw("same pe")
  
  c1.Draw()
  c1.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetPt2.png")
  c1.Close()

  #JetPt3
  c1 = ROOT.TCanvas("c1", "", 600, 600)
  c1.SetLogy()
  hJetPt3_real.GetXaxis().SetTitle("JetPt3/JetPt2 (Normalized)")
  hJetPt3_real.GetYaxis().SetTitle("Events")
  hJetPt3_real.SetTitle("Black = Real, Red = Generated")
  hJetPt3_real.SetLineColor(1)
  hJetPt3_real.Draw("pe")
  hJetPt3_fake.SetLineColor(2)
  hJetPt3_fake.Draw("same pe")
  
  c1.Draw()
  c1.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetPt3.png")
  c1.Close()

  #JetPt4
  c1 = ROOT.TCanvas("c1", "", 600, 600)
  c1.SetLogy()
  hJetPt4_real.GetXaxis().SetTitle("JetPt4/JetPt3 (Normalized)")
  hJetPt4_real.GetYaxis().SetTitle("Events")
  hJetPt4_real.SetTitle("Black = Real, Red = Generated")
  hJetPt4_real.SetLineColor(1)
  hJetPt4_real.Draw("pe")
  hJetPt4_fake.SetLineColor(2)
  hJetPt4_fake.Draw("same pe")
  
  c1.Draw()
  c1.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetPt4.png")
  c1.Close()

  #JetPt
  #c1 = ROOT.TCanvas("c1", "", 600, 600)
  #c1.SetLogy()
  #hJetPt_real.GetXaxis().SetTitle("JetPt1 + JetPt ratios (Normalized)")
  #hJetPt_real.GetYaxis().SetTitle("Events")
  #hJetPt_real.SetTitle("Black = Real, Red = Generated")
  #hJetPt_real.SetLineColor(1)
  #hJetPt_real.Draw("pe")
  #hJetPt_fake.SetLineColor(2)
  #hJetPt_fake.Draw("same pe")
  
  #c1.Draw()
  #c1.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetPt.png")
  #c1.Close()


  #JetEta1
  c2 = ROOT.TCanvas("c2", "", 600, 600)
  c2.SetLogy()
  hJetEta1_real.GetXaxis().SetTitle("JetEta1 (Normalized)")
  hJetEta1_real.GetYaxis().SetTitle("Events")
  hJetEta1_real.SetTitle("Black = Real, Red = Generated")
  hJetEta1_real.SetLineColor(1)
  hJetEta1_real.Draw("pe")
  hJetEta1_fake.SetLineColor(2)
  hJetEta1_fake.Draw("same pe")

  c2.Draw()
  c2.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetEta1.png")
  c2.Close()

  #JetEta2
  c2 = ROOT.TCanvas("c2", "", 600, 600)
  c2.SetLogy()
  hJetEta2_real.GetXaxis().SetTitle("JetEta2 (Normalized)")
  hJetEta2_real.GetYaxis().SetTitle("Events")
  hJetEta2_real.SetTitle("Black = Real, Red = Generated")
  hJetEta2_real.SetLineColor(1)
  hJetEta2_real.Draw("pe")
  hJetEta2_fake.SetLineColor(2)
  hJetEta2_fake.Draw("same pe")

  c2.Draw()
  c2.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetEta2.png")
  c2.Close()

  #JetEta3
  c2 = ROOT.TCanvas("c2", "", 600, 600)
  c2.SetLogy()
  hJetEta3_real.GetXaxis().SetTitle("JetEta3 (Normalized)")
  hJetEta3_real.GetYaxis().SetTitle("Events")
  hJetEta3_real.SetTitle("Black = Real, Red = Generated")
  hJetEta3_real.SetLineColor(1)
  hJetEta3_real.Draw("pe")
  hJetEta3_fake.SetLineColor(2)
  hJetEta3_fake.Draw("same pe")

  c2.Draw()
  c2.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetEta3.png")
  c2.Close()

  #JetEta4
  c2 = ROOT.TCanvas("c2", "", 600, 600)
  c2.SetLogy()
  hJetEta4_real.GetXaxis().SetTitle("JetEta4 (Normalized)")
  hJetEta4_real.GetYaxis().SetTitle("Events")
  hJetEta4_real.SetTitle("Black = Real, Red = Generated")
  hJetEta4_real.SetLineColor(1)
  hJetEta4_real.Draw("pe")
  hJetEta4_fake.SetLineColor(2)
  hJetEta4_fake.Draw("same pe")

  c2.Draw()
  c2.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetEta4.png")
  c2.Close()

  #JetEta
  #c2 = ROOT.TCanvas("c2", "", 600, 600)
  #c2.SetLogy()
  #hJetEta_real.GetXaxis().SetTitle("JetEta (Normalized)")
  #hJetEta_real.GetYaxis().SetTitle("Events")
  #hJetEta_real.SetTitle("Black = Real, Red = Generated")
  #hJetEta_real.SetLineColor(1)
  #hJetEta_real.Draw("pe")
  #hJetEta_fake.SetLineColor(2)
  #hJetEta_fake.Draw("same pe")

  #c2.Draw()
  #c2.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetEta.png")
  #c2.Close()


  #JetPhi1
  c3 = ROOT.TCanvas("c3", "", 600, 600)
  c3.SetLogy()
  hJetPhi1_real.GetXaxis().SetTitle("JetPhi1 (Normalized)")
  hJetPhi1_real.GetYaxis().SetTitle("Events")
  hJetPhi1_real.SetTitle("Black = Real, Red = Generated")
  hJetPhi1_real.SetLineColor(1)
  hJetPhi1_real.Draw("pe")
  hJetPhi1_fake.SetLineColor(2)
  hJetPhi1_fake.Draw("same pe")

  c3.Draw()
  c3.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetPhi1.png")
  c3.Close()

  #JetPhi2
  c3 = ROOT.TCanvas("c3", "", 600, 600)
  c3.SetLogy()
  hJetPhi2_real.GetXaxis().SetTitle("JetPhi2 (Normalized)")
  hJetPhi2_real.GetYaxis().SetTitle("Events")
  hJetPhi2_real.SetTitle("Black = Real, Red = Generated")
  hJetPhi2_real.SetLineColor(1)
  hJetPhi2_real.Draw("pe")
  hJetPhi2_fake.SetLineColor(2)
  hJetPhi2_fake.Draw("same pe")

  c3.Draw()
  c3.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetPhi2.png")
  c3.Close()

  #JetPhi3
  c3 = ROOT.TCanvas("c3", "", 600, 600)
  c3.SetLogy()
  hJetPhi3_real.GetXaxis().SetTitle("JetPhi3 (Normalized)")
  hJetPhi3_real.GetYaxis().SetTitle("Events")
  hJetPhi3_real.SetTitle("Black = Real, Red = Generated")
  hJetPhi3_real.SetLineColor(1)
  hJetPhi3_real.Draw("pe")
  hJetPhi3_fake.SetLineColor(2)
  hJetPhi3_fake.Draw("same pe")

  c3.Draw()
  c3.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetPhi3.png")
  c3.Close()

  #JetPhi4
  c3 = ROOT.TCanvas("c3", "", 600, 600)
  c3.SetLogy()
  hJetPhi4_real.GetXaxis().SetTitle("JetPhi4 (Normalized)")
  hJetPhi4_real.GetYaxis().SetTitle("Events")
  hJetPhi4_real.SetTitle("Black = Real, Red = Generated")
  hJetPhi4_real.SetLineColor(1)
  hJetPhi4_real.Draw("pe")
  hJetPhi4_fake.SetLineColor(2)
  hJetPhi4_fake.Draw("same pe")

  c3.Draw()
  c3.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetPhi4.png")
  c3.Close()

  #JetPhi
  #c3 = ROOT.TCanvas("c3", "", 600, 600)
  #c3.SetLogy()
  #hJetPhi_real.GetXaxis().SetTitle("JetPhi (Normalized)")
  #hJetPhi_real.GetYaxis().SetTitle("Events")
  #hJetPhi_real.SetTitle("Black = Real, Red = Generated")
  #hJetPhi_real.SetLineColor(1)
  #hJetPhi_real.Draw("pe")
  #hJetPhi_fake.SetLineColor(2)
  #hJetPhi_fake.Draw("same pe")

  #c3.Draw()
  #c3.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetPhi.png")
  #c3.Close()


  #JetMass1
  c4 = ROOT.TCanvas("c4", "", 600, 600)
  c4.SetLogy()
  hJetMass1_real.GetXaxis().SetTitle("JetMass1 (Normalized)")
  hJetMass1_real.GetYaxis().SetTitle("Events")
  hJetMass1_real.SetTitle("Black = Real, Red = Generated")
  hJetMass1_real.SetLineColor(1)
  hJetMass1_real.Draw("pe")
  hJetMass1_fake.SetLineColor(2)
  hJetMass1_fake.Draw("same pe")

  c4.Draw()
  c4.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetMass1.png")
  c4.Close()

  #JetMass2
  c4 = ROOT.TCanvas("c4", "", 600, 600)
  c4.SetLogy()
  hJetMass2_real.GetXaxis().SetTitle("JetMass2 (Normalized)")
  hJetMass2_real.GetYaxis().SetTitle("Events")
  hJetMass2_real.SetTitle("Black = Real, Red = Generated")
  hJetMass2_real.SetLineColor(1)
  hJetMass2_real.Draw("pe")
  hJetMass2_fake.SetLineColor(2)
  hJetMass2_fake.Draw("same pe")

  c4.Draw()
  c4.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetMass2.png")
  c4.Close()

  #JetMass3
  c4 = ROOT.TCanvas("c4", "", 600, 600)
  c4.SetLogy()
  hJetMass3_real.GetXaxis().SetTitle("JetMass3 (Normalized)")
  hJetMass3_real.GetYaxis().SetTitle("Events")
  hJetMass3_real.SetTitle("Black = Real, Red = Generated")
  hJetMass3_real.SetLineColor(1)
  hJetMass3_real.Draw("pe")
  hJetMass3_fake.SetLineColor(2)
  hJetMass3_fake.Draw("same pe")

  c4.Draw()
  c4.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetMass3.png")
  c4.Close()

  #JetMass4
  c4 = ROOT.TCanvas("c4", "", 600, 600)
  c4.SetLogy()
  hJetMass4_real.GetXaxis().SetTitle("JetMass4 (Normalized)")
  hJetMass4_real.GetYaxis().SetTitle("Events")
  hJetMass4_real.SetTitle("Black = Real, Red = Generated")
  hJetMass4_real.SetLineColor(1)
  hJetMass4_real.Draw("pe")
  hJetMass4_fake.SetLineColor(2)
  hJetMass4_fake.Draw("same pe")

  c4.Draw()
  c4.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetMass4.png")
  c4.Close()

  #JetMass
  #c4 = ROOT.TCanvas("c4", "", 600, 600)
  #c4.SetLogy()
  #hJetMass_real.GetXaxis().SetTitle("JetMass (Normalized)")
  #hJetMass_real.GetYaxis().SetTitle("Events")
  #hJetMass_real.SetTitle("Black = Real, Red = Generated")
  #hJetMass_real.SetLineColor(1)
  #hJetMass_real.Draw("pe")
  #hJetMass_fake.SetLineColor(2)
  #hJetMass_fake.Draw("same pe")

  #c4.Draw()
  #c4.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hJetMass.png")
  #c4.Close()


  #LepCharge
  c5 = ROOT.TCanvas("c5", "", 600, 600)
  c5.SetLogy()
  hLepCharge_real.GetXaxis().SetTitle("LepCharge (Normalized)")
  hLepCharge_real.GetYaxis().SetTitle("Events")
  hLepCharge_real.SetTitle("Black = Real, Red = Generated")
  hLepCharge_real.SetLineColor(1)
  hLepCharge_real.Draw("pe")
  hLepCharge_fake.SetLineColor(2)
  hLepCharge_fake.Draw("same pe")

  c5.Draw()
  c5.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hLepCharge.png")
  c5.Close()


  #LepPt
  c6 = ROOT.TCanvas("c6", "", 600, 600)
  c6.SetLogy()
  hLepPt_real.GetXaxis().SetTitle("LepPt (Normalized)")
  hLepPt_real.GetYaxis().SetTitle("Events")
  hLepPt_real.SetTitle("Black = Real, Red = Generated")
  hLepPt_real.SetLineColor(1)
  hLepPt_real.Draw("pe")
  hLepPt_fake.SetLineColor(2)
  hLepPt_fake.Draw("same pe")

  c6.Draw()
  c6.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hLepPt.png")
  c6.Close()
  
  
  #LepEta
  c7 = ROOT.TCanvas("c7", "", 600, 600)
  c7.SetLogy()
  hLepEta_real.GetXaxis().SetTitle("LepEta (Normalized)")
  hLepEta_real.GetYaxis().SetTitle("Events")
  hLepEta_real.SetTitle("Black = Real, Red = Generated")
  hLepEta_real.SetLineColor(1)
  hLepEta_real.Draw("pe")
  hLepEta_fake.SetLineColor(2)
  hLepEta_fake.Draw("same pe")

  c7.Draw()
  c7.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hLepEta.png")
  c7.Close()
  
  
  #LepPhi
  c8 = ROOT.TCanvas("c8", "", 600, 600)
  c8.SetLogy()
  hLepPhi_real.GetXaxis().SetTitle("LepPhi (Normalized)")
  hLepPhi_real.GetYaxis().SetTitle("Events")
  hLepPhi_real.SetTitle("Black = Real, Red = Generated")
  hLepPhi_real.SetLineColor(1)
  hLepPhi_real.Draw("pe")
  hLepPhi_fake.SetLineColor(2)
  hLepPhi_fake.Draw("same pe")

  c8.Draw()
  c8.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hLepPhi.png")
  c8.Close()
  
  
  #LepIsoPhoton
  c9 = ROOT.TCanvas("c9", "", 600, 600)
  c9.SetLogy()
  hLepIsoPhoton_real.GetXaxis().SetTitle("LepIsoPhoton (Normalized)")
  hLepIsoPhoton_real.GetYaxis().SetTitle("Events")
  hLepIsoPhoton_real.SetTitle("Black = Real, Red = Generated")
  hLepIsoPhoton_real.SetLineColor(1)
  hLepIsoPhoton_real.Draw("pe")
  hLepIsoPhoton_fake.SetLineColor(2)
  hLepIsoPhoton_fake.Draw("same pe")

  c9.Draw()
  c9.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hLepIsoPhoton.png")
  c9.Close()
  
  
  #LepIsoChHad
  c10 = ROOT.TCanvas("c11", "", 600, 600)
  c10.SetLogy()
  hLepIsoChHad_real.GetXaxis().SetTitle("LepIsoChHad (Normalized)")
  hLepIsoChHad_real.GetYaxis().SetTitle("Events")
  hLepIsoChHad_real.SetTitle("Black = Real, Red = Generated")
  hLepIsoChHad_real.SetLineColor(1)
  hLepIsoChHad_real.Draw("pe")
  hLepIsoChHad_fake.SetLineColor(2)
  hLepIsoChHad_fake.Draw("same pe")

  c10.Draw()
  c10.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hLepIsoChHad.png")
  c10.Close()
  
  
  #LepIsoNeuHad
  c11 = ROOT.TCanvas("c11", "", 600, 600)
  c11.SetLogy()
  hLepIsoNeuHad_real.GetXaxis().SetTitle("LepIsoNeuHad (Normalized)")
  hLepIsoNeuHad_real.GetYaxis().SetTitle("Events")
  hLepIsoNeuHad_real.SetTitle("Black = Real, Red = Generated")
  hLepIsoNeuHad_real.SetLineColor(1)
  hLepIsoNeuHad_real.Draw("pe")
  hLepIsoNeuHad_fake.SetLineColor(2)
  hLepIsoNeuHad_fake.Draw("same pe")

  c11.Draw()
  c11.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/hLepIsoNeuHad.png")
  c11.Close()
