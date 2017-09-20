#This function plots the features and other necessary values
#This function was specifically written for the ttbar datasets
#Dataset Shape: (x, 66)

import ROOT
import random

def plot_features(dataset):
  #Dataset is normalized
  #dataset_type: True if real dataset
  #              False if fake dataset

  #Sub-sampling real dataset
  #if dataset_type == True:
  #  dataset = dataset[0:100000, :]

  dataset_shape = dataset.shape
  num_rows = dataset_shape[0]
  num_cols = dataset_shape[1]

  jet_min_idx = 0
  jet_max_idx = 49
  lep_min_idx = 50
  lep_max_idx = 65

  #Initializing histograms
  hJetPt = ROOT.TH1D("hJetPt", "", 100, 0.0, 120)
  hJetEta = ROOT.TH1D("hJetEta", "", 100, -6.0, 6.0)
  hJetPhi = ROOT.TH1D("hJetPhi", "", 100, -4.0, 4.0)
  hJetMass = ROOT.TH1D("hJetMass", "", 100, 0.0, 60)
  hLepCharge = ROOT.TH1D("hLepCharge", "", 100, -1.1, 1.1)
  hLepPt = ROOT.TH1D("hLepPt", "", 100, 0.0, 180)
  hLepEta = ROOT.TH1D("hLepEta", "", 100, -6.0, 6.0)
  hLepPhi = ROOT.TH1D("hLepPhi", "", 100, -4.0, 4.0)
  hLepIsoPhoton = ROOT.TH1D("hLepIsoPhoton", "", 100, 0.0, 2.5)
  hLepIsoChHad = ROOT.TH1D("hLepIsoChHad", "", 100, 0.0, 4)
  hLepIsoNeuHad = ROOT.TH1D("hLepIsoNeuHad", "", 100, 0.0, 2.5)


  for i in range(num_rows):
    #analyzing jets (index 0-49)
    for jet_idx in range(jet_min_idx, jet_max_idx + 1):
      #JetPt
      if jet_idx%5 == 0:
        hJetPt.Fill(dataset[i][jet_idx])

      #JetEta 
      if jet_idx%5 == 1:
        hJetEta.Fill(dataset[i][jet_idx])

      #JetPhi
      if jet_idx%5 == 2:
        hJetPhi.Fill(dataset[i][jet_idx])

      #JetMass
      if jet_idx%5 == 3:
        hJetMass.Fill(dataset[i][jet_idx])

      #JetBtag
      #jet_idx%5 == 4:
      #NO NEED TO BE PLOTTED

    #analyzing leptons (index 50-65)
    for lep_idx in range(lep_min_idx, lep_max_idx + 1):
      #LepCharge
      if (lep_idx - lep_min_idx)%8 == 0:
        hLepCharge.Fill(dataset[i][lep_idx])

      #LepIsEle
      #if (lep_idx - lep_min_idx)%8 == 1:
      #NO NEED TO BE PLOTTED

      #LepPt
      if (lep_idx - lep_min_idx)%8 == 2:
        hLepPt.Fill(dataset[i][lep_idx])

      #LepEta
      if (lep_idx - lep_min_idx)%8 == 3:
        hLepEta.Fill(dataset[i][lep_idx])

      #LepPhi
      if (lep_idx - lep_min_idx)%8 == 4:
        hLepPhi.Fill(dataset[i][lep_idx])

      #LepIsoPhoton
      if (lep_idx - lep_min_idx)%8 == 5:
        hLepIsoPhoton.Fill(dataset[i][lep_idx])

      #LepIsoChHad
      if (lep_idx - lep_min_idx)%8 == 6:
        hLepIsoChHad.Fill(dataset[i][lep_idx])

      #LepIsoNeuHad
      if (lep_idx - lep_min_idx)%8 == 7:
        hLepIsoNeuHad.Fill(dataset[i][lep_idx])


  #JetPt
  c1 = ROOT.TCanvas("c1", "", 600, 600)
  #c1.SetLogy()
  hJetPt.GetXaxis().SetTitle("JetPt")
  hJetPt.GetYaxis().SetTitle("Events")
  hJetPt.Draw("pe")
  c1.Draw()
  c1.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/dataset_plots/hJetPt.png")

  #if dataset_type:
  #  c1.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/real/hJetPt_real.png")
  #else:
  #  c1.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/fake/hJetPt_fake.png")
  c1.Close()

  #outFile1 = ROOT.TFile.Open("/afs/cern.ch/user/k/kselvam/cernbox/TOPGEN_rootfiles/hJetPt.root", "RECREATE")
  #hJetPt.Write()
  #outFile1.Close() 


  #JetEta
  c2 = ROOT.TCanvas("c2", "", 600, 600)
  #c2.SetLogy()
  hJetEta.GetXaxis().SetTitle("JetEta")
  hJetEta.GetYaxis().SetTitle("Events")
  hJetEta.Draw("pe")
  c2.Draw()
  c2.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/dataset_plots/hJetEta.png")

  #if dataset_type:
  #  c2.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/real/hJetEta_real.png")
  #else:
  #  c2.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/fake/hJetEta_fake.png")
  c2.Close()

  #outFile2 = ROOT.TFile.Open("/afs/cern.ch/user/k/kselvam/cernbox/TOPGEN_rootfiles/hJetEta.root", "RECREATE")
  #hJetEta.Write()
  #outFile2.Close()


  ##JetPhi
  c3 = ROOT.TCanvas("c3", "", 600, 600)
  #c3.SetLogy()
  hJetPhi.GetXaxis().SetTitle("JetPhi")
  hJetPhi.GetYaxis().SetTitle("Events")
  hJetPhi.Draw("pe")
  c3.Draw()
  c3.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/dataset_plots/hJetPhi.png")

  #if dataset_type:
  #  c3.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/real/hJetPhi_real.png")
  #else:
  #  c3.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/fake/hJetPhi_fake.png")
  c3.Close()

  #outFile3 = ROOT.TFile.Open("/afs/cern.ch/user/k/kselvam/cernbox/TOPGEN_rootfiles/hJetPhi.root", "RECREATE")
  #hJetPhi.Write()
  #outFile3.Close()


  #JetMass
  c4 = ROOT.TCanvas("c4", "", 600, 600)
  #c4.SetLogy()
  hJetMass.GetXaxis().SetTitle("JetMass")
  hJetMass.GetYaxis().SetTitle("Events")
  hJetMass.Draw("pe")
  c4.Draw()
  c4.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/dataset_plots/hJetMass.png")

  #if dataset_type:
  #  c4.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/real/hJetMass_real.png")
  #else:
  #  c4.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/fake/hJetMass_fake.png")
  c4.Close()

  #outFile4 = ROOT.TFile.Open("/afs/cern.ch/user/k/kselvam/cernbox/TOPGEN_rootfiles/hJetMass.root", "RECREATE")
  #hJetMass.Write()
  #outFile4.Close()


  #LepCharge
  c5 = ROOT.TCanvas("c5", "", 600, 600)
  #c5.SetLogy()
  hLepCharge.GetXaxis().SetTitle("LepCharge")
  hLepCharge.GetYaxis().SetTitle("Events")
  hLepCharge.Draw("pe")
  c5.Draw()
  c5.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/dataset_plots/hLepCharge.png")

  #if dataset_type:
  #  c5.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/real/hLepCharge_real.png")
  #else:
  #  c5.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/fake/hLepCharge_fake.png")
  c5.Close()

  #outFile5 = ROOT.TFile.Open("/afs/cern.ch/user/k/kselvam/cernbox/TOPGEN_rootfiles/hLepCharge.root", "RECREATE")
  #hLepCharge.Write()
  #outFile5.Close()


  #LepPt
  c6 = ROOT.TCanvas("c6", "", 600, 600)
  #c6.SetLogy()
  hLepPt.GetXaxis().SetTitle("LepPt")
  hLepPt.GetYaxis().SetTitle("Events")
  hLepPt.Draw("pe")
  c6.Draw()
  c6.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/dataset_plots/hLepPt.png")

  #if dataset_type:
  #  c6.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/real/hLepPt_real.png")
  #else:
  #  c6.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/fake/hLepPt_fake.png")
  c6.Close()

  #outFile6 = ROOT.TFile.Open("/afs/cern.ch/user/k/kselvam/cernbox/TOPGEN_rootfiles/hLepPt.root", "RECREATE")
  #hLepPt.Write()
  #outFile6.Close()
  
  
  #LepEta
  c7 = ROOT.TCanvas("c7", "", 600, 600)
  #c7.SetLogy()
  hLepEta.GetXaxis().SetTitle("LepEta")
  hLepEta.GetYaxis().SetTitle("Events")
  hLepEta.Draw("pe")
  c7.Draw()
  c7.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/dataset_plots/hLepEta.png")

  #if dataset_type:
  #  c7.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/real/hLepEta_real.png")
  #else:
  #  c7.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/fake/hLepEta_fake.png")
  c7.Close()

  #outFile7 = ROOT.TFile.Open("/afs/cern.ch/user/k/kselvam/cernbox/TOPGEN_rootfiles/hLepEta.root", "RECREATE")
  #hLepEta.Write()
  #outFile7.Close()
  
  
  #LepPhi
  c8 = ROOT.TCanvas("c8", "", 600, 600)
  #c8.SetLogy()
  hLepPhi.GetXaxis().SetTitle("LepPhi")
  hLepPhi.GetYaxis().SetTitle("Events")
  hLepPhi.Draw("pe")
  c8.Draw()
  c8.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/dataset_plots/hLepPhi.png")

  #if dataset_type:
  #  c8.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/real/hLepPhi_real.png")
  #else:
  #  c8.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/fake/hLepPhi_fake.png")
  c8.Close()

  #outFile8 = ROOT.TFile.Open("/afs/cern.ch/user/k/kselvam/cernbox/TOPGEN_rootfiles/hLepPhi.root", "RECREATE")
  #hLepPhi.Write()
  #outFile8.Close()
  
  
  #LepIsoPhoton
  c9 = ROOT.TCanvas("c9", "", 600, 600)
  #c9.SetLogy()
  hLepIsoPhoton.GetXaxis().SetTitle("LepIsoPhoton")
  hLepIsoPhoton.GetYaxis().SetTitle("Events")
  hLepIsoPhoton.Draw("pe")
  c9.Draw()
  c9.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/dataset_plots/hLepIsoPhoton.png")

  #if dataset_type:
  #  c9.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/real/hLepIsoPhoton_real.png")
  #else:
  #  c9.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/fake/hLepIsoPhoton_fake.png")
  c9.Close()

  #outFile9 = ROOT.TFile.Open("/afs/cern.ch/user/k/kselvam/cernbox/TOPGEN_rootfiles/hLepIsoPhoton.root", "RECREATE")
  #hLepIsoPhoton.Write()
  #outFile9.Close()
  
  
  #LepIsoChHad
  c10 = ROOT.TCanvas("c11", "", 600, 600)
  #c10.SetLogy()
  hLepIsoChHad.GetXaxis().SetTitle("LepIsoChHad")
  hLepIsoChHad.GetYaxis().SetTitle("Events")
  hLepIsoChHad.Draw("pe")
  c10.Draw()
  c10.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/dataset_plots/hLepIsoChHad.png")

  #if dataset_type:
  #  c10.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/real/hLepIsoChHad_real.png")
  #else:
  #  c10.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/fake/hLepIsoChHad_fake.png")
  c10.Close()

  #outFile10 = ROOT.TFile.Open("/afs/cern.ch/user/k/kselvam/cernbox/TOPGEN_rootfiles/hLepIsoChHad.root", "RECREATE")
  #hLepIsoChHad.Write()
  #outFile10.Close()
  
  
  #LepIsoNeuHad
  c11 = ROOT.TCanvas("c11", "", 600, 600)
  #c11.SetLogy()
  hLepIsoNeuHad.GetXaxis().SetTitle("LepIsoNeuHad")
  hLepIsoNeuHad.GetYaxis().SetTitle("Events")
  hLepIsoNeuHad.Draw("pe")
  c11.Draw()
  c11.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/dataset_plots/hLepIsoNeuHad.png")

  #if dataset_type:
  #  c11.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/real/hLepIsoNeuHad_real.png")
  #else:
  #  c11.SaveAs("/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/feature_plots/fake/hLepIsoNeuHad_fake.png")
  c11.Close()

  #outFile11 = ROOT.TFile.Open("/afs/cern.ch/user/k/kselvam/cernbox/TOPGEN_rootfiles/hLepIsoNeuHad.root", "RECREATE")
  #hLepIsoNeuHad.Write()
  #outFile11.Close()
