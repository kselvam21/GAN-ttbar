#Contains functions to plot GAN losses

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def plot_loss(losses):
  plt.figure(figsize=(10,8))
  plt.plot(losses["d"], label='discriminative loss')
  plt.plot(losses["g"], label='generative loss')
  plt.title('GAN loss')
  plt.ylabel('loss')
  plt.xlabel('number of epochs')
  plt.legend()
  plt.savefig('/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/loss_plots/loss_trial9_20062017.png')
  plt.close()


def plot_dloss(losses):
  plt.figure(figsize=(10,8))
  plt.plot(losses["d"], label='discriminative loss')
  plt.title('GAN: Discriminative loss')
  plt.ylabel('loss')
  plt.xlabel('number of epochs')
  plt.savefig('/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/loss_plots/dloss_trial9_20062017.png')
  plt.close()


def plot_gloss(losses):
  plt.figure(figsize=(10,8))
  plt.plot(losses["g"], label='generative loss')
  plt.title('GAN: Generative loss')
  plt.ylabel('loss')
  plt.xlabel('number of epochs')
  plt.savefig('/afs/cern.ch/user/k/kselvam/cernbox/SYNC/GAN/loss_plots/gloss_trial9_20062017.png')
  plt.close()


