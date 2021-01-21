# Anomaly Detection for AIS Data Using Deep Neural Networks for Trajectory Prediction

This repository contains all code used to generate the findings of the master thesis "Anomaly Detection for AIS Data Using Deep Neural Networks for Trajectory Prediction" produced to obtain a masters degree in Mathematical Modelling and Computation from the Technical University of Denmark by Marie Normann Gadeberg. 

The project uses two models; a Convolutional Variational Autoencoder and a Variational Recurrent Neural Network. 

The implementation of the Variational Recurrent Neural Network tries to replicate the model proposed in "GeoTrackNet-A Maritime Anomaly Detector using Probabilistic Neural Network Representation of AIS Tracks and A Contrario Detection", Nguyen et al. 2019, https://arxiv.org/abs/1912.00682.

## Folder Summary
* Job Scripts: Contains job scripts used to train models on DTU HPC LSF 10 cluster
* scripts_main: Contains all main scripts
  - Config.py : configuration file containing hyperparameters
  - Preprocess.py : file to run to preprocess data
  - cvae.py : contains the CVAE model
  - get_log_px_train : used to get log_px og training set after finished training
  - train_cvae.py : file to run when training CVAE model
  - train_vrnn.py : file to fun when training VRNN model
  - vrnn.py : contains VRNN model
* scripts_plots: Contains all scripts used to generate plot for report
  - PlotResults.py : generates plots showing training stats
  - outlier_detect.py : generate all plots used in CVAE result section
  - sample.py : generate reconstructed paths using the VRNN model
* utils: contains all helper functions used in main scripts

## Usage
To train CVAE model for 30 epoch run:

´python train_cvae.py --num_epoch 30´

Here it is assumed that data lies in a folder one level up called "data", and the output will be saved in a folder one level up called "models". 
