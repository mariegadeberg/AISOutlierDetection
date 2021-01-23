# Anomaly Detection for AIS Data Using Deep Neural Networks for Trajectory Prediction

This repository contains all code used to generate the findings of the master thesis "Anomaly Detection for AIS Data Using Deep Neural Networks for Trajectory Prediction" produced to obtain a masters degree in Mathematical Modelling and Computation from the Technical University of Denmark by Marie Normann Gadeberg. 

The project uses two models; a Convolutional Variational Autoencoder and a Variational Recurrent Neural Network available in `cvae.py` and `vrnn.py` respectively. 

## File Summary
* Job Scripts: Contains job scripts used to train models on DTU HPC LSF 10 cluster
* Config.py: configuration file containing hyperparameters
* PlotResults.py: generates plots showing training stats
* Preprocess.py: file to run to preprocess data
* cvae.py: contains the CVAE model
* get_log_px_train: used to get log_px og training set after finished training
* outlier_detect.py: generate all plots used in CVAE result section
* requirements.txt: requirement file containing package versions used in project
* sample.py: generate reconstructed paths using the VRNN model
* train_cvae.py: file to run when training CVAE model
* train_vrnn.py: file to fun when training VRNN model
* vrnn.py: contains VRNN model
* utils_outliers: contains all helper functions for outlier_detect.py
* utils_preprocess: contains all helper functions for Prepocess.py
* utils_train: contains all helper functions used for both train files 

## Usage
To train CVAE model for 10 epoch run:

`python train_cvae.py --num_epoch 10`

Here it is assumed that data lies in a folder one level up called "data", and the output will be saved in a folder one level up called "models". 

To plot training stats of newly trained model run:

```
python PlotResults.py \
    --output_file "./models/bh10_noBN/output_10bh.txt" \
    --save_path "Figures/" \
    --model_type "CVAE" \
    --BN "False" \
    --gradient_path "./models/bh10_noBN/grad_10_bh.pcl" 
```
where:
- `output_file` is the directory to the file called "output" generated from the training file
- `save_path` shows directory to save figures
- `model_type` indicates which of the two available models has been used
- `BN` indicates whether batchnormalization of the approximate posterior has been used
- `gradient_path` is the directory to the "grad" file generated form the training file

To train VRNN for 10 epochs using batchsize 32, KL annealing, a batchnormalized approximate posterior and gradient clipping run:

```
python train_vrnn.py \
    --num_epoch 10 \
    --batchsize 32 \
    --kl_start 0.1 \
    --warm_up 10 \
    --gamma 0.6 \
    --bn_switch "True" \
    --clip_grad "True 
```
where:
- `kl_start` is the increase per epoch of the weight on the KL divergence
- `warm_up` is how many epochs before weight on KL divergence reaches 1
- `gamma` indicates the fixed mean imposed on the batch normalized approximate posterior
- `bn_switch` indicates that batch normalization of the approximate posterior should be used
- `clip_grad` indicates that gradient clipping should be used

For all files run `python filename.py --help` for full list of arguments available. 

## References
The implementation of the Variational Recurrent Neural Network tries to replicate the model proposed in "GeoTrackNet-A Maritime Anomaly Detector using Probabilistic Neural Network Representation of AIS Tracks and A Contrario Detection", Nguyen et al. 2019, https://arxiv.org/abs/1912.00682.

The general implementation of the Variational Autoencoder was done with inspiration from: https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch/blob/master/7_Unsupervised/7.2-EXE-variational-autoencoder.ipynb

The implementation of Mutual Information and Active Units was done with inspiration from: https://github.com/jxhe/vae-lagging-encoder
