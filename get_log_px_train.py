from utils.utils_outliers import *
import torch
import pickle
from cnnvae import CVAE
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", default="./CVAE/cvae_bh30_epochs.pt", help="path to model")
parser.add_argument("--train_ds", default="./data/train_bh_.pcl", help="path to training data")
parser.add_argument("--out_path", default="./CVAE/output_log_px.pcl")

args = parser.parse_args()

latent_features = Config.latent_shape
model = CVAE(latent_features)

state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)


with open(args.train_ds, "rb") as file:
    dataset = pickle.load(file)

out = get_log_px_train(dataset, model)

with open(args.out_path, "wb") as file:
    pickle.dump(out, file)


with open("/Volumes/MNG/outlier_outputs/output_log_px.pcl", "rb") as file:
    out = pickle.load(file)
