from model import VRNN
import torch
from torch.utils.data import DataLoader
from utils_preprocess import AISDataset
from Config import *
from utils_preprocess import PadCollate
import matplotlib.pyplot as plt
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pickle

# Setting arguments

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, default="./", help="Which directory to find the 'data' folder containing training, validation and test")
parser.add_argument("--num_epoch", type=int, default=1, help="How many epochs should run during training")
parser.add_argument("--beta", type=int, default=1, help="The size of the regularization coefficient 'beta'")
parser.add_argument("--save_dir", type=str, default="./models/")

args = parser.parse_args()

num_epoch = args.num_epoch
beta = args.beta
path = args.path
save_dir = args.save_dir

input_shape = Config.input_shape
latent_shape = Config.latent_shape

# Setup for training
writer = SummaryWriter()

model = VRNN(input_shape, latent_shape, beta)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epoch = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

train_ds = AISDataset(path+"data/train.pcl")
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)

val_ds = AISDataset(path+"data/val.pcl")
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=True)

# move the model to the device
model = model.to(device)
training_loss = []
val_loss = []
while epoch < num_epoch:

    epoch_train_loss = 0
    epoch_val_loss = 0

    epoch_train_kl = 0
    epoch_val_kl = 0

    epoch_train_logpx = 0
    epoch_val_logpx = 0

    model.train()

    for inputs in train_loader:
        inputs = inputs.to(device)

        loss, diagnostics = model(inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_kl += np.mean(diagnostics["kl"])
        epoch_train_logpx += np.mean(diagnostics["log_px"])

        epoch_train_loss += loss.data

    writer.add_scalar("Loss/train", epoch_train_loss, epoch)
    writer.add_scalar("KL/train", epoch_train_kl, epoch)
    writer.add_scalar("Log_px/train", epoch_train_logpx, epoch)

    model.eval()

    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.to(device)

            loss, diagnostics = model(inputs)

            epoch_val_kl += np.mean(diagnostics["kl"])
            epoch_val_logpx += np.mean(diagnostics["log_px"])

            epoch_val_loss += loss

        writer.add_scalar("Loss/validation", epoch_val_loss, epoch)
        writer.add_scalar("KL/validation", epoch_val_kl, epoch)
        writer.add_scalar("Log_px/validation", epoch_val_logpx, epoch)

    training_loss.append(epoch_train_loss/len(train_loader))
    val_loss.append(epoch_val_loss/len(val_loader))

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, training loss: {training_loss[-1]}, validation loss: {val_loss[-1]}')

    epoch += 1

writer.flush()
writer.close()

torch.save(model.state_dict(), save_dir+f"vrnn_{num_epoch}_epochs.pt")

with open(save_dir+f"training_loss_{num_epoch}_epochs.txt", "wb") as fp:
    pickle.dump(training_loss, fp)

with open(save_dir+f"validation_loss_{num_epoch}_epochs.txt", "wb") as fp:
    pickle.dump(val_loss, fp)


