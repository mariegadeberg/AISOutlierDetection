from model import VRNN
import torch
from torch.utils.data import DataLoader
from utils_preprocess import AISDataset
from Config import *
from utils_preprocess import PadCollate
import matplotlib.pyplot as plt
import argparse
from torch.utils.tensorboard import SummaryWriter

# Setting arguments

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, default="/", help="Which directory to find the 'data' folder containing training, validation and test")
parser.add_argument("--num_epoch", type=int, default=1, help="How many epochs should run during training")
parser.add_argument("--beta", type=int, default=1, help="The size of the regularization coefficient 'beta'")

args = parser.parse_args()

num_epoch = args.num_epoch
beta = args.beta
path = arg.path
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

test_ds = AISDataset(path+"data/test.pcl")
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=True)

# move the model to the device
model = model.to(device)
training_loss = []
val_loss = []
while epoch < num_epoch:

    epoch_train_loss = 0
    epoch_val_loss = 0

    model.train()

    for inputs in train_loader:
        inputs = inputs.to(device)

        loss, diagnostics = model(inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.data

    writer.add_scalar("Loss/train", epoch_train_loss, epoch)

    model.eval()

    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.to(device)

            loss, diagnostics = model(inputs)

            epoch_val_loss += loss

    writer.add_scalar("Loss/validation", epoch_val_loss, epoch)

    training_loss.append(epoch_train_loss/len(train_loader))
    val_loss.append(epoch_val_loss/len(val_loader))

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, training loss: {training_loss[-1]}, validation loss: {val_loss[-1]}')

    epoch += 1

writer.flush()
writer.close()




