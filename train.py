from model import VRNN
import torch
from torch.utils.data import DataLoader
from utils_preprocess import AISDataset
from Config import *
from utils_preprocess import PadCollate
import matplotlib.pyplot as plt

path = Config.path

num_epoch = 3
beta = 1
input_shape = 1907
latent_shape = 100

model = VRNN(input_shape, latent_shape, beta)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epoch = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

train_ds = AISDataset(path+"/Code2.0/local_files/data_split_cargo.pcl")

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=PadCollate(dim=0))

# move the model to the device
model = model.to(device)
loss_epochs = []
batch_loss = []
while epoch < num_epoch:
    model.train()
    train_loss = 0
    for inputs in train_loader:
        inputs = inputs.to(device)

        loss, loss_list = model(inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss.append(loss)

        train_loss += loss.data

    loss_epochs.append(train_loss / len(train_loader.dataset))

    print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, train_loss / len(train_loader.dataset)))
    epoch += 1





i=0
for data in train_loader:
    while i < 1:
        print(data.shape)
        i += 1



