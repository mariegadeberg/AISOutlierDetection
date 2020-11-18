from model import VRNN
import torch
from torch.utils.data import DataLoader
import pickle

num_epoch = 5
beta = 1
input_shape = 2410
latent_shape = 100

model = VRNN(input_shape, latent_shape, beta)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epoch = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

train_ds = AISDataset("Code2.0/local_files/test_data.pcl")

train_loader = torch.utils.data.DataLoader(train_ds, 1, True)

# move the model to the device
model = model.to(device)


while epoch < num_epoch:
    epoch += 1
    model.train()
    train_loss = 0

    for inputs in train_loader:
        inputs = inputs.to(device)

        loss, loss_list = model(inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss += loss.data

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


class AISDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path

        with open(self.path, "rb") as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return(len(self.dataset))

    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx]["FourHot"], dtype=torch.float)





i = 0
while i < 1:
    for inputs in dataloader:
        loss, loss_list = model(inputs)
        i+=1

