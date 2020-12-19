import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from Config import *
from model import VRNN
from utils_preprocess import AISDataset, TruncCollate, prep_mean

state_dict = torch.load("../HPCoutputs/models/bh_small100_fivo/vrnn_bh100_epochs.pt", map_location=torch.device('cpu'))
state_dict = torch.load("/Volumes/MNG/models/vrnn_bh16_epochs.pt", map_location=torch.device('cpu'))

mean_ = prep_mean("/Volumes/MNG/data/mean_bh.pcl")

train_ds = AISDataset("/Volumes/MNG/data/Small/train_bh_small.pcl", "/Volumes/MNG/data/mean_bh.pcl")
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=TruncCollate())

model = VRNN(Config.input_shape["bh"], Config.latent_shape, 1, mean_, Config.splits["bh"], len(train_loader))
model.load_state_dict(state_dict)

it = iter(train_loader)
inputs = next(it)
model.eval()
loss, diagnostics = model(inputs, 1)


lat_cols = pd.Float64Index(np.round(Config.lat_columns["bh"], 2))
long_cols = pd.Float64Index(np.round(Config.long_columns["bh"], 2))
breaks = Config.breaks["bh"]


plt.figure()
for k in range(0,2):
    lat_out = []
    long_out = []
    for i in range(len(diagnostics["log_px"])):
        t = diagnostics["log_px"][i,k,:] - mean_.numpy()[-1]
        lat, long, sog, cog = np.split(t, breaks)
        lat_out.append(lat_cols[np.argmax(lat)])
        long_out.append(long_cols[np.argmax(long)])
    #print(lat[np.argmax(lat)-10:np.argmax(lat)+10])
    #print(max(lat))
    #print(f"Average Latitude: {np.mean(lat_out)}")
    #print(f"Average Longitude: {np.mean(long_out)}")
    plt.plot(long_out, lat_out, ".-")
plt.show()

plt.figure()
for k in range(0, 10):
    plt.plot(diagnostics["logits"][:, k, :].mean(axis=1))
plt.show()

plt.figure()
for i in range(0, 32):
    t = inputs[i, :, :].numpy()
    lat_t, long_t, sog_t, cog_t = np.split(t, breaks, axis = 1)

    lat_t = lat_cols[np.argmax(lat_t, axis=1)]
    long_t = long_cols[np.argmax(long_t, axis=1)]

    plt.plot(long_t, lat_t, ".-")
plt.show()


lat, long, sog, cog = np.split(mean_.numpy()[-1], breaks)

plt.figure()
plt.plot(mean_.numpy()[-1])
plt.show()