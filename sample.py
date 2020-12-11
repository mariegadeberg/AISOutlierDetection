import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from Config import *
from model import VRNN
from utils_preprocess import AISDataset, TruncCollate

state_dict = torch.load("../HPCoutputs/models/bh15epoch/vrnn_bh15_epochs.pt", map_location=torch.device('cpu'))
#state_dict = torch.load("/Volumes/MNG/models/vrnn_bh10_epochs.pt", map_location=torch.device('cpu'))
model = VRNN(Config.input_shape["bh"], Config.latent_shape, 1)
model.load_state_dict(state_dict)

train_ds = AISDataset("/Volumes/MNG/data/train_bh_.pcl", "/Volumes/MNG/data/mean_bh.pcl")
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=10, shuffle=True, collate_fn=TruncCollate())

it = iter(train_loader)
inputs = next(it)
loss, diagnostics = model(inputs)


lat_cols = pd.Float64Index(np.round(Config.lat_columns["bh"], 2))
long_cols = pd.Float64Index(np.round(Config.long_columns["bh"], 2))
breaks = Config.breaks["bh"]


plt.figure()
for k in range(0, 10):
    lat_out = []
    long_out = []
    for i in range(len(diagnostics["log_px"])):
        t = diagnostics["log_px"][i,k,0,:]
        lat, long, sog, cog = np.split(t, breaks)
        lat_out.append(lat_cols[np.argmax(lat)])
        long_out.append(long_cols[np.argmax(long)])

    print(f"Average Latitude: {np.mean(lat_out)}")
    print(f"Average Longitude: {np.mean(long_out)}")
    plt.plot(lat_out, long_out, ".-")
plt.show()

plt.figure()
for k in range(0, 10):
    plt.plot(diagnostics["h"][:, k, 0])
plt.show()

plt.figure()
for i in range(0, 10):
    t = inputs[i, :, :].numpy()
    lat_t, long_t, sog_t, cog_t = np.split(t, breaks, axis = 1)

    lat_t = lat_cols[np.argmax(lat_t, axis=1)]
    long_t = long_cols[np.argmax(long_t, axis=1)]

    plt.plot(lat_t, long_t, ".-")
plt.show()



