import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from Config import *
from model import VRNN
from utils_preprocess import AISDataset, TruncCollate, prep_mean
from matplotlib.lines import Line2D

state_dict = torch.load("../HPCoutputs/models/bh30_meanMI/vrnn_bh30_epochs.pt", map_location=torch.device('cpu'))
state_dict = torch.load("/Volumes/MNG/HPCoutputs/models/bh30_meanklann/vrnn_bh30_epochs.pt", map_location=torch.device('cpu'))



with open("/Volumes/MNG/data/mean_bh.pcl", "rb") as f:
    mean_ = torch.tensor(pickle.load(f))

mean_logits = prep_mean("/Volumes/MNG/data/mean_bh.pcl")

train_ds = AISDataset("/Volumes/MNG/data/Small/train_bh_small.pcl", "/Volumes/MNG/data/mean_bh.pcl")
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=TruncCollate())

model = VRNN(Config.input_shape["bh"], Config.latent_shape, mean_logits,mean_, Config.splits["bh"], len(train_loader), gamma=0.6, bn_switch=False)
model.load_state_dict(state_dict)

it = iter(train_loader)
inputs = next(it)
model.eval()
loss, diagnostics = model(inputs, 1)


lat_cols = pd.Float64Index(np.round(Config.lat_columns["bh"], 2))
long_cols = pd.Float64Index(np.round(Config.long_columns["bh"], 2))
breaks = Config.breaks["bh"]


plt.figure()
for k in range(0,32):
    lat_out = []
    long_out = []
    for i in range(len(diagnostics["log_px"])):
        t = diagnostics["log_px"][i,k,:] #- mean_logits.numpy()[-1]
        lat, long, sog, cog = np.split(t, breaks)
        lat_out.append(lat_cols[np.argmax(lat)])
        long_out.append(long_cols[np.argmax(long)])
    #print(lat[np.argmax(lat)-10:np.argmax(lat)+10])
    #print(max(lat))
    #print(f"Average Latitude: {np.mean(lat_out)}")
    #print(f"Average Longitude: {np.mean(long_out)}")
    plt.plot(long_out, lat_out, "b.-", alpha=0.1)
#plt.title("Trajectory reconstruction for model trained \n 16 epochs subtracted mean logits")
plt.xlim(Config.ROI_boundary_long, Config.long_max)
plt.ylim(Config.lat_min, Config.ROI_boundary_lat)
plt.title("Reconstructed paths", fontsize=16)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.savefig("../Figures/recon_paths.png")
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


latm, longm, sogm, cogm = np.split(mean_.numpy()[-1], breaks)

X, Y = np.meshgrid(long_cols, lat_cols)
Z = np.concatenate([long, lat]).reshape(len(long_cols), len(lat_cols))


lat1 = np.array([lat])
long1 = np.array([long])[::-1]
tst = long1.transpose().dot(lat1)

tt = pd.DataFrame(tst.transpose(), index=lat_cols[::-1], columns=long_cols)

import seaborn as sns
from matplotlib.ticker import MultipleLocator
fig, ax = plt.subplots()
sns.heatmap(tt, cmap="RdPu")
plt.xticks(ticks=np.arange(0, len(long_cols)+1, 50.25), labels=np.arange(min(long_cols), max(long_cols), 0.5))
plt.yticks(ticks=np.arange(0, len(lat_cols)+1, 25.25), labels=np.arange(min(lat_cols), max(lat_cols)+0.25, 0.25)[::-1])
plt.show()

plt.figure()
plt.pcolormesh(X, Y, tst)
plt.show()

tst = pd.DataFrame(data={"long": long_cols, "lat": lat, "intensity":[long, lat]})

plt.figure()
plt.imshow([long + lat], cmap="hot", interpolation="nearest")
plt.show()

plt.figure()
plt.plot(mean_.numpy()[-1])
plt.show()


latm, longm, sogm, cogm = np.split(mean_logits.numpy()[-1], breaks)

plt.figure()
plt.plot(longm)
plt.show()

plt.figure()
plt.plot(latm*100)
plt.show()

legend_lines = [Line2D([0], [0], color="blue", lw=2),
                Line2D([0], [0], color="green", lw=2),
                Line2D([0], [0], color="red", lw=2)]

plt.figure()
for k in range(2,3):
    lat_out = []
    long_out = []
    for i in range(0,32):
        t = diagnostics["log_px"][i,k,:]
        q = inputs[k, i, :]
        lat, long, sog, cog = np.split(t, breaks)
        latt, longt, sogt, cogt = np.split(q, breaks)
        #plt.plot(long)
        #plt.plot(longt)
        plt.plot(lat_cols, lat, 'b', alpha=0.1)
        plt.plot(lat_cols, latt, 'g')
    plt.plot(lat_cols, latm, 'r.', markersize=2)
plt.title("Latitude of Reconstructed Path", fontsize=16)
plt.xlabel("Latitude", fontsize=12)
plt.ylabel("Logits", fontsize=12)
plt.legend(legend_lines, ["Logit vector for prediction", "One-hot encoding of true path", "Mean logits vector from dataset"])
plt.savefig("../Figures/lat_profile.png")
plt.show()


plt.figure()
for k in range(2,3):
    lat_out = []
    long_out = []
    for i in range(0,32):
        t = diagnostics["log_px"][i,k,:]
        q = inputs[k, i, :]
        lat, long, sog, cog = np.split(t, breaks)
        latt, longt, sogt, cogt = np.split(q, breaks)
        plt.plot(long_cols, long, 'b', alpha=0.1)
        plt.plot(long_cols, longt, 'g')
    plt.plot(long_cols, longm, 'r.', markersize=1.5)
plt.title("Longitude of Reconstructed Path", fontsize=16)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Logits", fontsize=12)
plt.legend(legend_lines, ["Logit vector for prediction", "One-hot encoding of true path", "Mean logits vector from dataset"])
plt.savefig("../Figures/long_profile.png")
plt.show()









tst = pd.DataFrame(inputs.numpy()[0, :, :])

y = tst.iloc[:,0:201]
z = tst.iloc[:,201:603]

x = np.transpose(y).dot(z)

import seaborn as sns

plt.figure()
sns.heatmap(x, cmap='Paired')
plt.show()