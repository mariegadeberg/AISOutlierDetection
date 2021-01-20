from cnnvae import CVAE
import torch
from utils_preprocess import *
from Config import *
import matplotlib.pyplot as plt
import seaborn as sns


ds = AISDataset_ImageOneHot("/Volumes/MNG/data/test_bh_.pcl", Config)
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)

ds_true = AISDataset_ImageOneHot("/Volumes/MNG/data/test_bh_.pcl", Config)
loader_true = torch.utils.data.DataLoader(ds_true, batch_size=32, shuffle=False)

it_t = iter(loader_true)
inputs_t = next(it_t)

it = iter(loader)
inputs = next(it)

input_shape = inputs[0].shape
latent_features = 100
init_kernel = 16

model = CVAE(latent_features)
state_dict = torch.load("/Volumes/MNG/HPCoutputs/models/CVAE/bh30nonorm/cvae_bh30_epochsnonorm.pt", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

inputs = inputs.unsqueeze(1)
recon, log_px = model.sample(inputs)
recon = recon.detach().numpy().squeeze()

recon_norm = np.zeros([32, 201, 402])
for i in range(32):
    recon_norm[i, :, :] = (recon[i, :, :] - recon[i, :, :].min()) / (recon[i, :, :].max() - recon[i, :, :].min())

recon_prob = np.exp(recon) / (1 + np.exp(recon))

recon_prob_norm = np.zeros([32, 201, 402])
for i in range(32):
    recon_prob_norm[i, :, :] = (recon_prob[i, :, :] - recon_prob[i, :, :].min()) / (recon_prob[i, :, :].max() - recon_prob[i, :, :].min())

mse = []
for i in range(len(inputs)):
    mse.append(((inputs.squeeze()[i, :, :] - recon_prob_norm[i, :, :])**2).mean(axis=0).mean().item())



plt.figure()
sns.heatmap(recon[30, :, :])
plt.show()

plt.figure()
sns.heatmap(recon_prob[30, :, :])
plt.show()

plt.figure()
sns.heatmap(inputs.squeeze()[30, :, :])
plt.show()

plt.figure()
sns.heatmap(inputs_t.squeeze()[15, :, :])
plt.show()

plt.figure()
sns.heatmap(log_px.squeeze().detach().numpy()[5, :, :])
plt.show()


# Using logpx
ds = AISDataset_Image("/Volumes/MNG/data/test_bh_.pcl", Config)
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)

it = iter(loader)
inputs = next(it)

latent_features = 100

model = CVAE(latent_features)
state_dict = torch.load("/Volumes/MNG/HPCoutputs/models/CVAE/bh30_noBN/cvae_bh30_epochs.pt", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

inputs = inputs.unsqueeze(1)
recon, log_px = model.sample(inputs)
recon = recon.detach().numpy().squeeze()
log_px = log_px.detach().numpy()

recon_prob = np.exp(recon) / (1 + np.exp(recon))

plt.figure()
sns.heatmap(recon.detach().numpy().squeeze()[18, :, :])
plt.show()

plt.figure()
sns.heatmap(inputs.squeeze()[4, :, :])
plt.show()

plt.figure()
sns.heatmap(recon_prob[18, :, :])
plt.show()


lat_cols = np.round(Config.lat_columns["bh"], 1)
long_cols = np.round(Config.long_columns["bh"], 1)


id = 1

plt.figure()
plt.subplot(2, 2, 1)
sns.heatmap(inputs_t[id, :, :], cmap="Reds")
plt.xticks(ticks=np.arange(0, len(long_cols)+1, 50), labels=long_cols[np.arange(0, len(long_cols)+1, 50)], fontsize=10)
plt.yticks(ticks=np.arange(0, len(lat_cols)+1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols)+1, 25)], fontsize=10)
plt.title("True Path")

plt.subplot(2, 2, 2)
sns.heatmap(inputs.squeeze()[id, :, :], cmap="Reds")
plt.xticks(ticks=np.arange(0, len(long_cols)+1, 50), labels=long_cols[np.arange(0, len(long_cols)+1, 50)], fontsize=10)
plt.yticks(ticks=np.arange(0, len(lat_cols)+1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols)+1, 25)], fontsize=10)
plt.title("Corrupted path")

plt.subplot(2, 2, 3)
sns.heatmap(recon[id, :, :], cmap="Reds")
plt.xticks(ticks=np.arange(0, len(long_cols)+1, 50), labels=long_cols[np.arange(0, len(long_cols)+1, 50)], fontsize=10)
plt.yticks(ticks=np.arange(0, len(lat_cols)+1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols)+1, 25)], fontsize=10)
plt.title("Logits")

plt.subplot(2, 2, 4)
sns.heatmap(recon_prob[id, :, :], cmap="Reds")
plt.xticks(ticks=np.arange(0, len(long_cols)+1, 50), labels=long_cols[np.arange(0, len(long_cols)+1, 50)], fontsize=10)
plt.yticks(ticks=np.arange(0, len(lat_cols)+1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols)+1, 25)], fontsize=10)
plt.title("Probabilities")
plt.tight_layout()
plt.savefig("../Figures/example_bo1.png")
plt.show()
