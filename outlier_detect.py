from cnnvae import CVAE
import torch
from utils_preprocess import *
from Config import *
import matplotlib.pyplot as plt
import seaborn as sns


ds = AISDataset_Image("/Volumes/MNG/data/test_bh_.pcl", Config)
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
latent_features = Config.latent_shape

model = CVAE(latent_features)
state_dict = torch.load("/Volumes/MNG/HPCoutputs/models/CVAE/bh15/vrnn_bh15_epochs.pt", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

mse = []
worst_img_batch = {"real": [],
                   "recon": []}
for inputs in loader:
    inputs = inputs.unsqueeze(1)
    recon = model.sample(inputs).detach().numpy().squeeze()

    recon_prob = np.exp(recon) / (1 + np.exp(recon))

    recon_prob_norm = np.zeros([inputs.size(0), len(Config.lat_columns["bh"]), len(Config.long_columns["bh"])])
    for i in range(inputs.size(0)):
        recon_prob_norm[i, :, :] = (recon_prob[i, :, :] - recon_prob[i, :, :].min()) / (
                    recon_prob[i, :, :].max() - recon_prob[i, :, :].min())

    mse_tmp = []
    for i in range(len(inputs)):
        m = ((inputs.squeeze()[i, :, :] - recon_prob_norm[i, :, :]) ** 2).mean(axis=0).mean().item()
        mse.append(m)
        mse_tmp.append(m)

    idx = np.argmax(mse_tmp)

    worst_img_batch["real"].append(inputs.squeeze()[idx, :, :])
    worst_img_batch["recon"].append(recon_prob_norm[idx, :, :])


#Some visualizations

plt.figure()
sns.heatmap(worst_img_batch["real"][6])
plt.show()

plt.figure()
sns.heatmap(worst_img_batch["recon"][6])
plt.show()