import pandas as pd
import matplotlib.pyplot as plt
import pickle

results = pd.read_csv("../HPCoutputs/models/bh30_meanMI/output_30bh.txt")
results = pd.read_csv("/Volumes/MNG/models/output_15bh.txt")


plt.figure()
plt.plot(results.training_loss)
#plt.plot(results.validation_loss)
plt.legend([f"Value at last epoch: {results.training_loss.iloc[-1]:.4f}"], fontsize=12)
plt.title("Training Loss 30 epochs", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("ELBO",  fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.savefig("../Figures/loss_30mean.eps")
plt.show()


plt.figure()
plt.plot(results.training_kl)
#plt.plot(results.validation_kl)
plt.legend([f"Value at last epoch: {results.training_kl.iloc[-1]:.4f}"], fontsize=12)
plt.title("KL divergence 30 epochs", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("KL divergence", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.savefig("../Figures/kl_30mean.eps")
plt.show()

plt.figure()
plt.plot(results.mi)
#plt.plot(results.validation_kl)
plt.legend([f"Value at last epoch: {results.mi.iloc[-1]:.4f}"], fontsize=12)
plt.title("Mutual information 30 epochs", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Mutual Information", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.savefig("../Figures/kl_30mean.eps")
plt.show()

plt.figure()
plt.plot(results.au)
#plt.plot(results.validation_kl)
plt.legend([f"Value at last epoch: {results.au.iloc[-1]:.4f}"], fontsize=12)
plt.title("Active Units 30 epochs", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Mutual Information", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.savefig("../Figures/kl_30mean.eps")
plt.show()


plt.figure()
plt.plot(results.training_logpx)
#plt.plot(results.validation_logpx)
plt.legend([f"Value at last epoch: {results.training_logpx.iloc[-1]:.4f}"])
plt.title("Logits for Bornholm model for 100 epochs")
#plt.ylim(-0.001, 0.01)
plt.show()

with open("../HPCoutputs/models/bh50_meanMI/grad_50_bh.pcl", "rb") as f:
    w_ave = pickle.load(f)

legend = []
plt.figure()
for name in w_ave.keys():
    #if name == "phi_z.0.weight" or name == "phi_x.0.weight" or name == "prior.0.weight":
    #    continue
    plt.plot(w_ave[name])
    legend.append([name])
plt.title("Trace of gradients through 1st epoch of training",  fontsize=16)
plt.legend(legend,  fontsize=12)
plt.xlabel("Steps",  fontsize=12)
plt.ylabel("Parameter value",  fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.ylim(-0.3, 0.1)
plt.show()
#plt.savefig("../Figures/gradient_flow_no_mean_zoom.eps")




with open("../HPCoutputs/models/bh_small100epoch/diagnostics_100_bh.pcl", "rb") as f:
    d2 = pickle.load(f)


with open("/Volumes/MNG/models/diagnostics_5_bh.pcl", "rb") as f:
    d2 = pickle.load(f)

plt.figure()
plt.plot(d2[50]["kl"])
plt.title("KL development through random chosen path")
plt.show()

plt.figure()
plt.plot(d2[0]["log_px"].sum(axis=2))
plt.title("Log_px development through random chosen path")
plt.show()

plt.figure()
plt.plot(d2[99]["h"][:,:])
plt.title("h development through random chosen path")
plt.show()

plt.figure()
plt.plot(d2[4]["mu_prior"].mean(axis=1), 'b')
plt.plot(d2[4]["mu_post"].mean(axis=1), 'r')
plt.legend(["Prior", "Posterior"])
plt.title("Mean of batch means from distributions through timesteps")
plt.show()

with open("/Volumes/MNG/models/diagnostics_1_bh.pcl", "rb") as f:
    d2 = pickle.load(f)


lat_cols = pd.Float64Index(np.round(Config.lat_columns["bh"], 2))
long_cols = pd.Float64Index(np.round(Config.long_columns["bh"], 2))

breaks = (201, 201+402, 201+402+31)

lat_out = []
long_out = []
for i in range(len(d2[0]["log_px"])):
    t = d2[0]["log_px"][i,0,:]
    lat, long, sog, cog = np.split(t, breaks)
    lat_out.append(lat_cols[np.argmax(lat)])
    long_out.append(long_cols[np.argmax(long)])

plt.figure()
plt.plot(long_out, lat_out, ".-")
plt.show()

import torch

state_dict = torch.load("../HPCoutputs/models/bh_small100epoch/vrnn_bh100_epochs.pt", map_location=torch.device('cpu'))
