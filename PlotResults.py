import pandas as pd
import matplotlib.pyplot as plt
import pickle

results = pd.read_csv("../HPCoutputs/models/bh15epoch/output_15bh.txt")
results = pd.read_csv("/Volumes/MNG/models/output_10bh.txt")



plt.figure()
plt.plot(results.training_loss)
#plt.plot(results.validation_loss)
#plt.legend(["Traning loss", "Validation loss"])
plt.title("Loss for Bornholm model for 15 epochs")
plt.show()


plt.figure()
plt.plot(results.training_kl)
#plt.plot(results.validation_kl)
#plt.legend(["Traning kl", "Validation kl"])
plt.title("KL for Bornholm model for 15 epochs")
plt.show()


plt.figure()
plt.plot(results.training_logpx)
#plt.plot(results.validation_logpx)
#plt.legend(["Traning logpx", "Validation logpx"])
plt.title("Logpx for Bornholm model for 15 epochs")
plt.show()

with open("../HPCoutputs/models/bh_small100epoch/diagnostics_100_bh.pcl", "rb") as f:
    d2 = pickle.load(f)

plt.figure()
plt.plot(d2[50]["kl"])
plt.title("KL development through random chosen path")
plt.show()

plt.figure()
plt.plot(d2[50]["log_px"])
plt.title("Log_px development through random chosen path")
plt.show()

plt.figure()
plt.plot(d2[14]["h"])
plt.title("h development through random chosen path")
plt.show()


with open("/Volumes/MNG/models/diagnostics_2_bh.pcl", "rb") as f:
    d2 = pickle.load(f)


lat_cols = pd.Float64Index(np.round(Config.lat_columns["bh"], 2))
long_cols = pd.Float64Index(np.round(Config.long_columns["bh"], 2))

breaks = (201, 201+402, 201+402+31)

lat_out = []
long_out = []
for i in range(len(d2[0]["log_px"])):
    t = d2[0]["log_px"][i,0,0,:]
    lat, long, sog, cog = np.split(t, breaks)
    lat_out.append(lat_cols[np.argmax(lat)])
    long_out.append(long_cols[np.argmax(long)])

plt.figure()
plt.plot(long_out, lat_out, ".-")
plt.show()

import torch

state_dict = torch.load("../HPCoutputs/models/bh_small100epoch/vrnn_bh100_epochs.pt", map_location=torch.device('cpu'))
