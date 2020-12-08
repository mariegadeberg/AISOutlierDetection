import pandas as pd
import matplotlib.pyplot as plt
import pickle

results = pd.read_csv("../HPCoutputs/models/sk15epoch/output_15sk.txt")
results2 = pd.read_csv("../HPCoutputs/models/bh15epoch/output_15bh.txt")


plt.figure()
plt.plot(results.training_loss)
plt.plot(results.validation_loss)
plt.legend(["Traning loss", "Validation loss"])
plt.title("Loss for Baelterne model for 15 epochs")
plt.show()


plt.figure()
plt.plot(results.training_kl)
plt.plot(results.validation_kl)
plt.legend(["Traning kl", "Validation kl"])
plt.title("KL for Baelterne model for 15 epochs")
plt.show()


plt.figure()
plt.plot(results.training_logpx)
plt.plot(results.validation_logpx)
plt.legend(["Traning logpx", "Validation logpx"])
plt.title("Logpx for Baelterne model for 15 epochs")
plt.show()

with open("../HPCoutputs/models/blt15epoch/diagnostics_15_blt.pcl", "rb") as f:
    d2 = pickle.load(f)

plt.figure()
plt.plot(d2[80]["kl"])
plt.title("KL development through random chosen path")
plt.show()

plt.figure()
plt.plot(d2[0]["log_px"])
plt.title("Log_px development through random chosen path")
plt.show()
