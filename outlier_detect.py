from cnnvae import CVAE
import torch
from utils_preprocess import *
from Config import *
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import random
random.seed(123)
import geopandas as gpd
from utils_outliers import *

crs = 'epsg:4326'

dk2 = gpd.read_file("/Volumes/MNG/gadm36_DNK_shp/gadm36_DNK_0.shp")
dk2 = dk2.to_crs(crs)
swe = gpd.read_file("/Volumes/MNG/gadm36_SWE_shp/gadm36_SWE_0.shp")
swe = swe.to_crs(crs)



latent_features = Config.latent_shape
model = CVAE(latent_features)

state_dict = torch.load("/Volumes/MNG/HPCoutputs/models/CVAE/bh30_noBN/cvae_bh30_epochs.pt", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

ds_true = AISDataset_Image("/Volumes/MNG/data/test_bh_.pcl", Config)
loader_true = torch.utils.data.DataLoader(ds_true, batch_size=32, shuffle=False)

ds_cut = AISDataset_ImageCut("/Volumes/MNG/data/test_bh_.pcl", Config)
loader_cut = torch.utils.data.DataLoader(ds_cut, batch_size=32, shuffle=False)

ds_flip = AISDataset_ImageFlipped("/Volumes/MNG/data/test_bh_.pcl", Config)
loader_flip = torch.utils.data.DataLoader(ds_flip, batch_size=32, shuffle=False)

ds_bo = AISDataset_ImageBlackout("/Volumes/MNG/data/test_bh_.pcl", Config, size=50)
loader_bo = torch.utils.data.DataLoader(ds_bo, batch_size=32, shuffle=False)

ds_shift = AISDataset_ImageShift("/Volumes/MNG/data/test_bh_.pcl", Config)
loader_shift = torch.utils.data.DataLoader(ds_shift, batch_size=32, shuffle=False)

ds_pass = AISDataset_ImageOneHot("/Volumes/MNG/data/train_bh_Pass30min.pcl", Config)
loader_pass = torch.utils.data.DataLoader(ds_pass, batch_size=32, shuffle=False)

ds_sail = AISDataset_Image("/Volumes/MNG/data/Sail/train_bh_Sail.pcl", Config)
loader_sail = torch.utils.data.DataLoader(ds_sail, batch_size=32, shuffle=False)



mse_true, worst_im_batch_true = calc_mse(loader_true, model, "true")
mse_cut, worst_im_batch_cut = calc_mse(loader_cut, model, "cut")
mse_flip, worst_im_batch_flip = calc_mse(loader_flip, model, "flipped")
mse_bo, worst_im_batch_bo = calc_mse(loader_bo, model, "blackout")


# Explore
print(f"True paths:")
print(f"--- max: {max(mse_true)}")
print(f"--- mean: {np.mean(mse_true)}")
print("")
print(f"Cut paths:")
print(f"--- max: {max(mse_cut)}")
print(f"--- mean: {np.mean(mse_cut)}")
print("")
print(f"Flipped paths:")
print(f"--- max: {max(mse_flip)}")
print(f"--- mean: {np.mean(mse_flip)}")
print("")
print(f"Blackout paths:")
print(f"--- max: {max(mse_bo)}")
print(f"--- mean: {np.mean(mse_bo)}")

#Some visualizations

plt.figure()
plt.subplot(2,1,1)
sns.heatmap(worst_im_batch_bo["real"][6])
plt.subplot(2,1,2)
sns.heatmap(worst_im_batch_bo["recon"][6])
plt.show()



it_true = iter(loader_true)
inputs_true = next(it_true)

it_cut = iter(loader_cut)
inputs_cut = next(it_cut)

it_flip = iter(loader_flip)
inputs_flip = next(it_flip)

it_bo = iter(loader_bo)
inputs_bo = next(it_bo)

it_shift = iter(loader_shift)
inputs_shift = next(it_shift)

it_pass = iter(loader_pass)
inputs_pass = next(it_pass)


plt.figure()

plt.subplot(2, 2, 1)
sns.heatmap(inputs_shift[19])
plt.title("Shift")
plt.xticks([])
plt.yticks([])


plt.subplot(2, 2, 2)
sns.heatmap(inputs_cut[19])
plt.title("Cut")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 3)
sns.heatmap(inputs_flip[19])
plt.title("Flipped")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 4)
sns.heatmap(inputs_bo[19])
plt.title("Blackout")
plt.xticks([])
plt.yticks([])

plt.show()


plt.figure()
sns.heatmap(inputs_pass[4])
plt.title("True")
plt.xticks([])
plt.yticks([])
plt.show()



plt.figure()
sns.distplot(mse_true)
sns.distplot(mse_cut)
sns.distplot(mse_flip)
sns.distplot(mse_bo)
plt.legend(["True", "Cut", "Flipped", "Blackout"])
plt.show()


plt.figure()
plt.subplot(2, 1, 1)
sns.heatmap(worst_im_batch_flip["recon"][14])
plt.subplot(2, 1, 2)
sns.heatmap(worst_im_batch_flip["real"][14])
plt.show()


# ------------------------------------------ UTILIZING LOG_PX --------------------------------------------






logits_true, log_px_true, probs_true, inputs_true = get_logpx(loader_true, model, "true")
logits_flip, log_px_flip, probs_flip, inputs_flip = get_logpx(loader_flip, model, "flip")
logits_shift, log_px_shift, probs_shift, inputs_shift = get_logpx(loader_shift, model, "shift")

logits_pass, log_px_pass, probs_pass, inputs_pass = get_logpx(loader_pass, model, "pass")
logits_sail, log_px_sail, probs_sail, inputs_sail = get_logpx(loader_sail, model, "sail")


logits_cut, log_px_cut, probs_cut, inputs_cut = get_corrupt_results(loader_cut, loader_true, model, "Cut")
logits_bo, log_px_bo, probs_bo, inputs_bo= get_corrupt_results(loader_bo, loader_true, model, "blackout")



plt.figure()
sns.distplot(log_px_true)
sns.distplot(log_px_cut)
sns.distplot(log_px_flip)
sns.distplot(log_px_bo)
sns.distplot(log_px_shift)
plt.legend(["True", "Cut", "Flipped", "Blackout", "Shift"])
plt.title("Distibution of Log_px")
plt.xlim(-2000, 500)
plt.show()

plt.figure()
sns.distplot(log_px_true)
sns.distplot(log_px_pass)
plt.legend(["Cargo and Tanker", "Passenger"])
plt.title("Distibution of Log_px")
#plt.xlim(-2000, 500)
plt.show()


plt.figure()
sns.distplot(log_px_true)
sns.distplot(log_px_flip)
sns.distplot(log_px_shift)
plt.xlim(-2000, 500)
plt.legend(["True", "Flipped", "Shifted"], fontsize=12)
plt.xlabel(r"$Log(p_x)$", fontsize=12)
plt.title("Density Plot of Log Probability of Input", fontsize=16)
plt.savefig("../Figures/distplot_2.png")
plt.show()



#def collect(loader):
#    inputs = []
#    for i in loader:
#        inputs.append(i)
#
#    inputs = [item for sublist in inputs for item in sublist]
#
#    return inputs
#
#inputs_true = collect(loader_true)
#inputs_cut = collect(loader_cut)
#inputs_flip = collect(loader_flip)
#inputs_bo = collect(loader_bo)
#inputs_shift = collect(loader_shift)

show_worst(logits_cut, log_px_cut, probs_cut, inputs_cut, inputs_true)
show_worst(logits_flip, log_px_flip, probs_flip, inputs_flip, inputs_true)
show_worst(logits_bo, log_px_bo, probs_bo, inputs_bo, inputs_true)
show_worst(logits_shift, log_px_shift, probs_shift, inputs_shift, inputs_true)


print(f"True paths:")
print(f"--- max: {min(log_px_true):.3f}")
print(f"--- mean: {np.mean(log_px_true):.3f}")
print("")
print(f"Cut paths:")
print(f"--- max: {min(log_px_cut):.3f}")
print(f"--- mean: {np.mean(log_px_cut):.3f}")
print("")
print(f"Flipped paths:")
print(f"--- max: {min(log_px_flip):.3f}")
print(f"--- mean: {np.mean(log_px_flip):.3f}")
print("")
print(f"Blackout paths:")
print(f"--- max: {min(log_px_bo):.3f}")
print(f"--- mean: {np.mean(log_px_bo):.3f}")
print("")
print(f"Shifted paths:")
print(f"--- max: {min(log_px_shift):.3f}")
print(f"--- mean: {np.mean(log_px_shift):.3f}")


plt.figure()
sns.heatmap(inputs_true[5])
plt.xticks([])
plt.yticks([])
plt.title("Example Path")
plt.show()

plt.figure()
sns.heatmap(inputs_true[15])
plt.xticks([])
plt.yticks([])
plt.title("True Path")
plt.show()

plt.figure()
sns.heatmap(logits_true[15].squeeze())
plt.xticks([])
plt.yticks([])
plt.title("Logits")
plt.show()

plt.figure()
sns.heatmap(probs_true[15].squeeze())
plt.xticks([])
plt.yticks([])
plt.title("Probability")
plt.show()



threshold = np.quantile(log_px_true, 0.05)

outliers_true = catch_outliers(log_px_true, threshold)
outliers_flip = catch_outliers(log_px_flip, threshold)
outliers_shift = catch_outliers(log_px_shift, threshold)

outliers_pass = catch_outliers(log_px_pass, threshold)




with open("/Volumes/MNG/data/train_bh_Pass30min.pcl", "rb") as file:
    dataset_pass = pickle.load(file)

with open("/Volumes/MNG/data/test_bh_.pcl", "rb") as file:
    dataset = pickle.load(file)


routes_tagged_true = tag_route_for_plt(dataset, random.choices(outliers_true, k=30))

routes_tagged_pass = tag_route_for_plt(dataset_pass, random.choices(outliers_pass, k=45))

legend_lines = [Line2D([0], [0], color="b", lw=2),
                Line2D([0], [0], color="r", lw=2)]
f, ax = plt.subplots()
dk2.plot(ax=ax, color="olivedrab")
swe.plot(ax=ax, color="olivedrab")
for i in range(len(routes_tagged_pass["lat"])):
    if routes_tagged_pass["anomaly"][i] == 1:
        continue
    else:
        c = "b"
        a = 0.2

    plt.plot(routes_tagged_pass["long"][i], routes_tagged_pass["lat"][i], c, alpha =a)
for i in range(len(routes_tagged_pass["lat"])):
    if routes_tagged_pass["anomaly"][i] == 1:
        c = "r"
        a = 1
    else:
        continue
    plt.plot(routes_tagged_pass["long"][i], routes_tagged_pass["lat"][i], c, alpha=a)
plt.title("Passenger Ships \n Paths Marked for Anomalies", fontsize=16)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.xlim(13, 17)
plt.ylim(54.5, 56.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(legend_lines, ["Normal", "Anomaly"], fontsize=12)
#plt.savefig("../Figures/anomalies.png")
plt.show()




lat_cols = np.round(Config.lat_columns["bh"], 1)
long_cols = np.round(Config.long_columns["bh"], 1)

tst = pd.DataFrame(inputs_true[109].detach().numpy(), columns=long_cols, index=lat_cols[::-1])

plt.figure()
sns.heatmap(tst, cmap="Reds")
plt.title("True Path", fontsize=16)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.xticks(ticks=np.arange(0, len(long_cols)+1, 50), labels=long_cols[np.arange(0, len(long_cols)+1, 50)], fontsize=10)
plt.yticks(ticks=np.arange(0, len(lat_cols)+1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols)+1, 25)], fontsize=10)
plt.tight_layout()
#plt.savefig("../Figures/ex_true2.png")
plt.show()



id = 129

plt.figure()
plt.subplot(2, 2, 1)
sns.set(font_scale=0.75)
sns.heatmap(inputs_true[id], cmap="Reds")
plt.xticks(ticks=np.arange(0, len(long_cols)+1, 50), labels=long_cols[np.arange(0, len(long_cols)+1, 50)], fontsize=8)
plt.yticks(ticks=np.arange(0, len(lat_cols)+1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols)+1, 25)], fontsize=8)
plt.title("True Path", fontsize=10)

plt.subplot(2, 2, 2)
sns.heatmap(inputs_true[id].squeeze(), cmap="Reds")
plt.xticks(ticks=np.arange(0, len(long_cols)+1, 50), labels=long_cols[np.arange(0, len(long_cols)+1, 50)], fontsize=8)
plt.yticks(ticks=np.arange(0, len(lat_cols)+1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols)+1, 25)], fontsize=8)
plt.title("Corrupted path", fontsize=10)

plt.subplot(2, 2, 3)
sns.heatmap(logits_true[id].squeeze(), cmap="Reds")
plt.xticks(ticks=np.arange(0, len(long_cols)+1, 50), labels=long_cols[np.arange(0, len(long_cols)+1, 50)], fontsize=8)
plt.yticks(ticks=np.arange(0, len(lat_cols)+1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols)+1, 25)], fontsize=8)
plt.title("Logits", fontsize=10)

plt.subplot(2, 2, 4)
sns.heatmap(probs_true[id].squeeze(), cmap="Reds")
plt.xticks(ticks=np.arange(0, len(long_cols)+1, 50), labels=long_cols[np.arange(0, len(long_cols)+1, 50)], fontsize=8)
plt.yticks(ticks=np.arange(0, len(lat_cols)+1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols)+1, 25)], fontsize=8)
plt.title("Probabilities", fontsize=10)
plt.tight_layout()
#plt.savefig("../Figures/example_shift1.png")
plt.show()


it_bo = iter(loader_bo)
inputs_bo_tst = next(it_bo)

plt.figure()
plt.subplot(2, 1, 1)
sns.heatmap(inputs_bo[12].squeeze())
plt.subplot(2,1,2)
sns.heatmap(inputs_bo_tst[12])
plt.show()

f, ax = plt.subplots()
dk2.plot(ax=ax)
swe.plot(ax=ax)
plt.xlim(14, 17)
plt.ylim(54.5, 56.5)
plt.show()


inputs_shift_plt = prep_route_for_pltShift(dataset, "shift")

f, ax = plt.subplots()
dk2.plot(ax=ax, color="olivedrab")
swe.plot(ax=ax, color="olivedrab")
for i in range(len(inputs_shift_plt["lat"])):
    plt.plot(inputs_shift_plt["long"][i], inputs_shift_plt["lat"][i], "b", alpha =0.1)
plt.title("Shifted Routes", fontsize=16)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.xlim(13, 17)
plt.ylim(54.5, 56.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
#plt.legend(["Shifted Routes"], fontsize=12)
plt.savefig("../Figures/shifted_routes_map.png")
plt.show()


inputs_flip_plt = prep_route_for_pltFlipped(dataset, "flip")

f, ax = plt.subplots()
dk2.plot(ax=ax, color="olivedrab")
swe.plot(ax=ax, color="olivedrab")
for i in range(len(inputs_flip_plt["lat"])):
    plt.plot(inputs_flip_plt["long"][i], inputs_flip_plt["lat"][i], "b", alpha =0.1)
plt.title("Flipped Routes", fontsize=16)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.xlim(13, 17)
plt.ylim(54.5, 56.5)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
#plt.legend(["Shifted Routes"], fontsize=12)
plt.savefig("../Figures/flipped_routes_map.png")
plt.show()
