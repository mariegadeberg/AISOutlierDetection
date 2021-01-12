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

crs = 'epsg:4326'

dk2 = gpd.read_file("/Volumes/MNG/gadm36_DNK_shp/gadm36_DNK_0.shp")
dk2 = dk2.to_crs(crs)
swe = gpd.read_file("/Volumes/MNG/gadm36_SWE_shp/gadm36_SWE_0.shp")
swe = swe.to_crs(crs)


def calc_mse(loader, model, name):
    print(f"Starting calculations for loader: {name}")
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

    return mse, worst_img_batch


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

def get_logpx(loader, model, name):
    print(f"Starting calculations for {name}")

    log_px_out = []
    logits_out = []
    probs_out = []
    inputs_out = []

    for inputs in loader:
        inputs = inputs.unsqueeze(1)

        logits, log_px = model.sample(inputs)
        logits = logits.detach().numpy()
        log_px = log_px.detach().numpy()

        probs = np.exp(logits) / (1 + np.exp(logits))

        logits_out.append(logits)
        log_px_out.append(log_px)
        probs_out.append(probs)

        inputs_out.append(inputs.squeeze())


    logits_out = [item for sublist in logits_out for item in sublist]
    log_px_out = [item for sublist in log_px_out for item in sublist]
    probs_out = [item for sublist in probs_out for item in sublist]
    inputs_out = [item for sublist in inputs_out for item in sublist]

    return logits_out, log_px_out, probs_out, inputs_out

def get_corrupt_results(loader_corrupt, loader_true, model, name):
    print(f"starting calculation for {name}")

    logits_c = []
    log_px_c = []
    probs_c = []
    inputs_out = []

    for inputs_true, inputs_corrupt in zip(loader_true, loader_corrupt):
        inputs_true = inputs_true.unsqueeze(1)
        inputs_corrupt = inputs_corrupt.unsqueeze(1)

        logits, log_px = model.sample_corrupt(inputs_corrupt, inputs_true)
        logits = logits.detach().numpy()
        log_px = log_px.detach().numpy()

        probs = np.exp(logits) / (1 + np.exp(logits))

        logits_c.append(logits)
        log_px_c.append(log_px)
        probs_c.append(probs)
        inputs_out.append(inputs_corrupt)

    logits_c = [item for sublist in logits_c for item in sublist]
    log_px_c = [item for sublist in log_px_c for item in sublist]
    probs_c = [item for sublist in probs_c for item in sublist]
    inputs_out = [item for sublist in inputs_out for item in sublist]

    return logits_c, log_px_c, probs_c, inputs_out


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


def show_worst(logits, log_px, probs, inputs, inputs_true):
    worst_idx = np.argmin(log_px)

    plt.figure()
    plt.subplot(2, 2, 1)
    sns.heatmap(inputs_true[worst_idx])
    plt.title("True Path")

    plt.subplot(2, 2, 2)
    sns.heatmap(inputs[worst_idx])
    plt.title("Corrupted path")

    plt.subplot(2,2, 3)
    sns.heatmap(logits[worst_idx].squeeze())
    plt.title("Logits")

    plt.subplot(2, 2, 4)
    sns.heatmap(probs[worst_idx].squeeze())
    plt.title("Probabilities")

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


def catch_outliers(log_px, threshold):
    outliers = np.where(log_px < threshold)
    return outliers[0]

threshold = np.quantile(log_px_true, 0.05)

outliers_true = catch_outliers(log_px_true, threshold)
outliers_flip = catch_outliers(log_px_flip, threshold)
outliers_shift = catch_outliers(log_px_shift, threshold)

outliers_pass = catch_outliers(log_px_pass, threshold)


def tag_route_for_plt(dataset, idx_ano):
    output = {"lat": [],
              "long": [],
              "anomaly": []}

    i = 0
    for path in dataset:

        tst = path["FourHot"].todense()

        lat, long, sog, cog = np.split(tst, Config.breaks["bh"], axis=1)

        lat_columns = Config.lat_columns["bh"]
        long_columns = Config.long_columns["bh"]

        idx_lat = np.where(lat > 0)[1]
        idx_long = np.where(long > 0)[1]

        lat_c = lat_columns[idx_lat]
        long_c = long_columns[idx_long]

        if lat_c.shape != long_c.shape:
            continue

        output["lat"].append(lat_c)
        output["long"].append(long_c)

        if i in idx_ano:
            output["anomaly"].append(1)
        else:
            output["anomaly"].append(0)

        i += 1

    return output

with open("/Volumes/MNG/data/train_bh_Pass30min.pcl", "rb") as file:
    dataset = pickle.load(file)

with open("/Volumes/MNG/data/test_bh_.pcl", "rb") as file:
    dataset = pickle.load(file)


routes_tagged_true = tag_route_for_plt(dataset, random.choices(outliers_true, k=30))

routes_tagged_true = tag_route_for_plt(dataset, random.choices(outliers_pass, k=45))

legend_lines = [Line2D([0], [0], color="b", lw=2),
                Line2D([0], [0], color="r", lw=2)]
f, ax = plt.subplots()
dk2.plot(ax=ax, color="olivedrab")
swe.plot(ax=ax, color="olivedrab")
for i in range(len(routes_tagged_true["lat"])):
    if routes_tagged_true["anomaly"][i] == 1:
        continue
    else:
        c = "b"
        a = 0.2

    plt.plot(routes_tagged_true["long"][i], routes_tagged_true["lat"][i], c, alpha =a)
for i in range(len(routes_tagged_true["lat"])):
    if routes_tagged_true["anomaly"][i] == 1:
        c = "r"
        a = 1
    else:
        continue
    plt.plot(routes_tagged_true["long"][i], routes_tagged_true["lat"][i], c, alpha=a)
plt.title("Paths Marked for Anomalies", fontsize=16)
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

tst = pd.DataFrame(inputs_true[50].detach().numpy(), columns=long_cols, index=lat_cols[::-1])

plt.figure()
sns.heatmap(tst, cmap="Reds")
plt.title("True Path", fontsize=16)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.xticks(ticks=np.arange(0, len(long_cols)+1, 50), labels=long_cols[np.arange(0, len(long_cols)+1, 50)], fontsize=10)
plt.yticks(ticks=np.arange(0, len(lat_cols)+1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols)+1, 25)], fontsize=10)
plt.tight_layout()
plt.savefig("../Figures/ex_true2.png")
plt.show()



id = 7

plt.figure()
plt.subplot(2, 2, 1)
sns.set(font_scale=0.75)
sns.heatmap(inputs_true[id], cmap="Reds")
plt.xticks(ticks=np.arange(0, len(long_cols)+1, 50), labels=long_cols[np.arange(0, len(long_cols)+1, 50)], fontsize=8)
plt.yticks(ticks=np.arange(0, len(lat_cols)+1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols)+1, 25)], fontsize=8)
plt.title("True Path", fontsize=10)

plt.subplot(2, 2, 2)
sns.heatmap(inputs_pass[id].squeeze(), cmap="Reds")
plt.xticks(ticks=np.arange(0, len(long_cols)+1, 50), labels=long_cols[np.arange(0, len(long_cols)+1, 50)], fontsize=8)
plt.yticks(ticks=np.arange(0, len(lat_cols)+1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols)+1, 25)], fontsize=8)
plt.title("Corrupted path", fontsize=10)

plt.subplot(2, 2, 3)
sns.heatmap(logits_pass[id].squeeze(), cmap="Reds")
plt.xticks(ticks=np.arange(0, len(long_cols)+1, 50), labels=long_cols[np.arange(0, len(long_cols)+1, 50)], fontsize=8)
plt.yticks(ticks=np.arange(0, len(lat_cols)+1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols)+1, 25)], fontsize=8)
plt.title("Logits", fontsize=10)

plt.subplot(2, 2, 4)
sns.heatmap(probs_pass[id].squeeze(), cmap="Reds")
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