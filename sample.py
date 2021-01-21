import torch
import pickle
import matplotlib.pyplot as plt
from Config import *
from vrnn import VRNN
from utils_preprocess import AISDataset, TruncCollate, prep_mean
from matplotlib.lines import Line2D
import geopandas as gpd
import argparse
import seaborn as sns

parser = argparse.ArgumentParser()

parser.add_argument("--dk_shp", type=str, default="/Volumes/MNG/gadm36_DNK_shp/gadm36_DNK_0.shp", help="path for dk shape file")
parser.add_argument("--swe_shp", type=str, default="/Volumes/MNG/gadm36_SWE_shp/gadm36_SWE_0.shp", help="path for swe shape file")
parser.add_argument("--model_path", type=str, default="/trained_models/bh30_norminput_klann/vrnn_bh30_epochs.pt", help="path to model")
parser.add_argument("--mean_path", type=str, default="/Volumes/MNG/data/mean_bh.pcl", help="path to mean pickle")
parser.add_argument("--test_path", type=str, default="/Volumes/MNG/data/test_bh_.pcl", help="path to test set")
parser.add_argument("--batch_size", type=int, default=32, help="batch size of test set")
parser.add_argument("--ROI", type=str, default="bh", choices={"bh", "sk", "blt"}, help="indicator of region of interest. Default of 'bh' means bornholm.")
parser.add_argument("--gamma", type=float, default=0.6, help="fixed mean to use if model uses batch normalization in inference")
parser.add_argument("--bn_switch", type=str, default="False", choices={"False", "True"}, help="whether the model uses batch normalization in inference.")
parser.add_argument("--save_fig", type=str, default="True", choices={"False", "True"}, help="whether to save the generated figures")
parser.add_argument("--save_path", type=str, default="/Figures/", help="path indicating where to save figures")

args = parser.parse_args()

#if not os.path.exists(args.save_path):
#    os.mkdir(args.save_path)

# change str to boolean
if args.bn_switch == "False":
    bn_switch = False
elif args.bn_switch == "True":
    bn_switch = True
else:
    print("bn_switch indicator not known")

if args.save_fig == "False":
    save_fig = False
elif args.save_fig == "True":
    save_fig = True
else:
    print("save_fig indicator not known")

#Prepare shapefile maps
crs = 'epsg:4326'

dk2 = gpd.read_file(args.dk_shp)
dk2 = dk2.to_crs(crs)
swe = gpd.read_file(args.swe_shp)
swe = swe.to_crs(crs)

# Load model and instate state_dict
state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))

with open(args.mean_path, "rb") as f:
    mean_ = torch.tensor(pickle.load(f))

mean_logits = prep_mean(args.mean_path)

train_ds = AISDataset(args.test_path, args.mean_path)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=TruncCollate())

model = VRNN(Config.input_shape[args.ROI], Config.latent_shape, mean_logits, mean_, Config.splits[args.ROI], len(train_loader), gamma=args.gamma, bn_switch=bn_switch)
model.load_state_dict(state_dict)

# To create sample plot get first set of inputs from data loader
it = iter(train_loader)
inputs = next(it)
model.eval()
loss, diagnostics = model(inputs, 1)

# Prepare axes of plot
lat_cols = pd.Float64Index(np.round(Config.lat_columns[args.ROI], 2))
long_cols = pd.Float64Index(np.round(Config.long_columns[args.ROI], 2))
breaks = Config.breaks[args.ROI]


f, ax = plt.subplots()
dk2.plot(ax=ax, color="olivedrab")
swe.plot(ax=ax, color="olivedrab")
for k in range(0,args.batch_size):
    lat_out = []
    long_out = []
    for i in range(len(diagnostics["log_px"])):
        t = diagnostics["log_px"][i,k,:]
        lat, long, sog, cog = np.split(t, breaks)
        lat_out.append(lat_cols[np.argmax(lat)])
        long_out.append(long_cols[np.argmax(long)])
    plt.plot(long_out, lat_out, "b.-", alpha=0.1)
plt.xlim(Config.ROI_boundary_long, Config.long_max)
plt.ylim(Config.lat_min, Config.ROI_boundary_lat)
plt.title("Reconstructed paths", fontsize=16)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
if save_fig:
    plt.savefig(args.save_path+"recon_paths.png")
plt.show()

f, ax = plt.subplots()
dk2.plot(ax=ax, color="olivedrab")
swe.plot(ax=ax, color="olivedrab")
for i in range(0, args.batch_size):
    t = inputs[i, :, :].numpy()
    lat_t, long_t, sog_t, cog_t = np.split(t, breaks, axis=1)

    lat_t = lat_cols[np.argmax(lat_t, axis=1)]
    long_t = long_cols[np.argmax(long_t, axis=1)]

    plt.plot(long_t, lat_t, ".-b")
plt.xlim(Config.ROI_boundary_long, Config.long_max)
plt.ylim(Config.lat_min, Config.ROI_boundary_lat)
plt.title("True paths", fontsize=16)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
if save_fig:
    plt.savefig(args.save_path+"true_paths.png")
plt.show()

# Prepare mean
latm, longm, sogm, cogm = np.split(mean_.numpy()[-1], breaks)

lat1 = np.array([latm])
long1 = np.array([longm])[::-1]
tt = np.transpose(lat1)[::-1] @ long1

fig, ax = plt.subplots(figsize=(6.4,4.8))
sns.heatmap(tt, cmap="RdPu")
sns.set(font_scale=0.75)
plt.xticks(ticks=np.arange(0, len(long_cols)+1, 50), labels=long_cols[np.arange(0, len(long_cols)+1, 50)], fontsize=10)
plt.yticks(ticks=np.arange(0, len(lat_cols)+1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols)+1, 25)], fontsize=10)
plt.title("Matrix Multiplication of Mean Longitude \n and Mean Latitude Vectors", fontsize=14)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.tight_layout()
if save_fig:
    plt.savefig(args.save_path+"heatmap_mean.png")
plt.show()

# Prepare mean logits
latm, longm, sogm, cogm = np.split(mean_logits.numpy()[-1], breaks)


legend_lines = [Line2D([0], [0], color="b", lw=2),
                Line2D([0], [0], color="g", lw=2),
                Line2D([0], [0], color="red", lw=2)]

plt.figure()
for k in range(2,3):
    lat_out = []
    long_out = []
    for i in range(len(diagnostics["log_px"])):
        t = diagnostics["log_px"][i,k, :]
        q = inputs[k, i, :]
        lat, long, sog, cog = np.split(t, breaks)
        latt, longt, sogt, cogt = np.split(q, breaks)
        plt.plot(lat_cols, lat, 'b', alpha=0.3)
        plt.plot(lat_cols, latt, 'g')
    plt.plot(lat_cols, latm, ".", c='red', markersize=2)
plt.title("Latitude of Reconstructed Path", fontsize=16)
plt.xlabel("Latitude", fontsize=12)
plt.ylabel("Logits", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(legend_lines, ["Logit vector for prediction", "One-hot encoding of true path", "Mean logits vector from dataset"], fontsize=12)
if save_fig:
    plt.savefig(args.save_path + "lat_profile.png")
plt.show()


plt.figure()
for k in range(2,3):
    lat_out = []
    long_out = []
    for i in range(len(diagnostics["log_px"])):
        t = diagnostics["log_px"][i,k,:]
        q = inputs[k, i, :]
        lat, long, sog, cog = np.split(t, breaks)
        latt, longt, sogt, cogt = np.split(q, breaks)
        plt.plot(long_cols, long, 'b', alpha=0.3)
        plt.plot(long_cols, longt, 'g')
    plt.plot(long_cols, longm,  ".", c='red', markersize=1.5)
plt.title("Longitude of Reconstructed Path", fontsize=16)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Logits", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(legend_lines, ["Logit vector for prediction", "One-hot encoding of true path", "Mean logits vector from dataset"], fontsize=12)
if save_fig:
    plt.savefig("../Figures/long_profile.png")
plt.show()


