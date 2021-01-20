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
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dk_shp", type=str, default="/Volumes/MNG/gadm36_DNK_shp/gadm36_DNK_0.shp", help="path for dk shape file")
parser.add_argument("--swe_shp", type=str, default="/Volumes/MNG/gadm36_SWE_shp/gadm36_SWE_0.shp", help="path for swe shape file")
parser.add_argument("--model_path", type=str, default="/Volumes/MNG/HPCoutputs/models/CVAE/bh30_noBN/cvae_bh30_epochs.pt", help="path to model")
parser.add_argument("--test_path", type=str, default="/Volumes/MNG/data/test_bh_.pcl", help="path to test set")
parser.add_argument("--pass_path", type=str, default="/Volumes/MNG/data/train_bh_Pass30min.pcl", help="path to passenger ship data set")
parser.add_argument("--save_fig", type=str, default="True", help="Whether to save the figures")
parser.add_argument("--ROI", type=str, default="bh", choices={"bh", "sk", "blt"}, help="indicator of region of interest. Default of 'bh' means bornholm.")
parser.add_argument("--save_path", type=str, default="Figures/", help="path to save images")
parser.add_argument("--make_calc", type=str, default="False", choices={"False", "True"}, help="Wheter to make all calculation. If false, calculations are read from file.")
parser.add_argument("--data_path", type=str, default="/Volumes/MNG/outlier_outputs/", help="Path to where output files from the calculation is")

args = parser.parse_args()

# change argument from str to boolean
if args.save_fig == "True":
    save_fig = True
elif args.save_fig == "False":
    save_fig = False
else:
    print("Indicator for save_fig not known")

if args.make_calc == "False":
    make_calc = False
elif args.make_calc == "True":
    make_calc = True
else:
    print("Indicator for make_calc not known")

# Prepare shapefile maps
crs = 'epsg:4326'

dk2 = gpd.read_file(args.dk_shp)
dk2 = dk2.to_crs(crs)
swe = gpd.read_file(args.swe_shp)
swe = swe.to_crs(crs)

if make_calc:
# Define model
    latent_features = Config.latent_shape
    model = CVAE(latent_features)

    state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    # Create different datasets
    ds_true = AISDataset_Image(args.test_path, Config)
    loader_true = torch.utils.data.DataLoader(ds_true, batch_size=32, shuffle=False)

    ds_cut = AISDataset_ImageCut(args.test_path, Config)
    loader_cut = torch.utils.data.DataLoader(ds_cut, batch_size=32, shuffle=False)

    ds_flip = AISDataset_ImageFlipped(args.test_path, Config)
    loader_flip = torch.utils.data.DataLoader(ds_flip, batch_size=32, shuffle=False)

    ds_bo = AISDataset_ImageBlackout(args.test_path, Config, size=50)
    loader_bo = torch.utils.data.DataLoader(ds_bo, batch_size=32, shuffle=False)

    ds_shift = AISDataset_ImageShift(args.test_path, Config)
    loader_shift = torch.utils.data.DataLoader(ds_shift, batch_size=32, shuffle=False)

    ds_pass = AISDataset_ImageOneHot(args.pass_path, Config)
    loader_pass = torch.utils.data.DataLoader(ds_pass, batch_size=32, shuffle=False)

    #ds_sail = AISDataset_Image("/Volumes/MNG/data/Sail/train_bh_Sail.pcl", Config)
    #loader_sail = torch.utils.data.DataLoader(ds_sail, batch_size=32, shuffle=False)



# ------------------------------------------ UTILIZING LOG_PX --------------------------------------------


    logits_true, log_px_true, probs_true, inputs_true = get_logpx(loader_true, model, "true")
    logits_flip, log_px_flip, probs_flip, inputs_flip = get_logpx(loader_flip, model, "flip")
    logits_shift, log_px_shift, probs_shift, inputs_shift = get_logpx(loader_shift, model, "shift")

    logits_cut, log_px_cut, probs_cut, inputs_cut = get_corrupt_results(loader_cut, loader_true, model, "Cut")
    logits_bo, log_px_bo, probs_bo, inputs_bo= get_corrupt_results(loader_bo, loader_true, model, "blackout")

    logits_pass, log_px_pass, probs_pass, inputs_pass = get_logpx(loader_pass, model, "pass")

    inputs_true_sparse = create_sparse_input(inputs_true)
    inputs_flip_sparse = create_sparse_input(inputs_flip)
    inputs_shift_sparse = create_sparse_input(inputs_shift)
    inputs_cut_sparse = create_sparse_input(inputs_cut)
    inputs_bo_sparse = create_sparse_input(inputs_bo)
    inputs_pass_sparse = create_sparse_input(inputs_pass)

    #with open("/Volumes/MNG/outlier_outputs/output_true.pcl", "wb") as file:
    #    pickle.dump([logits_true, log_px_true, probs_true, inputs_true_sparse], file)
#
    #with open("/Volumes/MNG/outlier_outputs/output_flip.pcl", "wb") as file:
    #    pickle.dump([logits_flip, log_px_flip, probs_flip, inputs_flip_sparse], file)
#
    #with open("/Volumes/MNG/outlier_outputs/output_shift.pcl", "wb") as file:
    #    pickle.dump([logits_shift, log_px_shift, probs_shift, inputs_shift_sparse], file)
#
    #with open("/Volumes/MNG/outlier_outputs/output_cut.pcl", "wb") as file:
    #    pickle.dump([logits_cut, log_px_cut, probs_cut, inputs_cut_sparse], file)
#
    #with open("/Volumes/MNG/outlier_outputs/output_bo.pcl", "wb") as file:
    #    pickle.dump([logits_bo, log_px_bo, probs_bo, inputs_bo_sparse], file)
#
    #with open("/Volumes/MNG/outlier_outputs/output_pass.pcl", "wb") as file:
    #    pickle.dump([logits_pass, log_px_pass, probs_pass, inputs_pass_sparse], file)


else:
    with open(args.data_path+"output_true.pcl", "rb") as file:
        logits_true, log_px_true, probs_true, inputs_true = pickle.load(file)

    with open(args.data_path + "output_flip.pcl", "rb") as file:
        logits_flip, log_px_flip, probs_flip, inputs_flip = pickle.load(file)

    with open(args.data_path + "output_shift.pcl", "rb") as file:
        logits_shift, log_px_shift, probs_shift, inputs_shift = pickle.load(file)

    with open(args.data_path + "output_cut.pcl", "rb") as file:
        logits_cut, log_px_cut, probs_cut, inputs_cut = pickle.load(file)

    with open(args.data_path + "output_bo.pcl", "rb") as file:
        logits_bo, log_px_bo, probs_bo, inputs_bo = pickle.load(file)

    with open(args.data_path + "output_pass.pcl", "rb") as file:
        logits_pass, log_px_pass, probs_pass, inputs_pass = pickle.load(file)



plt.figure()
sns.set(font_scale=1)
sns.distplot(log_px_true)
sns.distplot(log_px_cut)
sns.distplot(log_px_bo)
plt.xlim(-1000, 500)
plt.legend(["True", "Cut", "Blackout"], fontsize=10)
plt.xlabel(r"$\log p(x)$", fontsize=10)
plt.title("Density Plot of Log Probability of Input", fontsize=14)
if save_fig:
    plt.savefig(args.save_path+"distplot_corrupted.png")
plt.show()

plt.figure()
sns.set(font_scale=1)
sns.distplot(log_px_true)
sns.distplot(log_px_flip)
sns.distplot(log_px_shift)
plt.xlim(-2000, 500)
plt.legend(["True", "Flipped", "Shifted"], fontsize=10)
plt.xlabel(r"$\log p(x)$", fontsize=10)
plt.title("Density Plot of Log Probability of Input", fontsize=14)
if save_fig:
    plt.savefig(args.save_path + "distplot_anomalies.png")
plt.show()

plt.figure()
sns.distplot(log_px_true)
sns.distplot(log_px_pass)
plt.legend(["Cargo and Tanker", "Passenger"], fontsize=10)
plt.title("Density Plot of Log Probability of Input", fontsize=14)
plt.xlabel(r"$\log p(x)$", fontsize=10)
if save_fig:
    plt.savefig(args.save_path + "distplot_pass.png")
plt.xlim(-500, 100)
plt.show()

# Flag outliers
threshold = np.quantile(log_px_true, 0.05)

outliers_true = catch_outliers(log_px_true, threshold)
outliers_flip = catch_outliers(log_px_flip, threshold)
outliers_shift = catch_outliers(log_px_shift, threshold)

outliers_pass = catch_outliers(log_px_pass, threshold)


# Get maps with flagged outliers

with open(args.test_path, "rb") as file:
    dataset = pickle.load(file)

with open(args.pass_path, "rb") as file:
    dataset_pass = pickle.load(file)

routes_tagged_true = tag_route_for_plt(dataset, random.choices(outliers_true, k=30))
routes_tagged_pass = tag_route_for_plt(dataset_pass, random.choices(outliers_pass, k=45))

legend_lines = [Line2D([0], [0], color="b", lw=2),
                Line2D([0], [0], color="r", lw=2)]

get_anomaly_map(routes_tagged_true, dk2, swe, legend_lines, args.save_fig, "true", args.save_path)
get_anomaly_map(routes_tagged_pass, dk2, swe, legend_lines, args.save_fig, "pass", args.save_path)


lat_cols = np.round(Config.lat_columns[args.ROI], 1)
long_cols = np.round(Config.long_columns[args.ROI], 1)

get_example_inputs(inputs_true, inputs_flip, inputs_shift, long_cols, lat_cols, n_examples=2, save_fig=save_fig, save_path=args.save_path)

get_corrupt_recon(inputs_true, inputs_bo, logits_bo, probs_bo, long_cols, lat_cols, save_fig, args.save_path, n_examples=1, corr_name="bo")
get_corrupt_recon(inputs_true, inputs_cut, logits_cut, probs_cut, long_cols, lat_cols, save_fig, args.save_path, n_examples=1, corr_name="cut")

get_corrupt_recon(inputs_true, inputs_flip, logits_flip, probs_flip, long_cols, lat_cols, save_fig, args.save_path, n_examples=1, corr_name="flip")
get_corrupt_recon(inputs_true, inputs_shift, logits_shift, probs_shift, long_cols, lat_cols, save_fig, args.save_path, n_examples=1, corr_name="shift")



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
if save_fig:
    plt.savefig(args.save_path+"shifted_routes_map.png")
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
if save_fig:
    plt.savefig(args.save_path+"flipped_routes_map.png")
plt.show()


inputs_pass_plt = prep_route_for_plt(dataset_pass, "pass")

f, ax = plt.subplots()
dk2.plot(ax=ax, color="olivedrab")
swe.plot(ax=ax, color="olivedrab")
for i in range(len(inputs_pass_plt["lat"])):
    plt.plot(inputs_pass_plt["long"][i], inputs_pass_plt["lat"][i], "b", alpha =0.2)
plt.title("Passenger Ship Routes", fontsize=14)
plt.xlabel("Longitude", fontsize=10)
plt.ylabel("Latitude", fontsize=10)
plt.xlim(13, 17)
plt.ylim(54.5, 56.5)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
if save_fig:
    plt.savefig(args.save_path+"pass_routes_map.png")
plt.show()


inputs_true_plt = prep_route_for_plt(dataset, "true")

f, ax = plt.subplots()
dk2.plot(ax=ax, color="olivedrab")
swe.plot(ax=ax, color="olivedrab")
for i in range(100):
    plt.plot(inputs_true_plt["long"][i], inputs_true_plt["lat"][i], ".-b", alpha =0.5)
plt.title("Cargo and Tanker Ship Routes", fontsize=14)
plt.xlabel("Longitude", fontsize=10)
plt.ylabel("Latitude", fontsize=10)
plt.xlim(13, 17)
plt.ylim(54.5, 56.5)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
if save_fig:
    plt.savefig(args.save_path+"true_routes_map.png")
plt.show()


lengths_true, lengths_ano = get_lengths(dataset, outliers_true)

plt.figure()
sns.distplot(lengths_true)
sns.distplot(lengths_ano)
plt.legend(["Normal", "Anomalies"], fontsize=10)
plt.title("Distribution of Lengths of Paths \n Marked as Normal vs. Anomalous", fontsize=14)
plt.xlabel("Hours", fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
if save_fig:
    plt.savefig(args.save_path+"lengths_true.png")
plt.show()


print(f"True paths:")
print(f"--- max: {min(log_px_true):.3f}")
print(f"--- mean: {np.mean(log_px_true):.3f}")
print(f"--- std: {np.std(log_px_true):.3f}")
print("")
print(f"Cut paths:")
print(f"--- max: {min(log_px_cut):.3f}")
print(f"--- mean: {np.mean(log_px_cut):.3f}")
print(f"--- std: {np.std(log_px_cut):.3f}")
print(f"----t-stat: {calc_z(log_px_true, log_px_cut)}")
print("")
print(f"Flipped paths:")
print(f"--- max: {min(log_px_flip):.3f}")
print(f"--- mean: {np.mean(log_px_flip):.3f}")
print(f"--- std: {np.std(log_px_flip):.3f}")
print(f"----t-stat: {calc_z(log_px_true, log_px_flip)}")
print("")
print(f"Blackout paths:")
print(f"--- max: {min(log_px_bo):.3f}")
print(f"--- mean: {np.mean(log_px_bo):.3f}")
print(f"--- std: {np.std(log_px_bo):.3f}")
print(f"----t-stat: {calc_z(log_px_true, log_px_bo)}")
print("")
print(f"Shifted paths:")
print(f"--- max: {min(log_px_shift):.3f}")
print(f"--- mean: {np.mean(log_px_shift):.3f}")
print(f"--- std: {np.std(log_px_shift):.3f}")
print(f"----t-stat: {calc_z(log_px_true, log_px_shift)}")


l8, l12, l16, l20, l24 = id_length(dataset)

out_l8 = get_outlier_l(log_px_true, l8)
out_l12 = get_outlier_l(log_px_true, l12)
out_l16 = get_outlier_l(log_px_true, l16)
out_l20 = get_outlier_l(log_px_true, l20)
out_l24 = get_outlier_l(log_px_true, l24)

outliers_l = out_l8 + out_l12 + out_l16 + out_l20 + out_l24

routes_tagged_l = tag_route_for_plt(dataset, random.choices(outliers_l, k=30))

get_anomaly_map(routes_tagged_l, dk2, swe, legend_lines, True, "true", args.save_path)

lengths_l, lengths_ano_l = get_lengths(dataset, outliers_l)

plt.figure()
sns.distplot(lengths_l)
sns.distplot(lengths_ano_l)
plt.legend(["Normal", "Anomalies"], fontsize=10)
plt.title("Distribution of Lengths of Paths \n Marked as Normal vs. Anomalous", fontsize=14)
plt.xlabel("Hours", fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
#if save_fig:
plt.savefig(args.save_path+"lengths_l.png")
plt.show()


###
#mse_true, worst_im_batch_true = calc_mse(loader_true, model, "true")
#mse_cut, worst_im_batch_cut = calc_mse(loader_cut, model, "cut")
#mse_flip, worst_im_batch_flip = calc_mse(loader_flip, model, "flipped")
#mse_bo, worst_im_batch_bo = calc_mse(loader_bo, model, "blackout")
#
#
## Explore
#print(f"True paths:")
#print(f"--- max: {max(mse_true)}")
#print(f"--- mean: {np.mean(mse_true)}")
#print("")
#print(f"Cut paths:")
#print(f"--- max: {max(mse_cut)}")
#print(f"--- mean: {np.mean(mse_cut)}")
#print("")
#print(f"Flipped paths:")
#print(f"--- max: {max(mse_flip)}")
#print(f"--- mean: {np.mean(mse_flip)}")
#print("")
#print(f"Blackout paths:")
#print(f"--- max: {max(mse_bo)}")
#print(f"--- mean: {np.mean(mse_bo)}")
#
##Some visualizations
#
#it_true = iter(loader_true)
#inputs_true = next(it_true)
#
#it_cut = iter(loader_cut)
#inputs_cut = next(it_cut)
#
#it_flip = iter(loader_flip)
#inputs_flip = next(it_flip)
#
#it_bo = iter(loader_bo)
#inputs_bo = next(it_bo)
#
#it_shift = iter(loader_shift)
#inputs_shift = next(it_shift)
#
#it_pass = iter(loader_pass)
#inputs_pass = next(it_pass)
#
#show_worst(logits_cut, log_px_cut, probs_cut, inputs_cut, inputs_true)
#show_worst(logits_flip, log_px_flip, probs_flip, inputs_flip, inputs_true)
#show_worst(logits_bo, log_px_bo, probs_bo, inputs_bo, inputs_true)
#show_worst(logits_shift, log_px_shift, probs_shift, inputs_shift, inputs_true)
