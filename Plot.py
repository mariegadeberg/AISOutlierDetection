import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import descartes
from shapely.geometry import Point, Polygon
import os
import argparse
import pickle
import random
from matplotlib.lines import Line2D
from Config import *

## Creating parser arguments ##

parser = argparse.ArgumentParser(description="Parses command.")
parser.add_argument('--save_fig', type = str, default="False", help="Whether to save figures or not")
parser.add_argument('--no_traj', type = int, default=200, help="How many trajectories to plot")
parser.add_argument('--data_blt', type=str, default="data/train_blt_.pcl", help="path to data")
parser.add_argument('--data_bh', type=str, default="data/train_bh_.pcl", help="path to data")
parser.add_argument('--data_sk', type=str, default="data/train_sk_.pcl", help="path to data")

## Calling parsed arguments ##
args = parser.parse_args()
save_fig = args.save_fig
if save_fig == "True":
    save_fig = True
else:
    save_fig = False
no_traj = args.no_traj
data_blt = args.data_blt
data_bh = args.data_bh
data_sk = args.data_sk

path = Config.path
## Retrieving data ##
with open(path+data_blt, "rb") as f:
    data_blt = pickle.load(f)

with open(path+data_bh, "rb") as f:
    data_bh = pickle.load(f)

with open(path+data_sk, "rb") as f:
    data_sk = pickle.load(f)

#random.shuffle(data)

crs = 'epsg:4326'

#dk = gpd.read_file(path+"/Denmark_shapefile/dk_10km.shp")
#dk = dk.to_crs(crs)

dk2 = gpd.read_file(path+"/gadm36_DNK_shp/gadm36_DNK_0.shp")
dk2 = dk2.to_crs(crs)

legend_lines = [Line2D([0], [0], color="blue", lw=2),
                Line2D([0], [0], color="green", lw=2),
                Line2D([0], [0], color="red", lw=2)]


fig, ax = plt.subplots(figsize = (10,5))
#dk.plot(ax = ax, alpha = 0.4, color = "grey")
dk2.plot(ax = ax, alpha = 0.4, color = "red")
for traj in data_blt[0:no_traj]:
    geometry = [Point(xy) for xy in zip([item/600000 for item in traj["path"]["Longitude"]], [item/600000 for item in traj["path"]["Latitude"]])]
    t = gpd.GeoDataFrame(crs = crs, geometry = geometry)

    t.plot(ax = ax, markersize = 1, color = "blue")
for traj in data_bh[0:no_traj]:
    geometry = [Point(xy) for xy in zip([item/600000 for item in traj["path"]["Longitude"]], [item/600000 for item in traj["path"]["Latitude"]])]
    t = gpd.GeoDataFrame(crs = crs, geometry = geometry)

    t.plot(ax = ax, markersize = 1, color = "green")
for traj in data_sk[0:no_traj]:
    geometry = [Point(xy) for xy in zip([item/600000 for item in traj["path"]["Longitude"]], [item/600000 for item in traj["path"]["Latitude"]])]
    t = gpd.GeoDataFrame(crs = crs, geometry = geometry)

    t.plot(ax = ax, markersize = 1, color = "red")
plt.title(f"Map of Denmark with {no_traj} trajectories", fontsize = 15)
ax.legend(legend_lines, ["Baelterne", "Bornholm", "Skagen"])
if save_fig:
    plt.savefig(f"map{no_traj}.png")
plt.show()
