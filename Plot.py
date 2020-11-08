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

parser.add_argument('--save_fig', type = bool, default=False, help="Whether to save figures or not")

parser.add_argument('--no_traj', type = int, default=200, help="How many trajectories to plot")

## Calling parsed arguments ##
args = parser.parse_args()
save_fig = args.save_fig
no_traj = args.no_traj

path = Config.path

## Retrieving data ##
with open("Code2.0/local_files/data_split_cargo.pcl", "rb") as f:
    data_split_cargo = pickle.load(f)

with open("Code2.0/local_files/data_split_tank.pcl", "rb") as f:
    data_split_tank = pickle.load(f)

data = data_split_cargo + data_split_tank
random.shuffle(data)

crs = 'epsg:4326'

dk = gpd.read_file(path+"/Denmark_shapefile/dk_10km.shp")
dk = dk.to_crs(crs)

dk2 = gpd.read_file(path+"/gadm36_DNK_shp/gadm36_DNK_0.shp")
dk2 = dk2.to_crs(crs)

legend_lines = [Line2D([0], [0], color="blue", lw=2),
                Line2D([0], [0], color="green", lw=2)]


fig, ax = plt.subplots(figsize = (10,5))
dk.plot(ax = ax, alpha = 0.4, color = "grey")
dk2.plot(ax = ax, alpha = 0.4, color = "red")
for traj in split_tst[0]:
    geometry = [Point(xy) for xy in zip([item/600000 for item in traj["path"]["Longitude"]], [item/600000 for item in traj["path"]["Latitude"]])]
    t = gpd.GeoDataFrame(crs = crs, geometry = geometry)

    if traj["shiptype"] == "Cargo":
        c = "blue"
    elif traj["shiptype"] == "Tanker":
        c = "green"
    else:
        c = "purple"

    t.plot(ax = ax, markersize = 2, color = c)
plt.title(f"Map of Denmark with {no_traj} trajectories", fontsize = 15)
ax.legend(legend_lines, ["Cargo", "Tanker"])
if save_fig:
    plt.savefig(f"map{no_traj}.png")
plt.show()
