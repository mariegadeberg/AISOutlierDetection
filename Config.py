import os
import numpy as np
import pandas as pd

class Config(object):
    '''
    Configuration file for the project
    '''

    ## Setting path for project
    #os.chdir('/Users/mariegadeberg/OneDrive/Uni/Speciale/AIS')
    #path = os.getcwd()
    path = "/Volumes/MNG/"

    train_cuttime = pd.Timestamp(2020, 3, 10)
    val_cuttime = pd.Timestamp(2020, 3, 20)


    ## Defining Region of Interest
    lat_min = 54.5
    lat_max = 58.5
    long_min = 9
    long_max = 17

    ROI_boundary_long = 13
    ROI_boundary_lat = 56.5

    ## Set the maximum interval between AIS messages before splitting trajectory
    max_interval = 60 * 60 * 1 # 60*60*1 = 1h

    ## Set the maximum percentage of time with SOG < 0.1 knots before defined as moored or anchored
    threshold_moored = 0.8

    ## Set thresholds for durations
    threshold_trajlen = 20
    threshold_dur_min = 4 *60 * 60 # 4*60*60
    threshold_dur_max = 24*60*60 #24*60*60


    ## Time in seconds from epoch to T0 of dataset
    epoch_to_T0 = 1546297200

    freq = 10

    max_knot = 30
    sog_res = 1
    sog_columns = range(0, int(max_knot + sog_res), sog_res)

    cog_res = 5
    cog_columns = range(0, 365, cog_res)

    lat_long_res = 0.01

    lat_columns = {"bh": np.arange(lat_min, ROI_boundary_lat + lat_long_res, lat_long_res),
                   "sk": np.arange(ROI_boundary_lat, lat_max + lat_long_res, lat_long_res),
                   "blt": np.arange(lat_min, ROI_boundary_lat + lat_long_res, lat_long_res)}

    long_columns = {"bh": np.arange(ROI_boundary_long, long_max + lat_long_res, lat_long_res),
                    "sk": np.arange(long_min, ROI_boundary_long + lat_long_res, lat_long_res),
                    "blt": np.arange(long_min, ROI_boundary_long + lat_long_res, lat_long_res)}

    breaks = {"bh": (len(lat_columns["bh"]), len(lat_columns["bh"]) + len(long_columns["bh"]), len(lat_columns["bh"]) + len(long_columns["bh"]) + len(sog_columns)),
              "sk": (len(lat_columns["sk"]), len(lat_columns["sk"]) + len(long_columns["sk"]), len(lat_columns["sk"]) + len(long_columns["sk"]) + len(sog_columns)),
              "blt": (len(lat_columns["blt"]), len(lat_columns["blt"]) + len(long_columns["blt"]), len(lat_columns["blt"]) + len(long_columns["blt"]) + len(sog_columns))}


    #total_bins = len(sog_columns) + len(cog_columns) + len(lat_columns) + len(long_columns)

    # -------------------- Used during training ----------------------#

    input_shape = {"bh": len(sog_columns) + len(cog_columns) + len(lat_columns["bh"]) + len(long_columns["bh"]),
                   "sk": len(sog_columns) + len(cog_columns) + len(lat_columns["sk"]) + len(long_columns["sk"]),
                   "blt": len(sog_columns) + len(cog_columns) + len(lat_columns["blt"]) + len(long_columns["blt"])}

    latent_shape = 100

    lr = 0.0003

    splits = {"bh": (len(lat_columns["bh"]), len(long_columns["bh"]), len(sog_columns), len(cog_columns)),
              "sk": (len(lat_columns["sk"]), len(long_columns["sk"]), len(sog_columns), len(cog_columns)),
              "blt": (len(lat_columns["blt"]), len(long_columns["blt"]), len(sog_columns), len(cog_columns))}
