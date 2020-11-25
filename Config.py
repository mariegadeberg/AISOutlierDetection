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
    lat_min = 53
    lat_max = 58
    long_min = 4
    long_max = 17

    ROI_boundary_E = 9.5
    ROI_boundary_W = 13

    ## Set the maximum interval between AIS messages before splitting trajectory
    max_interval = 60 * 60 * 4

    ## Set the maximum percentage of time with SOG < 0.1 knots before defined as moored or anchored
    threshold_moored = 0.8

    ## Set thresholds for durations
    threshold_trajlen = 20
    threshold_dur_min = 4*60*60
    threshold_dur_max = 24*60*60


    ## Time in seconds from epoch to T0 of dataset
    epoch_to_T0 = 1546297200

    freq = 10

    max_knot = 30
    sog_res = 1
    sog_columns = range(0, int(max_knot + sog_res), sog_res)

    cog_res = 5
    cog_columns = range(0, 365, cog_res)

    lat_long_res = 0.01
    lat_columns = np.arange(lat_min, lat_max + lat_long_res, lat_long_res)
    long_columns = np.arange(long_min, long_max + lat_long_res, lat_long_res)

    # -------------------- Used during training ----------------------#

    input_shape = 1907
    latent_shape = 100