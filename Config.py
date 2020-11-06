import os

class Config(object):
    '''
    Configuration file for the project
    '''

    ## Setting path for project
    os.chdir('/Users/mariegadeberg/OneDrive/Uni/Speciale/AIS')
    path = os.getcwd()

    ## Defining Region of Interest
    lat_min = 53
    lat_max = 58
    long_min = 1
    long_max = 17

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