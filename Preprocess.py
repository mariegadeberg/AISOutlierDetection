import glob
import pickle
from Config import *
from utils_preprocess import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--save_data", type=bool, default=True, help="Whether to save the data generated in the preprocessing")

args = parser.parse_known_args()

## Defining constants from config file

path = Config.path
save_data = args.save_data

max_interval = Config.max_interval
epoch_to_T0 = Config.epoch_to_T0

threshold_trajlen = Config.threshold_trajlen
threshold_dur_min = Config.threshold_dur_min
threshold_dur_max = Config.threshold_dur_max
threshold_moored = Config.threshold_moored

freq = Config.freq

lat_min = Config.lat_min
lat_max = Config.lat_max
long_min = Config.long_min
long_max = Config.long_max

## Running preprocessing

#cargo_files = glob.glob(path + "/Data/aisMixJSONX_1912XX/Carg*.json")
tank_files = glob.glob(path + "/Data/aisMixJSONX_1912XX/Tank*.json")

#data_split_cargo, stats_cargo = split_and_collect_trajectories(cargo_files,
#                                                                "Cargo",
#                                                                max_interval,
#                                                                epoch_to_T0,
#                                                                threshold_trajlen,
#                                                                lat_min,
#                                                                lat_max,
#                                                                long_min,
#                                                                long_max,
#                                                                threshold_moored,
#                                                                threshold_dur_min,
#                                                                threshold_dur_max)

data_split_tank, stats_tank = split_and_collect_trajectories(tank_files,
                                                                "Tank",
                                                                max_interval,
                                                                epoch_to_T0,
                                                                threshold_trajlen,
                                                                lat_min,
                                                                lat_max,
                                                                long_min,
                                                                long_max,
                                                                threshold_moored,
                                                                threshold_dur_min,
                                                                threshold_dur_max,
                                                                freq)

if save_data:
    with open("Code/data_split_cargo.pcl", "wb") as f:
        pickle.dump(data_split_cargo, f)
    with open("Code/stats_cargo.pcl", "wb") as s:
        pickle.dump(stats_cargo, s)

    with open("Code/data_split_tank.pcl", "wb") as f:
        pickle.dump(data_split_tank, f)
    with open("Code/stats_tank.pcl", "wb") as s:
        pickle.dump(stats_tank, s)


#### EXPLORE ####

with open("Code/data_split_cargo.pcl", "rb") as f:
    data_split_cargo = pickle.load(f)

with open("Code/data_split_tank.pcl", "rb") as f:
    data_split_tank = pickle.load(f)

with open("Code/stats_cargo.pcl", "rb") as f:
    stats_cargo = pickle.load(f)

with open("Code/stats_tank.pcl", "rb") as f:
    stats_tank = pickle.load(f)




'''

## Explore data ###

max_diff = []
long_breaks = []
lengths = []
for t in data_split:
    try:
        max_diff.append(max(t["diff_AIS"]))
        long_breaks.append(len([i for i in t["diff_AIS"] if i > 3600]))
        lengths.append(len(t["path"]["Time"]))
    except:
        pass


plt.figure()
plt.hist(lengths, bins = 100)
plt.title("Histogram of lengths of paths after interpolation")
plt.show()

plt.figure()
plt.hist(max_diff, bins = 100)
plt.title("Histogram of largest interval between messages in path")
plt.show()

plt.figure()
plt.hist(long_breaks, bins = 100)
plt.title("Histogram of number of messages with more than 1 hour interval in path")
plt.show()

journey_time = [item["journey_time"] for item in data_split]
journey_time = pd.Series(journey_time)


## OLD ##

for file in example_lst:
    print(f"Preparing file: {file}")
    with open(path+file) as json_file:
        js = json.load(json_file)[0]

    time_moored = []

    try:
        time_moored = [int(time) for time, status in js["statushist"].items() if status == "Moored"]
    except:
        pass

    for time in time_moored:
        index_moored = [idx for idx, path in enumerate(js["path"]) if path[0] == time]

    for idx, val in enumerate(index_moored):
        if idx == 0:
            js1 = js.copy()
            js1["path"] = js1["path"][:index_moored[idx] - 1]
        elif idx == len(index_moored)-1:
            js1 = js.copy()
            js1["path"] = js1["path"][index_moored[idx]:]
        else:
            js1 = js.copy()
            js1["path"] = js1["path"][index_moored[idx-1]:index_moored[idx]-1]

        #print(len(js1["path"]))
        data_split.append(js1)

'''