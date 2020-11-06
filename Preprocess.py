import glob
import pickle
from Config import *
from utils_preprocess import *
import argparse
from Class_test import Preprocess

parser = argparse.ArgumentParser()

parser.add_argument("--save_data", type=bool, default=True, help="Whether to save the data generated in the preprocessing")

args = parser.parse_args()

## Defining constants from config file

path = Config.path
save_data = args.save_data

## Running preprocessing

#cargo_files = glob.glob(path + "/Data/aisMixJSONX_1912XX/Carg*.json")
tank_files = glob.glob(path + "/Data/aisMixJSONX_1912XX/Tank*.json")

data_split, stats = Preprocess(Config).split_and_collect_trajectories(tank_files, "Tank")

if save_data:
    #with open("Code/data_split_cargo.pcl", "wb") as f:
    #    pickle.dump(data_split_cargo, f)
    #with open("Code/stats_cargo.pcl", "wb") as s:
    #    pickle.dump(stats_cargo, s)

    with open("Code2.0/local_files/data_split_tank.pcl", "wb") as f:
        pickle.dump(data_split, f)
    with open("Code2.0/local_files/stats_tank.pcl", "wb") as s:
        pickle.dump(stats, s)


#### EXPLORE ####

#with open("Code/data_split_cargo.pcl", "rb") as f:
#    data_split_cargo = pickle.load(f)
#
#with open("Code/data_split_tank.pcl", "rb") as f:
#    data_split_tank = pickle.load(f)
#
#with open("Code/stats_cargo.pcl", "rb") as f:
#    stats_cargo = pickle.load(f)
#
#with open("Code/stats_tank.pcl", "rb") as f:
#    stats_tank = pickle.load(f)
#



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