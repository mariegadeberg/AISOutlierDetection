import glob
import pickle
from Config import *
import argparse
from utils_preprocess import Preprocess
import sys

parser = argparse.ArgumentParser()

parser.add_argument("--save_data", type=str, default="True", help="Whether to save the data generated in the preprocessing")
parser.add_argument("--name", type=str, default="", help="Add subscript to saved data")
parser.add_argument("--months", type=str, default="all", help="Which months to include in data")

args = parser.parse_args()

path = Config.path
save_data = args.save_data
name = args.name
months_ = args.months

if save_data == "True":
    save_data = True
elif save_data == "False":
    save_data = False
else:
    print("Argument for saving data not recognized")


## Running preprocessing
if months_ == "all":
    months = ["2001", "2002", "2003"]
    #months = ["1906", "1907", "1908", "1909", "1910", "1911", "1912", "2001", "2002", "2003"]
else:
    months = ["1911"]

ship_types = ["Carg", "Tank"]
#ship_types = ["Pass"]

ds_train_bh = []
ds_val_bh = []
ds_test_bh = []

ds_train_sk = []
ds_val_sk = []
ds_test_sk = []

ds_train_blt = []
ds_val_blt = []
ds_test_blt = []

stat = {}

for month in months:
    for cat in ship_types:
        files = glob.glob(path + "aisMixJSON_" + month + "XX/" + cat + "*.json")

        train, val, test, stats = Preprocess(Config).split_and_collect_trajectories(files, month=month, category=cat)

        ds_train_bh += train["bh"]
        ds_val_bh += val["bh"]
        ds_test_bh += test["bh"]

        ds_train_sk += train["sk"]
        ds_val_sk += val["sk"]
        ds_test_sk += test["sk"]

        ds_train_blt += train["blt"]
        ds_val_blt += val["blt"]
        ds_test_blt += test["blt"]

        stat[month] = stats

print(f"Processing done. Size of dataset follows.")
print("Bornholm:")
print(f"-----Train: {len(ds_train_bh)} samples of total size {sys.getsizeof(ds_train_bh)/1000} MB")
print(f"-----Validation: {len(ds_val_bh)} samples of total size {sys.getsizeof(ds_val_bh)/1000} MB")
print(f"-----Test: {len(ds_test_bh)} samples of total size {sys.getsizeof(ds_test_bh)/1000} MB")
print("Skagen:")
print(f"-----Train: {len(ds_train_sk)} samples of total size {sys.getsizeof(ds_train_sk)/1000} MB")
print(f"-----Validation: {len(ds_val_sk)} samples of total size {sys.getsizeof(ds_val_sk)/1000} MB")
print(f"-----Test: {len(ds_test_sk)} samples of total size {sys.getsizeof(ds_test_sk)/1000} MB")
print("Baelterne:")
print(f"-----Train: {len(ds_train_blt)} samples of total size {sys.getsizeof(ds_train_blt)/1000} MB")
print(f"-----Validation: {len(ds_val_blt)} samples of total size {sys.getsizeof(ds_val_blt)/1000} MB")
print(f"-----Test: {len(ds_test_blt)} samples of total size {sys.getsizeof(ds_test_blt)/1000} MB")


if save_data:
    print("Saving data...")
    with open(path + "data/train_bh_"+name+".pcl", "wb") as f:
        pickle.dump(ds_train_bh, f)
    with open(path + "data/train_sk_"+name+".pcl", "wb") as f:
        pickle.dump(ds_train_sk, f)
    with open(path + "data/train_blt_"+name+".pcl", "wb") as f:
        pickle.dump(ds_train_blt, f)

    with open(path + "data/val_bh_"+name+".pcl", "wb") as f:
        pickle.dump(ds_val_bh, f)
    with open(path + "data/val_sk_"+name+".pcl", "wb") as f:
        pickle.dump(ds_val_sk, f)
    with open(path + "data/val_blt_"+name+".pcl", "wb") as f:
        pickle.dump(ds_val_blt, f)

    with open(path + "data/test_bh_"+name+".pcl", "wb") as f:
        pickle.dump(ds_test_bh, f)
    with open(path + "data/test_sk_"+name+".pcl", "wb") as f:
        pickle.dump(ds_test_sk, f)
    with open(path + "data/test_blt_"+name+".pcl", "wb") as f:
        pickle.dump(ds_test_blt, f)

    with open(path + "data/stats"+name+".pcl", "wb") as f:
        pickle.dump(stat, f)
    print("Done!")


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