import pandas as pd
import json
import numpy as np


def check_moored(df, threshold_moored):
    perc_still = len(df[df.SOG <= 10]) / len(df)
    moored = perc_still > threshold_moored

    return perc_still, moored

def check_ROI(df, lat_min, lat_max, long_min, long_max):
    _latmin, _latmax = min(df.Latitude / 600000), max(df.Latitude / 600000)

    _longmin, _longmax = min(df.Longitude / 600000), max(df.Longitude / 600000)

    not_in_ROI = _latmin < lat_min or _latmax > lat_max or _longmin < long_min or _longmax > long_max

    return not_in_ROI


def resample_interpolate(path, epoch_to_T0, freq = 10):
    df = pd.DataFrame(path, columns=["Time", "Latitude", "Longitude", "SOG", "COG", "Header"])

    df["Time"] = pd.to_datetime(df["Time"] + epoch_to_T0, unit="s")
    df = df.set_index('Time').resample(str(freq)+'T').mean()
    df = df.interpolate("linear")
    df = df.reset_index(level=0, inplace=False)

    return df


def window(iterable, size=2):
    i = iter(iterable)
    win = []
    for e in range(0, size):
        win.append(next(i))
    yield win
    for e in i:
        win = win[1:] + [e]
        yield win

def check_duration(json, threshold_AIS, threshold_min, threshold_max):
    short_traj = len(json["path"]) < threshold_AIS
    dur_short = json["path"][-1][0]-json["path"][0][0] < threshold_min
    dur_long = json["path"][-1][0]-json["path"][0][0] > threshold_max

    disc = any([short_traj, dur_short, dur_long])

    return disc



def split_and_collect_trajectories(files,
                                   category,
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
                                   freq):
    print(f"Processing files of type: {category}")
    data_split = []
    disc_short = 0
    disc_ROI = 0
    disc_moored = 0
    disc_dur_min = 0

    for i, file in enumerate(files):
        print(f"Preparing file: {i}")
        with open(file) as json_file:
            js = json.load(json_file)[0]

        res = [js["path"][i + 1][0] - js["path"][i][0] for i in range(len(js["path"]) - 1)]

        break_points = [ids + 1 for ids, diff in enumerate(res) if diff > max_interval]

        paths = np.split(js["path"], break_points)

        for path in paths:

            # Dismissing path if less than **threshold** messages
            if len(path) < threshold_trajlen:
                disc_short += 1
                continue

            # Resamping and interpolating
            df = resample_interpolate(path, epoch_to_T0)

            # Check and split path if longer than **threshold**
            idx = (np.arange(0, len(df) * freq * 60, threshold_dur_max) / (freq * 60)).astype(int)[1:]

            subpaths = np.split(df, idx)

            for subpath in subpaths:

                # dismissing path if shorter than **threshold** duration
                if (subpath["Time"].iloc[-1] - subpath["Time"].iloc[0]).total_seconds() < threshold_dur_min:
                    disc_dur_min += 1
                    continue

                not_in_ROI = check_ROI(subpath, lat_min, lat_max, long_min, long_max)
                perc_still, moored = check_moored(subpath, threshold_moored)

                journey_time = df["Time"].iloc[-1] - df["Time"].iloc[0]

                if not_in_ROI:
                    disc_ROI += 1
                    continue
                elif moored:
                    disc_moored += 1
                    continue
                else:
                    js1 = js.copy()
                    js1["path"] = df.to_dict("list")
                    data_split.append(js1)
    stats = {"disc_short": disc_short,
             "disc_ROI": disc_ROI,
             "disc_moored": disc_moored,
             "disc_dur_min": disc_dur_min}

    return data_split, stats