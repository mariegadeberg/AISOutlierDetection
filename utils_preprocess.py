import pandas as pd
import json
import numpy as np

class Preprocess:
    def __init__(self, Config):

        self.max_interval = Config.max_interval
        self.epoch_to_T0 = Config.epoch_to_T0

        self.threshold_trajlen = Config.threshold_trajlen
        self.threshold_dur_min = Config.threshold_dur_min
        self.threshold_dur_max = Config.threshold_dur_max
        self.threshold_moored = Config.threshold_moored

        self.freq = Config.freq

        self.lat_min = Config.lat_min
        self.lat_max = Config.lat_max
        self.long_min = Config.long_min
        self.long_max = Config.long_max


    def check_moored(self, df):
        perc_still = len(df[df.SOG <= 10]) / len(df)
        moored = perc_still > self.threshold_moored

        return perc_still, moored

    def check_ROI(self, df):
        _latmin, _latmax = min(df.Latitude / 600000), max(df.Latitude / 600000)

        _longmin, _longmax = min(df.Longitude / 600000), max(df.Longitude / 600000)

        not_in_ROI = _latmin < self.lat_min or _latmax > self.lat_max or _longmin < self.long_min or _longmax > self.long_max

        return not_in_ROI

    def resample_interpolate(self, path):
        df = pd.DataFrame(path, columns=["Time", "Latitude", "Longitude", "SOG", "COG", "Header"])

        df["Time"] = pd.to_datetime(df["Time"] + self.epoch_to_T0, unit="s")
        df = df.set_index('Time').resample(str(self.freq) + 'T').mean()
        df = df.interpolate("linear")
        df = df.reset_index(level=0, inplace=False)

        return df

    def split_and_collect_trajectories(self, files, category):
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

            break_points = [ids + 1 for ids, diff in enumerate(res) if diff > self.max_interval]

            paths = np.split(js["path"], break_points)

            for path in paths:

                # Dismissing path if less than **threshold** messages
                if len(path) < self.threshold_trajlen:
                    disc_short += 1
                    continue

                # Resamping and interpolating
                df = self.resample_interpolate(path)

                # Check and split path if longer than **threshold**
                idx = (np.arange(0, len(df) * self.freq * 60, self.threshold_dur_max) / (self.freq * 60)).astype(int)[1:]

                subpaths = np.split(df, idx)

                for subpath in subpaths:

                    # dismissing path if shorter than **threshold** duration
                    if (subpath["Time"].iloc[-1] - subpath["Time"].iloc[0]).total_seconds() < self.threshold_dur_min:
                        disc_dur_min += 1
                        continue

                    not_in_ROI = self.check_ROI(subpath)
                    perc_still, moored = self.check_moored(subpath)

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
                        js1["journey_time"] = journey_time
                        data_split.append(js1)
        stats = {"disc_short": disc_short,
                 "disc_ROI": disc_ROI,
                 "disc_moored": disc_moored,
                 "disc_dur_min": disc_dur_min}

        return data_split, stats



