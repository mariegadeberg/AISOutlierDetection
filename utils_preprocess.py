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

        self.ROI_boundary_E = Config.ROI_boundary_E
        self.ROI_boundary_W = Config.ROI_boundary_W

        self.max_knot = Config.max_knot
        self.sog_res = Config.sog_res
        self.sog_columns = Config.sog_columns

        self.cog_res = Config.cog_res
        self.cog_columns = Config.cog_columns

        self.lat_long_res = Config.lat_long_res
        self.lat_columns = Config.lat_columns
        self.long_columns = Config.long_columns


    def check_moored(self, df):
        perc_still = len(df[df.SOG <= 10]) / len(df)
        moored = perc_still > self.threshold_moored

        return perc_still, moored

    def check_ROI(self, df):
        _latmin, _latmax = min(df.Latitude / 600000), max(df.Latitude / 600000)

        _longmin, _longmax = min(df.Longitude / 600000), max(df.Longitude / 600000)

        not_in_ROI = _latmin < self.lat_min or _latmax > self.lat_max or _longmin < self.long_min or _longmax > self.long_max

        return not_in_ROI

    def resample_interpolate(self, df):

        df["Time"] = pd.to_datetime(df["Time"] + self.epoch_to_T0, unit="s")
        df = df.set_index('Time').resample(str(self.freq) + 'T').mean()
        df = df.interpolate("linear")
        df = df.reset_index(level=0, inplace=False)

        return df

    def split_ROI(self, path):
        df = pd.DataFrame(path, columns=["Time", "Latitude", "Longitude", "SOG", "COG", "Header"])

        out_east = np.array([1 if val / 600000 < self.long_min else 0 for val in df.Longitude])
        eastern_ROI = np.array([1 if val / 600000 < self.ROI_boundary_E else 0 for val in df.Longitude])
        western_ROI = np.array([1 if val / 600000 > self.ROI_boundary_W else 0 for val in df.Longitude])

        breaks = np.concatenate((np.where(western_ROI[:-1] != western_ROI[1:])[0] + 1,
                                 np.where(eastern_ROI[:-1] != eastern_ROI[1:])[0] + 1,
                                 np.where(out_east[:-1] != out_east[1:])[0] + 1), axis=0)

        subpaths = np.array_split(df, np.sort(breaks))

        return subpaths

    def create_bins(self, feature_array, resolution, scaling_factor, upperbound=None, lowerbound=None):
        if upperbound:
            feature_array = [upperbound if coord > upperbound else coord for coord in feature_array]
        if lowerbound:
            feature_array = [lowerbound if coord < lowerbound else coord for coord in feature_array]

        feature_array = [np.round(coord / (resolution * scaling_factor)) * resolution for coord in feature_array]

        return feature_array

    def one_hot_and_fill(self, feature_array, columns):
        dummies = pd.get_dummies(feature_array)

        for col in columns:
            if col not in dummies.columns:
                dummies[col] = 0

        dummies = dummies.reindex(sorted(dummies.columns), axis=1)

        return dummies

    def four_hot_encode(self, path):
        sog = self.create_bins(path["path"]["SOG"], self.sog_res, scaling_factor=10, upperbound=self.max_knot)
        sog = self.one_hot_and_fill(sog, self.sog_columns)

        cog = self.create_bins(path["path"]["COG"], self.cog_res, scaling_factor=10)
        cog = self.one_hot_and_fill(cog, self.cog_columns)

        lat = self.create_bins(path["path"]["Latitude"], self.lat_long_res, scaling_factor=600000, upperbound=self.lat_max,
                          lowerbound=self..lat_min)
        lat = self.one_hot_and_fill(lat, self.lat_columns)

        long = self.create_bins(path["path"]["Latitude"], self.lat_long_res, scaling_factor=600000, upperbound=self.long_max,
                           lowerbound=self.long_min)
        long = self.one_hot_and_fill(long, self.long_columns)

        data = np.concatenate((lat, long, sog, cog), axis=1)

        return data

    def split_and_collect_trajectories(self, files, category):
        print(f"Processing files of type: {category}")
        data_split = []
        disc_short = 0
        disc_ROI = 0
        disc_moored = 0
        disc_dur_min = 0


        for i, file in enumerate(files):
            if i % 100 == 0:
                print(f"Preparing file {i}/{len(files)}")
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

                subpaths = self.split_ROI(path)

                for subpath in subpaths:

                    if any(subpath.Longitude < self.long_min * 600000):
                        disc_ROI += 1
                        continue

                    if any(subpath.Longitude > self.ROI_boundary_W * 600000):
                        disc_ROI += 1
                        continue

                    if len(subpath) < self.threshold_trajlen:
                        disc_short += 1
                        continue

                    if all(subpath.Longitude < self.ROI_boundary_E * 600000):
                        eastern = 1
                    else:
                        eastern = 0

                    # Resampling and interpolating
                    df = self.resample_interpolate(subpath)

                    # Check and split path if longer than **threshold**
                    idx = (np.arange(0, len(df) * self.freq * 60, self.threshold_dur_max) / (self.freq * 60)).astype(int)[1:]

                    subsubpaths = np.split(df, idx)

                    for subsubpath in subsubpaths:

                        # dismissing path if shorter than **threshold** duration
                        if (subsubpath["Time"].iloc[-1] - subsubpath["Time"].iloc[0]).total_seconds() < self.threshold_dur_min:
                            disc_dur_min += 1
                            continue

                        perc_still, moored = self.check_moored(subsubpath)

                        if moored:
                            disc_moored += 1
                            continue
                        else:
                            js1 = js.copy()
                            df1 = subsubpath.copy()
                            df1["Time"] = df1["Time"].astype(str)
                            js1["path"] = df1.to_dict("list")
                            js1["eastern"] = eastern
                            js1["FourHot"] = self.four_hot_encode(js1["path"])
                            data_split.append(js1)

        stats = {"disc_short": disc_short,
                 "disc_ROI": disc_ROI,
                 "disc_moored": disc_moored,
                 "disc_dur_min": disc_dur_min}

        return data_split, stats



