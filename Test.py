



### OLD ####

for i, file in enumerate(files):
    print(f"Preparing file: {i}")
    with open(file) as json_file:
        js = json.load(json_file)[0]

    res = [js["path"][i + 1][0] - js["path"][i][0] for i in range(len(js["path"]) - 1)]

    break_points = [ids + 1 for ids, diff in enumerate(res) if diff > max_interval]
    # break_points = [0] + [break_point + 1 for break_point in break_points] + [None]

    paths = np.split(js["path"], break_points)

    intervals = window(break_points)

    for idx, val in enumerate(intervals):
        start, stop = val[0], val[1]

        js1 = js.copy()
        js1["path"] = js1["path"][start:stop]

        js1["diff_AIS"] = [js1["path"][i + 1][0] - js1["path"][i][0] for i in range(len(js1["path"]) - 1)]

        df = resample_interpolate(js1, epoch_to_T0)
        js1["journey_time"] = df["Time"].iloc[-1] - df["Time"].iloc[0]

        short_traj = check_duration(js1, threshold_trajlen, threshold_dur_min, threshold_dur_max)
        not_in_ROI = check_ROI(df, lat_min, lat_max, long_min, long_max)
        perc_still, moored = check_moored(df, threshold_moored)

        if short_traj:
            print(f"Duration of trajectory out of scope - discarding entries {start}:{stop}")
            disc_short += 1
        elif not_in_ROI:
            print(f"Trajectory outside ROI - discarding entries {start}:{stop}")
            disc_ROI += 1
        elif moored:
            print(f"Speed under 0.1 knots for {perc_still * 100}% of trajectory - discarding entries {start}:{stop}")
            disc_moored += 1
        else:
            js1["path"] = df.to_dict("list")
            data_split.append(js1)
stats = {"disc_short": disc_short,
         "disc_ROI": disc_ROI,
         "disc_moored": disc_moored}