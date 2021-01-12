import numpy as np
from Config import *
import matplotlib.pyplot as plt
import seaborn as sns


def calc_mse(loader, model, name):
    print(f"Starting calculations for loader: {name}")
    mse = []
    worst_img_batch = {"real": [],
                       "recon": []}
    for inputs in loader:
        inputs = inputs.unsqueeze(1)
        recon = model.sample(inputs).detach().numpy().squeeze()

        recon_prob = np.exp(recon) / (1 + np.exp(recon))

        recon_prob_norm = np.zeros([inputs.size(0), len(Config.lat_columns["bh"]), len(Config.long_columns["bh"])])
        for i in range(inputs.size(0)):
            recon_prob_norm[i, :, :] = (recon_prob[i, :, :] - recon_prob[i, :, :].min()) / (
                        recon_prob[i, :, :].max() - recon_prob[i, :, :].min())

        mse_tmp = []
        for i in range(len(inputs)):
            m = ((inputs.squeeze()[i, :, :] - recon_prob_norm[i, :, :]) ** 2).mean(axis=0).mean().item()
            mse.append(m)
            mse_tmp.append(m)

        idx = np.argmax(mse_tmp)

        worst_img_batch["real"].append(inputs.squeeze()[idx, :, :])
        worst_img_batch["recon"].append(recon_prob_norm[idx, :, :])

    return mse, worst_img_batch


def get_logpx(loader, model, name):
    print(f"Starting calculations for {name}")

    log_px_out = []
    logits_out = []
    probs_out = []
    inputs_out = []

    for inputs in loader:
        inputs = inputs.unsqueeze(1)

        logits, log_px = model.sample(inputs)
        logits = logits.detach().numpy()
        log_px = log_px.detach().numpy()

        probs = np.exp(logits) / (1 + np.exp(logits))

        logits_out.append(logits)
        log_px_out.append(log_px)
        probs_out.append(probs)

        inputs_out.append(inputs.squeeze())


    logits_out = [item for sublist in logits_out for item in sublist]
    log_px_out = [item for sublist in log_px_out for item in sublist]
    probs_out = [item for sublist in probs_out for item in sublist]
    inputs_out = [item for sublist in inputs_out for item in sublist]

    return logits_out, log_px_out, probs_out, inputs_out


def get_corrupt_results(loader_corrupt, loader_true, model, name):
    print(f"starting calculation for {name}")

    logits_c = []
    log_px_c = []
    probs_c = []
    inputs_out = []

    for inputs_true, inputs_corrupt in zip(loader_true, loader_corrupt):
        inputs_true = inputs_true.unsqueeze(1)
        inputs_corrupt = inputs_corrupt.unsqueeze(1)

        logits, log_px = model.sample_corrupt(inputs_corrupt, inputs_true)
        logits = logits.detach().numpy()
        log_px = log_px.detach().numpy()

        probs = np.exp(logits) / (1 + np.exp(logits))

        logits_c.append(logits)
        log_px_c.append(log_px)
        probs_c.append(probs)
        inputs_out.append(inputs_corrupt)

    logits_c = [item for sublist in logits_c for item in sublist]
    log_px_c = [item for sublist in log_px_c for item in sublist]
    probs_c = [item for sublist in probs_c for item in sublist]
    inputs_out = [item for sublist in inputs_out for item in sublist]

    return logits_c, log_px_c, probs_c, inputs_out


def show_worst(logits, log_px, probs, inputs, inputs_true):
    worst_idx = np.argmin(log_px)

    plt.figure()
    plt.subplot(2, 2, 1)
    sns.heatmap(inputs_true[worst_idx])
    plt.title("True Path")

    plt.subplot(2, 2, 2)
    sns.heatmap(inputs[worst_idx])
    plt.title("Corrupted path")

    plt.subplot(2,2, 3)
    sns.heatmap(logits[worst_idx].squeeze())
    plt.title("Logits")

    plt.subplot(2, 2, 4)
    sns.heatmap(probs[worst_idx].squeeze())
    plt.title("Probabilities")

    plt.show()



def catch_outliers(log_px, threshold):
    outliers = np.where(log_px < threshold)
    return outliers[0]


def tag_route_for_plt(dataset, idx_ano):
    output = {"lat": [],
              "long": [],
              "anomaly": []}

    i = 0
    for path in dataset:

        tst = path["FourHot"].todense()

        lat, long, sog, cog = np.split(tst, Config.breaks["bh"], axis=1)

        lat_columns = Config.lat_columns["bh"]
        long_columns = Config.long_columns["bh"]

        idx_lat = np.where(lat > 0)[1]
        idx_long = np.where(long > 0)[1]

        lat_c = lat_columns[idx_lat]
        long_c = long_columns[idx_long]

        if lat_c.shape != long_c.shape:
            continue

        output["lat"].append(lat_c)
        output["long"].append(long_c)

        if i in idx_ano:
            output["anomaly"].append(1)
        else:
            output["anomaly"].append(0)

        i += 1

    return output


def prep_route_for_plt(dataset, kind):
    output = {"lat": [],
              "long": [],
              "kind": []}

    i = 0
    for path in dataset:

        tst = path["FourHot"].todense()

        lat, long, sog, cog = np.split(tst, Config.breaks["bh"], axis=1)

        lat_columns = Config.lat_columns["bh"]
        long_columns = Config.long_columns["bh"]

        idx_lat = np.where(lat > 0)[1]
        idx_long = np.where(long > 0)[1]

        lat_c = lat_columns[idx_lat]
        long_c = long_columns[idx_long]

        if lat_c.shape != long_c.shape:
            continue

        output["lat"].append(lat_c)
        output["long"].append(long_c)

        output["kind"].append(kind)

        i += 1

    return output


def prep_route_for_pltShift(dataset, kind):
    output = {"lat": [],
              "long": [],
              "kind": []}

    i = 0
    for path in dataset:

        tst = path["FourHot"].todense()

        lat, long, sog, cog = np.split(tst, Config.breaks["bh"], axis=1)

        lat = shift_lat(lat)

        lat_columns = Config.lat_columns["bh"]
        long_columns = Config.long_columns["bh"]

        idx_lat = np.where(lat > 0)[1]
        idx_long = np.where(long > 0)[1]

        lat_c = lat_columns[idx_lat]
        long_c = long_columns[idx_long]

        if lat_c.shape != long_c.shape:
            continue

        output["lat"].append(lat_c)
        output["long"].append(long_c)

        output["kind"].append(kind)

        i += 1

    return output


def shift_lat(lat, shift=5):
    lat_tst = lat.copy()
    present = lat_tst.sum(axis=0)
    shift_feasible = (present[0, shift] == 0).all()
    if shift_feasible:
        s = np.zeros([lat_tst.shape[0], shift])
        new_lat = np.concatenate((s, lat_tst), axis=1)[:, :-shift]
    elif (present[0, 0:shift] == 0).all():
        s = np.zeros([lat_tst.shape[0], shift])
        new_lat = np.concatenate((lat_tst, s), axis=1)[:, shift:]
    else:
        new_lat = lat
    return new_lat


def prep_route_for_pltFlipped(dataset, kind):
    output = {"lat": [],
              "long": [],
              "kind": []}

    i = 0
    for path in dataset:

        tst = path["FourHot"].todense()

        lat, long, sog, cog = np.split(tst, Config.breaks["bh"], axis=1)

        lat_columns = Config.lat_columns["bh"]
        long_columns = Config.long_columns["bh"]

        idx_lat = np.where(lat > 0)[1]
        idx_long = np.where(long > 0)[1]

        lat_c = lat_columns[idx_lat]
        long_c = long_columns[idx_long]

        if lat_c.shape != long_c.shape:
            continue

        output["lat"].append(lat_c[::-1])
        output["long"].append(long_c)

        output["kind"].append(kind)

        i += 1

    return output