import numpy as np
from Config import *
import matplotlib.pyplot as plt
import seaborn as sns
import random
random.seed(99)
from scipy import sparse
import torch


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
        inputs_out.append(inputs_corrupt.squeeze())

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

def get_lengths(dataset, outliers):
    output_true = []
    output_ano = []

    i = 0
    for path in dataset:
        l = len(path["FourHot"].todense())

        if i in outliers:
            output_ano.append(l*10/60)
        else:
            output_true.append(l*10/60)

        i += 1

    return output_true, output_ano

def calc_z(dist1, dist2):
    mu_1 = np.mean(dist1)
    mu_2 = np.mean(dist2)

    sigma1 = np.std(dist1) / np.sqrt(len(dist1))
    sigma2 = np.std(dist2) / np.sqrt(len(dist2))

    z = (mu_1-mu_2) / np.sqrt(sigma1**2 + sigma2**2)

    return z


def calc_v(dist1, dist2):
    s1 = np.std(dist1)
    s2 = np.std(dist2)

    n1 = len(dist1)
    n2 = len(dist2)

    v = (s1**2/n1+s2**2/n2)**2/((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))

    return v

def get_anomaly_map(routes_tagged, dk, swe, legend_lines, save_fig, name, save_path):
    f, ax = plt.subplots()
    dk.plot(ax=ax, color="olivedrab")
    swe.plot(ax=ax, color="olivedrab")
    for i in range(len(routes_tagged["lat"])):
        if routes_tagged["anomaly"][i] == 1:
            continue
        else:
            c = "b"
            a = 0.2

        plt.plot(routes_tagged["long"][i], routes_tagged["lat"][i], c, alpha=a)
    for i in range(len(routes_tagged["lat"])):
        if routes_tagged["anomaly"][i] == 1:
            c = "r"
            a = 1
        else:
            continue
        plt.plot(routes_tagged["long"][i], routes_tagged["lat"][i], c, alpha=a)
    if name == "true":
        plt.title("Paths Marked for Anomalies", fontsize=14)
    elif name == "pass":
        plt.title("Passenger Ship \n Paths Marked for Anomalies", fontsize=14)
    elif name == "l":
        plt.title("Paths Marked for Anomalies", fontsize=14)
    plt.xlabel("Longitude", fontsize=10)
    plt.ylabel("Latitude", fontsize=10)
    plt.xlim(13, 17)
    plt.ylim(54.5, 56.5)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(legend_lines, ["Normal", "Anomaly"], fontsize=10)
    if save_fig:
        plt.savefig(save_path + f"anomalies_{name}.png")
    plt.show()

def get_example_inputs(inputs_true, inputs_flip, inputs_shift, long_cols, lat_cols, n_examples, save_fig, save_path):
    idx = range(len(inputs_true))

    for i in range(n_examples):

        id = random.choice(idx)

        df_true = pd.DataFrame(inputs_true[id].todense(), columns=long_cols, index=lat_cols[::-1])
        df_flip = pd.DataFrame(inputs_flip[id].todense(), columns=long_cols, index=lat_cols[::-1])
        df_shift = pd.DataFrame(inputs_shift[id].todense(), columns=long_cols, index=lat_cols[::-1])

        plt.figure()
        sns.heatmap(df_true, cmap="Reds")
        sns.set(font_scale=1.5)
        plt.title("True Path", fontsize=22)
        plt.xlabel("Longitude", fontsize=18)
        plt.ylabel("Latitude", fontsize=18)
        plt.xticks(ticks=np.arange(0, len(long_cols) + 1, 50), labels=long_cols[np.arange(0, len(long_cols) + 1, 50)],
                   fontsize=16)
        plt.yticks(ticks=np.arange(0, len(lat_cols) + 1, 25),
                   labels=lat_cols[::-1][np.arange(0, len(lat_cols) + 1, 25)],
                   fontsize=16)
        plt.tight_layout()
        if save_fig:
            plt.savefig(save_path + f"ex_true{i}.png")
        plt.show()

        plt.figure()
        sns.heatmap(df_flip, cmap="Reds")
        sns.set(font_scale=1.5)
        plt.title("Flipped Path", fontsize=22)
        plt.xlabel("Longitude", fontsize=18)
        plt.ylabel("Latitude", fontsize=18)
        plt.xticks(ticks=np.arange(0, len(long_cols) + 1, 50), labels=long_cols[np.arange(0, len(long_cols) + 1, 50)],
                   fontsize=16)
        plt.yticks(ticks=np.arange(0, len(lat_cols) + 1, 25),
                   labels=lat_cols[::-1][np.arange(0, len(lat_cols) + 1, 25)],
                   fontsize=16)
        plt.tight_layout()
        if save_fig:
            plt.savefig(save_path + f"ex_flip{i}.png")
        plt.show()

        plt.figure()
        sns.heatmap(df_shift, cmap="Reds")
        sns.set(font_scale=1.5)
        plt.title("Shifted Path", fontsize=22)
        plt.xlabel("Longitude", fontsize=18)
        plt.ylabel("Latitude", fontsize=18)
        plt.xticks(ticks=np.arange(0, len(long_cols) + 1, 50), labels=long_cols[np.arange(0, len(long_cols) + 1, 50)],
                   fontsize=16)
        plt.yticks(ticks=np.arange(0, len(lat_cols) + 1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols) + 1, 25)],
                   fontsize=16)
        plt.tight_layout()
        if save_fig:
            plt.savefig(save_path+f"ex_shift{i}.png")
        plt.show()


def get_corrupt_recon(inputs_true, inputs_corr, logits_corr, probs_corr, long_cols, lat_cols, save_fig, save_path, n_examples, corr_name):

    idx = range(len(inputs_true))

    for i in range(n_examples):
        id = random.choice(idx)

        plt.figure()
        plt.subplot(2, 2, 1)
        sns.set(font_scale=0.75)
        sns.heatmap(inputs_true[id].todense(), cmap="Reds")
        plt.xticks(ticks=np.arange(0, len(long_cols) + 1, 50), labels=long_cols[np.arange(0, len(long_cols) + 1, 50)],
                   fontsize=8)
        plt.yticks(ticks=np.arange(0, len(lat_cols) + 1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols) + 1, 25)],
                   fontsize=8)
        plt.title("True Path", fontsize=10)

        plt.subplot(2, 2, 2)
        sns.heatmap(inputs_corr[id].todense(), cmap="Reds")
        plt.xticks(ticks=np.arange(0, len(long_cols) + 1, 50), labels=long_cols[np.arange(0, len(long_cols) + 1, 50)],
                   fontsize=8)
        plt.yticks(ticks=np.arange(0, len(lat_cols) + 1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols) + 1, 25)],
                   fontsize=8)
        plt.title("Corrupted path", fontsize=10)

        plt.subplot(2, 2, 3)
        sns.heatmap(logits_corr[id].squeeze(), cmap="Reds")
        plt.xticks(ticks=np.arange(0, len(long_cols) + 1, 50), labels=long_cols[np.arange(0, len(long_cols) + 1, 50)],
                   fontsize=8)
        plt.yticks(ticks=np.arange(0, len(lat_cols) + 1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols) + 1, 25)],
                   fontsize=8)
        plt.title("Logits", fontsize=10)

        plt.subplot(2, 2, 4)
        sns.heatmap(probs_corr[id].squeeze(), cmap="Reds")
        plt.xticks(ticks=np.arange(0, len(long_cols) + 1, 50), labels=long_cols[np.arange(0, len(long_cols) + 1, 50)],
                   fontsize=8)
        plt.yticks(ticks=np.arange(0, len(lat_cols) + 1, 25), labels=lat_cols[::-1][np.arange(0, len(lat_cols) + 1, 25)],
                   fontsize=8)
        plt.title("Probabilities", fontsize=10)
        plt.tight_layout()
        if save_fig:
            plt.savefig(save_path+f"example_{corr_name}{i}.png")
        plt.show()


def create_sparse_input(inputs):
    input_sparse = []
    for input in inputs:
        input_sparse.append(sparse.csr_matrix(input))

    return input_sparse

def get_outlier_l(log_px_true, l, threshold):

    id_ano = np.where([log_px_true[id] for id in l] < threshold)[0]

    return [l[i] for i in id_ano]

def id_length(dataset):
    l8 = []
    l12 = []
    l16 = []
    l20 = []
    l24 = []

    for i, path in enumerate(dataset):
        l = len(path["FourHot"].todense())*10/60

        if l < 8:
            l8.append(i)
        elif l < 12:
            l12.append(i)
        elif l < 16:
            l16.append(i)
        elif l < 20:
            l20.append(i)
        elif l <= 24:
            l24.append(i)

    return l8, l12, l16, l20, l24

def get_log_px_train(dataset, model):

    out = {"log_px": [],
           "length": []}

    for path in dataset:
        img = path["FourHot"].todense()

        l = len(img)

        lat, long, sog, cog = np.split(img, Config.breaks["bh"], axis=1)

        path_img = np.zeros([201, 402])
        for t in range(len(lat)):
            slice = np.transpose(lat[t, :])[::-1] @ long[t, :]
            path_img += slice

        item = torch.tensor(path_img / np.max(path_img, axis=1).max(), dtype=torch.float)

        item = item.unsqueeze(0).unsqueeze(1)

        _, log_px = model.sample(item)

        out["log_px"].append(log_px)
        out["length"].append(l)

    return out

def get_thresholds(output, percentile):
    l8 = []
    l12 = []
    l16 = []
    l20 = []
    l24 = []

    for i, l in enumerate(output["length"]):
        l = l * 10 / 60

        if l < 8:
            l8.append(i)
        elif l < 12:
            l12.append(i)
        elif l < 16:
            l16.append(i)
        elif l < 20:
            l20.append(i)
        elif l < 24:
            l24.append(i)

    t8 = np.quantile([output["log_px"][k] for k in l8], percentile)
    t12 = np.quantile([output["log_px"][k] for k in l12], percentile)
    t16 = np.quantile([output["log_px"][k] for k in l16], percentile)
    t20 = np.quantile([output["log_px"][k] for k in l20], percentile)
    t24 = np.quantile([output["log_px"][k] for k in l24], percentile)

    return t8, t12, t16, t20, t24