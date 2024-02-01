from scipy.interpolate import interp1d
import os
import numpy as np
import pickle as pkl
import scipy.signal as sig
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = "Times New Roman"
plt.rcParams['font.size'] = 18

def extract_tail(cs):  # crucial to tail clustering
    max_len = 60
    c_tail = cs[-max_len:]  # tail stage <= 60min
    v_max = np.max(c_tail)
    if v_max < 15:  # such curves could not hold tails
        cs = cs*32.0/np.max(cs)
        c_tail = cs[-max_len:]
    try:
        start_index = np.where(c_tail >= v_max-2)[0][-1]
    except:
        return cs, None
    c_tail = c_tail[start_index:]
    tail_len = c_tail.shape[0]
    if 5 <= tail_len <= max_len:
        c_bulk = cs[0:len(cs) - tail_len]
        tail_max = np.max(c_tail)
        if tail_max < 15:
            c_tail = c_tail*32/tail_max
        return c_bulk, c_tail
    else:
        return cs, None

def output_acn_tails(input_folder, output_folder, id):  # derive high-quality tails
    sample_paths = [f for f in os.listdir(input_folder) if f.endswith("pkl")]
    for pkl_path in sample_paths:
        input_pkl = f"{input_folder}/{pkl_path}"
        with open(input_pkl, "rb") as f:
            if id == "Real":
                cs = np.around(np.array(pkl.load(f)["current"]), 2)
            else:
                cs = np.around(np.array(pkl.load(f)), 2)
            _, c_tail = extract_tail(cs)
            if c_tail is None:
                continue
            x = np.arange(0, len(c_tail), 1)*60/4
            interp_func = interp1d(x, c_tail, kind="linear")
            xs = np.arange(np.min(x), np.max(x)+1, 1)
            ys = interp_func(xs)
            ys_filt = sig.medfilt(ys, kernel_size=5)
            output_pkl = f"{output_folder}/{pkl_path}"
            with open(f"{output_pkl}", "wb") as f:
                pkl.dump(ys_filt, f)
            output_png = f"{output_folder}/{pkl_path.replace('pkl', 'png')}"
            plt.plot(ys_filt)
            plt.savefig(output_png)
            plt.clf()
            print(f"{pkl_path} tail done!")

def read_tail_files(input_folder):
    sample_paths = [f for f in os.listdir(input_folder) if f.endswith("pkl")]
    tail_set, tail_padding_set = [], []
    for pkl_path in sample_paths:
        input_pkl = f"{input_folder}/{pkl_path}"
        tail_mask = np.zeros(60*15)
        with open(input_pkl, "rb") as f:
            cs = np.array(pkl.load(f))
        if "diffusion" or "cnn" in input_folder:
            low_indexes = np.where(cs < 2.5)
            cs = np.delete(cs, low_indexes)
        for i in range(15, 35, 5):  # both shape and magnitude could impact the clustering result
            if "gmm" in input_folder and i!=30:
                continue
            cs = cs*i/np.max(cs)
            tail_mask[0:len(cs)] = cs
            tail_set.append(cs.tolist())
            tail_padding_set.append(tail_mask.tolist())
    return tail_set, tail_padding_set

dark_color_set = ["crimson", "green", "grey", "orange", "blue", "purple", "saddlebrown"]
light_color_set = ["pink", "lightgreen", "lightgray", "moccasin", "lightblue", "plum", "sandybrown"]

def cluster_acn_tails(real_folder, diff_folder, cnn_folder, gmm_folder, gan_folder, aae_folder, n_clusters):
    real_tail_set, real_tail_padding_set = read_tail_files(real_folder)
    diff_tail_set, diff_tail_padding_set = read_tail_files(diff_folder)
    cnn_tail_set, cnn_tail_padding_set = read_tail_files(cnn_folder)
    gmm_tail_set, gmm_tail_padding_set = read_tail_files(gmm_folder)
    gan_tail_set, gan_tail_padding_set = read_tail_files(gan_folder)
    aae_tail_set, aae_tail_padding_set = read_tail_files(aae_folder)
    kmeans = KMeans(n_clusters=n_clusters, max_iter=400).fit(real_tail_padding_set)
    labelIds = kmeans.labels_
    diff_pred = kmeans.predict(diff_tail_padding_set)
    cnn_pred = kmeans.predict(cnn_tail_padding_set)
    gmm_pred = kmeans.predict(gmm_tail_padding_set)
    gan_pred = kmeans.predict(gan_tail_padding_set)
    aae_pred = kmeans.predict(aae_tail_padding_set)
    # plt.figure(figsize=(24, 7.8))
    real_cdf = plot_tail_cluster(real_tail_set, n_clusters, labelIds, "Real")
    diff_cdf = plot_tail_cluster(diff_tail_set, n_clusters, diff_pred, "Generated")
    cnn_cdf = plot_tail_cluster(cnn_tail_set, n_clusters, cnn_pred, "CNN")
    gmm_cdf = plot_tail_cluster(gmm_tail_set, n_clusters, gmm_pred, "GMM")
    gan_cdf = plot_tail_cluster(gan_tail_set, n_clusters, gan_pred, "TimeGAN")
    aae_cdf = plot_tail_cluster(aae_tail_set, n_clusters, aae_pred, "AAE")
    # for i in range(n_clusters):
    #     plt.subplot(3, n_clusters, 2*n_clusters+i+1)
    #     plt.plot(real_cdf[i], label="Real", linewidth=2.5, c="red")
    #     plt.plot(diff_cdf[i], label="Diffusion-att", linewidth=2.5, c="blue")
    #     plt.plot(cnn_cdf[i], label="Diffusion-cnn", linewidth=2.5, c="orange")
    #     plt.plot(gmm_cdf[i], label="GMM", linewidth=2.5, c="purple")
    #     x_pos = np.arange(0, 32+8, 8)
    #     x_ticks = np.arange(0, 32+8, 8)
    #     plt.xticks(x_pos, x_ticks)
    #     plt.xlim((0, 32))
    #     plt.ylim((0, 1))
    #     # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"output/tail/tail clusters.png")
    # plt.clf()
    output_cdf = {"real": real_cdf, "diffusion": diff_cdf, "cnn": cnn_cdf,
                  "gmm": gmm_cdf, "timegan": gan_cdf, "aae": aae_cdf}
    with open("output/tail/cdf.pkl", "wb") as f:
        pkl.dump(output_cdf, f)

def plot_tail_cluster(tail_set, n_clusters, labelIds, id):
    tail_cdf = dict()
    for i in range(n_clusters):
        cluster_Ids = np.where(labelIds == i)[0]
        cluster_tails = []
        for j in cluster_Ids:
            tail = tail_set[j]
            cluster_tails.append(tail)
        if id in ["Real", "Generated"]:
            cluster_lens = [len(t) for t in cluster_tails]
            len_thresh = int(np.percentile(cluster_lens, 80))
            cluster_tails_filt = []
            for c in cluster_tails:
                if len(c) > len_thresh:
                    c = c[0:len_thresh]
                cluster_tails_filt.append(c)
            if id == "Real":
                plt.subplot(3, n_clusters, i+1)
            else:
                plt.subplot(3, n_clusters, n_clusters+i+1)
            get_tail_region(cluster_tails_filt, i)
        # Tail CDF
        tail_x = []
        for k in cluster_tails:
            tail_x.extend(k)
        prob, bins, _ = plt.hist(x=tail_x, bins=32, range=(0, 32), density=True, cumulative=True)
        tail_cdf[i] = prob
    return tail_cdf

def get_tail_region(tail_set, i):
    max_len = max([len(s) for s in tail_set])
    min_last = min([s[-1] for s in tail_set])
    for s in tail_set:
        pad = np.linspace(s[-1], min_last, max_len-len(s)).tolist()
        s.extend(pad)
    ub = np.percentile(np.array(tail_set), 80, axis=0)
    lb = np.percentile(np.array(tail_set), 20, axis=0)
    x = np.arange(max_len)
    plt.stackplot(x, lb, ub-lb, colors=["white", light_color_set[i]], baseline="zero")
    medoid = np.percentile(np.array(tail_set), 50, axis=0)
    plt.plot(medoid, color=dark_color_set[i], linewidth=2, label=f"Medoid {i+1}")
    x_max = (max_len//50+1)*50
    x_pos = [0, x_max]
    x_ticks = [0, x_max]
    plt.xticks(x_pos, x_ticks)
    plt.xlim((0, x_max))
    y_pos = np.arange(0, 32+4, 8)
    y_ticks = np.arange(0, 32+4, 8)
    plt.yticks(y_pos, y_ticks)
    plt.ylim((0, 32))
    plt.legend()

def cal_cdf_distance():
    with open("output/tail/cdf.pkl", "rb") as f:
        cdf_dict = pkl.load(f)
    real_cdf, diff_cdf, cnn_cdf = cdf_dict["real"], cdf_dict["diffusion"], cdf_dict["cnn"]
    gmm_cdf, gan_cdf, aae_cdf = cdf_dict["gmm"], cdf_dict["timegan"], cdf_dict["aae"]
    cdf_dis = {}
    for i in range(7):
        real_i, diff_i, cnn_i = real_cdf[i], diff_cdf[i], cnn_cdf[i]
        gmm_i, gan_i, aae_i = gmm_cdf[i], gan_cdf[i], aae_cdf[i]
        diff_dis = np.mean(np.abs(real_i-diff_i))
        cnn_dis = np.mean(np.abs(real_i-cnn_i))
        gmm_dis = np.mean(np.abs(real_i-gmm_i))
        gan_dis = np.mean(np.abs(real_i-gan_i))
        aae_dis = np.mean(np.abs(real_i-aae_i))
        cdf_dis[i] = [np.around(diff_dis,4), np.around(cnn_dis,4), np.around(gmm_dis,4), np.around(gan_dis,4), np.around(aae_dis,4)]
    print("CDF distance", cdf_dis)

if __name__ == "__main__":
    # output_acn_tails("ACN-data/jpl/driver", "tails/real", "Real")
    # output_acn_tails("generation/diffusion/attention/driver/unconditional", "tails/diffusion", "Diffusion")
    # output_acn_tails("generation/diffusion/driver/tail-selected", "tails/diffusion", "Diffusion")
    # output_acn_tails("generation/diffusion/cnn/driver/unconditional", "tails/cnn", "cnn")
    # output_acn_tails("generation/gmm/driver", "tails/gmm", "gmm")
    # output_acn_tails("generation/aae/driver-alpha0.75", "tails/timegan", "timegan")
    # output_acn_tails("generation/aae/driver-alpha0.5", "tails/aae", "aae")
    # output_acn_tails("generation/diffusion/nonatt/driver/unconditional", "tails/nonatt", "nonatt")
    # output_acn_tails("generation/diffusion/attention/driver/unconditional-T30", "tails/T30", "T30")
    # output_acn_tails("generation/diffusion/attention/driver/unconditional-T40", "tails/T40", "T40")
    # output_acn_tails("generation/diffusion/attention/driver/unconditional-T60", "tails/T60", "T60")
    # output_acn_tails("generation/diffusion/attention/driver/unconditional-T70", "tails/T70", "T70")
    # cluster_acn_tails("tails/real_selected", "tails/diffusion_selected", "tails/cnn", "tails/gmm", "tails/timegan", "tails/aae", n_clusters=7)
    # cluster_acn_tails("tails/real_selected", "tails/nonatt", "tails/T30", "tails/T40", "tails/T60", "tails/T70", n_clusters=7)
    # cal_cdf_distance()
    pass
