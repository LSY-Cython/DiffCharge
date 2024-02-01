import numpy as np
import cvxpy as cp
import pickle as pkl
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import down_sample
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from collections import Counter

# LMP in DAM
daily_lmp = pd.read_csv("ACN-data/caiso_lmp_combined_2022-08-30.csv")["lmp_dam"].values*1e-3  # $/MWh -> $/kWh, 5-min
daily_lmp = daily_lmp.reshape((1, 288))

# test day setup, nominal voltage: 300V
day = "2019-08-30"
input_folder = "ACN-data/jpl/driver"
data_files = [f"{input_folder}/{f}" for f in os.listdir(input_folder) if day in f and f.endswith("pkl")]
ev_num = len(data_files)
print(f"{day} charged EV: {ev_num}")
real_curves = []
arrival_steps = []
for df in data_files:
    with open(df, "rb") as f:
        mask = [0]*288
        session_data = pkl.load(f)
        arrival_time = session_data["arrivalTime"]
        arrival_step = arrival_time[0]*12 + arrival_time[1]//5
        arrival_steps.append(arrival_step)
        current_series = down_sample(session_data["current"], 5)[1:]  # 1min -> 5min
        current_series = np.array(current_series)*32/np.max(current_series)
        mask[arrival_step:arrival_step+len(current_series)] = current_series.tolist()
        real_curves.append(mask)
real_load = np.sum(real_curves, axis=0)
real_curves = np.array(real_curves)*0.3

# sample of arrival time based on its frequency
counter = dict(Counter(arrival_steps))
freq = {}
for k, v in counter.items():
    freq[k] = v/len(arrival_steps)

# pick out diffusion-based representative samples
def select_generated_samples(input_folder):
    np.random.seed(0)
    data_files = [f"{input_folder}/{f}" for f in os.listdir(input_folder) if f.endswith("pkl")]
    gen_curves = []
    arr_samples = np.random.choice(a=list(freq.keys()), size=len(data_files), p=list(freq.values()))
    for i in range(len(data_files)):
        with open(data_files[i], "rb") as f:
            mask = [0]*288
            gen_series = pkl.load(f)
            if np.max(gen_series) == 0.0:
                continue
            gen_step = arr_samples[i]
            gen_series = down_sample(gen_series, 5)  # 1min -> 5min
            if np.max(gen_series) == 0.0:
                continue
            gen_series = np.array(gen_series)*32/(np.max(gen_series))
            end_index = gen_step+len(gen_series)
            if end_index > 288:
                gen_step -= (end_index-288)
            mask[gen_step:end_index] = gen_series.tolist()
            gen_curves.append(mask)
    X = np.array(gen_curves)
    kmeans = KMeans(n_clusters=ev_num, random_state=0, n_init="auto").fit(X)
    labels = kmeans.labels_
    # gen_reps = kmeans.cluster_centers_
    gen_reps = []
    for i in range(ev_num):
        indices = np.where(labels == i)[0]
        samples = X[indices]
        gen_reps.append(np.mean(samples, axis=0).tolist())
    # reps_indices = np.random.choice(a=range(len(gen_curves)), size=ev_num, replace=False)
    # gen_reps = np.array(gen_curves)[reps_indices]
    gen_load = np.sum(gen_reps, axis=0)
    return np.array(gen_reps)*0.3, gen_load

def bid_solver(ch_curves, id):
    # problem formulation
    ch_curves = ch_curves.T
    x = cp.Variable((288, ev_num))  # bid power for EV i at step t
    e1 = np.ones((ev_num, 1))
    e2 = np.ones((1, 288))
    constraints = [x >= 0, x <= 32]
    ch_cost = daily_lmp@x@e1
    ut_cost = cp.sum_squares(e2@(x-ch_curves)@e1) + cp.sum_squares(x-ch_curves)/ev_num
    # ut_cost = cp.Pnorm(e2@(x-ch_curves)@e1, p=2) + cp.Pnorm(x-ch_curves, p=2)
    obj = cp.Minimize(ch_cost+ut_cost*0.8*np.max(daily_lmp))
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=False)
    solution = x.value.T
    ch_value = daily_lmp@solution.T@e1
    ut_value = (cp.sum_squares(e2@(solution.T-ch_curves)@e1) + cp.sum_squares(solution.T-ch_curves)/ev_num).value*0.8*np.max(daily_lmp)
    oc = prob.value
    print(f"{id} optimal cost: {ch_value/12}, {ut_value/12}, {oc/12}")
    step_bids = np.sum(solution, axis=0)
    return step_bids

def draw_results(real_bids, diff_bids, aae_bids, gan_bids, gmm_bids):
    for i in range(24):
        real_bids[i * 12:(i + 1) * 12] = np.mean(real_bids[i * 12:(i + 1) * 12])
        diff_bids[i * 12:(i + 1) * 12] = np.mean(diff_bids[i * 12:(i + 1) * 12])
        aae_bids[i * 12:(i + 1) * 12] = np.mean(aae_bids[i * 12:(i + 1) * 12])
        gan_bids[i * 12:(i + 1) * 12] = np.mean(gan_bids[i * 12:(i + 1) * 12])
        gmm_bids[i * 12:(i + 1) * 12] = np.mean(gmm_bids[i * 12:(i + 1) * 12])
    fig = plt.figure(dpi=150)
    ax1 = fig.subplots()
    ax2 = ax1.twinx()
    l11, = ax1.plot(real_bids, label="Real", color="red")
    l12, = ax1.plot(diff_bids, label="DiffCharge", color="blue")
    # ax1.plot(gmm_bids, label="GMM", linestyle='dashdot', color="green")
    l13, = ax1.plot(aae_bids, label="VAEGAN", linestyle='dashed', color="green")
    l14, = ax1.plot(gan_bids, label="TimeGAN", linestyle='dotted', color="darkorange")
    # ax1.legend()
    fs = 12
    ax1.set_ylabel("Charging demand bids [kW]", fontsize=fs)
    ax1.set_xlabel("Time [h]", fontsize=fs)
    l2, = ax2.plot(daily_lmp[0], label="DAM price [$/kWh]", color="slategray")
    # ax2.legend()
    ax2.set_ylabel("DAM price [$/kWh]", fontsize=fs)

    lns = [l11, l12, l13, l14, l2]
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels, fontsize=9)
    plt.xlim((0, 24))
    positions = list(range(0, 288+1, 12))
    ticks = list(range(0, 24+1, 1))
    plt.xticks(positions, ticks)
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(f"output/dam/daily_bids.png")

if __name__ == "__main__":
    # produce stochastic samples
    diff_curves, diff_load = select_generated_samples("generation/diffusion/attention/driver/unconditional-T50-v2")
    aae_curves, aae_load = select_generated_samples("generation/aae/driver-alpha0.5")
    gan_curves, gan_load = select_generated_samples("generation/aae/driver-alpha0.75")
    gmm_curves, gmm_load = select_generated_samples("generation/gmm/driver")
    plt.plot(real_load, label="Real load")
    plt.plot(diff_load, label="Diffusion load")
    plt.plot(aae_load, label="AAE load")
    plt.plot(gan_load, label="TimeGAN load")
    plt.plot(gmm_load, label="GMM load")
    plt.legend()
    plt.savefig(f"output/dam/daily_load.png")
    plt.clf()
    # optimize DAM bids
    real_bids = bid_solver(real_curves, id="Real")
    diff_bids = bid_solver(diff_curves, id="Diffusion")
    aae_bids = bid_solver(aae_curves, id="AAE")
    gan_bids = bid_solver(gan_curves, id="TimeGAN")
    gmm_bids = bid_solver(gmm_curves, id="GMM")
    # compare results
    draw_results(real_bids, diff_bids, aae_bids, gan_bids, gmm_bids)



