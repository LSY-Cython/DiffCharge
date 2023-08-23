import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

def cmax_sample(input_folder, sample_num):
    cmax_set = []
    sample_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith("pkl")]
    for p in sample_paths:
        with open(p, "rb") as f:
            cs = np.array(pkl.load(f)["current"])
            cmax = np.max(cs)
            cmax_set.append(cmax)
    prob, bins, _ = plt.hist(x=np.array(cmax_set), bins=32, range=(0, 32), density=True)
    cmax_samples = []
    # Monte Carlo Sample
    cum_prob = np.cumsum(prob)
    rand_values = np.random.uniform(0, 1, sample_num)
    for i in rand_values:
        cmax = bins[np.where(cum_prob >= i)[0][0] + 1]
        cmax_samples.append(cmax)
    plt.clf()
    return cmax_samples

def pmax_sample(input_folder, sample_num):
    pmax_set = []
    sample_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith("pkl")]
    for p in sample_paths:
        with open(p, "rb") as f:
            ps = np.array(pkl.load(f)["power"])
            pmax = np.around(np.max(ps), 0)
            pmax_set.append(pmax)
    prob, bins, _ = plt.hist(x=np.array(pmax_set), bins=800, range=(0, 800), density=True)
    pmax_samples = []
    # Monte Carlo Sample
    cum_prob = np.cumsum(prob)
    rand_values = np.random.uniform(0, 1, sample_num)
    for i in rand_values:
        pmax = bins[np.where(cum_prob >= i)[0][0] + 1]
        pmax_samples.append(pmax)
    plt.clf()
    return pmax_samples