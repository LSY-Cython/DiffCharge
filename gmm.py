import numpy as np
from sklearn.mixture import GaussianMixture
import os
import pickle as pkl
import scipy.signal as sig
import matplotlib.pyplot as plt

def estimate_driver_gmm(input_folder, max_len):
    data_paths = [f for f in os.listdir(input_folder) if f.endswith("pkl")]
    current_set = []
    for p in data_paths:
        pkl_path = os.path.join(input_folder, p)
        with open(pkl_path, "rb") as f:
            current_data = pkl.load(f)["current"]
            current_mask = np.zeros(max_len)
            duration = len(current_data)
            if duration > max_len:
                continue
            current_mask[0:duration] = current_data
            current_set.append(current_mask.tolist())
    X = np.array(current_set)
    gmm = GaussianMixture(n_components=15, random_state=0, max_iter=200).fit(X)
    return gmm

def sample_driver_gmm(sample_num, output_folder):
    gmm = estimate_driver_gmm("ACN-data/jpl/driver", 720)
    samples, labels = gmm.sample(sample_num)
    for i in range(sample_num):
        x = samples[i]
        x = driver_postprocess(x)
        with open(f"{output_folder}/{i}.pkl", "wb") as f:
            pkl.dump(x, f)
        plt.plot(x)
        plt.savefig(f"{output_folder}/{i}.png")
        plt.clf()
        print(f"{i} output done!")

def driver_postprocess(x):
    x = sig.medfilt(x, 5)
    invalid_index = np.where(x < 0)[0]
    x[invalid_index] = 0
    valid_index = np.where(x < 1)[0]
    valid_index = valid_index[np.where(valid_index > 5)[0]]
    x_valid = x[0:valid_index[0] + 1]
    return x_valid

if __name__ == "__main__":
    sample_driver_gmm(1500, "generation/gmm/driver")
