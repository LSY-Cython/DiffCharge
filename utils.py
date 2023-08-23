import time
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

gmt_format = '%a, %d %b %Y %H:%M:%S GMT'
def gmt_to_datetime(gmt):
    time_struct = time.strptime(gmt, gmt_format)
    output_format = '%Y-%m-%d %H:%M:%S'
    return time.strftime(output_format, time_struct)

def gmt_to_timestamp(gmt):
    if type(gmt) is str:
        time_step = time.mktime(time.strptime(gmt, gmt_format))
        return time_step
    else:
        time_steps = [time.mktime(time.strptime(t, gmt_format)) for t in gmt]
        return time_steps

def interpolate_signal(timestamp, signal):
    interp_func = interp1d(timestamp, signal, kind="nearest")
    xs = np.arange(np.min(timestamp), np.max(timestamp)+1, 1)
    ys = interp_func(xs)
    return xs.tolist(), ys.tolist()

def down_sample(signal, scale):
    sample_index = np.arange(0, len(signal), scale).tolist()
    signal_down = np.array(signal)[sample_index].tolist()
    if len(signal) % scale != 0:
        signal_down = signal_down + [signal[-1]]
    return signal_down

def plot_session(pilot, current, title, path):
    plt.subplot(2, 1, 1)
    plt.plot(pilot, label="pilotSignal")
    plt.plot(current, label="chargingCurrent")
    plt.legend()
    plt.title(title)
    plt.subplot(2, 1, 2)
    plt.legend()
    plt.savefig(path)
    plt.clf()

def plot_heatmap(matrix, x, y, output_path):
    plt.figure(figsize=(12, 12), dpi=200)
    plt.imshow(matrix)
    plt.xticks(np.arange(len(x)), labels=x)
    plt.yticks(np.arange(len(y)), labels=y)
    plt.xlabel("Congestion Frequency")
    plt.ylabel("Pilot Values")
    for i in range(len(y)):
        for j in range(len(x)):
            plt.text(j, i, int(matrix[i, j]), ha="center", va="center", color="w")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path)

def plot_training_loss(*args, model_name, labels):
    sub_num = len(labels)
    if sub_num == 1:
        plt.plot(args[0], label=f"{labels[0]} Loss")
        plt.legend()
    else:
        for i in range(sub_num):
            plt.subplot(sub_num, 1, i+1)
            plt.plot(args[i], label=f"{labels[i]} Loss")
            plt.legend()
    plt.savefig(f"{model_name}_loss.png")
    plt.clf()

def plot_driver_generation(x1, x2, path):
    plt.subplot(2, 1, 1)
    plt.plot(x1, color="green", label="with zero padding")
    plt.plot(x2, color="orange", label="without zero padding")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(x2, label="Generated curve")
    img_path = f"{path}.png"
    plt.savefig(img_path)
    plt.clf()
    pkl_path = f"{path}.pkl"
    with open(f"{pkl_path}", "wb") as f:
        pkl.dump(x2, f)
    print(f"{path} generated done!")

def plot_station_generation(x, path):
    plt.plot(x, label="Generated charging station load")
    plt.savefig(f"{path}.png")
    plt.clf()
    with open(f"{path}.pkl", "wb") as f:
        pkl.dump(x, f)
    print(f"{path} generated done!")

def plot_reconstruction(real, rec, latent, path):
    plt.subplot(2, 1, 1)
    plt.plot(real, label="Real")
    plt.plot(rec, label="Reconstruction")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(latent, label="Latent")
    plt.legend()
    plt.savefig(path)
    plt.clf()
    print(f"{path} reconstructed done!")

def plot_prediction_error(error_set, output_path):
    plt.boxplot(error_set, sym=".", whis=1.5)
    plt.savefig(output_path)
    plt.clf()