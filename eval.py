import random
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from options import Options
from dataset import creat_dataloader
import scipy.signal as sig
from tail import extract_tail

plt.rcParams['font.sans-serif'] = "Times New Roman"
plt.rcParams['font.size'] = 20

def cal_current_pdf(folder, id):
    current_set = []
    sample_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith("pkl")]
    for i in range(len(sample_paths)):
        with open(sample_paths[i], "rb") as f:
            if id == "Real":
                current_series = pkl.load(f)["current"]
            else:
                current_series = pkl.load(f)
        try:
            c_bulk, c_tail = extract_tail(current_series)
        except:
            c_bulk = current_series
        # stochastic behaviors within the bulk stage
        for i in [8, 16, 32]:
            c_temp = np.array(c_bulk)/np.max(c_bulk)*i
            c_temp = np.around(c_temp, 2)
            current_set.extend(c_temp.tolist())
    # current rate
    zero_index = np.where(np.array(current_set) == 0.0)[0]
    current_valid = np.delete(current_set, zero_index)
    current_X = current_valid.reshape(-1, 1)
    current_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(current_X)
    current_samples = np.arange(0, 32, 0.1).reshape(-1, 1)
    current_density = np.exp(current_kde.score_samples(current_samples))
    return current_samples, current_density, current_valid

def config_current_axis():
    plt.xlabel(f"Charging rate [A]")
    plt.ylabel("Probability density")
    x_pos = np.arange(0, 32+4, 4)
    x_ticks = np.arange(0, 32+4, 4)
    plt.xticks(x_pos, x_ticks)
    plt.xlim((0, 32))
    plt.ylim((0, 0.2))
    plt.legend()

def battery_current_pdf(real_folder, gen_folder):
    real_position, real_density, real_current = cal_current_pdf(real_folder, "Real")
    gen_position, gen_density, gen_current = cal_current_pdf(gen_folder, "Generated")
    fig = plt.figure(figsize=(12.8, 6.8))
    fig.set_tight_layout(True)
    plt.subplot(1, 2, 1)
    plt.plot(real_position, real_density, color="purple", linewidth=2.5, label="KDE")
    plt.hist(x=real_current, bins=32, range=(0, 32), density=True, label="Real",
             facecolor="lightgreen", edgecolor="darkgreen")
    config_current_axis()
    plt.subplot(1, 2, 2)
    plt.plot(gen_position, gen_density, color="purple", linewidth=2.5, label="KDE")
    plt.hist(x=gen_current, bins=32, range=(0, 32), density=True, label="Generated",
             facecolor="lightblue", edgecolor="darkblue")
    config_current_axis()
    plt.savefig(f"output/battery/battery current pdf.png")
    plt.clf()

def cal_power_pdf(input_folder, id, bins, max_power):
    power_set = []
    sample_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith("pkl")]
    for p in sample_paths:
        with open(p, "rb") as f:
            if id == "Real":
                power_series = pkl.load(f)["power"]
                lc = "purple"
                fc = "lightgreen"
                ec = "darkgreen"
            else:
                power_series = pkl.load(f)
                lc = "purple"
                fc = "lightblue"
                ec = "darkblue"
            power_set.extend(power_series)
    power_X = np.array(power_set).reshape(-1, 1)
    power_kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(power_X)
    power_samples = np.arange(0, max_power, max_power//bins).reshape(-1, 1)
    power_density = np.exp(power_kde.score_samples(power_samples))
    plt.plot(power_samples, power_density, c=lc, linewidth=2, label="KDE")
    plt.hist(x=power_set, bins=bins, range=(0, max_power), density=True, facecolor=fc, edgecolor=ec, label=f"{id}")
    plt.legend()
    plt.xlabel(f"Charging power [kW]")
    plt.ylabel(f"Probability density")
    plt.xlim((0, max_power))
    plt.ylim((0, 0.1))

def station_power_pdf(real_folder, gen_folder, site, bins, max_power):
    fig = plt.figure(figsize=(11.8, 5.4))
    fig.set_tight_layout(True)
    plt.subplot(1, 2, 1)
    cal_power_pdf(real_folder, "Real", bins, max_power)
    plt.subplot(1, 2, 2)
    cal_power_pdf(gen_folder, "Generated", bins, max_power)
    plt.savefig(f"output/station/{site} charging power pdf.png")
    plt.clf()

def cal_duration_pdf(input_folder, id):
    duration_set = []
    sample_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith("pkl")]
    for p in sample_paths:
        with open(p, "rb") as f:
            if id == "Real":
                current_series = np.around(pkl.load(f)["current"], 2)
            else:
                current_series = np.around(pkl.load(f), 2)
            duration = np.where(np.array(current_series) > 0.5)[0].shape[0]
            if duration >= 710:
                duration_set.extend([105, 135]*2)  # [90, 120], [120, 150]
            if 90 <= duration < 710:  # 1.5h
                duration_set.append(duration)
    duration_X = np.array(duration_set).reshape(-1, 1)
    duration_kde = KernelDensity(kernel='gaussian', bandwidth=0.9).fit(duration_X)
    duration_samples = np.arange(90, 720, 15).reshape(-1, 1)
    duration_density = np.exp(duration_kde.score_samples(duration_samples))
    return duration_samples, duration_density, duration_set

def config_duration_axis():
    plt.xlabel(f"Charging duration [hour]")
    plt.ylabel(f"Probability density")
    plt.xlim((90, 720))
    plt.ylim((0, 0.005))
    x_pos = np.arange(90, 720+30, 30*3)
    x_ticks = np.arange(1.5, 12+1.5, 1.5)
    plt.xticks(x_pos, x_ticks)
    plt.legend()

def charging_duration_pdf(real_folder, gen_folder):
    real_position, real_density, real_duration = cal_duration_pdf(real_folder, "Real")
    gen_position, gen_density, gen_duration = cal_duration_pdf(gen_folder, "Generated")
    fig = plt.figure(figsize=(12.8, 6.8))
    fig.set_tight_layout(True)
    plt.subplot(1, 2, 1)
    # plt.plot(real_position, real_density, color="purple", linewidth=2.5, label="KDE")
    plt.hist(x=real_duration, bins=(720-90)//30, range=(90, 720), density=True, label="Real",
             facecolor="lightgreen", edgecolor="darkgreen")
    config_duration_axis()
    plt.subplot(1, 2, 2)
    # plt.plot(gen_position, gen_density, color="purple", linewidth=2.5, label="KDE")
    plt.hist(x=gen_duration, bins=(720-90)//30, range=(90, 720), density=True, label="Generated",
             facecolor="lightblue", edgecolor="darkblue")
    config_duration_axis()
    plt.savefig(f"output/battery/charging duration pdf.png")
    plt.clf()

if __name__ == "__main__":
    # battery_current_pdf("ACN-data/jpl/driver", "generation/diffusion/attention/driver/pdf-selected")
    # charging_duration_pdf("ACN-data/jpl/driver", "generation/diffusion/attention/driver/duration-selected")
    station_power_pdf("ACN-data/caltech/station/2019", "generation/diffusion/attention/station/caltech-epoch175",
                      "caltech", bins=25, max_power=100)
    station_power_pdf("ACN-data/jpl/station/2019", "generation/diffusion/attention/station/jpl-epoch175",
                      "jpl", bins=50, max_power=200)
    pass