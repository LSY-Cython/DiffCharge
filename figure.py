import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

plt.rcParams['font.sans-serif'] = "Times New Roman"
plt.rcParams['font.size'] = 20

def plot_noise():
    noise = np.random.rand(288)
    fig = plt.figure(figsize=(6, 4))
    fig.set_tight_layout(True)
    plt.plot(noise, linewidth=4)
    plt.axis('off')
    plt.savefig("output/figure/noise.png")

def plot_profiles():
    # with open("output/figure/battery/ppt1.pkl", "rb") as f:
    #     s1 = np.array(pkl.load(f)["current"])
    # with open("output/figure/battery/ppt2.pkl", "rb") as f:
    #     s2 = np.array(pkl.load(f)["current"])
    with open("output/figure/station/ppt1.pkl", "rb") as f:
        s1 = np.array(pkl.load(f)["power"])
    with open("output/figure/station/ppt2.pkl", "rb") as f:
        s2 = np.array(pkl.load(f)["power"])
    fig = plt.figure(figsize=(14, 6))
    fig.set_tight_layout(True)
    plt.subplot(1, 2, 1)
    plt.plot(s1, linewidth=6)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.plot(s2, linewidth=6)
    plt.axis('off')
    # plt.savefig("output/figure/ppt_samples1.png")
    plt.savefig("output/figure/ppt_samples2.png")
    plt.clf()

def config_time_axis(interval, y_max):
    plt.xlim((0, 24))
    positions = list(range(0, 288+1, 12*interval))
    ticks = list(range(0, 24+1, 1*interval))
    plt.xticks(positions, ticks)
    plt.ylim((0, y_max))
    plt.grid()

def exhibit_station_profile(input_folder, site, y_max):
    fig = plt.figure(figsize=(19.2, 10.8))
    fig.set_tight_layout(True)
    for i in range(3):
        plt.subplot(3, 3, i*3+1)
        with open(f"{input_folder}/real_{i}.pkl", "rb") as f:
            rp = pkl.load(f)["power"]
        plt.plot(rp, c="red", linewidth=3)
        config_time_axis(2, y_max)
        plt.subplot(3, 3, i*3+2)
        with open(f"{input_folder}/gen_{i}a.pkl", "rb") as f:
            gp0 = pkl.load(f)
        if site == "caltech" and i == 2:
            gp0 = gp0*75/np.max(gp0)
        plt.plot(gp0, c="blue", linewidth=3)
        config_time_axis(2, y_max)
        plt.subplot(3, 3, i*3+3)
        with open(f"{input_folder}/gen_{i}b.pkl", "rb") as f:
            gp1 = pkl.load(f)
        if site == "caltech" and i == 2:
            gp1 = gp1*75/np.max(gp1)
        plt.plot(gp1, c="blue", linewidth=3)
        config_time_axis(2, y_max)
    plt.savefig(f"output/figure/{site}_profile.png")

def segment_curve(data_file, pos, id):
    with open(data_file, "rb") as f:
        cs = np.array(pkl.load(f)["current"])
    bulk, tail = cs[0:pos], cs[pos:]
    plt.plot(np.arange(0, pos), bulk, c="#1f77b4", linewidth=4, label="Bulk stage")
    plt.plot(np.arange(pos, len(cs)), tail, c="#ff7f0e", linewidth=4, label="Absorption stage")
    plt.legend(loc="lower right")
    plt.xlabel(f"Time [1min]\n\n({id})")
    plt.ylabel(f"Charging rate [A]")
    plt.grid()

def charging_curve_samples():
    pos1, pos2 = 150, 380
    fig = plt.figure(figsize=(14, 6))
    fig.set_tight_layout(True)
    plt.subplot(1, 2, 1)
    segment_curve("output/figure/battery/sample1.pkl", pos1, "a")
    plt.subplot(1, 2, 2)
    segment_curve("output/figure/battery/sample2.pkl", pos2, "b")
    plt.savefig(f"output/figure/battery_samples.png")
    plt.clf()

def plot_load_profile(data_file, color, id):
    with open(data_file, "rb") as f:
        ps = np.array(pkl.load(f)["power"])
    plt.plot(ps, color=color, linewidth=4.0)
    plt.xlabel(f"Time [5min]\n\n({id})")
    plt.ylabel(f"Charging power [kW]")
    plt.grid()

def load_profile_samples():
    fig = plt.figure(figsize=(14, 6))
    fig.set_tight_layout(True)
    plt.subplot(1, 2, 1)
    plot_load_profile("output/figure/station/workplace.pkl", "#1f77b4", "a")
    plt.subplot(1, 2, 2)
    plot_load_profile("output/figure/station/campus.pkl", "#ff7f0e", "b")
    plt.savefig(f"output/figure/station_samples.png")
    plt.clf()

def plot_noise_schedule():
    plt.figure(figsize=(10, 8))
    beta_1 = 1e-4
    beta_T = 0.5
    T = 50
    t = np.arange(1, T+1, 1)
    beta_t = ((T-t)/(T-1)*np.sqrt(beta_1)+(t-1)/(T-1)*np.sqrt(beta_T))**2
    alpha_t = np.cumprod(1-beta_t)
    plt.plot(alpha_t**0.5, label="Mean weight", linewidth=4, c="#d62728")
    plt.plot((1-alpha_t)**0.5, label="Variance weight", linewidth=4, c="#2ca02c")
    # plt.xlim((0, 50))
    # plt.ylim((0, 1))
    plt.xlabel(f"Diffusion step t")
    plt.ylabel(f"Weight value")
    plt.legend()
    plt.grid(linestyle=":")
    plt.tight_layout()
    plt.savefig("output/figure/noise schedule.png")

if __name__ == "__main__":
    # exhibit_station_profile("output/figure/caltech", "caltech", 80)
    # exhibit_station_profile("output/figure/jpl", "jpl", 200)
    # charging_curve_samples()
    # load_profile_samples()
    # plot_noise()
    # plot_profiles()
    plot_noise_schedule()
    pass