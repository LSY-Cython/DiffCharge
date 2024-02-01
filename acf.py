import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = "Times New Roman"
plt.rcParams['font.size'] = 15

def auto_correlation(x, lag):
    mean, var = np.mean(x), np.var(x)
    x1 = x[0:len(x)-lag]
    x2 = x[lag:len(x)]
    ac = np.mean((x1-mean)*(x2-mean))/var
    return ac

def cal_acf(x):
    acf = []
    for lag in range(1, max_lag+1, 1):
        ac = auto_correlation(x, lag)
        acf.append(ac)
    return np.array(acf)

def output_acf(folder, id):
    sample_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith("pkl")]
    acf_set = []
    for sp in sample_paths:
        with open(sp, "rb") as f:
            if id == "Real":
                x = pkl.load(f)["current"]
            else:
                x = pkl.load(f)
        if len(x) < 90:
            continue
        x_acf = cal_acf(x)
        acf_set.append((sp, x_acf))
        img_path = sp.replace(".pkl", "_acf.png")
        # plt.plot(x_acf, color="red")
        # plt.ylim((0, 1))
        # plt.savefig(img_path)
        # plt.clf()
        print(f"{img_path} acf done!")
    with open(f"output/acf/{id}_acf.pkl", "wb") as f:
        pkl.dump(acf_set, f)

def acf_selection(input_x, acf_file, min_rank):
    with open(input_x, "rb") as f:
        if "real" in input_x:
            x = pkl.load(f)["current"]
        else:
            x = pkl.load(f)
    real_acf = cal_acf(x)
    with open(acf_file, "rb") as f:
        gen_acf = pkl.load(f)
    acf_dis = []
    for _, c in gen_acf:
        acf_dis.append(np.mean(np.abs(real_acf-c)))
    min_index = np.argsort(acf_dis)[min_rank]
    min_path = gen_acf[min_index][0]
    print(min_path, acf_dis[min_index])

def set_axis(x_max, y_max, y_min, x_inter, y_inter):
    # plt.xticks([0, x_max], [0, x_max])
    # plt.yticks([y_min, y_max], [y_min, y_max])
    plt.xticks(np.arange(0, x_max+x_inter, x_inter), np.arange(0, x_max+x_inter, x_inter))
    plt.yticks(np.arange(y_min, y_max+y_inter, y_inter), np.arange(y_min, y_max+y_inter, y_inter))
    plt.xlim((0, x_max))
    plt.ylim((y_min, y_max))
    plt.grid()

x_max = [240, 180, 120, 175, 200, 320, 300, 280]
y_max = [32, 32, 32, 32, 32, 16, 18, 18]
x_inter = [60, 45, 30, 35, 50, 80, 75, 70]
y_inter = [8, 8, 8, 8, 8, 4, 6, 6]

def plot_pair(input_folder):
    plt.figure(figsize=(24, 7.5))
    real_set, gen_set = [], []
    for i in range(1, 8+1, 1):
        plt.subplot(3, 8, i)
        with open(f"{input_folder}/data/real_x{i}.pkl", "rb") as f:
            real_x = np.array(pkl.load(f)["current"])
        real_x *= y_max[i-1]/np.max(real_x)*0.95
        real_set.append(real_x)
        plt.plot(real_x, c="red", linewidth=3)
        set_axis(x_max[i-1], y_max[i-1], 0, x_inter[i-1], y_inter[i-1])
    for j in range(1, 8+1, 1):
        plt.subplot(3, 8, 8+j)
        with open(f"{input_folder}/data/gen_x{j}.pkl", "rb") as f:
            gen_x = np.array(pkl.load(f))
        gen_x *= y_max[j-1]/np.max(gen_x)*0.95
        gen_set.append(gen_x)
        plt.plot(gen_x, c="blue", linewidth=3)
        set_axis(x_max[j-1], y_max[j-1], 0, x_inter[j-1], y_inter[j-1])
    for k in range(1, 8+1, 1):
        plt.subplot(3, 8, 16+k)
        real_acf = [1] + cal_acf(real_set[k-1]).tolist()
        gen_acf = [1] + cal_acf(gen_set[k-1]).tolist()
        plt.plot(real_acf, c="red", linewidth=3)
        plt.plot(gen_acf, c="blue", linewidth=3)
        set_axis(48, 1, -0.5, 12, 0.5)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.21, hspace=0.15)  # 调整子图间距
    plt.savefig(f"{input_folder}/acf.png")
    plt.clf()

if __name__ == "__main__":
    max_lag = 48
    # output_acf("generation/diffusion/cnn/driver/unconditional", id="CNN")
    # output_acf("generation/gmm/driver", id="GMM")
    # output_acf("generation/diffusion/attention/driver/new-epoch175-num1500", id="Generated")
    # output_acf("ACN-data/jpl/driver", id="Real")
    # acf_selection("output/acf/1053.pkl", "output/acf/Real_acf.pkl", min_rank=0)
    plot_pair("output/acf")
    pass
