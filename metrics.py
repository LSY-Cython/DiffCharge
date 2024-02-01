import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
from sklearn.manifold import TSNE

plt.rcParams['font.sans-serif'] = "Times New Roman"
plt.rcParams['font.size'] = 14

def marginal_score(real_folder, gen_folder, model_name):
    real_set, gen_set = [], []
    real_paths = [os.path.join(real_folder, f) for f in os.listdir(real_folder) if f.endswith("pkl")]
    for i in range(len(real_paths)):
        with open(real_paths[i], "rb") as f:
            # real_series = np.array(pkl.load(f)["current"])
            real_series = np.array(pkl.load(f)["power"])
            real_series /= np.max(real_series)
            real_set.extend(real_series.tolist())
    gen_paths = [os.path.join(gen_folder, f) for f in os.listdir(gen_folder) if f.endswith("pkl")]
    for i in range(len(gen_paths)):
        with open(gen_paths[i], "rb") as f:
            gen_series = pkl.load(f)
            if np.max(gen_series) == 0.0:
                continue
            gen_series /= np.max(gen_series)
            gen_set.extend(gen_series.tolist())
    real_den, _, _ = plt.hist(x=np.array(real_set), bins=100, range=(0, 1), density=True)
    gen_den, _, _ = plt.hist(x=np.array(gen_set), bins=100, range=(0, 1), density=True)
    plt.clf()
    real_den /= np.max(real_den)
    gen_den /= np.max(gen_den)
    marg_dist = np.mean(np.abs(real_den-gen_den))  # MAE
    print(f"Marginal score of {model_name}: {marg_dist}")

def tSNE_visualization_battery(real_folder, gen_folder, id, pos):
    real_set, gen_set = [], []
    real_paths = [os.path.join(real_folder, f) for f in os.listdir(real_folder) if f.endswith("pkl")]
    for i in range(len(real_paths)):
        with open(real_paths[i], "rb") as f:
            real_series = np.array(pkl.load(f)["current"])
            real_series /= np.max(real_series)
            real_mask = np.zeros(720)
            if len(real_series) > 720:
                continue
            real_mask[0:len(real_series)] = real_series
            real_set.append(real_mask.tolist())
    gen_paths = [os.path.join(gen_folder, f) for f in os.listdir(gen_folder) if f.endswith("pkl")]
    for i in range(len(gen_paths)):
        with open(gen_paths[i], "rb") as f:
            gen_series = pkl.load(f)
            if np.max(gen_series) == 0.0:
                continue
            gen_series /= np.max(gen_series)
            gen_mask = np.zeros(720)
            gen_mask[0:len(gen_series)] = gen_series
            gen_set.append(gen_mask.tolist())
    # if len(real_set) > 1500:
    #     real_set = real_set[0:1500]
    # if len(gen_set) > 1500:
    #     gen_set = gen_set[0:1500]
    X = np.array(real_set + gen_set)
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=15, n_iter=300, early_exaggeration=10.0).fit_transform(X)
    real_x, real_y = X_embedded[0:len(real_set)][:, 0], X_embedded[0:len(real_set)][:, 1]
    gen_x, gen_y = X_embedded[len(real_set):][:, 0], X_embedded[len(real_set):][:, 1]
    # fig = plt.figure(figsize=(5, 4), dpi=150)
    # fig.set_tight_layout(True)
    plt.subplot(1, 4, pos)
    plt.scatter(real_x, real_y, color="red", s=5, label="Real")
    plt.scatter(gen_x, gen_y, color="blue", s=5, label="Generated")
    plt.legend(loc="upper right", prop={'size': 10})
    plt.xlabel("x-value", font={'size': 14})
    plt.ylabel("y-value", font={'size': 14})
    plt.title(id, font={'size': 16.6})
    print(f"{id} t-SNE done")
    # plt.savefig(f"output/t-SNE_{id}.png")

def tSNE_visualization_station(real_folder, gen_folder, id, pos):
    real_set, gen_set = [], []
    real_paths = [os.path.join(real_folder, f) for f in os.listdir(real_folder) if f.endswith("pkl")]
    for i in range(len(real_paths)):
        with open(real_paths[i], "rb") as f:
            real_series = np.array(pkl.load(f)["power"])
            real_series /= np.max(real_series)
            if "nan" in real_series:
                continue
            real_set.append(real_series.tolist())
    gen_paths = [os.path.join(gen_folder, f) for f in os.listdir(gen_folder) if f.endswith("pkl")]
    for i in range(len(gen_paths)):
        with open(gen_paths[i], "rb") as f:
            gen_series = pkl.load(f)
            gen_series /= np.max(gen_series)
            gen_set.append(gen_series.tolist())
    X = np.array(real_set + gen_set)
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=5, n_iter=300, early_exaggeration=5.0).fit_transform(X)
    real_x, real_y = X_embedded[0:len(real_set)][:, 0], X_embedded[0:len(real_set)][:, 1]
    gen_x, gen_y = X_embedded[len(real_set):][:, 0], X_embedded[len(real_set):][:, 1]
    plt.subplot(1, 2, pos)
    plt.scatter(real_x, real_y, color="red", s=12, label="Real")
    plt.scatter(gen_x, gen_y, color="blue", s=12, label="Generated")
    plt.legend(loc="upper right", prop={'size': 10})
    plt.xlabel("x-value", font={'size': 14})
    plt.ylabel("y-value", font={'size': 14})
    plt.title(id, font={'size': 16.6})
    print(f"{id} t-SNE done")

if __name__ == "__main__":
    # marginal_score("ACN-data/jpl/driver", "generation/diffusion/attention/driver/pdf-selected", "DiffCharge")
    # marginal_score("ACN-data/jpl/driver", "generation/aae/driver-alpha0.5", "AAE-0.5")
    # marginal_score("ACN-data/jpl/driver", "generation/aae/driver-alpha0.75", "AAE-0.75")
    # marginal_score("ACN-data/jpl/driver", "generation/gmm/driver", "GMM")
    # marginal_score("ACN-data/jpl/driver", "generation/diffusion/nonatt/driver/unconditional", "Non-Attention")
    # marginal_score("ACN-data/jpl/driver", "generation/diffusion/attention/driver/unconditional-T30", "T30")
    # marginal_score("ACN-data/jpl/driver", "generation/diffusion/attention/driver/unconditional-T40", "T40")
    # marginal_score("ACN-data/jpl/driver", "generation/diffusion/attention/driver/unconditional-T60", "T60")
    # marginal_score("ACN-data/jpl/driver", "generation/diffusion/attention/driver/unconditional-T70", "T70")

    # fig = plt.figure(figsize=(15, 4), dpi=150)
    # tSNE_visualization_battery("ACN-data/jpl/driver", "generation/gmm/driver", id="GMM", pos=1)
    # tSNE_visualization_battery("ACN-data/jpl/driver", "generation/aae/driver-alpha0.5", id="VAEGAN", pos=2)
    # tSNE_visualization_battery("ACN-data/jpl/driver", "generation/aae/driver-alpha0.75", id="TimeGAN", pos=3)
    # tSNE_visualization_battery("ACN-data/jpl/driver", "generation/diffusion/attention/driver/unconditional", id="DiffCharge", pos=4)
    # plt.tight_layout()
    # plt.savefig("output/t-SNE_battery.png")

    # marginal_score("ACN-data/jpl/station/2019", "generation/diffusion/attention/station/jpl-epoch175", "jpl")
    # marginal_score("ACN-data/caltech/station/2019", "generation/diffusion/attention/station/caltech-epoch175", "caltech")

    fig = plt.figure(figsize=(8, 4), dpi=150)
    tSNE_visualization_station("ACN-data/jpl/station/2019", "generation/diffusion/attention/station/jpl-epoch175", id="JPL Station", pos=1)
    tSNE_visualization_station("ACN-data/caltech/station/2019", "generation/diffusion/attention/station/caltech-epoch175", id="Caltech Station", pos=2)
    plt.tight_layout()
    plt.savefig("output/t-SNE_station.png")
    pass