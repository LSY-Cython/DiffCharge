import random
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import pickle as pkl
import json

class BatteryDataset(Dataset):
    def __init__(self, data_file, isTrain):
        with open(data_file, "r") as f:
            if isTrain:
                self.data_paths = json.load(f)["train"]
            else:
                self.data_paths = json.load(f)["test"]

    def __getitem__(self, index):
        path = self.data_paths[index][0]
        label = self.data_paths[index][1]
        with open(path, "rb") as f:
            input_data = pkl.load(f)
        if type(input_data) is dict:
            series = input_data["current"]
        else:
            series = input_data
        mask = np.zeros((720, 1))
        if len(series) > 720:
            mask[:, 0] = series[0:720]
        else:
            mask[0:len(series), 0] = series
        series = torch.tensor(mask/(np.max(mask)+1e-3), dtype=torch.float32)
        label = torch.tensor([label], dtype=torch.float32)
        output_data = {"series": series, "label": label}
        return output_data

    def __len__(self):
        return len(self.data_paths)

class StationDataset(Dataset):
    def __init__(self, data_file, isTrain):
        with open(data_file, "r") as f:
            if isTrain:
                self.data_paths = json.load(f)["train"]
            else:
                self.data_paths = json.load(f)["test"]

    def __getitem__(self, index):
        path = self.data_paths[index][0]
        label = self.data_paths[index][1]
        with open(path, "rb") as f:
            input_data = pkl.load(f)
        if type(input_data) is dict:
            series = input_data["power"]
        else:
            series = input_data
        series = series.reshape(288, 1)
        series = torch.tensor(series/(np.max(series)+1e-3), dtype=torch.float32)
        label = torch.tensor([label], dtype=torch.float32)
        output_data = {"series": series, "label": label}
        return output_data

    def __len__(self):
        return len(self.data_paths)

def split_dataset(real_folder, gen_folder, model_name):
    real_paths = [f"{real_folder}/{f}" for f in os.listdir(real_folder) if f.endswith("pkl")]
    gen_paths = [f"{gen_folder}/{f}" for f in os.listdir(gen_folder) if f.endswith("pkl")]
    random.shuffle(real_paths)
    random.shuffle(gen_paths)
    real_idx, gen_idx = int(len(real_paths)*0.8), int(len(gen_paths)*0.8)
    real_train, real_test = real_paths[0:real_idx], real_paths[real_idx:]
    gen_train, gen_test = gen_paths[0:gen_idx], gen_paths[gen_idx:]
    real_train = list(zip(real_train, [1]*len(real_train)))
    real_test = list(zip(real_test, [1]*len(real_test)))
    gen_train = list(zip(gen_train, [0]*len(gen_train)))
    gen_test = list(zip(gen_test, [0]*len(gen_test)))
    with open(f"discrimination/{model_name}_dataset.json", "w") as f:
        json.dump({"train": real_train+gen_train, "test": real_test+gen_test}, f, indent=4)

def creat_dataloader(data_file, batch_size, shuffle, isTrain, level):
    if level == "battery":
        dataset = BatteryDataset(data_file, isTrain)
    else:
        dataset = StationDataset(data_file, isTrain)
    dataloader = DataLoader(dataset, batch_size, shuffle)
    return dataloader

class BatteryNet(nn.Module):
    def __init__(self):
        super(BatteryNet, self).__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
        self.lin = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.flip(x, dims=[1])  # refrain from harmful zero padding to hidden states
        hid_enc, (_, _) = self.rnn(x)  # (B, L, hidden_dim)
        x = hid_enc[:, -1, :]  # (B, hidden_dim)
        y = self.lin(x)  # (B, 1)
        return y

class StationNet(nn.Module):
    def __init__(self):
        super(StationNet, self).__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=16, num_layers=2, batch_first=True)
        self.lin = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        hid_enc, (_, _) = self.rnn(x)  # (B, L, hidden_dim)
        x = hid_enc[:, -1, :]  # (B, hidden_dim)
        y = self.lin(x)  # (B, 1)
        return y

class Classification:
    def __init__(self, data_loader, model_name, level):
        super(Classification, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if level == "battery":
            self.discriminator = BatteryNet().to(self.device)
            self.n_epochs = 150
        else:
            self.discriminator = BatteryNet().to(self.device)
            self.n_epochs = 50
        self.loss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.data_loader = data_loader
        self.model_name = model_name

    def train(self):
        epoch_loss = []
        for epoch in range(self.n_epochs):
            batch_loss = []
            for i, data in enumerate(self.data_loader):
                x = data["series"].to(self.device)
                y = data["label"].to(self.device)
                pred = self.discriminator(x)
                self.optimizer.zero_grad()
                loss = self.loss(pred, y)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(np.mean(batch_loss))
            print(f"epoch={epoch}/{self.n_epochs}, loss={epoch_loss[-1]}")
            save_path = f"weights/discrimination/{self.model_name}/epoch{epoch}.pt"
            torch.save(self.discriminator.state_dict(), save_path)
        plt.plot(epoch_loss, label="Cross entropy")
        plt.savefig(f"discrimination/{self.model_name}.png")

    def test(self, pt_file):
        weight = torch.load(pt_file, map_location=self.device)
        self.discriminator.load_state_dict(weight)
        self.discriminator.eval()
        test_error = []
        for i, data in enumerate(self.data_loader):
            x = data["series"].to(self.device)
            y = data["label"].to(self.device)
            pred = self.discriminator(x)
            loss = self.loss(pred, y)
            test_error.append(loss.item())
        mae = np.mean(test_error)
        print(f"Test cross entropy score: {mae}")
        return mae

if __name__ == "__main__":
    # level = "battery"
    # model_name = "DiffCharge"
    # split_dataset("ACN-data/jpl/driver", "generation/diffusion/attention/driver/new-epoch175-num1500", model_name)
    # model_name = "GMM"
    # split_dataset("ACN-data/jpl/driver", "generation/gmm/driver", model_name)
    # model_name = "TimeGAN"
    # split_dataset("ACN-data/jpl/driver", "generation/aae/driver-alpha0.75", model_name)
    # model_name = "AAE"
    # split_dataset("ACN-data/jpl/driver", "generation/aae/driver-alpha0.5", model_name)
    # model_name = "NonAtt"
    # split_dataset("ACN-data/jpl/driver", "generation/diffusion/nonatt/driver/unconditional", model_name)
    # model_name = "T30"
    # split_dataset("ACN-data/jpl/driver", "generation/diffusion/attention/driver/unconditional-T30", model_name)
    # model_name = "T40"
    # split_dataset("ACN-data/jpl/driver", "generation/diffusion/attention/driver/unconditional-T40", model_name)
    # model_name = "T60"
    # split_dataset("ACN-data/jpl/driver", "generation/diffusion/attention/driver/unconditional-T60", model_name)
    # model_name = "T70"
    # split_dataset("ACN-data/jpl/driver", "generation/diffusion/attention/driver/unconditional-T70", model_name)

    level = "station"
    # model_name = "jpl"
    # split_dataset("ACN-data/jpl/station/2019", "generation/diffusion/attention/station/jpl-epoch175", model_name)
    model_name = "caltech"
    split_dataset("ACN-data/caltech/station/2019", "generation/diffusion/attention/station/caltech-epoch175", model_name)

    train_loader = creat_dataloader(f"discrimination/{model_name}_dataset.json", batch_size=16, shuffle=True, isTrain=True, level=level)  # battery-32, station-16
    classifier = Classification(train_loader, model_name, level)
    classifier.train()

    test_loader = creat_dataloader(f"discrimination/{model_name}_dataset.json", batch_size=1, shuffle=True, isTrain=False, level=level)
    classifier = Classification(test_loader, model_name, level)
    test_errors = []
    n_epochs = classifier.n_epochs
    for i in range(n_epochs-10, n_epochs, 1):
        e = classifier.test(f"weights/discrimination/{model_name}/epoch{i}.pt")
        test_errors.append(e)
        print(f"epoch{i}: {e}")
    print(f"Final error: {np.mean(test_errors)}±{np.std(test_errors)}")