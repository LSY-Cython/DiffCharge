import json
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import random

class DriverDataset(Dataset):
    def __init__(self, data_file, isTrain):
        with open(data_file, "r") as f:
            if isTrain:
                self.data_paths = json.load(f)["train"]
            else:
                self.data_paths = json.load(f)["test"]

    def __getitem__(self, index):
        with open(self.data_paths[index], "rb") as f:
            input_data = pkl.load(f)
        # sample-wise min-max normalization
        try:
            current = np.array(input_data["current"])
        except:
            current = np.array(input_data)
        current = torch.tensor(current / np.max(current), dtype=torch.float32)
        # assume the full charging duration is 12 hours
        current_mask, valid_mask = torch.zeros(720, 1), torch.zeros(720, 1)
        duration = current.shape[0]
        if duration > 720:
            current_mask[:, 0] = current[0:720]
            valid_mask[:, 0] = 1
        else:
            current_mask[0:duration, 0] = current
            valid_mask[0:duration:, 0] = 1
        try:
            kwhRequested = input_data["kwhRequested"] / 100
            minAvailable = input_data["minAvailable"] / 1000
            condition = torch.tensor([kwhRequested, minAvailable], dtype=torch.float32)
        except:
            condition = torch.tensor([0, 0], dtype=torch.float32)
        output_data = {"current": current_mask, "condition": condition, "mask": valid_mask}
        return output_data

    def __len__(self):
        return len(self.data_paths)

class StationDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, "r") as f:
            self.data_paths = json.load(f)["train"]

    def __getitem__(self, index):
        path = self.data_paths[index][0]
        site = self.data_paths[index][1]
        with open(path, "rb") as f:
            input_data = pkl.load(f)
        power = input_data["power"]
        power = torch.tensor(power/np.max(power), dtype=torch.float32).view(-1, 1)
        if site == "caltech":
            condition = torch.tensor([1, 0], dtype=torch.float32)
        else:
            condition = torch.tensor([0, 1], dtype=torch.float32)
        # ev_num = input_data["number"] / 100
        # number = torch.tensor([ev_num], dtype=torch.float32)
        output_data = {"power": power, "condition": condition}
        return output_data

    def __len__(self):
        return len(self.data_paths)

def creat_dataloader(data_file, level, batch_size, shuffle, isTrain):
    if level == "station":
        dataset = StationDataset(data_file)
    else:
        dataset = DriverDataset(data_file, isTrain)
    dataloader = DataLoader(dataset, batch_size, shuffle)
    return dataloader

def create_driver_dataset(input_folder, output_path):
    pkl_paths = [f"{input_folder}/{f}" for f in os.listdir(input_folder) if f.endswith("pkl")]
    with open(output_path, "w") as f:
        json.dump({"train": pkl_paths, "test": []}, f, indent=4)

def create_station_dataset(caltech_folder, jpl_folder, output_path):
    caltech_paths = [(f"{caltech_folder}/{f}", "caltech") for f in os.listdir(caltech_folder) if f.endswith("pkl")]
    jpl_paths = [(f"{jpl_folder}/{f}", "jpl") for f in os.listdir(jpl_folder) if f.endswith("pkl")]
    pkl_paths = caltech_paths + jpl_paths
    with open(output_path, "w") as f:
        json.dump({"train": pkl_paths}, f, indent=4)

def count_extreme_value(input_path):
    pkl_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith("pkl")]
    kwhRequested, minAvailable, duration = list(), list(), list()
    for p in pkl_paths:
        with open(p, "rb") as f:
            raw_data = pkl.load(f)
            kwhRequested.append(raw_data["kwhRequested"])
            minAvailable.append(raw_data["minAvailable"])
            duration.append(len(raw_data["pilot"]))
    print(f"Max/Min kwhRequested: {max(kwhRequested)}/{min(kwhRequested)} {list(sorted(kwhRequested, reverse=True))}")
    print(f"Max/Min minAvailable: {max(minAvailable)}/{min(minAvailable)} {list(sorted(minAvailable, reverse=True))}")
    print(f"Max/Min duration: {max(duration)}/{min(duration)} {list(sorted(duration, reverse=True))}")

if __name__ == "__main__":
    # create_driver_dataset("ACN-data/jpl/driver", "ACN-data/jpl/driver_generation_dataset.json")
    # create_station_dataset("ACN-data/caltech/station/2019", "ACN-data/jpl/station/2019", "ACN-data/station_generation_dataset.json")
    pass