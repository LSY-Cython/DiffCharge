from diffusion import DDPM
from dataset import creat_dataloader
from options import Options
import sys
import numpy as np

if __name__ == "__main__":
    isTrain = False
    model_name = "diffusion"
    opt = Options(model_name, isTrain)
    modules = sys.modules
    data_file = f"ACN-data/{opt.level}_generation_dataset.json"
    data_loader = creat_dataloader(data_file, opt.level, opt.batch_size, opt.shuffle, isTrain)
    model = DDPM(opt, data_loader)
    if isTrain:
        model.train()
    else:
        best_epoch = 199
        condition = np.array([1, 0])
        model.sample(f"weights/{model_name}/{opt.network}/{opt.level}/{opt.cond_flag}/epoch{best_epoch}.pt", n_samples=500, condition=condition)