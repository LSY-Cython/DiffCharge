import torch

class Options:
    def __init__(self, model_name, isTrain):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_epochs = 200
        self.level = "station"  # ***
        self.seq_len = 288  # station——288, driver——720
        self.cond_flag = "conditional"  # ***
        if isTrain:
            self.batch_size = 4  # station——4, driver——8
            self.shuffle = True
        else:
            self.batch_size = 1
            self.shuffle = False
        if model_name == "diffusion":
            self.init_lr = 1e-3
            self.network = "attention"  # "attention" or "cnn"
            self.input_dim = 1
            self.hidden_dim = 48
            self.cond_dim = 2
            self.nhead = 4
            self.beta_start = 1e-4
            self.beta_end = 0.5
            self.n_steps = 50
            self.schedule = "quadratic"  # "linear"